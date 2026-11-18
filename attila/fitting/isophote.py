"""
Isophote fitting and radial profiles of the deblended targets.

This module wraps `photutils.isophote` to measure elliptical isophotes
on the products of the AstroPhot 2D deblending
(`attila.fitting.deblend_2D`) and to turn them into one-dimensional
quantities:

- `Ellipse_isophote` fits free ellipses to the partial residuals of
  each band (data minus the models of all the objects but the target),
  or samples the images along ellipses of fixed geometry, builds the
  2D model of the fitted isophotes and plots it;
- `IsophotesAnalysis` converts the isophote lists into circularized
  and major-axis intensity profiles and curves of growth, and lets the
  user mask ranges of the profile interactively.

In the ATTILA flow it sits between the 2D deblending and the 1D Sersic
fits of `attila.fitting.profiles`. All the lengths handled here (semi
major axes, centres, radii) are in pixels, and the intensities are in
the units of the images (per pixel).
"""
import numpy as np
from numpy.typing import ArrayLike

from astropy.io import ascii as asciii

from photutils.isophote import EllipseGeometry, EllipseSample, Isophote, IsophoteList, Ellipse, build_ellipse_model
from photutils.aperture import EllipticalAperture

from astropy.table import vstack, hstack

from typing import List, Dict, Tuple, Union, Optional

from attila import attila_logger
from attila.fitting.COG import COG
from attila.interactive.fitprep import _compute_cps_SB, _plot_data

import matplotlib.pyplot as plt
from matplotlib import colormaps as mcmaps
from matplotlib import colors as mcolors

import shelve

import os

import ipywidgets as widgets
from IPython.display import display

from copy import deepcopy

from functools import partial


def _exclude_bad_isophotes(isolist, bad_stops=[3,5], intensity_constraint=None, eps_max=0.85, save=True):
        """
        Return the isophotes that pass the quality cuts.

        An isophote is kept only if its mean intensity is larger than
        `intensity_constraint`, its `photutils` stop code is not in
        `bad_stops` and its ellipticity is smaller than `eps_max`. With
        the default `bad_stops` the fits flagged with stop code 3
        (singular matrix or too few points in the harmonic fit) and 5
        (ellipse diverged) are removed.

        Parameters
        ----------
        isolist : photutils.isophote.IsophoteList
            The isophotes to filter.
        bad_stops : list of int
            The `photutils` stop codes of the isophotes to exclude.
        intensity_constraint : float, optional
            The minimum mean intensity of an isophote, in image units.
            `None` means no cut on the intensity.
        eps_max : float
            The maximum ellipticity (`1 - b/a`) of an isophote.
        save : bool
            Not used.

        Returns
        -------
        photutils.isophote.IsophoteList
            The isophotes that pass all the cuts.
        """
        isophotes = []

        if intensity_constraint is None:
            intensity_constraint=-np.inf

        for iso in isolist:
            if iso.intens>intensity_constraint:
                if iso.stop_code not in bad_stops:
                    if iso.eps<eps_max:

                        isophotes.append(iso)

        return IsophoteList(isophotes)

def _fit_initial(ellipse_geometry, ellipse):
    """
    Refine an initial ellipse geometry with fits at growing radii.

    A single isophote is fitted ten times with `ellipse.fit_isophote`,
    at semi-major axes of 5, 7, ..., 23 pixels. After each fit the
    centre, the ellipticity and the position angle of the fitted
    isophote are copied into `ellipse_geometry`, which is modified in
    place.

    Parameters
    ----------
    ellipse_geometry : photutils.isophote.EllipseGeometry
        The geometry to update.
    ellipse : photutils.isophote.Ellipse
        The fitter, already bound to an image.

    Returns
    -------
    ellipse_geometry : photutils.isophote.EllipseGeometry
        The input geometry, with the parameters of the last fit and a
        semi-major axis of 23 pixels.
    isophote : photutils.isophote.Isophote
        The isophote of the last fit.
    """

    i = 0
    while (i < 10):

        ellipse_geometry.sma = 5 + 2*i

        isophote = ellipse.fit_isophote(sma=5 + 2*i)

        ellipse_geometry.x0 = isophote.x0
        ellipse_geometry.y0 = isophote.y0
        ellipse_geometry.sma = 5 + 2*i
        ellipse_geometry.eps = isophote.eps
        ellipse_geometry.pa = isophote.pa

        i += 1

    return ellipse_geometry, isophote


class Ellipse_isophote:
    """
    Isophotes of a deblended target in every band.

    The object takes the per-band results of the 2D deblending and
    measures the isophotes of the target either by fitting free
    ellipses to the partial residuals (`fit_isophotes`), or by sampling
    an image along ellipses of fixed geometry (`sample_isophotes`,
    `sample_isophotes_circ`, `cross_sample_isophotes`). Each of these
    methods overwrites the entry of `isophotes_dict` for the bands it
    works on. `bmodel` then builds the 2D image of the isophote model
    and `plot` compares it with the residuals.

    Attributes
    ----------
    bands : list of str
        The bands of the input dictionary.
    deblend2D_results : dict
        The per-band results of the 2D deblending (see `__init__`).
    isophotes_dict : dict
        The isophotes of every band,
        `{band: photutils.isophote.IsophoteList}`.
    start : int
        The smallest semi-major axis sampled, in pixels (1).
    stop : int
        The largest semi-major axis, in pixels: half the number of
        rows of the residual image of the first band, minus 2.
    sma0 : int
        The semi-major axis of the first fitted ellipse, in pixels
        (5).
    step : int
        The increment of the semi-major axis between isophotes (1);
        in pixels because `linear` is `True`.
    linear : bool
        If `True` the semi-major axis grows linearly, otherwise
        geometrically.
    fix_center : bool
        If `True` the centre is kept fixed in `fit_isophotes`.
    ellipse_models_dict : dict
        The 2D model image built from the isophotes of every band,
        set by `bmodel`.
    ellipse_noisymodels_dict : dict
        The same models with Gaussian noise added, set by `bmodel`.
    """

    def __init__(self, deblend2D_results: Dict[str, Dict[str, Union[ArrayLike, EllipseGeometry]]]):
        """
        Store the deblending results and the default fit settings.

        Parameters
        ----------
        deblend2D_results
            The results of the 2D deblending, one dictionary per band.
            The entries used by this class are `residuals` (2D partial
            residual image), `model` (2D image sampled by
            `sample_isophotes` and `sample_isophotes_circ`),
            `ell_geoms` (the `EllipseGeometry` of the target from the
            AstroPhot fit, in pixels and radians) and, for plotting
            only, `target` (the AstroPhot `Target_Image`, whose
            variance gives the noise map).
        """

        self.bands = list(deblend2D_results.keys())
        self.deblend2D_results = deblend2D_results

        self.isophotes_dict = {}
        for band in self.bands:
            self.isophotes_dict[band] = IsophoteList([])

        self.start = 1
        # Largest semi-major axis: half the image side (rows) minus a
        # 2 pixel margin, taken from the first band.
        self.stop = int(self.deblend2D_results[self.bands[0]]['residuals'].shape[0]/2.) - 2
        self.sma0 = 5
        self.step = 1
        self.linear = True
        self.fix_center = True

    def fit_isophotes(self):
        """
        Fit free elliptical isophotes to the residuals of every band.

        For each band a `photutils.isophote.Ellipse` is built on the
        partial residual image, starting from the geometry of the
        AstroPhot fit, and `fit_image` is run from `sma0` outwards to
        `stop` and inwards to the centre. The intensity along each
        ellipse is the median of the samples, with 10 iterations of
        sigma clipping (`nclip`), and the iterations on an isophote
        stop when fewer than 30 per cent of its points are valid
        (`fflag`). The central isophote (`sma == 0`), if present, is
        dropped.

        The result is stored in `isophotes_dict[band]`; the fitter and
        the list of the last band are also kept in `_ellipse` and
        `_isophote_list`.
        """

        for band in self.bands:

            deblend2D_result = self.deblend2D_results[band]

            data = deblend2D_result['residuals']
            ellipse_geometry = deblend2D_result['ell_geoms']

            self._ellipse = Ellipse(image=data, geometry=ellipse_geometry)

            self._isophote_list = self._ellipse.fit_image(
                sma0=self.sma0,
                minsma=0,
                maxsma=self.stop,
                step=self.step,
                linear=self.linear,
                fix_center=self.fix_center,
                integrmode='median',
                nclip=10,
                fflag=0.3
            )

            if self._isophote_list is not None:

                if len(self._isophote_list) > 0:

                    # Remove the central pixel "isophote".
                    if self._isophote_list[0].sma == 0:
                        self._isophote_list = self._isophote_list[1:]

            self.isophotes_dict[band] = self._isophote_list


    def sample_isophotes(self):
        """
        Sample the `model` image along ellipses of fixed geometry.

        No fit is performed. For each band the image stored under the
        `model` key of the deblending results is sampled along
        ellipses that all share the centre, the ellipticity and the
        position angle of the AstroPhot geometry (`ell_geoms`), with
        semi-major axes growing from `start` to `stop` in steps of
        `step`. The intensity of each isophote is the median along the
        ellipse. The isophotes are created with stop code 4, which
        `photutils` also uses when the fitting algorithm is turned
        off.

        Samples for which `EllipseSample.update` raises an
        `AttributeError` are skipped. The result replaces
        `isophotes_dict[band]`.
        """

        for band in self.bands:

            _isophote_list = []

            deblend2D_result = self.deblend2D_results[band]

            data = deblend2D_result['model']
            ellipse_geometry = deblend2D_result['ell_geoms']

            sma = self.start #ellipse_geometry.sma

            while True:

                # if (sma > self.stop):
                #     sma, self.step = _iso.sample.geometry.reset_sma(self.step)

                if (sma < self.start) or (sma > self.stop):
                    break

                sample = EllipseSample(
                    data,
                    sma=sma,
                    x0=ellipse_geometry.x0,
                    y0=ellipse_geometry.y0,
                    eps=ellipse_geometry.eps,
                    position_angle=ellipse_geometry.pa,
                    astep=self.step,
                    linear_growth=self.linear,
                    integrmode='median'
                )

                try:
                    # The four geometry values are passed as the
                    # `fixed_parameters` flags of the sample.
                    sample.update(fixed_parameters=[ellipse_geometry.x0,
                                                    ellipse_geometry.y0,
                                                    ellipse_geometry.eps,
                                                    ellipse_geometry.pa])

                    # Arguments: sample, niter=0, valid=True,
                    # stop_code=4.
                    _iso = Isophote(sample, 0, True, 4)

                    _isophote_list.append(_iso)

                except AttributeError:
                    pass

                    # _iso = Isophote(sample, 0, True, 4)

                sma = sample.geometry.update_sma(self.step)

            self.isophotes_dict[band] = IsophoteList(_isophote_list)

    def sample_isophotes_circ(self):
        """
        Sample the `model` image along concentric circles.

        This is the same as `sample_isophotes`, but the ellipticity of
        the sampling geometry is set to 0, so the image is sampled
        along circles of radius `start` to `stop` pixels centred on
        the AstroPhot centre. The result replaces
        `isophotes_dict[band]`.
        """

        for band in self.bands:

            _isophote_list = []

            deblend2D_result = self.deblend2D_results[band]

            data = deblend2D_result['model']
            ellipse_geometry = deblend2D_result['ell_geoms']

            sma = self.start #ellipse_geometry.sma

            while True:

                if (sma < self.start) or (sma > self.stop):
                    break

                sample = EllipseSample(
                    data,
                    sma=sma,
                    x0=ellipse_geometry.x0,
                    y0=ellipse_geometry.y0,
                    eps=0.0,
                    position_angle=ellipse_geometry.pa,
                    astep=self.step,
                    linear_growth=self.linear,
                    integrmode='median'
                )

                try:
                    sample.update(fixed_parameters=[ellipse_geometry.x0,
                                                    ellipse_geometry.y0,
                                                    ellipse_geometry.eps,
                                                    ellipse_geometry.pa])

                    _iso = Isophote(sample, 0, True, 4)

                    _isophote_list.append(_iso)

                except AttributeError:
                    pass

                    # _iso = Isophote(sample, 0, True, 4)

                sma = sample.geometry.update_sma(self.step)

            self.isophotes_dict[band] = IsophoteList(_isophote_list)



    def cross_sample_isophotes(self, target:str, reference:str):
        """
        Sample one band along the isophotes fitted in another band.

        The geometry (centre, semi-major axis, ellipticity, position
        angle) of every isophote of the `reference` band is reused to
        sample the partial residuals of the `target` band, taking the
        median along each ellipse. The two bands are therefore
        measured in identical apertures, as needed for colour
        profiles. The result replaces `isophotes_dict[target]`.

        Parameters
        ----------
        target
            The band whose residual image is sampled.
        reference
            The band whose isophotes provide the geometries.
        """

        deblend2D_result = self.deblend2D_results[target]

        _isophote_list_ = []
        for iso in self.isophotes_dict[reference]:

            geometry = iso.sample.geometry

            sample = EllipseSample(deblend2D_result['residuals'], geometry.sma, geometry=geometry, integrmode='median')

            sample.update()

            iso_ = Isophote(sample, 0, True, 4)
            _isophote_list_.append(iso_)

        self._isophote_list = IsophoteList(_isophote_list_)

        self.isophotes_dict[target] = self._isophote_list


    def bmodel(self, noise):
        """
        Build the 2D model image of the isophotes of every band.

        The model is built with `photutils.isophote.build_ellipse_model`
        on a grid with the shape of the residual image and stored in
        `ellipse_models_dict`. A second version, to which Gaussian
        noise of standard deviation `noise[band]` is added, is stored
        in `ellipse_noisymodels_dict`. If the model of a band cannot be
        built (`ValueError`), a message is printed and the band is
        skipped.

        Parameters
        ----------
        noise : dict
            The standard deviation of the noise to add in each band,
            `{band: float or numpy.ndarray}`, in image units.
        """

        self.ellipse_models_dict = {}
        self.ellipse_noisymodels_dict = {}
        for band in self.isophotes_dict.keys():
            try:
                self.ellipse_models_dict[band] =  build_ellipse_model(
                    self.deblend2D_results[band]['residuals'].shape,
                    self.isophotes_dict[band]
                )

                self.ellipse_noisymodels_dict[band] = self.ellipse_models_dict[band] + np.random.normal(
                                scale=noise[band],
                                size=self.ellipse_models_dict[band].shape
                            )

            except ValueError as e:
                print(f"Error processing band {band}: {e}")

    def monitor_band(self, change):
        """
        Redraw the figure when the band dropdown changes.

        Parameters
        ----------
        change : dict
            The `ipywidgets` change event; `change['new']` is the
            selected band.
        """

        new_band = change['new']
        if new_band != self.current_band:

            self.current_band = new_band
            self._plot(fig=self.fig, axs=self.axs)

    def _plot(self, fig, axs):
        """
        Draw the residuals, the isophote model and their difference.

        The three panels show, for the current band: the partial
        residuals with a subset of the isophotes overlaid (one every 5
        below a semi-major axis of 40 pixels, one every 10 beyond), the
        noisy isophote model, and the residuals minus the noise-free
        isophote model. The images are drawn by
        `attila.interactive.fitprep._plot_data` in units of the noise
        map, which is the square root of the variance of the AstroPhot
        target, with colour thresholds at 25 and 23.5 mag / arcsec^2.
        `bmodel` must have been called before.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to draw on.
        axs : sequence of matplotlib.axes.Axes
            The three axes of the figure.
        """

        sb_func = partial(self.sb_func_field, band=self.current_band)

        [ax.clear() for ax in self.axs]

        noisemap = np.sqrt(self.deblend2D_results[self.current_band]['target'].variance.detach().cpu().numpy())

        _plot_data(
            fig=fig,
            ax=axs[0],
            data=self.deblend2D_results[self.current_band]['residuals'],
            noise=noisemap,
            sb_func=sb_func,
            sb_upper=25,
            sb_max=23.5,
        )


        _plot_data(
            fig=fig,
            ax=axs[1],
            data=self.ellipse_noisymodels_dict[self.current_band],
            noise=noisemap,
            sb_func=sb_func,
            sb_upper=25,
            sb_max=23.5,
        )

        for i, isophote in enumerate(self.isophotes_dict[self.current_band]):

            # Thin out the overlaid isophotes: one every 5 in the
            # inner 40 pixels, one every 10 outside.
            if (i % 5 == 0) & (isophote.sma < 40):

                ell = EllipticalAperture(
                    positions=(isophote.x0, isophote.y0),
                    a=isophote.sma,
                    b=isophote.sma * (1 - isophote.eps),
                    theta=isophote.pa
                )

                ell.plot(ax=axs[0], color='C2', lw=1, alpha=0.75)

            elif (i % 10 == 0) & (isophote.sma >= 40):

                ell = EllipticalAperture(
                    positions=(isophote.x0, isophote.y0),
                    a=isophote.sma,
                    b=isophote.sma * (1 - isophote.eps),
                    theta=isophote.pa
                )

                ell.plot(ax=axs[0], color='C2', lw=1, alpha=0.75)

        _plot_data(
            fig=fig,
            ax=axs[2],
            data=(self.deblend2D_results[self.current_band]['residuals'] - self.ellipse_models_dict[self.current_band]),
            noise=noisemap,
            sb_func=sb_func,
            sb_upper=25,
            sb_max=23.5,
        )

        self.fig.canvas.draw_idle()

    def plot(self, sb_func_field):
        """
        Create the three-panel figure and the band dropdown.

        A 1 x 3 figure is created, an `ipywidgets` dropdown to select
        the band is displayed, and the panels are drawn for the first
        band in alphabetical order (see `_plot`). `bmodel` must have
        been called before.

        Parameters
        ----------
        sb_func_field : callable
            A function called as `sb_func_field(SB=..., band=...)`
            that converts a surface brightness in mag / arcsec^2 into
            image units per pixel for the given band, e.g.
            `functools.partial(_compute_cps_SB, field=field)`.
        """

        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))

        self.sb_func_field = sb_func_field

        self.current_band = sorted(self.bands)[0]

        self.dropdown_band = widgets.Dropdown(
            options=sorted(self.bands),
            description='Band:',
            value=self.current_band
        )

        display(self.dropdown_band)

        # self.load_band(self.current_band)

        self.dropdown_band.observe(self.monitor_band, names='value')

        self._plot(fig=self.fig, axs=self.axs)

    def save(self, filename: str):
        """
        Save the isophote lists of all the bands to a `shelve` file.

        Parameters
        ----------
        filename
            The path of the `shelve` database; each band is a key and
            the value is its `IsophoteList`.
        """
        with shelve.open(filename) as db:
            for band, isophote_list in self.isophotes_dict.items():
                db[band] = isophote_list


def load_isophotes(filename: str):
    """
    Load the isophote lists saved by `Ellipse_isophote.save`.

    Parameters
    ----------
    filename
        The path of the `shelve` database.

    Returns
    -------
    dict
        The isophotes of every band,
        `{band: photutils.isophote.IsophoteList}`.
    """

    isophote_dict = {}
    with shelve.open(filename) as db:
        for key in db.keys():
            isophote_dict[key] = db[key]

    return isophote_dict

from attila.interactive.profile_masking import InteractiveProfileMasking

def _plot_profile(fig, ax, x, y, y_err):
    """
    Plot a radial intensity profile on a logarithmic y axis.

    The profile is shifted vertically by subtracting the minimum of
    `y - y_err`, so that the lowest error bar starts at zero and the
    shifted values can be shown on a logarithmic scale. The points are
    drawn as a pickable scatter, so that they can be selected with the
    mouse, and the y limits are frozen.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the axis (not used).
    ax : matplotlib.axes.Axes
        The axis the profile is drawn on.
    x : numpy.ndarray
        The radii of the profile, in pixels.
    y : numpy.ndarray
        The intensities of the profile, in image units.
    y_err : numpy.ndarray
        The uncertainties on the intensities. They are used in the
        vertical shift before the check for `None`.
    """

    y_based = y - np.nanmin(y - y_err)

    sc = ax.scatter(x, y_based, color='C0', s=10, picker=True)

    if y_err is not None:
        ax.errorbar(x, y_based, yerr=y_err, ls='none', color='C0', marker='none', elinewidth=1, capsize=3)

    ax.set_yscale('log')

    ax.set_ylim(ax.get_ylim())


class IsophotesAnalysis:
    """
    Radial profiles and curves of growth derived from the isophotes.

    From the isophote list of every band the object builds the
    circularized profile (intensity against `sma * sqrt(1 - eps)`) and
    the major-axis profile (intensity against `sma`). On creation it
    also opens a figure showing the circularized profile of the `g`
    band, on which ranges of points can be selected with the mouse and
    turned into a mask with `apply_mask`. The profiles and the mask
    are the inputs of the 1D fits in `attila.fitting.profiles`.

    Attributes
    ----------
    bands : list of str
        The bands of the input dictionary.
    isolist_dict : dict
        The isophotes of every band.
    circularized_profile_dict : dict
        For every band, a dictionary with the keys `radius`,
        `radius_err`, `intensity` and `intensity_err`.
    maj_profile_dict : dict
        For every band, a dictionary with the keys `sma`, `sma_err`,
        `intensity` and `intensity_err`.
    fig, ax : matplotlib figure and axis
        The figure used for the interactive masking.
    band : str
        The band shown at creation and used to size the profile mask
        (`'g'`).
    ipm : attila.interactive.profile_masking.InteractiveProfileMasking
        The handler of the interactive selection, set by
        `plot_profiles`.
    mask : numpy.ndarray
        The profile mask, set by `apply_mask`.
    """

    def __init__(self, isolist_dict: Dict[str, IsophoteList]):
        """
        Build the profiles and open the interactive profile figure.

        Parameters
        ----------
        isolist_dict
            The isophotes of every band, as stored in
            `Ellipse_isophote.isophotes_dict` or returned by
            `load_isophotes`. It must contain the band `g`.
        """
        self.bands = list(isolist_dict.keys())
        self.isolist_dict = isolist_dict

        self.get_circularized_profile()
        self.get_maj_profile()

        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 4))

        self.band = 'g'

        self.plot_profiles(band=self.band, which='circularized')


    def get_COG(self):
        """
        Return the curve of growth of every band from the isophotes.

        The radius of each point is the circularized radius
        `sma * sqrt(1 - eps)` of the isophote, with the uncertainty
        propagated from the ellipticity error only, and the flux is
        the `tflux_c` attribute of the isophote list, which
        `photutils` defines as the sum of the pixels inside a circle
        of radius `sma`. The flux uncertainty is set to 0.

        Returns
        -------
        dict
            The curves of growth, `{band: COG}`, with radii in pixels
            and fluxes in image units.
        """

        COG_dict = {}

        for band, isolist in self.isolist_dict.items():
            sma = isolist.sma
            sma_err = 0

            integrated_flux = isolist.tflux_c

            ellipticity = isolist.eps
            ellipticity_err = isolist.ellip_err

            # Axis ratio q = b/a = 1 - eps, so its error is that of
            # the ellipticity.
            flattening = 1 - ellipticity
            flattening_err = ellipticity_err

            circularized_radius = sma * np.sqrt(flattening)
            circularized_radius_err = np.sqrt(
                    (np.sqrt(flattening) * sma_err) ** 2 +
                    (sma / (2 * np.sqrt(flattening)) * flattening_err) ** 2
                )

            COG_dict[band] = COG(circularized_radius, circularized_radius_err, integrated_flux, 0)

        return COG_dict

    def get_circularized_profile(self):
        """
        Build the circularized intensity profile of every band.

        The circularized radius of an isophote is
        `r = sma * sqrt(q)`, with `q = 1 - eps` the axis ratio, i.e.
        the radius of the circle with the same area as the ellipse.
        Its uncertainty is propagated from the ellipticity error, the
        semi-major axis being exact:
        `r_err = sma / (2 * sqrt(q)) * eps_err`. The intensity and its
        error are the mean intensity along the isophote and its
        uncertainty as given by `photutils` (`intens`, `int_err`).

        The result is stored in `circularized_profile_dict`, with
        radii in pixels and intensities in image units per pixel.
        """

        self.circularized_profile_dict = {}

        for band, isolist in self.isolist_dict.items():
            sma = isolist.sma
            sma_err = 0

            intensity = isolist.intens
            intensity_err = isolist.int_err

            ellipticity = isolist.eps
            ellipticity_err = isolist.ellip_err

            flattening = 1 - ellipticity
            flattening_err = ellipticity_err

            circularized_radius = sma * np.sqrt(flattening)
            circularized_radius_err = np.sqrt(
                    (np.sqrt(flattening) * sma_err) ** 2 +
                    (sma / (2 * np.sqrt(flattening)) * flattening_err) ** 2
                )

            self.circularized_profile_dict[band] = {'radius': circularized_radius,
                                               'radius_err':circularized_radius_err,
                                               'intensity': intensity,
                                               'intensity_err': intensity_err}

    def get_maj_profile(self):
        """
        Build the major-axis intensity profile of every band.

        The radius is the semi-major axis of the isophotes, with a
        zero uncertainty, and the intensity and its error are those of
        the isophotes (`intens`, `int_err`). The result is stored in
        `maj_profile_dict`, with radii in pixels and intensities in
        image units per pixel.
        """

        self.maj_profile_dict = {}

        for band, isolist in self.isolist_dict.items():
            sma = isolist.sma
            sma_err = np.zeros_like(sma)


            intensity = isolist.intens
            intensity_err = isolist.int_err

            self.maj_profile_dict[band] = {'sma': sma,
                                      'sma_err':sma_err,
                                      'intensity': intensity,
                                      'intensity_err': intensity_err}

    def apply_mask(self, mask=None):
        """
        Set the profile mask, by default from the interactive selection.

        The mask is only stored in the `mask` attribute; the profile
        dictionaries are not modified (the code that removed the
        masked points from them is commented out below).

        Parameters
        ----------
        mask : numpy.ndarray, optional
            Boolean array with one element per isophote, `True` for
            the points to exclude. `None` means that the mask is built
            from the ranges selected with the mouse on the profile
            figure, through `InteractiveProfileMasking.to_mask`.

        Raises
        ------
        RuntimeError
            If the mask cannot be built from the interactive
            selection.
        """

        self.mask = mask

        if self.mask is None:
            try:
                self.ipm.to_mask()

                self.mask = self.ipm.mask

            except Exception as e:
                raise RuntimeError("An error occurred while applying the mask: " + str(e))

        # if self.circularized_profile_dict is not None:

        #     #self.unmasked_circularized_profile_dict = deepcopy(self.circularized_profile_dict)
        #     #self.unmasked_maj_profile_dict = deepcopy(self.maj_profile_dict)

        #     for band, profile in self.circularized_profile_dict.items():
        #         for key in profile.keys():

        #             self.circularized_profile_dict[band][key] = self.circularized_profile_dict[band][key][~self.mask]

        # if self.maj_profile_dict is not None:
        #     for band, profile in self.maj_profile_dict.items():
        #         for key in profile.keys():

        #             self.maj_profile_dict[band][key] = self.maj_profile_dict[band][key][~self.mask]
        # else:
        #     raise ValueError("Both 'circularized_profile_dict' and 'maj_profile_dict' are empty. There is nothing to apply the mask to")



    def plot_profiles(self, band='g', which='circularized'):
        """
        Plot a profile and enable the interactive selection of ranges.

        The requested profile is drawn with `_plot_profile` on the
        figure of the object, and an `InteractiveProfileMasking`
        handler is connected to the matplotlib pick events: clicking
        on two points of the profile selects the range between them.
        The handler starts from an empty mask with one element per
        isophote of the band stored in the `band` attribute.

        Parameters
        ----------
        band : str
            The band of the profile to plot.
        which : str
            The profile to plot: `'circularized'` (intensity against
            circularized radius) or `'maj'` (intensity against
            semi-major axis).
        """

        if which == 'circularized':
            x = self.circularized_profile_dict[band]['radius']
            y = self.circularized_profile_dict[band]['intensity']
            y_err = self.circularized_profile_dict[band]['intensity_err']
        elif which == 'maj':
            x = self.maj_profile_dict[band]['sma']
            y = self.maj_profile_dict[band]['intensity']
            y_err = self.maj_profile_dict[band]['intensity_err']

        _plot_profile(self.fig, self.ax, x, y, y_err)

        self.ipm = InteractiveProfileMasking(self.fig, self.ax, x=x, y=y, mask=np.zeros_like(self.isolist_dict[self.band].sma))

        self.fig.canvas.mpl_connect('pick_event', self.ipm.on_click_select)





