"""
Curves of growth and circular aperture photometry.

This module provides the `COG` container, which stores a curve of
growth (enclosed flux as a function of radius) together with its
normalised version, and `AperturePhotometry`, which measures such
curves in concentric circular apertures on the partial residuals
produced by the AstroPhot 2D deblending (`attila.fitting.deblend_2D`).

In the ATTILA flow it sits after the 2D deblending: the residual image
of each band (data minus the models of all the neighbouring objects) is
integrated in apertures centred on the fitted centre of the target.
`COG` objects are also built from fitted isophotes by
`attila.fitting.isophote.IsophotesAnalysis.get_COG`.
"""
import numpy as np

import shelve

import os

class COG:
    """
    Curve of growth of a single object in a single band.

    The object stores the enclosed flux as a function of radius, the
    associated uncertainties, and the same curve normalised to its
    maximum. It can be saved to and loaded from a `shelve` database and
    drawn on a matplotlib axis.

    Attributes
    ----------
    radius : numpy.ndarray
        The radii at which the enclosed flux is measured. The module
        does not convert units: the values are pixels when the curve
        comes from `AperturePhotometry` or from the fitted isophotes.
    radius_err : numpy.ndarray or float
        The uncertainty on `radius`, in the same units.
    flux : numpy.ndarray
        The flux enclosed within each radius, in the units of the
        image the curve was measured on.
    flux_err : numpy.ndarray or float
        The uncertainty on `flux`, in the same units.
    normalized_flux : numpy.ndarray
        `flux` divided by its maximum value.
    normalized_flux_err : numpy.ndarray or float
        `flux_err` divided by the maximum of `flux`.
    """

    def __init__(self, radius, radius_err, flux, flux_err):
        """
        Store the curve of growth and compute its normalised version.

        The normalisation uses `numpy.max(flux)`, so the normalised
        curve peaks at 1 and a NaN in `flux` makes it entirely NaN.

        Parameters
        ----------
        radius : numpy.ndarray
            The radii of the curve of growth (pixels in the ATTILA
            usage), shape `(n_radii,)`.
        radius_err : numpy.ndarray or float
            The uncertainty on the radii, an array of shape
            `(n_radii,)` or a scalar (0 when the radii are exact).
        flux : numpy.ndarray
            The enclosed flux at each radius, in image units, shape
            `(n_radii,)`.
        flux_err : numpy.ndarray or float
            The uncertainty on the enclosed flux, an array of shape
            `(n_radii,)` or a scalar.
        """

        self.radius = radius
        self.radius_err = radius_err

        self.flux = flux
        self.flux_err = flux_err

        self.normalized_flux = self.flux / np.max(self.flux)
        self.normalized_flux_err = self.flux_err / np.max(self.flux)

    def get_half_light_radius(self):
        """
        Return the radius enclosing half of the maximum flux.

        The method looks for the two samples of the normalised curve
        that are closest to 0.5 from below and from above, and linearly
        interpolates the radius at a normalised flux of 0.5 between
        them.

        Returns
        -------
        float
            The radius at which the normalised flux equals 0.5, in the
            units of `radius`.
        """

        diff_from_half = self.normalized_flux - 0.5

        below = np.flatnonzero(diff_from_half < 0)
        above = np.flatnonzero(diff_from_half > 0)

        # Positions of the samples closest to 0.5 from below and above.
        low_idx = below[np.argmax(diff_from_half[below])]
        high_idx = above[np.argmin(diff_from_half[above])]

        bracket = [low_idx, high_idx]

        return np.interp(0.5, self.normalized_flux[bracket], self.radius[bracket])

    def save(self, path):
        """
        Save the curve of growth to a `shelve` database.

        The radii, the fluxes, their uncertainties and the normalised
        quantities are written under the keys `radius`, `radius_err`,
        `flux`, `flux_err`, `normalized_flux` and
        `normalized_flux_err`.

        Parameters
        ----------
        path : str
            The path of the `shelve` database, without extension (the
            `dbm` backend chooses the file names actually written).
        """

        with shelve.open(path) as db:
            db['radius'] = self.radius
            db['radius_err'] = self.radius_err
            db['flux'] = self.flux
            db['flux_err'] = self.flux_err
            db['normalized_flux'] = self.normalized_flux
            db['normalized_flux_err'] = self.normalized_flux_err

    @classmethod
    def load(cls, path):
        """
        Load a curve of growth from a `shelve` database.

        Only the radii, the fluxes and their uncertainties are read;
        the normalised quantities are recomputed by `__init__`.

        Parameters
        ----------
        path : str
            The path of the `shelve` database written by `save`.

        Returns
        -------
        COG
            The curve of growth stored in the database.
        """

        with shelve.open(path) as db:
            radius = db['radius']
            radius_err = db['radius_err']
            flux = db['flux']
            flux_err = db['flux_err']

        return cls(radius, radius_err, flux, flux_err)

    def plot(self, ax, normed, **kwargs):
        """
        Plot the curve of growth with its error bars.

        The curve is drawn as a line with `ax.plot`, and the
        uncertainties on radius and flux are overlaid with
        `ax.errorbar` (no markers, no connecting line).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis the curve is drawn on.
        normed : bool
            If `True` the normalised flux is plotted, otherwise the
            flux in image units.
        **kwargs
            Keyword arguments passed to `ax.plot`. If `color` is among
            them it is also used for the error bars; otherwise the
            error bars take the next colour of the matplotlib cycle.

        Returns
        -------
        matplotlib.axes.Axes
            The input axis.
        """

        if 'color' not in kwargs:
            color = None
        else:
            color = kwargs['color']

        if normed:
            ax.plot(self.radius, self.normalized_flux, **kwargs)

            ax.errorbar(
                self.radius,
                self.normalized_flux,
                xerr=self.radius_err,
                yerr=self.normalized_flux_err,
                color=color,
                marker='none',
                capsize=2,
                ls='none'
            )

            return ax

        if not normed:

            ax.plot(self.radius, self.flux, **kwargs)

            ax.errorbar(
                self.radius,
                self.flux,
                xerr=self.radius_err,
                yerr=self.flux_err,
                color=color,
                marker='none',
                capsize=2,
                ls='none'
            )

            return ax


from photutils.aperture import aperture_photometry
from typing import Dict, Union

from numpy.typing import ArrayLike
from photutils.isophote import EllipseGeometry
from photutils.aperture import CircularAperture

class AperturePhotometry:
    """
    Circular aperture photometry on the partial residuals of a target.

    For every band, the flux is summed with `photutils` in concentric
    circular apertures of growing radius, centred on the centre of the
    target fitted by AstroPhot, and the result is stored as a `COG`.
    The image used is the partial residual of the 2D deblending, i.e.
    the data from which the models of all the other objects have been
    subtracted.

    Attributes
    ----------
    bands : list of str
        The bands of the input dictionary.
    deblend2D_results : dict
        The per-band results of the 2D deblending (see `__init__`).
    start : int
        The radius of the smallest aperture in pixels (1).
    stop : int
        The upper limit of the aperture radii in pixels (exclusive):
        half the number of rows of the residual image of the first
        band, minus 2.
    step : int
        The increment between consecutive aperture radii in pixels
        (1).
    cogs : dict
        The curve of growth of every band, `{band: COG}`. It exists
        only after `get_aperture_photometry` has been called.
    """

    def __init__(self, deblend2D_results: Dict[str, Dict[str, Union[ArrayLike, EllipseGeometry]]]):
        """
        Store the deblending results and set the default radial grid.

        Parameters
        ----------
        deblend2D_results
            The results of the 2D deblending, one dictionary per band,
            as built by `attila.fitting.deblend_2D.AP_deblend`. The
            entries used here are `residuals` (2D partial residual
            image), `error` (2D noise map), `mask` (2D boolean array,
            `True` for excluded pixels) and `ell_geoms` (the
            `EllipseGeometry` of the target, whose `x0` and `y0` give
            the aperture centre in pixels).
        """

        self.bands = list(deblend2D_results.keys())
        self.deblend2D_results = deblend2D_results

        self.start = 1
        # Largest radius: half the image side (rows) minus a 2 pixel
        # margin, taken from the first band.
        self.stop = int(self.deblend2D_results[self.bands[0]]['residuals'].shape[0]/2.) - 2
        self.step = 1

    def get_aperture_photometry(self):
        """
        Measure the curve of growth of every band.

        For each band, circular apertures with radii
        `numpy.arange(start, stop, step)` pixels are centred on
        `(ell_geoms.x0, ell_geoms.y0)` and the flux inside each of them
        is summed on the partial residual image with
        `photutils.aperture.aperture_photometry`, which also propagates
        the noise map into the flux uncertainty and ignores the masked
        pixels. The curves are stored in the `cogs` attribute, with a
        radius uncertainty of 0.
        """

        self.cogs = {}

        radii = np.arange(self.start, self.stop, self.step)

        for band in self.bands:

            deblend2D_result = self.deblend2D_results[band]

            data = deblend2D_result['residuals']
            error = deblend2D_result['error']
            mask = deblend2D_result['mask']

            ellipse_geometry = deblend2D_result['ell_geoms']

            flux = []
            flux_err = []
            for aperture_radius in radii:

                aperture = CircularAperture(
                    positions=(ellipse_geometry.x0, ellipse_geometry.y0),
                    r=aperture_radius,
                )

                phot_table = aperture_photometry(
                    data=data,
                    apertures=aperture,
                    error=error,
                    mask=mask
                )

                flux.append(phot_table['aperture_sum'].value.squeeze())
                flux_err.append(phot_table['aperture_sum_err'].value.squeeze())



            flux = np.array(flux).squeeze()
            flux_err = np.array(flux_err).squeeze()

            cog = COG(
                radius=radii,
                radius_err=0,
                flux=flux,
                flux_err=flux_err
            )

            self.cogs[band] = cog


    def save(self, path):
        """
        Save the curves of growth of all the bands.

        One sub-directory per band is created under `path` if it does
        not exist, and the curve is written with `COG.save` to the
        `shelve` database `<path>/<band>/cog`.

        Parameters
        ----------
        path : str
            The output directory.
        """

        for band, cog in self.cogs.items():

            if not os.path.exists(f'{path}/{band}'):
                os.makedirs(f'{path}/{band}')

            cog.save(f'{path}/{band}/cog')

def load_cogs(path):
    """
    Load the curves of growth saved by `AperturePhotometry.save`.

    Every entry of the directory `path` is taken as a band name, and
    the `shelve` database `<path>/<band>/cog` is read with `COG.load`.

    Parameters
    ----------
    path : str
        The directory containing one sub-directory per band.

    Returns
    -------
    dict
        The curves of growth, `{band: COG}`.
    """

    cogs = {}
    for band in os.listdir(path):
        cogs[band] = COG.load(f'{path}/{band}/cog')

    return cogs



