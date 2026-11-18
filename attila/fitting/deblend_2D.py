"""
Two-dimensional deblending of a source and its neighbours (AstroPhot).

This module wraps AstroPhot (`astrophot`) to fit, independently in
every band, the cutout of a target source together with all the
neighbouring segments: one Sersic model per segment plus a flat sky,
collected in a group model. From the fit it derives the "partial
residuals" of the target, i.e. the data minus the models of everything
but the target, and the elliptical geometry of the target.

The module provides

- the functional interface to AstroPhot (`AP_initialize`,
  `AP_set_models`, `AP_fit_models`, `AP_get_residuals`,
  `read_AP_targets`, `load_results`);
- `AP_deblend`, which chains those steps, collects the outputs and
  saves them to disk;
- `ApGroupModelParser`, which converts a group model to and from a
  plain dictionary, and the `ipywidgets` helpers (`EditComponent`,
  `add_component`) that edit that dictionary with sliders.

In the ATTILA flow it comes after the `photutils` deblending of the
`Sources`, and its outputs feed the isophote fits
(`attila.fitting.isophote`) and the aperture photometry
(`attila.fitting.COG`). AstroPhot works with `torch` tensors and with
positions in arcsec on the tangent plane; the arrays returned by this
module are `numpy` arrays and the geometries are in pixels.
"""
import astrophot as ap

import numpy as np

from photutils.segmentation import SegmentationImage
from photutils.isophote.geometry import EllipseGeometry
from photutils.aperture import EllipticalAperture

from attila.field.field import Field
from attila.tiling.sources import Source, Sources

import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon

from astropy.io import fits

from typing import List, Dict, Tuple

import pandas as pd

from matplotlib import colormaps as mcmaps
from matplotlib import colors as mcolors

import os
import json

# AstroPhot-specific functions.


def AP_initialize(source:Source, field:Field, psf_path:str=None):
    """
    Build the AstroPhot target image of a source in every band.

    For each band of the multi-band cutout stored in `source.data`, an
    `astrophot.image.Target_Image` is created from the data, the pixel
    scale, the photometric zero point of the field, the variance (the
    square of the noise map) and the mask of the source.

    If `psf_path` is given, the PSF of each band is read from the FITS
    file `<psf_path>_<band>.fits`, its absolute value is taken, and it
    is attached to the target as a `PSF_Image`. The oversampling factor
    of the PSF with respect to the image is parsed from the path itself:
    it is the second element obtained by splitting `psf_path` at the
    underscores, and the pixel scale of the PSF is the image pixel
    scale divided by it.

    Parameters
    ----------
    source
        The source to fit. It must have the multi-band cutout in the
        `data` attribute (see `Source.get_data`) and a boolean `mask`
        attribute with the shape of the cutout.
    field
        The field the source belongs to; its `zpt` attribute must map
        each band to the zero point in magnitudes.
    psf_path
        The common prefix of the PSF files, one per band. `None` means
        that the targets have no PSF and the models are not convolved.

    Returns
    -------
    dict
        The AstroPhot targets, `{band: astrophot.image.Target_Image}`.

    Raises
    ------
    AttributeError
        If `source` has no `data` attribute.
    """

    if not hasattr(source, 'data'):
        raise AttributeError('Data attribute not found in source object')

    targets_dict = {}
    for image in source.data:

        if psf_path is not None:

            # The oversampling factor is encoded in the file prefix,
            # between the first and the second underscore.
            oversampling = float(psf_path.split('_')[1])

            psf_data = np.abs(fits.getdata(psf_path + f'_{image.band}.fits'))

            psf_target_guess = ap.image.PSF_Image(
                data=psf_data.astype(float),
                pixelscale=image.pixel_scale.value/oversampling,
            )

        else:
            psf_target_guess = None

        targets_dict[image.band] = ap.image.Target_Image(
            data=image.data.astype(float),
            pixelscale=image.pixel_scale.value,
            zeropoint=field.zpt[image.band],
            variance=(image.noise)**2,
            mask=source.mask,
            psf=psf_target_guess
        )
    return targets_dict

def read_AP_targets(targets_dict):
    """
    Extract the arrays of the AstroPhot targets as `numpy` arrays.

    Parameters
    ----------
    targets_dict : dict
        The AstroPhot targets, `{band: astrophot.image.Target_Image}`.
        Every target is accessed through its `psf.data`.

    Returns
    -------
    dict
        For every band, a dictionary with the keys `data` (image),
        `noise` (square root of the variance), `mask` and `psf` (PSF
        image), all 2D `numpy` arrays.
    """

    inputs_dict = {}

    for band, target in targets_dict.items():
        inputs_dict[band] = {}
        inputs_dict[band]['data'] = target.data.detach().cpu().numpy()
        inputs_dict[band]['noise'] = np.sqrt(target.variance.detach().cpu().numpy())
        inputs_dict[band]['mask'] = target.mask.detach().cpu().numpy()
        inputs_dict[band]['psf'] = target.psf.data.detach().cpu().numpy()

    return inputs_dict


def AP_set_models(source:Source, targets_dict:Dict[str, ap.image.Target_Image]):
    """
    Build the AstroPhot group model of a source in every band.

    The segmentation map of the cutout (`source.segment`) defines the
    components: every segment gets a `sersic galaxy model`, named
    `object NN` with `NN` the zero-padded label of the segment. For each
    segment the initial guesses are derived with the AstroPhot
    initialisation utilities:

    - the fitting window is the bounding box of the segment, enlarged
      by 5 pixels on every side and clipped to the image;
    - the centre is the flux-weighted centroid of the segment, in
      (x, y) pixels, multiplied by the pixel scale to obtain arcsec;
    - the position angle and the axis ratio come from the
      flux-weighted second moments of the segment.

    A `flat sky model` named `sky level` is added, and all the
    components are collected in a `group model`. Models are convolved
    with the PSF (`psf_mode='full'`) only if the target of the band has
    one. The models are not initialised here.

    Parameters
    ----------
    source
        The source to fit, with the `segment` and `data` attributes;
        `source.label` identifies the target among the segments.
    targets_dict
        The AstroPhot targets, `{band: Target_Image}`, as returned by
        `AP_initialize`.

    Returns
    -------
    models_dict : dict
        The group models, `{band: astrophot.models.AstroPhot_Model}`.
    target_name : str
        The name of the component of the target, `object NN` with `NN`
        the zero-padded `source.label`.
    """

    target_name = f"object {source.label:02d}"

    models_dict = {}
    for image in source.data:

        segment_img = SegmentationImage(source.segment.astype(int))

        # Windows are [[xmin, xmax], [ymin, ymax]] in pixels, keyed by
        # segment label.
        windows = ap.utils.initialize.windows_from_segmentation_map(segment_img)
        windows = ap.utils.initialize.scale_windows(
            windows, image_shape=image.shape, expand_scale=1, expand_border=5
        )

        centers = ap.utils.initialize.centroids_from_segmentation_map(segment_img, image.data)
        PAs = ap.utils.initialize.PA_from_segmentation_map(segment_img, image.data, centers)
        qs = ap.utils.initialize.q_from_segmentation_map(segment_img, image.data, centers, PAs)

        if targets_dict[image.band].has_psf:
            psf_mode = 'full'

        else:
            psf_mode = 'none'

        seg_models = []
        for win in windows:
            seg_models.append(
                ap.models.AstroPhot_Model(
                    name=f"object {win:02d}",
                    window=windows[win],
                    model_type="sersic galaxy model",
                    target=targets_dict[image.band],
                    parameters={
                        # Centroid from pixels to arcsec.
                        "center": np.array(centers[win]) * image.pixel_scale.value,
                        "PA": PAs[win],
                        "q": qs[win],
                    },
                    psf_mode=psf_mode
                )
            )

        sky = ap.models.AstroPhot_Model(
            name=f"sky level",
            model_type="flat sky model",
            target=targets_dict[image.band],
        )

        groupmodel = ap.models.AstroPhot_Model(
            name="group", models=[sky] + seg_models, target=targets_dict[image.band], model_type="group model"
        )

        # groupmodel.initialize()

        models_dict[image.band] = groupmodel

    return models_dict, target_name

def parse_AP_models(models_dict:Dict[str, ap.models.AstroPhot_Model]):
    """
    Placeholder for a parser of the AstroPhot models (not implemented).

    Parameters
    ----------
    models_dict
        The group models, `{band: AstroPhot_Model}`.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError

def AP_fit_models(models_dict:Dict[str, ap.models.AstroPhot_Model]):
    """
    Initialise and fit the group model of every band.

    Each group model is initialised with AstroPhot's automatic
    initialisation, which fills the parameters that have no value, and
    then fitted with `astrophot.fit.Iter`, the iterative optimizer
    that fits the components of a group one at a time (by default with
    Levenberg-Marquardt) and cycles until convergence. The bands are
    fitted independently and the models are updated in place.

    Parameters
    ----------
    models_dict
        The group models, `{band: AstroPhot_Model}`.

    Returns
    -------
    models_dict : dict
        The input dictionary, with the fitted models.
    results_dict : dict
        The optimizer object returned by the fit of every band,
        `{band: astrophot.fit.Iter}`.
    """

    results_dict = {}
    for band, groupmodel in models_dict.items():

        groupmodel.initialize()

        results_dict[band] = ap.fit.Iter(groupmodel, verbose=1).fit()

    return models_dict, results_dict

def AP_get_residuals(
        models_dict:Dict[str, ap.models.AstroPhot_Model],
        results_dict,
        targets_dict:Dict[str, ap.image.Target_Image],
        target_name='object 01'):
    """
    Compute the residuals and the target geometry from the fits.

    For every band three images are produced:

    - the partial residuals: the data minus the models of all the
      components of the group except `target_name` (the sky and the
      neighbouring objects), which isolates the target;
    - the total residuals: the data minus the full group model;
    - the model image of the target component alone.

    An AstroPhot model called with an image adds its flux to that
    image, so the loop over the components accumulates the models of
    all the non-target components in a single model image.

    The fitted parameters of the target are also converted into a
    `photutils` `EllipseGeometry`, in pixels: the centre is
    transformed from the tangent plane (arcsec) to pixel coordinates,
    the semi-major axis is the effective radius `Re` divided by the
    pixel size, the ellipticity is `1 - q`, and the position angle is
    the AstroPhot `PA` plus pi / 2, which converts it to the angle
    from the positive x axis used by `photutils`.

    Parameters
    ----------
    models_dict
        The fitted group models, `{band: AstroPhot_Model}`.
    results_dict : dict
        The optimizer objects returned by `AP_fit_models`, from which
        the fitted parameters of the target are read.
    targets_dict
        The AstroPhot targets, `{band: Target_Image}`.
    target_name : str
        The name of the component of the target in the group models.

    Returns
    -------
    resi_partial_dict : dict
        The partial residuals of every band, 2D `numpy` arrays.
    resi_total_dict : dict
        The total residuals of every band, 2D `numpy` arrays.
    model_partial_dict : dict
        The model image of the target alone in every band, 2D `numpy`
        arrays.
    ell_geoms_dict : dict
        The geometry of the target in every band,
        `{band: photutils.isophote.EllipseGeometry}`, with `sma` equal
        to the effective radius in pixels.
    """

    resi_partial_dict = {}
    model_partial_dict = {}
    resi_total_dict = {}
    ell_geoms_dict = {}

    for band, groupmodel in models_dict.items():

        target = targets_dict[band]
        result = results_dict[band]

        resi_partial = target.copy()
        resi_total = target.copy()

        modelimage = target.model_image()

        # Each call adds the component to `modelimage` in place, so
        # after the loop it holds the sum of the non-target models.
        for model in groupmodel.models.values():
            if model.name == target_name:
                continue

            modelimage_partial = model(image=modelimage)

        resi_partial -= modelimage_partial

        modelimage = target.model_image()

        modelimage_total = groupmodel(image=modelimage)

        resi_total -= modelimage_total

        resi_partial_dict[band] = resi_partial.data.detach().cpu().numpy()

        modelimage = target.model_image()

        modelimage_target = groupmodel.models[target_name](image=modelimage)

        model_partial_dict[band] = modelimage_target.data.detach().cpu().numpy()
        resi_total_dict[band] = resi_total.data.detach().cpu().numpy()

        q = result.model.parameters[target_name]['q'].value.detach().cpu().numpy()
        PA = result.model.parameters[target_name]['PA'].value.detach().cpu().numpy()

        centre = target.plane_to_pixel(result.model.parameters[target_name]['center'].value).detach().cpu().numpy()

        # Effective radius from arcsec to pixels.
        re = (result.model.parameters[target_name]['Re'].value / target.pixel_length).detach().cpu().numpy()

        ell_geoms_dict[band] = EllipseGeometry(
            x0=float(centre[0]),
            y0=float((centre[1])),
            sma=float(re),
            eps=1-float(q),
            pa=float(PA+np.pi/2)
        )

    return resi_partial_dict, resi_total_dict, model_partial_dict, ell_geoms_dict



def load_results(filepath):
    """
    Load the deblending results saved by `AP_deblend.save_results`.

    Every sub-directory of `filepath` is taken as a band. Of the files
    written by `save_results`, the data, the noise map and the PSF are
    not read back.

    Parameters
    ----------
    filepath : str
        The directory containing one sub-directory per band.

    Returns
    -------
    dict
        For every band, a dictionary with the keys `residuals`
        (partial residuals, 2D array), `mask` (2D array),
        `model_image` (model image of the target alone, 2D array),
        `model` (the AstroPhot group model loaded from
        `AstroPhot.yaml`), `target` (the AstroPhot `Target_Image`
        loaded from `target.fits`) and `ell_geoms` (the
        `EllipseGeometry` of the target, in pixels). Note that here
        `model` is the AstroPhot model, whereas in
        `AP_deblend.outputs` the same key holds the model image.
    """

    bands = []
    for _file in os.listdir(filepath):
        if os.path.isdir(os.path.join(filepath, _file)):
            bands.append(_file)

    outputs = {}
    for band in bands:

        outputs[band] = {}
        # outputs[band]['data'] = np.load(os.path.join(filepath, band, 'data.npy'))
        outputs[band]['residuals'] = np.load(os.path.join(filepath, band, 'residuals.npy'))
        outputs[band]['mask'] = np.load(os.path.join(filepath, band, 'mask.npy'))
        outputs[band]['model_image'] = np.load(os.path.join(filepath, band, 'model.npy'))
        # outputs[band]['error'] = np.load(os.path.join(filepath, band, 'error.npy'))
        # outputs[band]['psf'] = np.load(os.path.join(filepath, band, 'psf.npy'))

        outputs[band]['model'] = ap.models.AstroPhot_Model(filename=os.path.join(filepath, band, 'AstroPhot.yaml'))

        outputs[band]['target'] = ap.image.Target_Image(filename=os.path.join(filepath, band, 'target.fits'))

        with open(os.path.join(filepath, band, 'ell_geoms.json'), 'r') as file:
            geometry_dict = json.load(file)

        outputs[band]['ell_geoms'] = EllipseGeometry(**geometry_dict)

    return outputs

class AP_deblend:
    """
    Driver of the AstroPhot 2D deblending of one source.

    The object holds the AstroPhot targets of a source in every band
    and runs the fit of the group models on them. The typical sequence
    is:

    1. `AP_deblend(AP_initialize(source, field, psf_path))`;
    2. `set_models(*AP_set_models(source, targets_dict))`, optionally
       followed by `apply_prior_fit`;
    3. `fit`;
    4. `output_results`, then `save_results` and/or the plotting
       methods.

    Attributes
    ----------
    code : str
        The name of the fitting code, `'astrophot'`.
    inputs : dict
        For every band, the `data`, `noise`, `mask` and `psf` arrays of
        the target (see `read_AP_targets`).
    outputs : dict
        For every band, a dictionary with the keys `data`, `residuals`
        (partial residuals), `mask`, `model` (model image of the target
        alone), `error` (noise map), `ell_geoms` (`EllipseGeometry` of
        the target, in pixels) and `psf`. Set by `output_results`.
    """

    def __init__(self, target_dicts):
        """
        Store the AstroPhot targets and extract their arrays.

        Parameters
        ----------
        target_dicts : dict
            The AstroPhot targets, `{band: Target_Image}`, as returned
            by `AP_initialize`.
        """

        # self.source = source
        # self.field = field
        self.code = "astrophot"

        self._target_dicts = target_dicts

        self.inputs = read_AP_targets(self._target_dicts)


    def set_models(self, models_dict:Dict[str, ap.models.AstroPhot_Model], target_name:str='object 01'):
        """
        Set the group models to fit and the name of the target.

        Parameters
        ----------
        models_dict
            The group models, `{band: AstroPhot_Model}`, e.g. from
            `AP_set_models` or `ApGroupModelParser.get_model_from_dict`.
        target_name
            The name of the component of the target in the group
            models.
        """
        self._models_dict, self._target_name = models_dict, target_name

    def apply_prior_fit(self, prior_models_dict:Dict[str, ap.models.AstroPhot_Model]):
        """
        Copy the parameter values of a previous fit into the models.

        For every band of `prior_models_dict`, and for every component
        whose name is also present in the current group model, the
        values of all the parameters except the centre are copied from
        the previous model. The centres of the current models are left
        untouched.

        Parameters
        ----------
        prior_models_dict
            The group models of a previous fit,
            `{band: AstroPhot_Model}`.
        """

        for band in prior_models_dict.keys():
            prior_models = prior_models_dict[band].models
            new_models = self._models_dict[band].models

            for name in prior_models.keys():
                if name in new_models.keys():

                    for par_name in prior_models[name].parameters.names:
                        # AstroPhot names the two elements of the
                        # centre `center:0` and `center:1`.
                        if par_name != 'center:0' and par_name != 'center:1':
                            new_models[name].parameters[par_name].value = prior_models[name].parameters[par_name].value


    def fit(self):
        """Fit the group models of all bands with `AP_fit_models`."""
        self._models_dict, self._results_dict = AP_fit_models(models_dict=self._models_dict)

    def output_results(self):
        """
        Compute the residuals and collect the outputs of every band.

        The partial residuals, the total residuals, the model image of
        the target and its elliptical geometry are computed with
        `AP_get_residuals` and, together with the input arrays, stored
        in the `outputs` dictionary. The total residuals are kept only
        in the private attribute `_resi_total_dict`.
        """

        self._resi_partial_dict, self._resi_total_dict, self._model_partial_dict, self._ell_geoms_dict = AP_get_residuals(
            models_dict=self._models_dict,
            results_dict=self._results_dict,
            targets_dict=self._target_dicts,
            target_name=self._target_name
        )

        self.outputs = {}
        for band in self._results_dict.keys():

            # sky = self._models_dict[band]['sky']

            self.outputs[band] = {
                'data': self.inputs[band]['data'],
                'residuals': self._resi_partial_dict[band],
                'mask': self.inputs[band]['mask'],
                'model': self._model_partial_dict[band],
                'error': self.inputs[band]['noise'],
                'ell_geoms': self._ell_geoms_dict[band],
                'psf': self.inputs[band]['psf']
            }

    def save_results(self, filepath, plots=True):
        """
        Save the models, the targets and the output arrays to disk.

        For every band the following files are written in
        `<filepath>/<band>/`, creating the directories if needed:
        `AstroPhot.yaml` (the fitted group model), `target.fits` (the
        AstroPhot target), `data.npy`, `residuals.npy`, `mask.npy`,
        `model.npy`, `error.npy`, `psf.npy` (the arrays of `outputs`)
        and `ell_geoms.json` (the `x0`, `y0`, `sma`, `eps` and `pa` of
        the target geometry, in pixels and radians).

        If `plots` is `True`, the figures of `plot_inputs` and
        `plot_outputs` are saved as `<filepath>/inputs.jpg` and
        `<filepath>/outputs.jpg`. Both figures have two panels, one
        per band.

        Parameters
        ----------
        filepath : str
            The output directory.
        plots : bool
            If `True` the diagnostic figures are also saved.
        """

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for band in self.outputs.keys():

            if not os.path.exists(os.path.join(filepath, band)):
                os.makedirs(os.path.join(filepath, band))

            self._models_dict[band].save(filename=os.path.join(filepath, band, 'AstroPhot.yaml'))

            self._target_dicts[band].save(filename=os.path.join(filepath, band, 'target.fits'))

            np.save(os.path.join(filepath, band, 'data.npy'), self.outputs[band]['data'])
            np.save(os.path.join(filepath, band, 'residuals.npy'), self.outputs[band]['residuals'])
            np.save(os.path.join(filepath, band, 'mask.npy'), self.outputs[band]['mask'])
            np.save(os.path.join(filepath, band, 'model.npy'), self.outputs[band]['model'])
            np.save(os.path.join(filepath, band, 'error.npy'), self.outputs[band]['error'])
            np.save(os.path.join(filepath, band, 'psf.npy'), self.outputs[band]['psf'])

            geometry_dict = {
                'x0': self.outputs[band]['ell_geoms'].x0,
                'y0': self.outputs[band]['ell_geoms'].y0,
                'sma': self.outputs[band]['ell_geoms'].sma,
                'eps': self.outputs[band]['ell_geoms'].eps,
                'pa': self.outputs[band]['ell_geoms'].pa
            }

            with open(os.path.join(filepath, band, 'ell_geoms.json'), 'w+') as file:
                json.dump(geometry_dict, file, indent=4)

        if plots:

            fig_inp, axs_inp = plt.subplots(1, 2, figsize=(10, 5))

            self.plot_inputs(axs_inp)

            fig_inp.savefig(os.path.join(filepath, 'inputs.jpg'))


            fig_out, axs_out = plt.subplots(1, 2, figsize=(10, 5))

            self.plot_outputs(axs_out)

            fig_out.savefig(os.path.join(filepath, 'outputs.jpg'))






    def plot_inputs(self, axs:List[plt.Axes]):
        """
        Plot the signal-to-noise map of the input data in every band.

        The map is the data divided by the noise, with the masked
        pixels set to -3. It is drawn in two layers: a grey scale for
        signal-to-noise ratios between -3 and 3, and a logarithmic
        colour scale between 3 and 10 for the pixels above 3.

        Parameters
        ----------
        axs
            The axes to draw on, one per band, in the order of the
            `inputs` dictionary.
        """

        bottom = mcmaps['binary']
        top_parent = mcmaps['viridis']
        color_list = [
            bottom(0.99),
            top_parent(0.15),
            top_parent(0.33),
            top_parent(0.66),
            top_parent(0.99),
        ]
        top = mcolors.LinearSegmentedColormap.from_list("top", color_list, N=256)

        for i, (band, input) in enumerate(self.inputs.items()):

            masked_SNR_map = np.where(input['mask']==0, input['data']/input['noise'], -3) #np.ma.masked_where(input['mask'], input['data']/input['noise'])

            img = masked_SNR_map

            img_bot = np.where(img <= 3, img, 3)
            axs[i].imshow(img_bot, vmin=-3, vmax=3, origin='lower', cmap=bottom)

            if np.any(img >= 3):
                img_top = np.ma.masked_where(img <= 3, img)
                norm = mcolors.LogNorm(vmin=3, vmax=10)
                axs[i].imshow(img_top, norm=norm, origin='lower', cmap=top)

            # axs[i].imshow(masked_SNR_map, vmin=-3, vmax=5, origin='lower', cmap='viridis')

            axs[i].set_title(band)

    def plot_outputs(self, axs:List[plt.Axes]):
        """
        Plot the partial residuals and the target ellipse in every band.

        The partial residuals divided by the noise are drawn with the
        same two-layer colour scale as `plot_inputs` (masked pixels set
        to -3). The ellipse of the fitted target, with semi-major axis
        equal to the effective radius, and its centre are overlaid in
        red.

        Parameters
        ----------
        axs
            The axes to draw on, one per band, in the order of the
            `outputs` dictionary.
        """

        bottom = mcmaps['binary']
        top_parent = mcmaps['viridis']
        color_list = [
            bottom(0.99),
            top_parent(0.15),
            top_parent(0.33),
            top_parent(0.66),
            top_parent(0.99),
        ]
        top = mcolors.LinearSegmentedColormap.from_list("top", color_list, N=256)

        for i, (band, output) in enumerate(self.outputs.items()):

            input = self.inputs[band]

            masked_SNR_resi = np.where(input['mask']==0, output['residuals']/input['noise'], -3) #np.ma.masked_where(input['mask'], resi/input['noise'])

            img = masked_SNR_resi

            img_bot = np.where(img <= 3, img, 3)
            axs[i].imshow(img_bot, vmin=-3, vmax=3, origin='lower', cmap=bottom)

            if np.any(img >= 3):
                img_top = np.ma.masked_where(img <= 3, img)
                norm = mcolors.LogNorm(vmin=3, vmax=10)
                axs[i].imshow(img_top, norm=norm, origin='lower', cmap=top)

            # axs[i].imshow(masked_SNR_resi, origin='lower', cmap='viridis')

            ell_aperture = EllipticalAperture(
                positions=(output['ell_geoms'].x0, output['ell_geoms'].y0),
                a=output['ell_geoms'].sma,
                b=output['ell_geoms'].sma * (1 - output['ell_geoms'].eps),
                theta=output['ell_geoms'].pa
            )

            ell_aperture.plot(ax=axs[i], color='red', lw=2)

            axs[i].plot(output['ell_geoms'].x0, output['ell_geoms'].y0, 'rx', ms=10)

            axs[i].set_title(band)


class ApGroupModelParser(dict):
    """
    Plain-dictionary view of an AstroPhot group model.

    The parser converts a group model into a nested dictionary of
    `numpy` values, which can be edited by hand or with the
    `ipywidgets` sliders of `EditComponent`, and builds a new group
    model from the edited dictionary. The object is itself the
    dictionary, keyed by component name:

        {name: {'model_type': str,
                'parameters': {param: {'value': ...,
                                       'limits': [lo, hi]}},
                'window': {'center': ...,
                           'pixel_shape': ...}}}

    The `center` parameter has only the `value` entry. The window
    centre is in arcsec on the tangent plane and its shape in pixels,
    both in (x, y) order.

    Attribute access is redirected to the dictionary items, so
    `parser.name` is `parser['name']` and assigning an attribute
    creates an item. The only real attribute is `model`, the AstroPhot
    group model the dictionary was read from.

    Attributes
    ----------
    model : astrophot.models.AstroPhot_Model
        The group model given at creation.
    """

    def __init__(self, model:ap.models.AstroPhot_Model):
        """
        Store the group model and fill the dictionary from it.

        Parameters
        ----------
        model
            The AstroPhot group model to parse.
        """

        # Written to `__dict__` directly because `__setattr__` is
        # redirected to the dictionary items.
        self.__dict__['model'] = model
        self.get_dict_from_model()

    def __getattr__(self, item):
        """Return the dictionary item `item` for a missing attribute."""
        return super().__getitem__(item)

    def __setattr__(self, item, value):
        """Store `value` as the dictionary item `item`."""
        return super().__setitem__(item, value)


    def get_dict_from_model(self) -> None:
        """
        Fill the dictionary with the components of the group model.

        The dictionary is cleared and, for every component of `model`,
        the model type, the value and the limits of every parameter
        and the centre and pixel shape of the fitting window are
        stored as `numpy` values. The two elements of the centre
        (`center:0`, `center:1` in the AstroPhot parameter names) are
        merged in a single `center` entry, for which only the value is
        stored. A missing limit is stored as `None`.
        """

        self.clear()
        for name, component in self.model.models.items():

            self[name] = {}
            self[name]['model_type'] = component.model_type

            self[name]['parameters'] = {}
            for param_name in component.parameters.names:
                if param_name == 'center:0' or param_name == 'center:1':
                    self[name]['parameters']['center'] = {}
                    self[name]['parameters']['center']['value'] = component.parameters['center'].value.detach().cpu().numpy()
                else:
                    self[name]['parameters'][param_name] = {}
                    self[name]['parameters'][param_name]['value'] = component.parameters[param_name].value.detach().cpu().numpy()
                    self[name]['parameters'][param_name]['limits'] = []
                    for limit in component.parameters[param_name].limits:
                        if limit is not None:
                            self[name]['parameters'][param_name]['limits'].append(limit.detach().cpu().numpy())
                        else:
                            self[name]['parameters'][param_name]['limits'].append(None)

            self[name]['window'] = {}
            self[name]['window']['center'] = component.window.center.detach().cpu().numpy()
            self[name]['window']['pixel_shape'] = component.window.pixel_shape.detach().cpu().numpy()

    def get_model_from_dict(self, target) -> ap.models.AstroPhot_Model:
        """
        Build a new AstroPhot group model from the dictionary.

        Every component of the dictionary is turned into an AstroPhot
        model of its `model_type`, with its `parameters` and a window
        rebuilt from the stored centre and pixel shape with the pixel
        scale of `target`. The component named `sky level` is always
        built as a `flat sky model` without a window. PSF convolution
        is enabled (`psf_mode='full'`) if the target has a PSF. The
        components are collected in a group model named `group model`.

        Parameters
        ----------
        target : astrophot.image.Target_Image
            The target image the new models are attached to.

        Returns
        -------
        astrophot.models.AstroPhot_Model
            The new group model.
        """

        if target.has_psf:
            psf_mode = 'full'
        else:
            psf_mode = 'none'

        component_list = []
        for name, component in self.items():

            if name != 'sky level':
                window = ap.image.image_object.Window(
                    **component['window'],
                    pixelscale=target.pixelscale
                )

                component_list.append(
                    ap.models.AstroPhot_Model(
                        name=name,
                        model_type=component['model_type'],
                        parameters=component['parameters'],
                        window=window,
                        psf_mode=psf_mode,
                        target=target,
                    )
                )

            else:
                sky = ap.models.AstroPhot_Model(
                    name=f"sky level",
                    model_type="flat sky model",
                    target=target,
                    parameters=component['parameters'],
                )

                component_list.append(sky)

        groupmodel = ap.models.AstroPhot_Model(
            name="group model",
            model_type="group model",
            target=target,
            models=component_list,
        )

        return groupmodel

import ipywidgets as widgets
from IPython.display import display, clear_output

from functools import partial

from copy import deepcopy

def _create_box_x_slider(component):
    """
    Create the slider for the x size of a component window.

    Parameters
    ----------
    component : dict
        The component dictionary of an `ApGroupModelParser`.

    Returns
    -------
    ipywidgets.IntSlider
        A slider from 10 to 150 pixels in steps of 10, initialised at
        the first element of `component['window']['pixel_shape']`.
    """

    value = component['window']['pixel_shape']

    box_x_size_slider = widgets.IntSlider(
        value=value[0],
        min=10,
        max=150,
        step=10,
        description='Box X Size:',
        continuous_update=True
    )

    return box_x_size_slider

def _create_box_y_slider(component):
    """
    Create the slider for the y size of a component window.

    Parameters
    ----------
    component : dict
        The component dictionary of an `ApGroupModelParser`.

    Returns
    -------
    ipywidgets.IntSlider
        A slider from 10 to 150 pixels in steps of 10, initialised at
        the second element of `component['window']['pixel_shape']`.
    """

    value = component['window']['pixel_shape']

    box_y_size_slider = widgets.IntSlider(
        value=value[1],
        min=10,
        max=150,
        step=10,
        description='Box Y Size:',
        continuous_update=True
    )

    return box_y_size_slider

def _create_q_slider(component):
    """
    Create the slider for the axis ratio `q` of a component.

    The slider spans the limits stored for the parameter, in steps of
    0.001. An initial value that rounds to 1.00 or 0.00 is replaced by
    0.99 or 0.01. The parameter dictionary is also printed.

    Parameters
    ----------
    component : dict
        The component dictionary of an `ApGroupModelParser`, with a
        `q` parameter.

    Returns
    -------
    ipywidgets.FloatSlider
        The slider of the axis ratio.
    """

    print(component['parameters']['q'])

    value = component['parameters']['q']['value']
    limit_low = component['parameters']['q']['limits'][0]
    limit_high = component['parameters']['q']['limits'][1]

    # Keep the starting value strictly inside the (0, 1) interval.
    if np.round(value, 2) == 1.0:
        value = 0.99

    if np.round(value, 2) == 0.0:
        value = 0.01

    q_slider = widgets.FloatSlider(
        value=value,
        min=limit_low,
        max=limit_high,
        step=0.001,
        description='q:',
        continuous_update=True
    )
    return q_slider

def _create_PA_slider(component):
    """
    Create the slider for the position angle `PA` of a component.

    Parameters
    ----------
    component : dict
        The component dictionary of an `ApGroupModelParser`, with a
        `PA` parameter.

    Returns
    -------
    ipywidgets.FloatSlider
        A slider spanning the limits stored for the parameter, in
        steps of 0.1 rad.
    """

    value = component['parameters']['PA']['value']
    limit_low = component['parameters']['PA']['limits'][0]
    limit_high = component['parameters']['PA']['limits'][1]

    PA_slider = widgets.FloatSlider(
        value=value,
        min=limit_low,
        max=limit_high,
        step=0.1,
        description='PA:',
        continuous_update=True
    )
    return PA_slider

def _create_F_slider(component):
    """
    Create the slider for the sky level `F` of a flat sky component.

    `F` is the AstroPhot sky parameter, the base-10 logarithm of the
    sky flux per arcsec^2. The slider is bounded by
    `value - 0.1 * value` and `value + 0.1 * value`, in steps of 0.001.

    Parameters
    ----------
    component : dict
        The component dictionary of an `ApGroupModelParser`, with an
        `F` parameter.

    Returns
    -------
    ipywidgets.FloatSlider
        The slider of the sky level.
    """

    value = component['parameters']['F']['value']
    limit_low = value - 0.1*value
    limit_high = value + 0.1*value

    F_slider = widgets.FloatSlider(
        value=value,
        min=limit_low,
        max=limit_high,
        step=0.001,
        description='F:',
        continuous_update=True
    )
    return F_slider


def add_component(models_dict, new_type, name, window, priors):
    """
    Add a new component to the model dictionaries of all the bands.

    The component is added, in every band where `name` is not already
    present, as a dictionary with the given model type, window and
    parameters, in the format of `ApGroupModelParser`. The centre of
    the component is set to the centre of the window.

    If `priors` is `None`, default parameter values and limits are
    created for the requested model type the first time they are
    needed; the same dictionary is then used for all the bands.

    Parameters
    ----------
    models_dict : dict
        The model dictionaries, `{band: ApGroupModelParser}`. They are
        modified in place.
    new_type : str
        The AstroPhot model type of the new component:
        `'sersic galaxy model'`, `'moffat galaxy model'` or
        `'flat sky model'` when `priors` is `None`.
    name : str
        The name of the new component.
    window : dict
        The fitting window of the component, with the keys `center`
        (arcsec on the tangent plane, (x, y)) and `pixel_shape`
        (pixels, (x, y)).
    priors : dict or None
        The parameters of the component,
        `{param: {'value': ..., 'limits': [lo, hi]}}`. `None` means
        the defaults of the model type. It must contain a `center`
        entry, whose value is overwritten.

    Returns
    -------
    dict
        The input `models_dict`.

    Raises
    ------
    NotImplementedError
        If `priors` is `None` and `new_type` is not one of the three
        supported types.
    ValueError
        If the component has to be added and `window` is `None`.
    """

    for band in models_dict.keys():

        if priors is None:

            if new_type == "sersic galaxy model":
                priors = {
                    'center': {'value': np.array([0, 0]), 'limits': [None, None]},
                    'q': {'value': np.array(0.9), 'limits': [np.array(0), np.array(1)]},
                    'PA': {'value': np.array(0.5), 'limits': [np.array(0), np.array(np.pi)]},
                    'n': {'value': np.array(1), 'limits': [np.array(0), np.array(10)]},
                    'Re': {'value': np.array(1), 'limits': [np.array(0), None]},
                    'Ie': {'value': np.array(1), 'limits': [np.array(0), None]},
                }
            elif new_type == "moffat galaxy model":
                priors = {
                    'center': {'value': np.array([0, 0]), 'limits': [None, None]},
                    'q': {'value': np.array(0.9), 'limits': [np.array(0), np.array(1)]},
                    'PA': {'value': np.array(0.5), 'limits': [np.array(0), np.array(np.pi)]},
                    'n': {'value': np.array(1), 'limits': [np.array(0), np.array(10)]},
                    'Rd': {'value': np.array(10), 'limits': [np.array(0), None]},
                }
            elif new_type == "flat sky model":
                priors = {
                    'F': {'value': 1, 'limits': [np.array(-3), np.array(3)]},
                }
            else:
                raise NotImplementedError(f"Component {new_type} not found in the model.")

        component_names = models_dict[band].keys()

        if name not in component_names:

            if window is None:
                raise ValueError("Scemo!!")
            else:
                window = window

            priors['center']['value'] = window['center']

            models_dict[band][name] = {
                'model_type': new_type,
                'window': window,
                'parameters': priors,
            }

    return models_dict

def _update_component(component, new_type, priors):
    """
    Return a copy of a component with a new model type.

    The window of the input component is kept, while the parameters
    are replaced by `priors` or, if `priors` is `None`, by the default
    values and limits of the new model type (the same defaults as in
    `add_component`).

    Parameters
    ----------
    component : dict
        The component dictionary to convert.
    new_type : str
        The new AstroPhot model type: `'sersic galaxy model'`,
        `'moffat galaxy model'` or `'flat sky model'` when `priors` is
        `None`.
    priors : dict or None
        The parameters of the new component. `None` means the defaults
        of the model type.

    Returns
    -------
    dict
        The new component dictionary, with the keys `model_type`,
        `window` and `parameters`.

    Raises
    ------
    NotImplementedError
        If `priors` is `None` and `new_type` is not one of the three
        supported types.
    """

    if priors is None:

        if new_type == "sersic galaxy model":
            priors = {
                    'center': {'value': np.array([0, 0]), 'limits': [None, None]},
                    'q': {'value': np.array(0.9), 'limits': [np.array(0), np.array(1)]},
                    'PA': {'value': np.array(0.5), 'limits': [np.array(0), np.array(np.pi)]},
                    'n': {'value': np.array(1), 'limits': [np.array(0), np.array(10)]},
                    'Re': {'value': np.array(1), 'limits': [np.array(0), None]},
                    'Ie': {'value': np.array(1), 'limits': [np.array(0), None]},
                }
        elif new_type == "moffat galaxy model":
            priors = {
                    'center': {'value': np.array([0, 0]), 'limits': [None, None]},
                    'q': {'value': np.array(0.9), 'limits': [np.array(0), np.array(1)]},
                    'PA': {'value': np.array(0.5), 'limits': [np.array(0), np.array(np.pi)]},
                    'n': {'value': np.array(1), 'limits': [np.array(0), np.array(10)]},
                    'Rd': {'value': np.array(10), 'limits': [np.array(0), None]},
                }
        elif new_type == "flat sky model":
            priors = {
                    'F': {'value': 1, 'limits': [np.array(-3), np.array(3)]},
                }
        else:
            raise NotImplementedError(f"Component {new_type} not found in the model.")


    return {
        'model_type': new_type,
        'window': component['window'],
        'parameters': priors,
    }


class EditComponent:
    """
    Interactive editor of one component of the model dictionaries.

    The editor shows, in a Jupyter notebook, a dropdown with the model
    type of a component and a set of sliders for its window size and
    parameters. It works on the component of the reference band; the
    slider values are written to that component as they move, and the
    "Accept" button copies the edited component to all the bands.

    The usual sequence is `EditComponent(...)`, `initialize()`,
    `display()`.

    Attributes
    ----------
    models_dict : dict
        The model dictionaries, `{band: ApGroupModelParser}`.
    component : dict
        The component being edited.
    name : str
        The name of the component.
    model_type : str
        The current AstroPhot model type of the component.
    window : dict
        The window of the component at creation.
    dropdown : ipywidgets.Dropdown
        The model type selector.
    sliders : list
        The dropdown followed by the sliders of the current type.
    HBox : ipywidgets.HBox
        The box containing `sliders`.
    accept_button, reject_button : ipywidgets.Button
        The buttons created by `initialize`. Only the "Accept" button
        is connected to a callback and displayed.
    """
    def __init__(self, models_dict, reference_band, name):
        """
        Select the component to edit.

        Parameters
        ----------
        models_dict : dict
            The model dictionaries, `{band: ApGroupModelParser}`.
        reference_band : str
            The band whose component is loaded in the editor.
        name : str
            The name of the component to edit.
        """

        self.models_dict = models_dict

        self.component = self.models_dict[reference_band][name]
        self.name = name
        self.model_type = self.component['model_type']
        self.window = self.component['window']

    def initialize(self):
        """
        Create the dropdown, the sliders and the buttons.

        The dropdown is connected to `update_type` and the "Accept"
        button to `accept_changes`. The "Reject" button is created but
        not connected.
        """

        self.dropdown = widgets.Dropdown(
            options=['sersic galaxy model', 'moffat galaxy model', 'flat sky model'],
            description='Type:',
            value=self.model_type
        )
        self.sliders = [self.dropdown]
        self.get_paramter_widgets()

        self.dropdown.observe(self.update_type, names='value')

        self.accept_button = widgets.Button(description="Accept")

        self.reject_button = widgets.Button(description="Reject")

        self.accept_button.on_click(self.accept_changes)


        # self.reject_button.on_click(self.reject_changes)


    def get_paramter_widgets(self):
        """
        Build the sliders of the current model type and connect them.

        The `sliders` list is rebuilt as the dropdown followed by:

        - for Sersic and Moffat models, the x and y window size
          sliders and the `q` and `PA` sliders;
        - for a flat sky model, the `F` slider.

        The `q`, `PA` and `F` sliders are connected to
        `get_slider_value`. The window size sliders are connected to
        `get_box_slider_value` for the Sersic model only. Finally the
        widgets are packed in the `HBox` attribute.
        """

        self.sliders = [self.dropdown]

        if self.model_type == "sersic galaxy model":
            self.sliders.append(_create_box_x_slider(self.component))
            self.sliders.append(_create_box_y_slider(self.component))
            self.sliders.append(_create_q_slider(self.component))
            self.sliders.append(_create_PA_slider(self.component))
        elif self.model_type == "moffat galaxy model":
            self.sliders.append(_create_box_x_slider(self.component))
            self.sliders.append(_create_box_y_slider(self.component))
            self.sliders.append(_create_q_slider(self.component))
            self.sliders.append(_create_PA_slider(self.component))
        elif self.model_type == "flat sky model":
            self.sliders.append(_create_F_slider(self.component))

        # Slider positions in the list: 0 dropdown, 1 box x, 2 box y,
        # 3 q, 4 PA (galaxy models); 0 dropdown, 1 F (sky model).
        if self.model_type == 'sersic galaxy model':
            slider_q = partial(self.get_slider_value, parameter_name='q')
            self.sliders[3].observe(slider_q, names='value')
            slider_pa = partial(self.get_slider_value, parameter_name='PA')
            self.sliders[4].observe(slider_pa, names='value')

            slider_box_x = partial(self.get_box_slider_value, which='x')
            self.sliders[1].observe(slider_box_x, names='value')
            slider_box_y = partial(self.get_box_slider_value, which='y')
            self.sliders[2].observe(slider_box_y, names='value')

        elif self.model_type == 'moffat galaxy model':
            slider_q = partial(self.get_slider_value, parameter_name='q')
            self.sliders[3].observe(slider_q, names='value')
            slider_pa = partial(self.get_slider_value, parameter_name='PA')
            self.sliders[4].observe(slider_pa, names='value')

        elif self.model_type == 'flat sky model':
            slider_F = partial(self.get_slider_value, parameter_name='F')
            self.sliders[1].observe(slider_F, names='value')

        self.HBox = widgets.HBox(self.sliders)

    def update_type(self, change):
        """
        Change the model type of the component from the dropdown.

        The component is replaced by a new one of the selected type
        with default parameters and the same window (see
        `_update_component`), and the sliders are rebuilt and
        redisplayed. A debug message is printed.

        Parameters
        ----------
        change : dict
            The `ipywidgets` change event; `change['new']` is the
            selected model type.
        """
        print(f"Dropdown changed to: {change['new']}")  # Debug

        self.model_type = change['new']

        for band in self.models_dict.keys():
            self.component = _update_component(self.component, self.model_type, None)

        self.get_paramter_widgets()
        self.refresh_display()

    def get_slider_value(self, change, parameter_name):
        """
        Write a slider value to a parameter of the component.

        Parameters
        ----------
        change : dict
            The `ipywidgets` change event; `change['new']` is the new
            value, which is rounded to 3 decimals.
        parameter_name : str
            The name of the parameter to update (`q`, `PA` or `F`).
        """
        self.component['parameters'][parameter_name]['value'] = np.round(change['new'], 3)

    def get_box_slider_value(self, change, which):
        """
        Write a slider value to the window size of the component.

        Parameters
        ----------
        change : dict
            The `ipywidgets` change event; `change['new']` is the new
            size in pixels.
        which : str
            The side of the window to update, `'x'` or `'y'`.
        """
        if which == 'x':
            self.component['window']['pixel_shape'][0] = change['new']
        elif which == 'y':
            self.component['window']['pixel_shape'][1] = change['new']

    def refresh_display(self):
        """Clear the cell output and display the widgets again."""
        print(f"Refreshing display for component: {self.name}")  # Debug
        clear_output(wait=True)
        display(self.HBox)
        display(self.accept_button)

    def display(self):
        """Display the sliders and the "Accept" button."""
        print(f"Displaying component: {self.name}")  # Debug
        display(self.HBox)
        display(self.accept_button)



    def accept_changes(self, change=None):
        """
        Copy the edited component to all the bands.

        A deep copy of the component is stored under its name in the
        model dictionary of every band.

        Parameters
        ----------
        change : ipywidgets.Button, optional
            The button that triggered the callback (not used).
        """

        for band in self.models_dict.keys():
            self.models_dict[band][self.name] = deepcopy(self.component)




# Retired code, kept for reference. The display of the model, of the
# component windows and the mask drawing / dilation that `PlotModel`,
# `_plot_image` and `plot_window` provided are now handled by
# `PlotApModel` and `_plot_window` in `attila.interactive.fitprep`.
# `InteractiveWindowSelector` has no counterpart there.

# from matplotlib import colormaps as mcmaps
# from matplotlib.colors import Normalize 
# from matplotlib.widgets import Button
# from threading import Event
# import shapely as sh
# from shapely.plotting import plot_polygon
# from attila.interactive.masking_segment import InteractiveMaskDrawer, InteractiveMaskDilator

# def _plot_window_dict(ax, window_dict, pixel_scale, **kwargs):

#     dx = window_dict['pixel_shape'][0]
#     dy = window_dict['pixel_shape'][1]

#     center_arcsec = window_dict['center']

#     center_pix = [center_arcsec[0] / pixel_scale, center_arcsec[1] / pixel_scale]

#     # ax.add_patch(plt.Rectangle((x0, y0), dx, dy, fill=False, edgecolor='C1', lw=2, **kwargs))

#     x0 = center_pix[0] - dx / 2
#     y0 = center_pix[1] - dy / 2


#     rct = sh.Polygon([(x0, y0), (x0 + dx, y0), (x0 + dx, y0 + dy), (x0, y0 + dy)])
#     plt_poly = plot_polygon(rct, ax=ax, add_points=False, ls='--', ec='C1', fc='none', hatch='//', alpha=1)

#     return ax, plt_poly

# class InteractiveWindowSelector:
#     def __init__(self, fig, ax, pixelscale):
#         self.fig = fig
#         self.ax = ax
#         self.pixelscale = pixelscale
#         self.window_dict = None 
#         self.clicks = []

#         self.windows = []

#     def on_click(self, event):

#         # Handle the click event
#         if event.inaxes == self.ax:

#             if event.button == 1:  # Left click
#                 self.clicks.append((event.xdata, event.ydata))

#                 if len(self.clicks) == 1:

#                     self.ax.plot(self.clicks[0][0], self.clicks[0][1], marker='o', mec='C1', mfc=None) 

#                 elif len(self.clicks) == 2:
#                     center = self.clicks[0]

#                     diagonal = np.array(self.clicks[1])

#                     dx = int(2 * abs(diagonal[0] - center[0]))
#                     dy = int(2 * abs(diagonal[1] - center[1]))

#                     numpy_pixelscale = self.pixelscale.detach().cpu().numpy()

#                     center_arcsec = np.array(center) * np.array([numpy_pixelscale[0][0], numpy_pixelscale[1][1]])

#                     # self.ax.figure.canvas.mpl_disconnect(self.cid)

#                     self.window_dict = {
#                         'center': center_arcsec,
#                         'pixel_shape': [dx, dy],
#                     }

#                     self.center_plot = self.ax.plot(self.clicks[1][0], self.clicks[1][1], marker='x', mec='C1', mfc=None)
#                     self.ax, self.plt_poly = _plot_window_dict(self.ax, self.window_dict, pixel_scale=numpy_pixelscale[0][0])

#                     self.windows.append(self.window_dict)
#                     self.fig.canvas.draw()

#             elif event.button == 3:
#                 self.clicks = []

#                 self.center_plot[0].remove()

#                 self.plt_poly.remove()

#                 self.fig.canvas.draw()

#             elif event.button == 2:  # Middle click
#                 self.ax.figure.canvas.mpl_disconnect(self.cid)


# def _plot_image(ax, image, vlims, colormaps, mask=None, **kwargs):

#     if mask is not None:
#         image = np.where(mask, 0, image)

#     bottom_image = np.where(image < vlims[0], vlims[0], image)
#     bottom_image = np.where(bottom_image > vlims[1], vlims[1], bottom_image)

#     top_image = np.where(image > vlims[2], vlims[2], image)
#     top_image = np.where(top_image < vlims[1], np.nan, top_image)

#     ax.imshow(bottom_image, cmap=colormaps[0], vmin=vlims[0], vmax=vlims[1], origin='lower', **kwargs)

#     ax.imshow(top_image, cmap=colormaps[1], vmin=vlims[1], vmax=vlims[2], origin='lower', **kwargs)

#     return ax

# def plot_window(ax, window, **kwargs):

#     dx = window.pixel_shape[0].detach().cpu().numpy()
#     dy = window.pixel_shape[1].detach().cpu().numpy()

#     center_arcsec = window.center.detach().cpu().numpy()

#     pixel_scale = window.pixelscale.detach().cpu().numpy()

#     center_pix = [center_arcsec[0] / pixel_scale[0][0], center_arcsec[1] / pixel_scale[1][1]]

#     x0 = center_pix[0] - dx / 2
#     y0 = center_pix[1] - dy / 2

#     ax.add_patch(plt.Rectangle((x0, y0), dx, dy, fill=False, edgecolor='C1', lw=2, **kwargs))

#     return ax

# class PlotModel:
#     def __init__(self, results):
#         self.results = results
#         self.current_band = list(results.keys())[0]

#         # reapply the standard masks

#         for band in results.keys():
#             results[band]['target'].mask = results[band]['mask']

#         self.working_component_name = None

#         self.dropdown_band = widgets.Dropdown(
#             options=list(results.keys()),
#             description='Band:',
#             value=self.current_band
#         )

#         display(self.dropdown_band)

#         self.models_dict = {band: ApGroupModelParser(self.results[band]['model']) for band in self.results.keys()}

#         # Initialize the figure and axes
#         self.fig, (self.ax_target, self.ax_model, self.ax_resi) = plt.subplots(1, 3, figsize=(15, 5))
#         self.vmed = 10
#         self.vmax = 50
#         self.bottom_cmap = mcmaps['coolwarm']
#         self.top_cmap = mcmaps['RdYlGn']

#         # Load the initial band data
#         self.load_band(self.current_band)

#         # Observe changes in the dropdown
#         self.dropdown_band.observe(self.monitor_band, names='value')

#         self.selector = None
#         self.masker = None

        

#     def load_band(self, band):
#         """Load the data for the selected band."""
#         result_band = self.results[band]
#         self.model = result_band['model']
#         self.target = result_band['target']
#         self.model.target = self.target

#         self.plot()

#     def monitor_band(self, change):
#         """Handle changes in the dropdown value."""
#         new_band = change['new']
#         if new_band != self.current_band:
#             self.current_band = new_band
#             self.load_band(new_band)
        

#     def plot(self):
        
#         target_data = self.target.data.detach().cpu().numpy()
#         target_variance = self.target.variance.detach().cpu().numpy()

#         target_noise = np.sqrt(target_variance)

#         self.ax_target.imshow(target_data/target_variance, cmap='viridis')

#         self.ax_target = _plot_image(
#             self.ax_target, 
#             target_data/target_noise, 
#             vlims=[-3, self.vmed, self.vmax], 
#             colormaps=[self.bottom_cmap, self.top_cmap],
#             mask=self.target.mask.detach().cpu().numpy(),
#         )

#         if self.working_component_name is not None:

#             if self.working_component_name in self.model.models.keys():

#                 base_image = self.target.model_image()

#                 for model in self.model.models.values():
#                     if model.name != self.working_component_name:
#                         continue

#                     model_partial = model(image=base_image)

#                 model_image = model_partial.data.detach().cpu().numpy()

#                 self.ax_model = _plot_image(
#                     self.ax_model,
#                     model_image/target_noise,
#                     vlims=[-3, self.vmed, self.vmax], 
#                     colormaps=[self.bottom_cmap, self.top_cmap],
#                 )

#                 resi_image = target_data - model_image

#                 self.ax_resi = _plot_image(
#                     self.ax_resi,
#                     resi_image/target_noise,
#                     vlims=[-3, self.vmed, self.vmax], 
#                     colormaps=[self.bottom_cmap, self.top_cmap],
#                     mask=self.target.mask.detach().cpu().numpy(),
#                 )

#                 self.ax_target = plot_window(self.ax_target, self.model.models[self.working_component_name].window)

#                 self.ax_model = plot_window(self.ax_model, self.model.models[self.working_component_name].window)

#                 self.ax_resi = plot_window(self.ax_resi, self.model.models[self.working_component_name].window)

#             else:

#                 self.ax_model.axis('off')
#                 self.ax_resi.axis('off')

#                 # selector.cid = self.fig.canvas.mpl_connect('button_press_event', selector.on_click)

#                 self.selector = InteractiveWindowSelector(self.fig, self.ax_target, self.target.pixelscale)

#         else:

#             base_image = self.target.model_image()

#             model_total = self.model(image=base_image)

#             model_image = model_total.data.detach().cpu().numpy()

#             self.ax_model = _plot_image(
#                 self.ax_model,
#                 model_image/target_noise,
#                 vlims=[-3, self.vmed, self.vmax], 
#                 colormaps=[self.bottom_cmap, self.top_cmap],
#             )

#             resi_image = target_data - model_image

#             self.ax_resi = _plot_image(
#                 self.ax_resi,
#                 resi_image/target_noise,
#                 vlims=[-3, self.vmed, self.vmax], 
#                 colormaps=[self.bottom_cmap, self.top_cmap],
#                 mask=self.target.mask.detach().cpu().numpy(),
#             )

#         self.fig.canvas.draw()

#     def plot_component(self, component_name):
        
#         self.working_component_name = component_name

#         self.ax_target.clear()

#         self.ax_model.clear()

#         self.ax_resi.clear()

#         self.plot()


#     def edit_mask(self, which='target', zoom=(False, 100)):
        
#         self.ax_target.clear()

#         self.ax_model.clear()

#         self.ax_resi.clear()

#         xlim = -0.1 * self.target.data.shape[0], self.target.data.shape[0] * 1.1
#         ylim = -0.1 * self.target.data.shape[1], self.target.data.shape[1] * 1.1

#         if zoom[0]:
#             xlim = (xlim[0] + xlim[1]) /2. - zoom[1]/2., (xlim[0] + xlim[1]) /2. + zoom[1]/2.
#             ylim = (ylim[0] + ylim[1]) /2. - zoom[1]/2., (ylim[0] + ylim[1]) /2. + zoom[1]/2.

#         self.ax_target.set_xlim(xlim)
#         self.ax_target.set_ylim(ylim)

#         self.ax_model.set_xlim(xlim)
#         self.ax_model.set_ylim(ylim)

#         self.ax_resi.set_xlim(xlim)
#         self.ax_resi.set_ylim(ylim)

#         self.plot()

#         if which == 'target':
#             draw_axis = self.ax_target
#         elif which == 'model':
#             draw_axis = self.ax_model
#         elif which == 'resi':
#             draw_axis = self.ax_resi

#         self.edit_mask_handle = InteractiveMaskDrawer(self.fig, draw_axis, self.target.mask.detach().cpu().numpy())

#         # self.maskcid = self.fig.canvas.mpl_connect('button_press_event', self.masker.on_click)

#         return self.edit_mask_handle
    
#     def dilate_mask(self, which='target'):

#         self.ax_target.clear()

#         self.ax_model.clear()

#         self.ax_resi.clear()

#         # xlim = -0.1 * self.target.data.shape[0], self.target.data.shape[0] * 1.1
#         # ylim = -0.1 * self.target.data.shape[1], self.target.data.shape[1] * 1.1

#         # self.ax_target.set_xlim(xlim)
#         # self.ax_target.set_ylim(ylim)

#         # self.ax_model.set_xlim(xlim)
#         # self.ax_model.set_ylim(ylim)

#         # self.ax_resi.set_xlim(xlim)
#         # self.ax_resi.set_ylim(ylim)

#         self.plot()

#         if which == 'target':
#             draw_axis = self.ax_target
#         elif which == 'model':
#             draw_axis = self.ax_model
#         elif which == 'resi':
#             draw_axis = self.ax_resi

#         self.dilate_mask_handle = InteractiveMaskDilator(self.fig, draw_axis, self.target.mask.detach().cpu().numpy())

#         # self.maskcid = self.fig.canvas.mpl_connect('button_press_event', self.masker.on_click)

#         return self.dilate_mask_handle