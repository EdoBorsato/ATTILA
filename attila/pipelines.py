"""
Template of the three stages of the ATTILA detection pipeline.

The module chains the building blocks of the package into three
functions, each taking the output of the previous one:

1. `combine` builds the detection array: the g + r signal-to-noise map
   of the field, smoothed with a Gaussian kernel. It is computed tile
   by tile and mosaicked back to the full image.
2. `detect` segments the detection array tile by tile, merges the tile
   segmentations into one map, turns the segments into `Sources`,
   computes their Galactic extinction and photometric properties, and
   writes them to `./output/detection/`.
3. `deblend` splits the blended sources with `photutils`, measures the
   deblended sources again and writes them to `./output/deblend/`.

The signatures are such that the stages can be chained as
`deblend(*detect(*combine(field, tiling)))`.

The functions are a reference implementation: the parameters of the
detection (kernel width, threshold, minimum area, deblending levels)
are hard-coded, the field must have the bands `'g'` and `'r'`, and the
output is written to `./output/`, relative to the current working
directory.
"""

from attila.field.field import Field

from astropy import units as u

import numpy as np

from astropy.convolution import Gaussian2DKernel, convolve_fft
from functools import partial

from attila.tiling.tilingrid import Tiling, _combine_labels
from attila.tiling.healpix import HealpixTiling, detect_in_cells
from attila.utils.image_utils import gaussian_smooth
from attila.tiling.tiles import ConvolutionTile, BaseTile, DetectionTile
from attila.detection.detection import segment
from attila.tiling.sources import Sources
from attila.properties.extinction import Extinction
from attila.properties.properties import Properties

from scipy import ndimage

from tqdm import tqdm

import warnings
from photutils.segmentation import SegmentationImage

import os

from attila import attila_logger

from typing import Dict

from attila.deblending.deblending import deblend as photutils_deblend
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from os import path

def _measure_and_save(sources: Sources, field: Field, extinction_tiling, threshold, nconn):
    """
    Compute the extinction and the properties of detected sources.

    This is the common end of the detection stage (steps 5 to 7 of
    `detect`): the Galactic extinction is computed at the centres of
    the tiles of `extinction_tiling` and assigned to the sources, the
    image cutouts are extracted, the measurements of
    `Properties.detection_properties` are run and the sources are
    saved in `./output/detection/<threshold>_<nconn>/`.

    Parameters
    ----------
    sources
        The detected sources.
    field
        The field with the images of the bands, in counts.
    extinction_tiling : Tiling or HealpixTiling
        The tiling whose `centres` and `polygons` define the grid of
        extinction values.
    threshold : float
        The detection threshold, used in the name of the output
        folder.
    nconn : int
        The minimum number of connected pixels, used in the name of
        the output folder.

    Returns
    -------
    sources : Sources
        The sources, with their `properties` table.
    ext : Extinction
        The extinction object, holding the grid of extinction values.
    """

    attila_logger.logger.info("Compute the extinction from the detection sources...")

    ext = Extinction(
        field = field,
        extinction_map_name = 'sfd',
        bands=['SLOAN_SDSS.g', 'SLOAN_SDSS.r'],
        )

    ext.compute_extinction_from_tiling(coord_list=extinction_tiling.centres,
                                    regions=extinction_tiling.polygons)

    A_lambda = ext.apply_grid_to_sources(sources)

    attila_logger.logger.info("Load the data")

    sources.get_data(field.multibandimages)

    attila_logger.logger.info("Compute the properties of the sources...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        props = Properties(sources, field)

        props.A = A_lambda
        sources = props.detection_properties()

    attila_logger.logger.info("All done! Outputting the detection results...")

    os.makedirs(f'./output/detection/{threshold}_{nconn}', exist_ok=True)

    sources.save_light(f'./output/detection/{threshold}_{nconn}')

    return sources, ext


def _snr_smoothed(data_g, data_r, noise_g, noise_r, sigma):
    """
    Return the smoothed g + r signal-to-noise image.

    Parameters
    ----------
    data_g, data_r : numpy.ndarray
        The images of the two bands, in counts.
    noise_g, noise_r : numpy.ndarray
        The RMS noise maps of the two bands.
    sigma : float
        The standard deviation of the Gaussian kernel in pixels.

    Returns
    -------
    numpy.ndarray
        `(g + r) / sqrt(noise_g^2 + noise_r^2)`, smoothed with
        `attila.utils.image_utils.gaussian_smooth`.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = (data_g + data_r) / np.sqrt(noise_g**2 + noise_r**2)

    return gaussian_smooth(snr, sigma)


def _combine_healpix(field: Field, tiling: HealpixTiling, sigma: float = 2.0, n_workers: int = 4):
    """
    Build the detection array on a HEALPix tiling.

    It is the counterpart of `combine` for a `HealpixTiling`. For
    every cell, the g and r images and noise maps are cut out with a
    margin as wide as the smoothing kernel, the signal-to-noise image
    is computed and smoothed, and the part that covers the cell is
    written into the full-size detection array. Thanks to the margin
    the values do not depend on the cell they are computed in, so
    nothing has to be combined where the cutouts overlap, and the
    cells can be processed in parallel.

    Parameters
    ----------
    field
        The field with the bands `'g'` and `'r'`.
    tiling
        The tiling of the field.
    sigma
        The standard deviation of the Gaussian kernel in pixels.
    n_workers
        The number of threads that process the cells.

    Returns
    -------
    output_detarray_dict : dict
        `{'gr': {'detection_array': numpy.ndarray}}`, 32-bit floats.
    field : Field
        The input field, unchanged.
    tiling : HealpixTiling
        A new tiling that also includes the band `'gr'`.
    """
    output_detarray_dict = {
        'gr': {
            'detection_array': np.zeros(field.multibandimages['g'].shape, dtype=np.float32),
        }
    }

    tiling = tiling.add_band('gr', field.multibandimages.property_dict('header')['g'])

    input_dict = field.multibandimages.property_dict(['data', 'noise'])
    input_dict = {band: input_dict[band] for band in ('g', 'r')}

    # Half-size of the kernel: the margin that makes the smoothed
    # values independent of the cutout.
    margin = int(np.ceil(4 * sigma)) + 1

    def run(i):
        combine_tile = tiling.get_tile_data(
            index = i,
            Tile = BaseTile,
            input_dict = input_dict,
            margin_pixels = margin
        )

        convolve_tile = ConvolutionTile(
            shape=combine_tile.shape,
            overlap=combine_tile.overlap,
            bands=['gr'],
            index=i
        )

        convolve_tile.inner_box = {'gr': combine_tile.inner_box['g']}
        convolve_tile.outer_box = {'gr': combine_tile.outer_box['g']}

        convolve_tile.add_data(
            band = 'gr',
            key = 'detection_array',
            arr = _snr_smoothed(
                combine_tile.data2D['g']['data'],
                combine_tile.data2D['r']['data'],
                combine_tile.data2D['g']['noise'],
                combine_tile.data2D['r']['noise'],
                sigma
            ),
            reg = combine_tile.reg['g']['data']
        )

        tiling.tile_to_mosaic(
            tile = convolve_tile,
            band = 'gr',
            output_dict = output_detarray_dict
        )

    with ThreadPool(n_workers) as pool:
        list(tqdm(pool.imap(run, range(len(tiling.cell_ids))), total=len(tiling.cell_ids)))

    attila_logger.logger.info("Convolution and combination done.")

    return output_detarray_dict, field, tiling


def _detect_healpix(output_detarray_dict: Dict, field: Field, tiling: HealpixTiling, threshold, nconn, n_workers: int = 4):
    """
    Detect and measure the sources on a HEALPix tiling.

    It is the counterpart of `detect` for a `HealpixTiling`. The
    cells are segmented independently with
    `attila.tiling.healpix.detect_in_cells`: every source belongs to
    the cell that contains its peak, and a source cut by the border
    of a cutout is detected again on a larger one, so no mosaic of
    labels is built and nothing is relabelled. Segments that touch
    after the closing are merged (`segment(merge_touching=True)`), as
    on a rectangular tiling. The extinction grid is
    made of the HEALPix cells of order 10 (3.4 arcmin), or of the
    cells of the tiling if they are smaller. The HEALPix cell of every
    source is written in the column `healpix_cell` of the properties.

    Parameters
    ----------
    output_detarray_dict
        The detection array, as returned by `combine`.
    field
        The field with the images of the bands, in counts.
    tiling
        The tiling returned by `combine`, which includes the band
        `'gr'`.
    threshold : float
        The detection threshold in units of sigma.
    nconn : int
        The minimum number of connected pixels of a source.
    n_workers
        The number of threads that process the cells.

    Returns
    -------
    tuple
        The same six values as `detect`.
    """
    # Touching segments are merged, as the relabelling of the mosaic
    # does on a rectangular tiling.
    detection_function = partial(
        segment,
        nconn = nconn,
        threshold = threshold,
        fill = True,
        merge_touching = True
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sources = detect_in_cells(
            det_arr = output_detarray_dict['gr']['detection_array'],
            tiling = tiling,
            band = 'gr',
            detection_function = detection_function,
            n_workers = n_workers
        )

    extinction_tiling = HealpixTiling(
        header_dict = tiling.header_dict,
        order = max(tiling.order, 10)
    )

    extinction_tiling.make_tiles()

    sources, ext = _measure_and_save(sources, field, extinction_tiling, threshold, nconn)

    sources.properties['healpix_order'] = tiling.order
    sources.properties['healpix_cell'] = [source.cell_id for source in sources]

    sources.properties.to_csv(path.join(f'./output/detection/{threshold}_{nconn}', 'properties.csv'))

    return sources, ext, field, tiling, threshold, nconn


def combine(field: Field, tiling: Tiling):
    """
    Build the smoothed g + r signal-to-noise detection array.

    With a `HealpixTiling` the work is done by `_combine_healpix`,
    which processes the cells independently; the description below
    applies to the rectangular `Tiling`.

    For every tile of the tiling, the cutouts of the g and r images and
    of their noise maps are combined into the signal-to-noise ratio of
    the summed image,

        S/N = (g + r) / sqrt(noise_g^2 + noise_r^2)

    which is then convolved with a 2D Gaussian kernel of standard
    deviation 2 pixels (`astropy.convolution.convolve_fft`) to enhance
    faint extended sources. The g and r images must share the same
    pixel grid.

    The smoothed tiles are pasted into an array with the shape of the
    g image, initially filled with zeros. Where a tile meets values
    already written (the zeros, or another tile in the overlap regions)
    the pixel-wise maximum is kept. As a consequence the detection
    array has no negative values, and the pixels not covered by any
    tile are 0.

    The detection array is registered in the tiling as a new band named
    `'gr'`, with the header of the g image. This creates a new `Tiling`
    object with the same tile shape and overlap and remakes its tiles;
    the tiling given in input is not modified.

    All the warnings raised while a tile is processed are silenced.

    Parameters
    ----------
    field
        The field with the images and the noise maps of the bands
        `'g'` and `'r'`, in counts.
    tiling
        The tiling of the field. Only its definition (headers, tile
        shape, overlap, grid type) is used: the tiles are remade when
        the `'gr'` band is added.

    Returns
    -------
    output_detarray_dict : dict
        The detection array, as
        `{'gr': {'detection_array': numpy.ndarray}}`. The array has
        the shape of the g image, axis order (y, x), and is in units
        of sigma (dimensionless signal-to-noise).
    field : Field
        The input field, unchanged.
    tiling : Tiling
        The new tiling, with the additional band `'gr'`.
    """

    attila_logger.logger.info("starting the preprocessing to compute the desired detection array")

    if isinstance(tiling, HealpixTiling):
        return _combine_healpix(field, tiling)

    # Smoothing kernel: a Gaussian with a sigma of 2 pixels.
    kernel = Gaussian2DKernel(x_stddev=2)

    partial_convolve = partial(convolve_fft, kernel=kernel)

    output_detarray_dict = {
        'gr': {
            'detection_array': np.zeros(field.multibandimages['g'].shape),
        }
    }

    def combine_func(arr1, arr2):
        """Return the pixel-wise maximum of the mosaic and the tile."""
        return np.maximum(arr1, arr2)

    tiling = tiling.add_band('gr', field.multibandimages.property_dict('header')['g'])

    for i in tqdm(range(len(tiling.centres))):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Cutouts of the data and noise of every band on tile i.
            combine_tile = tiling.get_tile_data(
                index = i,
                Tile = BaseTile,
                input_dict = field.multibandimages.property_dict(['data', 'noise'])
            )

            # Sum of the two bands and its noise, added in quadrature.
            combined_data = combine_tile.data2D['g']['data'] + combine_tile.data2D['r']['data']
            combined_noise = np.sqrt(combine_tile.data2D['g']['noise']**2 +
                                    combine_tile.data2D['r']['noise']**2)

            convolve_tile = ConvolutionTile(
                shape=combine_tile.shape,
                overlap=combine_tile.overlap,
                bands=['gr'],
                index=i
            )

            # The S/N tile inherits the WCS of the g-band cutout.
            convolve_tile.add_data(
                band = 'gr',
                key = 'detection_array',
                arr = combined_data/combined_noise,
                reg = combine_tile.reg['g']['data']
            )

            convolve_tile.apply_convolution_function(
                band='gr',
                convolution_function = partial_convolve
            )

            tiling.tile_to_mosaic(
                    tile = convolve_tile,
                    band='gr',
                    output_dict = output_detarray_dict,
                    combine_func = combine_func
                )

    attila_logger.logger.info("Convolution and combination done.")

    return output_detarray_dict, field, tiling

def detect(output_detarray_dict: Dict, field: Field, tiling: Tiling):
    """
    Detect the sources on the detection array and measure them.

    With a `HealpixTiling` the work is done by `_detect_healpix`, in
    which every source belongs to the cell that contains its peak and
    steps 2 and 3 below (mosaicking and relabelling) do not exist.

    The stage performs these steps:

    1. Segmentation. Every tile of the detection array is segmented
       with `attila.detection.detection.segment`: a source is a group
       of at least `nconn` = 100 connected pixels above `threshold` =
       1 (in units of sigma, since the array is a signal-to-noise
       map), and the holes of the segments are closed (`fill=True`).
    2. Mosaicking. The tile segmentations are pasted into a full-size
       map. Each tile numbers its segments from 1, so the labels of
       different tiles are not consistent: where tiles overlap the
       larger label is kept (`_combine_labels`), which preserves every
       pixel that is detected in at least one tile.
    3. Relabelling. The mosaic is relabelled globally with
       `scipy.ndimage.label` and 8-connectivity. Every connected group
       of detected pixels becomes one source with a unique label, so
       segments that touch each other are merged.
    4. Sources. The segments are converted to a `Sources` container
       (`Sources.from_segmentation_image`), each source holding the
       cutouts of its segment and of the detection array.
    5. Extinction. The Galactic extinction in the g and r SDSS filters
       is computed at the centres of the tiles
       (`Extinction.compute_extinction_from_tiling`) and assigned to
       every source according to the tile polygon that contains its
       centre (`Extinction.apply_grid_to_sources`). The tiles used are
       those of `extinction_tiling`, a dedicated tiling with tiles of
       3 arcmin, since the reddening varies on scales smaller than
       the detection tiles.
    6. Properties. The image cutouts of the sources are extracted
       from the images of the field and the measurements of
       `Properties.detection_properties` are run, with the extinction
       of step 5.
    7. Output. The sources are saved with `Sources.save_light` in
       `./output/detection/<threshold>_<nconn>/`.

    The filter curves and the SFD dust map are read from the
    `synphot_filters` and `dustmaps` folders of the repository.

    Parameters
    ----------
    output_detarray_dict
        The detection array, as
        `{'gr': {'detection_array': numpy.ndarray}}`, as returned by
        `combine`.
    field
        The field with the images of the bands, in counts. It must
        have the attribute `zpt` required by `Properties`.
    tiling
        The tiling returned by `combine`, which includes the band
        `'gr'`.

    Returns
    -------
    sources : Sources
        The detected sources, with their `properties` table.
    ext : Extinction
        The extinction object, holding the grid of extinction values.
    field : Field
        The input field, unchanged.
    tiling : Tiling
        The input tiling, unchanged.
    threshold : int
        The detection threshold that was used, in units of sigma.
    nconn : int
        The minimum number of connected pixels that was used.
    """

    output_dict_segmap = {
        'gr': {
            'segmap_data': np.zeros(field.multibandimages['g'].shape),
        }
    }

    # Minimum number of connected pixels and S/N threshold of a source.
    nconn = 100
    threshold = 1

    attila_logger.logger.info(f"The detection will done with: nconn = {nconn} and threshold = {threshold}")

    if isinstance(tiling, HealpixTiling):
        return _detect_healpix(output_detarray_dict, field, tiling, threshold, nconn)

    detection_function = partial(
        segment,
        nconn = nconn,
        threshold = threshold,
        fill = True
        )

    # Element-wise rule to merge the labels of overlapping tiles.
    combine_func = np.vectorize(_combine_labels)

    for i in tqdm(range(len(tiling.centres))):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            detection_tile = tiling.get_tile_data(
                index = i,
                Tile = DetectionTile,
                input_dict = output_detarray_dict
            )

            detection_tile.apply_detection_function('gr', detection_function)

            tiling.tile_to_mosaic(
                tile = detection_tile,
                band='gr',
                output_dict = output_dict_segmap,
                combine_func = combine_func)

    # The labels of the mosaic are only meaningful as "detected or
    # not": relabel the connected regions (8-connectivity) globally.
    stitched_segmap_dict = {'gr': ndimage.label(
        output_dict_segmap['gr']['segmap_data'],
        structure=np.ones((3, 3)),
        output=np.uint32
        )[0]}

    attila_logger.logger.info("Now I can stack each detection on top of the other! This might take a while...")

    sources = Sources.from_segmentation_image(
        segmap = SegmentationImage(stitched_segmap_dict['gr']),
        full_det_arr = output_detarray_dict['gr']['detection_array'],
        header = tiling.header_dict['gr']
    )

    tile_shape_sky = (3 * u.arcmin, 3 * u.arcmin)

    extinction_tiling = Tiling(
        header_dict = tiling.header_dict,
        tile_shape_sky = tile_shape_sky,
        grid_type = 'grid'
        )

    extinction_tiling.make_tiles()

    sources, ext = _measure_and_save(sources, field, extinction_tiling, threshold, nconn)

    return sources, ext, field, tiling, threshold, nconn

def _deblend_source(source, nlevels):
    """
    Deblend one source and attach the result to it.

    The function is defined at module level so that it can be sent to
    the worker processes of a `multiprocessing.Pool`.

    Parameters
    ----------
    source : Source
        The source to deblend.
    nlevels : int
        The number of multi-threshold levels.

    Returns
    -------
    Source
        The same source, with the deblended segmentation image in its
        attribute `deblend`.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        source.deblend = photutils_deblend(
            detection_array=source.det_arr,
            segmap=SegmentationImage(source.label*source.segment.astype(int)),
            nlevels=nlevels,
        )
        return source


def deblend(sources: Sources, ext: Extinction, field: Field, tiling: Tiling, threshold: int, nconn: int):
    """
    Deblend the detected sources and measure the deblended ones.

    The stage performs these steps:

    1. Deblending. Every source is deblended on the cutout of its
       detection array with `attila.deblending.deblending.deblend`
       (the multi-threshold and watershed algorithm of `photutils`),
       using 10 threshold levels and the default minimum area and
       contrast. The segmentation given to the deblender contains the
       segment of the source alone, with its label. The resulting
       segmentation image is assigned to the attribute `deblend` of
       the source. The sources are processed by a pool of 8 worker
       processes and replace the content of `sources`, in the same
       order.
    2. Sources. A new `Sources` container is built with
       `Sources.from_deblend`: a source split into several components
       is replaced by one source per component, each keeping the label
       of its parent in `previous_label`; the others are kept as they
       are.
    3. Extinction. The grid computed in `detect` is applied to the
       deblended sources (`Extinction.apply_grid_to_sources`).
    4. Properties. The image cutouts are extracted and the
       measurements of `Properties.detection_properties` are run, as
       in `detect`.
    5. Output. The deblended sources are saved with
       `Sources.save_light` in
       `./output/deblend/<threshold>_<nconn>_<nlevels>/`.

    Parameters
    ----------
    sources
        The detected sources, as returned by `detect`.
    ext
        The extinction object returned by `detect`, with the grid of
        extinction values already computed.
    field
        The field with the images of the bands, in counts.
    tiling
        The tiling of the field. It is not used.
    threshold
        The detection threshold used in `detect`; it only enters the
        name of the output folder.
    nconn
        The minimum number of connected pixels used in `detect`; it
        only enters the name of the output folder.

    Returns
    -------
    Sources
        The deblended sources, with their `properties` table.
    """

    attila_logger.logger.info("deblending the data...")

    # Number of multi-threshold deblending levels.
    nlevels = 10

    with Pool(processes=8) as pool:
        sources.sources = list(tqdm(
            pool.imap(partial(_deblend_source, nlevels=nlevels), sources.sources),
            total=len(sources)
        ))

    deblended_sources = Sources.from_deblend(sources)

    A_lambda = ext.apply_grid_to_sources(deblended_sources)

    attila_logger.logger.info("Load the data")

    deblended_sources.get_data(field.multibandimages)

    props = Properties(deblended_sources, field)
    props.A = A_lambda

    attila_logger.logger.info("Compute the properties of the sources...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deblended_sources = props.detection_properties()

    attila_logger.logger.info("All done! Outputting the detection results...")

    os.makedirs(f'./output/deblend/{threshold}_{nconn}_{nlevels}', exist_ok=True)

    deblended_sources.save_light(f'./output/deblend/{threshold}_{nconn}_{nlevels}')

    return deblended_sources
