
from astropy.io import fits

from attila.field.field import Field, Image, MultibandImage

from astropy.cosmology.realizations import Planck18

from astropy import units as u

import numpy as np
import matplotlib.pyplot as plt

from astropy.convolution import Gaussian2DKernel, convolve_fft
from functools import partial

from attila.tiling.tilingrid import Tiling, _combine_labels
from attila.tiling.tiles import ConvolutionTile, BaseTile, DetectionTile
from attila.detection.detection import segment
from attila.tiling.sources import Sources
from attila.properties.extinction import Extinction
from attila.properties.properties import Properties

from scipy import ndimage

from tqdm import tqdm

import warnings
from photutils.segmentation import SegmentationImage

from os import path
import os

from attila import attila_logger

from typing import Dict
from numpy.typing import ArrayLike

from attila.deblending.deblending import deblend as photutils_deblend
from multiprocessing import Pool
import warnings

def combine(field: Field, tiling: Tiling):

    attila_logger.logger.info("starting the preprocessing to compute the desired detection array")

    kernel = Gaussian2DKernel(x_stddev=2)

    partial_convolve = partial(convolve_fft, kernel=kernel)

    output_detarray_dict = {
        'gr': {
            'detection_array': np.zeros(field.multibandimages['g'].shape),
        }
    }

    def combine_func(arr1, arr2):
        return np.maximum(arr1, arr2)

    tiling = tiling.add_band('gr', field.multibandimages.property_dict('header')['g'])

    for i in tqdm(range(len(tiling.centres))):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            combine_tile = tiling.get_tile_data(
                index = i,
                Tile = BaseTile, 
                input_dict = field.multibandimages.property_dict(['data', 'noise'])
            )

            combined_data = combine_tile.data2D['g']['data'] + combine_tile.data2D['r']['data']
            combined_noise = np.sqrt(combine_tile.data2D['g']['noise']**2 + 
                                    combine_tile.data2D['r']['noise']**2)
            
            convolve_tile = ConvolutionTile(
                shape=combine_tile.shape,
                overlap=combine_tile.overlap,
                bands=['gr'], 
                index=i
            )

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

    output_dict_segmap = {
        'gr': {
            'segmap_data': np.zeros(field.multibandimages['g'].shape),
        }
    }

    nconn = 100
    threshold = 1

    attila_logger.logger.info(f"The detection will done with: nconn = {nconn} and threshold = {threshold}")

    detection_function = partial(
        segment, 
        nconn = nconn, 
        threshold = threshold,
        fill = True 
        )

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

    attila_logger.logger.info("Compute the extinction from the detection sources...")

    tile_shape_sky = (3 * u.arcmin, 3 * u.arcmin) 

    extinction_tiling = Tiling(
        header_dict = tiling.header_dict,
        tile_shape_sky = tile_shape_sky, 
        grid_type = 'grid'
        )

    extinction_tiling.make_tiles()

    ext = Extinction(
        field = field,
        extinction_map_name = 'sfd', 
        filters_folder='../synphot_filters/',
        extinction_maps_folder='../dustmaps/',
        bands=['SLOAN_SDSS.g', 'SLOAN_SDSS.r'],
        )

    ext.compute_extinction_from_tiling(coord_list=tiling.centres, 
                                    regions=tiling.polygons)

    A_lambda = ext.apply_grid_to_sources(sources)

    attila_logger.logger.info("Load the data")

    data_dict = field.multibandimages.property_dict(["data"])

    sources.get_data(np.asarray([data_dict[band]['data'].astype(np.float32) for band in field.bands]))


    attila_logger.logger.info("Compute the properties of the sources...")

    with warnings.catch_warnings():

        props = Properties(sources, field)

        props.A = A_lambda
        sources = props.detection_properties()

    attila_logger.logger.info("All done! Outputting the detection results...")

    if path.exists('./output/detection') == False:
        os.mkdir('./output/detection')

    if path.exists(f'./output/detection/{threshold}_{nconn}') == False:
        os.mkdir(f'./output/detection/{threshold}_{nconn}')

    if path.exists('./output/detection') == False:
        os.mkdir('./output/detection')

    sources.save_light(f'./output/detection/{threshold}_{nconn}')

    return sources, ext, field, tiling, threshold, nconn

def deblend(sources: Sources, ext: Extinction, field: Field, tiling: Tiling, threshold: int, nconn: int):

    attila_logger.logger.info("deblending the data...")

    nlevels = 10

    def photutils_deblend_source(source):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            source.deblend = photutils_deblend(
                detection_array=source.det_arr, 
                segmap=SegmentationImage(source.label*source.segment.astype(int)), 
                nlevels=10, 
            )
            return source


    with Pool(processes=8) as pool:
        list(tqdm(pool.imap_unordered(photutils_deblend_source, sources), total=len(sources)))

    # for source in tqdm(sources):
    #     source.deblend = photutils_deblend(
    #         detection_array=source.det_arr, 
    #         segmap=SegmentationImage(source.label*source.segment.astype(int)), 
    #         nlevels=nlevels, 
    #     )

    deblended_sources = Sources.from_deblend(sources)

    A_lambda = ext.apply_grid_to_sources(deblended_sources)

    attila_logger.logger.info("Load the data")

    data_dict = field.multibandimages.property_dict(["data"])

    deblended_sources.get_data(np.asarray([data_dict[band]['data'].astype(np.float32) for band in field.bands]))

    props = Properties(deblended_sources, field)
    props.A = A_lambda

    attila_logger.logger.info("Compute the properties of the sources...")

    with warnings.catch_warnings():
        deblended_sources = props.detection_properties()

    attila_logger.logger.info("All done! Outputting the detection results...")

    if path.exists('./output/deblend') == False:
        os.mkdir('./output/deblend')

    if path.exists(f'./output/deblend/{threshold}_{nconn}_{nlevels}') == False:
        os.mkdir(f'./output/deblend/{threshold}_{nconn}_{nlevels}')

    deblended_sources.save_light(f'./output/deblend/{threshold}_{nconn}_{nlevels}')

        # return deblended_sources, ext, field, tiling