import rpack

import numpy as np
import matplotlib.pyplot as plt
import shapely as sh
from shapely.plotting import plot_polygon

from functools import partial

from attila.tiling.sources import Sources

from photutils.aperture import BoundingBox
import shapely.geometry as sh

def Polygon_to_BoundingBox(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    return BoundingBox(ixmin=int(minx), ixmax=int(maxx), iymin=int(miny), iymax=int(maxy))

def _get_wh_from_Polygon(bbox, padding=0):
        
        width = int(bbox.bounds[2] - bbox.bounds[0]) + 2*padding
        height = int(bbox.bounds[3] - bbox.bounds[1]) + 2*padding
        return (width, height)

def _add_Polygon_data(input, output, input_polygons, output_polygons, check_func=None):

    x_min_0, y_min_0, x_max_0, y_max_0 = input_polygons.bounds
    x_min_0, y_min_0, x_max_0, y_max_0 = int(x_min_0), int(y_min_0), int(x_max_0), int(y_max_0)

    x_min, y_min, x_max, y_max = output_polygons.bounds
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    if check_func is None:
        output[y_min:y_max, x_min:x_max] = input[y_min_0:y_max_0, x_min_0:x_max_0]

    else:
        output[y_min:y_max, x_min:x_max] = check_func(input[y_min_0:y_max_0, x_min_0:x_max_0])

    return output

    # optimized_selected_error[y_min:y_max, x_min:x_max] = pgc007748.obs_dict['g']['noise'][y_min_0:y_max_0, x_min_0:x_max_0]
    
    # otimized_selected_segmap_data[y_min:y_max, x_min:x_max] = segmap_dict['g'].data[y_min_0:y_max_0, x_min_0:x_max_0]
    # otimized_selected_segmap_data[y_min:y_max, x_min:x_max] = np.where(otimized_selected_segmap_data[y_min:y_max, x_min:x_max] == label, label, 0)

def _get_relative_polygon(polygon):

    minx, miny, maxx, maxy = polygon.bounds

    return sh.box(0, 0, maxx-minx, maxy-miny)

def BoundingBox_to_Polygon(boundingbox, padding=0, shape=None):
    if shape is None:
        maxx,maxy=np.inf,np.inf
    else:
        maxx,maxy=shape
    return sh.box(
        max(boundingbox.ixmin - padding, 0),
        max(boundingbox.iymin - padding, 0),   
        min(boundingbox.ixmax + padding, maxy),
        min(boundingbox.iymax + padding, maxx)
    )

class Packing:

    def __init__(self, polygons):
        
        self.polygons = [_get_relative_polygon(polygon) for polygon in polygons]    

        self.total_area = sum([polygon.area for polygon in self.polygons])


    def get_rectangular_packing(self, max_axis_ratio, which='width'):
        
        # Get the width and height of each bounding box
        whs = [_get_wh_from_Polygon(bbox) for bbox in self.polygons]

        padded_whs = [_get_wh_from_Polygon(bbox, padding=1) for bbox in self.polygons]

        # Get the packing

        positions = rpack.pack(sizes=padded_whs)

        self.optimized_polygons = [
                    sh.box(pos[0], pos[1], pos[0] + size[0], pos[1] + size[1]) 
                    for size, pos in zip(whs, positions)
                ]
        self.bounding_shape = rpack.bbox_size(padded_whs, positions)[::-1]

        if which == 'width':
            if self.bounding_shape[1] > self.bounding_shape[0]:

                self.optimized_polygons = [
                    sh.box(pos[1], pos[0], pos[1] + size[1], pos[0] + size[0]) 
                    for size, pos in zip(whs, positions)
                ]

                self.bounding_shape = self.bounding_shape[::-1]

            
        elif which == 'height':
            if self.bounding_shape[0] > self.bounding_shape[1]:

                self.optimized_polygons = [
                    sh.box(pos[1], pos[0], pos[1] + size[1], pos[0] + size[0]) 
                    for size, pos in zip(whs, positions)
                ]

                self.bounding_shape = self.bounding_shape[::-1]


    def plot_packing(self, min_size=10):

        if not hasattr(self, 'optimized_polygons'):
            raise AttributeError('You need to run get_rectangular_packing() before adding data')
        
        y_pix, x_pix = self.bounding_shape

        axis_ratio = x_pix/y_pix

        fig, ax = plt.subplots(figsize=(min_size*axis_ratio, min_size))

        for polygon in self.optimized_polygons:
            plot_polygon(polygon, ax=ax, add_points=False, facecolor='C0', edgecolor='k', alpha=0.5)

        ax.set_aspect('equal')

        plt.show()

    def add_data_from_array(self, data, check_func=None, check_condition=None):

        if not hasattr(self, 'optimized_polygons'):
            raise AttributeError('You need to run get_rectangular_packing() before adding data')

        self.optimized_data = np.zeros(self.bounding_shape, dtype=data.dtype)

        for i in range(len(self.polygons)):

            partial_check_func = partial(check_func, check_value=check_condition[i]) if check_func is not None else None

            self.optimized_data = _add_Polygon_data(
                input = data, 
                output = self.optimized_data, 
                input_polygons = self.polygons[i], 
                output_polygons = self.optimized_polygons[i], 
                check_func = partial_check_func, 
                )

        return self.optimized_data
    
    def packed_from_sources(self, sources:Sources, band:str):

        self.optimized_data = np.zeros(self.bounding_shape)
        self.optimized_noise = np.zeros(self.bounding_shape)
        self.optimized_segment = np.zeros(self.bounding_shape, dtype=int)

        for i, source in enumerate(sources):

            self.optimized_data = _add_Polygon_data(
                input = source.data[band].data, 
                output = self.optimized_data, 
                input_polygons = self.polygons[i], 
                output_polygons = self.optimized_polygons[i], 
                )
            
            self.optimized_noise = _add_Polygon_data(
                input = source.data[band].noise, 
                output = self.optimized_noise, 
                input_polygons = self.polygons[i], 
                output_polygons = self.optimized_polygons[i], 
                )
            
            self.optimized_segment = _add_Polygon_data(
                input = source.label * source.segment, 
                output = self.optimized_segment, 
                input_polygons = self.polygons[i], 
                output_polygons = self.optimized_polygons[i], 
                )
            
        return self.optimized_data, self.optimized_noise, self.optimized_segment
        


