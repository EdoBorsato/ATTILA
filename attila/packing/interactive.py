import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

from shapely.geometry import Point
from matplotlib import colormaps as mcmaps
from matplotlib import colors as mcolors

from attila.packing.packing import Packing
from attila.tiling.sources import Sources, Source, BoundingBox_to_Polygon
from attila.field.field import Field

import shapely as sh 
from shapely.plotting import plot_polygon

from scipy import ndimage
from photutils.segmentation import SegmentationImage

import pathlib

import pandas as pd

import json

from rasterio.features import shapes

from astropy.convolution import convolve, Gaussian2DKernel

def _get_polygons_from_segment(segment, invert=False):

    if invert:
        mask = np.where(segment==0, True, False)
    else:
        mask = np.where(segment==0, False, True)

    shape_polygon_list = list(shapes(segment.astype(np.int32), mask=mask, connectivity=8))

    return [sh.geometry.shape(geo_poly[0]) for geo_poly in shape_polygon_list]

def _correct_for_polygon_shift(polygon):

    return sh.affinity.translate(polygon, xoff=-0.5, yoff=-0.5)

def _compute_cps_SB(field:Field, SB:float, band:str):

    if not hasattr(field, "zpt"):
        raise AttributeError("Field object must have a zpt attribute")

    pixel_scale = field.multibandimages[band].pixel_scale.value

    zpt = field.zpt[band]

    obs_SB = SB - zpt - 5*np.log10(pixel_scale)

    if hasattr(field, "redshift"):
        obs_SB += 2.5*np.log10((1 + field.redshift)**4)

    return 10**(-0.4*obs_SB)


def _plot(ax, data, noise, segment, polygons, threshold_upper, threshold_max):

    # define things related to the colormap
    bottom = mcmaps['coolwarm']
    top_parent = mcmaps['RdYlGn']
    color_list = [
        bottom(0.99), 
        top_parent(0.15), 
        top_parent(0.33), 
        top_parent(0.66), 
        top_parent(0.99), 
    ]
    top = mcolors.LinearSegmentedColormap.from_list("top", color_list, N=256)

    # define the images to be displayed

    threshold_upper_SNR = threshold_upper / np.nanmedian(noise)
    threshold_max_SNR = threshold_max / np.nanmedian(noise)

    img = data/noise
    img_bot = np.where(img <= threshold_upper_SNR, img, threshold_upper_SNR)
    ax.imshow(img_bot, vmin=-3, vmax=threshold_upper_SNR, origin='lower', cmap=bottom)

    if np.any(img >= threshold_upper_SNR):
        img_top = np.ma.masked_where(img <= threshold_upper_SNR, img)
        norm = mcolors.LogNorm(vmin=threshold_upper_SNR, vmax=threshold_max_SNR)
        ax.imshow(img_top, norm=norm, origin='lower', cmap=top)

    for polygon in polygons:

        polygon = _correct_for_polygon_shift(polygon)

        plot_polygon(polygon, ax=ax, add_points=False, facecolor='none', edgecolor='C0', alpha=1, lw=2)

    # for name, polygon in zip(names, polygons):
    #     upper_right_corner = polygon.bounds[2] - 5, polygon.bounds[3] - 5
    #     ax.text(upper_right_corner[0], upper_right_corner[1], name, fontsize=8, color='k', 
    #                     ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
    segment_polys = _get_polygons_from_segment(segment, invert=True)
    for polygon in segment_polys:

        polygon = _correct_for_polygon_shift(polygon)

    #     color = 'brown'

    #     plot_polygon(polygon, ax=ax, add_points=False, facecolor='none', edgecolor=color, hatch='//', alpha=1, lw=1)

    ax.set_xlim(-.5, data.shape[1]-0.5)
    ax.set_ylim(-0.1, data.shape[0]-0.5)
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    kernel = Gaussian2DKernel(x_stddev=1.5, y_stddev=1.5)

    convolved_image = convolve(img, kernel, boundary='extend')

    levels = np.logspace(np.log10(2), np.log10(threshold_max_SNR.value), num=10)

    ax.contour(convolved_image, levels=levels, colors='k', linewidths=1, linestyles='dashed', alpha=0.5)


class InteractivePlotter:
    def __init__(self, sources: Sources, field:Field):

        self.sources = sources

        self.polygons = [BoundingBox_to_Polygon(source.bbox) for source in sources]

        self.packing = Packing(self.polygons)

        self.packing.get_rectangular_packing(max_axis_ratio=0.6)

        self.data, self.noise, self.segment = self.packing.packed_from_sources(sources, 'g')

        self.threshold_upper = _compute_cps_SB(field, 25, 'g')
        self.threshold_max = _compute_cps_SB(field, 23.5, 'g')

        # self.names = [str(source.label) for source in sources]

        self.selected_polygons = []

        self.classification = {}
        for source in sources:
            _id = f"{source.previous_label}_{source.label}"
            self.classification[_id] = '1'

        # self.classification = np.full(len(self.polygons), '1', dtype=str)

        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 12 * self.data.shape[0]/self.data.shape[1]))
        
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.plot_data()

    def plot_data(self):
        _plot(self.ax, self.data, self.noise, self.segment, self.packing.optimized_polygons, self.threshold_upper, self.threshold_max)
        plt.show()


    def on_click(self, event):
        click_point = Point(event.xdata, event.ydata)
        for i, polygon in enumerate(self.packing.optimized_polygons):
            if polygon.contains(click_point):
                self.selected_polygon_index = i
                # print(f"Polygon {i} selected")

                self.highlight_polygon(polygon, "1")
                
                break

    def on_key_press(self, event):
        if event.key == 'q':
            plt.close(self.fig)

        if hasattr(self, 'selected_polygon_index'):
            if (event.key =='2') | (event.key =='1') | (event.key =='0'):

                selected_source = self.sources[self.selected_polygon_index]

                _id = f"{selected_source.previous_label}_{selected_source.label}"

                self.classification[_id] = event.key

                self.highlight_polygon(self.packing.optimized_polygons[self.selected_polygon_index], event.key)

    def save(self, filename:str):

        with open(f"output/classification/{filename}.json", 'w+') as f:
            json.dump(self.classification, f, indent=4)

        return self.classification

    def highlight_polygon(self, polygon, classification):

        polygon = _correct_for_polygon_shift(polygon)

        if classification == '2':
            edgecolor = 'C1'

            plot_polygon(polygon, ax=self.ax, add_points=False, facecolor='none', edgecolor=edgecolor, alpha=1, lw=2)
            self.fig.canvas.draw()

        elif classification == '1':
            edgecolor = 'C0'
        
            plot_polygon(polygon, ax=self.ax, add_points=False, facecolor='none', edgecolor=edgecolor, alpha=1, lw=2)
            self.fig.canvas.draw()

        elif classification == '0':
            edgecolor = 'red'
        
            plot_polygon(polygon, ax=self.ax, add_points=False, facecolor='none', edgecolor=edgecolor, alpha=1, lw=2)
            self.fig.canvas.draw()

        # elif self.classification[self.selected_polygon_index] == '--':
        #     edgecolor = 'C0'
        
            plot_polygon(polygon, ax=self.ax, add_points=False, facecolor='none', edgecolor=edgecolor, alpha=1, lw=2)
            self.fig.canvas.draw()