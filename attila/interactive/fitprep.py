import astrophot as ap

import numpy as np

# from photutils.segmentation import SegmentationImage
# from photutils.isophote.geometry import EllipseGeometry
# from photutils.aperture import EllipticalAperture


import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon

# from astropy.io import fits

from typing import List, Dict, Tuple

# import pandas as pd

from matplotlib import colormaps as mcmaps
from matplotlib import colors as mcolors

# import os
# import json

import ipywidgets as widgets
from IPython.display import display

from functools import partial

# from copy import deepcopy

from matplotlib import colormaps as mcmaps
# from matplotlib.colors import Normalize 
# from matplotlib.widgets import Button
# from threading import Event
# import shapely as sh
from shapely.plotting import plot_polygon

# from attila.field.field import Field
# from attila.tiling.sources import Source, Sources
from attila.interactive.masking_segment import InteractiveMaskDrawer, InteractiveMaskDilator, _project_polygon
from attila.interactive.masking_segment import InteractiveSegmentDrawer, InteractiveSegmentSelector, _project_polygon

# from attila.fitting.deblend_2D import AP_deblend, load_results
from attila.packing.interactive import _compute_cps_SB, _correct_for_polygon_shift, _get_polygons_from_segment
from astropy.convolution import convolve, Gaussian2DKernel


def _plot_data(fig, ax, data, noise, sb_func, sb_upper, sb_max):

    threshold_upper = sb_func(SB=sb_upper)
    threshold_max = sb_func(SB=sb_max)

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

    threshold_upper_SNR = threshold_upper / np.nanmedian(noise)
    threshold_max_SNR = threshold_max / np.nanmedian(noise)

    img = data/noise
    img_bot = np.where(img <= threshold_upper_SNR, img, threshold_upper_SNR)
    im1 = ax.imshow(img_bot, vmin=-3, vmax=threshold_upper_SNR, origin='lower', cmap=bottom)

    if np.any(img >= threshold_upper_SNR):
        img_top = np.ma.masked_where(img <= threshold_upper_SNR, img)
        norm = mcolors.LogNorm(vmin=threshold_upper_SNR, vmax=threshold_max_SNR)
        im2 = ax.imshow(img_top, norm=norm, origin='lower', cmap=top)

    # ax.set_xlim(-.5, data.shape[1]-0.5)
    # ax.set_ylim(-0.1, data.shape[0]-0.5)
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    kernel = Gaussian2DKernel(x_stddev=1.5, y_stddev=1.5)

    convolved_image = convolve(img, kernel, boundary='extend')
    
    levels = np.logspace(np.log10(2), np.log10(threshold_max_SNR.value), num=10)

    ax.contour(convolved_image, levels=levels, colors='k', linewidths=1, linestyles='dashed', alpha=0.5)

    # cbar1 = fig.colorbar(im1, ax=ax, orientation='vertical', pad=0.01, aspect=20, location='left')

    # cbar1.ax.axhline(threshold_upper_SNR, color='w', linestyle='-', linewidth=5)
    # cbar1.ax.text(0.5, 0.95, f'$\\mu$ {sb_upper}', color='w', transform=cbar1.ax.transAxes, ha='center', va='bottom', fontsize=8)

    # cbar2 = fig.colorbar(im2, ax=ax, orientation='vertical', pad=0.01, aspect=20)

    # cbar2.ax.axhline(threshold_max_SNR, color='w', linestyle='-', linewidth=5)
    # cbar2.ax.text(0.5, 0.95, f'$\\mu$ {sb_max}', color='w', transform=cbar2.ax.transAxes, ha='center', va='bottom', fontsize=8)


def _plot_mask(fig, ax, mask):

    plot_mask = np.where(mask > 0, mask, np.nan)

    ax.imshow(plot_mask, origin='lower', cmap='gray', alpha=0.5)

    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

def _plot_segment(fig, ax, segment):

    ax.set_aspect('equal')

    ax.set_xlim(0, segment.shape[1])
    ax.set_ylim(0, segment.shape[0])

    polys = _get_polygons_from_segment(segment)

    polys = [_correct_for_polygon_shift(poly) for poly in polys]

    for i, poly in enumerate(polys):
        plot_polygon(poly, ax=ax, add_points=False, ls='-', lw=3, ec=f'C{i}', fc='none', hatch='//', alpha=1)

    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)


def _plot_window(ax, window, **kwargs):

    dx = window.pixel_shape[0].detach().cpu().numpy()
    dy = window.pixel_shape[1].detach().cpu().numpy()

    center_arcsec = window.center.detach().cpu().numpy()

    pixel_scale = window.pixelscale.detach().cpu().numpy()

    center_pix = [center_arcsec[0] / pixel_scale[0][0], center_arcsec[1] / pixel_scale[1][1]]

    x0 = center_pix[0] - dx / 2
    y0 = center_pix[1] - dy / 2

    ax.add_patch(plt.Rectangle((x0, y0), dx, dy, fill=False, edgecolor='C1', lw=2, **kwargs))

    return ax, center_pix, (dx, dy)

def _plot_model(fig, ax, model_image, noise, sb_func, sb_upper, sb_max):

    threshold_upper = sb_func(SB=sb_upper)
    threshold_max = sb_func(SB=sb_max)

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

    threshold_upper_SNR = threshold_upper / np.nanmedian(noise)
    threshold_max_SNR = threshold_max / np.nanmedian(noise)

    median_noise = np.nanmedian(noise)

    img = model_image / median_noise
    img_bot = np.where(img <= threshold_upper_SNR, img, threshold_upper_SNR)
    im1 = ax.imshow(img_bot, vmin=-3, vmax=threshold_upper_SNR, origin='lower', cmap=bottom)

    if np.any(img >= threshold_upper_SNR):
        img_top = np.ma.masked_where(img <= threshold_upper_SNR, img)
        norm = mcolors.LogNorm(vmin=threshold_upper_SNR, vmax=threshold_max_SNR)
        im2 = ax.imshow(img_top, norm=norm, origin='lower', cmap=top)

    kernel = Gaussian2DKernel(x_stddev=1.5, y_stddev=1.5)

    convolved_image = convolve(img, kernel, boundary='extend')
    
    levels = np.logspace(np.log10(2), np.log10(threshold_max_SNR.value), num=10)

    ax.contour(convolved_image, levels=levels, colors='k', linewidths=1, linestyles='dashed', alpha=0.5)

    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)


def _plot_residual(fig, ax, residual, noise, sb_func, sb_upper, sb_max):

    threshold_upper = sb_func(SB=sb_upper)
    threshold_max = sb_func(SB=sb_max)

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

    threshold_upper_SNR = threshold_upper / np.nanmedian(noise)
    threshold_max_SNR = threshold_max / np.nanmedian(noise)

    img = residual / noise
    img_bot = np.where(img <= threshold_upper_SNR, img, threshold_upper_SNR)
    im1 = ax.imshow(img_bot, vmin=-3, vmax=threshold_upper_SNR, origin='lower', cmap=bottom)

    if np.any(img >= threshold_upper_SNR):
        img_top = np.ma.masked_where(img <= threshold_upper_SNR, img)
        norm = mcolors.LogNorm(vmin=threshold_upper_SNR, vmax=threshold_max_SNR)
        im2 = ax.imshow(img_top, norm=norm, origin='lower', cmap=top)

    kernel = Gaussian2DKernel(x_stddev=1.5, y_stddev=1.5)

    convolved_image = convolve(img, kernel, boundary='extend')
    
    levels = np.logspace(np.log10(2), np.log10(threshold_max_SNR.value), num=10)

    ax.contour(convolved_image, levels=levels, colors='k', linewidths=1, linestyles='dashed', alpha=0.5)

    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # cbar1 = fig.colorbar(im1, ax=ax, orientation='vertical', pad=0.01, aspect=20, location='left')

    # cbar1.ax.axhline(threshold_upper_SNR, color='w', linestyle='-', linewidth=5)
    # cbar1.ax.text(0.5, 0.95, f'$\\mu$ {sb_upper}', color='w', transform=cbar1.ax.transAxes, ha='center', va='bottom', fontsize=8)

    # cbar2 = fig.colorbar(im2, ax=ax, orientation='vertical', pad=0.01, aspect=20)

    # cbar2.ax.axhline(threshold_max_SNR, color='w', linestyle='-', linewidth=5)
    # cbar2.ax.text(0.5, 0.95, f'$\\mu$ {sb_max}', color='w', transform=cbar2.ax.transAxes, ha='center', va='bottom', fontsize=8)


class PlotTarget:
    def __init__(self, source, field):

        self.source = source
        self.field = field

        self.mask_selector = None
        self.mask_drawer = None   
        self.segment_selector = None
        self.segment_drawer = None

        self.sb_upper = 25
        self.sb_max = 23.5

        self.fig, self.axs = plt.subplots(1, 2, figsize=(15, 6))

        self.current_band = sorted(self.source.data.bands)[0]

        self.dropdown_band = widgets.Dropdown(
            options=sorted(self.source.data.bands),
            description='Band:',
            value=self.current_band
        )

        display(self.dropdown_band)

        self.load_band(self.current_band)

        self.dropdown_band.observe(self.monitor_band, names='value')

        

    def load_band(self, band):

        self.data = self.source.data[band].data
        self.segment = self.source.segment
        self.noise = self.source.data[band].noise
        self.mask = self.source.mask

        [ax.clear() for ax in self.axs]

        self.plot()

    def monitor_band(self, change):

        new_band = change['new']
        if new_band != self.current_band:

            self.current_band = new_band
            self.load_band(new_band)

    def plot(self):

        sb_func = partial(_compute_cps_SB, field=self.field, band=self.current_band)

        _plot_data(
            fig=self.fig, 
            ax=self.axs[0], 
            data=self.data, 
            noise=self.noise, 
            sb_func=sb_func, 
            sb_upper=self.sb_upper, 
            sb_max=self.sb_max
        )

        _plot_mask(
            fig=self.fig, 
            ax=self.axs[0], 
            mask=self.mask
        )

        _plot_segment(
            fig=self.fig, 
            ax=self.axs[1], 
            segment=self.segment
        )

        _plot_mask(
            fig=self.fig, 
            ax=self.axs[1], 
            mask=self.mask
        )

        self.fig.canvas.draw()

    def draw_mask(self, zoom=(False, 100)):

        [ax.clear() for ax in self.axs]

        xlim = -0.1 * self.data.shape[0], self.data.shape[0] * 1.1
        ylim = -0.1 * self.data.shape[1], self.data.shape[1] * 1.1

        if zoom[0]:
            xlim = (xlim[0] + xlim[1]) /2. - zoom[1]/2., (xlim[0] + xlim[1]) /2. + zoom[1]/2.
            ylim = (ylim[0] + ylim[1]) /2. - zoom[1]/2., (ylim[0] + ylim[1]) /2. + zoom[1]/2.

        for ax in self.axs:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        self.plot()

        self.draw_mask_handle = InteractiveMaskDrawer(self.fig, self.axs[0], self.mask)

        return self.draw_mask_handle
    
    def dilate_mask(self):

        [ax.clear() for ax in self.axs]

        self.plot()

        self.dilate_mask_handle = InteractiveMaskDilator(self.fig, self.axs[0], self.mask)

        return self.dilate_mask_handle
    
    def draw_segment(self, zoom=(False, 100)):

        [ax.clear() for ax in self.axs]

        xlim = -0.1 * self.data.shape[0], self.data.shape[0] * 1.1
        ylim = -0.1 * self.data.shape[1], self.data.shape[1] * 1.1

        if zoom[0]:
            xlim = (xlim[0] + xlim[1]) /2. - zoom[1]/2., (xlim[0] + xlim[1]) /2. + zoom[1]/2.
            ylim = (ylim[0] + ylim[1]) /2. - zoom[1]/2., (ylim[0] + ylim[1]) /2. + zoom[1]/2.

            for ax in self.axs:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

        self.plot()

        self.draw_segment_handle = InteractiveSegmentDrawer(self.fig, self.axs[0], self.segment)

        return self.draw_segment_handle
    
    def remove_segment(self):

        [ax.clear() for ax in self.axs]

        self.plot()

        self.remove_segment_handle = InteractiveSegmentSelector(self.fig, self.axs[1], self.segment)

        return self.remove_segment_handle
    


class PlotApModel:
    def __init__(self, results, source, field):

        self.results = results
        self.source = source
        self.field = field

        for band in sorted(self.source.data.bands):
            results[band]['target'].mask = source.mask
        self.mask_selector = None
        self.mask_drawer = None   
        self.segment_selector = None
        self.segment_drawer = None

        self.sb_upper = 25
        self.sb_max = 23.5

        self.fig, self.axs = plt.subplots(1, 2, figsize=(15, 6))

        self.current_band = sorted(self.source.data.bands)[0]

        # dropdown for the band selection

        self.dropdown_band = widgets.Dropdown(
            options=sorted(self.source.data.bands),
            description='Band:',
            value=self.current_band
        )

        display(self.dropdown_band)
        self.dropdown_band.observe(self.monitor_band, names='value')

        # dropdown for the data type selection

        self.current_data_type = 'data'

        self.dropdown_data_type = widgets.Dropdown(
            options=['data', 'residuals'],
            description='Data Type:',
            value=self.current_data_type
        )

        display(self.dropdown_data_type)
        self.dropdown_data_type.observe(self.monitor_data_type, names='value')

        self.load_band()
        

    def load_band(self):

        self.target = self.results[self.current_band]['target']

        if self.current_data_type == 'data':
            self.data = self.target.data.detach().cpu().numpy()

        self.model = self.results[self.current_band]['model']
        self.model.target = self.target

        # define the model_image

        base_image = self.target.model_image()
        model_total = self.model(image=base_image)
        self.model_image = model_total.data.detach().cpu().numpy()

        if self.current_data_type == 'residuals':
            self.data = self.target.data.detach().cpu().numpy() - self.model_image

        self.segment = self.source.segment
        self.noise = self.source.data[self.current_band].noise
        self.mask = self.source.mask

        [ax.clear() for ax in self.axs]

        self.plot()

    def monitor_band(self, change):

        new_band = change['new']
        if new_band != self.current_band:

            self.current_band = new_band
            self.load_band()

    def monitor_data_type(self, change):

        new_data_type = change['new']
        if new_data_type != self.current_data_type:

            self.current_data_type = new_data_type
            self.load_band()

    
    def plot(self):
        sb_func = partial(_compute_cps_SB, field=self.field, band=self.current_band)

        # plot the first axis

        _plot_data(
            fig=self.fig, 
            ax=self.axs[0], 
            data=self.data, 
            noise=self.noise, 
            sb_func=sb_func, 
            sb_upper=self.sb_upper, 
            sb_max=self.sb_max
        )

        _plot_mask(
            fig=self.fig, 
            ax=self.axs[0], 
            mask=self.mask
        )

        # plot the second axis

        _plot_model(
            fig=self.fig, 
            ax=self.axs[1], 
            model_image=self.model_image,
            noise=self.noise,
            sb_func=sb_func, 
            sb_upper=self.sb_upper, 
            sb_max=self.sb_max
        )

        for name, _model in self.results[self.current_band]['model'].models.items():

            window = _model.window

            self.axs[1], center_pix, size = _plot_window(
                ax=self.axs[1], 
                window=window, 
                alpha=0.5
            )

            self.axs[1].text(
                center_pix[0] + 1.5*size[0]/2, 
                center_pix[1] + 1.5*size[1]/2,  #
                name, 
                color='C1', 
                fontsize=6,
                ha='center', 
                va='center', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )



        _plot_segment(
            fig=self.fig, 
            ax=self.axs[1], 
            segment=self.segment
        )

        _plot_mask(
            fig=self.fig, 
            ax=self.axs[1], 
            mask=self.mask
        )

        self.fig.canvas.draw()


    def draw_mask(self, zoom=(False, 100)):

        [ax.clear() for ax in self.axs]

        xlim = -0.1 * self.data.shape[0], self.data.shape[0] * 1.1
        ylim = -0.1 * self.data.shape[1], self.data.shape[1] * 1.1

        if zoom[0]:
            xlim = (xlim[0] + xlim[1]) /2. - zoom[1]/2., (xlim[0] + xlim[1]) /2. + zoom[1]/2.
            ylim = (ylim[0] + ylim[1]) /2. - zoom[1]/2., (ylim[0] + ylim[1]) /2. + zoom[1]/2.

        for ax in self.axs:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        self.plot()

        self.draw_mask_handle = InteractiveMaskDrawer(self.fig, self.axs[0], self.mask)

        return self.draw_mask_handle
    
    def dilate_mask(self):

        [ax.clear() for ax in self.axs]

        self.plot()

        self.dilate_mask_handle = InteractiveMaskDilator(self.fig, self.axs[0], self.mask)

        return self.dilate_mask_handle
    
    def draw_segment(self, zoom=(False, 100)):

        [ax.clear() for ax in self.axs]

        xlim = -0.1 * self.data.shape[0], self.data.shape[0] * 1.1
        ylim = -0.1 * self.data.shape[1], self.data.shape[1] * 1.1

        if zoom[0]:
            xlim = (xlim[0] + xlim[1]) /2. - zoom[1]/2., (xlim[0] + xlim[1]) /2. + zoom[1]/2.
            ylim = (ylim[0] + ylim[1]) /2. - zoom[1]/2., (ylim[0] + ylim[1]) /2. + zoom[1]/2.

            for ax in self.axs:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

        self.plot()

        self.draw_segment_handle = InteractiveSegmentDrawer(self.fig, self.axs[0], self.segment)

        return self.draw_segment_handle
    
    def remove_segment(self):

        [ax.clear() for ax in self.axs]

        self.plot()

        self.remove_segment_handle = InteractiveSegmentSelector(self.fig, self.axs[1], self.segment)

        return self.remove_segment_handle   