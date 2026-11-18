"""
Interactive inspection of a source and of its AstroPhot model.

This module provides the notebook plots used to prepare (and check)
the 2D AstroPhot fit of a source, after detection and deblending:

- `PlotTarget` shows the cutout of a `Source` (signal-to-noise image,
  mask and segment outlines) with a band dropdown.
- `PlotApModel` shows the data (or the residuals) next to the AstroPhot
  model image with the windows of its components, with band and data
  type dropdowns.

Both classes can hand the figure to the editing tools of
`attila.interactive.masking_segment` to draw / dilate the mask and to
draw / remove segments.

The images are displayed in signal-to-noise units with a two-part
colour map split at a surface brightness threshold: a linear
`coolwarm` scale below `sb_upper` and a logarithmic red-yellow-green
scale between `sb_upper` and `sb_max` (both in mag / arcsec^2,
converted to pixel values by `_compute_cps_SB` of
`attila.packing.interactive`).
"""

import numpy as np

# from photutils.segmentation import SegmentationImage
# from photutils.isophote.geometry import EllipseGeometry
# from photutils.aperture import EllipticalAperture

import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon

# from astropy.io import fits

# import pandas as pd

from matplotlib import colormaps as mcmaps
from matplotlib import colors as mcolors

# import os
# import json

import ipywidgets as widgets
from IPython.display import display

from functools import partial

# from copy import deepcopy

# from matplotlib.colors import Normalize 
# from matplotlib.widgets import Button
# from threading import Event
# import shapely as sh

# from attila.field.field import Field
# from attila.tiling.sources import Source, Sources
from attila.interactive.masking_segment import InteractiveMaskDrawer, InteractiveMaskDilator
from attila.interactive.masking_segment import InteractiveSegmentDrawer, InteractiveSegmentSelector

# from attila.fitting.deblend_2D import AP_deblend, load_results
from attila.packing.interactive import _compute_cps_SB, _correct_for_polygon_shift, _get_polygons_from_segment
from astropy.convolution import convolve, Gaussian2DKernel


def _plot_data(fig, ax, data, noise, sb_func, sb_upper, sb_max):
    """
    Show a data image in signal-to-noise units with a split colour map.

    The image `data / noise` is drawn twice on `ax`: pixels up to the
    `sb_upper` threshold with a linear `coolwarm` map starting at
    S/N = -3, and pixels above it with a logarithmic map running up to
    the `sb_max` threshold. The two surface brightness thresholds are
    converted to pixel values with `sb_func` and to S/N with the median
    of `noise`. Dashed contours of the S/N image smoothed with a
    Gaussian of sigma 1.5 pixels are overplotted at 10 logarithmic
    levels between S/N = 2 and the `sb_max` threshold; this step reads
    `.value` of the threshold, so it expects an `astropy` quantity.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure of `ax` (only used by the commented-out colour bars).
    ax : matplotlib.axes.Axes
        The axis to draw on.
    data : numpy.ndarray
        The image, shape `(n_rows, n_cols)`, in the flux units of the
        zero point used by `sb_func`.
    noise : numpy.ndarray
        The noise (RMS) map of `data`, same shape and units.
    sb_func : callable
        A function called as `sb_func(SB=value)` that converts a surface
        brightness in mag / arcsec^2 to a pixel value.
    sb_upper : float
        The surface brightness, in mag / arcsec^2, at which the colour
        map is split (the fainter of the two thresholds).
    sb_max : float
        The surface brightness, in mag / arcsec^2, at which the upper
        colour map saturates (the brighter threshold).
    """

    threshold_upper = sb_func(SB=sb_upper)
    threshold_max = sb_func(SB=sb_max)

    # The upper colour map starts from the red end of `coolwarm`, so the
    # two maps join without a colour jump at the threshold.
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

    # Convert the thresholds from pixel values to signal-to-noise.
    threshold_upper_SNR = threshold_upper / np.nanmedian(noise)
    threshold_max_SNR = threshold_max / np.nanmedian(noise)

    # The lower layer is clipped at the threshold; the upper layer,
    # drawn on top, only shows the pixels above it.
    img = data/noise
    img_bot = np.where(img <= threshold_upper_SNR, img, threshold_upper_SNR)
    ax.imshow(img_bot, vmin=-3, vmax=threshold_upper_SNR, origin='lower', cmap=bottom)

    if np.any(img >= threshold_upper_SNR):
        img_top = np.ma.masked_where(img <= threshold_upper_SNR, img)
        norm = mcolors.LogNorm(vmin=threshold_upper_SNR, vmax=threshold_max_SNR)
        ax.imshow(img_top, norm=norm, origin='lower', cmap=top)

    # ax.set_xlim(-.5, data.shape[1]-0.5)
    # ax.set_ylim(-0.1, data.shape[0]-0.5)
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Overplot contours of the smoothed signal-to-noise image.
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
    """
    Overlay the masked pixels of a mask as a translucent grey layer.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure of `ax` (not used).
    ax : matplotlib.axes.Axes
        The axis to draw on.
    mask : numpy.ndarray
        The mask, shape `(n_rows, n_cols)`; pixels `> 0` are shown and
        the others are transparent.
    """

    plot_mask = np.where(mask > 0, mask, np.nan)

    ax.imshow(plot_mask, origin='lower', cmap='gray', alpha=0.5)

    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

def _plot_segment(fig, ax, segment):
    """
    Draw the outline of every segment of a segmentation map.

    The segments are converted to polygons with
    `_get_polygons_from_segment` and shifted by half a pixel so that
    they line up with an `imshow` of the same cutout. Each polygon is
    hatched with a colour of the default cycle (`C0`, `C1`, ...). The
    axis limits are set to the extent of the map.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure of `ax` (not used).
    ax : matplotlib.axes.Axes
        The axis to draw on.
    segment : numpy.ndarray
        The segmentation map, shape `(n_rows, n_cols)`; 0 is background.
    """

    ax.set_aspect('equal')

    ax.set_xlim(0, segment.shape[1])
    ax.set_ylim(0, segment.shape[0])

    polys = _get_polygons_from_segment(segment)

    polys = [_correct_for_polygon_shift(poly) for poly in polys]

    for i, poly in enumerate(polys):
        plot_polygon(poly, ax=ax, add_points=False, ls='-', lw=3, ec=f'C{i}', fc='none', hatch='//', alpha=1)

    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)


def _plot_window(ax, window, **kwargs):
    """
    Draw the rectangular window of an AstroPhot model component.

    The window centre is converted from arcsec to pixels by dividing by
    the diagonal terms of the window pixel scale matrix, then the
    rectangle of the window is added to `ax`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    window : astrophot.image.Window
        The window of the model component.
    **kwargs
        Extra keyword arguments passed to `matplotlib.pyplot.Rectangle`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The input axis.
    center_pix : list of float
        The (x, y) centre of the window in pixels.
    size : tuple
        The (dx, dy) size of the window in pixels.
    """

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
    """
    Show a model image with the split colour map of `_plot_data`.

    The display is the same as in `_plot_data`, except that the model
    is divided by the median of `noise` (a single number) instead of
    the noise map pixel by pixel.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure of `ax` (not used).
    ax : matplotlib.axes.Axes
        The axis to draw on.
    model_image : numpy.ndarray
        The model image, shape `(n_rows, n_cols)`, in the same units as
        the data.
    noise : numpy.ndarray
        The noise (RMS) map of the data, same shape.
    sb_func : callable
        A function called as `sb_func(SB=value)` that converts a surface
        brightness in mag / arcsec^2 to a pixel value.
    sb_upper : float
        The surface brightness, in mag / arcsec^2, at which the colour
        map is split.
    sb_max : float
        The surface brightness, in mag / arcsec^2, at which the upper
        colour map saturates.
    """

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
    ax.imshow(img_bot, vmin=-3, vmax=threshold_upper_SNR, origin='lower', cmap=bottom)

    if np.any(img >= threshold_upper_SNR):
        img_top = np.ma.masked_where(img <= threshold_upper_SNR, img)
        norm = mcolors.LogNorm(vmin=threshold_upper_SNR, vmax=threshold_max_SNR)
        ax.imshow(img_top, norm=norm, origin='lower', cmap=top)

    kernel = Gaussian2DKernel(x_stddev=1.5, y_stddev=1.5)

    convolved_image = convolve(img, kernel, boundary='extend')
    
    levels = np.logspace(np.log10(2), np.log10(threshold_max_SNR.value), num=10)

    ax.contour(convolved_image, levels=levels, colors='k', linewidths=1, linestyles='dashed', alpha=0.5)

    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)


def _plot_residual(fig, ax, residual, noise, sb_func, sb_upper, sb_max):
    """
    Show a residual image with the split colour map of `_plot_data`.

    The image `residual / noise` is displayed exactly as `_plot_data`
    displays `data / noise`. The function is not called in this module.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure of `ax` (only used by the commented-out colour bars).
    ax : matplotlib.axes.Axes
        The axis to draw on.
    residual : numpy.ndarray
        The data minus model image, shape `(n_rows, n_cols)`.
    noise : numpy.ndarray
        The noise (RMS) map of the data, same shape.
    sb_func : callable
        A function called as `sb_func(SB=value)` that converts a surface
        brightness in mag / arcsec^2 to a pixel value.
    sb_upper : float
        The surface brightness, in mag / arcsec^2, at which the colour
        map is split.
    sb_max : float
        The surface brightness, in mag / arcsec^2, at which the upper
        colour map saturates.
    """

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
    ax.imshow(img_bot, vmin=-3, vmax=threshold_upper_SNR, origin='lower', cmap=bottom)

    if np.any(img >= threshold_upper_SNR):
        img_top = np.ma.masked_where(img <= threshold_upper_SNR, img)
        norm = mcolors.LogNorm(vmin=threshold_upper_SNR, vmax=threshold_max_SNR)
        ax.imshow(img_top, norm=norm, origin='lower', cmap=top)

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
    """
    Two-panel interactive view of a source cutout.

    The left panel shows the signal-to-noise image of the selected band
    with the mask on top; the right panel shows the segment outlines
    with the same mask. Creating the object opens the figure and
    displays a dropdown to change band. The `draw_mask`, `dilate_mask`,
    `draw_segment` and `remove_segment` methods redraw the figure and
    return an editing object of `attila.interactive.masking_segment`,
    whose handler the caller must then connect to the canvas.

    Attributes
    ----------
    source : Source
        The source being displayed.
    field : Field
        The field the source belongs to (zero points, pixel scales).
    sb_upper : float
        The surface brightness at which the colour map is split,
        25 mag / arcsec^2.
    sb_max : float
        The surface brightness at which the colour map saturates,
        23.5 mag / arcsec^2.
    fig, axs
        The matplotlib figure and its two axes.
    current_band : str
        The band being displayed.
    dropdown_band : ipywidgets.Dropdown
        The band selector.
    data, noise : numpy.ndarray
        The image and noise cutouts of the current band.
    segment, mask : numpy.ndarray
        The segmentation map and the mask of the source (references to
        the arrays stored on `source`).
    """
    def __init__(self, source, field):
        """
        Open the figure, display the band dropdown and plot the source.

        Parameters
        ----------
        source : Source
            The source to display. It must carry per-band image cutouts
            in `source.data` (with `data` and `noise`), a `segment` map
            and a `mask` array.
        field : Field
            The field of the source; it must have the `zpt` attribute
            needed by `_compute_cps_SB`.
        """

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
        """
        Load the cutouts of a band from the source and redraw.

        Parameters
        ----------
        band : str
            The band to display.
        """

        self.data = self.source.data[band].data
        self.segment = self.source.segment
        self.noise = self.source.data[band].noise
        self.mask = self.source.mask

        [ax.clear() for ax in self.axs]

        self.plot()

    def monitor_band(self, change):
        """
        React to the band dropdown and reload when the band changes.

        Parameters
        ----------
        change : dict
            The ipywidgets change notification; `change['new']` is the
            selected band.
        """

        new_band = change['new']
        if new_band != self.current_band:

            self.current_band = new_band
            self.load_band(new_band)

    def plot(self):
        """Draw the S/N image, the segment outlines and the mask."""

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
        """
        Redraw the figure and return a tool to draw mask polygons.

        Both axes are cleared, their limits are set and the source is
        plotted again. The limits are the cutout size enlarged by 10% on
        each side, or a box of side `zoom[1]` around the centre when
        `zoom[0]` is true; the x limits are computed from
        `data.shape[0]` and the y limits from `data.shape[1]`.

        Parameters
        ----------
        zoom : tuple
            `(flag, size)`: if `flag` is true the view is restricted to
            a box of `size` pixels around the cutout centre.

        Returns
        -------
        InteractiveMaskDrawer
            The drawer bound to the left (data) axis; connect its
            `on_click_draw` to `button_press_event` and call `to_mask()`
            when done.
        """

        [ax.clear() for ax in self.axs]

        xlim = -0.1 * self.data.shape[1], self.data.shape[1] * 1.1
        ylim = -0.1 * self.data.shape[0], self.data.shape[0] * 1.1

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
        """
        Redraw the figure and return a tool to dilate mask blobs.

        Returns
        -------
        InteractiveMaskDilator
            The dilator bound to the left (data) axis; connect its
            `on_click_dilate` to `button_press_event` and call
            `to_mask()` when done.
        """

        [ax.clear() for ax in self.axs]

        self.plot()

        self.dilate_mask_handle = InteractiveMaskDilator(self.fig, self.axs[0], self.mask)

        return self.dilate_mask_handle
    
    def draw_segment(self, zoom=(False, 100)):
        """
        Redraw the figure and return a tool to draw new segments.

        Both axes are cleared and the source is plotted again. The axis
        limits are only set when `zoom[0]` is true, to a box of side
        `zoom[1]` around the centre of the 10% enlarged cutout (x
        limits from `data.shape[0]`, y limits from `data.shape[1]`).

        Parameters
        ----------
        zoom : tuple
            `(flag, size)`: if `flag` is true the view is restricted to
            a box of `size` pixels around the cutout centre.

        Returns
        -------
        InteractiveSegmentDrawer
            The drawer bound to the left (data) axis; connect its
            `on_click_draw` to `button_press_event` and call
            `to_segment()` when done.
        """

        [ax.clear() for ax in self.axs]

        xlim = -0.1 * self.data.shape[1], self.data.shape[1] * 1.1
        ylim = -0.1 * self.data.shape[0], self.data.shape[0] * 1.1

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
        """
        Redraw the figure and return a tool to delete segments.

        Returns
        -------
        InteractiveSegmentSelector
            The selector bound to the right (segment) axis; connect its
            `on_click_remove` to `button_press_event`. It edits the
            segmentation map of the source in place.
        """

        [ax.clear() for ax in self.axs]

        self.plot()

        self.remove_segment_handle = InteractiveSegmentSelector(self.fig, self.axs[1], self.segment)

        return self.remove_segment_handle
    


class PlotApModel:
    """
    Two-panel interactive view of a source and its AstroPhot model.

    The left panel shows the data, or the data minus model residuals,
    of the selected band in signal-to-noise units with the mask on top.
    The right panel shows the model image with the window and the name
    of every model component, the segment outlines and the mask.
    Creating the object opens the figure and displays two dropdowns
    (band and data type). The editing methods are the same as in
    `PlotTarget`.

    Attributes
    ----------
    results : dict
        The fit products per band: `results[band]['target']` is the
        AstroPhot target image and `results[band]['model']` the group
        model.
    source : Source
        The source being displayed.
    field : Field
        The field the source belongs to.
    sb_upper, sb_max : float
        The surface brightness thresholds of the colour map, 25 and
        23.5 mag / arcsec^2.
    fig, axs
        The matplotlib figure and its two axes.
    current_band : str
        The band being displayed.
    current_data_type : str
        `'data'` or `'residuals'`.
    target, model
        The AstroPhot target image and model of the current band.
    data : numpy.ndarray
        The image shown in the left panel (data or residuals).
    model_image : numpy.ndarray
        The model evaluated on the target pixel grid.
    noise, segment, mask : numpy.ndarray
        The noise cutout of the current band, the segmentation map and
        the mask of the source.
    """
    def __init__(self, results, source, field):
        """
        Open the figure, display the dropdowns and plot the model.

        The mask of the source is assigned to the `mask` of the target
        image of every band, so `results` is modified in place.

        Parameters
        ----------
        results : dict
            The fit products, indexed as `results[band]['target']` and
            `results[band]['model']`.
        source : Source
            The fitted source, with per-band cutouts in `source.data`, a
            `segment` map and a `mask` array.
        field : Field
            The field of the source; it must have the `zpt` attribute
            needed by `_compute_cps_SB`.
        """

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

        # Dropdown for the band selection.

        self.dropdown_band = widgets.Dropdown(
            options=sorted(self.source.data.bands),
            description='Band:',
            value=self.current_band
        )

        display(self.dropdown_band)
        self.dropdown_band.observe(self.monitor_band, names='value')

        # Dropdown for the data type selection.

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
        """
        Load the target and the model of the current band and redraw.

        The model of the band is attached to the target, evaluated on an
        empty model image of the target and converted to a numpy array.
        The image to show is the target data, or the target data minus
        the model image when the data type is `'residuals'`.
        """

        self.target = self.results[self.current_band]['target']

        if self.current_data_type == 'data':
            self.data = self.target.data.detach().cpu().numpy()

        self.model = self.results[self.current_band]['model']
        self.model.target = self.target

        # Define the model image by sampling the model on the target
        # grid.

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
        """
        React to the band dropdown and reload when the band changes.

        Parameters
        ----------
        change : dict
            The ipywidgets change notification; `change['new']` is the
            selected band.
        """

        new_band = change['new']
        if new_band != self.current_band:

            self.current_band = new_band
            self.load_band()

    def monitor_data_type(self, change):
        """
        React to the data type dropdown and reload when it changes.

        Parameters
        ----------
        change : dict
            The ipywidgets change notification; `change['new']` is
            `'data'` or `'residuals'`.
        """

        new_data_type = change['new']
        if new_data_type != self.current_data_type:

            self.current_data_type = new_data_type
            self.load_band()

    
    def plot(self):
        """
        Draw the data / residuals and the model with its windows.

        Left axis: the current image in S/N units and the mask. Right
        axis: the model image, one labelled rectangle per component of
        the model (its AstroPhot window), the segment outlines and the
        mask.
        """
        sb_func = partial(_compute_cps_SB, field=self.field, band=self.current_band)

        # Plot the first axis.

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

        # Plot the second axis.

        _plot_model(
            fig=self.fig, 
            ax=self.axs[1], 
            model_image=self.model_image,
            noise=self.noise,
            sb_func=sb_func, 
            sb_upper=self.sb_upper, 
            sb_max=self.sb_max
        )

        # Draw the window of every model component, labelled with the
        # component name beyond its upper right corner.
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
        """
        Redraw the figure and return a tool to draw mask polygons.

        Both axes are cleared, their limits are set and the figure is
        plotted again. The limits are the cutout size enlarged by 10% on
        each side, or a box of side `zoom[1]` around the centre when
        `zoom[0]` is true; the x limits are computed from
        `data.shape[0]` and the y limits from `data.shape[1]`.

        Parameters
        ----------
        zoom : tuple
            `(flag, size)`: if `flag` is true the view is restricted to
            a box of `size` pixels around the cutout centre.

        Returns
        -------
        InteractiveMaskDrawer
            The drawer bound to the left (data) axis; connect its
            `on_click_draw` to `button_press_event` and call `to_mask()`
            when done.
        """

        [ax.clear() for ax in self.axs]

        xlim = -0.1 * self.data.shape[1], self.data.shape[1] * 1.1
        ylim = -0.1 * self.data.shape[0], self.data.shape[0] * 1.1

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
        """
        Redraw the figure and return a tool to dilate mask blobs.

        Returns
        -------
        InteractiveMaskDilator
            The dilator bound to the left (data) axis; connect its
            `on_click_dilate` to `button_press_event` and call
            `to_mask()` when done.
        """

        [ax.clear() for ax in self.axs]

        self.plot()

        self.dilate_mask_handle = InteractiveMaskDilator(self.fig, self.axs[0], self.mask)

        return self.dilate_mask_handle
    
    def draw_segment(self, zoom=(False, 100)):
        """
        Redraw the figure and return a tool to draw new segments.

        Both axes are cleared and the figure is plotted again. The axis
        limits are only set when `zoom[0]` is true, to a box of side
        `zoom[1]` around the centre of the 10% enlarged cutout (x
        limits from `data.shape[0]`, y limits from `data.shape[1]`).

        Parameters
        ----------
        zoom : tuple
            `(flag, size)`: if `flag` is true the view is restricted to
            a box of `size` pixels around the cutout centre.

        Returns
        -------
        InteractiveSegmentDrawer
            The drawer bound to the left (data) axis; connect its
            `on_click_draw` to `button_press_event` and call
            `to_segment()` when done.
        """

        [ax.clear() for ax in self.axs]

        xlim = -0.1 * self.data.shape[1], self.data.shape[1] * 1.1
        ylim = -0.1 * self.data.shape[0], self.data.shape[0] * 1.1

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
        """
        Redraw the figure and return a tool to delete segments.

        Returns
        -------
        InteractiveSegmentSelector
            The selector bound to the right (model) axis; connect its
            `on_click_remove` to `button_press_event`. It edits the
            segmentation map of the source in place.
        """

        [ax.clear() for ax in self.axs]

        self.plot()

        self.remove_segment_handle = InteractiveSegmentSelector(self.fig, self.axs[1], self.segment)

        return self.remove_segment_handle   