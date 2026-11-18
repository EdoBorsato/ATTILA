"""
Interactive visual classification of the detected sources.

The module provides `InteractivePlotter`, a `matplotlib` figure that
shows all the sources of a `Sources` container at once. Their cutouts
are packed into a single image by `attila.packing.packing.Packing` and
displayed as a g-band signal-to-noise map, with the bounding box of
each source drawn on top. The user clicks on a source to select it and
presses `0`, `1` or `2` to assign it a class; `q` closes the figure.
The classification can then be saved to a JSON file.

The helper functions build the outlines of the segments, convert a
surface-brightness level to counts per pixel and draw the image.
"""
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
    """
    Vectorise the regions of a segmentation map into polygons.

    The connected regions of equal value (8-connectivity) are traced
    with `rasterio.features.shapes`. By default only the labelled
    pixels are traced, giving one polygon per connected segment; with
    `invert` the background is traced instead.

    The vertices lie on the pixel corners, with the corner of the first
    pixel at (0, 0); see `_correct_for_polygon_shift`.

    Parameters
    ----------
    segment : numpy.ndarray
        The 2D segmentation map, axis order (y, x), with 0 for the
        background.
    invert : bool
        If True, return the polygons of the background (the pixels
        equal to 0) instead of those of the segments.

    Returns
    -------
    list of shapely.Polygon
        The outlines, with vertices in (x, y) pixel-corner
        coordinates.
    """

    if invert:
        mask = np.where(segment==0, True, False)
    else:
        mask = np.where(segment==0, False, True)

    shape_polygon_list = list(shapes(segment.astype(np.int32), mask=mask, connectivity=8))

    return [sh.geometry.shape(geo_poly[0]) for geo_poly in shape_polygon_list]

def _correct_for_polygon_shift(polygon):
    """
    Shift a polygon by half a pixel towards the origin.

    The boxes and outlines have their vertices on the pixel corners,
    while `imshow` puts the centre of the first pixel at (0, 0). The
    shift of -0.5 pixels along both axes aligns the two conventions.

    Parameters
    ----------
    polygon : shapely.Polygon
        The polygon in (x, y) pixel-corner coordinates.

    Returns
    -------
    shapely.Polygon
        The translated polygon.
    """

    return sh.affinity.translate(polygon, xoff=-0.5, yoff=-0.5)

def _compute_cps_SB(field:Field, SB:float, band:str):
    """
    Convert a surface brightness to counts per pixel.

    The observed level is

        SB - zpt - 5 log10(pixel_scale)

    to which `10 log10(1 + z)` is added when the field has a
    `redshift`, so that `SB` is read as a surface brightness corrected
    for the `(1 + z)^4` cosmological dimming. The result is
    `10 ** (-0.4 * level)`. No extinction is applied.

    Parameters
    ----------
    field
        The field providing the zero point (`field.zpt[band]`), the
        pixel scale of the band and, optionally, the redshift.
    SB
        The surface brightness in mag / arcsec^2.
    band
        The band of the conversion.

    Returns
    -------
    float or astropy.units.Quantity
        The pixel value in the counts of the image, per pixel. It is a
        `Quantity` when `field.redshift` is one.

    Raises
    ------
    AttributeError
        If the field has no attribute `zpt`.
    """

    if not hasattr(field, "zpt"):
        raise AttributeError("Field object must have a zpt attribute")

    pixel_scale = field.multibandimages[band].pixel_scale.value

    zpt = field.zpt[band]

    obs_SB = SB - zpt - 5*np.log10(pixel_scale)

    if hasattr(field, "redshift"):
        obs_SB += 2.5*np.log10((1 + field.redshift)**4)

    return 10**(-0.4*obs_SB)


def _plot(ax, data, noise, segment, polygons, threshold_upper, threshold_max):
    """
    Draw the packed signal-to-noise image with the source boxes.

    The image shown is `data / noise`. It is drawn in two layers so
    that faint and bright pixels use different colour scales: the
    pixels below the level `threshold_upper` with a linear `coolwarm`
    scale starting at -3, and the pixels above it with a logarithmic
    red-yellow-green scale reaching up to `threshold_max`. The two
    thresholds, given in counts per pixel, are converted to
    signal-to-noise by dividing them by the median of `noise`.

    The boxes of the sources are outlined in blue, and dashed black
    contours are drawn on the image smoothed with a Gaussian of
    standard deviation 1.5 pixels, at 10 levels logarithmically spaced
    between a signal-to-noise of 2 and the upper threshold.

    The outlines of the background regions of `segment` are computed,
    but the code that draws them is disabled.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    data : numpy.ndarray
        The packed image in counts, axis order (y, x).
    noise : numpy.ndarray
        The packed noise map in counts, with the shape of `data`.
    segment : numpy.ndarray
        The packed segmentation map, with the shape of `data`.
    polygons : list of shapely.Polygon
        The box of each source in the packed image, in (x, y)
        pixel-corner coordinates.
    threshold_upper : float or astropy.units.Quantity
        The level, in counts per pixel, that separates the two colour
        scales.
    threshold_max : astropy.units.Quantity
        The level, in counts per pixel, at which the upper colour
        scale and the contour levels end. Its ratio to the median
        noise is read through the attribute `value`.
    """

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

    # Lower layer: the image clipped at the upper threshold.
    img = data/noise
    img_bot = np.where(img <= threshold_upper_SNR, img, threshold_upper_SNR)
    ax.imshow(img_bot, vmin=-3, vmax=threshold_upper_SNR, origin='lower', cmap=bottom)

    # Upper layer: only the pixels above the threshold, in log scale.
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
    """
    A figure to classify the sources by eye.

    On creation the cutouts of the sources are packed into one image
    (g band) and displayed with `_plot`. A mouse click inside the box
    of a source selects it; pressing `0`, `1` or `2` then stores that
    key as the class of the selected source and redraws its box in
    red, blue (`C0`) or orange (`C1`) respectively. Pressing `q` closes
    the figure. Every source starts in class `'1'`. The meaning of the
    three classes is not defined in this module.

    Attributes
    ----------
    sources : Sources
        The sources shown in the figure.
    polygons : list of shapely.Polygon
        The bounding box of each source in the full image.
    packing : Packing
        The packing of the boxes; `packing.optimized_polygons` are the
        boxes in the displayed image.
    data, noise, segment : numpy.ndarray
        The packed g-band image, noise map and segmentation map.
    threshold_upper, threshold_max : float or astropy.units.Quantity
        The counts per pixel corresponding to 25 and 23.5
        mag / arcsec^2 in the g band, used for the colour scales.
    classification : dict
        The class of each source, `'0'`, `'1'` or `'2'`, keyed by the
        string `"<previous_label>_<label>"`.
    selected_polygon_index : int
        The index of the source selected by the last click; it does
        not exist before the first click.
    fig, ax
        The `matplotlib` figure and axes.
    """
    def __init__(self, sources: Sources, field:Field):
        """
        Pack the sources, open the figure and connect the events.

        The constructor ends by calling `plot_data`, which shows the
        figure.

        Parameters
        ----------
        sources
            The sources to classify. They must hold their cutouts as
            images with a `'g'` band (see
            `Packing.packed_from_sources`).
        field
            The field of the sources; it must have the attribute `zpt`
            with the zero point of the `'g'` band.
        """

        self.sources = sources

        self.polygons = [BoundingBox_to_Polygon(source.bbox) for source in sources]

        self.packing = Packing(self.polygons)

        self.packing.get_rectangular_packing(max_axis_ratio=0.6)

        self.data, self.noise, self.segment = self.packing.packed_from_sources(sources, 'g')

        # Surface-brightness levels of the colour scales, in
        # mag / arcsec^2, converted to counts per pixel.
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
        """Draw the packed image on the axes and show the figure."""
        _plot(self.ax, self.data, self.noise, self.segment, self.packing.optimized_polygons, self.threshold_upper, self.threshold_max)
        plt.show()


    def on_click(self, event):
        """
        Select the source whose box contains the clicked point.

        The index of the source is stored in `selected_polygon_index`
        and its box is redrawn in the colour of class `'1'`. The stored
        class of the source is not changed by the click.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse-press event; `xdata` and `ydata` are the (x, y)
            pixel coordinates in the packed image.
        """
        click_point = Point(event.xdata, event.ydata)
        for i, polygon in enumerate(self.packing.optimized_polygons):
            if polygon.contains(click_point):
                self.selected_polygon_index = i
                # print(f"Polygon {i} selected")

                self.highlight_polygon(polygon, "1")

                break

    def on_key_press(self, event):
        """
        Close the figure or set the class of the selected source.

        The key `q` closes the figure. If a source has been selected,
        the keys `0`, `1` and `2` store that character as its class in
        `classification` and redraw its box in the matching colour.

        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            The key-press event.
        """
        if event.key == 'q':
            plt.close(self.fig)

        if hasattr(self, 'selected_polygon_index'):
            if (event.key =='2') | (event.key =='1') | (event.key =='0'):

                selected_source = self.sources[self.selected_polygon_index]

                _id = f"{selected_source.previous_label}_{selected_source.label}"

                self.classification[_id] = event.key

                self.highlight_polygon(self.packing.optimized_polygons[self.selected_polygon_index], event.key)

    def save(self, filename:str):
        """
        Write the classification to a JSON file.

        The file is `output/classification/<filename>.json`, relative
        to the current working directory; the folder must exist.

        Parameters
        ----------
        filename
            The name of the file, without folder and extension.

        Returns
        -------
        dict
            The `classification` dictionary.
        """

        with open(f"output/classification/{filename}.json", 'w+') as f:
            json.dump(self.classification, f, indent=4)

        return self.classification

    def highlight_polygon(self, polygon, classification):
        """
        Redraw the outline of a box in the colour of a class.

        The colours are orange (`C1`) for `'2'`, blue (`C0`) for `'1'`
        and red for `'0'`. The new outline is drawn over the existing
        ones, which are not removed; for class `'0'` it is drawn
        twice. Any other value draws nothing.

        Parameters
        ----------
        polygon : shapely.Polygon
            The box in the packed image, in (x, y) pixel-corner
            coordinates.
        classification : str
            The class, `'0'`, `'1'` or `'2'`.
        """

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
