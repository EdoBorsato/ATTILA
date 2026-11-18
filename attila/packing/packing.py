"""
Packing of source cutouts into one compact rectangular image.

The detected sources are scattered over a large field, so inspecting
them one by one means browsing an image that is mostly empty sky. This
module rearranges the bounding boxes of the sources side by side, with
no overlap, inside a rectangle that is as small as possible (the
rectangle packing problem, solved with the `rpack` package) and copies
the pixels of every source to its new position. The result is a single
image that shows all the sources at once; it is what
`attila.packing.interactive.InteractivePlotter` displays for the visual
classification.

Boxes are handled in two forms: `photutils.aperture.BoundingBox`
(integer pixel limits, as stored in the sources) and `shapely` boxes
with vertices in (x, y) pixel coordinates. The helpers at the top of
the module convert between the two and copy the pixel data.
"""
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
    """
    Convert a `shapely` polygon to a `photutils` bounding box.

    Parameters
    ----------
    polygon : shapely.Polygon
        The polygon, with vertices in (x, y) pixel coordinates.

    Returns
    -------
    photutils.aperture.BoundingBox
        The bounding box of the polygon, with its limits truncated to
        integers.
    """
    minx, miny, maxx, maxy = polygon.bounds
    return BoundingBox(ixmin=int(minx), ixmax=int(maxx), iymin=int(miny), iymax=int(maxy))

def _get_wh_from_Polygon(bbox, padding=0):
        """
        Return the width and height of the bounds of a polygon.

        Parameters
        ----------
        bbox : shapely.Polygon
            The polygon, with vertices in (x, y) pixel coordinates.
        padding : int
            The number of pixels added on each side, so that each
            dimension grows by `2 * padding`.

        Returns
        -------
        tuple of int
            The `(width, height)` in pixels, i.e. the extent along x
            and along y, truncated to integers before the padding.
        """

        width = int(bbox.bounds[2] - bbox.bounds[0]) + 2*padding
        height = int(bbox.bounds[3] - bbox.bounds[1]) + 2*padding
        return (width, height)

def _add_Polygon_data(input, output, input_polygons, output_polygons, check_func=None):
    """
    Copy a rectangular region of an array into another array.

    The pixels of `input` inside the bounds of `input_polygons` are
    written into `output` at the bounds of `output_polygons`,
    overwriting what was there. The bounds are truncated to integers
    and used as the slices `[y_min:y_max, x_min:x_max]`; the two
    regions must have the same shape.

    Parameters
    ----------
    input : numpy.ndarray
        The 2D array, axis order (y, x), the pixels are read from.
    output : numpy.ndarray
        The 2D array, axis order (y, x), the pixels are written to. It
        is modified in place.
    input_polygons : shapely.Polygon
        The single box, in (x, y) pixel coordinates of `input`, to
        read.
    output_polygons : shapely.Polygon
        The single box, in (x, y) pixel coordinates of `output`, to
        write.
    check_func : callable
        A function applied to the pixels read from `input` before they
        are written; it takes and returns an array of the same shape.
        `None` means the pixels are copied unchanged.

    Returns
    -------
    numpy.ndarray
        The `output` array.
    """

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
    """
    Return a box of the same size as a polygon, moved to the origin.

    Parameters
    ----------
    polygon : shapely.Polygon
        The polygon, with vertices in (x, y) pixel coordinates.

    Returns
    -------
    shapely.Polygon
        The box with lower left corner in (0, 0) and the width and
        height of the bounds of `polygon`. It describes the full extent
        of a cutout in the pixel coordinates of the cutout itself.
    """

    minx, miny, maxx, maxy = polygon.bounds

    return sh.box(0, 0, maxx-minx, maxy-miny)

def BoundingBox_to_Polygon(boundingbox, padding=0, shape=None):
    """
    Convert a `photutils` bounding box to a `shapely` box.

    The box can be enlarged by a padding and is clipped to the image:
    the lower limits are never below 0 and, if `shape` is given, the
    upper limits never exceed the image size.

    Parameters
    ----------
    boundingbox : photutils.aperture.BoundingBox
        The bounding box to convert.
    padding : int
        The number of pixels added on each side of the box.
    shape : tuple of int
        The shape `(n_y, n_x)` of the image, as a `numpy` shape: the
        first element limits the y coordinates and the second the x
        coordinates. `None` means no upper limit.

    Returns
    -------
    shapely.Polygon
        The box with vertices in (x, y) pixel coordinates.
    """
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
    """
    A compact, non-overlapping arrangement of rectangular cutouts.

    The object is built from the boxes of the cutouts, of which only
    the sizes are kept. `get_rectangular_packing` computes where each
    box goes in the packed image; `packed_from_sources` and
    `add_data_from_array` then fill the packed image with pixels.

    Attributes
    ----------
    polygons : list of shapely.Polygon
        The input boxes moved to the origin: box `i` spans
        `(0, 0)` to `(width_i, height_i)`.
    total_area : float
        The sum of the areas of the boxes in pixels.
    optimized_polygons : list of shapely.Polygon
        The position of each box in the packed image, in (x, y) pixel
        coordinates, in the order of `polygons`. Set by
        `get_rectangular_packing`.
    bounding_shape : tuple of int
        The shape `(n_y, n_x)` of the packed image. Set by
        `get_rectangular_packing`.
    optimized_data : numpy.ndarray
        The packed image. Set by `add_data_from_array` or
        `packed_from_sources`.
    optimized_noise : numpy.ndarray
        The packed noise map. Set by `packed_from_sources`.
    optimized_segment : numpy.ndarray
        The packed segmentation map. Set by `packed_from_sources`.
    """

    def __init__(self, polygons):
        """
        Store the sizes of the boxes to pack.

        Parameters
        ----------
        polygons : list of shapely.Polygon
            The boxes of the cutouts, in (x, y) pixel coordinates.
            Their position is discarded: only the width and the height
            are kept.
        """

        self.polygons = [_get_relative_polygon(polygon) for polygon in polygons]

        self.total_area = sum([polygon.area for polygon in self.polygons])


    def get_rectangular_packing(self, max_axis_ratio, which='width'):
        """
        Compute the position of every box in the packed image.

        The boxes, each enlarged by 1 pixel on every side, are packed
        with `rpack.pack`, a heuristic that looks for an arrangement
        with a small enclosing rectangle. Each box is then placed at the
        corner returned for its enlarged version with its original
        size, which leaves a gap of at least 2 pixels between adjacent
        boxes.

        The arrangement can be transposed (x and y swapped) depending
        on `which`: with `'width'` it is transposed when it is wider
        than tall, so that the final image has `n_x <= n_y`; with
        `'height'` it is transposed when it is taller than wide, so
        that `n_y <= n_x`. Any other value keeps the arrangement of
        `rpack`.

        The results are stored in `optimized_polygons` and
        `bounding_shape`; nothing is returned.

        Parameters
        ----------
        max_axis_ratio : float
            Not used.
        which : str
            The side of the packed image that is made the shorter one,
            `'width'` or `'height'`, as described above.
        """

        # Get the width and height of each bounding box
        whs = [_get_wh_from_Polygon(bbox) for bbox in self.polygons]

        padded_whs = [_get_wh_from_Polygon(bbox, padding=1) for bbox in self.polygons]

        # Get the packing: `positions` holds the (x, y) corner of each
        # padded box.

        positions = rpack.pack(sizes=padded_whs)

        self.optimized_polygons = [
                    sh.box(pos[0], pos[1], pos[0] + size[0], pos[1] + size[1])
                    for size, pos in zip(whs, positions)
                ]
        # `rpack.bbox_size` returns (width, height); reverse it to the
        # numpy order (n_y, n_x).
        self.bounding_shape = rpack.bbox_size(padded_whs, positions)[::-1]

        if which == 'width':
            if self.bounding_shape[1] > self.bounding_shape[0]:

                # Transpose the layout by swapping x and y.
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
        """
        Plot the packed boxes as semi-transparent rectangles.

        A new figure is created and shown with `plt.show()`.

        Parameters
        ----------
        min_size : float
            The height of the figure in inches; the width follows from
            the aspect ratio of the packed image.

        Raises
        ------
        AttributeError
            If `get_rectangular_packing` has not been run.
        """

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
        """
        Fill the packed image with pixels read from a single array.

        For every box, the region of `data` given by the corresponding
        element of `polygons` is copied to the position of the box in
        the packed image. Since `polygons` holds the boxes moved to the
        origin, the region read for box `i` is
        `data[0:height_i, 0:width_i]`.

        Before being written, the pixels of box `i` can be transformed
        with `check_func(pixels, check_value=check_condition[i])`, for
        example to keep only the pixels of a segmentation map equal to
        the label of the source.

        Parameters
        ----------
        data : numpy.ndarray
            The 2D array, axis order (y, x), the pixels are read from.
        check_func : callable
            A function taking the pixels of a box and the keyword
            argument `check_value`, and returning an array of the same
            shape. `None` means the pixels are copied unchanged.
        check_condition : sequence
            The value passed as `check_value` for each box, in the
            order of `polygons`. It is required when `check_func` is
            given.

        Returns
        -------
        numpy.ndarray
            The packed image, of shape `bounding_shape` and with the
            data type of `data`; the pixels outside the boxes are 0.
            It is also stored in `optimized_data`.

        Raises
        ------
        AttributeError
            If `get_rectangular_packing` has not been run.
        """

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
        """
        Fill the packed image, noise and segmentation from the sources.

        The cutout of every source is copied in full to the position of
        its box. The sources must be in the same order as the boxes
        given to the constructor, and must hold their cutouts as
        images, so that `source.data[band]` has the attributes `data`
        and `noise` (see `Sources.get_data` with a `MultibandImage`).

        In the packed segmentation map the pixels of the segment of a
        source carry its label and all the others are 0.

        Parameters
        ----------
        sources
            The sources to pack.
        band
            The band of the cutouts to use.

        Returns
        -------
        optimized_data : numpy.ndarray
            The packed image in counts, shape `bounding_shape`; 0
            outside the boxes.
        optimized_noise : numpy.ndarray
            The packed noise map, with the same shape.
        optimized_segment : numpy.ndarray
            The packed integer segmentation map, with the same shape.
        """

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



