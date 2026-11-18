"""
HEALPix tiling of a field, with detection by cell ownership.

`HealpixTiling` is an alternative to `attila.tiling.tilingrid.Tiling`
in which the tiles are the cells of the HEALPix tessellation of the
sky (nested scheme, ICRS). The cells are defined on the sky and not by
the data: a cell identifier points to the same region for every band,
epoch and survey, the cells have the same area at every declination,
and their identifiers are the index used by HATS / LSDB catalogues.

Only the index of a tile is HEALPix. The data of a tile are cut out of
the images on their own pixel grid, as the bounding box of the cell
plus a margin, so nothing is resampled.

The tiles are processed independently and their results are not
stitched together. `detect_in_cells` segments the cutout of every
cell and assigns every source to exactly one cell:

- a source belongs to the cell that contains its peak, the pixel with
  the highest value of the detection array;
- a source that is cut by the border of a cutout is not trusted there:
  it is detected again on a box that contains the whole source.

A source is therefore never split at the border between two tiles and
never counted twice, whatever its size.
"""
import multiprocessing
import pickle

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from os import path
from threading import Lock
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

import shapely as sh

from astropy import units as u
from astropy.coordinates import ICRS
from astropy.io import fits
from astropy.wcs import WCS
from astropy_healpix import HEALPix
from photutils.aperture import BoundingBox
from scipy import ndimage
from tqdm import tqdm

from attila.tiling.sources import Source, Sources, terrace_values
from attila.tiling.tiles import BaseTile


def _clipped_box(box, shape):
    """
    Clip a pixel box to an image.

    Parameters
    ----------
    box : tuple of int
        The box `(y0, y1, x0, x1)`, with exclusive upper limits.
    shape : tuple of int
        The shape `(n_y, n_x)` of the image.

    Returns
    -------
    tuple of int
        The box restricted to the image.
    """
    y0, y1, x0, x1 = box

    return max(y0, 0), min(y1, shape[0]), max(x0, 0), min(x1, shape[1])


def _touches_border(obj_slices, box, shape):
    """
    Tell whether an object reaches the border of its cutout.

    A border of the cutout that coincides with the border of the image
    does not count: the object cannot continue beyond it.

    Parameters
    ----------
    obj_slices : tuple of slice
        The bounding box of the object inside the cutout, `(y, x)`.
    box : tuple of int
        The box `(y0, y1, x0, x1)` of the cutout in the image.
    shape : tuple of int
        The shape `(n_y, n_x)` of the image.

    Returns
    -------
    bool
        `True` if the object may continue outside the cutout.
    """
    y0, y1, x0, x1 = box
    sl_y, sl_x = obj_slices

    return ((sl_y.start == 0 and y0 > 0)
            or (sl_y.stop == y1 - y0 and y1 < shape[0])
            or (sl_x.start == 0 and x0 > 0)
            or (sl_x.stop == x1 - x0 and x1 < shape[1]))


class HealpixTiling:
    """
    A tiling of the common footprint of the bands in HEALPix cells.

    The tiles are the HEALPix cells of a given order (nested scheme,
    ICRS) that fall on the images of all the bands. The object offers
    the same interface as `attila.tiling.tilingrid.Tiling`: the tile
    centres and footprints (`centres`, `polygons`), the extraction of
    the data of a tile (`get_tile_data`) and the writing of a tile
    into a full-size array (`tile_to_mosaic`).

    The side of a cell is about 58.6 deg / 2^order: 13.7 arcmin at
    order 8, 6.9 arcmin at order 9 and 3.4 arcmin at order 10.

    Attributes
    ----------
    header_dict : dict
        For every band, the FITS header of the full image.
    wcs_dict : dict
        For every band, the `astropy.wcs.WCS` of the full image.
    order : int
        The HEALPix order of the cells.
    margin_sky : float
        The margin added around the cells in the cutouts, in degrees.
    healpix : astropy_healpix.HEALPix
        The tessellation.
    cell_ids : numpy.ndarray
        The HEALPix identifiers of the tiles (nested), sorted. Set by
        `make_tiles`.
    centres : astropy.coordinates.SkyCoord
        The centres of the cells. Set by `make_tiles`.
    polygons : list of shapely.Polygon
        The footprints of the cells, with vertices in (RA, Dec)
        degrees. Set by `make_tiles`.
    """

    def __init__(
            self,
            header_dict: Dict[str, fits.Header],
            order: int = 9,
            margin_sky: u.Quantity = 1 * u.arcmin,
        ) -> None:
        """
        Store the headers and define the tessellation.

        Parameters
        ----------
        header_dict
            For every band, the FITS header of the full image.
        order
            The HEALPix order of the cells.
        margin_sky
            The margin added around every cell in the cutouts, as an
            angle. It only affects the speed of the detection: sources
            larger than the margin are handled anyway, but each of
            them costs one more segmentation.
        """
        self.header_dict = dict(header_dict)
        self.order = int(order)
        self.margin_sky = margin_sky.to(u.deg).value

        self.healpix = HEALPix(nside=2**self.order, order='nested', frame=ICRS())

        self.cell_ids = None
        self.centres = None
        self.polygons = None

        self._boxes = {}
        self._lock = Lock()

    def __getstate__(self):
        """Return the state to pickle, without the lock."""
        state = self.__dict__.copy()
        state.pop('_lock', None)

        return state

    def __setstate__(self, state):
        """Restore the pickled state and create a new lock."""
        self.__dict__.update(state)
        self._lock = Lock()

    @property
    def bands(self):
        """The names of the bands."""
        return list(self.header_dict.keys())

    @property
    def wcs_dict(self):
        """The WCS of every band, built from the headers."""
        if getattr(self, '_wcs_dict', None) is None or set(self._wcs_dict) != set(self.header_dict):
            self._wcs_dict = {band: WCS(header) for band, header in self.header_dict.items()}

        return self._wcs_dict

    @property
    def cell_size_sky(self):
        """The side of a cell (square root of its area) in degrees."""
        return self.healpix.pixel_resolution.to(u.deg).value

    def _image_shape(self, band):
        """Return the shape `(n_y, n_x)` of the image of a band."""
        header = self.header_dict[band]

        return int(header['NAXIS2']), int(header['NAXIS1'])

    def _pixel_scale(self, band):
        """Return the mean pixel scale of a band in degrees."""
        scales = self.wcs_dict[band].proj_plane_pixel_scales()

        return float(np.mean([scale.to_value(u.deg) for scale in scales]))

    def margin_pixels(self, band):
        """
        Return the margin of the cutouts of a band in pixels.

        Parameters
        ----------
        band : str
            The name of the band.

        Returns
        -------
        int
            The margin, rounded up.
        """
        return int(np.ceil(self.margin_sky / self._pixel_scale(band)))

    def cells_of_pixels(self, x, y, band):
        """
        Return the HEALPix cell of pixel positions.

        Parameters
        ----------
        x, y : array_like
            The pixel coordinates (0-based, pixel centres at integers).
        band : str
            The band whose WCS is used.

        Returns
        -------
        numpy.ndarray
            The nested HEALPix identifier of every position.
        """
        # A WCS object is not safe to call from several threads.
        with self._lock:
            lon, lat = self.wcs_dict[band].all_pix2world(np.atleast_1d(x), np.atleast_1d(y), 0)

        return self.healpix.lonlat_to_healpix(lon * u.deg, lat * u.deg)

    def _cells_on_image(self, band):
        """
        Find the cells that fall on the image of a band.

        The image is sampled on a lattice four times finer than the
        cells, which finds every cell that lies inside the image, and
        at every pixel of its border, which finds the cells that are
        only partly on the image, however small the part.

        Parameters
        ----------
        band : str
            The name of the band.

        Returns
        -------
        numpy.ndarray
            The sorted identifiers of the cells.
        """
        n_y, n_x = self._image_shape(band)

        step = max(self.cell_size_sky / self._pixel_scale(band) / 4.0, 1.0)

        xs = np.unique(np.append(np.arange(0, n_x - 1, step), n_x - 1))
        ys = np.unique(np.append(np.arange(0, n_y - 1, step), n_y - 1))

        grid_x, grid_y = np.meshgrid(xs, ys)

        border_x = np.arange(n_x)
        border_y = np.arange(n_y)

        x = np.concatenate([
            grid_x.ravel(), border_x, border_x,
            np.zeros(n_y), np.full(n_y, n_x - 1)
        ])
        y = np.concatenate([
            grid_y.ravel(), np.zeros(n_x), np.full(n_x, n_y - 1),
            border_y, border_y
        ])

        return np.unique(self.cells_of_pixels(x, y, band))

    def _cell_boxes(self, cell_ids, band):
        """
        Return the pixel bounding boxes of cells on the image of a band.

        The border of every cell is sampled with 16 points per side,
        converted to pixels, and enlarged by two pixels, so that the
        box contains all the pixels of the cell.

        Parameters
        ----------
        cell_ids : numpy.ndarray
            The identifiers of the cells.
        band : str
            The name of the band.

        Returns
        -------
        numpy.ndarray
            One row `(y0, y1, x0, x1)` per cell, clipped to the image,
            with exclusive upper limits. A cell outside the image has
            `y1 <= y0` or `x1 <= x0`.
        """
        lon, lat = self.healpix.boundaries_lonlat(cell_ids, step=16)

        x, y = self.wcs_dict[band].all_world2pix(lon.to_value(u.deg), lat.to_value(u.deg), 0)

        n_y, n_x = self._image_shape(band)

        y0 = np.clip(np.floor(y.min(axis=1)).astype(int) - 2, 0, n_y)
        y1 = np.clip(np.ceil(y.max(axis=1)).astype(int) + 3, 0, n_y)
        x0 = np.clip(np.floor(x.min(axis=1)).astype(int) - 2, 0, n_x)
        x1 = np.clip(np.ceil(x.max(axis=1)).astype(int) + 3, 0, n_x)

        return np.stack([y0, y1, x0, x1], axis=1)

    def make_tiles(
            self,
            coverage_mask_dict: Optional[Dict[str, npt.ArrayLike]] = None,
            min_coverage: float = 0.0,
        ) -> None:
        """
        Select the cells of the tiling.

        A cell is kept if it falls on the image of every band. If
        coverage masks are given, now or in a previous call, the cell
        must also have, in every band, a fraction of covered pixels in
        its bounding box larger than `min_coverage`.

        The results are stored in `cell_ids`, `centres` and
        `polygons`.

        Parameters
        ----------
        coverage_mask_dict
            For every band, a full-image array that is non-zero where
            the band has valid data. If None, the masks stored by a
            previous call, if any, are used.
        min_coverage
            The minimum covered fraction, between 0 and 1.
        """
        if coverage_mask_dict is not None:
            self.coverage_mask_dict = coverage_mask_dict
            self.min_coverage = min_coverage

        bands = self.bands

        cell_ids = self._cells_on_image(bands[0])

        for band in bands[1:]:
            cell_ids = np.intersect1d(cell_ids, self._cells_on_image(band))

        boxes = {band: self._cell_boxes(cell_ids, band) for band in bands}

        keep = np.ones(cell_ids.size, dtype=bool)

        for band in bands:
            box = boxes[band]
            keep &= (box[:, 1] > box[:, 0]) & (box[:, 3] > box[:, 2])

        if hasattr(self, 'coverage_mask_dict'):
            for band in bands:
                mask = self.coverage_mask_dict[band]

                for i in np.flatnonzero(keep):
                    y0, y1, x0, x1 = boxes[band][i]
                    covered = np.count_nonzero(mask[y0:y1, x0:x1]) / ((y1 - y0) * (x1 - x0))

                    if covered <= self.min_coverage:
                        keep[i] = False

        if not keep.any():
            raise ValueError("No HEALPix cell falls on the images of all the bands.")

        self.cell_ids = cell_ids[keep]
        self._boxes = {band: boxes[band][keep] for band in bands}

        self.centres = self.healpix.healpix_to_skycoord(self.cell_ids)

        lon, lat = self.healpix.boundaries_lonlat(self.cell_ids, step=1)

        self.polygons = [
            sh.Polygon(np.column_stack([lon_i.to_value(u.deg), lat_i.to_value(u.deg)]))
            for lon_i, lat_i in zip(lon, lat)
        ]

    def add_band(self, band: str, header: fits.Header):
        """
        Return a new tiling that also includes an additional band.

        The cells are selected again, so that they fall on the images
        of all the bands including the new one. The coverage masks
        given to `make_tiles`, if any, are applied again; a mask for
        the new band is not required. The current object is not
        modified.

        Parameters
        ----------
        band
            The name of the band to add.
        header
            The FITS header of the full image in that band.

        Returns
        -------
        HealpixTiling
            The new tiling, with `make_tiles` already called.
        """
        new_header_dict = dict(self.header_dict)
        new_header_dict[band] = header

        new_tiling = HealpixTiling(new_header_dict, self.order, self.margin_sky * u.deg)

        if hasattr(self, 'coverage_mask_dict'):
            masks = dict(self.coverage_mask_dict)
            masks.setdefault(band, np.ones(new_tiling._image_shape(band), dtype=bool))

            new_tiling.make_tiles(masks, self.min_coverage)

        else:
            new_tiling.make_tiles()

        return new_tiling

    def cell_box(self, index: int, band: str, margin_pixels: Optional[int] = None):
        """
        Return the pixel box of a tile on the image of a band.

        Parameters
        ----------
        index
            The position of the tile in `cell_ids`.
        band
            The name of the band.
        margin_pixels
            The margin added around the bounding box of the cell, in
            pixels. If None, the margin of the tiling is used.

        Returns
        -------
        inner : tuple of int
            The bounding box `(y0, y1, x0, x1)` of the cell, with
            exclusive upper limits.
        outer : tuple of int
            The same box enlarged by the margin and clipped to the
            image.
        """
        if self.cell_ids is None:
            raise ValueError("The tiles are not defined yet: call 'make_tiles' first.")

        if margin_pixels is None:
            margin_pixels = self.margin_pixels(band)

        y0, y1, x0, x1 = (int(v) for v in self._boxes[band][index])

        outer = _clipped_box(
            (y0 - margin_pixels, y1 + margin_pixels, x0 - margin_pixels, x1 + margin_pixels),
            self._image_shape(band)
        )

        return (y0, y1, x0, x1), outer

    def get_tile_data(
            self,
            index: int,
            Tile: BaseTile,
            input_dict: Dict[str, Dict[str, npt.ArrayLike]],
            margin_pixels: Optional[int] = None,
        ) -> BaseTile:
        """
        Extract the data of one tile from full-image arrays.

        For every band and every named array of `input_dict`, the
        bounding box of the cell plus the margin is cut out of the
        array, on its own pixel grid. The cutouts are views of the
        input arrays, not copies, and are stored in the tile together
        with the WCS of the cutout.

        The tile also receives the attributes `cell_id` (the HEALPix
        identifier), and `inner_box` and `outer_box`, which give for
        every band the box of the cell and that of the cutout in the
        full image as `(y0, y1, x0, x1)`.

        Parameters
        ----------
        index
            The position of the tile in `cell_ids`.
        Tile
            The class of the tile to create.
        input_dict
            The full-image arrays, as `input_dict[band][name]`.
        margin_pixels
            The margin of the cutout in pixels. If None, the margin of
            the tiling is used.

        Returns
        -------
        BaseTile
            The tile, an instance of `Tile`.
        """
        bands = list(input_dict.keys())

        tile = Tile(
            shape=(self.cell_size_sky, self.cell_size_sky),
            overlap=self.margin_sky,
            bands=bands,
            index=index
        )

        tile.cell_id = int(self.cell_ids[index])
        tile.inner_box = {}
        tile.outer_box = {}

        for band in bands:
            inner, outer = self.cell_box(index, band, margin_pixels)

            tile.inner_box[band] = inner
            tile.outer_box[band] = outer

            y0, y1, x0, x1 = outer

            wcs = self.wcs_dict[band][y0:y1, x0:x1]

            for name, data in input_dict[band].items():
                tile.add_data(band, name, data[y0:y1, x0:x1], wcs)

        return tile

    def tile_to_mosaic(
            self,
            tile: BaseTile,
            band: str,
            output_dict: Dict[str, Dict[str, npt.ArrayLike]],
        ) -> None:
        """
        Write the arrays of a tile into full-image arrays.

        For every key of `output_dict[band]`, the part of the tile
        array that corresponds to the bounding box of the cell is
        copied into the full-image array; the margin is discarded.
        The boxes of neighbouring cells overlap, so some pixels are
        written by more than one tile: this is harmless only if the
        tiles hold the same values there, as is the case for an array
        smoothed with a margin at least as wide as the kernel.

        Parameters
        ----------
        tile
            The tile, as returned by `get_tile_data`.
        band
            The band to write.
        output_dict
            The full-image arrays, as `output_dict[band][key]`. They
            are modified in place.
        """
        iy0, iy1, ix0, ix1 = tile.inner_box[band]
        oy0, _, ox0, _ = tile.outer_box[band]

        for key in output_dict[band].keys():
            small = tile.data2D[band][key]

            output_dict[band][key][iy0:iy1, ix0:ix1] = small[iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0]

    def plot_tiles(self, figsize: Tuple[int, int] = (10, 10)):
        """
        Plot the footprints of the bands and of the cells.

        Parameters
        ----------
        figsize
            The size of the figure in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : matplotlib.axes.Axes
            The axes, in (RA, Dec) degrees with RA increasing to the
            left.
        """
        import matplotlib.pyplot as plt
        from shapely.plotting import plot_polygon

        fig, ax = plt.subplots(figsize=figsize)

        for i, band in enumerate(self.bands):
            footprint = sh.Polygon(self.wcs_dict[band].calc_footprint())
            plot_polygon(footprint, ax=ax, add_points=False, facecolor='none', edgecolor=f'C{i}', lw=2, label=band)

        for polygon in self.polygons:
            plot_polygon(polygon, ax=ax, add_points=False, facecolor='none', edgecolor='k', lw=0.5)

        ax.invert_xaxis()
        ax.set_xlabel('RA [deg]')
        ax.set_ylabel('Dec [deg]')
        ax.legend()

        return fig, ax

    def save(self, filename: str) -> None:
        """
        Pickle the tiling to `<filename>.pkl`.

        Parameters
        ----------
        filename
            The path of the output file, without the extension.
        """
        with open(path.join(filename + '.pkl'), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        """
        Load a tiling saved with `save`.

        Parameters
        ----------
        filename
            The path of the file, without the `.pkl` extension.

        Returns
        -------
        HealpixTiling
            The tiling.
        """
        with open(path.join(filename + '.pkl'), 'rb') as f:
            return pickle.load(f)


def _segment_cutout(det_arr, box, detection_function):
    """
    Segment a cutout of the detection array.

    Parameters
    ----------
    det_arr : numpy.ndarray
        The full detection array.
    box : tuple of int
        The box `(y0, y1, x0, x1)` of the cutout.
    detection_function : callable
        The function that segments an array. It returns an integer
        label map, or an object with such a map in its attribute
        `data` (a photutils `SegmentationImage`).

    Returns
    -------
    cutout : numpy.ndarray
        The cutout of the detection array (a view).
    labels : numpy.ndarray
        The label map of the cutout, 0 for the background.
    """
    y0, y1, x0, x1 = box

    cutout = det_arr[y0:y1, x0:x1]

    segmap = detection_function(cutout)

    return cutout, np.asarray(getattr(segmap, 'data', segmap))


def _stored(det, segment, terrace):
    """
    Copy the cutout of the detection array that a source stores.

    The pixels outside the segment are set to zero: they belong to
    the background or to other sources and are not used. If
    `terrace` is given, the values inside the segment are replaced
    by their terrace levels (`attila.tiling.sources.terrace`), which
    a packed file reproduces exactly.

    Parameters
    ----------
    det : numpy.ndarray
        The cutout of the detection array on the box of the source.
    segment : numpy.ndarray
        The boolean footprint of the source in the box.
    terrace : tuple of float or None
        The threshold and the relative step of the terraces.

    Returns
    -------
    numpy.ndarray
        The cutout, as 32-bit floats.
    """
    stored = np.zeros(det.shape, dtype=np.float32)

    if terrace is None:
        stored[segment] = det[segment]
    else:
        stored[segment] = terrace_values(det[segment], *terrace)

    return stored


def _candidate(cutout, labels, label, obj_slices, box, det_terrace=None):
    """
    Describe one segment of a cutout in full-image coordinates.

    Parameters
    ----------
    cutout : numpy.ndarray
        The cutout of the detection array.
    labels : numpy.ndarray
        The label map of the cutout.
    label : int
        The label of the segment.
    obj_slices : tuple of slice
        The bounding box of the segment inside the cutout.
    box : tuple of int
        The box `(y0, y1, x0, x1)` of the cutout in the image.

    Returns
    -------
    dict
        The footprint (`segment`), the detection cutout (`det_arr`),
        the bounding box in the image (`bbox`, as
        `(iymin, iymax, ixmin, ixmax)`), the area in pixels and the
        position `(y, x)` of the peak in the image.
    """
    y0, _, x0, _ = box
    sl_y, sl_x = obj_slices

    segment = labels[obj_slices] == label
    det = cutout[obj_slices]

    # Peak of the detection array on the segment. `argmax` returns the
    # first maximum in raster order, which does not depend on the
    # cutout the segment is seen in.
    flat = np.argmax(np.where(segment, det, -np.inf))
    peak_y, peak_x = np.unravel_index(flat, segment.shape)

    return {
        'segment': segment,
        'det_arr': _stored(det, segment, det_terrace),
        'bbox': (y0 + sl_y.start, y0 + sl_y.stop, x0 + sl_x.start, x0 + sl_x.stop),
        'area': int(np.count_nonzero(segment)),
        'peak': (int(y0 + sl_y.start + peak_y), int(x0 + sl_x.start + peak_x)),
    }


# Side in pixels of the blocks of the coarse map that is used to find
# the extent of the sources cut by a cutout.
_BLOCK = 8


class LazyDetectionArray:
    """
    A detection array that is computed piece by piece, on request.

    The object stands for a full-size detection array without holding
    it in memory. Slicing it with two slices, `array[y0:y1, x0:x1]`,
    computes that box only: the function given at construction is
    called on the box enlarged by a margin, and the margin is trimmed
    from its result. With a margin at least as wide as the footprint
    of the operation (the half-size of a smoothing kernel) the values
    are the same as those of the array computed in one piece.

    Attributes
    ----------
    shape : tuple of int
        The shape `(n_y, n_x)` of the full array.
    margin : int
        The margin in pixels added around a box before computing it.
    """

    def __init__(self, shape: Tuple[int, int], function: Callable, margin: int) -> None:
        """
        Store the shape, the function and the margin.

        Parameters
        ----------
        shape
            The shape `(n_y, n_x)` of the full array.
        function
            The function that computes a piece of the array: it
            receives a box `(y0, y1, x0, x1)` in full-image pixels and
            returns the array of that box.
        margin
            The margin in pixels added around a requested box.
        """
        self.shape = tuple(shape)
        self.function = function
        self.margin = int(margin)

    @property
    def ndim(self):
        """The number of dimensions of the array."""
        return 2

    def __getitem__(self, key):
        """
        Compute a box of the array.

        Parameters
        ----------
        key : tuple of slice
            The two slices `(y, x)` of the box, with unit steps.

        Returns
        -------
        numpy.ndarray
            The array of the box.
        """
        sl_y, sl_x = key

        y0, y1, _ = sl_y.indices(self.shape[0])
        x0, x1, _ = sl_x.indices(self.shape[1])

        oy0, oy1, ox0, ox1 = _clipped_box(
            (y0 - self.margin, y1 + self.margin, x0 - self.margin, x1 + self.margin), self.shape
        )

        piece = self.function((oy0, oy1, ox0, ox1))

        return piece[y0 - oy0:y1 - oy0, x0 - ox0:x1 - ox0]

    def materialize(self, block: int = 4096) -> np.ndarray:
        """
        Compute the full array, block by block.

        Parameters
        ----------
        block
            The side in pixels of the blocks that are computed.

        Returns
        -------
        numpy.ndarray
            The full array, as 32-bit floats.
        """
        full = np.zeros(self.shape, dtype=np.float32)

        for y0 in range(0, self.shape[0], block):
            for x0 in range(0, self.shape[1], block):
                full[y0:y0 + block, x0:x0 + block] = self[y0:y0 + block, x0:x0 + block]

        return full


def _coarse_piece(cutout, box, shape, threshold):
    """
    Reduce a cutout to the blocks of the coarse map it contains.

    The coarse map divides the image into blocks of `_BLOCK` pixels
    and marks a block if any of its pixels is above the threshold.
    Only the blocks that lie entirely inside the cutout are returned.

    Parameters
    ----------
    cutout : numpy.ndarray
        The cutout of the detection array.
    box : tuple of int
        The box `(y0, y1, x0, x1)` of the cutout in the image.
    shape : tuple of int
        The shape `(n_y, n_x)` of the image.
    threshold : float
        The detection threshold.

    Returns
    -------
    tuple
        The first block along y and x, and the boolean array of the
        blocks.
    """
    y0, y1, x0, x1 = box

    def block_range(start, stop, size):
        first = -(-start // _BLOCK)
        # The last block of the image may be incomplete.
        last = -(-size // _BLOCK) if stop == size else stop // _BLOCK

        return first, max(last, first)

    by0, by1 = block_range(y0, y1, shape[0])
    bx0, bx1 = block_range(x0, x1, shape[1])

    above = cutout > threshold

    piece = np.zeros((by1 - by0, bx1 - bx0), dtype=bool)

    for i, block_y in enumerate(range(by0, by1)):
        rows = above[block_y * _BLOCK - y0:min((block_y + 1) * _BLOCK, y1) - y0]

        if not rows.any():
            continue

        columns = rows.any(axis=0)

        for j, block_x in enumerate(range(bx0, bx1)):
            piece[i, j] = columns[block_x * _BLOCK - x0:min((block_x + 1) * _BLOCK, x1) - x0].any()

    return by0, bx0, piece


def _to_sources(candidates, header):
    """
    Turn the segments found in a cutout into a `Sources` container.

    Parameters
    ----------
    candidates : list of dict
        The segments, as returned by `_candidate`, each with its
        `cell_id`.
    header : astropy.io.fits.Header
        The header of the mosaic.

    Returns
    -------
    Sources
        The sources, labelled from 1 in the order of `candidates`,
        with the attributes `cell_id` and `peak`.
    """
    sources_list = []

    for label, candidate in enumerate(candidates, start=1):
        iy0, iy1, ix0, ix1 = candidate['bbox']

        source = Source(
            segment=candidate['segment'],
            det_arr=candidate['det_arr'],
            label=label,
            area=candidate['area'],
            bbox=BoundingBox(ixmin=ix0, ixmax=ix1, iymin=iy0, iymax=iy1),
            previous_label=0,
        )

        source.cell_id = candidate['cell_id']
        source.peak = candidate['peak']

        sources_list.append(source)

    return Sources(sources_list, header)


def _detect_cell(det_arr, tiling, index, band, detection_function, threshold=None, header=None, measure=None,
                 det_terrace=None):
    """
    Detect, and optionally measure, the sources of one cell.

    The cutout of the cell is taken from the detection array (and
    computed at this point if the array is lazy) and segmented. A
    segment that does not reach the border of the cutout is complete:
    it is kept if its peak lies in the cell. A segment that reaches
    the border and overlaps the bounding box of the cell is reported
    as open, to be detected again on a larger box. The cutout is not
    kept: the sources hold copies of their own boxes only.

    Parameters
    ----------
    det_arr : numpy.ndarray or LazyDetectionArray
        The full detection array.
    tiling : HealpixTiling
        The tiling.
    index : int
        The position of the cell in `tiling.cell_ids`.
    band : str
        The band of the tiling the detection array belongs to.
    detection_function : callable
        The function that segments an array.
    threshold : float, optional
        If given, the blocks of the coarse map covered by the cutout
        are also returned.
    header : astropy.io.fits.Header, optional
        The header of the mosaic, stored in the returned container.
    measure : callable, optional
        A function applied to the sources of the cell; its result is
        returned with them.

    Returns
    -------
    sources : Sources
        The complete sources owned by the cell.
    measured : object
        The result of `measure`, or None.
    seeds : list of tuple
        For every open segment, the position `(y, x)` in the image of
        one of its pixels and the box of the cutout it was seen in.
    coarse : tuple or None
        The piece of the coarse map (see `_coarse_piece`).
    """
    inner, outer = tiling.cell_box(index, band, max(tiling.margin_pixels(band), _BLOCK))

    cutout, labels = _segment_cutout(det_arr, outer, detection_function)

    coarse = None if threshold is None else _coarse_piece(cutout, outer, det_arr.shape, threshold)

    iy0, iy1, ix0, ix1 = inner
    oy0, _, ox0, _ = outer

    candidates, seeds = [], []

    for label, obj_slices in enumerate(ndimage.find_objects(labels), start=1):
        if obj_slices is None:
            continue

        sl_y, sl_x = obj_slices

        # Skip the segments that do not overlap the box of the cell.
        if (oy0 + sl_y.stop <= iy0 or oy0 + sl_y.start >= iy1
                or ox0 + sl_x.stop <= ix0 or ox0 + sl_x.start >= ix1):
            continue

        if _touches_border(obj_slices, outer, det_arr.shape):
            segment = labels[obj_slices] == label
            seed_y, seed_x = np.argwhere(segment)[0]

            seeds.append(((int(oy0 + sl_y.start + seed_y), int(ox0 + sl_x.start + seed_x)), outer))

        else:
            candidates.append(_candidate(cutout, labels, label, obj_slices, outer, det_terrace))

    owned = []

    if candidates:
        peaks = np.array([candidate['peak'] for candidate in candidates])
        cells = tiling.cells_of_pixels(peaks[:, 1], peaks[:, 0], band)

        cell_id = int(tiling.cell_ids[index])

        for candidate, cell in zip(candidates, cells):
            if cell == cell_id:
                candidate['cell_id'] = cell_id
                owned.append(candidate)

        owned.sort(key=lambda candidate: candidate['peak'])

    sources = _to_sources(owned, header)

    measured = measure(sources) if (measure is not None and len(sources) > 0) else None

    return sources, measured, seeds, coarse


def _resolve_open(det_arr, seeds, box, detection_function, det_terrace=None):
    """
    Detect again the sources that were cut by the border of a cutout.

    The box is segmented; every seed pixel selects a segment. If one
    of the selected segments still reaches a border of the box, the
    box is doubled in size around its centre and segmented again,
    until all of them are complete or the box is the whole image.

    Parameters
    ----------
    det_arr : numpy.ndarray or LazyDetectionArray
        The full detection array.
    seeds : list of tuple of int
        The positions `(y, x)` in the image of one pixel of each
        source to detect.
    box : tuple of int
        The first box `(y0, y1, x0, x1)` to segment. It must contain
        all the seeds.
    detection_function : callable
        The function that segments an array.

    Returns
    -------
    list of dict
        The complete segments (see `_candidate`), without duplicates.
        A seed that is not in a segment of the box gives nothing.
    """
    shape = det_arr.shape
    box = _clipped_box(box, shape)

    while True:
        cutout, labels = _segment_cutout(det_arr, box, detection_function)

        wanted = sorted({int(labels[y - box[0], x - box[2]]) for y, x in seeds} - {0})

        objects = ndimage.find_objects(labels)

        whole_image = box == (0, shape[0], 0, shape[1])

        if whole_image or not any(_touches_border(objects[label - 1], box, shape) for label in wanted):
            return [_candidate(cutout, labels, label, objects[label - 1], box, det_terrace) for label in wanted]

        y0, y1, x0, x1 = box
        half_y, half_x = (y1 - y0 + 1) // 2, (x1 - x0 + 1) // 2

        box = _clipped_box((y0 - half_y, y1 + half_y, x0 - half_x, x1 + half_x), shape)


# The arguments of the worker processes of `detect_in_cells`, which
# inherit them when forked.
_SHARED_DETECTION = None


def _detect_cell_shared(index):
    """Run `_detect_cell` on the shared arguments, in a worker process."""
    return _detect_cell(index=index, **_SHARED_DETECTION)


def _resolve_open_shared(task):
    """Run `_resolve_open` on the shared arguments, in a worker process."""
    return _resolve_open(_SHARED_DETECTION['det_arr'], task[0], task[1], _SHARED_DETECTION['detection_function'],
                         _SHARED_DETECTION['det_terrace'])


def _iter_tasks(function, shared_function, tasks, n_workers, backend, progress):
    """
    Apply a function to a list of tasks and yield the results in order.

    The results are produced one at a time, so that each can be used
    and released before the next one is taken. In parallel at most
    `2 * n_workers` tasks are running or waiting to be taken.

    Parameters
    ----------
    function : callable
        The function applied in this process (serial and threads).
    shared_function : callable
        The module-level counterpart applied by the worker processes,
        which reads its other arguments from `_SHARED_DETECTION`.
    tasks : list
        The tasks.
    n_workers : int
        The number of workers.
    backend : str
        `'threads'` or `'processes'`.
    progress : bool
        If `True` a progress bar is shown.

    Yields
    ------
    object
        The result of every task, in the order of the tasks.
    """
    bar = tqdm(total=len(tasks), disable=not progress)

    if n_workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            yield function(task)
            bar.update()

    else:
        if backend == 'processes':
            pool = multiprocessing.get_context('fork').Pool(n_workers)
            submit = partial(pool.apply_async, shared_function)
            result_of = lambda future: future.get()
        else:
            pool = ThreadPoolExecutor(max_workers=n_workers)
            submit = partial(pool.submit, function)
            result_of = lambda future: future.result()

        try:
            pending = deque()
            tasks = iter(tasks)

            while True:
                # Keep a bounded number of tasks in flight, so that
                # finished results do not pile up in memory.
                while len(pending) < 2 * n_workers:
                    try:
                        task = next(tasks)
                    except StopIteration:
                        break

                    pending.append(submit((task,)) if backend == 'processes' else submit(task))

                if not pending:
                    break

                yield result_of(pending.popleft())
                bar.update()

        finally:
            if backend == 'processes':
                pool.terminate()
            else:
                pool.shutdown(wait=True)

    bar.close()


def detect_in_cells(
        det_arr,
        tiling: HealpixTiling,
        band: str,
        detection_function: Callable,
        header: Optional[fits.Header] = None,
        n_workers: int = 1,
        progress: bool = True,
        threshold: Optional[float] = None,
        backend: str = 'threads',
        measure: Optional[Callable] = None,
        det_step: Optional[float] = None,
        sink: Optional[Callable] = None,
    ) -> Sources:
    """
    Detect the sources of a field cell by cell, without stitching.

    Every cell of the tiling is segmented independently on its cutout
    (the bounding box of the cell plus the margin), and every source
    is assigned to one cell:

    1. A segment that does not reach the border of the cutout is
       complete. It is kept by the cell that contains its peak, the
       pixel with the highest value of the detection array, and
       ignored by the other cells that see it.
    2. A segment that reaches the border of a cutout is not complete
       there. If it overlaps the box of the cell it is detected
       again on a box that contains the whole segment, and then
       assigned to the cell of its peak.

    The box of step 2 is found in one step if `threshold` is given:
    a coarse map of the pixels above the threshold, 64 times smaller
    than the image, is assembled from the cutouts of the cells, and
    all the cut segments of a region of that map are detected
    together on the box of the region. Without `threshold`, or if a
    segment is still cut, the box is doubled until the segment fits.

    The result does not depend on the order or on the margin of the
    tiling, nor on the number of workers, and equals the segmentation
    of the whole array at once, provided that `detection_function`
    gives a complete segment the same footprint whatever the cutout
    it is seen in.

    Memory. Only the cutouts being segmented are in memory, one per
    worker, if `det_arr` is a `LazyDetectionArray`. If `sink` is
    given the sources are also handed over cell by cell and not
    kept, so that the memory in use does not grow with the size of
    the field.

    The sources are labelled from 1: first those of the cells, in the
    order of the cells and, within a cell, of the position of their
    peak; then those that were cut by a cutout, sorted by cell and
    peak. The cell of every source is stored in its attribute
    `cell_id`, and the peak in `peak`.

    Parameters
    ----------
    det_arr : numpy.ndarray or LazyDetectionArray
        The full detection array, shape `(n_y, n_x)`.
    tiling
        The tiling, with the tiles already made.
    band
        The band of the tiling whose WCS describes `det_arr`.
    detection_function
        The function that segments an array: it receives a 2D array
        and returns a label map or a photutils `SegmentationImage`.
    header
        The FITS header stored in the returned container. If None,
        the header of `band` in the tiling is used.
    n_workers
        The number of workers that process the cells and the cut
        segments.
    progress
        If `True` a progress bar is shown.
    threshold
        The threshold used by `detection_function`, in the units of
        `det_arr`. It is only used to find the extent of the cut
        segments; a value lower than the real one is safe.
    backend
        `'threads'` runs the workers as threads of this process;
        `'processes'` runs them as forked processes, which do not
        share the Python interpreter lock, at the cost of sending
        the sources back to this process. Where forking is not
        available threads are used.
    measure
        A function applied by the workers to the sources of a cell,
        a `Sources` container with provisional labels. Its result is
        passed to `sink`.
    sink
        A function `sink(sources, measured)` called in this process
        with the sources of one cell at a time, with their final
        labels, and the result of `measure`. If given, the sources
        are not kept and an empty container is returned.

    det_step
        If given, the cutouts of the detection array stored in the
        sources are terraced: the values are replaced by levels
        that differ by this relative step, starting at `threshold`
        (`attila.tiling.sources.terrace_values`). Terraced cutouts
        compress well and are saved exactly by `write_pack`. It
        needs `threshold`. The detection itself always uses
        `det_arr` as it is.

    Returns
    -------
    Sources
        The detected sources, or an empty container if `sink` is
        given.
    """
    global _SHARED_DETECTION

    if not isinstance(det_arr, LazyDetectionArray):
        det_arr = np.asarray(det_arr)

    if backend not in ('threads', 'processes'):
        raise ValueError("backend must be either 'threads' or 'processes'")

    if backend == 'processes' and 'fork' not in multiprocessing.get_all_start_methods():
        backend = 'threads'

    if header is None:
        header = tiling.header_dict[band]

    if det_step is not None and threshold is None:
        raise ValueError("det_step needs the threshold of the detection")

    det_terrace = None if det_step is None else (float(threshold), float(det_step))

    shared = dict(det_arr=det_arr, tiling=tiling, band=band, detection_function=detection_function,
                  threshold=threshold, header=header, measure=measure, det_terrace=det_terrace)

    def run(index):
        return _detect_cell(index=index, **shared)

    def resolve(task):
        return _resolve_open(det_arr, task[0], task[1], detection_function, det_terrace)

    kept = []
    peaks_found = set()
    next_label = 1

    def deliver(sources, measured):
        """Give the final labels to the sources and pass them on."""
        nonlocal next_label

        for source in sources:
            source.label = next_label
            next_label += 1

            peaks_found.add(source.peak)

        if sink is not None:
            sink(sources, measured)
        else:
            kept.extend(sources.sources)

    n_y, n_x = det_arr.shape
    coarse = np.zeros((-(-n_y // _BLOCK), -(-n_x // _BLOCK)), dtype=bool) if threshold is not None else None

    seeds = []

    _SHARED_DETECTION = shared

    try:
        results = _iter_tasks(run, _detect_cell_shared, list(range(len(tiling.cell_ids))), n_workers, backend, progress)

        for sources, measured, cell_seeds, piece in results:
            seeds += cell_seeds

            if piece is not None:
                by0, bx0, blocks = piece
                coarse[by0:by0 + blocks.shape[0], bx0:bx0 + blocks.shape[1]] |= blocks

            if len(sources) > 0:
                deliver(sources, measured)

        # Segments cut by a cutout: group them, so that each group is
        # detected again once.
        tasks = []

        if seeds and threshold is not None:
            coarse_labels, _ = ndimage.label(coarse, structure=np.ones((3, 3)))
            coarse_boxes = ndimage.find_objects(coarse_labels)

            groups = {}

            for seed, box in seeds:
                groups.setdefault(int(coarse_labels[seed[0] // _BLOCK, seed[1] // _BLOCK]), []).append((seed, box))

            for label, members in groups.items():
                if label == 0:
                    # Not above the threshold given: fall back on the
                    # cutouts the segments were seen in.
                    tasks += [([seed], box) for seed, box in members]
                    continue

                sl_y, sl_x = coarse_boxes[label - 1]

                # One more block on each side: the closing of the
                # segments and the border test need a free border.
                box = ((sl_y.start - 1) * _BLOCK, (sl_y.stop + 1) * _BLOCK,
                       (sl_x.start - 1) * _BLOCK, (sl_x.stop + 1) * _BLOCK)

                tasks.append(([seed for seed, _ in members], box))

        else:
            tasks = [([seed], box) for seed, box in seeds]

        # Smallest boxes first: their order fixes the labels, and the
        # large ones then run with fewer of the others in memory.
        tasks.sort(key=lambda task: ((task[1][1] - task[1][0]) * (task[1][3] - task[1][2]), task[1]))

        valid_cells = set(int(cell) for cell in tiling.cell_ids)

        late = {}

        for group in _iter_tasks(resolve, _resolve_open_shared, tasks, n_workers, backend, False):
            candidates = [candidate for candidate in group
                          if candidate['peak'] not in peaks_found and candidate['peak'] not in late]

            if not candidates:
                continue

            peaks = np.array([candidate['peak'] for candidate in candidates])
            cells = tiling.cells_of_pixels(peaks[:, 1], peaks[:, 0], band)

            for candidate, cell in zip(candidates, cells):
                # A source whose peak is in a cell that is not part
                # of the tiling is dropped, as the complete ones are.
                if int(cell) in valid_cells:
                    candidate['cell_id'] = int(cell)
                    late[candidate['peak']] = candidate

    finally:
        _SHARED_DETECTION = None

    # The sources that were cut, one at a time: they can be large.
    for candidate in sorted(late.values(), key=lambda candidate: (candidate['cell_id'], candidate['peak'])):
        sources = _to_sources([candidate], header)

        measured = measure(sources) if measure is not None else None

        deliver(sources, measured)

    return Sources(kept, header)
