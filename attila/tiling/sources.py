"""
Per-source cutouts of the detection and deblending segmentation.

After the detection step the full-mosaic segmentation map is split
into one `Source` per label: the bounding box of the source in
mosaic pixels, its boolean footprint inside the box and the cutout
of the S/N detection array. `Sources` is the collection of all the
sources of a field; it builds them from a photutils
`SegmentationImage` or from the deblending results, finds the
neighbours of a source, extracts the image data of every source and
saves / loads the collection in a light on-disk format. The later
steps (extinction, properties, deblending, 2D fits) work on these
objects.

The on-disk layout is flat: the files of a source are
`<root>/sources/source_<previous_label>_<label>.segmap.npy`,
`...det_arr.npy` and `...metadata.json`, all in the same folder.
Collections written by earlier versions, with one folder per source
(`source_<previous_label>_<label>/segmap.npy` and so on), can still be
read by `Source.load_light`, `Sources.load_light` and
`Sources.predict`. `compress` and `decompress` pack the many small
files of a folder with zstd and a trained dictionary.
"""
from photutils.segmentation import SegmentationImage
import numpy as np
from numpy.typing import ArrayLike

from attila.field.field import Image, MultibandImage
# from attila.packing.packing import BoundingBox_to_Polygon, Polygon_to_BoundingBox

from photutils.aperture import BoundingBox

from astropy import units as u

from tqdm import tqdm

import io
import os
from os import path

import json

from typing import Union, List, Optional

import pandas as pd

import shapely as sh

from astropy.wcs import WCS
from astropy.io import fits

from astropy.coordinates import SkyCoord

import warnings

def terrace_codes(values, threshold, step):
    """
    Return the terrace level of the values of a detection array.

    The levels are `threshold * (1 + step)**k`, so that the contrast
    between two consecutive levels is the same everywhere. Level 0
    holds the values up to the threshold (and NaN); level `k >= 1`
    holds the values between `threshold * (1 + step)**(k - 1)` and
    `threshold * (1 + step)**k`.

    Parameters
    ----------
    values : numpy.ndarray
        The values.
    threshold : float
        The detection threshold, which is the edge of the first
        level: a value is above the threshold if and only if its
        level is larger than zero.
    step : float
        The relative step between two levels, for example 0.001.

    Returns
    -------
    numpy.ndarray
        The levels, as unsigned 16-bit integers.
    """
    values = np.asarray(values, dtype=np.float64)

    codes = np.zeros(values.shape, dtype=np.uint16)

    above = values > threshold

    levels = 1 + np.floor(np.log(values[above] / threshold) / np.log1p(step))

    codes[above] = np.clip(levels, 1, np.iinfo(np.uint16).max).astype(np.uint16)

    return codes


def terrace_decode(codes, threshold, step):
    """
    Return the values of terrace levels.

    Level 0 gives the threshold and level `k >= 1` the geometric
    centre of its interval, `threshold * (1 + step)**(k - 0.5)`.
    The function is the inverse of `terrace_codes` on the values it
    returns, so terraced values survive any number of round trips.

    Parameters
    ----------
    codes : numpy.ndarray
        The levels.
    threshold : float
        The detection threshold.
    step : float
        The relative step between two levels.

    Returns
    -------
    numpy.ndarray
        The values, as 32-bit floats.
    """
    codes = np.asarray(codes, dtype=np.float64)

    values = threshold * np.exp((codes - 0.5) * np.log1p(step))

    return np.where(codes > 0, values, threshold).astype(np.float32)


def terrace_values(values, threshold, step):
    """
    Replace the values of a detection array by their terrace levels.

    See `terrace_codes` and `terrace_decode`.

    Parameters
    ----------
    values : numpy.ndarray
        The values.
    threshold : float
        The detection threshold.
    step : float
        The relative step between two levels.

    Returns
    -------
    numpy.ndarray
        The terraced values, as 32-bit floats.
    """
    return terrace_decode(terrace_codes(values, threshold, step), threshold, step)


# Columns of the table of integers stored in a packed file.
_PACK_COLUMNS = ('label', 'previous_label', 'area', 'iymin', 'iymax', 'ixmin', 'ixmax', 'cell_id', 'peak_y', 'peak_x')


def write_pack(filename, sources, threshold=None, step=None, level=9):
    """
    Save a group of sources, for example those of a cell, in one file.

    The file holds, for all the sources together: a table of
    integers (label, parent label, area, bounding box and, if the
    sources have them, HEALPix cell and peak), the footprints as
    packed bits, and the values of the detection array inside the
    footprints. The pixels of a box outside the footprint are not
    stored and are read back as zero.

    If `threshold` and `step` are given the values are stored as
    terrace levels (`terrace_codes`), which is exact for cutouts
    that were terraced with the same parameters and much smaller;
    otherwise they are stored as 32-bit floats.

    The arrays are written in the `.npz` format and compressed with
    zstd (`<filename>.npz.zst`). Without the `zstandard` package the
    deflate compression of numpy is used instead (`<filename>.npz`).

    Parameters
    ----------
    filename : str
        The path of the file, without extension.
    sources : iterable of Source
        The sources. Their detection cutouts must be present.
    threshold : float, optional
        The detection threshold of the terraces.
    step : float, optional
        The relative step of the terraces.
    level : int
        The zstd compression level.

    Returns
    -------
    str
        The path of the file that was written.
    """
    sources = list(sources)

    meta = np.array([[
        source.label, source.previous_label, source.area,
        source.bbox.iymin, source.bbox.iymax, source.bbox.ixmin, source.bbox.ixmax,
        getattr(source, 'cell_id', -1),
        *getattr(source, 'peak', (-1, -1))
    ] for source in sources], dtype=np.int64).reshape(len(sources), len(_PACK_COLUMNS))

    segments = [np.asarray(source.segment, dtype=bool) for source in sources]

    inside = [np.asarray(source.det_arr)[segment] for source, segment in zip(sources, segments)]
    inside = np.concatenate(inside) if inside else np.zeros(0, dtype=np.float32)

    terraced = threshold is not None and step is not None

    arrays = {
        'meta': meta,
        'mask': np.packbits(np.concatenate([segment.ravel() for segment in segments]) if segments else np.zeros(0, bool)),
        'terrace': np.array([threshold, step] if terraced else [np.nan, np.nan], dtype=np.float64),
        'det': terrace_codes(inside, threshold, step) if terraced else inside.astype(np.float32),
    }

    try:
        import zstandard
    except ImportError:
        np.savez_compressed(filename + '.npz', **arrays)

        return filename + '.npz'

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)

    with open(filename + '.npz.zst', 'wb') as handle:
        handle.write(zstandard.ZstdCompressor(level=level).compress(buffer.getvalue()))

    return filename + '.npz.zst'


def read_pack(filename, keep_det_arr=True):
    """
    Read the sources saved by `write_pack`.

    Parameters
    ----------
    filename : str
        The path of the file, with its extension.
    keep_det_arr : bool
        If `False` the detection cutouts are not rebuilt and the
        footprints are used in their place, as `Source.load_light`
        does.

    Returns
    -------
    list of Source
        The sources, in the order they were saved, with the
        attributes `cell_id` and `peak` if they were stored.
    """
    if filename.endswith('.zst'):
        import zstandard

        with open(filename, 'rb') as handle:
            arrays = np.load(io.BytesIO(zstandard.ZstdDecompressor().decompress(handle.read())))
    else:
        arrays = np.load(filename)

    meta, det = arrays['meta'], arrays['det']
    threshold, step = arrays['terrace']

    shapes = [(row[4] - row[3], row[6] - row[5]) for row in meta]
    sizes = np.array([shape[0] * shape[1] for shape in shapes], dtype=np.int64)

    mask = np.unpackbits(arrays['mask'], count=int(sizes.sum())).astype(bool)

    if keep_det_arr and not np.isnan(step):
        det = terrace_decode(det, threshold, step)

    sources, pixel, value = [], 0, 0

    for row, shape, size in zip(meta, shapes, sizes):
        values = dict(zip(_PACK_COLUMNS, (int(item) for item in row)))

        segment = mask[pixel:pixel + size].reshape(shape)
        pixel += size

        if keep_det_arr:
            det_arr = np.zeros(shape, dtype=np.float32)
            det_arr[segment] = det[value:value + values['area']]
        else:
            det_arr = segment.copy()

        value += values['area']

        source = Source(
            segment = segment,
            det_arr = det_arr,
            label = values['label'],
            area = values['area'],
            bbox = BoundingBox(ixmin=values['ixmin'], ixmax=values['ixmax'], iymin=values['iymin'], iymax=values['iymax']),
            previous_label = values['previous_label']
        )

        if values['cell_id'] >= 0:
            source.cell_id = values['cell_id']
            source.peak = (values['peak_y'], values['peak_x'])

        sources.append(source)

    return sources


def Polygon_to_BoundingBox(polygon):
    """
    Convert a shapely polygon to a photutils `BoundingBox`.

    Parameters
    ----------
    polygon : shapely.Polygon
        The polygon, in pixel coordinates `(x, y)`.

    Returns
    -------
    photutils.aperture.BoundingBox
        The box given by the bounds of the polygon, truncated to
        integers.
    """
    minx, miny, maxx, maxy = polygon.bounds
    return BoundingBox(ixmin=int(minx), ixmax=int(maxx), iymin=int(miny), iymax=int(maxy))

def BoundingBox_to_Polygon(boundingbox, padding=0, shape=None):
    """
    Convert a photutils `BoundingBox` to a shapely rectangle.

    The box can be enlarged by a padding on all sides. The lower
    bounds are clipped at 0; if `shape` is given, the upper x bound is
    clipped at `shape[1]` and the upper y bound at `shape[0]`.

    Parameters
    ----------
    boundingbox : photutils.aperture.BoundingBox
        The box, in pixel coordinates.
    padding : int
        The number of pixels added on every side.
    shape : tuple of int, optional
        The shape `(ny, nx)` of the array the box refers to. If None
        the upper bounds are not clipped.

    Returns
    -------
    shapely.Polygon
        The rectangle with corners `(ixmin, iymin)` and
        `(ixmax, iymax)` after padding and clipping.
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

def _get_bbox_image(image, bbox, wcs):
    """
    Cut an image on the bounding box of a source.

    The centre of the box is converted to sky coordinates with `wcs`
    and the box sides are converted to arcsec with the mean pixel scale
    of `wcs`; the cutout is then delegated to the `cutout` method of
    the image.

    Parameters
    ----------
    image : Image or MultibandImage
        The full image to cut.
    bbox : photutils.aperture.BoundingBox
        The box of the source in full-mosaic pixel coordinates.
    wcs : astropy.wcs.WCS
        The WCS of the mosaic the box refers to.

    Returns
    -------
    Image or MultibandImage
        The cutout, of the same type as `image`.
    """

    position = wcs.pixel_to_world(float(bbox.center[1]), float(bbox.center[0]))
    # Compute the size of the region in arcsec. `bbox.center` above
    # is in (y, x) order.

    pixel_scale = np.mean([_ps.to(u.arcsec).value for _ps in wcs.proj_plane_pixel_scales()])
    size_x = float(bbox.ixmax - bbox.ixmin) * pixel_scale  # Size in arcsec along the x axis.
    size_y = float(bbox.iymax - bbox.iymin) * pixel_scale  # Size in arcsec along the y axis.

    return image.cutout(position=position, size=(size_y * u.arcsec, size_x * u.arcsec))

def _same_grid(input, header, wcs):
    """
    Tell whether an image has the pixel grid of a mosaic.

    Parameters
    ----------
    input : object
        The input of `get_data`. Anything that is not an `Image` or a
        `MultibandImage` gives `False`.
    header : astropy.io.fits.Header or None
        The header of the mosaic.
    wcs : astropy.wcs.WCS
        The WCS of the mosaic.

    Returns
    -------
    bool
        `True` if every band has the shape given by the header and a
        WCS equal to `wcs`.
    """
    if isinstance(input, Image):
        images = [input]
    elif isinstance(input, MultibandImage):
        images = list(input)
    else:
        return False

    if header is None or 'NAXIS1' not in header:
        return False

    shape = (int(header['NAXIS2']), int(header['NAXIS1']))

    try:
        return all(
            image.data.shape == shape and image.wcs.wcs.compare(wcs.wcs, tolerance=1e-10)
            for image in images
        )
    except Exception:
        return False

def _get_bbox_arr(arr, bbox):
        """
        Cut an array on a bounding box along its last two axes.

        Parameters
        ----------
        arr : numpy.ndarray
            The array to cut, with 2, 3 or 4 dimensions and `(y, x)` as
            the last two axes.
        bbox : photutils.aperture.BoundingBox
            The box in the pixel coordinates of `arr`.

        Returns
        -------
        numpy.ndarray
            A view of `arr` restricted to the box; the leading axes are
            kept. For any other number of dimensions None is returned.
        """
        if arr.ndim == 2:
            return arr[bbox.iymin:bbox.iymax, bbox.ixmin:bbox.ixmax]
        elif arr.ndim == 3:
            return arr[:, bbox.iymin:bbox.iymax, bbox.ixmin:bbox.ixmax]
        elif arr.ndim == 4:
            return arr[:, :, bbox.iymin:bbox.iymax, bbox.ixmin:bbox.ixmax]

class Source:
    """
    A detected source: its footprint and detection-array cutout.

    Attributes
    ----------
    segment : numpy.ndarray
        The boolean footprint of the source inside its bounding box,
        with shape `(ny, nx)` of the box. For the combined sources
        built by `combine_sources` it is instead a label map.
    det_arr : numpy.ndarray or None
        The cutout of the S/N detection array on the bounding box.
    label : int
        The label of the source after the current step.
    previous_label : int
        The label of the parent source from the detection step, 0 if
        the source does not come from a deblending.
    area : int
        The number of pixels of the footprint.
    bbox : photutils.aperture.BoundingBox
        The bounding box in full-mosaic pixel coordinates.
    data_types : list of str
        The kinds of data extracted with `get_data` so far (`"array"`
        and / or `"image"`).
    data : numpy.ndarray, Image or MultibandImage
        The data cutout, set by `get_data`.
    mask : numpy.ndarray
        The boolean mask of the neighbours to exclude, only set on the
        combined sources returned by `Sources.sources_from_radius` and
        `Sources.sources_from_centre`.
    deblend : photutils.segmentation.SegmentationImage
        The deblended segmentation of the source cutout, set by the
        pipeline and read by `Sources.from_deblend`.
    """
    def __init__(self, 
                segment: ArrayLike,
                det_arr: ArrayLike,
                label: int,
                area: int,
                bbox: BoundingBox,
                previous_label=None) -> None:
        """
        Store the footprint, the detection cutout and the labels.

        Parameters
        ----------
        segment
            The boolean footprint of the source inside `bbox`, with axes
            `(y, x)`.
        det_arr
            The cutout of the S/N detection array on `bbox`.
        label
            The label of the source after the current step.
        area
            The number of pixels of the footprint.
        bbox
            The bounding box in full-mosaic pixel coordinates.
        previous_label : int, optional
            The label of the parent source from the detection step. If
            None it is set to 0 (not deblended).
        """
    
        self.segment = segment
        self.det_arr = det_arr
        self.label = int(label)
        self.area = int(area)
        self.bbox = bbox
        self.data_types = []
        if previous_label is None:
            self.previous_label=0
        else:
            self.previous_label=int(previous_label)


    @property
    def centre(self):
        """
        The centre `(x, y)` of the bounding box in mosaic pixels.

        It is the mid-point between the `min` and `max` edges of `bbox`
        along each axis.
        """
        return (self.bbox.ixmin + self.bbox.ixmax) / 2, (self.bbox.iymin + self.bbox.iymax) / 2

    def get_data(self, input: Union[ArrayLike, Image, MultibandImage], wcs=None, same_grid=False) -> None:
        """
        Extract the data of the source from a full-mosaic input.

        A numpy array is sliced on the bounding box (last two axes) and
        `"array"` is recorded in `data_types`. An `Image` or a
        `MultibandImage` is cut around the centre of the box through its
        `cutout` method, or sliced on the box with `cutout_pixels` if
        `same_grid` is `True`, and `"image"` is recorded. The result is stored
        in the attribute `data`, replacing any previous value. Inputs of
        other types are ignored. Warnings raised during the extraction
        are silenced.

        Parameters
        ----------
        input
            The full-mosaic array or image.
        wcs : astropy.wcs.WCS, optional
            The WCS of the mosaic, needed for `Image` and
            `MultibandImage` inputs unless `same_grid` is `True`.
        same_grid : bool
            `True` if the input image has the pixel grid the bounding
            box refers to. The box is then sliced directly, which is
            much faster and gives the same pixels.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if isinstance(input, np.ndarray):
                self.data = _get_bbox_arr(input, self.bbox)

                if "array" not in self.data_types:
                    self.data_types.append("array")            

            elif (isinstance(input, Image)) | (isinstance(input, MultibandImage)):
                if same_grid:
                    self.data = input.cutout_pixels(self.bbox.iymin, self.bbox.iymax, self.bbox.ixmin, self.bbox.ixmax)
                else:
                    self.data = _get_bbox_image(input, self.bbox, wcs=wcs)

                if "image" not in self.data_types:
                    self.data_types.append("image")

    def save_light(self, filename: str, keep_det_arr: bool=True) -> None:
        """
        Save the footprint, the detection cutout and the metadata.

        Three flat files are written, using `filename` as a common
        prefix: `<filename>.segmap.npy` (the footprint as unsigned 8-bit
        integers), `<filename>.det_arr.npy` (the detection cutout as
        32-bit floats, if `keep_det_arr`) and `<filename>.metadata.json`
        (label, area, the four edges of the bounding box and the
        previous label). The image data extracted with `get_data` are
        not saved.

        Parameters
        ----------
        filename
            The path prefix of the files of the source, without
            extension.
        keep_det_arr
            Whether the detection-array cutout is saved too.
        """

        # Previous layout, one folder per source:
        # if not path.exists(filename):
        #     os.mkdir(filename)

        #np.save(path.join(filename, f"segmap.npy"), self.segment.astype(np.uint8))

        np.save(filename + ".segmap.npy", self.segment.astype(np.uint8))

        if keep_det_arr:
            #np.save(path.join(filename, f"det_arr.npy"), self.det_arr.astype(np.float32))
            np.save(filename + ".det_arr.npy", self.det_arr.astype(np.float32))


        metadata = {
            "label": self.label, 
            "area": self.area, 
            "ixmin": self.bbox.ixmin, 
            "ixmax": self.bbox.ixmax, 
            "iymin": self.bbox.iymin, 
            "iymax": self.bbox.iymax, 
            "previous_label": self.previous_label, 
            }
        
        #with open(path.join(filename, "metadata.json"), "w+") as handle:
        with open(filename + ".metadata.json", "w+") as handle:
            json.dump(metadata, handle, indent=4)

        
    # def save_full(self, filename: str) -> None:

    #     self.save_light(filename)

    #     if "array" in self.data_types:
    #         np.save(path.join(filename, f"data.npy"), self.data)

    #     if "image" in self.data_types:
    #          with open(path.join(filename, "image") + '.pkl', 'wb') as file:
    #             pickle.dump(self.data, file)

    @classmethod
    def load_light(cls, filename: str, keep_det_arr: bool) -> "Source":
        """
        Load a source saved with `save_light`.

        The footprint is read from `<filename>.segmap.npy` as a boolean
        array. If `keep_det_arr` is True, `det_arr` is read from
        `<filename>.det_arr.npy` as 32-bit floats; otherwise the
        boolean footprint is used in its place. If `filename` is a
        folder, the files `segmap.npy`, `det_arr.npy` and
        `metadata.json` inside it are read instead (the layout of
        earlier versions).

        Parameters
        ----------
        filename
            The path prefix of the files of the source, without
            extension, or the folder of the source.
        keep_det_arr
            Whether the `.det_arr.npy` file is read.

        Returns
        -------
        Source
            The loaded source.
        """

        # One folder per source in the old layout, a common file prefix
        # in the current one.
        sep = "/" if path.isdir(filename) else "."

        with open(filename + sep + "metadata.json") as handle:
            metadata = json.load(handle)

        if keep_det_arr:
            det_arr = np.load(filename + sep + "det_arr.npy").astype(np.float32)
        else:
            det_arr = np.load(filename + sep + "segmap.npy").astype(bool)

        return cls(
            segment = np.load(filename + sep + "segmap.npy").astype(bool),
            det_arr = det_arr,
            label = metadata["label"],
            area = metadata["area"],
            bbox = BoundingBox(
                ixmin=metadata["ixmin"],
                ixmax=metadata["ixmax"],
                iymin=metadata["iymin"],
                iymax=metadata["iymax"],
            ),
            previous_label = metadata["previous_label"]
        )
    
    # @classmethod
    # def load_full(cls, filename: str) -> None:
            
    #         source = cls.load_light(filename)
    
    #         if path.exists(path.join(filename, "data.npy")):
    #             source.data = np.load(path.join(filename, "data.npy"))
    #             source.data_types.append("array")
    
    #         if path.exists(path.join(filename, "image.pkl")):
    #             with open(path.join(filename, "image.pkl"), 'rb') as file:
    #                 source.data = pickle.load(file)
    #                 source.data_types.append("image")
    
    #         return source

    def drop_det_array(self) -> None:
        """Release the detection-array cutout by setting it to None."""
        self.det_arr = None

class Sources:
    """
    The collection of the sources of a field.

    The object behaves like a sequence: it can be iterated over,
    indexed with an integer (returning a `Source`) or a slice
    (returning a new `Sources`), measured with `len` and concatenated
    with `+` to a `Source` or to another `Sources`. The iteration state
    is kept on the object itself, so two loops over the same object
    cannot be nested.

    Attributes
    ----------
    sources : list of Source
        The sources.
    header : astropy.io.fits.Header or None
        The FITS header of the mosaic the bounding boxes refer to.
    wcs : astropy.wcs.WCS
        The WCS built from `header`.
    properties : pandas.DataFrame
        The table of the measured properties, set by other modules or
        by `load_light`; saved by `save_light` if present.
    """
    def __init__(self, sources:List[Source], header: Optional[fits.Header]=None) -> None:
        """
        Store the list of sources and build the WCS from the header.

        Parameters
        ----------
        sources
            The sources of the collection.
        header
            The FITS header of the mosaic. If None a default WCS is
            built.
        """
        self.sources= sources
        self.header=header
        self.wcs = WCS(self.header)

    def __iter__(self):
        """
        Reset the internal index and return the object as iterator.
        """
        self._index = 0
        return self

    def __next__(self):
        """Return the next source of the iteration."""
        if self._index < len(self.sources):
            result = self.sources[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration
        
    def __getitem__(self, index):
        """
        Return a source (integer index) or a sub-collection (slice).

        Raises
        ------
        TypeError
            If `index` is neither an `int` nor a `slice`.
        """
        if isinstance(index, slice):
            return Sources(self.sources[index], header=self.header)
        elif isinstance(index, int):
            return self.sources[index]
        else:
            raise TypeError("Invalid argument type.")
        
    def __len__(self):
        """Return the number of sources."""
        return len(self.sources)
    
    def __add__(self, other):
        """
        Return a new collection with a source or a collection appended.

        The header of the left operand is kept.

        Raises
        ------
        TypeError
            If `other` is neither a `Source` nor a `Sources`.
        """
        if isinstance(other, Source):
            return Sources(self.sources + [other], header=self.header)
        elif isinstance(other, Sources):
            return Sources(self.sources + other.sources, header=self.header)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Sources' and '{}'".format(type(other).__name__))
        
    def __radd__(self, other):
        """
        Return a new collection with a source prepended.

        This implements `source + sources`. The header of `self` is
        kept.

        Raises
        ------
        TypeError
            If `other` is not a `Source`.
        """
        if isinstance(other, Source):
            return Sources([other] + self.sources, header=self.header)
        else:
            raise TypeError("Unsupported operand type(s) for +: '{}' and 'Sources'".format(type(other).__name__))

    def sort_by_area(self):
        """Sort the sources in place by decreasing area."""
        self.sources = sorted(self.sources, key=lambda x: x.area)[::-1]
    
    def sources_from_radius(self, source, size, deblending_radius, return_sources=False):
        """
        Build a combined source from a source and its neighbours.

        A square region of side `size` pixels, centred on the centre of
        the bounding box of `source`, is defined together with a
        circular "deblending region" of radius `deblending_radius`
        around the same point. All the sources of the collection with a
        `label` different from that of `source` and whose bounding box
        intersects the square (found with a shapely `STRtree`) are the
        neighbours. They are split in two groups:

        - neighbours with more than half of their bounding-box area
          inside the deblending region are kept as separate components,
          together with `source` itself (which comes first);
        - the other neighbours are masked: their footprint, limited to
          the square, is set to True in a boolean mask of shape `(size,
          size)`.

        The kept sources are merged by `combine_sources` into a single
        `Source` covering the square, whose `segment` is a label map
        (the i-th kept source has value i + 1) and whose `label` is the
        value assigned to `source` (1). The mask is attached to it as
        the attribute `mask`.

        Parameters
        ----------
        source : Source
            The reference source.
        size : int
            The side of the square region in pixels.
        deblending_radius : float
            The radius in pixels of the region within which the
            neighbours are kept as separate components rather than
            masked.
        return_sources : bool
            Whether the collection of the kept sources is returned too.

        Returns
        -------
        source_large : Source
            The combined source covering the square region.
        region_sources : Sources
            The reference source followed by the kept neighbours, only
            returned if `return_sources` is True.

        Raises
        ------
        ValueError
            If `deblending_radius` is larger than `size / 2`.
        """

        label = source.label
        previous_label = source.previous_label

        if deblending_radius > size/2.:
            raise ValueError("Deblending radius must be smaller than the size of the source")

        max_bbox = BoundingBox(
            ixmin=int(source.centre[0] - size/2), 
            ixmax=int(source.centre[0] + size/2), 
            iymin=int(source.centre[1] - size/2), 
            iymax=int(source.centre[1] + size/2))

        max_polygon = BoundingBox_to_Polygon(max_bbox)

        deblending_region = sh.Point(source.centre).buffer(deblending_radius)

        other_sources = []
        for s in self.sources:

            if s.previous_label != previous_label:
                if s.label != label:
                    other_sources.append(s)

            elif (s.previous_label == previous_label) and (s.label != label):
                other_sources.append(s)

        # other_sources = Sources([s for s in self.sources if ((s.label != label) & (s.previous_label != previous_label))], self.header)

        bboxes = [s.bbox for s in other_sources]

        polygons = [BoundingBox_to_Polygon(bbox) for bbox in bboxes]

        tree = sh.strtree.STRtree(polygons)

        neighbours_idx = tree.query(max_polygon, predicate='intersects').tolist()

        neighbours = Sources([other_sources[int(i)] for i in neighbours_idx], self.header)

        neighbours_polygons = [BoundingBox_to_Polygon(n.bbox) for n in neighbours]

        intersections = [sh.intersection(np, deblending_region) for np in neighbours_polygons]

        # A neighbour is "good" (kept as a component) if more than
        # half of its bounding box lies in the deblending region,
        # otherwise it is "bad" (masked).
        good_neighbours = []
        bad_neighbours = []
        for n, intersection in zip(neighbours, intersections):
            if intersection.area > BoundingBox_to_Polygon(n.bbox).area/2.:
                good_neighbours.append(n)
            else:
                bad_neighbours.append(n)

        # TODO: check that this is OK.
        region_sources = source + Sources([n for n in good_neighbours], self.header)

        mask = np.zeros((size, size)).astype(bool)
        for bad_source in bad_neighbours:

            bad_source_intersection = sh.intersection(BoundingBox_to_Polygon(bad_source.bbox), max_polygon)

            _ixmin, _iymin, _ixmax, _iymax = bad_source_intersection.bounds

            # Overlap in the pixel coordinates of the neighbour ...
            source_ixmin = int(_ixmin - bad_source.bbox.ixmin)
            source_iymin = int(_iymin - bad_source.bbox.iymin)
            source_ixmax = int(_ixmax - bad_source.bbox.ixmin)
            source_iymax = int(_iymax - bad_source.bbox.iymin)

            # ... and in those of the square region.
            cutout_ixmin = int(_ixmin - max_bbox.ixmin)
            cutout_iymin = int(_iymin - max_bbox.iymin)
            cutout_ixmax = int(_ixmax - max_bbox.ixmin)
            cutout_iymax = int(_iymax - max_bbox.iymin)

            relative_mask = np.where(bad_source.segment>0, True, False)
            relative_mask = relative_mask[source_iymin:source_iymax, source_ixmin:source_ixmax]

            mask[cutout_iymin:cutout_iymax, cutout_ixmin:cutout_ixmax] += relative_mask
            

        source_large = combine_sources(
            sources=region_sources, 
            reference_label=source.label, 
            bbox=Polygon_to_BoundingBox(max_polygon), 
            )

        source_large.mask = np.where(mask>0, True, False)

        if return_sources:
            return source_large, region_sources

        else:
            return source_large
        
    def sources_from_centre(self, centre, size, deblending_radius, source=None, return_sources=False):
        """
        Build a combined source from the sources around a position.

        This is the variant of `sources_from_radius` for an arbitrary
        centre, which does not need to coincide with a source.

        A square region of side `size` pixels, centred on the pixel
        position of `centre`, is defined together with a circular
        "deblending region" of radius `deblending_radius` around the
        same point. The candidate sources (all the sources of the
        collection or, if `source` is given, those with a `label`
        different from that of `source`) whose bounding box intersects
        the square (found with a shapely `STRtree`) are the neighbours.
        They are split in two groups:

        - neighbours with more than half of their bounding-box area
          inside the deblending region are kept as separate components;
          if `source` is given it is added as the first component;
        - the other neighbours are masked: their footprint, limited to
          the square, is set to True in a boolean mask of shape `(size,
          size)`.

        The kept sources are merged by `combine_sources` into a single
        `Source` covering the square, whose `segment` is a label map
        (the i-th kept source has value i + 1) and whose `label` is the
        value assigned to `source` (1 if `source` is None). The mask is
        attached to it as the attribute `mask`.

        Parameters
        ----------
        centre : astropy.coordinates.SkyCoord
            The sky position of the centre of the region, converted to
            pixels with the WCS of the collection.
        size : int
            The side of the square region in pixels.
        deblending_radius : float
            The radius in pixels of the region within which the
            neighbours are kept as separate components rather than
            masked.
        source : Source, optional
            The reference source. If None all the sources are treated as
            neighbours.
        return_sources : bool
            Whether the collection of the kept sources is returned too.

        Returns
        -------
        source_large : Source
            The combined source covering the square region.
        region_sources : Sources
            The kept sources, only returned if `return_sources` is True.

        Raises
        ------
        ValueError
            If `centre` is not a `SkyCoord`, or if `deblending_radius`
            is larger than `size / 2`.
        """

        if isinstance(centre, SkyCoord):
            centre_pix = self.wcs.all_world2pix(centre.ra.deg, centre.dec.deg, 0)
        else:
            raise ValueError("Coordinates need to be in a SkyCoord object")

        if deblending_radius > size/2.:
            raise ValueError("Deblending radius must be smaller than the size of the source")

        max_bbox = BoundingBox(
            ixmin=int(centre_pix[0] - size/2), 
            ixmax=int(centre_pix[0] + size/2), 
            iymin=int(centre_pix[1] - size/2), 
            iymax=int(centre_pix[1] + size/2))

        max_polygon = BoundingBox_to_Polygon(max_bbox)

        deblending_region = sh.Point(centre_pix).buffer(deblending_radius)

        other_sources = []
        if source is not None:

            label = source.label
            previous_label = source.previous_label
            
            for s in self.sources:

                if s.previous_label != previous_label:
                    if s.label != label:
                        other_sources.append(s)

                elif (s.previous_label == previous_label) and (s.label != label):
                    other_sources.append(s)

        else: 
            for s in self.sources:
                other_sources.append(s)

        bboxes = [s.bbox for s in other_sources]

        polygons = [BoundingBox_to_Polygon(bbox) for bbox in bboxes]

        tree = sh.strtree.STRtree(polygons)

        neighbours_idx = tree.query(max_polygon, predicate='intersects').tolist()

        neighbours = Sources([other_sources[int(i)] for i in neighbours_idx], self.header)

        neighbours_polygons = [BoundingBox_to_Polygon(n.bbox) for n in neighbours]

        intersections = [sh.intersection(np, deblending_region) for np in neighbours_polygons]

        # A neighbour is "good" (kept as a component) if more than
        # half of its bounding box lies in the deblending region,
        # otherwise it is "bad" (masked).
        good_neighbours = []
        bad_neighbours = []
        for n, intersection in zip(neighbours, intersections):
            if intersection.area > BoundingBox_to_Polygon(n.bbox).area/2.:
                good_neighbours.append(n)
            else:
                bad_neighbours.append(n)

        if source is not None:
            # TODO: check that this is OK.
            region_sources = source + Sources([n for n in good_neighbours], self.header)
        else:
            # TODO: check that this is OK.
            region_sources = Sources([n for n in good_neighbours], self.header)

        mask = np.zeros((size, size)).astype(bool)
        for bad_source in bad_neighbours:

            bad_source_intersection = sh.intersection(BoundingBox_to_Polygon(bad_source.bbox), max_polygon)

            _ixmin, _iymin, _ixmax, _iymax = bad_source_intersection.bounds

            # Overlap in the pixel coordinates of the neighbour ...
            source_ixmin = int(_ixmin - bad_source.bbox.ixmin)
            source_iymin = int(_iymin - bad_source.bbox.iymin)
            source_ixmax = int(_ixmax - bad_source.bbox.ixmin)
            source_iymax = int(_iymax - bad_source.bbox.iymin)

            # ... and in those of the square region.
            cutout_ixmin = int(_ixmin - max_bbox.ixmin)
            cutout_iymin = int(_iymin - max_bbox.iymin)
            cutout_ixmax = int(_ixmax - max_bbox.ixmin)
            cutout_iymax = int(_iymax - max_bbox.iymin)

            relative_mask = np.where(bad_source.segment>0, True, False)
            relative_mask = relative_mask[source_iymin:source_iymax, source_ixmin:source_ixmax]

            mask[cutout_iymin:cutout_iymax, cutout_ixmin:cutout_ixmax] += relative_mask
            
        if source is not None:
            reference_label = source.label
        else:
            reference_label = 0

        source_large = combine_sources(
            sources=region_sources, 
            reference_label=reference_label, 
            bbox=Polygon_to_BoundingBox(max_polygon), 
            )

        source_large.mask = np.where(mask>0, True, False)

        if return_sources:
            return source_large, region_sources

        else:
            return source_large
    

    
    def get_data(self, input: Union[np.ndarray, Image, MultibandImage]) -> None:
        """
        Extract the data of every source from a full-mosaic input.

        `Source.get_data` is called on every source with the WCS of the
        collection, showing a progress bar. If the input is an image
        with the same pixel grid as the collection (same shape and
        same WCS in every band), the bounding boxes are sliced
        directly instead of being converted to sky coordinates.

        Parameters
        ----------
        input
            The full-mosaic array or image.
        """
        same_grid = _same_grid(input, self.header, self.wcs)

        for source in tqdm(self.sources):
            source.get_data(input, self.wcs, same_grid=same_grid)


    @classmethod
    def from_segmentation_image(cls, segmap: SegmentationImage, full_det_arr: ArrayLike, header) -> "Sources":
        """
        Build the collection from a full-mosaic segmentation map.

        One `Source` is created per label of the segmentation image: its
        bounding box, its area, the boolean footprint of that label
        inside the box and the cutout of the detection array on the box.
        The `previous_label` of every source is set to 0.

        Parameters
        ----------
        segmap
            The segmentation image of the full mosaic.
        full_det_arr
            The S/N detection array of the full mosaic, with the same
            shape as `segmap`.
        header : astropy.io.fits.Header
            The FITS header of the mosaic.

        Returns
        -------
        Sources
            The collection, with the sources in the order of
            `segmap.labels`.
        """

        labels = segmap.labels
        bboxes = segmap.bbox

        sources_list = []
        for i in tqdm(range(len(labels))):

            bbox_segment_data = _get_bbox_arr(segmap.data, bboxes[i])
            det_arr = _get_bbox_arr(full_det_arr, bboxes[i])

            segment = np.where(
                bbox_segment_data == labels[i], 
                True, 
                False)

            sources_list.append(
                Source(
                    segment = segment,
                    det_arr = det_arr,
                    label = labels[i],
                    area = segmap.areas[i],
                    bbox = bboxes[i], 
                    previous_label=0,
                )
            )

        return cls(sources_list, header)
    
    @classmethod
    def from_deblend(cls, sources) -> "Sources":
        """
        Build a new collection from the deblending of each source.

        Every source must carry the attribute `deblend`, the photutils
        `SegmentationImage` obtained by deblending its cutout. If it
        contains more than one label, the source is replaced by one new
        `Source` per label: the footprint and the detection array are
        cut on the bounding box of the child, which is shifted to
        full-mosaic pixel coordinates by adding the lower corner of the
        parent box; `label` is the label in the deblended map and
        `previous_label` is the label of the parent. A source with a
        single deblended label is kept unchanged.

        Parameters
        ----------
        sources : Sources
            The collection whose sources have been deblended.

        Returns
        -------
        Sources
            The new collection, with the header of `sources`.
        """
        sources_list = []
        # If the source has been deblended, replace it with its
        # children.
        for source in sources:
            if source.deblend.nlabels>1:
                for i,label in enumerate(source.deblend.labels):

                    segment = np.where(
                        source.deblend.data == label, 
                        True, 
                        False)
                    
                    resized_segment=_get_bbox_arr(segment, source.deblend.bbox[i])
                    resized_det_arr=_get_bbox_arr(source.det_arr, source.deblend.bbox[i])

                    # if hasattr(source, "data"):
                    #     source.get_data(source.data, sources.wcs)

                    # Shift the child box from the parent cutout to
                    # full-mosaic pixel coordinates.
                    new_bbox=BoundingBox(
                        source.bbox.ixmin+source.deblend.bbox[i].ixmin,
                        source.bbox.ixmin+source.deblend.bbox[i].ixmax,
                        source.bbox.iymin+source.deblend.bbox[i].iymin,
                        source.bbox.iymin+source.deblend.bbox[i].iymax
                        )

                    sources_list.append(
                        Source(
                            segment = resized_segment,
                            det_arr = resized_det_arr,
                            label = label, 
                            # The parent label, from the segmentation
                            # (detection) step.
                            previous_label = source.label,
                            area = source.deblend.areas[i],
                            bbox = new_bbox
                        )
                    )
            else:
                sources_list.append(source)
            
        return cls(sources_list, sources.header)
    
    @classmethod
    def predict(cls, finding_region, path_to_detection):
        """
        Find the stored sources whose bounding box touches a region.

        The bounding box of every source stored in
        `<path_to_detection>/sources/` is read from its metadata file,
        `source_<id>.metadata.json` or, in the layout of earlier
        versions, `source_<id>/metadata.json`. The identifiers are
        `"<previous_label>_<label>"`.

        Parameters
        ----------
        finding_region : shapely.Geometry
            The region of interest in full-mosaic pixel coordinates.
        path_to_detection : str
            The root folder of the stored collection.

        Returns
        -------
        numpy.ndarray
            The identifiers of the sources whose bounding box intersects
            the region.
        """

        def json_open(filename):
            """Return the content of a JSON file."""
            with open(filename, "r") as file:
                return json.load(file)
            
        path_to_sources = path.join(path_to_detection, "sources")

        sources_names = sorted({
            name.split('.')[0] for name in os.listdir(path_to_sources)
            if name.startswith("source_")
        })
        sources_ids = ['_'.join(name.split("_")[1:]) for name in sources_names]

        metadatas = []
        for name in sources_names:
            source_path = path.join(path_to_sources, name)
            if path.isdir(source_path):
                metadatas.append(json_open(path.join(source_path, 'metadata.json')))
            else:
                metadatas.append(json_open(source_path + '.metadata.json'))

        bboxes = [BoundingBox(
            ixmin=metadata["ixmin"],
            ixmax=metadata["ixmax"],
            iymin=metadata["iymin"],
            iymax=metadata["iymax"]
        ) for metadata in metadatas]

        polygons = [BoundingBox_to_Polygon(bbox) for bbox in bboxes]

        tree = sh.strtree.STRtree(polygons)

        intersecting_ids = tree.query(finding_region, predicate='intersects').tolist()

        return np.array(sources_ids)[np.array(intersecting_ids)]
        
    
    def save_light(self, filename: str, keep_det_arr: bool=True) -> None:
        """
        Save the collection in the light, flat on-disk format.

        The folder `filename` and its sub-folder `sources` are created
        if missing. Every source is saved by `Source.save_light` with
        the file prefix `sources/source_<previous_label>_<label>`, so
        that all the files lie in the same folder. If the collection has
        the attribute `properties`, it is written to `properties.csv`.
        The header is written to `header.hdr`, overwriting an existing
        file.

        Parameters
        ----------
        filename
            The root folder of the stored collection.
        keep_det_arr
            Whether the detection-array cutouts are saved too.
        """

        if not path.exists(filename):
            os.mkdir(filename)
        if not path.exists(path.join(filename, "sources")):
            os.mkdir(path.join(filename, "sources"))

        for source in tqdm(self.sources):

            source_path = path.join(filename, "sources", "source") + f"_{source.previous_label}_{source.label}"

            source.save_light(source_path, keep_det_arr)

        if hasattr(self, "properties"):
            getattr(self, "properties").to_csv(path.join(filename, "properties.csv"))

        self.header.tofile(path.join(filename, "header.hdr"), overwrite=True)
        

    @classmethod
    def load_light(cls, filename: str, ids=None, keep_det_arr: bool=True) -> "Sources":
        """
        Load a collection saved with `save_light`.

        The sources are read from the `cells` sub-folder of `filename`
        if it exists (packed layout, see `write_pack`), sorted by
        label; otherwise from the `sources` sub-folder. If `ids` is None, every source of that folder is
        loaded once, in sorted order of name. The header is read from
        `header.hdr` and, if `properties.csv` exists, it is loaded in
        the attribute `properties`.

        Parameters
        ----------
        filename
            The root folder of the stored collection.
        ids : iterable of str, optional
            The identifiers `"<previous_label>_<label>"` of the sources
            to load. If None the whole `sources` folder is loaded.
        keep_det_arr
            Whether the detection-array cutouts are read (see
            `Source.load_light`).

        Returns
        -------
        Sources
            The loaded collection.
        """

        sources_list = []
        path_to_sources = path.join(filename, "sources")
        path_to_cells = path.join(filename, "cells")

        if path.isdir(path_to_cells):
            # Packed layout: one file per cell.
            for name in tqdm(sorted(os.listdir(path_to_cells))):
                sources_list += read_pack(path.join(path_to_cells, name), keep_det_arr=keep_det_arr)

            if ids is not None:
                wanted = {str(id) for id in ids}
                sources_list = [source for source in sources_list
                                if f"{source.previous_label}_{source.label}" in wanted]

            sources_list.sort(key=lambda source: (source.previous_label, source.label))

            names = []

        elif ids is None:
            filenames=os.listdir(path_to_sources)

            # Each source owns up to three files sharing one prefix.
            names = sorted({name.split('.')[0] for name in filenames})

        else:
            names=[f'source_{id}' for id in ids]
        for name in tqdm(names):
            sources_list.append(Source.load_light(path.join(path_to_sources, name), keep_det_arr=keep_det_arr))

        header=fits.Header.fromfile(path.join(filename, "header.hdr"))
        sources = cls(sources_list, header)

        if path.exists(path.join(filename, "properties.csv")):
            properties = pd.read_csv(path.join(filename, "properties.csv"))
            sources.properties = properties

        return sources
    
    def drop_det_arrays(self) -> None:
        """Release the detection-array cutout of every source."""
        for source in self.sources:
            source.drop_det_array()



from photutils.aperture import BoundingBox
def bbox_shape_of_neighbours(neighbours, bbox=None):
    """
    Return the bounding box enclosing a set of sources.

    Parameters
    ----------
    neighbours : iterable of Source
        The sources to enclose.
    bbox : photutils.aperture.BoundingBox, optional
        An additional box to enclose. If None only the sources are
        used.

    Returns
    -------
    bbox : photutils.aperture.BoundingBox
        The smallest box containing all the boxes.
    shape : tuple of int
        The shape `(ny, nx)` of that box.
    """
    
    bboxes = [neighbour.bbox for neighbour in neighbours]

    if bbox is not None:
        bboxes.append(bbox)
    
    xmin = np.min([bbox.ixmin for bbox in bboxes])
    xmax = np.max([bbox.ixmax for bbox in bboxes])
    ymin = np.min([bbox.iymin for bbox in bboxes])
    ymax = np.max([bbox.iymax for bbox in bboxes])

    return BoundingBox(xmin, xmax, ymin, ymax), (ymax - ymin, xmax - xmin)

def combine_sources(sources, reference_label, bbox=None):
    """
    Merge several sources into one source covering a given box.

    A label map with the shape of `bbox` is filled by adding, for the
    i-th source, the value i + 1 on the part of its footprint that
    falls inside the box; pixels shared by several footprints hold the
    sum of their values. The returned `Source` has this map as
    `segment`, an array of zeros as `det_arr`, `area` 0 and, as
    `label`, the value given to the source whose label equals
    `reference_label` (1 if there is none).

    Parameters
    ----------
    sources : iterable of Source
        The sources to merge, in full-mosaic pixel coordinates.
    reference_label : int
        The `label` of the reference source among `sources`.
    bbox : photutils.aperture.BoundingBox
        The box of the combined source in full-mosaic pixel
        coordinates. If None, the smallest box containing all the
        sources is used.

    Returns
    -------
    Source
        The combined source.
    """
    
    if bbox is None:
        bbox, shape = bbox_shape_of_neighbours(sources)
    else:
        shape = (bbox.iymax - bbox.iymin, bbox.ixmax - bbox.ixmin)

    large_segmap_data = np.zeros(shape)
    large_segmap_det_arr = np.zeros(shape)

    relative_reference_label = 1

    for i, source in enumerate(sources):

        source_intersection = sh.intersection(BoundingBox_to_Polygon(source.bbox), BoundingBox_to_Polygon(bbox))

        _ixmin, _iymin, _ixmax, _iymax = source_intersection.bounds

        # Overlap in the pixel coordinates of the source ...
        source_ixmin = int(_ixmin - source.bbox.ixmin)
        source_iymin = int(_iymin - source.bbox.iymin)
        source_ixmax = int(_ixmax - source.bbox.ixmin)
        source_iymax = int(_iymax - source.bbox.iymin)

        # ... and in those of the combined box.
        cutout_ixmin = int(_ixmin - bbox.ixmin)
        cutout_iymin = int(_iymin - bbox.iymin)
        cutout_ixmax = int(_ixmax - bbox.ixmin)
        cutout_iymax = int(_iymax - bbox.iymin)

        relative_segment_data = source.segment[source_iymin:source_iymax, source_ixmin:source_ixmax]

        large_segmap_data[cutout_iymin:cutout_iymax, cutout_ixmin:cutout_ixmax] += (i+1)*relative_segment_data

        if source.label == reference_label:
            relative_reference_label = i+1

    return Source(large_segmap_data, large_segmap_det_arr, label=relative_reference_label, area=0, bbox=bbox)

def clean_stored_sources(path_to_sources: str) -> None:
    """
    Delete the stored detection-array cutouts of a collection.

    The files `<name>.det_arr.npy` are removed from the `sources`
    sub-folder of `path_to_sources`, to save disk space.

    Parameters
    ----------
    path_to_sources
        The root folder of the stored collection.
    """

    sources_dir = path.join(path_to_sources, "sources")

    filenames=os.listdir(sources_dir)

    names = sorted({name.split('.')[0] for name in filenames})

    for name in tqdm(names):
        path_to_det_arr = path.join(sources_dir, name + ".det_arr.npy")

        if os.path.exists(path_to_det_arr):
            os.remove(path_to_det_arr)

def compress(input_dir: str, output_dir: str) -> None: 
    """
    Compress every file of a folder with zstd and a shared dictionary.

    A zstd dictionary of 300 KiB is trained on the first 1000 files of
    `input_dir` (in the order returned by `os.listdir`) and written to
    `output_dir + "_dict.zstd"`, next to the output folder. Every file
    is then compressed with that dictionary and written to
    `<output_dir>/<name>.zst`, where `<name>` is the full file name, so
    that `decompress` restores the original names. A shared dictionary
    is used because the files (one small set per source) are many,
    small and similar. The output folder is created if missing.

    Parameters
    ----------
    input_dir
        The folder with the files to compress.
    output_dir
        The folder receiving the `.zst` files, without a trailing path
        separator.
    """

    # Imported here so that `zstandard` is needed only for compression.
    import zstandard as zstd

    _files = os.listdir(input_dir)

    os.makedirs(output_dir, exist_ok=True)

    path_to_files = [path.join(input_dir, _f) for _f in _files]

    sample_files = path_to_files[:1000]

    samples = []
    for _f in sample_files:
        with open(_f, "rb") as f:
            samples.append(f.read())

    # Train a 300 KiB dictionary on the sample files.
    dict_trainer = zstd.train_dictionary(300 * 1024, samples) 

    with open(output_dir + '_dict.zstd', "wb") as f: 
        f.write(dict_trainer.as_bytes())

    with open(output_dir + "_dict.zstd", "rb") as f:
        dict_data = zstd.ZstdCompressionDict(f.read())

    cctx = zstd.ZstdCompressor(dict_data=dict_data)

    for _f, _full in tqdm(zip(_files, path_to_files), total=len(_files)):

        with open(_full, "rb") as f:
            data = f.read()

        compressed = cctx.compress(data)

        output_filepath = path.join(output_dir, _f + '.zst')

        with open(output_filepath, "wb") as fout:
            fout.write(compressed)


def decompress(compressed_dir: str, output_dir: str) -> None:
    """
    Decompress a folder written by `compress`.

    The dictionary is read from `compressed_dir + "_dict.zstd"`. Every
    `.zst` file of `compressed_dir` is decompressed and written to
    `output_dir` under the same name without the `.zst` suffix. The
    output folder is created if missing.

    Parameters
    ----------
    compressed_dir
        The folder with the `.zst` files, without a trailing path
        separator.
    output_dir
        The folder receiving the decompressed files.
    """

    import zstandard as zstd

    compressed_dict = compressed_dir + '_dict.zstd'
    
    os.makedirs(output_dir, exist_ok=True)

    with open(compressed_dict, "rb") as _f:
        dict_data = zstd.ZstdCompressionDict(_f.read())

    dctx = zstd.ZstdDecompressor(dict_data=dict_data)

    for filename in tqdm(os.listdir(compressed_dir)):
        if filename.endswith(".zst"):
            input_path = os.path.join(compressed_dir, filename)
            output_path = os.path.join(output_dir, filename[:-4])

            with open(input_path, "rb") as fin:
                compressed_data = fin.read()
                decompressed_data = dctx.decompress(compressed_data)

            with open(output_path, "wb") as fout:
                fout.write(decompressed_data)

