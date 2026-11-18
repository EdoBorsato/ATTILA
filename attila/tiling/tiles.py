"""
Tile containers holding the multi-band data of one sky tile.

A tile is the unit of work of the detection pipeline: `Tiling`
(`attila.tiling.tilingrid`) creates one tile object per sky tile,
fills it with the cutouts of the full-image arrays and, once the
tile has been processed, pastes its results back into the mosaic.

`BaseTile` is the generic container. `ConvolutionTile` and
`DetectionTile` add the methods that apply a smoothing function and
a detection function to the detection array of the tile.
"""
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS

from typing import List, Union, Optional
from numpy.typing import ArrayLike

from os import path, mkdir

from photutils.segmentation import SegmentationImage

import json

class BaseTile():
    """
    A container for the multi-band data of one tile of the mosaic.

    The object stores, for every band, a dictionary of named 2D arrays
    (for example the science data, the noise, the detection array or a
    segmentation map) together with the registration information (WCS)
    of each array. It can be written to and read from a folder, cropped
    of its overlap region and queried for a subset of its data.

    Attributes
    ----------
    shape : tuple
        The sky shape of the tile in degrees.
    overlap : float
        The overlap with the neighbouring tiles in degrees.
    bands : list of str
        The names of the bands.
    working_bands : list of str
        Initially the same list as `bands`.
    index : int
        The index identifying the tile in the tiling.
    data2D : dict
        The arrays, as `data2D[band][key]`.
    data_keywords : list of str
        The names (`key`) of the arrays added so far, over all bands.
    reg : dict
        The registration information, as `reg[band][key]`.
    metadata : dict
        The summary dictionary written to `metadata.json`.
    """

    def __init__(self, 
                 shape, 
                 overlap, 
                 bands, 
                 index, 
                 ) -> None:
        """
        Store the geometry and create the empty data containers.

        Parameters
        ----------
        shape : tuple
            The sky shape of the tile in degrees. Note that
            `remove_overlap` still treats it as a shape in pixels.
        overlap : float
            The overlap with the neighbouring tiles in degrees. Note
            that `remove_overlap` still uses it as a number of pixels.
        bands : list of str
            The names of the bands.
        index : int
            The index identifying the tile in the tiling.
        """
        
        self.shape = shape
        self.overlap = overlap
        self.bands = bands
        self.index = index
        self.working_bands = bands

        self.data2D = {}
        for band in self.bands:
            self.data2D[band] = {}

        self.data_keywords = []

        self.reg = {}
        for band in self.bands:
            self.reg[band] = {}

        self._metadata = None
        
    @property
    def metadata(self):
        """The metadata dictionary of the object."""
        return self._metadata
    
    @metadata.setter
    def metadata(self, metadata):
        """Set the metadata dictionary."""
        self._metadata = metadata

    @metadata.getter
    def metadata(self):
        """
        Return the metadata dictionary, building it on first access.

        If no metadata have been set, a dictionary with the keys
        `index`, `shape`, `bands`, `overlap` (in degrees) and
        `data_keywords` is created from the current attributes and
        cached. Later changes of the attributes are not propagated to
        the cached dictionary, except for `data_keywords`, which is
        stored by reference.

        Returns
        -------
        dict
            The metadata dictionary.
        """

        if self._metadata is None:
            self._metadata = {"index": self.index,  # The tile index.
                            "shape": tuple(self.shape),  # Two values.
                            "bands": self.bands,  # A list of strings.
                            "overlap": float(self.overlap),  # In degrees.
                            "data_keywords": self.data_keywords  # A list of strings.
                            }

        return self._metadata

    def add_data(
            self, 
            band: str,
            key: Union[List[str], str],
            arr: Union[List[ArrayLike], ArrayLike],
            reg: Optional[Union[List[WCS], WCS]]=None):
        """
        Store one or more arrays, with their registration, for a band.

        If `key`, `arr` and `reg` are all lists, they are stored element
        by element; otherwise they are stored as a single entry. An
        existing entry with the same key is overwritten. New keys are
        appended to `data_keywords`.

        Parameters
        ----------
        band
            The band the data belong to.
        key
            The name, or list of names, under which the data are stored.
        arr
            The 2D array, or list of 2D arrays, with axes `(y, x)`.
        reg
            The registration information of the array (its WCS), or a
            list of them. If None, None is stored in `reg[band][key]`.
        """

        if (isinstance(key, List) and isinstance(arr, List) and isinstance(reg, List)):
            for _key, _arr, _reg in zip(key, arr, reg):
                self.data2D[band][_key] = _arr
                self.reg[band][_key] = _reg

                if _key not in self.data_keywords:
                    self.data_keywords.append(_key)

        else:
            self.data2D[band][key] = arr
            self.reg[band][key] = reg

            if key not in self.data_keywords:
                self.data_keywords.append(key)

    # TODO: `to_folder` and `from_folder` might be outdated. They might
    # still be useful for "full control" or higher customisation
    # purposes. A pickling system might be preferable though.
    def to_folder(self, folder_path):
        """
        Write the metadata and the arrays to a folder tree.

        The layout is `<folder_path>/<index>/metadata.json` plus one
        sub-folder per band containing one file per data keyword:
        `<key>.npy` (unsigned 32-bit integers) if the array is a
        photutils `SegmentationImage`, otherwise `<key>.fits` (32-bit
        floats) with a header built from the WCS stored in
        `reg[band][key]`, if any. Missing folders are created (one level at a
        time) and existing FITS files are overwritten.

        Parameters
        ----------
        folder_path : str
            The existing folder in which the `<index>` folder is
            created.
        """

        box_path = path.join(folder_path, f"{self.index}")

        if not path.exists(box_path):
            mkdir(box_path)
        
        with open(path.join(box_path, "metadata.json"), "w") as handle:
            json.dump(self.metadata, handle, indent=4)

        for band in self.bands:
            box_band_path = path.join(box_path, f"{band}")

            if not path.exists(box_band_path):
                mkdir(box_band_path)
        
            for key in self.data_keywords:
                data_path = path.join(box_band_path, f"{key}")

                if isinstance(self.data2D[band][key], SegmentationImage):

                    np.save(data_path+".npy", self.data2D[band][key].data.astype(np.uint32))

                else:
                    wcs = self.reg[band].get(key)
                    header = wcs.to_header() if wcs is not None else None

                    fits.writeto(data_path+".fits", self.data2D[band][key].astype(np.float32), header=header, overwrite=True)


    @classmethod
    def from_folder(cls, folder_path, index):
        """
        Build an object from a folder written by `to_folder`.

        The constructor arguments (`shape`, `overlap`, `bands`, `index`)
        are taken from `metadata.json` and the full metadata dictionary
        is attached to the new object. For every band and every data
        keyword, a key stored as `<key>.npy` is wrapped in a photutils
        `SegmentationImage`; any other key is read from `<key>.fits` as
        32-bit floats, together with the WCS of its header.

        Parameters
        ----------
        folder_path : str
            The folder containing the `<index>` folder.
        index : int
            The index of the tile to load.

        Returns
        -------
        BaseTile
            The loaded object (an instance of `cls`).
        """

        box_path = path.join(folder_path, f"{index}")

        with open(path.join(box_path, "metadata.json")) as handle:
            metadata = json.load(handle)

        min_metadata = {key: metadata[key] for key in  ["shape", "overlap", "bands", "index"]}
        min_metadata["shape"] = tuple(min_metadata["shape"])

        box = cls(**min_metadata)

        box.metadata = metadata

        for band in box.bands:
            box_band_path = path.join(box_path, f"{band}")

            for key in metadata["data_keywords"]:
                data_path = path.join(box_band_path, f"{key}")

                if path.exists(data_path + ".npy"):
                    # Segmentation maps are stored as .npy files.
                    segmap_data = np.load(data_path + ".npy")
                    segmap = SegmentationImage(segmap_data)

                    box.add_data(band, key, segmap)

                else:
                    hdr = fits.getheader(data_path + ".fits")

                    box.add_data(band, key, fits.getdata(data_path + ".fits").astype(np.float32), WCS(hdr))

        return box
    
    # For now only the overlaps on the right and bottom sides of the
    # array are removed (this is the most natural way of doing it given
    # how the boxes are computed). If the way the overlap is defined
    # changes, how the overlaps are thrown away can be reconsidered.

    # TODO: move this method somewhere else, together with other
    # customised methods.
    def remove_overlap(self, directions):
        """
        Crop the overlap region from the given sides of all arrays.

        Every array of every band is sliced in place, removing `overlap`
        rows or columns for each requested direction: `"top"` removes
        the first rows, `"bottom"` the last rows, `"left"` the first
        columns and `"right"` the last columns (rows are the first array
        axis). `overlap` is used here as a number of pixels. For a
        `SegmentationImage` the `data` attribute is replaced.

        Afterwards `shape` is reduced by `overlap` along both axes,
        whatever the directions requested, `overlap` is set to 0 and the
        metadata are updated accordingly.

        Parameters
        ----------
        directions : list of str
            The sides to crop, among `"top"`, `"bottom"`, `"left"` and
            `"right"`.

        Raises
        ------
        ValueError
            If `overlap` is 0.
        """
        
        if self.overlap != 0:

            slice_ops = {
                        'top': (slice(self.overlap, None), slice(None)),
                        'bottom': (slice(None, -self.overlap), slice(None)),
                        'left': (slice(None), slice(self.overlap, None)),
                        'right': (slice(None), slice(None, -self.overlap))
                    }
        
            for band in self.bands:
                for key in self.data_keywords:
                    for direction in directions:

                        slice_op = slice_ops[direction]

                        if isinstance(self.data2D[band][key], SegmentationImage):
                            self.data2D[band][key].data = self.data2D[band][key].data[slice_op]
                        else:
                             self.data2D[band][key] = self.data2D[band][key][slice_op]

            self.shape = (self.shape[0]-self.overlap, self.shape[1]-self.overlap)
            self.overlap = 0

            self.metadata["shape"] = self.shape
            self.metadata["overlap"] = 0

        else:
            raise ValueError("No overlap to remove.")
        
    def extract_data(
            self,
            bands: List[str],
            keywords: List[str],
            ):
        """
        Return a subset of the stored arrays as a nested dictionary.

        Either `bands` or `keywords` can be a list with a single
        element, but the intended use is not to have both of them so at
        the same time.

        Parameters
        ----------
        bands
            The bands to extract.
        keywords
            The names of the arrays to extract for every band.

        Returns
        -------
        dict
            The arrays as `extracted[band][keyword]`. They are
            references to the stored arrays, not copies.
        """
        extracted_data={}
        for band in bands:
            extracted_data[band]={}

            for keyword in keywords:
                extracted_data[band][keyword]=self.data2D[band][keyword]

        return extracted_data
    

class ConvolutionTile(BaseTile):
    """
    A tile whose detection array can be convolved.

    It is used in the smoothing step of the detection pipeline, where
    the detection array of each tile is convolved (for example with a
    Gaussian kernel) before the sources are detected.
    """

    def apply_convolution_function(self, band, convolution_function):
        """
        Replace the detection array of a band with its convolution.

        Parameters
        ----------
        band : str
            The band whose `"detection_array"` is convolved.
        convolution_function : callable
            The function taking the 2D detection array and returning the
            convolved array.

        Raises
        ------
        ValueError
            If no `"detection_array"` has been added to the tile.
        """

        if 'detection_array' not in self.data_keywords:
            raise ValueError("Detection array not found in data keywords")

        self.add_data(band, "detection_array",  convolution_function(self.data2D[band]["detection_array"]))


class DetectionTile(BaseTile):
    """
    A tile on which the source detection is run.

    It is used in the detection step of the pipeline, where a detection
    function (for example a thresholding with photutils
    `detect_sources`) turns the detection array of each tile into a
    segmentation map.
    """

    def fix_zeros(self, fill):
        """
        Replace the zeros of the detection array with a fill value.

        The replacement is applied to the `"detection_array"` of every
        band of the tile.

        Parameters
        ----------
        fill : float
            The value assigned to the pixels equal to 0.
        """

        for band in self.bands:
            self.data2D[band]["detection_array"] = np.where(self.data2D[band]["detection_array"]==0, 
                                                            fill, self.data2D[band]["detection_array"])

    def apply_detection_function(self, band, detection_function):
        """
        Run a detection function on the detection array of a band.

        The `data` attribute of the object returned by the function (the
        integer label array of a segmentation image) is stored in the
        tile under the name `"segmap_data"`.

        Parameters
        ----------
        band : str
            The band whose `"detection_array"` is used.
        detection_function : callable
            The function taking the 2D detection array and returning an
            object with a `data` attribute, such as a photutils
            `SegmentationImage`.

        Raises
        ------
        ValueError
            If no `"detection_array"` has been added to the tile.
        """

        if 'detection_array' not in self.data_keywords:
            raise ValueError("Detection array not found in data keywords")

        self.add_data(band, "segmap_data",  detection_function(self.data2D[band]["detection_array"]).data)


    