"""
Base classes for the per-tile "box" containers.

`AbstractBox` declares the interface of a box: a container of the
multi-band data of one tile that can be saved to a folder, loaded
back and cropped of its overlap region. `BaseBox` is the older
concrete container, still subclassed by `DetectionBox`
(`attila.detection.detection`) and `DeblendingBox`
(`attila.deblending.deblending`).

`BaseBox` is a near-duplicate of `attila.tiling.tiles.BaseTile`.
"""

import numpy as np

# from abc import ABC, abstractmethod, abstractproperty, abstractclassmethod

from astropy.io import fits
from astropy.wcs import WCS

from typing import List

from os import path, mkdir

from photutils.segmentation import SegmentationImage

import json
from abc import ABC, abstractmethod

class AbstractBox(ABC):
    """
    The interface of a container of the multi-band data of one tile.

    Concrete boxes must provide the geometry (`shape`, `overlap`), the
    list of `bands`, the names and content of the stored arrays
    (`data_keywords`, `data2D`), the folder input / output and the
    removal of the overlap region. `BaseBox` and `BaseTile` implement
    the same members without inheriting from this class.
    """

    @property
    @abstractmethod
    def shape(self):
        """The shape of the box."""
        pass

    @property
    @abstractmethod
    def overlap(self):
        """The overlap of the box with its neighbours."""
        pass

    @property
    @abstractmethod
    def bands(self):
        """The names of the bands stored in the box."""
        pass

    @property
    @abstractmethod
    def data_keywords(self):
        """The names of the arrays stored in the box."""
        pass

    @property
    @abstractmethod
    def data2D(self):
        """The stored arrays, as `data2D[band][key]`."""
        pass

    @abstractmethod
    def to_folder(self, folder_path):
        """
        Write the content of the box to a folder.

        Parameters
        ----------
        folder_path : str
            The folder in which the box is written.
        """
        pass

    @classmethod
    @abstractmethod
    def from_folder(cls, folder_path, index):
        """
        Build a box from a folder written by `to_folder`.

        Parameters
        ----------
        folder_path : str
            The folder containing the stored boxes.
        index : int
            The index of the box to load.
        """
        pass

    @abstractmethod
    def remove_overlap(self, directions):
        """
        Crop the overlap region from the given sides of the box.

        Parameters
        ----------
        directions : list of str
            The sides from which the overlap is removed.
        """
        pass


class BaseBox():
    """
    A container for the multi-band data of one box of the mosaic.

    The object stores, for every band, a dictionary of named 2D arrays
    (for example the science data, the noise, the detection array or a
    segmentation map) together with the registration information (WCS)
    of each array. It can be written to and read from a folder, cropped
    of its overlap region and queried for a subset of its data.

    This class is the older near-duplicate of `BaseTile` of
    `attila.tiling.tiles`. It is still the parent of `DetectionBox`
    and `DeblendingBox`. Unlike `BaseTile`, it stores the data and
    the registration information with two separate methods,
    `add_data` and `add_reg`. It does not inherit from `AbstractBox`.

    Attributes
    ----------
    shape : tuple
        The shape of the box, as passed to the constructor.
    overlap : float or int
        The overlap with the neighbouring boxs.
    bands : list of str
        The names of the bands.
    working_bands : list of str
        Initially the same list as `bands`.
    index : int
        The index identifying the box in the tiling.
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
            The shape of the box. `remove_overlap` treats it as a shape
            in pixels.
        overlap : float or int
            The overlap with the neighbouring boxs. `remove_overlap`
            uses it as a number of pixels.
        bands : list of str
            The names of the bands.
        index : int
            The index identifying the box in the tiling.
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
        `index`, `shape`, `bands`, `overlap` (cast to an integer) and
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
                            "overlap": int(self.overlap),  # An integer.
                            "data_keywords": self.data_keywords  # A list of strings.
                            }

        return self._metadata



    def add_data(self, band, key, arr):
        """
        Store a named array for a band.

        An existing entry with the same key is overwritten. A new key is
        appended to `data_keywords`.

        Parameters
        ----------
        band : str
            The band the data belong to.
        key : str
            The name under which the array is stored.
        arr : numpy.ndarray or SegmentationImage
            The 2D data, with axes `(y, x)`.
        """

        self.data2D[band][key] = arr

        if key not in self.data_keywords:
            self.data_keywords.append(key)
            
    def add_reg(self, band, key, reg):
        """
        Store a named piece of registration information for a band.

        Parameters
        ----------
        band : str
            The band the information belongs to.
        key : str
            The name under which it is stored; `to_folder` reads the key
            `"wcs"`.
        reg : astropy.wcs.WCS
            The registration information, typically the WCS of the
            arrays.
        """

        self.reg[band][key] = reg


    def to_folder(self, folder_path):
        """
        Write the metadata and the arrays to a folder tree.

        The layout is `<folder_path>/<index>/metadata.json` plus one
        sub-folder per band containing one file per data keyword:
        `<key>.npy` (unsigned 32-bit integers) if the array is a
        photutils `SegmentationImage`, otherwise `<key>.fits` (32-bit
        floats) with a header built from the WCS stored in
        `reg[band]["wcs"]`. Missing folders are created (one level at a
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
                    fits.writeto(data_path+".fits", self.data2D[band][key].astype(np.float32), header=self.reg[band]["wcs"].to_header(), overwrite=True)

    @classmethod
    def from_folder(cls, folder_path, index):
        """
        Build an object from a folder written by `to_folder`.

        The constructor arguments (`shape`, `overlap`, `bands`, `index`)
        are taken from `metadata.json` and the full metadata dictionary
        is attached to the new object. For every band and every data
        keyword, the key `"segmap"` is read from `<key>.npy` and wrapped
        in a photutils `SegmentationImage`; any other key is read from
        `<key>.fits` as 32-bit floats. After each key the WCS built from
        the last FITS header read is registered with `add_reg` under the
        name `"wcs"`.

        Parameters
        ----------
        folder_path : str
            The folder containing the `<index>` folder.
        index : int
            The index of the box to load.

        Returns
        -------
        BaseBox
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

                if key == "segmap":
                    # Segmentation maps are stored as .npy files.
                    # segmap_data = fits.getdata(data_path)

                    segmap_data = np.load(data_path + ".npy")
                    segmap = SegmentationImage(segmap_data)

                    box.add_data(band, key, segmap)

                else:
                    box.add_data(band, key, fits.getdata(data_path + ".fits").astype(np.float32))

                    hdr = fits.getheader(data_path + ".fits")

                box.add_reg(band, "wcs", WCS(hdr))

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
        exctracted_data={}
        for band in bands:
            exctracted_data[band]={}

            for keyword in keywords:
                exctracted_data[band][keyword]=self.data2D[band][keyword]

        return exctracted_data

