import numpy as np

# from abc import ABC, abstractmethod, abstractproperty, abstractclassmethod

from attila.abstract.abstract_things import AbstractBox

from astropy.io import fits
from astropy.wcs import WCS

from typing import List, Union, Optional
from numpy.typing import ArrayLike

from os import path, mkdir

from photutils.segmentation import SegmentationImage

import json

class BaseTile():

    def __init__(self, 
                 shape, 
                 overlap, 
                 bands, 
                 index, 
                 ) -> None:
        
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
        return self._metadata
    
    @metadata.setter
    def metadata(self, metadata):
        self._metadata = metadata

    @metadata.getter
    def metadata(self):

        if self._metadata is None:
            self._metadata = {"index": self.index,  # This is a list of two integers
                            "shape": tuple(self.shape),  # This is a list of two integers
                            "bands": self.bands,  # This is a list of strings
                            "overlap": int(self.overlap),  # This is an integer
                            "data_keywords": self.data_keywords  # This is a list of strings
                            }

        return self._metadata

    def add_data(
            self, 
            band: str,
            key: Union[List[str], str],
            arr: Union[List[ArrayLike], ArrayLike],
            reg: Optional[Union[List[WCS], WCS]]=None):

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

    # This thing might be outdated. They still might be useful for 'full control'
    # or higher customization purposes. I might want to a have a pickling system though
    def to_folder(self, folder_path):

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
                    # segmap_data = fits.getdata(data_path)

                    segmap_data = np.load(data_path + ".npy")
                    segmap = SegmentationImage(segmap_data)

                    box.add_data(band, key, segmap)

                else:
                    box.add_data(band, key, fits.getdata(data_path + ".fits").astype(np.float32))

                    hdr = fits.getheader(data_path + ".fits")

                box.add_reg(band, "wcs", WCS(hdr))

        return box
    
    """
    For now I only remove overlaps on the right and bottom sides
    of the array (this is the most natural way of doing it given 
    how the boxes are computed). If I will change the way the 
    overlap is defined I can consider changing how the overlaps
    are thrown away. 
    """


#this method should be placed in a different place with customized methods
    def remove_overlap(self, directions):
        
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
        extract data as dict{dict} from a list of bands and a list of keywords
        assuming that either bands or keywords can be a list of a single element, 
        but not both at the same time
        """
        exctracted_data={}
        for band in bands:
            exctracted_data[band]={}

            for keyword in keywords:
                exctracted_data[band][keyword]=self.data2D[band][keyword]

        return exctracted_data
    
"""
BlaBla
"""

class ConvolutionTile(BaseTile):

    def apply_convolution_function(self, band, convolution_function):

        if 'detection_array' not in self.data_keywords:
            raise ValueError("Detection array not found in data keywords")

        self.add_data(band, "detection_array",  convolution_function(self.data2D[band]["detection_array"]))

"""
BlaBla
"""

class DetectionTile(BaseTile):

    def fix_zeros(self, fill):

        for band in self.bands:
            self.data2D[band]["detection_array"] = np.where(self.data2D[band]["detection_array"]==0, 
                                                            fill, self.data2D[band]["detection_array"])

    def apply_detection_function(self, band, detection_function):

        if 'detection_array' not in self.data_keywords:
            raise ValueError("Detection array not found in data keywords")

        self.add_data(band, "segmap_data",  detection_function(self.data2D[band]["detection_array"]).data)


    