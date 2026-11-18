from photutils.segmentation import SegmentationImage
import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt

from attila.field.field import Image, MultibandImage
# from attila.packing.packing import BoundingBox_to_Polygon, Polygon_to_BoundingBox

from photutils.aperture import BoundingBox
from photutils.segmentation import Segment

from astropy import units as u

from scipy.spatial import KDTree

from tqdm import tqdm

from attila.attila_logger import logger

import os
from os import path

import pickle
import json

from typing import Union, List, Optional

import pandas as pd

import shapely as sh

from astropy.wcs import WCS
from astropy.io import fits

from numpy.typing import ArrayLike
from astropy.coordinates import SkyCoord

import warnings

def Polygon_to_BoundingBox(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    return BoundingBox(ixmin=int(minx), ixmax=int(maxx), iymin=int(miny), iymax=int(maxy))

def BoundingBox_to_Polygon(boundingbox, padding=0, shape=None):
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

    # Astropy Cutout2D expects scalar world coordinates here; length-1 arrays
    # can fail on newer versions when converted to pixel positions.
    x_center = float(bbox.center[1])
    y_center = float(bbox.center[0])
    position = wcs.pixel_to_world(x_center, y_center)

    pixel_scale = np.mean([_ps.to(u.arcsec).value for _ps in wcs.proj_plane_pixel_scales()])
    size_x = float((bbox.ixmax - bbox.ixmin) * pixel_scale)
    size_y = float((bbox.iymax - bbox.iymin) * pixel_scale)

    return image.cutout(position=position, size=(size_y * u.arcsec, size_x * u.arcsec))

def _get_bbox_arr(arr, bbox):
        if arr.ndim == 2:
            return arr[bbox.iymin:bbox.iymax, bbox.ixmin:bbox.ixmax]
        elif arr.ndim == 3:
            return arr[:, bbox.iymin:bbox.iymax, bbox.ixmin:bbox.ixmax]
        elif arr.ndim == 4:
            return arr[:, :, bbox.iymin:bbox.iymax, bbox.ixmin:bbox.ixmax]

class Source:
    def __init__(self, 
                segment: ArrayLike,
                det_arr: ArrayLike,
                label: int,
                area: int,
                bbox: BoundingBox,
                previous_label=None) -> None:
    
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
        return (self.bbox.ixmin + self.bbox.ixmax) / 2, (self.bbox.iymin + self.bbox.iymax) / 2

    def get_data(self, input: Union[ArrayLike, Image, MultibandImage], wcs=None) -> None:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if isinstance(input, np.ndarray):
                self.data = _get_bbox_arr(input, self.bbox)

                if "array" not in self.data_types:
                    self.data_types.append("array")            

            elif (isinstance(input, Image)) | (isinstance(input, MultibandImage)):
                self.data = _get_bbox_image(input, self.bbox, wcs=wcs)

                if "image" not in self.data_types:
                    self.data_types.append("image")

    def save_light(self, filename: str, keep_det_arr: bool=True) -> None:

        if not path.exists(filename):
            os.mkdir(filename)

        np.save(path.join(filename, f"segmap.npy"), self.segment.astype(np.uint8))

        if keep_det_arr:
            np.save(path.join(filename, f"det_arr.npy"), self.det_arr.astype(np.float32))

        metadata = {
            "label": self.label, 
            "area": self.area, 
            "ixmin": self.bbox.ixmin, 
            "ixmax": self.bbox.ixmax, 
            "iymin": self.bbox.iymin, 
            "iymax": self.bbox.iymax, 
            "previous_label": self.previous_label, 
            }
        
        with open(path.join(filename, "metadata.json"), "w+") as handle:
            json.dump(metadata, handle, indent=4)

    @classmethod
    def load_light(cls, filename: str, keep_det_arr: bool) -> None:

        with open(filename+"/metadata.json") as handle:
            metadata = json.load(handle)

        if keep_det_arr:
            det_arr = np.load(filename+"/det_arr.npy").astype(bool)
        else:
            det_arr = np.load(filename+"/segmap.npy").astype(bool)

        return cls(
            segment = np.load(filename+"/segmap.npy").astype(bool),
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

    def drop_det_array(self) -> None:
        self.det_arr = None

class Sources:
    def __init__(self, sources:List[Source], header: Optional[fits.Header]=None) -> None:
        self.sources= sources
        self.header=header
        self.wcs = WCS(self.header)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.sources):
            result = self.sources[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration
        
    def __getitem__(self, index):
        if isinstance(index, slice):
            return Sources(self.sources[index], header=self.header)
        elif isinstance(index, int):
            return self.sources[index]
        else:
            raise TypeError("Invalid argument type.")
        
    def __len__(self):
        return len(self.sources)
    
    def __add__(self, other):
        if isinstance(other, Source):
            return Sources(self.sources + [other], header=self.header)
        elif isinstance(other, Sources):
            return Sources(self.sources + other.sources, header=self.header)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Sources' and '{}'".format(type(other).__name__))
        
    def __radd__(self, other):
        if isinstance(other, Source):
            return Sources([other] + self.sources, header=self.header)
        else:
            raise TypeError("Unsupported operand type(s) for +: '{}' and 'Sources'".format(type(other).__name__))

    def sort_by_area(self):
        self.sources = sorted(self.sources, key=lambda x: x.area)[::-1]

    def sources_from_radius(self, source, size, deblending_radius, return_sources=False):

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

        good_neighbours = []
        bad_neighbours = []
        for n, intersection in zip(neighbours, intersections):
            if intersection.area > BoundingBox_to_Polygon(n.bbox).area/2.:
                good_neighbours.append(n)
            else:
                bad_neighbours.append(n)

        region_sources = source + Sources([n for n in good_neighbours], self.header) ## OK??

        mask = np.zeros((size, size)).astype(bool)
        for bad_source in bad_neighbours:

            bad_source_intersection = sh.intersection(BoundingBox_to_Polygon(bad_source.bbox), max_polygon)

            _ixmin, _iymin, _ixmax, _iymax = bad_source_intersection.bounds

            source_ixmin = int(_ixmin - bad_source.bbox.ixmin)
            source_iymin = int(_iymin - bad_source.bbox.iymin)
            source_ixmax = int(_ixmax - bad_source.bbox.ixmin)
            source_iymax = int(_iymax - bad_source.bbox.iymin)

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

        if isinstance(centre, SkyCoord):
            centre_pix = self.wcs.all_world2pix(centre.ra.deg, centre.dec.deg, 0)
        else:
            raise ValueError("Coordinates need to be in a SkyCoord object")

        # source = self.sources[int(np.where([s.label == label for s in self.sources])[0][0])]

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

        # other_sources = Sources([s for s in self.sources if ((s.label != label) & (s.previous_label != previous_label))], self.header)

        bboxes = [s.bbox for s in other_sources]

        polygons = [BoundingBox_to_Polygon(bbox) for bbox in bboxes]

        tree = sh.strtree.STRtree(polygons)

        neighbours_idx = tree.query(max_polygon, predicate='intersects').tolist()

        neighbours = Sources([other_sources[int(i)] for i in neighbours_idx], self.header)

        neighbours_polygons = [BoundingBox_to_Polygon(n.bbox) for n in neighbours]

        intersections = [sh.intersection(np, deblending_region) for np in neighbours_polygons]

        good_neighbours = []
        bad_neighbours = []
        for n, intersection in zip(neighbours, intersections):
            if intersection.area > BoundingBox_to_Polygon(n.bbox).area/2.:
                good_neighbours.append(n)
            else:
                bad_neighbours.append(n)

        if source is not None:
            region_sources = source + Sources([n for n in good_neighbours], self.header) ## OK??
        else:
            region_sources = Sources([n for n in good_neighbours], self.header) ## OK??

        mask = np.zeros((size, size)).astype(bool)
        for bad_source in bad_neighbours:

            bad_source_intersection = sh.intersection(BoundingBox_to_Polygon(bad_source.bbox), max_polygon)

            _ixmin, _iymin, _ixmax, _iymax = bad_source_intersection.bounds

            source_ixmin = int(_ixmin - bad_source.bbox.ixmin)
            source_iymin = int(_iymin - bad_source.bbox.iymin)
            source_ixmax = int(_ixmax - bad_source.bbox.ixmin)
            source_iymax = int(_iymax - bad_source.bbox.iymin)

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
        for source in tqdm(self.sources):
            source.get_data(input, self.wcs)


    @classmethod
    def from_segmentation_image(cls, segmap: SegmentationImage, full_det_arr: ArrayLike, header) -> None:

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

            sources = cls(sources_list, header)

        return sources
    
    @classmethod
    def from_deblend(cls, sources) -> None:
        sources_list = []
        #if the source has been deblended:
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
                            previous_label = source.label, #label della segmentazione
                            area = source.deblend.areas[i],
                            bbox = new_bbox
                        )
                    )
            else:
                sources_list.append(source)
            
        return cls(sources_list, sources.header)
    
    @classmethod
    def predict(cls, finding_region, path_to_detection):

        def json_open(filename):
            with open(filename, "r") as file:
                return json.load(file)
            
        path_to_sources = path.join(path_to_detection, "sources")

        sources_names = os.listdir(path_to_sources)
        sources_ids = ['_'.join(name.split("_")[1:]) for name in sources_names if name.startswith("source_")]
            
        sources_paths = [path.join(path_to_sources, f) for f in sources_names]
        
        metadatas = [json_open(path.join(source_path, 'metadata.json')) for source_path in sources_paths]

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

        if not path.exists(filename):
            os.mkdir(filename)
        if not path.exists(path.join(filename, "sources")):
            os.mkdir(path.join(filename, "sources"))

        for source in self.sources:

            source_path = path.join(filename, "sources", "source") + f"_{source.previous_label}_{source.label}"

            if not path.exists(source_path):
                os.mkdir(source_path)

            source.save_light(source_path, keep_det_arr)

        if hasattr(self, "properties"):
            getattr(self, "properties").to_csv(path.join(filename, "properties.csv"))

        self.header.tofile(path.join(filename, "header.hdr"), overwrite=True)
        

    @classmethod
    def load_light(cls, filename: str, ids=None, keep_det_arr: bool=True) -> None:

        sources_list = []
        if ids is None:
            names=os.listdir(filename+ "sources/")
        else:
            names=[f'source_{id}' for id in ids]
        for name in tqdm(names):
            sources_list.append(Source.load_light(filename + "sources/" + name, keep_det_arr=keep_det_arr))

        header=fits.Header.fromfile(path.join(filename, "header.hdr"))
        sources = cls(sources_list, header)

        if path.exists(path.join(filename, "properties.csv")):
            properties = pd.read_csv(path.join(filename, "properties.csv"))
            sources.properties = properties

        return sources
    
    def drop_det_arrays(self) -> None:
        for source in self.sources:
            source.drop_det_array()



from photutils.aperture import BoundingBox
def bbox_shape_of_neighbours(neighbours, bbox=None):
    
    bboxes = [neighbour.bbox for neighbour in neighbours]

    if bbox is not None:
        bboxes.append(bbox)
    
    xmin = np.min([bbox.ixmin for bbox in bboxes])
    xmax = np.max([bbox.ixmax for bbox in bboxes])
    ymin = np.min([bbox.iymin for bbox in bboxes])
    ymax = np.max([bbox.iymax for bbox in bboxes])

    return BoundingBox(xmin, xmax, ymin, ymax), (ymax - ymin, xmax - xmin)

def combine_sources(sources, reference_label, bbox=None):
    
    bbox, shape = bbox, (bbox.iymax - bbox.iymin, bbox.ixmax - bbox.ixmin)

    large_segmap_data = np.zeros(shape)
    large_segmap_det_arr = np.zeros(shape)

    relative_refence_label = 1

    for i, source in enumerate(sources):

        source_intersection = sh.intersection(BoundingBox_to_Polygon(source.bbox), BoundingBox_to_Polygon(bbox))

        _ixmin, _iymin, _ixmax, _iymax = source_intersection.bounds

        source_ixmin = int(_ixmin - source.bbox.ixmin)
        source_iymin = int(_iymin - source.bbox.iymin)
        source_ixmax = int(_ixmax - source.bbox.ixmin)
        source_iymax = int(_iymax - source.bbox.iymin)

        cutout_ixmin = int(_ixmin - bbox.ixmin)
        cutout_iymin = int(_iymin - bbox.iymin)
        cutout_ixmax = int(_ixmax - bbox.ixmin)
        cutout_iymax = int(_iymax - bbox.iymin)

        relative_segment_data = source.segment[source_iymin:source_iymax, source_ixmin:source_ixmax]

        large_segmap_data[cutout_iymin:cutout_iymax, cutout_ixmin:cutout_ixmax] += (i+1)*relative_segment_data

        if source.label == reference_label:
            relative_refence_label = i+1

    return Source(large_segmap_data, large_segmap_det_arr, label=relative_refence_label, area=0, bbox=bbox)

def clean_stored_sources(path_to_sources: str) -> None:

    names = os.listdir(path_to_sources + "sources/")

    for name in names:
        path_to_det_arr = path_to_sources + "sources/" + name + "/det_arr.npy"

        if os.path.exists(path_to_det_arr):
            os.remove(path_to_det_arr)


   


