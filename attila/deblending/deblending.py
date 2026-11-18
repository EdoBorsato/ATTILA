import numpy as np 

from scipy.ndimage import gaussian_filter, grey_closing

from photutils.background import Background2D, MedianBackground

from photutils.segmentation import detect_threshold, detect_sources, deblend_sources, SourceCatalog
from photutils.segmentation import SegmentationImage

from astropy.stats import SigmaClip, sigma_clip, sigma_clipped_stats, biweight_location

from attila.abstract.abstract_things import BaseBox

from astropy.convolution import convolve, convolve_fft

from scipy.ndimage import grey_closing

from astropy.convolution import Kernel2D
import shelve


def deblend(
        detection_array, 
        segmap, 
        npixels=5, 
        nlevels=8, 
        contrast=1/8., 
        mode='exponential'
        ):

    # segmap.remove_border_labels(border_width=1)
    
    deblmap = deblend_sources(data=detection_array, 
                              segment_img=segmap, 
                              npixels=npixels, 
                              nlevels=nlevels,
                              contrast=contrast,
                              mode=mode,
                              relabel=False,
                              progress_bar=False, 
                              connectivity=8)
    
    if deblmap is None:
        deblmap = SegmentationImage(np.zeros(detection_array.shape, dtype=int))
    
    return deblmap

class DeblendingBox(BaseBox):

    # def __init__(
    #         self,
    #         shape, 
    #         overlap, 
    #         bands, 
    #         index):

        # super().__init__(shape=shape, overlap=overlap, bands=bands, index=index)

        # self.segmap_dict = {band: SegmentationImage(self.data2D[band]["segmap_data"]) for band in self.bands}

        

    def apply_deblending(self, band, deblending_function):

        self.segmap = SegmentationImage(self.data2D[band]["segmap_data"].astype(np.uint16))

        # for band in self.bands:

        deblmap = deblending_function(
            detection_array=self.data2D[band]["detection_array"], 
            segmap=self.segmap)
        
        # deblmap.relabel_consecutive()

        self.add_data(band, "deblmap_data",  deblmap.data)

        return deblmap.max_label
    

class MultiSegmentationImage:

    def __init__(self, segmap_dict):

        self.segmap_dict = segmap_dict

        self.levels = list(segmap_dict.keys())

    def __getitem__(self, level):
        return self.segmap_dict[level]
    
    def match_levels(self, level_1, level_2):

        segmap_2 = self.segmap_dict[level_2]

        segmap_1 = self.segmap_dict[level_1]
         
        self.matched_labels = np.empty(segmap_2.labels.shape)

        for j, _label in enumerate(segmap_2.labels):

            i = np.where(segmap_2.labels == _label)[0][0]
            
            slices = segmap_2.slices[i]

            slice_2 = segmap_2.data[slices]

            slice_1 = segmap_1.data[slices]

            print(np.unique(slice_1[slice_2 == _label]).squeeze(), _label)

            self.matched_labels[j] = float(np.unique(slice_1[slice_2 == _label]).squeeze())

    # def save_multi_segmentation_image(multi_segmap, filename):
    #     with shelve.open(filename) as db:
    #         db['multi_segmap'] = multi_segmap


