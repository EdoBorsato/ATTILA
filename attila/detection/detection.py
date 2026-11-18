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



def _compute_threshold(arr, arg):

    if isinstance(arg, float) or isinstance(arg, int): 
        # If the second argument is a float, do nothing
       return arg
    
    elif callable(arg): #Nicola: vorrei ci fosse la possibilià di inserire threshold=np.ndarray 
        # If the second argument is callable, call it with arr to produce a float
        return arg(arr)
    else:
        # Handle other cases (e.g., invalid input)
        raise ValueError("Second argument must be either a float or a callable.")

def segment(detection_array, 
                    nconn,
                    threshold,
                    remove_borders=False,
                    mask=None, 
                    fill=False):

    threshold_value = _compute_threshold(detection_array, threshold)

    segmap = detect_sources(
        detection_array, 
        threshold=threshold_value, 
        npixels=nconn, 
        connectivity=8,
        mask=mask
    )

    if remove_borders:
        segmap.remove_border_labels(border_width=1)

    if segmap is None:
        segmap = SegmentationImage(np.zeros(detection_array.shape, dtype=int))

    if fill:
        segmap.data = grey_closing(segmap.data, size=(3, 3))

        # threshmap = np.zeros(detection_array.shape, dtype=int)

    # else:
    #     closed_segmap = grey_closing(segmap.data, size=(4, 4))
    #     # segmap = SegmentationImage(closed_segmap)

    #     threshmap = threshold * np.where(segmap.data>0, 1, 0).astype(np.float32)
    
    return segmap

# def block_background(detection_array, size=100):

#     sc = SigmaClip(sigma_lower=3.0, sigma_upper=3.0, maxiters=5)

#     bkg = biweight_location(sc(detection_array)[~sc(detection_array).mask]) * np.ones(detection_array.shape)

#     return detection_array - bkg, bkg

class DetectionBox(BaseBox):

    # def __init__(self, 
    #              shape, 
    #              overlap, 
    #              bands, 
    #              index):
        
    #     super().__init__(shape, overlap, bands, index,)

    def fix_zeros(self, fill):

        for band in self.bands:
            self.data2D[band]["detection_array"] = np.where(self.data2D[band]["detection_array"]==0, 
                                                            fill, self.data2D[band]["detection_array"])

    # def convolve_over_detection_array(self, kernel):

    #     for band in self.bands:
    #         # self.data2D[band]["detection_array"] = convolve_fft(self.data2D[band]["detection_array"], kernel)
    #         self.data2D[band]["detection_array"] = convolve(self.data2D[band]["detection_array"], kernel)

    #     self.metadata["convolved"] = True

    # def filter_detection_array(self, filter_function):

    #     for band in self.bands:
    #         self.data2D[band]["detection_array"] = filter_function(self.data2D[band]["detection_array"])

    #     self.metadata["filtered"] = True

    # def remove_background(self, background_function):
        
    #     for band in self.bands:
           
    #         # if "bkgmap" not in self.data_keywords:
    #         self.data2D[band]["detection_array"], bkgmap = background_function(self.data2D[band]["detection_array"])

    #         self.add_data(band, "bkgmap",  bkgmap)

    #         # else:
    #         #     self.data2D[band]["detection_array"] -= self.data2D[band]["bkgmap"]
            
    def apply_detection_function(self, band, detection_function):
         
        # new_metadata = self.metadata
        # new_metadata["max_label"] = {}
        self.segmentation_images_dict={}

        # for band in self.bands:
        segmap = detection_function(self.data2D[band]["detection_array"])

        self.add_data(band, "segmap_data",  segmap.data)
        self.segmentation_images_dict[band]=segmap
    
        #     new_metadata["max_label"][band] = int(segmap.max_label)
        # self.metadata = new_metadata

