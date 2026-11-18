"""
Deblending of detected segments into individual sources.

A segment found by `attila.detection.detection.segment` can contain
several overlapping objects. This module provides `deblend`, a wrapper
around `photutils.segmentation.deblend_sources`, which splits such a
segment by multi-thresholding followed by a watershed. In
`attila.pipelines.deblend` it is applied source by source, on the
cutout of the detection array of each source; the result is turned
into new sources by `attila.tiling.sources.Sources.from_deblend`.

The module also contains `DeblendingBox`, the deblending step for the
older `attila.abstract.abstract_things.BaseBox` container, and
`MultiSegmentationImage`, a holder of several segmentation maps of the
same image (for example the detection and the deblended one) with a
method to match their labels.
"""
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
    """
    Deblend the sources of a segmentation image.

    Each segment is re-thresholded at `nlevels` levels between its
    minimum and maximum value in `detection_array`; the local peaks
    found in this way seed a watershed that divides the segment. A
    peak becomes a separate source only if it has at least `npixels`
    connected pixels (8-connectivity) and holds at least a fraction
    `contrast` of the total flux of the parent segment. This is the
    algorithm of `photutils.segmentation.deblend_sources`, which is
    called with `relabel=False`, so the output labels are not made
    consecutive.

    If `deblend_sources` returns `None`, an all-zero segmentation image
    is returned instead.

    Parameters
    ----------
    detection_array : numpy.ndarray
        The 2D array, axis order (y, x), the segmentation was made on.
        In the pipeline it is the smoothed signal-to-noise map.
    segmap : photutils.segmentation.SegmentationImage
        The segmentation image to deblend, with the shape of
        `detection_array`.
    npixels : int
        The minimum number of connected pixels of a deblended source.
    nlevels : int
        The number of thresholding levels.
    contrast : float
        The minimum fraction of the flux of the parent segment, between
        0 and 1, that a local peak needs to be deblended. 0 deblends
        every peak, 1 disables the deblending.
    mode : str
        The spacing of the thresholding levels between the minimum and
        the maximum of the segment: `'exponential'`, `'linear'` or
        `'sinh'`.

    Returns
    -------
    photutils.segmentation.SegmentationImage
        The deblended segmentation image, with the shape of
        `detection_array`.
    """

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
    """
    A box of image data whose segmentation map is deblended.

    The class extends `attila.abstract.abstract_things.BaseBox`, which
    stores, for every band, a dictionary of named 2D arrays in
    `data2D[band][key]`. It expects the `"detection_array"` and
    `"segmap_data"` entries and adds `"deblmap_data"`.

    The pipeline in `attila.pipelines` does not use this class: there
    the deblending is done per source, not per box.

    Attributes
    ----------
    segmap : photutils.segmentation.SegmentationImage
        The segmentation image of the band given to the last call of
        `apply_deblending`, before the deblending.
    """

    # def __init__(
    #         self,
    #         shape,
    #         overlap,
    #         bands,
    #         index):

        # super().__init__(shape=shape, overlap=overlap, bands=bands, index=index)

        # self.segmap_dict = {band: SegmentationImage(self.data2D[band]["segmap_data"]) for band in self.bands}



    def apply_deblending(self, band, deblending_function):
        """
        Deblend the segmentation map of one band and store the result.

        The label image `data2D[band]["segmap_data"]` is cast to
        unsigned 16-bit integers, which limits the labels to 65535,
        wrapped in a `SegmentationImage` and passed to the deblending
        function together with the detection array. The deblended
        labels are stored as `data2D[band]["deblmap_data"]`.

        Parameters
        ----------
        band : str
            The band to deblend.
        deblending_function : callable
            A function called with the keyword arguments
            `detection_array` and `segmap` and returning a
            `photutils.segmentation.SegmentationImage`, typically
            `deblend` with its other arguments fixed by
            `functools.partial`.

        Returns
        -------
        int
            The largest label of the deblended segmentation image.
        """

        self.segmap = SegmentationImage(self.data2D[band]["segmap_data"].astype(np.uint16))

        # for band in self.bands:

        deblmap = deblending_function(
            detection_array=self.data2D[band]["detection_array"],
            segmap=self.segmap)

        # deblmap.relabel_consecutive()

        self.add_data(band, "deblmap_data",  deblmap.data)

        return deblmap.max_label


class MultiSegmentationImage:
    """
    A set of segmentation images of the same field, keyed by level.

    A level is any hashable name that identifies one segmentation of
    the image, for example the detection map and its deblended version.
    The object gives access to the images by level and can match the
    labels of one level to those of another.

    Attributes
    ----------
    segmap_dict : dict
        The `photutils.segmentation.SegmentationImage` of each level.
    levels : list
        The level names, in the order of `segmap_dict`.
    matched_labels : numpy.ndarray
        The result of the last call of `match_levels`; it does not
        exist before.
    """

    def __init__(self, segmap_dict):
        """
        Store the segmentation images and their level names.

        Parameters
        ----------
        segmap_dict : dict
            The `photutils.segmentation.SegmentationImage` of each
            level, keyed by level name. All the images must have the
            same shape and refer to the same pixels.
        """

        self.segmap_dict = segmap_dict

        self.levels = list(segmap_dict.keys())

    def __getitem__(self, level):
        """Return the segmentation image of the given level."""
        return self.segmap_dict[level]

    def match_levels(self, level_1, level_2):
        """
        Match every label of one level to a label of another level.

        For each segment of `level_2`, the labels that `level_1` has on
        the same pixels are collected. The method expects them to be a
        single value, i.e. every segment of `level_2` to lie entirely
        inside one segment of `level_1` (or on its background, label
        0), as it happens when `level_2` is a deblending of `level_1`.
        If a segment covers more than one label of `level_1` the
        conversion to `float` fails.

        The result is stored in `matched_labels`, which has one element
        per label of `level_2`, in the order of its `labels`; nothing
        is returned. Each matched pair is also printed.

        Parameters
        ----------
        level_1
            The level whose labels are looked up (the parent level).
        level_2
            The level whose segments are matched (the child level).
        """

        segmap_2 = self.segmap_dict[level_2]

        segmap_1 = self.segmap_dict[level_1]

        self.matched_labels = np.empty(segmap_2.labels.shape)

        for j, _label in enumerate(segmap_2.labels):

            i = np.where(segmap_2.labels == _label)[0][0]

            # Work on the bounding-box cutout of the segment only.
            slices = segmap_2.slices[i]

            slice_2 = segmap_2.data[slices]

            slice_1 = segmap_1.data[slices]

            print(np.unique(slice_1[slice_2 == _label]).squeeze(), _label)

            self.matched_labels[j] = float(np.unique(slice_1[slice_2 == _label]).squeeze())

    # def save_multi_segmentation_image(multi_segmap, filename):
    #     with shelve.open(filename) as db:
    #         db['multi_segmap'] = multi_segmap


