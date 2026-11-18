"""
Deblending of detected segments into individual sources.

A segment found by `attila.detection.detection.segment` can contain
several overlapping objects. This module provides `deblend`, a wrapper
around `photutils.segmentation.deblend_sources`, which splits such a
segment by multi-thresholding followed by a watershed. In
`attila.pipelines.deblend` it is applied source by source, on the
cutout of the detection array of each source; the result is turned
into new sources by `attila.tiling.sources.Sources.from_deblend`.
"""
import numpy as np

from photutils.segmentation import deblend_sources
from photutils.segmentation import SegmentationImage


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
