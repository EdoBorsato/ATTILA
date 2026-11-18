"""
Segmentation of a detection array into sources.

This module provides `segment`, a thin wrapper around
`photutils.segmentation.detect_sources`, which turns a detection array
(in ATTILA the smoothed g + r signal-to-noise map built by
`attila.pipelines.combine`) into a segmentation map: an integer image
where 0 is the background and every connected group of pixels above the
threshold carries its own label.

In the pipeline `segment` is applied tile by tile through
`attila.tiling.tiles.DetectionTile.apply_detection_function`, and the
per-tile maps are then mosaicked and relabelled in
`attila.pipelines.detect`.
"""
import numbers

import numpy as np

from scipy.ndimage import grey_closing

from photutils.segmentation import detect_sources
from photutils.segmentation import SegmentationImage


def _compute_threshold(arr, arg):
    """
    Resolve the detection threshold to a number.

    The threshold can be given directly as a number, or as a callable
    that derives it from the detection array (for example a function
    returning a multiple of its standard deviation).

    Parameters
    ----------
    arr : numpy.ndarray
        The 2D detection array the threshold applies to.
    arg : float, int or callable
        The threshold, in the units of `arr`, or a callable taking
        `arr` and returning it.

    Returns
    -------
    float
        The threshold in the units of `arr`. A number is returned
        unchanged; for a callable, whatever it returns.

    Raises
    ------
    ValueError
        If `arg` is neither a real number nor a callable.
    """

    if isinstance(arg, numbers.Real):
        # A number is used as the threshold as it is.
        return arg

    # TODO (Nicola): allow the threshold to be given as a numpy.ndarray,
    # i.e. as a per-pixel threshold map.
    elif callable(arg):
        # A callable is evaluated on the array to produce the threshold.
        return arg(arr)
    else:
        # Any other type is rejected.
        raise ValueError("Second argument must be either a float or a callable.")

def segment(detection_array,
                    nconn,
                    threshold,
                    remove_borders=False,
                    mask=None,
                    fill=False):
    """
    Segment a detection array into labelled sources.

    A source is a group of at least `nconn` pixels, connected along
    edges or corners (8-connectivity), whose values are all above the
    threshold. The work is done by
    `photutils.segmentation.detect_sources`; this wrapper adds the
    resolution of the threshold, the optional removal of the sources
    touching the border and the optional closing of small holes.

    When `fill` is True a 3 x 3 grey-scale closing
    (`scipy.ndimage.grey_closing`) is applied to the label image. It
    fills holes and gaps about one pixel wide inside and between the
    segments; because it acts on the label values, a filled pixel takes
    the label given by the closing (a maximum followed by a minimum
    over the 3 x 3 neighbourhood), and pixels on the boundary between
    two touching segments can change label.

    If no source is found `detect_sources` returns `None`, which is
    replaced by an all-zero segmentation image. The border removal is
    executed before this replacement.

    Parameters
    ----------
    detection_array : numpy.ndarray
        The 2D array to segment, with axis order (y, x). In the
        pipeline it is a signal-to-noise map, so the threshold is in
        units of sigma.
    nconn : int
        The minimum number of connected pixels of a source.
    threshold : float, int or callable
        The detection threshold in the units of `detection_array`, or
        a callable that computes it from `detection_array` (see
        `_compute_threshold`).
    remove_borders : bool
        If True, remove the sources with at least one pixel on the
        outermost row or column of the array.
    mask : numpy.ndarray
        A boolean array with the shape of `detection_array` where True
        marks the pixels excluded from the detection. `None` means no
        pixel is excluded.
    fill : bool
        If True, apply the 3 x 3 closing described above to the label
        image.

    Returns
    -------
    photutils.segmentation.SegmentationImage
        The segmentation image, with the shape of `detection_array`:
        0 for the background and a positive integer label for the
        pixels of each source.
    """

    threshold_value = _compute_threshold(detection_array, threshold)

    segmap = detect_sources(
        detection_array,
        threshold=threshold_value,
        npixels=nconn,
        connectivity=8,
        mask=mask
    )

    # `detect_sources` returns None when nothing is detected; use an
    # empty map so that the callers always receive a SegmentationImage.
    if segmap is None:
        segmap = SegmentationImage(np.zeros(detection_array.shape, dtype=int))

    elif remove_borders:
        segmap.remove_border_labels(border_width=1)

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
