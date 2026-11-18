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
import warnings

import numpy as np

from scipy import ndimage

from photutils.segmentation import deblend_sources
from photutils.segmentation import SegmentationImage

# The fast watershed below extends a private class of photutils. It
# is defined only if photutils still provides the pieces it needs;
# otherwise `deblend` falls back on `deblend_sources`.
try:
    from photutils.segmentation.deblend import (
        _DeblendParams,
        _SingleSourceDeblender,
        _get_labels,
    )
    from photutils.segmentation.utils import _make_binary_structure

    _FAST_AVAILABLE = True

except ImportError:  # pragma: no cover
    _SingleSourceDeblender = object
    _FAST_AVAILABLE = False


class LocalWatershedDeblender(_SingleSourceDeblender):
    """
    The photutils deblender of one source, with a faster watershed.

    This is a subclass defined in ATTILA: photutils itself is not
    modified, and `photutils.segmentation.deblend_sources` behaves as
    always. Only the method `apply_watershed` is replaced; the
    thresholds, the markers and the selection criteria are those of
    photutils.

    What photutils does. After the candidate peaks (markers) of a
    source are found, the source is divided among them by a
    watershed. If a child holds less than the fraction `contrast` of
    the flux of the source, the faintest child is removed and the
    watershed is run again, on the whole source, to share its pixels
    among the others; this is repeated until every child passes. A
    large source has many faint candidates, so the whole source is
    flooded 40 to 100 times and the time grows almost as the square
    of its area.

    What changes here. When a child is removed, the pixels of the
    other children are seeds of the next watershed and cannot change
    hands: only the pixels of the removed child are assigned again,
    from the children that touch them. The watershed is therefore
    run only on the bounding box of the removed child, enlarged by
    one pixel to include its neighbours, and the fluxes of the
    children that gain pixels are updated instead of being summed
    again over the source.

    The result is the label map of photutils, pixel by pixel,
    except where the watershed has to break a tie. Pixels with
    exactly the same value are flooded in an order that depends on
    the internal state of the watershed, which is not the same on a
    box and on the whole source; this only happens with repeated
    values, such as those of terraced cutouts. On the 26133 sources
    of the Hydra S mosaic, with terraces of 0.1%, 26131 label maps
    are identical; in the one that was examined, 16 of 580 pixels
    go to the other child. `tests/test_deblending.py` checks the
    identity on images without repeated values. In addition, the
    fluxes are accumulated in a different order, so a child exactly
    at the contrast limit could be treated differently because of
    rounding.
    """

    def apply_watershed(self, markers):
        """
        Divide the source among the markers that pass the contrast.

        Parameters
        ----------
        markers : numpy.ndarray
            The label map of the candidate peaks.

        Returns
        -------
        numpy.ndarray
            The label map of the deblended source. The labels are
            not consecutive if a marker was removed.
        """
        from skimage.segmentation import watershed

        # First watershed, on the whole source, as photutils does.
        markers = watershed(-self.data, markers, mask=self.segment_mask,
                            connectivity=self.footprint)

        labels = _get_labels(markers)

        if labels.size == 1:
            return markers

        fluxes = ndimage.sum_labels(self.data, markers, index=labels)

        flux = dict(zip(labels.tolist(), fluxes.astype(float).tolist()))

        boxes = dict(zip(range(1, int(markers.max()) + 1), ndimage.find_objects(markers)))

        n_y, n_x = markers.shape

        while len(flux) > 1:
            current = np.array(sorted(flux))
            fraction = np.array([flux[label] for label in current]) / self.source_sum

            if not np.any(fraction < self.contrast):
                break

            # Remove only the faintest child, one at a time, because
            # several faint children could combine to pass.
            removed = int(current[np.argmin(fraction)])

            sl_y, sl_x = boxes[removed]

            # Its box plus one pixel: all the neighbours it can be
            # flooded from.
            box = (slice(max(sl_y.start - 1, 0), min(sl_y.stop + 1, n_y)),
                   slice(max(sl_x.start - 1, 0), min(sl_x.stop + 1, n_x)))

            local = markers[box].copy()
            hole = local == removed
            local[hole] = 0

            local = watershed(-self.data[box], local, mask=self.segment_mask[box],
                              connectivity=self.footprint)

            markers[box] = local

            # Give the flux and the extent of the removed child to
            # the children that took its pixels.
            gained = local[hole]
            added = np.bincount(gained, weights=self.data[box][hole].astype(float))

            del flux[removed]
            del boxes[removed]

            for label in np.unique(gained).tolist():
                flux[label] += added[label]

                own_y, own_x = boxes[label]

                boxes[label] = (slice(min(own_y.start, sl_y.start), max(own_y.stop, sl_y.stop)),
                                slice(min(own_x.start, sl_x.start), max(own_x.stop, sl_x.stop)))

        return markers


def _deblend_fast(detection_array, segmap, npixels, nlevels, contrast, mode):
    """
    Deblend a segmentation image with `LocalWatershedDeblender`.

    The function follows the serial branch of
    `photutils.segmentation.deblend_sources` with `relabel=False`
    and `connectivity=8`, and returns the same label map: the
    children of a source receive new labels above the largest label
    of the input, in the order of the input labels; a source that is
    not divided keeps its label.

    Parameters
    ----------
    detection_array : numpy.ndarray
        The 2D array the segmentation was made on.
    segmap : photutils.segmentation.SegmentationImage
        The segmentation image to deblend.
    npixels, nlevels, contrast, mode
        As in `deblend`.

    Returns
    -------
    photutils.segmentation.SegmentationImage
        The deblended segmentation image.
    """
    if contrast == 1:
        return segmap.copy()

    labels = segmap.labels

    # Sources too small to hold two children are not deblended.
    labels = labels[segmap.areas[segmap.get_indices(labels)] >= (npixels * 2)]

    params = _DeblendParams(npixels, _make_binary_structure(detection_array.ndim, 8),
                            nlevels, contrast, mode)

    deblended = segmap.data.copy()
    max_label = segmap.max_label

    for label, index in zip(labels, segmap.get_indices(labels)):
        source_slice = segmap.slices[index]

        children = LocalWatershedDeblender(
            detection_array[source_slice], segmap.data[source_slice], label, params
        ).deblend_source()

        if children is not None:
            mask = children > 0

            deblended[source_slice][mask] = children[mask] + max_label
            max_label += len(_get_labels(children[mask]))

    return SegmentationImage(deblended)


def deblend(
        detection_array,
        segmap,
        npixels=5,
        nlevels=8,
        contrast=1/8.,
        mode='exponential',
        fast=True
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

    By default (`fast=True`) the same algorithm is run by
    `LocalWatershedDeblender`, an ATTILA subclass of the photutils
    deblender that gives the same label map and is 10 to 30 times
    faster on large sources. `fast=False` calls photutils directly.

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
    fast : bool
        If `True` the watershed of `LocalWatershedDeblender` is
        used; if `False`, or if the installed photutils does not
        provide the classes it extends, `deblend_sources` is called.

    Returns
    -------
    photutils.segmentation.SegmentationImage
        The deblended segmentation image, with the shape of
        `detection_array`.
    """

    if fast and not _FAST_AVAILABLE:
        warnings.warn("The fast deblender does not support this version of photutils: "
                      "using photutils.segmentation.deblend_sources.")

    if fast and _FAST_AVAILABLE:
        return _deblend_fast(np.asarray(detection_array), segmap, npixels, nlevels, contrast, mode)

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
