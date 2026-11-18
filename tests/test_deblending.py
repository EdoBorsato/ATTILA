"""
Check the fast deblender of ATTILA against photutils.

`attila.deblending.deblending.LocalWatershedDeblender` must return
the same label map as `photutils.segmentation.deblend_sources`. Run
with `pytest tests/test_deblending.py` after updating photutils.
"""
import numpy as np
import pytest

from photutils.segmentation import SegmentationImage, detect_sources

from attila.deblending.deblending import deblend
from attila.utils.image_utils import gaussian_smooth


def _blended_field(seed, n_sources, size=300):
    """Return a smoothed noisy image of overlapping Gaussian sources."""
    rng = np.random.default_rng(seed)

    y, x = np.mgrid[:size, :size]

    image = rng.normal(0.0, 1.0, (size, size))

    for _ in range(n_sources):
        x0, y0 = rng.uniform(40, size - 40, 2)
        sigma = rng.uniform(3, 15)
        amplitude = 10 ** rng.uniform(0.5, 2.5)

        image += amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    return gaussian_smooth(image, 2.0).astype(np.float32)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("contrast", [1 / 8., 0.01, 0.001])
@pytest.mark.parametrize("mode", ["exponential", "linear"])
def test_same_label_map_as_photutils(seed, contrast, mode):
    """The fast deblender reproduces photutils pixel by pixel."""
    image = _blended_field(seed, n_sources=40)

    segmap = detect_sources(image, threshold=1.0, npixels=25, connectivity=8)

    for label in segmap.labels[np.argsort(-segmap.areas)][:5]:
        box = segmap.slices[segmap.get_index(label)]
        single = SegmentationImage((segmap.data[box] == label).astype(int) * int(label))

        kwargs = dict(npixels=5, nlevels=10, contrast=contrast, mode=mode)

        reference = deblend(image[box], single, fast=False, **kwargs)
        fast = deblend(image[box], single, fast=True, **kwargs)

        assert np.array_equal(reference.data, fast.data)


def test_whole_segmentation_image():
    """A segmentation image with many sources gives the same labels."""
    image = _blended_field(11, n_sources=60, size=400)

    segmap = detect_sources(image, threshold=1.0, npixels=25, connectivity=8)

    reference = deblend(image, segmap, nlevels=10, contrast=0.01, fast=False)
    fast = deblend(image, segmap, nlevels=10, contrast=0.01, fast=True)

    assert reference.nlabels > segmap.nlabels
    assert np.array_equal(reference.data, fast.data)
