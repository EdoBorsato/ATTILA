# Changelog

## 0.2.0 (in development)

- Added `attila.tiling.healpix`: tiles defined as HEALPix cells and
  detection by cell ownership (`detect_in_cells`), with no stitching
  of the tile segmentations. `pipelines.combine` and
  `pipelines.detect` accept a `HealpixTiling`.
- Added `segment(merge_touching=True)`, which merges the segments
  that touch after the closing, as the relabelling of the mosaic did.
- Added `gaussian_smooth`, a separable Gaussian smoothing that
  ignores invalid pixels.

## 0.1.0 (2026-11-18)

First tagged version of ATTILA, the code presented in
Borsato et al. 2026, A&A, 714, A223.

- Documented the whole package: docstrings, `README.md` and
  `KNOWN_ISSUES.md`.
- Added the packaging metadata (`pip install -e .`), the BSD 3-Clause
  licence, `CITATION.cff` and `attila.__version__`.
- Fixed bugs in the computation of the Galactic extinction and in the
  PSF convolution of the 1D profile fits.
- `attila.tiling.sources` replaces `sources_dev`; catalogues stored
  with the older layout are still read.
- General clean-up: unused code removed and several minor fixes.
- Added a profiling harness for the detection pipeline.
