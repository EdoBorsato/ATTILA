# Changelog

## 0.2.0 (in development)

- Added `attila.tiling.healpix`: tiles defined as HEALPix cells and
  detection by cell ownership (`detect_in_cells`), with no stitching
  of the tile segmentations. `pipelines.combine` and
  `pipelines.detect` accept a `HealpixTiling`.
- Added `segment(merge_touching=True)`, which merges the segments
  that touch after the closing, as the relabelling of the mosaic did.
- Faster measurements: the image cutouts of the sources are sliced
  directly when the images share the pixel grid of the catalogue
  (`Image.cutout_pixels`, with the WCS built only on request), the
  sky coordinates of the centroids are computed in one call, and
  `Properties.detection_properties(n_workers=...)` measures stacks of
  sources in parallel. The tables are unchanged.
- `detect_in_cells`: sources cut by a cutout are grouped and detected
  again in parallel on a box found in one step (`threshold`), and the
  cells can run in worker processes (`backend='processes'`).
- Streaming HEALPix detection: `combine` returns a
  `LazyDetectionArray` that computes the cutout of a cell when it is
  requested, reading only that part of the images; `detect` segments,
  measures and saves the sources cell by cell and releases the
  cutout. The memory in use scales with the number of workers and
  with the largest source, not with the field. The catalogue is
  unchanged.
- `Field(max_abs_value=1e6)`: data pixels with a larger absolute
  value are set to NaN in memory when the field is loaded (the files
  are not modified); the counts are in `Field.masked_pixels`.
- `detect_in_cells(det_dtype=numpy.float16)` stores the detection
  cutouts of the sources in half precision. Off by default.
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
