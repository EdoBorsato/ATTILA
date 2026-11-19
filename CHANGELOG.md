# Changelog

## 0.2.0 (19 November 2026)

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
- Faster deblending of large sources, without modifying photutils.
  `attila.deblending.deblending.LocalWatershedDeblender` is a
  subclass, defined in ATTILA, of the photutils class that deblends
  one source; it replaces one method, `apply_watershed`.
  - What photutils does: the source is divided among the candidate
    peaks by a watershed; if a child holds less than `contrast` of
    the flux, the faintest child is removed and the watershed is run
    again on the whole source, until every child passes. Large
    sources have many faint candidates, so the whole source is
    flooded 40 to 100 times.
  - What ATTILA does: when a child is removed, the pixels of the
    other children cannot change, so the watershed is run only on
    the box of the removed child. Thresholds, markers and selection
    criteria are those of photutils.
  - Effect on the Hydra S mosaic (26133 sources, 10 processes):
    209 s -> 19 s; the largest source, 191 s -> 17 s. 26131 label
    maps are identical to photutils; the differences are pixels
    with exactly equal values, see `KNOWN_ISSUES.md`.
  - `deblend(..., fast=False)` calls
    `photutils.segmentation.deblend_sources` as before; this is
    also what happens, with a warning, if a future photutils no
    longer provides the class. `tests/test_deblending.py` compares
    the two and should be run after updating photutils.
- `pipelines.deblend` hands the largest sources to the pool first,
  one at a time, and restores the order of the catalogue.
- Measured against 0.1.0 on the Hydra S mosaic (21112 x 14611
  pixels, same machine, the driver's parameters, two deblending
  passes): 2011 s -> 212 s, peak memory 10.3 GB -> 3.5 GB, output
  1612 MB in 246489 files -> 154 MB in 297 files. The catalogues
  have 26127 and 26133 detected sources.
- Streaming deblending for HEALPix tilings (`_deblend_healpix`):
  every packed file of the detection is a task; a worker reads it,
  deblends and measures its sources and writes a packed file of
  the result. No source is sent between processes and the main
  process only collects the tables. Hydra S mosaic: 145 s and
  5.9 GB -> 26 s and 2.0 GB, same table.
- `catalogue.json` (`write_catalogue`, `read_catalogue`) declares
  the packed files of a catalogue, with the cell, the number of
  sources and the largest area of each. The stages hand over this
  file instead of the sources; `Sources.load_light` reads it.
- With HEALPix tiles `detect` no longer returns the sources in
  memory by default (`keep_sources=False`).
- Packed catalogues: the HEALPix detection saves one zstd file per
  cell (`write_pack`, `read_pack`, folder `cells/`) instead of three
  files per source; `Sources.load_light` reads both layouts. Only
  the pixels inside the footprints are stored.
- Terraced detection cutouts (`detect_in_cells(det_step=...)`,
  `terrace_values`): the values stored in the sources are levels
  with a constant relative step (0.1% by default in the pipeline),
  saved exactly as level indices.
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
