# Changelog

## 0.1.0 (2026-11-18)

First tagged version: the code used for the Hydra runs, documented
and with the bugs found by a full code review fixed. Performance is
unchanged. The open issues are listed in `KNOWN_ISSUES.md`.

### Results change with respect to earlier, untagged code

- The Galactic extinction is in magnitudes,
  `A = -2.5 log10(F_ext / F)`. It was a flux difference, which
  under-corrected every magnitude and surface brightness (0.242
  instead of `A_g` = 0.302 at E(B-V) = 0.08).
- Every source gets the extinction of one tile; rows could be
  shifted between sources when tiles overlap.
- `attila.pipelines.detect` evaluates the extinction on a 3 arcmin
  tiling instead of the detection tiles.
- Magnitudes are apparent when the field has no distance; they were
  offset by +5.
- `sb_peak` only uses the pixels of the segment.
- `FitProfile` convolves the Sersic model with the PSF kernel it is
  given (`convolve_model=False` switches it off). The kernel was
  ignored before.
- The tile selection by coverage keeps the well-covered tiles; it
  was inverted. `Tiling.add_band` keeps the selection.
- `compute_pixel_scale` returns the mean of the two axes.

### Changed

- `attila.tiling.sources` is the former `sources_dev`: flat file
  layout and zstd compression. Catalogues stored with one folder per
  source are still read. `sources_dev` remains as an alias.
- `pipelines.deblend` returns the deblended sources.
- `FitProfile`: `mask` is optional; new arguments `kernel_step` and
  `convolve_model`.
- Renamed `tansform_to_SB` / `tansform_from_SB` to `transform_*` and
  `get_paramter_widgets` to `get_parameter_widgets`.
- Reloaded sources keep the S/N values of `det_arr`.
- The filter curves and dust maps default to the folders of the
  repository.

### Removed

- `attila.abstract` (`AbstractBox`, `BaseBox`), `DetectionBox`,
  `DeblendingBox`, `MultiSegmentationImage`.
- `LogProbability_SersicIS`, `FitProfile_noSky`.
- `attila.mocks`, which no longer ran.
- Unused helpers and the retired plotting code at the end of
  `deblend_2D.py`.

### Fixed

- About 25 code paths that could not run, in the tiling, the tile
  and source input / output, the curve of growth, the AstroPhot
  wrappers and the interactive tools.

### Added

- Docstrings and comments for the whole package (PEP 257, numpydoc
  sections).
- `README.md`, `KNOWN_ISSUES.md`, a BSD 3-Clause licence, packaging
  metadata (`pip install -e .`) and `attila.__version__`.
- `profiling/profile_detection_pipeline.py`, a profiling harness for
  the detection pipeline.
