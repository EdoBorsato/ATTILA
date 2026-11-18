# Known issues

Open bugs and inconsistencies (November 2026), followed by the list of
what the clean-up commit fixed. Items are located by function name,
because line numbers move. "Checked" means the behaviour was reproduced
with a small test; everything else comes from static reading.

## 1. Open: results

- **The 1D PSF convolution is a cut, not a 2D convolution**
  (`FitProfile`). The Sersic model is mirrored around the centre and
  convolved with a cut through the PSF. It recovers the input
  parameters of a profile blurred the same way (checked), but for
  sources not much larger than the PSF a 2D convolution would differ.
  Fits made before this version were not convolved at all: the
  kernel was ignored.
- **Negative S/N is clipped** (`combine`). The mosaic starts at zero
  and tiles are merged with `np.maximum`, so the detection array has
  no negative values. Probably harmless for detection; it matters for
  anything that uses `det_arr` statistics.
- **Tiles ignore cos(Dec)** (`_make_grid_tiling`). Polygons and grid
  steps are in RA/Dec coordinate degrees, while `Cutout2D` reads the
  tile size as a true sky angle. At Dec = -27 deg a cutout spans
  about 13% more RA than its polygon, so tiles overlap by roughly
  2 arcmin more than `overlap_sky` in RA, and the polygons used for
  the extinction lookup and for `plot_tiles` do not match the
  cutouts. No coverage is lost.
- **`Properties.area` uses the first band's pixel scale** for all
  bands.
- **Counts or counts per second, and the magnitude system**, are not
  recorded: the zero points are taken as given. To be generalised at
  field construction in the next version, together with the
  conversion of weight maps to RMS (`Field(noise_type="weights")`
  stores the weights as noise).
- **Labels of deblended sources** are unique only together with
  `previous_label` (the parent). This is how the files are named;
  the `label` column alone has duplicates.

## 2. Open: code that cannot run as written

- `remove_overlap` (`BaseTile`): slices by `overlap` as pixels while
  `Tiling` passes degrees, and shrinks both axes whatever the
  directions.
- `Tiling.make_tiles` with a list of tile shapes: the setter and
  `_make_sparse_tiling` now accept it, `make_tiles` still refuses it.
- `Image.cutout`: keeps the parent header next to the cutout WCS.
- `combine_sources`: overlapping footprints are summed into labels
  that belong to no source.

## 3. Open: inconsistencies

- **Duplicated code.** `sources_from_radius` and
  `sources_from_centre`; `_plot_data`, `_plot_model`,
  `_plot_residual` and `packing.interactive._plot`;
  `BoundingBox_to_Polygon` and `Polygon_to_BoundingBox` in two
  modules; the default priors in `add_component` and
  `_update_component`.
- **Results dictionary.** `AP_deblend.outputs[band]['model']` is an
  image, `load_results()[band]['model']` is the AstroPhot model.
  `Ellipse_isophote.sample_isophotes` and `AperturePhotometry` need
  the first layout, `Ellipse_isophote._plot` the second.
- **`_combine_labels`** equals `np.maximum` (every branch takes the
  maximum) and is applied through `np.vectorize`.
- **Cosmology.** `Field.compute_physical` reads the distance as
  comoving and uses the angular-diameter (proper) scale, chosen to
  match the NED calculator.
- **Logging.** `attila_logger` calls `logging.basicConfig` at import
  and opens `example.log` in the working directory.
- **Imports.** `isophote.py` imports from `interactive.fitprep`, so
  the fitting code needs ipywidgets and rasterio at import time.
- **Naming.** `to_mag` returns a surface brightness; `ml_sample`
  returns the median; `apply_mask` only stores the mask;
  `Packing.max_axis_ratio` and `BaseTile.working_bands` are unused.
- **Interactive tools.** Undo does not rebuild the polygon in
  `InteractiveSegmentDrawer`. The middle-click branch of
  `InteractiveSegmentSelector` is a no-op. `InteractiveMaskDilator`
  connects a new key handler at every click.
- **Style.** The code lines were not reformatted: `pycodestyle`
  still reports long lines, trailing whitespace and spacing around
  operators.

## 4. Open: performance (next version)

Measured on the central 6000 x 6000 pixels of the Hydra mosaic.

- `detect` + properties on a rectangular tiling: 10 s go into
  `np.vectorize(_combine_labels)`. The per-source WCS constructions
  that took about half of the time were removed in 0.2.0.
- `combine`: the FFT convolution per tile and reading the images from
  the SSD dominate.
- Deblending: a handful of very large segments take almost all the
  time (3 of 150 sampled sources took 72 of 75 s), so the pool is
  limited by its slowest items.

## 5. Open: HEALPix tiling (0.2.0, in development)

- Memory on the Hydra S mosaic (21112 x 14611 pixels, images
  memory-mapped): the detection alone peaks 0.5 GB above the bare
  process with one worker, set by the largest group of sources cut
  by a cutout (a box of 3080 x 3056 pixels); the whole stage with
  measurements peaks at 1.4 GB with one worker and 3.1 GB with 8
  processes.
- Speed on the same field, detection, measurements and saving:
  100 s with one worker, 27 s with 8 processes. The cutouts of the
  cells overlap, so part of the smoothing is computed more than
  once; `_combine_healpix(lazy=False)` computes the array once and
  keeps it in memory. The measurements only run in parallel with
  `backend='processes'`.
- The sources returned by `detect` (`keep_sources=True`) and the
  deblending stage still hold every source in memory.
- Compared with version 0.1.0 on the central 6000 x 6000 pixels of
  the Hydra S mosaic (threshold 3, 25 pixels, sigma 3): 3210 of the
  3212 sources of 0.1.0 are recovered, 98.3% with an identical
  footprint, and their magnitudes agree within 0.01 mag (the
  extinction grids differ). The remaining differences are at the
  image border, where the smoothing is not the same (2 small sources
  only in 0.1.0), and in a corner of the crop that the rectangular
  tiles did not cover (102 sources only in the new catalogue).
- The cell polygons are in (RA, Dec) degrees and are not handled
  across RA = 0.
- Deblending and properties still run per source, not per cell.

## 6. Open: comparison scripts (`mtobjects/core`, not tracked)

- `match_known_sources_to_mto.py`, `check_attila_footprint_and_match.
  py`: `add_prefix` is applied to columns that already carry the
  prefix (`catalog_catalog_name`, `mto_mto_ra_deg`). The default
  matching radii differ (10 and 2.5 arcsec).
- `check_attila_footprint_and_match.py`: the footprint test ignores
  `--pixel-origin` and does not check for blank pixels.
- `run_mto_*.py`: `os.chdir(ROOT)` at import, so relative arguments
  resolve from `mtobjects/`.
- `detection_pipeline.py`: `contrast` only names the output
  folder; the applied value is hard-coded in each `debfunc`.
  `npixels=24/2` passes a float.

## 7. Fixed in version 0.1.0

Results:

- Extinction is now in magnitudes, `A = -2.5 log10(F_ext / F)`
  (checked: `A_g` = 0.302 at E(B-V) = 0.08; it was stored as 0.242).
  Catalogues made before the fix are under-corrected.
- `apply_grid_to_sources` returns exactly one row per source (lowest
  tile index in overlaps, nearest tile outside every polygon).
- Without a distance, magnitudes are apparent (`dist = 10 pc`); they
  were offset by +5.
- `sb_peak` only looks at the pixels of the segment.
- `Source.load_light` keeps `det_arr` as floats.
- `Tiling.add_band` keeps the coverage selection.
- `_check_nan_in_data` keeps the tiles covered by more than 90% in
  every band (it kept the ones badly covered in every band).
- `compute_pixel_scale` returns the mean of the two axes.
- `FitProfile` applies the PSF kernel it is given; `convolve_model=
  False` switches the convolution off, `kernel_step` sets the kernel
  sampling, and `mask` is optional.
- The driver scripts next to the data evaluate the extinction on the
  3 arcmin `extinction_tiling`, as `attila.pipelines.detect` does
  (they used the detection tiles).

Crashes:

- `Sources.from_segmentation_image` (empty segmentation, collection
  rebuilt at every source); `_make_grid_tiling` (tile larger than the
  field); `pipelines.deblend` (unpicklable worker, results dropped,
  nothing returned); `pipelines.detect` (stacked array given to
  `get_data`); `compute_extinction_from_sources`; `Field` with an
  unknown `noise_type`; `MultibandImage.plot_color` overlays;
  `BaseTile.to_folder` / `from_folder`; `Tiling.load_tile`;
  `_make_sparse_tiling`; the list branch of the `tile_shape_sky`
  setter; `Sources.predict`; `Sources.load_light` loading every
  source once per file; `compress` / `decompress` round trip;
  `combine_sources(bbox=None)`; `COG.get_half_light_radius`;
  `FitProfile` default prior; `read_AP_targets` without a PSF;
  `add_component` for a flat sky and shared dictionaries between
  bands; `_plot_profile` without errors; the `None` branches of
  `to_mask` / `to_segment`; right click in `InteractiveMaskDilator`;
  `segment(..., remove_borders=True)` with no detection;
  `_compute_threshold` with a numpy scalar.

Clean-up:

- `sources.py` is the former `sources_dev.py` (flat layout, zstd) and
  still reads the old one-folder-per-source catalogues;
  `sources_dev` is an alias.
- Removed, with a copy in the local `archive/` folder:
  `attila/abstract` (`AbstractBox`, `BaseBox`), `DetectionBox`,
  `DeblendingBox`, `MultiSegmentationImage`,
  `LogProbability_SersicIS`, `FitProfile_noSky`,
  `_exclude_bad_isophotes`, `_fit_initial`, `parse_AP_models`,
  `tiling_types`, and the commented-out block at the end of
  `deblend_2D.py` (`PlotModel`, `InteractiveWindowSelector`).
- Renamed: `tansform_to_SB` / `tansform_from_SB` to `transform_*`,
  `get_paramter_widgets` to `get_parameter_widgets`,
  `exctracted_data` to `extracted_data`.
- 107 unused imports, duplicated imports, the shadowed `shapely`
  import, Italian error messages, duplicated drawing calls, the
  doubled polygon in `InteractiveSegmentDrawer`, the swapped axis
  limits in `fitprep.py`, the profile mask that left out the last
  picked point, output folders created with `os.makedirs`, warnings
  actually silenced in `detect` / `deblend`.
