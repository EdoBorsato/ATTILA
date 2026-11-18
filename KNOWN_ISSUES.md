# Known issues

Bugs and inconsistencies found by reading the code (November 2026).
Nothing here has been fixed: the documentation pass changed comments
and docstrings only. Items are located by function name, because line
numbers move. "Checked" means the behaviour was reproduced with a small
test; everything else comes from static reading.

## 1. Results affected in the production pipeline

- **Extinction is not in magnitudes** (`Extinction.
  compute_extinction_from_tiling`, checked). `Observation.effstim()`
  is called without a unit and returns PHOTLAM, so `abs(mag - mag_ext)`
  is a flux difference of a unit-amplitude flat spectrum, about
  `1 - 10**(-0.4 A)`. With the SDSS g filter:

  | E(B-V) | Stored value | True A_g |
  | ---: | ---: | ---: |
  | 0.05 | 0.159 | 0.189 |
  | 0.08 | 0.242 | 0.302 |
  | 0.20 | 0.498 | 0.753 |

  Every magnitude and surface brightness is under-corrected.
- **Extinction rows can be misassigned** (`Extinction.
  apply_grid_to_sources`, checked on a toy case). `STRtree.query(...,
  predicate='within')` returns one row per (source, tile) pair. With
  overlapping tiles a source in the overlap gets two rows, and a source
  outside every polygon gets none. `Properties` then pairs the rows
  with the sources through `zip`, shifting the values silently.
- **The 3 arcmin extinction tiling is never used** (`detect` in
  `pipelines.py` and in the driver scripts). `extinction_tiling` is
  built, but the extinction is evaluated on the detection tiles
  (17 arcmin in the Hydra run).
- **Apparent magnitudes are offset by 5** (`Properties.mag`,
  `Properties.mag_aper`). Without `field.distance` the code sets
  `dist = 1 pc`, and `m - 5 log10(d / pc) + 5` then gives `m + 5`.
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
  cutouts. No coverage is lost, but the tiling is not what the
  parameters say.
- **`Tiling.add_band` rebuilds the tiles** and drops the coverage
  masks, so any tile selection made in the saved tiling is lost when
  `combine` adds the `gr` band.
- **`sb_peak` uses `nanmax(image * segment)`**, so pixels outside the
  segment count as zero. A segment with only negative counts gives an
  infinite surface brightness.
- **`Properties.area` uses the first band's pixel scale** for all
  bands.
- **`Source.load_light` casts `det_arr` to bool**, so the S/N values
  are lost when sources are reloaded.

## 2. Code that cannot run as written

- `Sources.from_segmentation_image`: `sources = cls(...)` is indented
  inside the loop. It is rebuilt for every source, and an empty
  segmentation raises `UnboundLocalError` (checked).
- `_make_grid_tiling`: a tile larger than the field gives no tiles,
  because `shapely.overlaps` is false when one polygon contains the
  other (checked).
- `pipelines.deblend`: the worker is a local closure, which
  `multiprocessing` cannot pickle (checked). Its results are also
  discarded, `nlevels=10` is a literal, and nothing is returned.
- `pipelines.detect`: `Properties.sb_mean` fails on the stacked
  `ndarray` passed to `get_data` (checked); the driver scripts pass a
  `MultibandImage` instead.
- `Extinction.compute_extinction_from_sources`: `np.zeros(n, (m))`
  passes the shape as dtype, and `self.sources` does not exist.
- `Field.__init__`: a `noise_type` other than `"rms"` or `"weights"`
  leaves `noise` undefined. With `"weights"` the weight map is stored
  as noise without conversion.
- `Image.cutout`: keeps the parent header next to the cutout WCS.
- `MultibandImage.plot_color`: tests the keys `detection` /
  `deblending` but reads `segment` / `deblend` and `self.segment`.
- `BaseTile.from_folder` calls `add_reg`, which only `BaseBox` has.
  `BaseTile.to_folder` reads `reg[band]["wcs"]`, a key that is never
  written. In both `from_folder` versions `hdr` is undefined when the
  first keyword is `segmap`.
- `remove_overlap` (both classes): slices by `overlap` as pixels while
  `Tiling` passes degrees, and shrinks both axes whatever the
  directions.
- `Tiling.load_tile`: `tiling_table.loc["index" == index]` is
  `.loc[False]`.
- `_make_sparse_tiling`: appends `centres` (the whole list); the list
  branch divides a tuple. The list branch of the `tile_shape_sky`
  setter calls `.to` on tuples.
- `_check_nan_in_data`: collects tiles with at most 90% coverage and
  keeps those that are bad in every band, which looks inverted.
- `Sources.predict`: ids and paths are filtered differently, so they
  can misalign. The `sources_dev` version still expects one folder
  per source.
- `sources_dev.Sources.load_light` with `ids=None`: every source is
  loaded once per file (two or three times).
- `sources_dev.compress` / `decompress`: the round trip loses the
  `.npy` / `.json` suffix; `compress` does not create `output_dir`.
- `combine_sources(..., bbox=None)`: the default is dereferenced
  immediately. Overlapping footprints are summed into labels that
  belong to no source.
- `COG.get_half_light_radius`: `np.argwhere` arrays are used as slice
  bounds (`TypeError` on numpy 2, checked), and the slice excludes the
  upper point.
- `FitProfile` / `FitProfile_noSky`: the default prior is overwritten
  with `None` by `super().__init__`; `kernel` is ignored;
  `self.initialize()` does not exist. `FitProfile_noSky` uses the
  five-parameter defaults for a four-parameter likelihood.
- `read_AP_targets`: reads `target.psf.data` even when no PSF was
  given.
- `add_component`: `priors['center']` raises `KeyError` for a flat
  sky; the same dicts are shared between bands.
- `_plot_profile`: uses `y_err` before testing it for `None`.
- `InteractiveMask.to_mask`, `InteractiveSegment.to_segment`,
  `InteractiveProfileMask.to_mask`: the `mask is None` branch calls
  `self.mask.shape`.
- `InteractiveMaskDilator.on_click_dilate`: a right click before any
  arrow key raises `AttributeError`.
- `segment(..., remove_borders=True)`: calls a method on `segmap`
  before the `None` check.
- `attila/mocks/mocks.py`: `sizes` is a flat array of length 2N used
  as N pairs; `pixel_to_arcsec.to_value()` is called on a float; the
  outputs are written inside the loop; the PSF path, the image path
  and `ag.Kernel2D` are stale.

## 3. Inconsistencies

- **Duplicated code.** `BaseBox` and `BaseTile`; `sources.py` and
  `sources_dev.py`; `sources_from_radius` and `sources_from_centre`;
  `FitProfile` and `FitProfile_noSky`; `_plot_data`, `_plot_model`,
  `_plot_residual` and `packing.interactive._plot`;
  `BoundingBox_to_Polygon` and `Polygon_to_BoundingBox` in two
  modules; the default priors in `add_component` and
  `_update_component`; `DetectionBox` and `DetectionTile`.
- **Results dictionary.** `AP_deblend.outputs[band]['model']` is an
  image, `load_results()[band]['model']` is the AstroPhot model.
  `Ellipse_isophote.sample_isophotes` and `AperturePhotometry` need
  the first layout, `Ellipse_isophote._plot` the second.
- **`_combine_labels`.** The comment says "pick the second", the code
  takes the maximum in every branch, so it equals `np.maximum`.
- **Cosmology.** `Field.compute_physical` reads the distance as
  comoving and uses the proper scale; `mocks.py` uses the comoving
  scale. `mocks.py` adds extinction and dimming where
  `Properties._compute_sb` subtracts them.
- **Paths.** `save_light` uses `path.join`; `load_light` and
  `clean_stored_sources` concatenate and need a trailing slash.
  Scripts hard-code `/home/edoardo/...`, `/media/edoardo/...` and
  relative `../synphot_filters/`. `os.mkdir` calls need the parent
  folder to exist.
- **Logging.** `attila_logger` calls `logging.basicConfig` at import
  and opens `example.log` in the working directory.
- **Imports.** `isophote.py` imports from `interactive.fitprep`, so
  the fitting code needs ipywidgets and rasterio at import time.
  There are 113 unused imports, 13 imports below the top of a module,
  and `shapely` is shadowed by `shapely.geometry` in `packing.py`.
- **Annotations.** Several `-> None` functions return objects
  (`get_tile_data`, `plot_tiles`, `load_light`,
  `from_segmentation_image`, `from_deblend`). `compute_pixel_scale`
  returns a `Quantity`, is hinted `float`, and returns the first-axis
  scale instead of the mean (`np.array(a, b)` reads `b` as dtype).
- **Naming.** `tansform_to_SB`, `get_paramter_widgets`,
  `exctracted_data`, "tilees"; `to_mag` returns a surface brightness;
  `ml_sample` returns the median; `apply_mask` only stores the mask;
  `Packing.max_axis_ratio` is unused; the field variable is called
  `PGC007748` in the Hydra scripts.
- **Interactive tools.** `InteractiveSegmentDrawer` appends each
  accepted polygon twice. Undo does not rebuild the polygon. The
  profile mask excludes the last picked point that the plot shows as
  masked. `xlim` / `ylim` use swapped axes in `fitprep.py`. The
  middle-click branch of `InteractiveSegmentSelector` is a no-op.
  `highlight_polygon` draws class 0 twice. `InteractiveMaskDilator`
  connects a new key handler at every click.
- **Unused or dead code.** `_exclude_bad_isophotes`, `_fit_initial`,
  `_plot_residual`, `parse_AP_models`, `AbstractBox`,
  `MultiSegmentationImage` (with a debug `print`), `tiling_types`,
  `working_bands`, the code after `raise` in
  `Properties.add_function`, and about 360 lines of commented-out
  code at the end of `deblend_2D.py`.
- **Warnings.** `with warnings.catch_warnings():` without a filter
  suppresses nothing (`detect`, `deblend`).
- **Style.** `pycodestyle` reports about 2000 findings in `attila/`,
  mostly long lines (555), trailing whitespace (714) and spacing
  around `=` and operators (301). The code was not reformatted.

## 4. Comparison scripts (`mtobjects/core`)

- `match_known_sources_to_mto.py`, `check_attila_footprint_and_match.
  py`: `add_prefix` is applied to columns that already carry the
  prefix (`catalog_catalog_name`, `mto_mto_ra_deg`). The default
  matching radii differ (10 and 2.5 arcsec).
- `check_attila_footprint_and_match.py`: the footprint test ignores
  `--pixel-origin` and does not check for blank pixels.
- `run_mto_*.py`: `os.chdir(ROOT)` at import, so relative arguments
  resolve from `mtobjects/`.
- `detection_pipeline_A_mag_correct.py`: `contrast` only names the
  output folder; the applied value is hard-coded in each `debfunc`.
  `npixels=24/2` passes a float.
