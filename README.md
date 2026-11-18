# ATTILA

Detection, deblending and fitting of low-surface-brightness galaxies in
wide-field, multi-band imaging.

The package takes a set of co-registered images with their noise maps,
splits the sky into tiles, builds a signal-to-noise detection image,
segments it, measures the detected sources and deblends them. Selected
sources can then be modelled in 2D with AstroPhot and analysed with
isophotes, curves of growth and 1D Sersic fits.

## Processing flow

1. **Field** (`attila.field`): load the images and noise maps of every
   band; attach distance, redshift and zero points.
2. **Tiling** (`attila.tiling.tilingrid`): cover the common footprint
   of the bands with overlapping RA/Dec tiles; cut tiles out and paste
   them back into a mosaic.
3. **Combine** (`attila.pipelines.combine`): per tile, compute
   `(g + r) / sqrt(noise_g^2 + noise_r^2)`, smooth it with a Gaussian
   kernel and mosaic the result into the detection array.
4. **Detect** (`attila.pipelines.detect`): threshold each tile with
   `photutils.segmentation.detect_sources`, merge the tile labels,
   relabel the mosaic and store one `Source` per segment.
5. **Measure** (`attila.properties`): Galactic extinction (SFD map,
   G23 curve, synphot filters), magnitudes, surface brightnesses and
   centroids, written to `properties.csv`.
6. **Deblend** (`attila.pipelines.deblend`): multi-threshold
   deblending of every source with `photutils`, then measure again.
7. **Fit** (`attila.fitting`, `attila.interactive`): AstroPhot Sersic
   models of a target and its neighbours, isophotes on the partial
   residuals, curves of growth and 1D Sersic MCMC.

## Repository layout

| Path | Lines | Content |
| --- | ---: | --- |
| `attila/pipelines.py` | 498 | Template of the three pipeline stages: `combine`, `detect`, `deblend`. |
| `attila/attila_logger.py` | 26 | Package logger (console and `example.log`). |
| `attila/field/field.py` | 485 | `Field`, `Image`, `MultibandImage`. |
| `attila/tiling/tilingrid.py` | 1028 | `Tiling`, grid and sparse tile builders, `guess_shape`. |
| `attila/tiling/tiles.py` | 491 | `BaseTile`, `ConvolutionTile`, `DetectionTile`. |
| `attila/tiling/sources.py` | 1238 | `Source`, `Sources`, one folder per saved source. |
| `attila/tiling/sources_dev.py` | 1369 | Development variant: flat file layout and zstd compression. |
| `attila/abstract/abstract_things.py` | 473 | `AbstractBox`, `BaseBox` (older tile container). |
| `attila/detection/detection.py` | 284 | `segment`, `DetectionBox`. |
| `attila/deblending/deblending.py` | 271 | `deblend`, `DeblendingBox`, `MultiSegmentationImage`. |
| `attila/properties/extinction.py` | 349 | `Extinction`. |
| `attila/properties/properties.py` | 793 | `Properties`. |
| `attila/utils/image_utils.py` | 211 | Unit conversions and small array helpers. |
| `attila/packing/packing.py` | 458 | `Packing`: rectangle packing of source cutouts. |
| `attila/packing/interactive.py` | 463 | `InteractivePlotter`: visual classification. |
| `attila/fitting/deblend_2D.py` | 1879 | AstroPhot wrappers, `AP_deblend`, model editing widgets. |
| `attila/fitting/isophote.py` | 963 | `Ellipse_isophote`, `IsophotesAnalysis`. |
| `attila/fitting/COG.py` | 403 | `COG`, `AperturePhotometry`. |
| `attila/fitting/profiles.py` | 1058 | 1D Sersic likelihoods and emcee fits. |
| `attila/interactive/fitprep.py` | 1068 | `PlotTarget`, `PlotApModel`. |
| `attila/interactive/masking_segment.py` | 835 | Interactive mask and segment editors. |
| `attila/interactive/profile_masking.py` | 164 | Interactive masking of 1D profiles. |
| `attila/mocks/mocks.py` | 280 | Stand-alone script simulating mock Sersic galaxies. |
| `profiling/` | | Profiling harness for the detection pipeline. |
| `mtobjects/` | | Vendored MTObjects (C. Haigh) and comparison scripts in `mtobjects/core/`. |
| `synphot_filters/` | | DECam, SDSS and Gaia filter transmission curves. |
| `NOTES_future_features.md` | | Design notes: HEALPix tiling, model-based deblending. |
| `KNOWN_ISSUES.md` | | Bugs and inconsistencies found by code review. |

Not tracked: `dustmaps/` (SFD maps, 129 MB), `attila.zip` (old
snapshot), FITS / CSV / PNG products under `mtobjects/`, and
`profiling/results/`.

## Requirements

Python 3.12 with `numpy`, `scipy`, `astropy`, `photutils`, `shapely`,
`pandas`, `matplotlib`, `tqdm`, `dustmaps`, `dust_extinction` and
`synphot` for the detection pipeline.

The fitting and interactive modules also need `astrophot`, `emcee`,
`corner`, `ipywidgets`, `rasterio` and `rpack`. `attila/mocks` needs
`autogalaxy`, and `attila/tiling/sources_dev.py` needs `zstandard`.

There is no installer yet: scripts add the repository root to
`sys.path`.

## Running the detection pipeline

The pipeline is driven by a script that lives next to the data and
defines the stages with the parameters of that data set. It expects
`<prefix>_<band>_data.fits` and `<prefix>_<band>_noise.fits`, a saved
`Tiling`, and the `synphot_filters/` and `dustmaps/` folders.
`attila/pipelines.py` is the in-package template of such a script.

## Profiling

`profiling/profile_detection_pipeline.py` imports a driver script and
runs its stages under `cProfile`. It also times the deblending of
individual sources, which the multiprocessing pool hides from the
profiler.

```bash
# Central 6000 x 6000 pixels, a few minutes.
python profiling/profile_detection_pipeline.py --crop 6000 \
    --tile-arcmin 8

# Full mosaic, wall times and memory only.
python profiling/profile_detection_pipeline.py --no-cprofile

# Inspect one stage.
python -m pstats profiling/results/<run>/detect.prof
```

## Documentation style

Comments follow PEP 8 and docstrings follow PEP 257 with numpydoc
sections, wrapped at 72 characters.
