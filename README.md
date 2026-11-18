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
| `attila/pipelines.py` | 470 | Template of the three pipeline stages: `combine`, `detect`, `deblend`. |
| `attila/attila_logger.py` | 26 | Package logger (console and `example.log`). |
| `attila/field/field.py` | 481 | `Field`, `Image`, `MultibandImage`. |
| `attila/tiling/tilingrid.py` | 1015 | `Tiling`, grid and sparse tile builders, `guess_shape`. |
| `attila/tiling/tiles.py` | 473 | `BaseTile`, `ConvolutionTile`, `DetectionTile`. |
| `attila/tiling/sources.py` | 1377 | `Source`, `Sources`, flat file layout, zstd compression. |
| `attila/tiling/sources_dev.py` | 12 | Deprecated alias of `sources.py`, kept for old scripts. |
| `attila/detection/detection.py` | 163 | `segment`. |
| `attila/deblending/deblending.py` | 85 | `deblend`. |
| `attila/properties/extinction.py` | 325 | `Extinction`. |
| `attila/properties/properties.py` | 781 | `Properties`. |
| `attila/utils/image_utils.py` | 205 | Unit conversions and small array helpers. |
| `attila/packing/packing.py` | 457 | `Packing`: rectangle packing of source cutouts. |
| `attila/packing/interactive.py` | 453 | `InteractivePlotter`: visual classification of UDG candidates. |
| `attila/fitting/deblend_2D.py` | 1499 | AstroPhot wrappers, `AP_deblend`, model editing widgets. |
| `attila/fitting/isophote.py` | 867 | `Ellipse_isophote`, `IsophotesAnalysis`. |
| `attila/fitting/COG.py` | 398 | `COG`, `AperturePhotometry`. |
| `attila/fitting/profiles.py` | 649 | 1D Sersic + sky likelihood and emcee fit (`FitProfile`). |
| `attila/interactive/fitprep.py` | 1061 | `PlotTarget`, `PlotApModel`. |
| `attila/interactive/masking_segment.py` | 830 | Interactive mask and segment editors. |
| `attila/interactive/profile_masking.py` | 156 | Interactive masking of 1D profiles. |
| `attila/mocks/mocks.py` | 254 | Stand-alone script simulating mock Sersic galaxies (stale). |
| `profiling/` | | Profiling harness for the detection pipeline. |
| `synphot_filters/` | | DECam, SDSS and Gaia filter transmission curves. |
| `NOTES_future_features.md` | | Design notes: HEALPix tiling, model-based deblending. |
| `KNOWN_ISSUES.md` | | Open bugs and inconsistencies, and what was fixed. |

Not tracked: `dustmaps/` (SFD maps, 129 MB), `attila.zip` (old
snapshot), `mtobjects/` (MTObjects comparison, not published yet),
`archive/` (local copies of retired code) and `profiling/results/`.

## Requirements

Python 3.12 with `numpy`, `scipy`, `astropy`, `photutils`, `shapely`,
`pandas`, `matplotlib`, `tqdm`, `dustmaps`, `dust_extinction` and
`synphot` for the detection pipeline.

The fitting and interactive modules also need `astrophot`, `emcee`,
`corner`, `ipywidgets`, `rasterio` and `rpack`. `attila/mocks` needs
`autogalaxy`, and `compress` / `decompress` in
`attila/tiling/sources.py` need `zstandard`.

There is no installer yet: scripts add the repository root to
`sys.path`.

## Running the detection pipeline

The pipeline is driven by a script that lives next to the data and
defines the stages with the parameters of that data set. It expects
`<prefix>_<band>_data.fits` and `<prefix>_<band>_noise.fits`, a saved
`Tiling`, and the `synphot_filters/` and `dustmaps/` folders (by
default those at the repository root). `attila/pipelines.py` is the
in-package template of such a script:

```python
from attila import pipelines

sources = pipelines.deblend(
    *pipelines.detect(*pipelines.combine(field, tiling))
)
```

Tile shapes and overlaps are sky angles, stored in degrees. Images
are in counts with one zero point per band (`field.zpt`).

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
