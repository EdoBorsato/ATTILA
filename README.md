# ATTILA

Detection, deblending and fitting of low-surface-brightness galaxies in
wide-field, multi-band imaging.

The package takes a set of co-registered images with their noise maps,
splits the sky into tiles, builds a signal-to-noise detection image,
segments it, measures the detected sources and deblends them. Selected
sources can then be modelled in 2D with AstroPhot and analysed with
isophotes, curves of growth and 1D Sersic fits.

This repository is under active development.

## Processing flow

1. **Field** (`attila.field`): load the images and noise maps of every
   band; attach distance, redshift and zero points.
2. **Tiling**: split the common footprint of the bands into tiles,
   either HEALPix cells (`attila.tiling.healpix`) or overlapping
   RA/Dec rectangles (`attila.tiling.tilingrid`). The data of a tile
   are always cut out on the native pixel grid.
3. **Combine** (`attila.pipelines.combine`): per tile, compute
   `(g + r) / sqrt(noise_g^2 + noise_r^2)` and smooth it with a
   Gaussian kernel. With HEALPix cells this is done when the cell is
   detected, so only the cutouts being worked on are in memory; with
   rectangles the result is written into a full detection array.
4. **Detect** (`attila.pipelines.detect`): segment each tile with
   `photutils.segmentation.detect_sources` and store one `Source` per
   segment. With HEALPix cells a source belongs to the cell that
   contains its peak; with rectangles the tile labels are merged and
   the mosaic is relabelled.
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
| `attila/tiling/healpix.py` | 982 | `HealpixTiling`, `detect_in_cells`: HEALPix tiles, no stitching. |
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
| `profiling/` | | Profiling harness for the detection pipeline. |
| `synphot_filters/` | | DECam, SDSS and Gaia filter transmission curves. |
| `NOTES_future_features.md` | | Design notes: HEALPix tiling, model-based deblending. |
| `CHANGELOG.md` | | What changed in each version. |
| `KNOWN_ISSUES.md` | | Open bugs and inconsistencies, and what was fixed. |

Not tracked: `dustmaps/` (SFD maps, 129 MB), `attila.zip` (old
snapshot), `mtobjects/` (MTObjects comparison, not published yet),
`archive/` (local copies of retired code) and `profiling/results/`.

## Installation

ATTILA needs Python 3.10 or later; it is developed with Python 3.12.
Install it in editable mode from a clone of the repository, so that
the filter curves in `synphot_filters/` are found:

```bash
git clone https://github.com/EdoBorsato/ATTILA.git
cd ATTILA

# Optional: a dedicated environment.
conda create -n attila python=3.12
conda activate attila

pip install -e .                         # detection pipeline
pip install -e ".[fitting,compression]"  # everything
```

The plain install covers detection, deblending and the measurements.
The `fitting` extra adds the AstroPhot models, the isophotes, the 1D
fits and the interactive tools; `compression` adds the zstd
compression of the source catalogues.

The SFD dust maps (about 130 MB) are not in the repository. Download
them once into a `dustmaps/` folder at the repository root:

```bash
python -c "from dustmaps.config import config; \
config['data_dir'] = 'dustmaps'; \
import dustmaps.sfd; dustmaps.sfd.fetch()"
```

This also makes `dustmaps/` the default data folder of the `dustmaps`
package. Maps stored elsewhere can be used by passing
`extinction_maps_folder` to `Extinction`.

Check the installation with:

```bash
python -c "import attila; print(attila.__version__)"
```

## Requirements

The detection pipeline uses `numpy`, `scipy`, `astropy`, `photutils`,
`shapely`, `pandas`, `matplotlib`, `tqdm`, `dustmaps`,
`dust_extinction` and `synphot`.

The fitting and interactive modules also use `astrophot`, `emcee`,
`corner`, `ipywidgets`, `rasterio` and `rpack`; `compress` /
`decompress` in `attila/tiling/sources.py` use `zstandard`. All of
them are installed by the commands above.

## Running the detection pipeline

With HEALPix tiles:

```python
from astropy import units as u

from attila import pipelines
from attila.field.field import Field
from attila.tiling.healpix import HealpixTiling

field = Field({"g": "path/prefix_g", "r": "path/prefix_r"})
field.zpt = {"g": 30.0, "r": 30.0}

# Order 9 cells are 6.9 arcmin wide, order 10 cells 3.4 arcmin.
tiling = HealpixTiling(
    field.multibandimages.property_dict("header"),
    order=9,
    margin_sky=1 * u.arcmin,
)
tiling.make_tiles()

sources = pipelines.deblend(
    *pipelines.detect(*pipelines.combine(field, tiling))
)
```

The catalogue does not depend on the order or on the margin: a source
larger than a cell is detected whole and assigned to the cell of its
peak, stored in the column `healpix_cell`.


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

## Citing ATTILA

If you use ATTILA, please cite the presentation paper:
Borsato et al. 2026, A&A, 714, A223,
[doi:10.1051/0004-6361/202660975](https://doi.org/10.1051/0004-6361/202660975)
([arXiv:2605.21598](https://arxiv.org/abs/2605.21598)). The full
reference is in `CITATION.cff`.

## Licence

BSD 3-Clause, see `LICENSE`.
