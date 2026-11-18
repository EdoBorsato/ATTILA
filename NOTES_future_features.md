# ATTILA — future features (notes, 2026-10-09)

Ideas only: nothing implemented yet. Came out of a discussion during interbars work.

## 1. HATS tiling (Hierarchical Adaptive Tiling Scheme)

Replace the RA/Dec-rectangle tiling (`tiling/tilingrid.py`) with HEALPix cells, keeping
the same interface (`centres`, `polygons`, `get_tile_data`, `tile_to_mosaic`) so the
detection / deblend pipelines run unchanged.

- **Tile = HEALPix cell + margin.** Order 9 is about 6.9', order 10 about 3.4'. Tiles are
  equal-area at any declination, with no RA = 0/360 wrap and no missing cos(dec).
- **Survey-independent tile IDs.** The same cell ID points to LS / VST / DES / Euclid data,
  so multiwavelength combination is a join on cell ID, not a WCS intersection.
- **Footprints as MOCs** (`mocpy`). The cross-band intersection is a MOC intersection,
  correct on the sphere. This replaces the shapely polygons.
- **Ownership by centroid instead of stitching.** Process each cell with its margin and keep
  the sources whose centroid lies inside the cell. No full-image `np.zeros(field.shape)`
  mosaic and no global `ndimage.label`. HATS "margin caches" do exactly this.
- **Native pixels per band.** Cut out each band on its own grid around the cell; only the
  tile *index* is HEALPix. Do NOT reproject to HiPS / HEALPix pixels for fitting, because it
  correlates the noise and degrades the PSF. HiPS is fine for browsing and for the coarse
  diffuse layer.
- **Adaptive splitting.** Go deeper where it's dense. HATS splits by row count; ATTILA could
  split by source density or blend size (small cells in cluster cores, large in empty fields).
- **Output catalogue as HATS Parquet** (`hats-import`). It can then be cross-matched through
  LSDB with Gaia (star halos), LS DR10 Tractor (neighbour models) and spec-z catalogues.
- **Open problem: sources larger than the margin** (big galaxies, BCG). They belong to the
  low-order layer (coarse-grid detection, like the two-scale mask in the halo code).

Libraries: `hats`, `lsdb`, `hats-import`, `mocpy`, `cdshealpix` / `astropy-healpix`.

## 2. Model-based deblending + diffuse light (ICL / star halos)

Multi-scale scene model whose levels map onto the HEALPix orders:

1. **Global layer (low order, coarse / quadtree-binned data):**
   - sky (very low order);
   - bright-star halos: extended PSF (core from stacked field stars, wings from saturated
     stars) times one linear amplitude per star, anchored by its Gaia magnitude. Ghosts /
     reflections (VST) are extra templates. Check first whether LS DR10 already subtracts them;
   - ICL: smooth, restricted linear basis (B-splines / bilinear patches on the quadtree in
     place of griddata), with a smoothness regulariser.
   Keep halo removal separate from sky.
2. **Local layer (high order, footprint-only, GPU-batched):** MGE sources (PSF as a Gaussian
   mixture means analytic convolution and analytic erf integrals over binned pixels), sparse
   NNLS for the fluxes, model-weighted flux redistribution for attribution (scarlet-style).
   Scarlet / free-form morphology only for blends flagged by coherent residuals.
3. **Alternate 1 <-> 2 (backfitting)**, or a joint Schur / Woodbury solve for the linear parts.
   The current halo code (sunrise_core_testing/Hydra on T7: `halo_model_tile.py`) is already
   this loop with masks standing in for the sources. Upgrade: replace the masks with source
   models, so mask dilation stops deciding how much galaxy outskirt the halo model absorbs.

**Caveat: identifiability.** The ICL / galaxy outskirts / sky split is degenerate, and under
backfitting whatever is fitted first takes the ambiguous light. Restrict the global basis
(no power on galaxy scales), give galaxies explicit outskirts, and calibrate the split on
mocks with a penalty (as for the smooth/mesh lambda in interbars). Never fit that penalty
to the data.

Uncertainties: Laplace / Hessian at the optimum (JAX). Nested sampling does not scale here.

## 3. Bugs noticed while reading (not fixed)

- `tiling/tilingrid.py` `_check_nan_in_data`: looks inverted. It collects tiles with <= 90%
  coverage, then KEEPS those that are bad in every band.
- Overlap units: `Tiling` passes `overlap_sky` in **degrees** to the tile; `int(self.overlap)`
  then gives 0 in the metadata, and `remove_overlap` slices by it as if it were pixels.
- `_make_sparse_tiling`: appends `centres` (the whole list) instead of `centre`.
- `_combine_labels`: the comment says "pick the second", the code picks the max.
- `Tiling.load_tile`: `tiling_table.loc["index" == index]` evaluates to `.loc[False]`.
- `mocks/mocks.py`: `ag.Kernel2D` no longer exists in the current PyAutoGalaxy stack;
  `pipelines.py` hardcodes relative paths (`../synphot_filters/`, `../dustmaps/`).
