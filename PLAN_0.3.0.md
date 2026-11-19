# Plan proposal: ATTILA 0.3.0, model-based deblending with scarlet2

Status: proposal, 18 November 2026. Nothing here is implemented. The
facts about scarlet2 come from its documentation and package pages,
read on that day; what could not be checked is marked "not verified".

## 1. Goal

Replace the two photutils deblending passes, and possibly the
AstroPhot fits, with one scene model per blend, fitted with scarlet2
(the JAX version of scarlet).

Why this removes the two passes: photutils assigns every pixel to
exactly one child, and the answer depends on the `contrast` chosen,
which is why the driver runs it twice (0.1 and 0.01) and keeps both
catalogues. A scene model gives every source its own image, lets
sources overlap, and shares the flux of a pixel among them. There is
one result, and the question "how deep to split" becomes "how many
sources to seed", which can be decided once from the data.

Why it may supersede AstroPhot: the scene model already delivers what
`AP_deblend` is used for, a model of the target and of its neighbours
and the residual image with the neighbours subtracted, for every
source instead of for selected targets. Whether it does so well
enough for low-surface-brightness galaxies is the question the first
milestone answers.

## 2. What scarlet2 is (checked)

| Item | Fact |
| --- | --- |
| Version | 0.5.1, released 3 April 2026; classified as beta |
| Licence | MIT (compatible with BSD-3) |
| Python | 3.10 or later |
| Dependencies | `jax`, `equinox`; `optax`, `numpyro`, `h5py` for fitting, sampling and saving |
| Paper | Melchior et al. 2026, JOSS 11, 9646 |
| Hardware | runs on CPU; can use GPUs |

How a fit is set up:

- **Data.** `Observation(data, weights, psf, wcs, channels)`, with
  arrays of shape (bands, height, width), inverse-variance weights
  and one PSF image per band (`ArrayPSF`: odd size, centred).
- **Model.** A `Scene` on a model frame holds `Source` objects, each
  the product of a spectrum (one amplitude per band) and a
  morphology. The morphology is a free pixel image by default;
  `SersicMorphology`, `GaussianMorphology` and a wavelet
  (`StarletMorphology`) also exist, and `PointSource` uses the PSF.
- **Seeding.** Version 0.5.1 has no source finder
  (`init.hierarchical_sources`, named in the documentation, is not in
  the installed package). `init.from_gaussian_moments` gives a
  spectrum and a Gaussian morphology for a given position.
- **Fit.** `scene.fit(obs, max_iter, e_rel)`, gradient descent with
  an optax optimiser. Only the parameters that are declared are
  fitted, so parts of a model can be held fixed.
- **Outputs.** The image of each source, the rendered model,
  `measure.flux`, `measure.forced_photometry`, approximate errors
  from the diagonal of the Hessian (`infer.uncertainty`) and full
  sampling with NUTS.
- **Several data sets.** Observations with different PSFs, pixel
  scales and WCS are fitted together on a common model frame.
- **Low surface brightness.** The documentation has a worked example
  of an LSB galaxy with compact neighbours, using free pixel models
  and a regulariser (`PairSimilarity`) that discourages sources from
  copying each other. It reports that the LSB model "is still
  picking up some flux from other components" and is "quite grainy".

Not verified, or absent:

- **No sky component.** No background or sky class is documented.
- **No timings.** The documentation gives no run times, memory use or
  size limits. Its examples are about 60 x 50 pixels with 9 sources.
- **Masks** appear to be pixels of zero weight; this is inferred.
- **Neural priors** (`galaxygrad`) exist for 32 and 64 pixel boxes,
  trained on HSC. They are optional and probably unsuitable for
  large diffuse sources.
- The documentation contradicts itself in places (optimiser name,
  default values, type of the position list). The installed code is
  the reference.

## 3. What ATTILA has to provide

- **PSF per band in `Field`.** Today the PSF only enters the AstroPhot
  path. The Hydra S folder has `epsf_1_g.fits` and `epsf_1_r.fits`,
  23 x 23 pixels and centred, which is the format scarlet2 needs.
  One PSF per band and field is the starting point.
- **Noise.** The weights are `1 / noise**2` from the noise maps
  already loaded. Blanked pixels get zero weight.
- **Sky.** A local sky estimate per blend, subtracted before the fit
  or fitted as a flat component. This is the main risk for LSB
  sources and has to be designed, not borrowed.
- **JAX.** Not installed in the `sunrise` environment. The machine
  has 12 cores, 30 GB and no GPU, so the first tests are CPU only.

## 4. Design

The 0.2.0 architecture carries over unchanged: the unit of work is a
packed file of one HEALPix cell, read by a worker, and the result is
another packed file declared in `catalogue.json`.

- **A scene is one detected segment.** ATTILA's detection already
  isolates the blends: every segment, with a margin, is one scene.
  Scenes are independent, so they parallelise as the deblending does
  now.
- **Seeds.** Two candidates to compare: the peaks photutils finds
  (the markers of the current deblender, before the contrast cut),
  and scarlet2's own wavelet footprints inside the segment.
- **Model per source.** Free pixel morphology by default; a point
  source for stars; a Sersic as an option when a parametric result
  is wanted (the AstroPhot use case).
- **What is stored.** Per source: its model image (terraced and
  packed, as the detection cutouts), its spectrum, its box, and the
  parent segment. Per scene: the fit diagnostics (iterations, final
  chi-square).
- **What is measured.** Fluxes per band from the models, with the
  Hessian errors; the current `Properties` run on the data with the
  other sources subtracted, so that the columns of 0.2.0 keep their
  meaning; the fraction of blended flux per source.
- **Footprints.** For tools that need one label per pixel, each pixel
  goes to the source with the largest model flux there. This is
  derived, not the primary product.
- **Downstream.** Isophotes, curves of growth and the 1D Sersic fit
  take the neighbour-subtracted image as they do now from AstroPhot.

Sizes need different treatment:

| Segment size | Share of sources (Hydra S) | Treatment |
| --- | --- | --- |
| Below 10^4 px | 99.3% | one scene, as is |
| 10^4 to 10^5 px | 0.6% | one scene; timing to be measured |
| Above 10^5 px | 26 sources | binned fit, or keep the watershed, to be decided on evidence |

The largest segment has 6.5 million pixels. A free pixel model of
that size, on CPU, is the case most likely not to work.

## 5. Milestones

Each one ends with something that can be checked.

**M0. Feasibility test (go or no-go).**
Install JAX and scarlet2 in a separate environment. Fit about 30
Hydra S segments by hand: isolated sources, typical blends, LSB
candidates from the paper with neighbours, one star, one segment
above 10^5 px. Measure time and memory per scene against size and
number of sources, on CPU.
Exit: a table of timings; side-by-side images of data, model and
residual; a comparison of the LSB fluxes with the AstroPhot fits of
the paper. Decision on whether to continue, and on the sky.

**M1. Inputs.**
PSF per band in `Field`; weights and masks; a function that builds
an `Observation` from a segment. Local sky method chosen from M0.
Exit: tests that the observation of a synthetic segment has the
right shapes, weights and PSF.

**M2. One blend.**
`attila/deblending/scene.py`: build the scene of a segment, seed it,
fit it, return the sources. Seeding method chosen by comparison on
the M0 sample.
Exit: tests on synthetic blends of known fluxes (two Sersic profiles
at several separations and flux ratios, with and without a star);
recovered fluxes within the errors.

**M3. Pipeline stage.**
A streaming stage that reads the detection catalogue and writes a
model catalogue, with the size tiers of section 4. JIT compilation
is shared between scenes of the same shape, so scenes are padded to
a few standard sizes.
Exit: full run on Hydra S; time, memory and disk reported as for
0.2.0.

**M4. Measurements and downstream.**
Catalogue columns from the models; neighbour-subtracted images
feeding isophotes, curves of growth and the 1D fit.
Exit: the fitting example of the paper reproduced without AstroPhot.

**M5. Validation.**
Injection of synthetic LSB galaxies into the real images, next to
bright neighbours and in empty sky: completeness and flux bias
against surface brightness, compared with the 0.2.0 path. Comparison
of the full catalogue with the two photutils passes.
Exit: a short report with those curves; this decides whether
AstroPhot and the photutils passes are removed or kept as options.

**M6. Release.**
Documentation, changelog, `jax` and `scarlet2` as an optional extra,
version 0.3.0.

## 5b. First results of M0 (19 November 2026)

Environment `attila-jax`: scarlet2 0.5.1, JAX 0.12.1, CPU. 29 Hydra S
segments (10 targets of the paper, 6 isolated, 8 blends, 3 of 10^4 to
10^5 px, one of 2 x 10^5 px, one star), g and r, free pixel models,
2000 iterations at most. Scripts, tables and images are in
`profiling/results/m0_scarlet2/`.

Timings:

- One iteration takes 0.4 ms for a 45 px box, 1 ms at 100 px, 4 ms at
  450 px, 10 ms at 500 px with 5 sources, 67 ms at 1150 px with 11
  sources. Memory is 0.5 GB per process, 2.2 GB for the largest.
- Fixed cost per scene, in a process that has already fitted one:
  0.2 s when the scenes are padded to the same size, 0.7 s otherwise.
- A typical scene costs 1 to 3 s. All 26133 segments: about 10 to 15
  hours of one core, so 1 to 2 hours on this machine; a few minutes
  if only the segments with several seeds are fitted.

Quality:

- `init.from_gaussian_moments` gave 11 px model boxes for most
  sources; a free model cannot grow out of its box, and half of the
  flux was lost. Taking the box and the first guess from the
  watershed child of ATTILA fixes it: the low-surface-brightness
  targets are fitted to the noise.
- Free models also fit the noise (chi-square per pixel 0.55 to 0.8)
  and absorb faint undetected neighbours inside their box.
- Bright sources fit badly (chi-square per pixel 10 to 2500): their
  cores hold saturated or corrupted pixels that are not masked.
- Fluxes of the paper's targets against the 1D profile fits. The
  `sma` magnitude of `fitting_results.hdf5` integrates a circular
  Sersic profile with the semi-major-axis radius, so the `circ`
  magnitude is the one to compare with. For 9 targets, scarlet2 minus
  `circ` (g, r), median and scatter:
  - local sky subtracted (median of the undetected pixels of the
    box): +0.65, +0.78 mag, scatter 0.4;
  - no sky subtracted: -0.12, +0.10 mag, scatter 0.6 to 0.7;
  - model boxes twice as large: +0.41, +0.76 mag, scatter 1 to 2.4,
    with two fits that fail.
  The total flux of a free pixel model is set by the sky level and by
  the box, not by the data. Before any pipeline work the test needs a
  sky term fitted with the scene and a constrained or parametric
  model (Sersic) for the faint sources.

## 6. Risks

- **Speed on CPU.** 26133 scenes, each a few hundred iterations.
  Unknown until M0. If too slow: fit only the segments that
  photutils splits or that have more than one seed (about 1000 of
  26133 on Hydra S), and treat the rest as single sources.
- **Sky and diffuse light.** With free pixel models, an extended
  source can absorb sky residuals or the light of a star halo. This
  is the degeneracy described in `NOTES_future_features.md`; the
  global layer (halos, intracluster light) stays out of 0.3.0, so
  the affected regions should be flagged, not trusted.
- **Free models on faint sources.** The documentation's own LSB
  example is noisy and leaks flux. Constraints or a parametric model
  for the faint source may be needed; M0 and M5 measure it.
- **Beta software.** Five releases in two years and inconsistent
  documentation. Pin the version, keep the scarlet2 calls in one
  module, and keep the photutils deblender as fallback, as done for
  the watershed wrapper.
- **Reproducibility.** JAX results can differ between CPU and GPU at
  rounding level; the tests need tolerances, not identity.

## 7. Out of scope for 0.3.0

- The global layer: bright-star halos, intracluster light, large
  scale sky (section 2 of `NOTES_future_features.md`).
- Spatially varying PSF.
- Joint fits of several surveys. The design allows it later, since
  scarlet2 supports it and the HEALPix cells are survey independent.
- HATS output of the catalogues.

## 8. Before starting

0.2.0 should be closed first: release notes, the Zenodo record of
0.1.0, and the comparison with 0.1.0 (plots in
`profiling/results/v010_vs_v020/`).

## 9. Questions

1. Is a GPU available anywhere (institute cluster)? It changes what
   is feasible for the large segments.
2. Which bands should enter the model: g and r only, or more?
3. Are the `epsf_1_*` images the PSFs to use, and is there one set
   per pointing (S, E, SE)?
4. For the paper's targets, should the model be free-form, or do you
   need Sersic parameters from the same fit?
5. Should AstroPhot stay as an option after 0.3.0, whatever M5 says?
6. Is a separate conda environment for JAX acceptable, or must it
   live in `sunrise`?
