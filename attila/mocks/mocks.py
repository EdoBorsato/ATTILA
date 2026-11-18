"""
Script that simulates an image of mock Sersic galaxies.

This file is a script, not a library module: it has no functions, all
its statements run when it is imported or executed, and the paths of
its input files are hard-coded. It was written for the field of
PGC 7748 (z = 0.04171, pixel scale 0.2 arcsec, g band).

The steps are:

1. read the g-band noise map of the field, of which only the median is
   used as the noise level of the mock image;
2. draw `N_samples` random Sersic galaxies, uniformly in Sersic index,
   effective radius (kpc), axis ratio, position angle and central
   intensity (the latter between the counts that correspond to the two
   central surface brightnesses of `mu0_range`);
3. give every galaxy a rectangular box of about 3 effective radii per
   side of the centre and pack the boxes without overlap into a
   compact rectangle with `rpack`, which defines the positions of the
   galaxies and the size of the mock image;
4. simulate each galaxy, convolved with the PSF, with PyAutoGalaxy and
   add it to the mock image inside its box;
5. add Gaussian noise and write the noisy image to `mock_data.npy` and
   the table of the input parameters to `input_parameters.csv`, both
   in the current working directory.

The output is meant as a test image with known input parameters for
the detection and fitting steps of ATTILA.
"""
import numpy as np

import autogalaxy as ag

from astropy.io import fits
from astropy.wcs import WCS

from astropy.cosmology.realizations import Planck18 as cosmo

import shapely as sh

import rpack

from tqdm import tqdm

import pandas as pd

# ----------------------------------------------------------------------
# Input data: the g-band noise (RMS) map of the field and its WCS.
# ----------------------------------------------------------------------
noise = fits.getdata("/media/edoardo/T7/sunrise/images/PGC7748/PGC7748_g_noise.fits")
hdr = fits.getheader("/media/edoardo/T7/sunrise/images/PGC7748/PGC7748_g_noise.fits")
wcs = WCS(hdr)

# ----------------------------------------------------------------------
# Distance and scale conversions at the redshift of the field.
# ----------------------------------------------------------------------
z_BGC_PGC7748 = 0.04171

# Angular size of 1 kpc, from the comoving scale of Planck18.
kpc_to_arcsec = cosmo.arcsec_per_kpc_comoving(z_BGC_PGC7748)

arcsec_to_kpc = 1/kpc_to_arcsec

# Pixel scale in arcsec / pixel.
pixel_to_arcsec = 0.2

# Physical size of a pixel, in kpc / pixel.
pixel_to_kpc = arcsec_to_kpc * pixel_to_arcsec

# ----------------------------------------------------------------------
# Photometric calibration of the g band. These terms relate the counts
# per pixel I to the surface brightness used in this script:
#     mu = g_zpt + Ag_mean + dimming + pix_size_corr - 2.5 log10(I)
# ----------------------------------------------------------------------
# Zero point in magnitudes.
g_zpt = 30.8

# Mean Galactic extinction in the g band, in magnitudes.
Ag_mean = 0.0921

# Cosmological surface-brightness dimming, 10 log10(1 + z) mag.
dimming = 2.5 * np.log10((1 + z_BGC_PGC7748)**4)

# Pixel-area term, 5 log10(pixel scale in arcsec).
pix_size_corr = 5 * np.log10(pixel_to_arcsec)

# ----------------------------------------------------------------------
# Ranges of the uniform distributions of the Sersic parameters.
# ----------------------------------------------------------------------
# x_range = [500, 3500]
# y_range = [500, 3500]

# Sersic index.
n_range = [0.8, 1.5]

# Effective radius in kpc.
Re_range = [0.5, 5]

# mu0_range = [23, 26]

# Axis ratio (minor over major).
q_range = [0.7, 1]

# Position angle in deg.
pa_range = [0, 90]

# Number of mock galaxies.
N_samples = 100

# ----------------------------------------------------------------------
# Random draw of the parameters, one value per galaxy.
# ----------------------------------------------------------------------
# x_sample = np.random.uniform(x_range[0], x_range[1], N_samples)
# y_sample = np.random.uniform(y_range[0], y_range[1], N_samples)

n_sample = np.random.uniform(n_range[0], n_range[1], N_samples)

Re_kpc_sample = np.random.uniform(Re_range[0], Re_range[1], N_samples)
# mu0_sample = np.random.uniform(mu0_range[0], mu0_range[1], N_samples)

q_sample = np.random.uniform(q_range[0], q_range[1], N_samples)
pa_sample = np.random.uniform(pa_range[0], pa_range[1], N_samples)

# Effective radius in pixels.
Re_sample = Re_kpc_sample / pixel_to_kpc.to_value()

# Central surface brightness range in mag / arcsec^2, faint to bright,
# converted to central intensities in counts per pixel by inverting the
# relation given above.
mu0_range = [26, 23.5]

I0_range = 10**((np.asarray(mu0_range) - g_zpt - Ag_mean - dimming - pix_size_corr)/(-2.5))

# The draw is uniform in intensity, hence not uniform in magnitude.
I0_sample = np.random.uniform(I0_range[0], I0_range[1], N_samples)

mu0_sample = g_zpt + Ag_mean + dimming + pix_size_corr - 2.5*np.log10(I0_sample)

# Sersic constant b_n, linear approximation in the Sersic index.
b = 1.9992*n_sample-0.3271

# Intensity at the effective radius, I_e = I_0 exp(-b_n), which is the
# normalisation expected by the Sersic profile of PyAutoGalaxy.
Ie_sample = I0_sample / np.exp(b)

# ----------------------------------------------------------------------
# Box of every galaxy and packing of the boxes. The box sides, in
# pixels, are built from 2 * 3 effective radii along the major axis and
# 2 * 3 * q effective radii along the minor axis, projected with the
# position angle.
# ----------------------------------------------------------------------
widths = np.maximum(2 * Re_sample * 3 * np.sin(np.radians(pa_sample)),
                    2 * Re_sample * q_sample * 3 * np.cos(np.radians(pa_sample)))

heights = np.maximum(2 * Re_sample * 3 * np.cos(np.radians(pa_sample)),
                    2 * Re_sample * q_sample * 3 * np.sin(np.radians(pa_sample)))

# Note: this is a 1D array of length 2 * N_samples, with all the widths
# followed by all the heights.
sizes = np.concatenate([widths, heights], axis=0)

# Integer (width, height) of every box, as required by `rpack`.
sizes_tuple = [(int(w), int(h)) for w, h in zip(widths, heights)]

# (x, y) of the lower left corner of every box in the packed layout.
bottom_left_corners = rpack.pack(sizes_tuple)

# Centre of every box: its corner plus half of the matching element of
# `sizes`.
x, y = np.array([np.array(blc) + 0.5*np.array(size) for blc, size in zip(bottom_left_corners, sizes)]).T

bboxes = [sh.box(x[i] - 0.5*widths[i], y[i] - 0.5*heights[i], x[i] + 0.5*widths[i], y[i] + 0.5*heights[i]) for i in range(len(sizes))]

# Shape of the mock image; `rpack.bbox_size` returns (width, height)
# and the reversal gives the numpy order (n_y, n_x).
bounding_shape = rpack.bbox_size(sizes, bottom_left_corners)[::-1]

# ----------------------------------------------------------------------
# PSF, noise level and empty mock image.
# ----------------------------------------------------------------------
psf = ag.Kernel2D.from_fits('../PGC007748/2D/PSF_model.fits', hdu=0, pixel_scales=0.2)

# A single noise level, the median of the real noise map, is used for
# the whole mock image.
median_noise = np.median(noise)

noisemap = np.full(bounding_shape, median_noise)

mock_data = np.zeros(bounding_shape)

# The simulator convolves the model with the PSF and adds neither sky
# background nor Poisson noise; the noise is added separately below.
simulator = ag.SimulatorImaging(
    exposure_time=1.0, psf=psf, background_sky_level=0.0, add_poisson_noise=False, noise_if_add_noise_false=0.1
)

# ----------------------------------------------------------------------
# Simulation of every galaxy on its own grid and insertion in the mock
# image. The noisy image and the table of the input parameters are
# rebuilt and written to disk at every iteration of the loop.
# ----------------------------------------------------------------------
for i, size, blc in tqdm(zip(range(len(sizes)), sizes, bottom_left_corners), total=len(sizes)):

    # Grid of the box, centred on the galaxy, 0.2 arcsec per pixel.
    grid = ag.Grid2D.uniform(
            shape_native=size,
            pixel_scales=0.2,
        )

    galaxy = ag.Galaxy(
        redshift=z_BGC_PGC7748,
        bulge=ag.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=q_sample[i], angle=pa_sample[i]),
            intensity=Ie_sample[i],
            effective_radius=Re_sample[i]*pixel_to_arcsec.to_value(),
            sersic_index=n_sample[i],
        ),
    )

    # plane = ag.Plane(galaxies=[galaxy])

    dataset = simulator.via_galaxies_from(galaxies=[galaxy], grid=grid)

    xmin, y_min = blc

    xmax, y_max = xmin + size[0], y_min + size[1]

    # Add the simulated image to the region of the box.
    mock_data[y_min:y_max, xmin:xmax] += dataset.data.native

    # Gaussian noise with the median noise level as standard deviation.
    noisy_mock_data = np.random.normal(mock_data, noisemap)

    # Table of the true parameters: box centre (x, y) in pixels of the
    # mock image, effective radius in pixels (`Re`) and in kpc
    # (`Re_kpc`), axis ratio, position angle in deg, Sersic index,
    # intensity at the effective radius in counts per pixel, central
    # surface brightness in mag / arcsec^2 and box polygon.
    reference_table = pd.DataFrame({'x': x,
                                    'y': y,
                                    'Re': Re_sample,
                                    'q': q_sample,
                                    'pa': pa_sample,
                                    'n': n_sample,
                                    'Ie': Ie_sample,
                                    'mu0': mu0_sample,
                                    'Re_kpc': Re_kpc_sample,
                                    'bbox': bboxes})

    reference_table.to_csv(f'input_parameters.csv', index=False)

    np.save('mock_data.npy', noisy_mock_data)

