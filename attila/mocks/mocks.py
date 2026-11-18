import numpy as np
import matplotlib.pyplot as plt

import autogalaxy as ag
import autogalaxy.plot as aplt

from astropy.io import fits
from astropy.wcs import WCS

from attila.utils.image_utils import compute_pixel_scale

from astropy import units as u
from astropy.cosmology.realizations import Planck18 as cosmo

from astropy.cosmology import units as cu

import shapely as sh

from shapely.plotting import  plot_polygon, plot_points

from attila.utils.image_utils import compute_pixel_scale

import rpack

import copy

from tqdm import tqdm

import pandas as pd

noise = fits.getdata("/media/edoardo/T7/sunrise/images/PGC7748/PGC7748_g_noise.fits")
hdr = fits.getheader("/media/edoardo/T7/sunrise/images/PGC7748/PGC7748_g_noise.fits")
wcs = WCS(hdr)


z_BGC_PGC7748 = 0.04171

kpc_to_arcsec = cosmo.arcsec_per_kpc_comoving(z_BGC_PGC7748)

arcsec_to_kpc = 1/kpc_to_arcsec

pixel_to_arcsec = 0.2

pixel_to_kpc = arcsec_to_kpc * pixel_to_arcsec


g_zpt = 30.8

Ag_mean = 0.0921

dimming = 2.5 * np.log10((1 + z_BGC_PGC7748)**4)

pix_size_corr = 5 * np.log10(pixel_to_arcsec)


# x_range = [500, 3500]
# y_range = [500, 3500]

n_range = [0.8, 1.5]

Re_range = [0.5, 5]

# mu0_range = [23, 26]

q_range = [0.7, 1]

pa_range = [0, 90]

N_samples = 100


# x_sample = np.random.uniform(x_range[0], x_range[1], N_samples)
# y_sample = np.random.uniform(y_range[0], y_range[1], N_samples)

n_sample = np.random.uniform(n_range[0], n_range[1], N_samples)

Re_kpc_sample = np.random.uniform(Re_range[0], Re_range[1], N_samples)
# mu0_sample = np.random.uniform(mu0_range[0], mu0_range[1], N_samples)

q_sample = np.random.uniform(q_range[0], q_range[1], N_samples)
pa_sample = np.random.uniform(pa_range[0], pa_range[1], N_samples)


Re_sample = Re_kpc_sample / pixel_to_kpc.to_value()


mu0_range = [26, 23.5]

I0_range = 10**((np.asarray(mu0_range) - g_zpt - Ag_mean - dimming - pix_size_corr)/(-2.5))

I0_sample = np.random.uniform(I0_range[0], I0_range[1], N_samples)

mu0_sample = g_zpt + Ag_mean + dimming + pix_size_corr - 2.5*np.log10(I0_sample)

b = 1.9992*n_sample-0.3271

Ie_sample = I0_sample / np.exp(b)


widths = np.maximum(2 * Re_sample * 3 * np.sin(np.radians(pa_sample)), 
                    2 * Re_sample * q_sample * 3 * np.cos(np.radians(pa_sample)))

heights = np.maximum(2 * Re_sample * 3 * np.cos(np.radians(pa_sample)), 
                    2 * Re_sample * q_sample * 3 * np.sin(np.radians(pa_sample)))

sizes = np.concatenate([widths, heights], axis=0)

sizes_tuple = [(int(w), int(h)) for w, h in zip(widths, heights)]

bottom_left_corners = rpack.pack(sizes_tuple)

x, y = np.array([np.array(blc) + 0.5*np.array(size) for blc, size in zip(bottom_left_corners, sizes)]).T

bboxes = [sh.box(x[i] - 0.5*widths[i], y[i] - 0.5*heights[i], x[i] + 0.5*widths[i], y[i] + 0.5*heights[i]) for i in range(len(sizes))]

bounding_shape = rpack.bbox_size(sizes, bottom_left_corners)[::-1]


psf = ag.Kernel2D.from_fits('../PGC007748/2D/PSF_model.fits', hdu=0, pixel_scales=0.2)

median_noise = np.median(noise)

noisemap = np.full(bounding_shape, median_noise)

mock_data = np.zeros(bounding_shape)

simulator = ag.SimulatorImaging(
    exposure_time=1.0, psf=psf, background_sky_level=0.0, add_poisson_noise=False, noise_if_add_noise_false=0.1
)


for i, size, blc in tqdm(zip(range(len(sizes)), sizes, bottom_left_corners), total=len(sizes)):

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

    mock_data[y_min:y_max, xmin:xmax] += dataset.data.native

    noisy_mock_data = np.random.normal(mock_data, noisemap)


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




