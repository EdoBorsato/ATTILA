import numpy as np

import shelve

import os

class COG:

    def __init__(self, radius, radius_err, flux, flux_err):

        self.radius = radius
        self.radius_err = radius_err

        self.flux = flux
        self.flux_err = flux_err

        self.normalized_flux = self.flux / np.max(self.flux)
        self.normalized_flux_err = self.flux_err / np.max(self.flux)

    def get_half_light_radius(self):

        diff_from_half = self.normalized_flux - 0.5

        diff_from_half_low = np.abs(diff_from_half[diff_from_half<0])

        diff_from_half_high = np.abs(diff_from_half[diff_from_half>0])

        low_idx = np.argwhere(diff_from_half == -diff_from_half_low.min())

        high_idx = np.argwhere(diff_from_half == diff_from_half_high.min())

        return np.interp(0.5, self.normalized_flux[low_idx:high_idx], self.radius[low_idx:high_idx])
    
    def save(self, path):

        with shelve.open(path) as db:
            db['radius'] = self.radius
            db['radius_err'] = self.radius_err
            db['flux'] = self.flux
            db['flux_err'] = self.flux_err
            db['normalized_flux'] = self.normalized_flux
            db['normalized_flux_err'] = self.normalized_flux_err

    @classmethod
    def load(cls, path):

        with shelve.open(path) as db:
            radius = db['radius']
            radius_err = db['radius_err']
            flux = db['flux']
            flux_err = db['flux_err']

        return cls(radius, radius_err, flux, flux_err)

    def plot(self, ax, normed, **kwargs):

        if 'color' not in kwargs:
            color = None
        else:
            color = kwargs['color']

        if normed:
            ax.plot(self.radius, self.normalized_flux, **kwargs)

            ax.errorbar(
                self.radius, 
                self.normalized_flux, 
                xerr=self.radius_err, 
                yerr=self.normalized_flux_err, 
                color=color,
                marker='none',
                capsize=2, 
                ls='none'
            )

            return ax
        
        if not normed:

            ax.plot(self.radius, self.flux, **kwargs)

            ax.errorbar(
                self.radius, 
                self.flux, 
                xerr=self.radius_err, 
                yerr=self.flux_err, 
                color=color,
                marker='none',
                capsize=2, 
                ls='none'
            )

            return ax
        

from photutils.aperture import aperture_photometry
from typing import List, Dict, Tuple, Union, Optional

from numpy.typing import ArrayLike
from photutils.isophote import EllipseGeometry
from photutils.aperture import CircularAperture

class AperturePhotometry:

    def __init__(self, deblend2D_results: Dict[str, Dict[str, Union[ArrayLike, EllipseGeometry]]]):

        self.bands = list(deblend2D_results.keys())
        self.deblend2D_results = deblend2D_results

        self.start = 1
        self.stop = int(self.deblend2D_results[self.bands[0]]['residuals'].shape[0]/2.) - 2
        self.step = 1

    def get_aperture_photometry(self):

        self.cogs = {}

        radii = np.arange(self.start, self.stop, self.step)

        for band in self.bands:

            deblend2D_result = self.deblend2D_results[band]

            data = deblend2D_result['residuals']
            error = deblend2D_result['error']
            mask = deblend2D_result['mask']

            ellipse_geometry = deblend2D_result['ell_geoms']

            flux = []
            flux_err = []
            for aperture_radius in radii:

                aperture = CircularAperture(
                    positions=(ellipse_geometry.x0, ellipse_geometry.y0),
                    r=aperture_radius,
                )

                phot_table = aperture_photometry(
                    data=data, 
                    apertures=aperture, 
                    error=error, 
                    mask=mask
                )

                flux.append(phot_table['aperture_sum'].value.squeeze())
                flux_err.append(phot_table['aperture_sum_err'].value.squeeze())

            

            flux = np.array(flux).squeeze()
            flux_err = np.array(flux_err).squeeze()

            cog = COG(
                radius=radii, 
                radius_err=0, 
                flux=flux,
                flux_err=flux_err
            )

            self.cogs[band] = cog


    def save(self, path):

        for band, cog in self.cogs.items():

            if not os.path.exists(f'{path}/{band}'):
                os.makedirs(f'{path}/{band}')

            cog.save(f'{path}/{band}/cog')

def load_cogs(path):

    cogs = {}
    for band in os.listdir(path):
        cogs[band] = COG.load(f'{path}/{band}/cog')

    return cogs


    