

from typing import List, Union
from attila.tiling.sources import Source, Sources
from attila.field.field import Field
import numpy as np
import pandas as pd
from functools import partial
from photutils.centroids import centroid_quadratic

from astropy.wcs import WCS


import astropy.units as u

def _compute_mag(eps, zpt, A, dist):
    return -2.5 * np.log10(eps) + zpt - A - 5 * np.log10(dist.to(u.pc).value) + 5

def _compute_sb(eps, zpt, scale, A, z):
    return -2.5 * np.log10(eps) + zpt + 5 * np.log10(scale) - A - 10 * np.log10(1 + z)

def _compute_area_kpc(area_pixel, scale_kpc):
    return area_pixel * scale_kpc**2

def _compute_area_arcsec(area_pixel, scale_arcsec):
    return area_pixel * scale_arcsec**2

def _area_pixel(area_pixel):
    return area_pixel

def _label(sources: Sources):
    return [source.label for source in sources]

class Properties:

    def __init__(self, sources: Sources, field: Field, bands=None):
        self.sources = sources
        self.field = field

        if bands is None:
            self.bands = field.bands
        else:
            self.bands = bands

        self.zpt = np.zeros(len(self.bands))
        for band in self.bands:
            self.zpt[self.bands.index(band)] = field.zpt[band]

        self.scale_arcsec = np.asarray([self.field.multibandimages[band].pixel_scale.value for band in self.bands])

        self.A = np.zeros((len(self.sources), len(self.bands)))

        self.radius = 10

        self.functions = [self.previous_label, 
                          self.label, 
                          self.area_pixel, 
                          self.area, 
                          self.mag, 
                          self.sb_mean, 
                          self.sb_peak,
                          #self.extinction
                        #   self._sum,
                          self.mag_aper, 
                          self.rel_x_centroid_quad, 
                          self.rel_y_centroid_quad,
                          self.x_centroid_quad, 
                          self.y_centroid_quad,
                          self.ra_centroid_quad,
                          self.dec_centroid_quad
                          
                          ]

    def area(self):
        
        area_pixel = [source.area * u.pix**2 for source in self.sources]

        if hasattr(self.field, 'angular_scale_dict'):
            scale_arcsec_kpc = [self.field.angular_scale_dict[band] for band in self.bands]

            scale_pixel_kpc = [scale_arcsec * scale_arcsec_kpc for scale_arcsec, scale_arcsec_kpc in zip(self.scale_arcsec, scale_arcsec_kpc)]

            return [_compute_area_kpc(area_pixel[i], scale_pixel_kpc[0]).value for i in range(len(area_pixel))]
        
        else:
            return [_compute_area_arcsec(area_pixel[i], self.scale_arcsec[0]).value for i in range(len(area_pixel))]
        
    def area_pixel(self):
        return [source.area for source in self.sources]
    
    def label(self):
        return _label(self.sources)
    
    def previous_label(self):
        return [source.previous_label for source in self.sources]
    
    def extinction(self):
        return self.A
    
    def add_function(self, function):
        raise NotImplementedError('This method is not implemented yet.')
    
        function.__get__(self)

        self.functions += [getattr(self, function.__name__)]
    
    def mag(self):

        if hasattr(self.field, 'distance') is False:
            dist = 1 * u.pc
        else:
            dist = self.field.distance

        # eps = [np.nansum(source.data, axis=(1,2), where=source.segment) for source in self.sources]

        eps = np.zeros((len(self.sources), len(self.bands)))

        for i, source in enumerate(self.sources):
            for j, image in enumerate(source.data):

                eps[i, j] = np.nansum(image.data, where=source.segment)
                
        # eps = [np.nansum(source.data, axis=(1,2)) for source in self.sources]

        mag_segm = np.zeros((len(eps), len(self.bands)))
        for i, _eps, _A in zip(range(len(eps)), eps, self.A):
            mag_segm[i] = _compute_mag(_eps, self.zpt, _A, dist)

        return mag_segm

    def sb_mean(self):

        if hasattr(self.field, 'redshift') is False:
            redshift = 0
        else:
            redshift = self.field.redshift  

        eps = np.zeros((len(self.sources), len(self.bands)))

        for i, source in enumerate(self.sources):
            for j, image in enumerate(source.data):

                eps[i, j] = np.nansum(image.data/source.area, where=source.segment)

        sb_mean = np.zeros((len(eps), len(self.bands)))
        for i, _eps, _A in zip(range(len(eps)), eps, self.A):
            sb_mean[i] = _compute_sb(_eps, self.zpt, self.scale_arcsec, _A, redshift)

        return sb_mean

    def sb_peak(self):

        if hasattr(self.field, 'redshift') is False:
            redshift = 0
        else:
            redshift = self.field.redshift  


        eps = np.zeros((len(self.sources), len(self.bands)))

        for i, source in enumerate(self.sources):
            for j, image in enumerate(source.data):

                eps[i, j] = np.nanmax(image.data * source.segment)

        sb_peak = np.zeros((len(eps), len(self.bands)))
        for i, _eps, _A in zip(range(len(eps)), eps, self.A):
            sb_peak[i] = _compute_sb(_eps, self.zpt, self.scale_arcsec, _A, redshift)

        return sb_peak
    
    ### UNTESTED FUNCTIONS ###

    def _centroid_quad(self):
        """
        Calcola il centroide quadratico 2D dell'oggetto nella cutout image.
        """
        centroids_quad = np.zeros((len(self.sources), len(self.bands), 6))

        for i, source in enumerate(self.sources):

            bbox = source.bbox
            
            for j, image in enumerate(source.data):

                guess = [source.centre[0] - source.bbox.ixmin, source.centre[1] - source.bbox.iymin]

                mask = np.where(source.segment>0, 0, 1).astype(bool) 

                if np.max(source.segment.shape) % 2 == 0:
                    size = np.max(source.segment.shape) + 1
                else:
                    size = np.max(source.segment.shape)

                try:
                    centroids_quad[i, j, :2] = centroid_quadratic(image.data, xpeak=guess[0], ypeak=guess[1], mask=mask, fit_boxsize=size)
                except ValueError:
                    centroids_quad[i, j, :2] = [np.nan, np.nan]

                if np.isnan(centroids_quad[i, j, 0]):
                    centroids_quad[i, j, :2] = guess

                centroids_quad[i, j, 2] = centroids_quad[i, j, 0] + bbox.ixmin
                centroids_quad[i, j, 3] = centroids_quad[i, j, 1] + bbox.iymin

                centroids_quad[i, j, 4] = WCS(self.sources.header).all_pix2world(centroids_quad[i, j, 2], centroids_quad[i, j, 3], 0)[0]
                centroids_quad[i, j, 5] = WCS(self.sources.header).all_pix2world(centroids_quad[i, j, 2], centroids_quad[i, j, 3], 0)[1]

        return centroids_quad
    
    def rel_x_centroid_quad(self):

        if hasattr(self, '_centroids'):
            centroids_quad = self._centroids
        else:
            centroids_quad = self._centroid_quad()

        return centroids_quad[:, :, 0]
    
    def rel_y_centroid_quad(self):

        if hasattr(self, '_centroids'):
            centroids_quad = self._centroids
        else:
            centroids_quad = self._centroid_quad()

        return centroids_quad[:, :, 1]
    
    def x_centroid_quad(self):

        if hasattr(self, '_centroids'):
            centroids_quad = self._centroids
        else:
            centroids_quad = self._centroid_quad()

        return centroids_quad[:, :, 2]

    def y_centroid_quad(self):

        if hasattr(self, '_centroids'):
            centroids_quad = self._centroids
        else:
            centroids_quad = self._centroid_quad()

        return centroids_quad[:, :, 3]
    
    def ra_centroid_quad(self):

        if hasattr(self, '_centroids'):
            centroids_quad = self._centroids
        else:
            centroids_quad = self._centroid_quad()

        return centroids_quad[:, :, 4]
    
    def dec_centroid_quad(self):

        if hasattr(self, '_centroids'):
            centroids_quad = self._centroids
        else:
            centroids_quad = self._centroid_quad()

        return centroids_quad[:, :, 5]


    def mag_aper(self):
        """
        Calcola la magnitudine dell'oggetto usando il centroide calcolato dalla funzione `cutout_centroid_quad`.
        """

        if hasattr(self.field, 'distance') is False:
            dist = 1 * u.pc
        else:
            dist = self.field.distance

        self._centroids = self._centroid_quad()
        
        self._rel_centroids = self._centroids[:, :, :2]

        eps = np.zeros((len(self.sources), len(self.bands))) 
        for i, (source, centroid_bands) in enumerate(zip(self.sources, self._rel_centroids)):
            for j, (image, centroid) in enumerate(zip(source.data, centroid_bands)):

                y, x = np.indices(image.data.shape)

                distance = np.sqrt((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2)  # Calcola la distanza dal centro

                aperture_mask = distance <= self.radius

                eps[i, j] = np.nansum(image.data, where=aperture_mask)

        mag_aper = np.zeros((len(eps), len(self.bands)))
        for i, _eps, _A in zip(range(len(eps)), eps, self.A):
            mag_aper[i] = _compute_mag(_eps, self.zpt, _A, dist)

        return mag_aper  
          
    ### END OF UNTESTED FUNCTIONS ###
    
    def _sum(self):
        eps = np.zeros((len(self.sources), len(self.bands)))

        for i, source in enumerate(self.sources):
            for j, image in enumerate(source.data):

                eps[i, j] = np.nansum(image.data, where=source.segment)

        sum = np.zeros((len(eps), len(self.bands))) 
        for i, _eps, _A in zip(range(len(eps)), eps, self.A):
            sum[i] = _eps

        return sum
    
    def _mean(self):

        eps = [np.nanmean(source.data , axis=(1,2), where=source.segment) for source in self.sources]

        mean = np.zeros((len(eps), len(self.bands))) 
        for i, _eps, _A in zip(range(len(eps)), eps, self.A):
            mean[i] = _eps

        return mean
    
    def _std(self):
        eps = [np.nanstd(source.data , axis=(1,2), where=source.segment) for source in self.sources]

        std = np.zeros((len(eps), len(self.bands))) 
        for i, _eps, _A in zip(range(len(eps)), eps, self.A):
            std[i] = _eps

        return std

    def detection_properties(self):
        
        df = pd.DataFrame()
        for func in self.functions:

            results = np.asarray(func())

            if len(results.shape) == 1:
                df[func.__name__] = results
            else:
                for band, result in zip(self.bands, results.T):
                    df[func.__name__ + '_' + band] = result
        
        self.sources.properties = df
        return self.sources