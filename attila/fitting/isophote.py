import numpy as np
from numpy.typing import ArrayLike

from astropy.io import ascii as asciii

from photutils.isophote import EllipseGeometry, EllipseSample, Isophote, IsophoteList, Ellipse, build_ellipse_model
from photutils.aperture import EllipticalAperture

from astropy.table import vstack, hstack

from typing import List, Dict, Tuple, Union, Optional

from attila import attila_logger
from attila.fitting.COG import COG
from attila.interactive.fitprep import _compute_cps_SB, _plot_data

import matplotlib.pyplot as plt
from matplotlib import colormaps as mcmaps
from matplotlib import colors as mcolors

import shelve

import os

import ipywidgets as widgets
from IPython.display import display

from copy import deepcopy

from functools import partial


def _exclude_bad_isophotes(isolist, bad_stops=[3,5], intensity_constraint=None, eps_max=0.85, save=True):
        """
        It excludes all isophotes with stop_code= 3 or 5; 
        It excludes all isophotes with intens<intensity_constraint
        It excludes all isophotes with eps>eps_max
        """
        isophotes = []

        if intensity_constraint is None:
            intensity_constraint=-np.inf

        for iso in isolist:
            if iso.intens>intensity_constraint:
                if iso.stop_code not in bad_stops:
                    if iso.eps<eps_max:

                        isophotes.append(iso)

        return IsophoteList(isophotes)

def _fit_initial(ellipse_geometry, ellipse):

    i = 0
    while (i < 10):

        ellipse_geometry.sma = 5 + 2*i

        isophote = ellipse.fit_isophote(sma=5 + 2*i)

        ellipse_geometry.x0 = isophote.x0
        ellipse_geometry.y0 = isophote.y0
        ellipse_geometry.sma = 5 + 2*i
        ellipse_geometry.eps = isophote.eps
        ellipse_geometry.pa = isophote.pa

        i += 1

    return ellipse_geometry, isophote


class Ellipse_isophote:

    def __init__(self, deblend2D_results: Dict[str, Dict[str, Union[ArrayLike, EllipseGeometry]]]):

        self.bands = list(deblend2D_results.keys())
        self.deblend2D_results = deblend2D_results

        self.isophotes_dict = {}
        for band in self.bands:
            self.isophotes_dict[band] = IsophoteList([])

        self.start = 1
        self.stop = int(self.deblend2D_results[self.bands[0]]['residuals'].shape[0]/2.) - 2
        self.sma0 = 5
        self.step = 1
        self.linear = True
        self.fix_center = True
        
    def fit_isophotes(self):

        for band in self.bands:

            deblend2D_result = self.deblend2D_results[band]

            data = deblend2D_result['residuals']
            ellipse_geometry = deblend2D_result['ell_geoms']

            self._ellipse = Ellipse(image=data, geometry=ellipse_geometry)

            self._isophote_list = self._ellipse.fit_image(
                sma0=self.sma0,
                minsma=0, 
                maxsma=self.stop, 
                step=self.step, 
                linear=self.linear, 
                fix_center=self.fix_center,
                integrmode='median',
                nclip=10, 
                fflag=0.3
            )

            if self._isophote_list is not None:

                if len(self._isophote_list) > 0:

                    if self._isophote_list[0].sma == 0:
                        self._isophote_list = self._isophote_list[1:]

            self.isophotes_dict[band] = self._isophote_list


    def sample_isophotes(self):

        for band in self.bands:

            _isophote_list = []

            deblend2D_result = self.deblend2D_results[band]

            data = deblend2D_result['model']
            ellipse_geometry = deblend2D_result['ell_geoms']

            sma = self.start #ellipse_geometry.sma

            while True:

                # if (sma > self.stop):
                #     sma, self.step = _iso.sample.geometry.reset_sma(self.step)

                if (sma < self.start) or (sma > self.stop):
                    break

                sample = EllipseSample(
                    data, 
                    sma=sma,
                    x0=ellipse_geometry.x0,
                    y0=ellipse_geometry.y0,
                    eps=ellipse_geometry.eps,
                    position_angle=ellipse_geometry.pa,
                    astep=self.step, 
                    linear_growth=self.linear,
                    integrmode='median'
                )

                try:
                    sample.update(fixed_parameters=[ellipse_geometry.x0, 
                                                    ellipse_geometry.y0, 
                                                    ellipse_geometry.eps, 
                                                    ellipse_geometry.pa])
                    
                    _iso = Isophote(sample, 0, True, 4)

                    _isophote_list.append(_iso)
                    
                except AttributeError:
                    pass
                    
                    # _iso = Isophote(sample, 0, True, 4)

                sma = sample.geometry.update_sma(self.step)
        
            self.isophotes_dict[band] = IsophoteList(_isophote_list)
    
    def sample_isophotes_circ(self):

        for band in self.bands:

            _isophote_list = []

            deblend2D_result = self.deblend2D_results[band]

            data = deblend2D_result['model']
            ellipse_geometry = deblend2D_result['ell_geoms']

            sma = self.start #ellipse_geometry.sma

            while True:

                if (sma < self.start) or (sma > self.stop):
                    break

                sample = EllipseSample(
                    data, 
                    sma=sma,
                    x0=ellipse_geometry.x0,
                    y0=ellipse_geometry.y0,
                    eps=0.0,
                    position_angle=ellipse_geometry.pa,
                    astep=self.step, 
                    linear_growth=self.linear,
                    integrmode='median'
                )

                try:
                    sample.update(fixed_parameters=[ellipse_geometry.x0, 
                                                    ellipse_geometry.y0, 
                                                    ellipse_geometry.eps, 
                                                    ellipse_geometry.pa])
                    
                    _iso = Isophote(sample, 0, True, 4)

                    _isophote_list.append(_iso)
                    
                except AttributeError:
                    pass
                    
                    # _iso = Isophote(sample, 0, True, 4)

                sma = sample.geometry.update_sma(self.step)

            self.isophotes_dict[band] = IsophoteList(_isophote_list)


    
    def cross_sample_isophotes(self, target:str, reference:str):

        deblend2D_result = self.deblend2D_results[target]

        _isophote_list_ = []
        for iso in self.isophotes_dict[reference]:

            geometry = iso.sample.geometry

            sample = EllipseSample(deblend2D_result['residuals'], geometry.sma, geometry=geometry, integrmode='median')

            sample.update()

            iso_ = Isophote(sample, 0, True, 4)
            _isophote_list_.append(iso_)

        self._isophote_list = IsophoteList(_isophote_list_)

        self.isophotes_dict[target] = self._isophote_list

    
    def bmodel(self, noise):

        self.ellipse_models_dict = {}
        self.ellipse_noisymodels_dict = {}
        for band in self.isophotes_dict.keys():
            try:
                self.ellipse_models_dict[band] =  build_ellipse_model(
                    self.deblend2D_results[band]['residuals'].shape, 
                    self.isophotes_dict[band]
                )

                self.ellipse_noisymodels_dict[band] = self.ellipse_models_dict[band] + np.random.normal(
                                scale=noise[band], 
                                size=self.ellipse_models_dict[band].shape
                            )

            except ValueError as e:
                print(f"Error processing band {band}: {e}")

    def monitor_band(self, change):

        new_band = change['new']
        if new_band != self.current_band:

            self.current_band = new_band
            self._plot(fig=self.fig, axs=self.axs)

    def _plot(self, fig, axs):

        sb_func = partial(self.sb_func_field, band=self.current_band)

        [ax.clear() for ax in self.axs]

        noisemap = np.sqrt(self.deblend2D_results[self.current_band]['target'].variance.detach().cpu().numpy())

        _plot_data(
            fig=fig, 
            ax=axs[0],
            data=self.deblend2D_results[self.current_band]['residuals'],
            noise=noisemap,
            sb_func=sb_func,
            sb_upper=25, 
            sb_max=23.5,
        )


        _plot_data(
            fig=fig, 
            ax=axs[1],
            data=self.ellipse_noisymodels_dict[self.current_band],
            noise=noisemap,
            sb_func=sb_func,
            sb_upper=25, 
            sb_max=23.5,
        )

        for i, isophote in enumerate(self.isophotes_dict[self.current_band]):

            if (i % 5 == 0) & (isophote.sma < 40):

                ell = EllipticalAperture(
                    positions=(isophote.x0, isophote.y0),
                    a=isophote.sma,
                    b=isophote.sma * (1 - isophote.eps),
                    theta=isophote.pa
                )

                ell.plot(ax=axs[0], color='C2', lw=1, alpha=0.75)

            elif (i % 10 == 0) & (isophote.sma >= 40):

                ell = EllipticalAperture(
                    positions=(isophote.x0, isophote.y0),
                    a=isophote.sma,
                    b=isophote.sma * (1 - isophote.eps),
                    theta=isophote.pa
                )

                ell.plot(ax=axs[0], color='C2', lw=1, alpha=0.75)

        _plot_data(
            fig=fig, 
            ax=axs[2],
            data=(self.deblend2D_results[self.current_band]['residuals'] - self.ellipse_models_dict[self.current_band]),
            noise=noisemap,
            sb_func=sb_func,
            sb_upper=25, 
            sb_max=23.5,
        )

        self.fig.canvas.draw_idle()

    def plot(self, sb_func_field):

        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))

        self.sb_func_field = sb_func_field

        self.current_band = sorted(self.bands)[0]

        self.dropdown_band = widgets.Dropdown(
            options=sorted(self.bands),
            description='Band:',
            value=self.current_band
        )

        display(self.dropdown_band)

        # self.load_band(self.current_band)

        self.dropdown_band.observe(self.monitor_band, names='value')

        self._plot(fig=self.fig, axs=self.axs)

    def save(self, filename: str):
        with shelve.open(filename) as db:
            for band, isophote_list in self.isophotes_dict.items():
                db[band] = isophote_list


def load_isophotes(filename: str):

    isophote_dict = {}
    with shelve.open(filename) as db:
        for key in db.keys():
            isophote_dict[key] = db[key]

    return isophote_dict

from attila.interactive.profile_masking import InteractiveProfileMasking

def _plot_profile(fig, ax, x, y, y_err):

    y_based = y - np.nanmin(y - y_err)

    sc = ax.scatter(x, y_based, color='C0', s=10, picker=True)

    if y_err is not None:
        ax.errorbar(x, y_based, yerr=y_err, ls='none', color='C0', marker='none', elinewidth=1, capsize=3)

    ax.set_yscale('log')

    ax.set_ylim(ax.get_ylim())


class IsophotesAnalysis:

    def __init__(self, isolist_dict: Dict[str, IsophoteList]):
        self.bands = list(isolist_dict.keys())
        self.isolist_dict = isolist_dict

        self.get_circularized_profile()
        self.get_maj_profile()

        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 4))

        self.band = 'g'

        self.plot_profiles(band=self.band, which='circularized')


    def get_COG(self):

        COG_dict = {}

        for band, isolist in self.isolist_dict.items():
            sma = isolist.sma
            sma_err = 0

            integrated_flux = isolist.tflux_c

            ellipticity = isolist.eps
            ellipticity_err = isolist.ellip_err

            flattening = 1 - ellipticity
            flattening_err = ellipticity_err

            circularized_radius = sma * np.sqrt(flattening)
            circularized_radius_err = np.sqrt(
                    (np.sqrt(flattening) * sma_err) ** 2 +
                    (sma / (2 * np.sqrt(flattening)) * flattening_err) ** 2
                )

            COG_dict[band] = COG(circularized_radius, circularized_radius_err, integrated_flux, 0)

        return COG_dict
    
    def get_circularized_profile(self):

        self.circularized_profile_dict = {}

        for band, isolist in self.isolist_dict.items():
            sma = isolist.sma
            sma_err = 0

            intensity = isolist.intens
            intensity_err = isolist.int_err

            ellipticity = isolist.eps
            ellipticity_err = isolist.ellip_err

            flattening = 1 - ellipticity
            flattening_err = ellipticity_err

            circularized_radius = sma * np.sqrt(flattening)
            circularized_radius_err = np.sqrt(
                    (np.sqrt(flattening) * sma_err) ** 2 +
                    (sma / (2 * np.sqrt(flattening)) * flattening_err) ** 2
                )

            self.circularized_profile_dict[band] = {'radius': circularized_radius, 
                                               'radius_err':circularized_radius_err, 
                                               'intensity': intensity, 
                                               'intensity_err': intensity_err}
    
    def get_maj_profile(self):

        self.maj_profile_dict = {}

        for band, isolist in self.isolist_dict.items():
            sma = isolist.sma
            sma_err = np.zeros_like(sma)


            intensity = isolist.intens
            intensity_err = isolist.int_err

            self.maj_profile_dict[band] = {'sma': sma, 
                                      'sma_err':sma_err, 
                                      'intensity': intensity, 
                                      'intensity_err': intensity_err}

    def apply_mask(self, mask=None):

        self.mask = mask

        if self.mask is None:
            try:
                self.ipm.to_mask()

                self.mask = self.ipm.mask

            except Exception as e:
                raise RuntimeError("An error occurred while applying the mask: " + str(e))

        # if self.circularized_profile_dict is not None:

        #     #self.unmasked_circularized_profile_dict = deepcopy(self.circularized_profile_dict)
        #     #self.unmasked_maj_profile_dict = deepcopy(self.maj_profile_dict)

        #     for band, profile in self.circularized_profile_dict.items():
        #         for key in profile.keys():
                    
        #             self.circularized_profile_dict[band][key] = self.circularized_profile_dict[band][key][~self.mask]
                
        # if self.maj_profile_dict is not None:
        #     for band, profile in self.maj_profile_dict.items():
        #         for key in profile.keys():
                    
        #             self.maj_profile_dict[band][key] = self.maj_profile_dict[band][key][~self.mask]
        # else:
        #     raise ValueError("Both 'circularized_profile_dict' and 'maj_profile_dict' are empty. There is nothing to apply the mask to")



    def plot_profiles(self, band='g', which='circularized'):

        if which == 'circularized':
            x = self.circularized_profile_dict[band]['radius']
            y = self.circularized_profile_dict[band]['intensity']
            y_err = self.circularized_profile_dict[band]['intensity_err']
        elif which == 'maj':
            x = self.maj_profile_dict[band]['sma']
            y = self.maj_profile_dict[band]['intensity']
            y_err = self.maj_profile_dict[band]['intensity_err']

        _plot_profile(self.fig, self.ax, x, y, y_err)

        self.ipm = InteractiveProfileMasking(self.fig, self.ax, x=x, y=y, mask=np.zeros_like(self.isolist_dict[self.band].sma))

        self.fig.canvas.mpl_connect('pick_event', self.ipm.on_click_select)

        


