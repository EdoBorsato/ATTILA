import astrophot as ap

import numpy as np

from photutils.segmentation import SegmentationImage
from photutils.isophote.geometry import EllipseGeometry
from photutils.aperture import EllipticalAperture

from attila.field.field import Field
from attila.tiling.sources import Source, Sources

import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon

from astropy.io import fits

from typing import List, Dict, Tuple

import pandas as pd

from matplotlib import colormaps as mcmaps
from matplotlib import colors as mcolors

import os
import json

"""
Astrophot specific function
"""


def AP_initialize(source:Source, field:Field, psf_path:str=None):

    if not hasattr(source, 'data'):
        raise AttributeError('Data attribute not found in source object')
    
    targets_dict = {}
    for image in source.data:

        if psf_path is not None:

            oversampling = float(psf_path.split('_')[1])

            psf_data = np.abs(fits.getdata(psf_path + f'_{image.band}.fits'))

            psf_target_guess = ap.image.PSF_Image(
                data=psf_data.astype(float),
                pixelscale=image.pixel_scale.value/oversampling,
            )

        else:
            psf_target_guess = None

        targets_dict[image.band] = ap.image.Target_Image(
            data=image.data.astype(float),
            pixelscale=image.pixel_scale.value,
            zeropoint=field.zpt[image.band],
            variance=(image.noise)**2,  
            mask=source.mask,
            psf=psf_target_guess
        )
    return targets_dict

def read_AP_targets(targets_dict):

    inputs_dict = {}

    for band, target in targets_dict.items():
        inputs_dict[band] = {}
        inputs_dict[band]['data'] = target.data.detach().cpu().numpy()
        inputs_dict[band]['noise'] = np.sqrt(target.variance.detach().cpu().numpy())
        inputs_dict[band]['mask'] = target.mask.detach().cpu().numpy()
        inputs_dict[band]['psf'] = target.psf.data.detach().cpu().numpy()

    return inputs_dict


def AP_set_models(source:Source, targets_dict:Dict[str, ap.image.Target_Image]):

    target_name = f"object {source.label:02d}"

    models_dict = {}
    for image in source.data:

        segment_img = SegmentationImage(source.segment.astype(int))
        
        windows = ap.utils.initialize.windows_from_segmentation_map(segment_img)
        windows = ap.utils.initialize.scale_windows(
            windows, image_shape=image.shape, expand_scale=1, expand_border=5
        )

        centers = ap.utils.initialize.centroids_from_segmentation_map(segment_img, image.data)
        PAs = ap.utils.initialize.PA_from_segmentation_map(segment_img, image.data, centers)
        qs = ap.utils.initialize.q_from_segmentation_map(segment_img, image.data, centers, PAs)

        if targets_dict[image.band].has_psf:
            psf_mode = 'full'

        else:
            psf_mode = 'none'

        seg_models = []
        for win in windows:
            seg_models.append(
                ap.models.AstroPhot_Model(
                    name=f"object {win:02d}",
                    window=windows[win],
                    model_type="sersic galaxy model",
                    target=targets_dict[image.band],
                    parameters={
                        "center": np.array(centers[win]) * image.pixel_scale.value,
                        "PA": PAs[win],
                        "q": qs[win],
                    },
                    psf_mode=psf_mode
                )
            )

        sky = ap.models.AstroPhot_Model(
            name=f"sky level",
            model_type="flat sky model",
            target=targets_dict[image.band],
        )

        groupmodel = ap.models.AstroPhot_Model(
            name="group", models=[sky] + seg_models, target=targets_dict[image.band], model_type="group model"
        )

        # groupmodel.initialize()

        models_dict[image.band] = groupmodel

    return models_dict, target_name

def parse_AP_models(models_dict:Dict[str, ap.models.AstroPhot_Model]):
    raise NotImplementedError

def AP_fit_models(models_dict:Dict[str, ap.models.AstroPhot_Model]):

    results_dict = {}
    for band, groupmodel in models_dict.items():
    
        groupmodel.initialize()

        results_dict[band] = ap.fit.Iter(groupmodel, verbose=1).fit()

    return models_dict, results_dict

def AP_get_residuals(
        models_dict:Dict[str, ap.models.AstroPhot_Model], 
        results_dict,
        targets_dict:Dict[str, ap.image.Target_Image], 
        target_name='object 01'):
    
    resi_partial_dict = {}
    model_partial_dict = {}
    resi_total_dict = {}
    ell_geoms_dict = {}

    for band, groupmodel in models_dict.items():

        target = targets_dict[band]
        result = results_dict[band]
        
        resi_partial = target.copy()
        resi_total = target.copy()

        modelimage = target.model_image()

        for model in groupmodel.models.values():
            if model.name == target_name:
                continue

            modelimage_partial = model(image=modelimage)

        resi_partial -= modelimage_partial
        
        modelimage = target.model_image()

        modelimage_total = groupmodel(image=modelimage)

        resi_total -= modelimage_total

        resi_partial_dict[band] = resi_partial.data.detach().cpu().numpy()
        
        modelimage = target.model_image()

        modelimage_target = groupmodel.models[target_name](image=modelimage)

        model_partial_dict[band] = modelimage_target.data.detach().cpu().numpy()
        resi_total_dict[band] = resi_total.data.detach().cpu().numpy()

        q = result.model.parameters[target_name]['q'].value.detach().cpu().numpy()
        PA = result.model.parameters[target_name]['PA'].value.detach().cpu().numpy()

        centre = target.plane_to_pixel(result.model.parameters[target_name]['center'].value).detach().cpu().numpy()

        re = (result.model.parameters[target_name]['Re'].value / target.pixel_length).detach().cpu().numpy()

        ell_geoms_dict[band] = EllipseGeometry(
            x0=float(centre[0]), 
            y0=float((centre[1])), 
            sma=float(re), 
            eps=1-float(q), 
            pa=float(PA+np.pi/2)
        )

    return resi_partial_dict, resi_total_dict, model_partial_dict, ell_geoms_dict



def load_results(filepath):
        
    bands = []
    for _file in os.listdir(filepath):
        if os.path.isdir(os.path.join(filepath, _file)):
            bands.append(_file)

    outputs = {}
    for band in bands:

        outputs[band] = {}
        # outputs[band]['data'] = np.load(os.path.join(filepath, band, 'data.npy'))
        outputs[band]['residuals'] = np.load(os.path.join(filepath, band, 'residuals.npy'))
        outputs[band]['mask'] = np.load(os.path.join(filepath, band, 'mask.npy'))
        outputs[band]['model_image'] = np.load(os.path.join(filepath, band, 'model.npy'))
        # outputs[band]['error'] = np.load(os.path.join(filepath, band, 'error.npy'))
        # outputs[band]['psf'] = np.load(os.path.join(filepath, band, 'psf.npy'))

        outputs[band]['model'] = ap.models.AstroPhot_Model(filename=os.path.join(filepath, band, 'AstroPhot.yaml'))

        outputs[band]['target'] = ap.image.Target_Image(filename=os.path.join(filepath, band, 'target.fits'))

        with open(os.path.join(filepath, band, 'ell_geoms.json'), 'r') as file:
            geometry_dict = json.load(file)

        outputs[band]['ell_geoms'] = EllipseGeometry(**geometry_dict)

    return outputs

class AP_deblend:

    def __init__(self, target_dicts):

        # self.source = source
        # self.field = field
        self.code = "astrophot"

        self._target_dicts = target_dicts 

        self.inputs = read_AP_targets(self._target_dicts)
        

    def set_models(self, models_dict:Dict[str, ap.models.AstroPhot_Model], target_name:str='object 01'):
        self._models_dict, self._target_name = models_dict, target_name

    def apply_prior_fit(self, prior_models_dict:Dict[str, ap.models.AstroPhot_Model]):

        for band in prior_models_dict.keys():
            prior_models = prior_models_dict[band].models
            new_models = self._models_dict[band].models

            for name in prior_models.keys():
                if name in new_models.keys():
                    
                    for par_name in prior_models[name].parameters.names:
                        if par_name != 'center:0' and par_name != 'center:1':
                            new_models[name].parameters[par_name].value = prior_models[name].parameters[par_name].value
                    

    def fit(self): 
        self._models_dict, self._results_dict = AP_fit_models(models_dict=self._models_dict)

    def output_results(self):

        self._resi_partial_dict, self._resi_total_dict, self._model_partial_dict, self._ell_geoms_dict = AP_get_residuals(
            models_dict=self._models_dict, 
            results_dict=self._results_dict, 
            targets_dict=self._target_dicts, 
            target_name=self._target_name
        )

        self.outputs = {}
        for band in self._results_dict.keys():

            # sky = self._models_dict[band]['sky']

            self.outputs[band] = {
                'data': self.inputs[band]['data'],
                'residuals': self._resi_partial_dict[band],
                'mask': self.inputs[band]['mask'],
                'model': self._model_partial_dict[band],
                'error': self.inputs[band]['noise'],
                'ell_geoms': self._ell_geoms_dict[band], 
                'psf': self.inputs[band]['psf']
            }

    def save_results(self, filepath, plots=True):

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for band in self.outputs.keys():

            if not os.path.exists(os.path.join(filepath, band)):
                os.makedirs(os.path.join(filepath, band))

            self._models_dict[band].save(filename=os.path.join(filepath, band, 'AstroPhot.yaml'))

            self._target_dicts[band].save(filename=os.path.join(filepath, band, 'target.fits'))

            np.save(os.path.join(filepath, band, 'data.npy'), self.outputs[band]['data'])
            np.save(os.path.join(filepath, band, 'residuals.npy'), self.outputs[band]['residuals'])
            np.save(os.path.join(filepath, band, 'mask.npy'), self.outputs[band]['mask'])
            np.save(os.path.join(filepath, band, 'model.npy'), self.outputs[band]['model'])
            np.save(os.path.join(filepath, band, 'error.npy'), self.outputs[band]['error'])
            np.save(os.path.join(filepath, band, 'psf.npy'), self.outputs[band]['psf'])

            geometry_dict = {
                'x0': self.outputs[band]['ell_geoms'].x0, 
                'y0': self.outputs[band]['ell_geoms'].y0,
                'sma': self.outputs[band]['ell_geoms'].sma, 
                'eps': self.outputs[band]['ell_geoms'].eps, 
                'pa': self.outputs[band]['ell_geoms'].pa
            }

            with open(os.path.join(filepath, band, 'ell_geoms.json'), 'w+') as file:
                json.dump(geometry_dict, file, indent=4)

        if plots:

            fig_inp, axs_inp = plt.subplots(1, 2, figsize=(10, 5))

            self.plot_inputs(axs_inp)

            fig_inp.savefig(os.path.join(filepath, 'inputs.jpg'))


            fig_out, axs_out = plt.subplots(1, 2, figsize=(10, 5))

            self.plot_outputs(axs_out)

            fig_out.savefig(os.path.join(filepath, 'outputs.jpg'))

    




    def plot_inputs(self, axs:List[plt.Axes]):

        bottom = mcmaps['binary']
        top_parent = mcmaps['viridis']
        color_list = [
            bottom(0.99), 
            top_parent(0.15), 
            top_parent(0.33), 
            top_parent(0.66), 
            top_parent(0.99), 
        ]
        top = mcolors.LinearSegmentedColormap.from_list("top", color_list, N=256)

        for i, (band, input) in enumerate(self.inputs.items()):

            masked_SNR_map = np.where(input['mask']==0, input['data']/input['noise'], -3) #np.ma.masked_where(input['mask'], input['data']/input['noise'])

            img = masked_SNR_map

            img_bot = np.where(img <= 3, img, 3)
            axs[i].imshow(img_bot, vmin=-3, vmax=3, origin='lower', cmap=bottom)

            if np.any(img >= 3):
                img_top = np.ma.masked_where(img <= 3, img)
                norm = mcolors.LogNorm(vmin=3, vmax=10)
                axs[i].imshow(img_top, norm=norm, origin='lower', cmap=top)

            # axs[i].imshow(masked_SNR_map, vmin=-3, vmax=5, origin='lower', cmap='viridis')
            
            axs[i].set_title(band)
        
    def plot_outputs(self, axs:List[plt.Axes]):

        bottom = mcmaps['binary']
        top_parent = mcmaps['viridis']
        color_list = [
            bottom(0.99), 
            top_parent(0.15), 
            top_parent(0.33), 
            top_parent(0.66), 
            top_parent(0.99), 
        ]
        top = mcolors.LinearSegmentedColormap.from_list("top", color_list, N=256)

        for i, (band, output) in enumerate(self.outputs.items()):

            input = self.inputs[band]

            masked_SNR_resi = np.where(input['mask']==0, output['residuals']/input['noise'], -3) #np.ma.masked_where(input['mask'], resi/input['noise'])

            img = masked_SNR_resi

            img_bot = np.where(img <= 3, img, 3)
            axs[i].imshow(img_bot, vmin=-3, vmax=3, origin='lower', cmap=bottom)

            if np.any(img >= 3):
                img_top = np.ma.masked_where(img <= 3, img)
                norm = mcolors.LogNorm(vmin=3, vmax=10)
                axs[i].imshow(img_top, norm=norm, origin='lower', cmap=top)

            # axs[i].imshow(masked_SNR_resi, origin='lower', cmap='viridis')

            ell_aperture = EllipticalAperture(
                positions=(output['ell_geoms'].x0, output['ell_geoms'].y0),
                a=output['ell_geoms'].sma,
                b=output['ell_geoms'].sma * (1 - output['ell_geoms'].eps),
                theta=output['ell_geoms'].pa
            )

            ell_aperture.plot(ax=axs[i], color='red', lw=2)

            axs[i].plot(output['ell_geoms'].x0, output['ell_geoms'].y0, 'rx', ms=10)
            
            axs[i].set_title(band)


class ApGroupModelParser(dict):

    def __init__(self, model:ap.models.AstroPhot_Model):

        self.__dict__['model'] = model
        self.get_dict_from_model()

    def __getattr__(self, item):
        return super().__getitem__(item)

    def __setattr__(self, item, value):
        return super().__setitem__(item, value)
        

    def get_dict_from_model(self) -> None:
        
        self.clear()
        for name, component in self.model.models.items():
            
            self[name] = {}
            self[name]['model_type'] = component.model_type

            self[name]['parameters'] = {}
            for param_name in component.parameters.names:
                if param_name == 'center:0' or param_name == 'center:1':
                    self[name]['parameters']['center'] = {}
                    self[name]['parameters']['center']['value'] = component.parameters['center'].value.detach().cpu().numpy()
                else:
                    self[name]['parameters'][param_name] = {}
                    self[name]['parameters'][param_name]['value'] = component.parameters[param_name].value.detach().cpu().numpy()
                    self[name]['parameters'][param_name]['limits'] = []
                    for limit in component.parameters[param_name].limits:
                        if limit is not None:
                            self[name]['parameters'][param_name]['limits'].append(limit.detach().cpu().numpy())
                        else:
                            self[name]['parameters'][param_name]['limits'].append(None)

            self[name]['window'] = {}
            self[name]['window']['center'] = component.window.center.detach().cpu().numpy()
            self[name]['window']['pixel_shape'] = component.window.pixel_shape.detach().cpu().numpy()

    def get_model_from_dict(self, target) -> ap.models.AstroPhot_Model:

        if target.has_psf:
            psf_mode = 'full'
        else:
            psf_mode = 'none'
        
        component_list = []
        for name, component in self.items():

            if name != 'sky level':
                window = ap.image.image_object.Window(
                    **component['window'], 
                    pixelscale=target.pixelscale
                ) 

                component_list.append(
                    ap.models.AstroPhot_Model(
                        name=name,
                        model_type=component['model_type'],
                        parameters=component['parameters'],
                        window=window, 
                        psf_mode=psf_mode,
                        target=target,
                    )
                )
            
            else:
                sky = ap.models.AstroPhot_Model(
                    name=f"sky level",
                    model_type="flat sky model",
                    target=target,
                    parameters=component['parameters'],
                )

                component_list.append(sky)

        groupmodel = ap.models.AstroPhot_Model(
            name="group model",
            model_type="group model",
            target=target,
            models=component_list,
        )

        return groupmodel
    
import ipywidgets as widgets
from IPython.display import display, clear_output

from functools import partial

from copy import deepcopy

def _create_box_x_slider(component):

    value = component['window']['pixel_shape']

    box_x_size_slider = widgets.IntSlider(
        value=value[0],
        min=10,
        max=150,
        step=10,
        description='Box X Size:',
        continuous_update=True
    )

    return box_x_size_slider

def _create_box_y_slider(component):

    value = component['window']['pixel_shape']

    box_y_size_slider = widgets.IntSlider(
        value=value[1],
        min=10,
        max=150,
        step=10,
        description='Box Y Size:',
        continuous_update=True
    )

    return box_y_size_slider

def _create_q_slider(component):

    print(component['parameters']['q'])

    value = component['parameters']['q']['value']
    limit_low = component['parameters']['q']['limits'][0]
    limit_high = component['parameters']['q']['limits'][1]

    if np.round(value, 2) == 1.0:
        value = 0.99

    if np.round(value, 2) == 0.0:
        value = 0.01

    q_slider = widgets.FloatSlider(
        value=value,
        min=limit_low,
        max=limit_high,
        step=0.001,
        description='q:',
        continuous_update=True
    )
    return q_slider

def _create_PA_slider(component):

    value = component['parameters']['PA']['value']
    limit_low = component['parameters']['PA']['limits'][0]
    limit_high = component['parameters']['PA']['limits'][1]

    PA_slider = widgets.FloatSlider(
        value=value,
        min=limit_low,
        max=limit_high,
        step=0.1,
        description='PA:',
        continuous_update=True
    )
    return PA_slider
    
def _create_F_slider(component):

    value = component['parameters']['F']['value']
    limit_low = value - 0.1*value
    limit_high = value + 0.1*value

    F_slider = widgets.FloatSlider(
        value=value,
        min=limit_low,
        max=limit_high,
        step=0.001,
        description='F:',
        continuous_update=True
    )
    return F_slider


def add_component(models_dict, new_type, name, window, priors):

    for band in models_dict.keys():

        if priors is None:

            if new_type == "sersic galaxy model":
                priors = {
                    'center': {'value': np.array([0, 0]), 'limits': [None, None]}, 
                    'q': {'value': np.array(0.9), 'limits': [np.array(0), np.array(1)]},
                    'PA': {'value': np.array(0.5), 'limits': [np.array(0), np.array(np.pi)]}, 
                    'n': {'value': np.array(1), 'limits': [np.array(0), np.array(10)]},
                    'Re': {'value': np.array(1), 'limits': [np.array(0), None]},
                    'Ie': {'value': np.array(1), 'limits': [np.array(0), None]},
                }
            elif new_type == "moffat galaxy model":
                priors = {
                    'center': {'value': np.array([0, 0]), 'limits': [None, None]}, 
                    'q': {'value': np.array(0.9), 'limits': [np.array(0), np.array(1)]},
                    'PA': {'value': np.array(0.5), 'limits': [np.array(0), np.array(np.pi)]}, 
                    'n': {'value': np.array(1), 'limits': [np.array(0), np.array(10)]},
                    'Rd': {'value': np.array(10), 'limits': [np.array(0), None]},
                }
            elif new_type == "flat sky model":
                priors = {
                    'F': {'value': 1, 'limits': [np.array(-3), np.array(3)]},
                }
            else:
                raise NotImplementedError(f"Component {new_type} not found in the model.")
            
        component_names = models_dict[band].keys()

        if name not in component_names:
                    
            if window is None:
                raise ValueError("Scemo!!")
            else:
                window = window

            priors['center']['value'] = window['center']

            models_dict[band][name] = {
                'model_type': new_type,
                'window': window,
                'parameters': priors,
            }
            
    return models_dict

def _update_component(component, new_type, priors):

    if priors is None:

        if new_type == "sersic galaxy model":
            priors = {
                    'center': {'value': np.array([0, 0]), 'limits': [None, None]}, 
                    'q': {'value': np.array(0.9), 'limits': [np.array(0), np.array(1)]},
                    'PA': {'value': np.array(0.5), 'limits': [np.array(0), np.array(np.pi)]}, 
                    'n': {'value': np.array(1), 'limits': [np.array(0), np.array(10)]},
                    'Re': {'value': np.array(1), 'limits': [np.array(0), None]},
                    'Ie': {'value': np.array(1), 'limits': [np.array(0), None]},
                }
        elif new_type == "moffat galaxy model":
            priors = {
                    'center': {'value': np.array([0, 0]), 'limits': [None, None]}, 
                    'q': {'value': np.array(0.9), 'limits': [np.array(0), np.array(1)]},
                    'PA': {'value': np.array(0.5), 'limits': [np.array(0), np.array(np.pi)]}, 
                    'n': {'value': np.array(1), 'limits': [np.array(0), np.array(10)]},
                    'Rd': {'value': np.array(10), 'limits': [np.array(0), None]},
                }
        elif new_type == "flat sky model":
            priors = {
                    'F': {'value': 1, 'limits': [np.array(-3), np.array(3)]},
                }
        else:
            raise NotImplementedError(f"Component {new_type} not found in the model.")


    return {
        'model_type': new_type,
        'window': component['window'],
        'parameters': priors,
    }


class EditComponent:
    def __init__(self, models_dict, reference_band, name):

        self.models_dict = models_dict

        self.component = self.models_dict[reference_band][name]
        self.name = name
        self.model_type = self.component['model_type']
        self.window = self.component['window']

    def initialize(self):

        self.dropdown = widgets.Dropdown(
            options=['sersic galaxy model', 'moffat galaxy model', 'flat sky model'],
            description='Type:',
            value=self.model_type
        )
        self.sliders = [self.dropdown]
        self.get_paramter_widgets()

        self.dropdown.observe(self.update_type, names='value')

        self.accept_button = widgets.Button(description="Accept")

        self.reject_button = widgets.Button(description="Reject")

        self.accept_button.on_click(self.accept_changes)


        # self.reject_button.on_click(self.reject_changes)

            
    def get_paramter_widgets(self):

        self.sliders = [self.dropdown]

        if self.model_type == "sersic galaxy model":
            self.sliders.append(_create_box_x_slider(self.component))
            self.sliders.append(_create_box_y_slider(self.component))
            self.sliders.append(_create_q_slider(self.component))
            self.sliders.append(_create_PA_slider(self.component))
        elif self.model_type == "moffat galaxy model":
            self.sliders.append(_create_box_x_slider(self.component))
            self.sliders.append(_create_box_y_slider(self.component))            
            self.sliders.append(_create_q_slider(self.component))
            self.sliders.append(_create_PA_slider(self.component))
        elif self.model_type == "flat sky model":
            self.sliders.append(_create_F_slider(self.component))

        if self.model_type == 'sersic galaxy model':
            slider_q = partial(self.get_slider_value, parameter_name='q')
            self.sliders[3].observe(slider_q, names='value')
            slider_pa = partial(self.get_slider_value, parameter_name='PA')
            self.sliders[4].observe(slider_pa, names='value')

            slider_box_x = partial(self.get_box_slider_value, which='x')
            self.sliders[1].observe(slider_box_x, names='value')
            slider_box_y = partial(self.get_box_slider_value, which='y')
            self.sliders[2].observe(slider_box_y, names='value')

        elif self.model_type == 'moffat galaxy model':
            slider_q = partial(self.get_slider_value, parameter_name='q')
            self.sliders[3].observe(slider_q, names='value')
            slider_pa = partial(self.get_slider_value, parameter_name='PA')
            self.sliders[4].observe(slider_pa, names='value')

        elif self.model_type == 'flat sky model':
            slider_F = partial(self.get_slider_value, parameter_name='F')
            self.sliders[1].observe(slider_F, names='value')

        self.HBox = widgets.HBox(self.sliders)

    def update_type(self, change):
        print(f"Dropdown changed to: {change['new']}")  # Debug

        self.model_type = change['new']

        for band in self.models_dict.keys():
            self.component = _update_component(self.component, self.model_type, None)

        self.get_paramter_widgets()
        self.refresh_display()

    def get_slider_value(self, change, parameter_name):
        self.component['parameters'][parameter_name]['value'] = np.round(change['new'], 3)

    def get_box_slider_value(self, change, which):
        if which == 'x':
            self.component['window']['pixel_shape'][0] = change['new']
        elif which == 'y':
            self.component['window']['pixel_shape'][1] = change['new']

    def refresh_display(self):
        print(f"Refreshing display for component: {self.name}")  # Debug
        clear_output(wait=True)
        display(self.HBox)
        display(self.accept_button)

    def display(self):
        print(f"Displaying component: {self.name}")  # Debug
        display(self.HBox)
        display(self.accept_button)

    

    def accept_changes(self, change=None):

        for band in self.models_dict.keys():
            self.models_dict[band][self.name] = deepcopy(self.component)


        

# from matplotlib import colormaps as mcmaps
# from matplotlib.colors import Normalize 
# from matplotlib.widgets import Button
# from threading import Event
# import shapely as sh
# from shapely.plotting import plot_polygon
# from attila.interactive.masking_segment import InteractiveMaskDrawer, InteractiveMaskDilator

# def _plot_window_dict(ax, window_dict, pixel_scale, **kwargs):

#     dx = window_dict['pixel_shape'][0]
#     dy = window_dict['pixel_shape'][1]

#     center_arcsec = window_dict['center']

#     center_pix = [center_arcsec[0] / pixel_scale, center_arcsec[1] / pixel_scale]

#     # ax.add_patch(plt.Rectangle((x0, y0), dx, dy, fill=False, edgecolor='C1', lw=2, **kwargs))

#     x0 = center_pix[0] - dx / 2
#     y0 = center_pix[1] - dy / 2


#     rct = sh.Polygon([(x0, y0), (x0 + dx, y0), (x0 + dx, y0 + dy), (x0, y0 + dy)])
#     plt_poly = plot_polygon(rct, ax=ax, add_points=False, ls='--', ec='C1', fc='none', hatch='//', alpha=1)

#     return ax, plt_poly

# class InteractiveWindowSelector:
#     def __init__(self, fig, ax, pixelscale):
#         self.fig = fig
#         self.ax = ax
#         self.pixelscale = pixelscale
#         self.window_dict = None 
#         self.clicks = []

#         self.windows = []

#     def on_click(self, event):

#         # Handle the click event
#         if event.inaxes == self.ax:

#             if event.button == 1:  # Left click
#                 self.clicks.append((event.xdata, event.ydata))

#                 if len(self.clicks) == 1:

#                     self.ax.plot(self.clicks[0][0], self.clicks[0][1], marker='o', mec='C1', mfc=None) 

#                 elif len(self.clicks) == 2:
#                     center = self.clicks[0]

#                     diagonal = np.array(self.clicks[1])

#                     dx = int(2 * abs(diagonal[0] - center[0]))
#                     dy = int(2 * abs(diagonal[1] - center[1]))

#                     numpy_pixelscale = self.pixelscale.detach().cpu().numpy()

#                     center_arcsec = np.array(center) * np.array([numpy_pixelscale[0][0], numpy_pixelscale[1][1]])

#                     # self.ax.figure.canvas.mpl_disconnect(self.cid)

#                     self.window_dict = {
#                         'center': center_arcsec,
#                         'pixel_shape': [dx, dy],
#                     }

#                     self.center_plot = self.ax.plot(self.clicks[1][0], self.clicks[1][1], marker='x', mec='C1', mfc=None)
#                     self.ax, self.plt_poly = _plot_window_dict(self.ax, self.window_dict, pixel_scale=numpy_pixelscale[0][0])

#                     self.windows.append(self.window_dict)
#                     self.fig.canvas.draw()

#             elif event.button == 3:
#                 self.clicks = []

#                 self.center_plot[0].remove()

#                 self.plt_poly.remove()

#                 self.fig.canvas.draw()

#             elif event.button == 2:  # Middle click
#                 self.ax.figure.canvas.mpl_disconnect(self.cid)


# def _plot_image(ax, image, vlims, colormaps, mask=None, **kwargs):

#     if mask is not None:
#         image = np.where(mask, 0, image)

#     bottom_image = np.where(image < vlims[0], vlims[0], image)
#     bottom_image = np.where(bottom_image > vlims[1], vlims[1], bottom_image)

#     top_image = np.where(image > vlims[2], vlims[2], image)
#     top_image = np.where(top_image < vlims[1], np.nan, top_image)

#     ax.imshow(bottom_image, cmap=colormaps[0], vmin=vlims[0], vmax=vlims[1], origin='lower', **kwargs)

#     ax.imshow(top_image, cmap=colormaps[1], vmin=vlims[1], vmax=vlims[2], origin='lower', **kwargs)

#     return ax

# def plot_window(ax, window, **kwargs):

#     dx = window.pixel_shape[0].detach().cpu().numpy()
#     dy = window.pixel_shape[1].detach().cpu().numpy()

#     center_arcsec = window.center.detach().cpu().numpy()

#     pixel_scale = window.pixelscale.detach().cpu().numpy()

#     center_pix = [center_arcsec[0] / pixel_scale[0][0], center_arcsec[1] / pixel_scale[1][1]]

#     x0 = center_pix[0] - dx / 2
#     y0 = center_pix[1] - dy / 2

#     ax.add_patch(plt.Rectangle((x0, y0), dx, dy, fill=False, edgecolor='C1', lw=2, **kwargs))

#     return ax

# class PlotModel:
#     def __init__(self, results):
#         self.results = results
#         self.current_band = list(results.keys())[0]

#         # reapply the standard masks

#         for band in results.keys():
#             results[band]['target'].mask = results[band]['mask']

#         self.working_component_name = None

#         self.dropdown_band = widgets.Dropdown(
#             options=list(results.keys()),
#             description='Band:',
#             value=self.current_band
#         )

#         display(self.dropdown_band)

#         self.models_dict = {band: ApGroupModelParser(self.results[band]['model']) for band in self.results.keys()}

#         # Initialize the figure and axes
#         self.fig, (self.ax_target, self.ax_model, self.ax_resi) = plt.subplots(1, 3, figsize=(15, 5))
#         self.vmed = 10
#         self.vmax = 50
#         self.bottom_cmap = mcmaps['coolwarm']
#         self.top_cmap = mcmaps['RdYlGn']

#         # Load the initial band data
#         self.load_band(self.current_band)

#         # Observe changes in the dropdown
#         self.dropdown_band.observe(self.monitor_band, names='value')

#         self.selector = None
#         self.masker = None

        

#     def load_band(self, band):
#         """Load the data for the selected band."""
#         result_band = self.results[band]
#         self.model = result_band['model']
#         self.target = result_band['target']
#         self.model.target = self.target

#         self.plot()

#     def monitor_band(self, change):
#         """Handle changes in the dropdown value."""
#         new_band = change['new']
#         if new_band != self.current_band:
#             self.current_band = new_band
#             self.load_band(new_band)
        

#     def plot(self):
        
#         target_data = self.target.data.detach().cpu().numpy()
#         target_variance = self.target.variance.detach().cpu().numpy()

#         target_noise = np.sqrt(target_variance)

#         self.ax_target.imshow(target_data/target_variance, cmap='viridis')

#         self.ax_target = _plot_image(
#             self.ax_target, 
#             target_data/target_noise, 
#             vlims=[-3, self.vmed, self.vmax], 
#             colormaps=[self.bottom_cmap, self.top_cmap],
#             mask=self.target.mask.detach().cpu().numpy(),
#         )

#         if self.working_component_name is not None:

#             if self.working_component_name in self.model.models.keys():

#                 base_image = self.target.model_image()

#                 for model in self.model.models.values():
#                     if model.name != self.working_component_name:
#                         continue

#                     model_partial = model(image=base_image)

#                 model_image = model_partial.data.detach().cpu().numpy()

#                 self.ax_model = _plot_image(
#                     self.ax_model,
#                     model_image/target_noise,
#                     vlims=[-3, self.vmed, self.vmax], 
#                     colormaps=[self.bottom_cmap, self.top_cmap],
#                 )

#                 resi_image = target_data - model_image

#                 self.ax_resi = _plot_image(
#                     self.ax_resi,
#                     resi_image/target_noise,
#                     vlims=[-3, self.vmed, self.vmax], 
#                     colormaps=[self.bottom_cmap, self.top_cmap],
#                     mask=self.target.mask.detach().cpu().numpy(),
#                 )

#                 self.ax_target = plot_window(self.ax_target, self.model.models[self.working_component_name].window)

#                 self.ax_model = plot_window(self.ax_model, self.model.models[self.working_component_name].window)

#                 self.ax_resi = plot_window(self.ax_resi, self.model.models[self.working_component_name].window)

#             else:

#                 self.ax_model.axis('off')
#                 self.ax_resi.axis('off')

#                 # selector.cid = self.fig.canvas.mpl_connect('button_press_event', selector.on_click)

#                 self.selector = InteractiveWindowSelector(self.fig, self.ax_target, self.target.pixelscale)

#         else:

#             base_image = self.target.model_image()

#             model_total = self.model(image=base_image)

#             model_image = model_total.data.detach().cpu().numpy()

#             self.ax_model = _plot_image(
#                 self.ax_model,
#                 model_image/target_noise,
#                 vlims=[-3, self.vmed, self.vmax], 
#                 colormaps=[self.bottom_cmap, self.top_cmap],
#             )

#             resi_image = target_data - model_image

#             self.ax_resi = _plot_image(
#                 self.ax_resi,
#                 resi_image/target_noise,
#                 vlims=[-3, self.vmed, self.vmax], 
#                 colormaps=[self.bottom_cmap, self.top_cmap],
#                 mask=self.target.mask.detach().cpu().numpy(),
#             )

#         self.fig.canvas.draw()

#     def plot_component(self, component_name):
        
#         self.working_component_name = component_name

#         self.ax_target.clear()

#         self.ax_model.clear()

#         self.ax_resi.clear()

#         self.plot()


#     def edit_mask(self, which='target', zoom=(False, 100)):
        
#         self.ax_target.clear()

#         self.ax_model.clear()

#         self.ax_resi.clear()

#         xlim = -0.1 * self.target.data.shape[0], self.target.data.shape[0] * 1.1
#         ylim = -0.1 * self.target.data.shape[1], self.target.data.shape[1] * 1.1

#         if zoom[0]:
#             xlim = (xlim[0] + xlim[1]) /2. - zoom[1]/2., (xlim[0] + xlim[1]) /2. + zoom[1]/2.
#             ylim = (ylim[0] + ylim[1]) /2. - zoom[1]/2., (ylim[0] + ylim[1]) /2. + zoom[1]/2.

#         self.ax_target.set_xlim(xlim)
#         self.ax_target.set_ylim(ylim)

#         self.ax_model.set_xlim(xlim)
#         self.ax_model.set_ylim(ylim)

#         self.ax_resi.set_xlim(xlim)
#         self.ax_resi.set_ylim(ylim)

#         self.plot()

#         if which == 'target':
#             draw_axis = self.ax_target
#         elif which == 'model':
#             draw_axis = self.ax_model
#         elif which == 'resi':
#             draw_axis = self.ax_resi

#         self.edit_mask_handle = InteractiveMaskDrawer(self.fig, draw_axis, self.target.mask.detach().cpu().numpy())

#         # self.maskcid = self.fig.canvas.mpl_connect('button_press_event', self.masker.on_click)

#         return self.edit_mask_handle
    
#     def dilate_mask(self, which='target'):

#         self.ax_target.clear()

#         self.ax_model.clear()

#         self.ax_resi.clear()

#         # xlim = -0.1 * self.target.data.shape[0], self.target.data.shape[0] * 1.1
#         # ylim = -0.1 * self.target.data.shape[1], self.target.data.shape[1] * 1.1

#         # self.ax_target.set_xlim(xlim)
#         # self.ax_target.set_ylim(ylim)

#         # self.ax_model.set_xlim(xlim)
#         # self.ax_model.set_ylim(ylim)

#         # self.ax_resi.set_xlim(xlim)
#         # self.ax_resi.set_ylim(ylim)

#         self.plot()

#         if which == 'target':
#             draw_axis = self.ax_target
#         elif which == 'model':
#             draw_axis = self.ax_model
#         elif which == 'resi':
#             draw_axis = self.ax_resi

#         self.dilate_mask_handle = InteractiveMaskDilator(self.fig, draw_axis, self.target.mask.detach().cpu().numpy())

#         # self.maskcid = self.fig.canvas.mpl_connect('button_press_event', self.masker.on_click)

#         return self.dilate_mask_handle