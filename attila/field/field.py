import numpy as np
import numpy.typing as npt
import pickle


from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy import units as u

from functools import cached_property

from typing import List, Dict, Tuple, Union, Optional, Iterator

from attila.utils.image_utils import compute_pixel_scale

from os import path

from astropy.cosmology.realizations import Planck18

from astropy.cosmology import units as cu

from astropy.visualization import make_lupton_rgb

from copy import deepcopy

class Field:
    

    def __init__(
            self, 
            obs_path_dict: Dict[str, str],
            extension=".fits",
            noise_type="rms",
            axis=0 #(int)
        ) -> None:
        
        self.bands = list(obs_path_dict.keys())

        # self.obs_dict = {}
        # self.header_dict = {}

        self.multibandimages = MultibandImage([])

        for band in self.bands:

            data_path = obs_path_dict[band] + "_data"+extension

            data = fits.open(data_path)[axis].data
            header = fits.open(data_path)[axis].header

            if noise_type == "rms":
                noise_path = obs_path_dict[band] + "_noise"+extension
                noise = fits.open(noise_path)[axis].data
            elif noise_type == "weights":
                noise_path = obs_path_dict[band] + "_weights"+extension
                noise = fits.open(noise_path)[axis].data

            # print(noise_path)

            header_copy = deepcopy(header)

            self.multibandimages.append(
                Image(
                    band=band, 
                    data=data, 
                    noise=noise, 
                    pixel_scale=compute_pixel_scale(WCS(header)), 
                    header=header, 
                    wcs=WCS(header)
                )
            )

            # self.obs_dict[band] = {"data": data, "noise": noise}
            # self.header_dict[band] = header

        # self.wcs_dict = {band: WCS(header) for band, header in self.header_dict.items()}

        # self.pixel_scale_dict = {band: compute_pixel_scale(wcs) for band, wcs in self.wcs_dict.items()}


    def compute_physical(
            self, 
            cosmology,
            distance: u.Quantity
            ) -> None:
    
        self.cosmology = cosmology

        self.distance = distance.to(u.Mpc)

        self.redshift = self.distance.to(cu.redshift, cu.redshift_distance(cosmology, kind='comoving', zmax=1000))

        self.angular_scale_dict = {band: (1/self.cosmology.arcsec_per_kpc_proper(self.redshift)) for band in self.bands}
        

class Image:
    def __init__(
            self, 
            band: str,
            data: np.ndarray, 
            noise: np.ndarray, 
            pixel_scale: float,
            header: fits.Header,
            wcs: WCS,
        ) -> None:
        
        self.band = band
        self.data = data
        self.noise = noise
        self.pixel_scale = pixel_scale
        self.wcs = wcs
        self.header = header   

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape
    
    @property
    def SNR(self) -> np.ndarray:
        return self.data / self.noise

    def cutout(self, position: SkyCoord, size: u.Quantity) -> Cutout2D:

        data_cutout = Cutout2D(self.data, position, size, wcs=self.wcs, mode='partial')
        noise_cutout = Cutout2D(self.noise, position, size, wcs=self.wcs, mode='partial')

        # updated_header = header.copy().update(data_cutout.wcs.to_header())

        return Image(
            band=self.band,
            data=data_cutout.data,
            noise=noise_cutout.data,
            pixel_scale=compute_pixel_scale(WCS(self.header)), 
            wcs=data_cutout.wcs, 
            header=self.header)
    
    def plot(self, ax, additional_plotting=None, **kwargs):

        self.ax = ax

        self.ax.imshow(self.SNR, **kwargs)

        if isinstance(additional_plotting, dict):  
            if "detection" in additional_plotting:
                additional_plotting["detection"].imshow(ax=self.ax, cmap=additional_plotting["detection"].cmap, alpha=0.25)

            if "deblending" in additional_plotting:
                additional_plotting["deblending"].imshow(ax=self.ax, cmap=additional_plotting["deblending"].cmap, alpha=0.1)


        self.ax.axis("off")

class MultibandImage:
    def __init__(
            self, 
            image_list: List[Image]
        ) -> None:
        
        self.image_list = image_list

        self.bands = [image.band for image in image_list]

    def __getitem__(self, band: str) -> Image:
        return self.image_list[self.bands.index(band)]

    def __iter__(self) -> Iterator[Image]:
        return iter(self.image_list)

    def append(self, image: Image) -> None:
        self.image_list.append(image)
        self.bands.append(image.band)
    
    def cutout(self, position: SkyCoord, size: u.Quantity):
        return MultibandImage([image.cutout(
            position, 
            size
            ) for image in self.image_list])

    def property_dict(self, key: Union[List, str]) -> Dict[str, np.ndarray]:

        if isinstance(key, list):
            return {image.band: {k: getattr(image, k) for k in key} for image in self.image_list}
        if isinstance(key, str):
            return {image.band: getattr(image, key) for image in self.image_list}

    def plot_color(self, ax, additional_plotting=None, Q=25, stretch=75):

        self.ax = ax

        if len(self.bands) == 1:
            raise ValueError("The image must have at least two bands.")
        
        elif len(self.bands) == 2:
            blue_image_data = self[self.bands[0]].data
            red_image_data = self[self.bands[1]].data

            green_image_data = np.stack((blue_image_data, red_image_data), axis=-1).mean(axis=-1)

        elif len(self.bands) == 3:
            blue_image_data = self[self.bands[0]].data
            green_image_data = self[self.bands[1]].data
            red_image_data = self[self.bands[2]].data

        else:
            raise ValueError("The image must have at most three bands. Decide how to deal with it")
        
        color_image = make_lupton_rgb(
                red_image_data, 
                green_image_data, 
                blue_image_data, 
                Q=Q, stretch=stretch)

        self.ax.imshow(color_image, origin='lower')

        if isinstance(additional_plotting, dict):  
            if "detection" in additional_plotting:
                additional_plotting['segment'].imshow(ax=self.ax, cmap=self.segment.cmap, alpha=0.25)

            if "deblending" in additional_plotting:
                additional_plotting['deblend'].imshow(ax=self.ax, cmap=self.deblend.cmap, alpha=0.1)

        self.ax.axis("off")

            




    
        