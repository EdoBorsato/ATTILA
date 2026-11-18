import numpy as np
import numpy.typing as npt
import astropy.units as u

from astropy.wcs import WCS

from typing import Dict, Union

def to_mag(array: npt.ArrayLike, zpt: float, pixel_scale: float) -> npt.ArrayLike:

    return -2.5*np.log10(array) + zpt + 5*np.log10(pixel_scale)

def compute_pixel_scale(wcs: WCS) -> float:

    return ((np.array(wcs.proj_plane_pixel_scales()[0].to_value(), 
                    wcs.proj_plane_pixel_scales()[1].to_value()).mean()*u.deg).to(u.arcsec))/u.pixel

def tansform_to_SB(data: Union[float, npt.ArrayLike], zpt: float, pixel_scale:float):

    return -2.5*np.log10(data) + zpt + 5*np.log10(pixel_scale)

def tansform_from_SB(data: Union[float, npt.ArrayLike], zpt: float, pixel_scale:float):

    return 10**((data - 5*np.log10(pixel_scale) - zpt)/(-2.5)) #-2.5*np.log10(data) + zpt + 5*np.log10(pixel_scale)

def transform_to_mag(data: Union[float, npt.ArrayLike], zpt: float):

    return -2.5*np.log10(data) + zpt

def pad_to_shape(array, shape):
    # Calculate the padding for each dimension
    pad_width = [(0, max(0, shape[i] - array.shape[i])) for i in range(2)]

    # Pad the array with zeros
    padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)

    return padded_array




"""def trasform_to_mag_2D(array: npt.ArrayLike, zpt: Dict[str, float]):

    if array.ndim == 4:
        return np.asarray([[(to_mag(slice_single, zpt_i, pix_scale_i)) for slice_single in slice_array] for 
                slice_array, zpt_i, pix_scale_i, in zip(array, zpt_dict.values(), self.pixel_scale_dict.values())])

    elif array.ndim == 3:
        return np.asarray([to_mag(slice_single, zpt_i, pix_scale_i) for 
                            slice_single, zpt_i, pix_scale_i, in zip(array, self.zpt_dict.values(), self.pixel_scale_dict.values())])


def trasform_to_mag_1D(self, array: npt.ArrayLike):

    if array.ndim == 3:
        return np.asarray([[(to_mag(value, zpt_i, pix_scale_i)) for value in slice_single] for 
                slice_single, zpt_i, pix_scale_i, in zip(array, self.zpt_dict.values(), self.pixel_scale_dict.values())])

    elif array.ndim == 2:
        return np.asarray([to_mag(value, zpt_i, pix_scale_i) for 
                            value, zpt_i, pix_scale_i, in zip(array, self.zpt_dict.values(), self.pixel_scale_dict.values())])"""