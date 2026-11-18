"""
Unit conversions and small array helpers shared across ATTILA.

The module converts between pixel values in counts and magnitudes or
surface brightnesses, derives the pixel scale of an image from its WCS
and pads arrays to a common shape. `compute_pixel_scale` is used by
`attila.field.field.Field` when the images are loaded, so the value it
returns is the pixel scale used by the rest of the package.
"""
import numpy as np
import numpy.typing as npt
import astropy.units as u

from astropy.wcs import WCS

from typing import Dict, Union

def to_mag(array: npt.ArrayLike, zpt: float, pixel_scale: float) -> npt.ArrayLike:
    """
    Convert counts per pixel to a surface brightness.

    The conversion is

        mu = -2.5 log10(array) + zpt + 5 log10(pixel_scale)

    which, despite the name of the function, includes the pixel-area
    term and therefore yields a surface brightness and not a magnitude.
    It is the same expression as `tansform_to_SB`. Non-positive values
    give `nan` or `inf`.

    Parameters
    ----------
    array
        The pixel values in counts per pixel.
    zpt
        The photometric zero point in magnitudes.
    pixel_scale
        The pixel scale in arcsec / pixel.

    Returns
    -------
    numpy.ndarray
        The surface brightness in mag / arcsec^2, with the shape of
        `array`.
    """

    return -2.5*np.log10(array) + zpt + 5*np.log10(pixel_scale)

def compute_pixel_scale(wcs: WCS) -> float:
    """
    Return the pixel scale of an image from its WCS.

    The scales of the two image axes are read from
    `wcs.proj_plane_pixel_scales()`. In the `numpy.array` call the
    scale of the second axis is passed in the position of the `dtype`
    argument, so the array holds the scale of the first axis only and
    the mean reduces to that single value. The result is the pixel
    scale along the first WCS axis; the two are equal for square
    pixels.

    The WCS is assumed to have celestial axes in degrees.

    Parameters
    ----------
    wcs
        The world coordinate system of the image.

    Returns
    -------
    astropy.units.Quantity
        The pixel scale in arcsec / pixel. Note that this is a
        `Quantity` and not a plain `float` as the return annotation
        says.
    """

    return ((np.array(wcs.proj_plane_pixel_scales()[0].to_value(),
                    wcs.proj_plane_pixel_scales()[1].to_value()).mean()*u.deg).to(u.arcsec))/u.pixel

def tansform_to_SB(data: Union[float, npt.ArrayLike], zpt: float, pixel_scale:float):
    """
    Convert counts per pixel to a surface brightness.

    The conversion is

        mu = -2.5 log10(data) + zpt + 5 log10(pixel_scale)

    with no correction for extinction or cosmological dimming. It is
    the inverse of `tansform_from_SB`.

    Parameters
    ----------
    data
        The pixel values in counts per pixel.
    zpt
        The photometric zero point in magnitudes.
    pixel_scale
        The pixel scale in arcsec / pixel.

    Returns
    -------
    float or numpy.ndarray
        The surface brightness in mag / arcsec^2.
    """

    return -2.5*np.log10(data) + zpt + 5*np.log10(pixel_scale)

def tansform_from_SB(data: Union[float, npt.ArrayLike], zpt: float, pixel_scale:float):
    """
    Convert a surface brightness to counts per pixel.

    The conversion is the inverse of `tansform_to_SB`:

        counts = 10 ** ((data - 5 log10(pixel_scale) - zpt) / -2.5)

    Parameters
    ----------
    data
        The surface brightness in mag / arcsec^2.
    zpt
        The photometric zero point in magnitudes.
    pixel_scale
        The pixel scale in arcsec / pixel.

    Returns
    -------
    float or numpy.ndarray
        The pixel values in counts per pixel.
    """

    # The trailing comment is the forward relation that is inverted.
    return 10**((data - 5*np.log10(pixel_scale) - zpt)/(-2.5)) #-2.5*np.log10(data) + zpt + 5*np.log10(pixel_scale)

def transform_to_mag(data: Union[float, npt.ArrayLike], zpt: float):
    """
    Convert counts to a magnitude.

    The conversion is `m = -2.5 log10(data) + zpt`, with no pixel-area
    term.

    Parameters
    ----------
    data
        The flux in counts.
    zpt
        The photometric zero point in magnitudes.

    Returns
    -------
    float or numpy.ndarray
        The magnitude.
    """

    return -2.5*np.log10(data) + zpt

def pad_to_shape(array, shape):
    """
    Pad a 2D array with zeros up to a target shape.

    The zeros are appended at the end of each axis (after the last row
    and after the last column), so the input keeps its position at the
    origin of the output. An axis that is already at least as long as
    requested is left unchanged; the array is never cropped.

    Parameters
    ----------
    array : numpy.ndarray
        The 2D array to pad.
    shape : tuple of int
        The target shape `(n_rows, n_columns)` in pixels.

    Returns
    -------
    numpy.ndarray
        The padded array, a new array.
    """
    # Calculate the padding for each dimension
    pad_width = [(0, max(0, shape[i] - array.shape[i])) for i in range(2)]

    # Pad the array with zeros
    padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)

    return padded_array




# The block below is disabled code, kept for reference: it converted
# stacks of per-band 2D images or 1D profiles to magnitudes with
# `to_mag`. It refers to `self` attributes that do not exist in this
# module.
#
# def trasform_to_mag_2D(array: npt.ArrayLike, zpt: Dict[str, float]):
#
#     if array.ndim == 4:
#         return np.asarray([[(to_mag(slice_single, zpt_i, pix_scale_i)) for slice_single in slice_array] for
#                 slice_array, zpt_i, pix_scale_i, in zip(array, zpt_dict.values(), self.pixel_scale_dict.values())])
#
#     elif array.ndim == 3:
#         return np.asarray([to_mag(slice_single, zpt_i, pix_scale_i) for
#                             slice_single, zpt_i, pix_scale_i, in zip(array, self.zpt_dict.values(), self.pixel_scale_dict.values())])
#
#
# def trasform_to_mag_1D(self, array: npt.ArrayLike):
#
#     if array.ndim == 3:
#         return np.asarray([[(to_mag(value, zpt_i, pix_scale_i)) for value in slice_single] for
#                 slice_single, zpt_i, pix_scale_i, in zip(array, self.zpt_dict.values(), self.pixel_scale_dict.values())])
#
#     elif array.ndim == 2:
#         return np.asarray([to_mag(value, zpt_i, pix_scale_i) for
#                             value, zpt_i, pix_scale_i, in zip(array, self.zpt_dict.values(), self.pixel_scale_dict.values())])
