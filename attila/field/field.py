"""
Containers for the single-band and multi-band images of a field.

This module is the entry point of the pipeline flow. `Field` reads
the FITS data and noise maps of every band of an observed field;
each band is stored as an `Image` (data, noise, WCS, header, pixel
scale) and the bands are collected in a `MultibandImage`. The same
classes are used for the cutouts made around tiles and sources.
"""
import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy import units as u

from typing import List, Dict, Tuple, Union, Iterator

from attila.utils.image_utils import compute_pixel_scale

from astropy.cosmology import units as cu

from astropy.visualization import make_lupton_rgb


class Field:
    """
    An observed field: the images and noise maps of all its bands.

    The images are read from FITS files at construction and stored in
    `multibandimages`. `compute_physical` optionally attaches a
    distance and the corresponding physical scale to the field.

    Attributes
    ----------
    bands : list of str
        The names of the bands, in the order of `obs_path_dict`.
    multibandimages : MultibandImage
        The images of all bands.
    cosmology : astropy.cosmology.Cosmology
        The cosmology, set by `compute_physical`.
    distance : astropy.units.Quantity
        The distance of the field in Mpc, set by `compute_physical`.
    redshift : astropy.units.Quantity
        The redshift derived from `distance`, set by
        `compute_physical`.
    angular_scale_dict : dict
        For every band, the physical scale in kpc / arcsec, set by
        `compute_physical`.
    """
    

    def __init__(
            self, 
            obs_path_dict: Dict[str, str],
            extension=".fits",
            noise_type="rms",
            axis=0  # Index (int) of the FITS HDU to read.
        ) -> None:
        """
        Read the data and noise FITS files of every band.

        For every band the files `<prefix>_data<extension>` and,
        depending on `noise_type`, `<prefix>_noise<extension>` or
        `<prefix>_weights<extension>` are opened, where `<prefix>` is
        the value of `obs_path_dict` for that band. The WCS and the
        pixel scale are derived from the header of the data file. The
        array read from the noise or weights file is stored unchanged in
        `Image.noise`.

        Parameters
        ----------
        obs_path_dict
            For every band, the path prefix of its FITS files (without
            the `_data` / `_noise` / `_weights` suffix and the
            extension).
        extension : str
            The file extension, including the dot.
        noise_type : str
            `"rms"` to read the `_noise` file, `"weights"` to read the
            `_weights` file. The weights are not converted to an RMS.
        axis : int
            The index of the HDU, in each FITS file, from which the data
            and the header are read.

        Raises
        ------
        ValueError
            If `noise_type` is neither `"rms"` nor `"weights"`.
        """

        if noise_type not in ("rms", "weights"):
            raise ValueError("noise_type must be either 'rms' or 'weights'")

        self.bands = list(obs_path_dict.keys())

        # self.obs_dict = {}
        # self.header_dict = {}

        self.multibandimages = MultibandImage([])

        for band in self.bands:

            data_path = obs_path_dict[band] + "_data"+extension

            # The files are left open: the arrays are memory-mapped.
            hdu = fits.open(data_path)[axis]
            data = hdu.data
            header = hdu.header

            if noise_type == "rms":
                noise_path = obs_path_dict[band] + "_noise"+extension
                noise = fits.open(noise_path)[axis].data
            elif noise_type == "weights":
                noise_path = obs_path_dict[band] + "_weights"+extension
                noise = fits.open(noise_path)[axis].data

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
        """
        Attach a distance and the derived physical quantities.

        The distance is converted to Mpc and interpreted as a comoving
        distance to obtain the redshift in the given cosmology. The
        physical scale at that redshift, the inverse of
        `cosmology.arcsec_per_kpc_proper`, is stored for every band (the
        value is the same for all bands); it is the scale given by the
        angular diameter distance, and this convention was chosen
        because it matches the NED cosmology calculator. The results
        are stored in the
        attributes `cosmology`, `distance`, `redshift` and
        `angular_scale_dict`.

        Parameters
        ----------
        cosmology : astropy.cosmology.Cosmology
            The cosmology used for the conversions.
        distance
            The distance of the field, as a length quantity.
        """
    
        self.cosmology = cosmology

        self.distance = distance.to(u.Mpc)

        self.redshift = self.distance.to(cu.redshift, cu.redshift_distance(cosmology, kind='comoving', zmax=1000))

        self.angular_scale_dict = {band: (1/self.cosmology.arcsec_per_kpc_proper(self.redshift)) for band in self.bands}
        

class Image:
    """
    A single-band image with its noise map and astrometry.

    Attributes
    ----------
    band : str
        The name of the band.
    data : numpy.ndarray
        The 2D science image, with axes `(y, x)`.
    noise : numpy.ndarray
        The 2D noise map, with the same shape as `data`.
    pixel_scale
        The pixel scale, as returned by `compute_pixel_scale` (arcsec
        per pixel).
    wcs : astropy.wcs.WCS
        The WCS of `data`.
    header : astropy.io.fits.Header
        The FITS header the image comes from.
    ax : matplotlib.axes.Axes
        The axes of the last call to `plot`.
    """
    def __init__(
            self, 
            band: str,
            data: np.ndarray, 
            noise: np.ndarray, 
            pixel_scale: float,
            header: fits.Header,
            wcs: WCS,
        ) -> None:
        """
        Store the data, the noise and the astrometry of the image.

        Parameters
        ----------
        band
            The name of the band.
        data
            The 2D science image, with axes `(y, x)`.
        noise
            The 2D noise map, with the same shape as `data`.
        pixel_scale
            The pixel scale (arcsec per pixel).
        header
            The FITS header of the image.
        wcs
            The WCS of `data`.
        """
        
        self.band = band
        self.data = data
        self.noise = noise
        self.pixel_scale = pixel_scale
        self.wcs = wcs
        self.header = header   

    @property
    def shape(self) -> Tuple[int, int]:
        """The shape `(ny, nx)` of the data array."""
        return self.data.shape
    
    @property
    def SNR(self) -> np.ndarray:
        """The signal-to-noise map, `data / noise`, pixel by pixel."""
        return self.data / self.noise

    def cutout(self, position: SkyCoord, size: u.Quantity) -> Cutout2D:
        """
        Return a cutout of the image around a sky position.

        The data and the noise are cut with `astropy.nddata.Cutout2D` in
        `"partial"` mode, so a cutout crossing the image border is
        padded with the `Cutout2D` fill value. The new image carries the
        WCS of the cutout, while `header` is the header of the parent
        image and the pixel scale is recomputed from that header.

        Parameters
        ----------
        position
            The sky position of the centre of the cutout.
        size
            The size of the cutout, as accepted by `Cutout2D`: a single
            quantity for a square or a `(ny, nx)` pair, in angular units
            or pixels.

        Returns
        -------
        Image
            The cutout as a new image of the same band.
        """

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
        """
        Show the signal-to-noise map of the image on the given axes.

        The segmentation maps given in `additional_plotting` are drawn
        on top, semi-transparent. The axis frame is hidden and the axes
        are stored in the attribute `ax`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which the image is drawn.
        additional_plotting : dict, optional
            The overlays. The recognised keys are `"detection"` (drawn
            with alpha 0.25) and `"deblending"` (alpha 0.1); the values
            are objects with an `imshow` method and a `cmap` attribute,
            such as a photutils `SegmentationImage`. If None nothing is
            overlaid.
        **kwargs
            Passed to `matplotlib.axes.Axes.imshow`.
        """

        self.ax = ax

        self.ax.imshow(self.SNR, **kwargs)

        if isinstance(additional_plotting, dict):  
            if "detection" in additional_plotting:
                additional_plotting["detection"].imshow(ax=self.ax, cmap=additional_plotting["detection"].cmap, alpha=0.25)

            if "deblending" in additional_plotting:
                additional_plotting["deblending"].imshow(ax=self.ax, cmap=additional_plotting["deblending"].cmap, alpha=0.1)


        self.ax.axis("off")

class MultibandImage:
    """
    An ordered collection of single-band images of the same region.

    The images can be accessed by band name (`multiband["g"]`) and
    iterated over in the order they were added.

    Attributes
    ----------
    image_list : list of Image
        The single-band images.
    bands : list of str
        The names of the bands, in the same order as `image_list`.
    ax : matplotlib.axes.Axes
        The axes of the last call to `plot_color`.
    """
    def __init__(
            self, 
            image_list: List[Image]
        ) -> None:
        """
        Store the list of images and the names of their bands.

        Parameters
        ----------
        image_list
            The single-band images. The list is stored by reference and
            is extended in place by `append`.
        """
        
        self.image_list = image_list

        self.bands = [image.band for image in image_list]

    def __getitem__(self, band: str) -> Image:
        """Return the image of the given band name."""
        return self.image_list[self.bands.index(band)]

    def __iter__(self) -> Iterator[Image]:
        """Iterate over the single-band images."""
        return iter(self.image_list)

    def append(self, image: Image) -> None:
        """Add an image, and its band name, to the collection."""
        self.image_list.append(image)
        self.bands.append(image.band)
    
    def cutout(self, position: SkyCoord, size: u.Quantity):
        """
        Return the cutouts of all bands around a sky position.

        Parameters
        ----------
        position
            The sky position of the centre of the cutout.
        size
            The size of the cutout, as in `Image.cutout`.

        Returns
        -------
        MultibandImage
            The collection of the single-band cutouts.
        """
        return MultibandImage([image.cutout(
            position, 
            size
            ) for image in self.image_list])

    def property_dict(self, key: Union[List, str]) -> Dict[str, np.ndarray]:
        """
        Collect one or more attributes of the images in a dictionary.

        Parameters
        ----------
        key
            The name of an `Image` attribute (for example `"data"`,
            `"noise"`, `"wcs"` or `"header"`) or a list of such names.

        Returns
        -------
        dict
            If `key` is a string, the attribute of every band as `{band:
            value}`. If it is a list, a nested dictionary `{band: {name:
            value}}`. For any other type None is returned.
        """

        if isinstance(key, list):
            return {image.band: {k: getattr(image, k) for k in key} for image in self.image_list}
        if isinstance(key, str):
            return {image.band: getattr(image, key) for image in self.image_list}

    def plot_color(self, ax, additional_plotting=None, Q=25, stretch=75):
        """
        Show a colour composite of the bands on the given axes.

        The composite is built with
        `astropy.visualization.make_lupton_rgb`. With two bands the
        first is used as blue, the second as red and their
        pixel-by-pixel mean as green. With three bands they are used, in
        order, as blue, green and red. The image is drawn with the
        origin in the lower left corner, the axis frame is hidden and
        the axes are stored in the attribute `ax`.

        If `additional_plotting` contains the keys `"detection"` or
        `"deblending"`, the segmentation images stored under them are
        overlaid with their own colour maps, as in `Image.plot`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which the image is drawn.
        additional_plotting : dict, optional
            The segmentation maps to overlay (see above). If None
            nothing is overlaid.
        Q : float
            The asinh softening parameter of the Lupton scaling.
        stretch : float
            The linear stretch of the Lupton scaling.

        Raises
        ------
        ValueError
            If the collection has one band or more than three bands.
        """

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
                additional_plotting["detection"].imshow(ax=self.ax, cmap=additional_plotting["detection"].cmap, alpha=0.25)

            if "deblending" in additional_plotting:
                additional_plotting["deblending"].imshow(ax=self.ax, cmap=additional_plotting["deblending"].cmap, alpha=0.1)

        self.ax.axis("off")

            




    
        