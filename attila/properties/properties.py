"""
Photometric and geometric properties of the detected sources.

The `Properties` class measures, for every source of a
`attila.tiling.sources.Sources` container and for every band, its area,
magnitude, mean and peak surface brightness, aperture magnitude and
centroid. The measurements are collected in a `pandas.DataFrame` that
is attached to the container as `sources.properties` and saved to
`properties.csv` by `Sources.save_light`.

In the flow of the package this step comes after the detection (and
again after the deblending): the sources must already hold their image
cutouts (`Sources.get_data`) and the Galactic extinction computed by
`attila.properties.extinction.Extinction` is assigned to
`Properties.A` before the measurement.

The module-level functions hold the photometric formulae; the fluxes
are in the image counts and are converted with the zero points of the
field.
"""

from attila.tiling.sources import Sources
from attila.field.field import Field
import multiprocessing
import warnings

from copy import copy

import numpy as np
import pandas as pd
from photutils.centroids import centroid_quadratic


import astropy.units as u

def _compute_mag(eps, zpt, A, dist):
    """
    Convert a flux in counts to an extinction-corrected magnitude.

    The magnitude is

        -2.5 log10(eps) + zpt - A - 5 log10(dist / pc) + 5

    that is the apparent magnitude `-2.5 log10(eps) + zpt`, corrected
    for the extinction `A` and reduced by the distance modulus
    `5 log10(dist / 10 pc)`. With the true distance of the source this
    is its absolute magnitude. With `dist` equal to 10 pc, the value
    used by `Properties` when the field has no distance, the distance
    modulus vanishes and the result is the extinction-corrected
    apparent magnitude.

    Parameters
    ----------
    eps : float or numpy.ndarray
        The flux in counts. Non-positive values give `nan` or `inf`.
    zpt : float or numpy.ndarray
        The photometric zero point in magnitudes.
    A : float or numpy.ndarray
        The extinction in magnitudes.
    dist : astropy.units.Quantity
        The distance of the source, in any unit of length.

    Returns
    -------
    float or numpy.ndarray
        The magnitude, with the broadcast shape of the inputs.
    """
    return -2.5 * np.log10(eps) + zpt - A - 5 * np.log10(dist.to(u.pc).value) + 5

def _compute_sb(eps, zpt, scale, A, z):
    """
    Convert counts per pixel to a corrected surface brightness.

    The surface brightness is

        -2.5 log10(eps) + zpt + 5 log10(scale) - A - 10 log10(1 + z)

    where `5 log10(scale)` converts from per pixel to per arcsec^2,
    `A` removes the extinction and `10 log10(1 + z)` removes the
    `(1 + z)^4` cosmological surface-brightness dimming.

    Parameters
    ----------
    eps : float or numpy.ndarray
        The flux in counts per pixel. Non-positive values give `nan`
        or `inf`.
    zpt : float or numpy.ndarray
        The photometric zero point in magnitudes.
    scale : float or numpy.ndarray
        The pixel scale in arcsec / pixel.
    A : float or numpy.ndarray
        The extinction in magnitudes.
    z : float
        The redshift of the source; 0 applies no dimming correction.

    Returns
    -------
    float or numpy.ndarray
        The surface brightness in mag / arcsec^2, with the broadcast
        shape of the inputs.
    """
    return -2.5 * np.log10(eps) + zpt + 5 * np.log10(scale) - A - 10 * np.log10(1 + z)

def _compute_area_kpc(area_pixel, scale_kpc):
    """
    Convert an area in pixels to a physical area.

    Parameters
    ----------
    area_pixel : float or astropy.units.Quantity
        The area in pixels.
    scale_kpc : float or astropy.units.Quantity
        The linear size of a pixel in kpc / pixel.

    Returns
    -------
    float or astropy.units.Quantity
        The area in kpc^2, `area_pixel * scale_kpc**2`.
    """
    return area_pixel * scale_kpc**2

def _compute_area_arcsec(area_pixel, scale_arcsec):
    """
    Convert an area in pixels to an angular area.

    Parameters
    ----------
    area_pixel : float or astropy.units.Quantity
        The area in pixels.
    scale_arcsec : float or astropy.units.Quantity
        The pixel scale in arcsec / pixel.

    Returns
    -------
    float or astropy.units.Quantity
        The area in arcsec^2, `area_pixel * scale_arcsec**2`.
    """
    return area_pixel * scale_arcsec**2

def _area_pixel(area_pixel):
    """Return the area in pixels unchanged."""
    return area_pixel

def _label(sources: Sources):
    """Return the list of the labels of the sources."""
    return [source.label for source in sources]

# The object measured by the worker processes of
# `Properties.detection_properties`, which inherit it when forked.
_SHARED_PROPERTIES = None


def _stack_table(bounds):
    """
    Measure one stack of sources in a worker process.

    Parameters
    ----------
    bounds : tuple of int
        The first source of the stack and the one after the last.

    Returns
    -------
    pandas.DataFrame
        The table of the stack.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        return _SHARED_PROPERTIES._stack(*bounds)._table()


class Properties:
    """
    The measurements made on a set of sources.

    Each measurement is a method without arguments that returns either
    one value per source, or an array of shape `(n_sources, n_bands)`.
    `detection_properties` runs the methods listed in `functions` and
    arranges their results in a table, one row per source, which is
    stored in `sources.properties`.

    The sources must hold their image cutouts in `source.data` (see
    `Sources.get_data`), with the bands in the same order as `bands`.
    The magnitudes and surface brightnesses are corrected for the
    extinction in `A`, which is zero until the caller assigns it,
    typically with the output of `Extinction.apply_grid_to_sources`.

    If the field has the `distance` and `redshift` attributes (set by
    `Field.compute_physical`) they enter the magnitudes and the surface
    brightnesses, and the areas are in kpc^2; see the single methods.

    Attributes
    ----------
    sources : Sources
        The sources that are measured.
    field : Field
        The field the sources were detected in.
    bands : list of str
        The bands of the measurements.
    zpt : numpy.ndarray
        The zero point of each band in magnitudes, shape `(n_bands,)`.
    scale_arcsec : numpy.ndarray
        The pixel scale of each band in arcsec / pixel, shape
        `(n_bands,)`.
    A : numpy.ndarray
        The extinction of each source in each band in magnitudes,
        shape `(n_sources, n_bands)`.
    radius : int
        The radius in pixels of the circular aperture of `mag_aper`,
        10 by default.
    functions : list of callable
        The measurements run by `detection_properties`, in the order
        of the columns of the output table. The name of each method is
        the name of its column, or the prefix of its per-band columns.
    """

    def __init__(self, sources: Sources, field: Field, bands=None):
        """
        Set up the measurements for a set of sources.

        The zero points are read from `field.zpt`, a dictionary keyed
        by band that is not created by `Field` itself and must have
        been assigned to the field beforehand. The pixel scales are
        read from the images of the field.

        Parameters
        ----------
        sources
            The sources to measure.
        field
            The field the sources belong to. It provides the zero
            points, the pixel scales and, when available, the distance,
            the redshift and the physical scale.
        bands : list of str
            The bands to measure. `None` means all the bands of the
            field.
        """
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

        # No extinction correction until the caller assigns `A`.
        self.A = np.zeros((len(self.sources), len(self.bands)))

        # Aperture radius of `mag_aper`, in pixels (a deliberate choice).
        self.radius = 10

        # The order of this list is the order of the output columns.
        # `mag_aper` comes before the centroid columns because it
        # caches the centroids, which are then not computed again.
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
        """
        Return the area of every source in physical or angular units.

        If the field has the attribute `angular_scale_dict` (the
        physical scale in kpc / arcsec set by
        `Field.compute_physical`) the pixel area of the segment is
        converted to kpc^2, otherwise to arcsec^2. In both cases the
        pixel scale of the first band is used for all the sources.

        Returns
        -------
        list of float
            The area of each source, in kpc^2 or in arcsec^2.
        """

        area_pixel = [source.area * u.pix**2 for source in self.sources]

        if hasattr(self.field, 'angular_scale_dict'):
            scale_arcsec_kpc = [self.field.angular_scale_dict[band] for band in self.bands]

            # (arcsec / pixel) * (kpc / arcsec) = kpc / pixel, per band.
            scale_pixel_kpc = [scale_arcsec * scale_arcsec_kpc for scale_arcsec, scale_arcsec_kpc in zip(self.scale_arcsec, scale_arcsec_kpc)]

            return [_compute_area_kpc(area_pixel[i], scale_pixel_kpc[0]).value for i in range(len(area_pixel))]

        else:
            return [_compute_area_arcsec(area_pixel[i], self.scale_arcsec[0]).value for i in range(len(area_pixel))]

    def area_pixel(self):
        """Return the number of pixels in the segment of each source."""
        return [source.area for source in self.sources]

    def label(self):
        """Return the label of every source."""
        return _label(self.sources)

    def previous_label(self):
        """
        Return the label of the parent segment of every source.

        For a deblended source this is the label that its parent had in
        the detection segmentation; it is 0 for a source that was not
        deblended.
        """
        return [source.previous_label for source in self.sources]

    def extinction(self):
        """Return the extinction array `A`, in magnitudes."""
        return self.A

    def add_function(self, function):
        """
        Register a user-defined measurement (not implemented).

        Parameters
        ----------
        function : callable
            The measurement to append to `functions`.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError('This method is not implemented yet.')

    def mag(self):
        """
        Compute the magnitude of every source inside its segment.

        The flux of a source in a band is the sum of the counts of the
        pixels of its segment, ignoring the `nan` values. It is
        converted with `_compute_mag` using the zero point of the band,
        the extinction `A` and the distance of the field.

        If the field has the attribute `distance` the result is an
        absolute magnitude. Otherwise the distance is set to 10 pc and
        the result is the extinction-corrected apparent magnitude (see
        `_compute_mag`).

        Returns
        -------
        numpy.ndarray
            The magnitudes, shape `(n_sources, n_bands)`.
        """

        if hasattr(self.field, 'distance') is False:
            dist = 10 * u.pc
        else:
            dist = self.field.distance

        # eps = [np.nansum(source.data, axis=(1,2), where=source.segment) for source in self.sources]

        # Total counts inside the segment, per source and per band.
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
        """
        Compute the mean surface brightness of every source.

        The mean counts per pixel of a source in a band are the sum of
        the counts inside its segment divided by the number of pixels
        of the segment (`source.area`). They are converted with
        `_compute_sb`, which applies the zero point and the pixel scale
        of the band, the extinction `A` and the cosmological dimming at
        the redshift of the field. If the field has no attribute
        `redshift` the redshift is 0 and no dimming is removed.

        Returns
        -------
        numpy.ndarray
            The mean surface brightnesses in mag / arcsec^2, shape
            `(n_sources, n_bands)`.
        """

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
        """
        Compute the peak surface brightness of every source.

        The peak of a source in a band is the largest value, in counts
        per pixel, of its cutout inside its segment. It is converted with
        `_compute_sb` exactly as in `sb_mean`.

        Returns
        -------
        numpy.ndarray
            The peak surface brightnesses in mag / arcsec^2, shape
            `(n_sources, n_bands)`.
        """

        if hasattr(self.field, 'redshift') is False:
            redshift = 0
        else:
            redshift = self.field.redshift


        eps = np.zeros((len(self.sources), len(self.bands)))

        for i, source in enumerate(self.sources):
            for j, image in enumerate(source.data):

                eps[i, j] = np.nanmax(np.where(source.segment, image.data, np.nan))

        sb_peak = np.zeros((len(eps), len(self.bands)))
        for i, _eps, _A in zip(range(len(eps)), eps, self.A):
            sb_peak[i] = _compute_sb(_eps, self.zpt, self.scale_arcsec, _A, redshift)

        return sb_peak

    ### UNTESTED FUNCTIONS ###

    def _centroid_quad(self):
        """
        Compute the quadratic centroid of every source in every band.

        The centroid is found with `centroid_quadratic` of
        `photutils.centroids`, which fits a 2D quadratic polynomial to
        the cutout of the source and returns the position of its
        maximum. The pixels outside the segment are masked, the
        initial guess is the centre of the bounding box and the fitting
        box is the longer side of the cutout, made odd. When the fit
        raises a `ValueError` or returns `nan`, the centre of the
        bounding box is used instead.

        The six values stored for each source and band are, in order:

        - 0, 1: the (x, y) centroid in pixels relative to the lower
          left corner of the bounding box of the source;
        - 2, 3: the (x, y) centroid in pixels in the full image,
          0-based;
        - 4, 5: the right ascension and declination in deg, from the
          WCS of the header of the sources.

        Returns
        -------
        numpy.ndarray
            The centroids, shape `(n_sources, n_bands, 6)`.
        """
        centroids_quad = np.zeros((len(self.sources), len(self.bands), 6))

        for i, source in enumerate(self.sources):

            bbox = source.bbox

            for j, image in enumerate(source.data):

                # Centre of the bounding box, relative to its corner.
                guess = [source.centre[0] - source.bbox.ixmin, source.centre[1] - source.bbox.iymin]

                # True where the pixel is outside the segment.
                mask = np.where(source.segment>0, 0, 1).astype(bool)

                # The fitting box size must be an odd number of pixels.
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

                # Shift from cutout to full-image pixel coordinates.
                centroids_quad[i, j, 2] = centroids_quad[i, j, 0] + bbox.ixmin
                centroids_quad[i, j, 3] = centroids_quad[i, j, 1] + bbox.iymin


        # Sky coordinates of all the centroids, in one call.
        if len(self.sources) > 0:
            ra, dec = self.sources.wcs.all_pix2world(centroids_quad[:, :, 2].ravel(), centroids_quad[:, :, 3].ravel(), 0)

            centroids_quad[:, :, 4] = np.reshape(ra, centroids_quad.shape[:2])
            centroids_quad[:, :, 5] = np.reshape(dec, centroids_quad.shape[:2])

        return centroids_quad

    def rel_x_centroid_quad(self):
        """
        Return the x centroid of every source relative to its cutout.

        The centroids cached by `mag_aper` are used when available,
        otherwise they are computed with `_centroid_quad`.

        Returns
        -------
        numpy.ndarray
            The x coordinates in pixels from the left edge of the
            bounding box, shape `(n_sources, n_bands)`.
        """

        if hasattr(self, '_centroids'):
            centroids_quad = self._centroids
        else:
            centroids_quad = self._centroid_quad()

        return centroids_quad[:, :, 0]

    def rel_y_centroid_quad(self):
        """
        Return the y centroid of every source relative to its cutout.

        The centroids cached by `mag_aper` are used when available,
        otherwise they are computed with `_centroid_quad`.

        Returns
        -------
        numpy.ndarray
            The y coordinates in pixels from the bottom edge of the
            bounding box, shape `(n_sources, n_bands)`.
        """

        if hasattr(self, '_centroids'):
            centroids_quad = self._centroids
        else:
            centroids_quad = self._centroid_quad()

        return centroids_quad[:, :, 1]

    def x_centroid_quad(self):
        """
        Return the x centroid of every source in the full image.

        The centroids cached by `mag_aper` are used when available,
        otherwise they are computed with `_centroid_quad`.

        Returns
        -------
        numpy.ndarray
            The 0-based x pixel coordinates, shape
            `(n_sources, n_bands)`.
        """

        if hasattr(self, '_centroids'):
            centroids_quad = self._centroids
        else:
            centroids_quad = self._centroid_quad()

        return centroids_quad[:, :, 2]

    def y_centroid_quad(self):
        """
        Return the y centroid of every source in the full image.

        The centroids cached by `mag_aper` are used when available,
        otherwise they are computed with `_centroid_quad`.

        Returns
        -------
        numpy.ndarray
            The 0-based y pixel coordinates, shape
            `(n_sources, n_bands)`.
        """

        if hasattr(self, '_centroids'):
            centroids_quad = self._centroids
        else:
            centroids_quad = self._centroid_quad()

        return centroids_quad[:, :, 3]

    def ra_centroid_quad(self):
        """
        Return the right ascension of the centroid of every source.

        The centroids cached by `mag_aper` are used when available,
        otherwise they are computed with `_centroid_quad`.

        Returns
        -------
        numpy.ndarray
            The right ascensions in deg, shape `(n_sources, n_bands)`.
        """

        if hasattr(self, '_centroids'):
            centroids_quad = self._centroids
        else:
            centroids_quad = self._centroid_quad()

        return centroids_quad[:, :, 4]

    def dec_centroid_quad(self):
        """
        Return the declination of the centroid of every source.

        The centroids cached by `mag_aper` are used when available,
        otherwise they are computed with `_centroid_quad`.

        Returns
        -------
        numpy.ndarray
            The declinations in deg, shape `(n_sources, n_bands)`.
        """

        if hasattr(self, '_centroids'):
            centroids_quad = self._centroids
        else:
            centroids_quad = self._centroid_quad()

        return centroids_quad[:, :, 5]


    def mag_aper(self):
        """
        Compute the aperture magnitude of every source.

        The flux of a source in a band is the sum of the counts of the
        pixels of its cutout whose centre lies within `radius` pixels
        of the quadratic centroid computed by `_centroid_quad`. All the
        pixels of the aperture are summed, whether or not they belong
        to the segment, but the aperture is truncated at the edges of
        the cutout, which is the bounding box of the source. The flux
        is converted with `_compute_mag` as in `mag`, so the result is
        an absolute magnitude only if the field has a `distance`.

        The centroids are stored in `_centroids`, and their cutout
        coordinates in `_rel_centroids`, so that the centroid methods
        called afterwards do not compute them again.

        Returns
        -------
        numpy.ndarray
            The aperture magnitudes, shape `(n_sources, n_bands)`.
        """

        if hasattr(self.field, 'distance') is False:
            dist = 10 * u.pc
        else:
            dist = self.field.distance

        self._centroids = self._centroid_quad()

        self._rel_centroids = self._centroids[:, :, :2]

        eps = np.zeros((len(self.sources), len(self.bands)))
        for i, (source, centroid_bands) in enumerate(zip(self.sources, self._rel_centroids)):
            for j, (image, centroid) in enumerate(zip(source.data, centroid_bands)):

                y, x = np.indices(image.data.shape)

                distance = np.sqrt((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2)  # Distance from the centroid, in pixels.

                aperture_mask = distance <= self.radius

                eps[i, j] = np.nansum(image.data, where=aperture_mask)

        mag_aper = np.zeros((len(eps), len(self.bands)))
        for i, _eps, _A in zip(range(len(eps)), eps, self.A):
            mag_aper[i] = _compute_mag(_eps, self.zpt, _A, dist)

        return mag_aper

    ### END OF UNTESTED FUNCTIONS ###

    def _sum(self):
        """
        Compute the total counts of every source inside its segment.

        Returns
        -------
        numpy.ndarray
            The sums in counts, ignoring `nan` values, shape
            `(n_sources, n_bands)`.
        """
        eps = np.zeros((len(self.sources), len(self.bands)))

        for i, source in enumerate(self.sources):
            for j, image in enumerate(source.data):

                eps[i, j] = np.nansum(image.data, where=source.segment)

        sum = np.zeros((len(eps), len(self.bands)))
        for i, _eps, _A in zip(range(len(eps)), eps, self.A):
            sum[i] = _eps

        return sum

    def _mean(self):
        """
        Compute the mean counts per pixel of every source.

        The mean is taken over the pixels of the segment, ignoring the
        `nan` values. Unlike the other measurements, this method
        reduces `source.data` over its last two axes, so the cutouts
        must be a `numpy.ndarray` of shape `(n_bands, n_y, n_x)`.

        Returns
        -------
        numpy.ndarray
            The means in counts per pixel, shape
            `(n_sources, n_bands)`.
        """

        eps = [np.nanmean(source.data , axis=(1,2), where=source.segment) for source in self.sources]

        mean = np.zeros((len(eps), len(self.bands)))
        for i, _eps, _A in zip(range(len(eps)), eps, self.A):
            mean[i] = _eps

        return mean

    def _std(self):
        """
        Compute the standard deviation of the counts of every source.

        The standard deviation is taken over the pixels of the segment,
        ignoring the `nan` values. As in `_mean`, the cutouts must be a
        `numpy.ndarray` of shape `(n_bands, n_y, n_x)`.

        Returns
        -------
        numpy.ndarray
            The standard deviations in counts per pixel, shape
            `(n_sources, n_bands)`.
        """
        eps = [np.nanstd(source.data , axis=(1,2), where=source.segment) for source in self.sources]

        std = np.zeros((len(eps), len(self.bands)))
        for i, _eps, _A in zip(range(len(eps)), eps, self.A):
            std[i] = _eps

        return std

    def _table(self):
        """
        Run all the measurements and return them as a table.

        Returns
        -------
        pandas.DataFrame
            One row per source, with the columns described in
            `detection_properties`.
        """
        df = pd.DataFrame()

        for func in self.functions:

            results = np.asarray(func())

            if len(results.shape) == 1:
                df[func.__name__] = results
            else:
                # Transpose to iterate over the bands, not the sources.
                for band, result in zip(self.bands, results.T):
                    df[func.__name__ + '_' + band] = result

        return df

    def _stack(self, start, stop):
        """
        Return a copy of this object restricted to a stack of sources.

        Parameters
        ----------
        start, stop : int
            The stack holds the sources `start` to `stop - 1`.

        Returns
        -------
        Properties
            An object that measures only those sources, with the
            corresponding rows of the extinction `A`.
        """
        stack = copy(self)

        stack.sources = Sources(self.sources.sources[start:stop], self.sources.header)
        stack.A = np.asarray(self.A)[start:stop]

        # The measurement methods are bound to this object: bind the
        # same ones to the stack.
        stack.functions = [getattr(stack, func.__name__) for func in self.functions]

        for cached in ('_centroids', '_rel_centroids'):
            stack.__dict__.pop(cached, None)

        return stack

    def detection_properties(self, n_workers=1, stack_size=64):
        """
        Run all the measurements and attach the table to the sources.

        Every callable of `functions` is run in order. A result with
        one value per source becomes a column named after the method
        (for example `label`, `area`); a result of shape
        `(n_sources, n_bands)` becomes one column per band, named
        `<method>_<band>` (for example `mag_g`, `sb_mean_r`). The rows
        follow the order of the sources.

        The table is stored in `sources.properties`, replacing any
        previous one; `Sources.save_light` writes it to
        `properties.csv`.

        With `n_workers` larger than 1 the sources are split into
        stacks of `stack_size` consecutive sources, which are measured
        by a pool of worker processes; the stacks are handed out one
        at a time, so a stack with a very large source does not hold
        the others back. The table is the same as with one worker.
        The worker processes are forked, so that they share the
        images without copying them; where forking is not available
        the measurements run in this process.

        Parameters
        ----------
        n_workers : int
            The number of worker processes.
        stack_size : int
            The number of sources in a stack.

        Returns
        -------
        Sources
            The same `sources` object given to the constructor, with
            the `properties` attribute set to the `pandas.DataFrame`
            of the measurements.
        """

        n_sources = len(self.sources)

        if n_workers > 1 and n_sources > stack_size and 'fork' in multiprocessing.get_all_start_methods():
            global _SHARED_PROPERTIES

            bounds = [(start, min(start + stack_size, n_sources)) for start in range(0, n_sources, stack_size)]

            _SHARED_PROPERTIES = self

            try:
                with multiprocessing.get_context('fork').Pool(n_workers) as pool:
                    tables = list(pool.imap(_stack_table, bounds))
            finally:
                _SHARED_PROPERTIES = None

            df = pd.concat(tables, ignore_index=True)

        else:
            df = self._table()

        self.sources.properties = df
        return self.sources
