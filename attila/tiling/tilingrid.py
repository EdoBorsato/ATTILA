"""
Sky tiling of a multi-band field and mosaicking of tile results.

A wide-field mosaic is too large to be processed in one go, so the
pipeline splits it into rectangular tiles defined on the sky. This
module provides:

- `Tiling`, which computes the sky region common to all bands, lays
  out the tiles (a regular overlapping grid or user-given centres),
  extracts the per-tile cutouts with `astropy.nddata.Cutout2D` and
  pastes per-tile results back into full-size arrays;
- `guess_shape`, which proposes a tile size and grid layout matching
  a target tile area;
- private helpers that build the tile polygons with `shapely`.

All tile geometry is expressed in degrees of right ascension and
declination. In the pipeline flow the module sits between `Field`
(the images) and the detection step that runs on every tile.
"""
import numpy as np
import numpy.typing as npt

from typing import Tuple, Literal, Optional, List, Dict, Callable, Union

from attila.tiling.tiles import BaseTile

import astropy.units as u
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.nddata.utils import add_array, overlap_slices
from astropy.io import fits
import warnings

import shapely as sh
from shapely.plotting import plot_polygon
import matplotlib.pyplot as plt
import pandas as pd

import warnings

from os import path
import pickle

# The supported ways of combining the footprints of the bands.
tiling_types = ['intersection']

# The supported tile layouts: a regular grid or given centres.
grid_types = ['grid','sparse']

# LABELS
#
# Design note on the "label-like" combination of overlapping values,
# defined for segmentation maps and similar integer arrays where the
# background (no source) is 0 and the sources are positive integers.
#
# If the two overlapping labels are equal: do nothing.
# If one is 0 but not both: pick the larger one.
# If they are both different and non-zero: pick the second.
#
# The latter condition is in line with how the matching will be done:
#
#     match 1 --> 2; combine 1 and 2; match 2 --> 1
#
# In `_combine_labels` below, the last case returns the larger label.

def _combine_labels(label1, label2):
    """
    Combine two overlapping integer labels into one.

    This is the "label-like" combination intended for segmentation maps
    and similar integer arrays, where 0 marks the background (no source)
    and positive integers mark the sources.

    - If the two labels are equal, that label is returned.
    - If exactly one of them is 0, the non-zero (larger) one is
      returned.
    - If they are different and both non-zero, the larger one is
      returned.

    The function works on scalars, not on arrays.

    Parameters
    ----------
    label1 : int
        The label already present in the mosaic.
    label2 : int
        The label coming from the tile being added.

    Returns
    -------
    int
        The combined label.
    """

    if label1 == label2:
        return label2
    
    elif (label1 != label2) & (label1 * label2 == 0):
        return max([label1, label2])
    
    elif (label1 != label2) & (label1 * label2 != 0):
        return max([label1, label2])

def _add_array_func(array_large, array_small, position, func):
    """
    Combine a small array into a larger one at a given position.

    This is a variant of `astropy.nddata.utils.add_array` in which the
    overlapping region is not summed but combined with a user function:
    `array_large[region] = func(array_large[region],
    array_small[region])`. The part of the small array falling outside
    the large array is ignored. The large array is modified in place.

    Parameters
    ----------
    array_large : numpy.ndarray
        The large array, modified in place.
    array_small : numpy.ndarray
        The small array to combine into `array_large`. It must not be
        larger than `array_large` along any axis.
    position : tuple
        The pixel position of the centre of `array_small` in the
        coordinates of `array_large`, in `(y, x)` order.
    func : callable
        The function `func(large_region, small_region)` returning the
        combined values of the overlapping region.

    Returns
    -------
    numpy.ndarray
        The modified `array_large`.

    Raises
    ------
    ValueError
        If `array_small` is larger than `array_large` along any axis.
    """


    # Check if large array is not smaller
    if all(
        large_shape >= small_shape
        for (large_shape, small_shape) in zip(array_large.shape, array_small.shape)
    ):
        
        large_slices, small_slices = overlap_slices(
            array_large.shape, array_small.shape, position
        )

        array_large[large_slices] = func(array_large[large_slices], array_small[small_slices])
        return array_large
    else:
        raise ValueError("Can't add array. Small array too large.")
    

def _add_quantity_tuple(t, unit):
    """Return the tuple `t` with every element multiplied by `unit`."""
    return tuple([elem * unit for elem in t])

def _make_grid_tiling(
        polygon: sh.Polygon,
        shape: Union[Tuple[float, float], List[Tuple[float, float]]],
        overlap: float,
        ):
    """
    Lay out a regular grid of overlapping tiles over a sky polygon.

    The grid starts at the corner of the polygon bounding box with the
    minimum right ascension and declination and advances by
    `side - overlap` along each axis, so that neighbouring tiles share a
    strip of width `overlap`. The tiles are rectangles in the (RA, Dec)
    coordinate plane: no `cos(Dec)` factor is applied to the RA side.

    A tile is kept if it lies entirely within the polygon, or if it
    partially overlaps it and the common area is larger than one tenth
    of the tile area. The expected number of tiles (a float, computed
    from the bounding box) is printed.

    Parameters
    ----------
    polygon
        The sky region to tile, with vertices in (RA, Dec) degrees.
    shape
        The tile sides `(side_ra, side_dec)` in degrees.
    overlap
        The overlap between neighbouring tiles in degrees.

    Returns
    -------
    lskycoord : list of astropy.coordinates.SkyCoord
        The centre of every kept tile (ICRS).
    lpolygons : list of shapely.Polygon
        The footprint of every kept tile in (RA, Dec) degrees.
    """
            
    # Initialise the tile parameters from the polygon bounding box.
    ra_min,dec_min,ra_max,dec_max=polygon.bounds
    
    Deltara =(ra_max-ra_min)
    Deltadec=(dec_max-dec_min)

    side_ra=shape[0]
    side_dec=shape[1]
    
    Nmaxtile=(Deltara/(side_ra-overlap))*(Deltadec/(side_dec-overlap))
    print('Creating {} tilees'.format(Nmaxtile))
    
    lskycoord=[]
    lpolygons=[]

    # Create the tiles row by row, from the minimum declination up.
    while dec_min<dec_max:
        ra_min=polygon.bounds[0]
        while ra_min<ra_max:
            poly_tile=_make_tile_polygon((ra_min,dec_min),(side_ra,side_dec))
            # if sh.overlaps(intersection,poly_tile):
            if (sh.overlaps(polygon,poly_tile) and sh.intersection(polygon,poly_tile).area>poly_tile.area/10.) or sh.within(poly_tile,polygon):
                lskycoord.append(SkyCoord((ra_min+side_ra/2.)*u.deg, 
                                            (dec_min+side_dec/2.)*u.deg,
                                            frame="icrs", unit="deg"))
                lpolygons.append(poly_tile)
            
            ra_min+=side_ra-overlap
        dec_min+=side_dec-overlap
    
    return lskycoord, lpolygons

def _make_sparse_tiling(
        polygon: sh.Polygon,
        shapes: Union[Tuple[float, float], List[Tuple[float, float]]],
        centres: List[SkyCoord]
        ):
    """
    Build the tiles centred on a list of user-given sky positions.

    For every centre a rectangular tile is built in the (RA, Dec) plane.
    Tiles that neither overlap the polygon nor lie within it are
    discarded with a warning.

    If `shapes` is a tuple the same shape is used for all tiles. If it
    is a list, the polygon of tile `j` is built with `shapes[j]`, while
    its lower corner is computed from `shapes[0]` and `shapes[1]`.

    For every kept tile the whole `centres` object (not the single
    centre) is appended to the returned list of centres.

    Parameters
    ----------
    polygon
        The sky region to tile, with vertices in (RA, Dec) degrees.
    shapes
        The tile sides `(side_ra, side_dec)` in degrees, or a list with
        one such tuple per centre.
    centres
        The requested tile centres.

    Returns
    -------
    lskycoord : list
        One entry per kept tile (see above for its content).
    lpolygons : list of shapely.Polygon
        The footprint of every kept tile in (RA, Dec) degrees.
    """
    lskycoord, lpolygons = [], []

    if type(shapes) is tuple:
        for centre in centres:
            poly_tile = _make_tile_polygon((centre.ra.deg-shapes[0]/2., centre.dec.deg-shapes[1]/2.), shapes)

            if not (sh.overlaps(polygon, poly_tile) or sh.within(poly_tile, polygon)):
                warnings.warn("The tile with centre {} do not overlap with the wcs_dict. The centre is removed.".format(centre))
                                                                                                                       
            else:
                lskycoord.append(centres)
                lpolygons.append(poly_tile)
    
    if type(shapes) is list:
        for j, centre in enumerate(centres):
            poly_tile = _make_tile_polygon((centre.ra.deg-shapes[0]/2., centre.dec.deg-shapes[1]/2.), shapes[j])

            if not (sh.overlaps(polygon, poly_tile) or sh.within(poly_tile, polygon)):
                warnings.warn("The tile with centre {} do not overlap with the wcs_dict. The centre is removed.".format(centre))
                                                                                                                       
            else:
                lskycoord.append(centres)
                lpolygons.append(poly_tile)

    return lskycoord,lpolygons
 

def _make_tile_polygon(corner: Tuple[float, float],
                      shape: Tuple[float, float]
                      ) -> sh.Polygon:
    """
    Return the rectangular polygon of a tile in the (RA, Dec) plane.

    Parameters
    ----------
    corner
        The `(ra, dec)` of the tile corner with minimum RA and Dec, in
        degrees.
    shape
        The tile sides `(side_ra, side_dec)` in degrees.

    Returns
    -------
    shapely.Polygon
        The closed rectangle with the four tile corners.
    """
    
    polygon=sh.Polygon([(corner[0],corner[1]),
                        (corner[0],corner[1]+shape[1]),
                        (corner[0]+shape[0],corner[1]+shape[1]),
                        (corner[0]+shape[0],corner[1]),
                        (corner[0],corner[1])])
    return polygon

def _check_nan_in_data(centres, polygons, coverage_mask_dict, shape, wcs_dict):
    """
    Select tiles according to the coverage of the data.

    For every tile and every band the coverage mask is cut out on the
    tile footprint (`Cutout2D`, `mode="partial"`) and the fraction of
    non-zero pixels is computed. A tile is flagged in a band when this
    fraction is lower than or equal to 0.9. The tiles returned are those
    flagged in all the bands; the other tiles are dropped.

    Parameters
    ----------
    centres : list of astropy.coordinates.SkyCoord
        The tile centres.
    polygons : list of shapely.Polygon
        The tile footprints, in the same order as `centres`.
    coverage_mask_dict : dict
        For every band, the full-image coverage mask (non-zero where
        there are valid data).
    shape : tuple of float
        The tile shape in degrees, passed to `Cutout2D` as its `size`.
    wcs_dict : dict
        For every band, the `astropy.wcs.WCS` of the full image.

    Returns
    -------
    updated_centres : list
        The centres of the selected tiles.
    updated_polygons : list
        The footprints of the selected tiles.
    """
    bands = list(coverage_mask_dict.keys())

    indexes = []

    for i, centre in enumerate(centres):

        for band in bands:
            wcs = wcs_dict[band]

            coverage_mask = coverage_mask_dict[band]
            coverage_cutout = Cutout2D(coverage_mask, centre, _add_quantity_tuple(shape, u.deg), wcs=wcs, mode='partial')

            # Flag the tile in this band if at most 90% of its pixels
            # are covered.
            # if not coverage_cutout.data.all():
            if np.count_nonzero(coverage_cutout.data)/coverage_cutout.data.size <= 0.9:
                indexes.append(i)
          
    
    # Keep the indexes that were flagged in every band.
    unique_indexes, N = np.unique(indexes, return_counts=True)
   
    idx = np.where(N == len(bands))[0]

    intersection_unique_indexes = unique_indexes[idx]

    updated_centres = np.asarray(centres, dtype=object)[intersection_unique_indexes]
    updated_polygons = np.asarray(polygons, dtype=object)[intersection_unique_indexes]

    return updated_centres.tolist(), updated_polygons.tolist()     

class Tiling:
    """
    A tiling of the sky region shared by a set of single-band images.

    The object is built from the FITS headers of the bands. It computes
    the intersection of the band footprints, lays out the tiles with
    `make_tiles`, returns the data of one tile with `get_tile_data` and
    writes the result of a tile back into full-image arrays with
    `tile_to_mosaic`.

    Attributes
    ----------
    header_dict : dict
        For every band, the FITS header of the full image. Setting it
        also rebuilds `wcs_dict`.
    wcs_dict : dict
        For every band, the `astropy.wcs.WCS` built from the header.
    tile_shape_sky : tuple of float
        The tile sides in degrees.
    overlap_sky : float
        The overlap between neighbouring tiles in degrees.
    grid_type : str
        Either `"grid"` (regular grid) or `"sparse"` (tiles centred on
        given positions).
    centres : list of astropy.coordinates.SkyCoord
        The tile centres. Before `make_tiles` it holds the value passed
        to the constructor.
    polygons : list of shapely.Polygon
        The tile footprints in (RA, Dec) degrees, set by `make_tiles`.
    wcs_polygon_list : list of shapely.Polygon
        The sky footprint of every band.
    wcs_intersection : shapely.Polygon
        The sky region common to all bands.
    coverage_mask_dict : dict
        The coverage masks, only present if passed to `make_tiles`.
    """

    def __init__(
            self, 
            header_dict: Dict[str, fits.Header],
            tile_shape_sky: Union[Tuple[Quantity, Quantity], List[Tuple[Quantity, Quantity]]], #or List(Tuple[Quantity, Quantity])
            overlap_sky: Quantity = 0 * u.arcsec, 
            grid_type: Literal[grid_types] = "grid", 
            centres: SkyCoord = None, 
        ) -> None:
        """
        Store the tiling parameters and intersect the band footprints.

        Parameters
        ----------
        header_dict
            For every band, the FITS header of the full image, from
            which the WCS is built.
        tile_shape_sky
            The two tile sides as angular quantities, converted to
            degrees. The grid tiling uses the first as the RA side and
            the second as the Dec side. A list of such tuples is
            accepted by the setter but not by `make_tiles`.
        overlap_sky
            The overlap between neighbouring tiles as an angular
            quantity, converted to degrees.
        grid_type
            `"grid"` for a regular grid covering the common footprint,
            `"sparse"` for tiles centred on `centres`.
        centres
            The tile centres, required if `grid_type` is `"sparse"`.
            With `"grid"` they are computed by `make_tiles`.

        Raises
        ------
        ValueError
            If the footprints of the bands have no common region.
        """
        
        self.header_dict = header_dict
        self.tile_shape_sky = tile_shape_sky
        self.overlap_sky = overlap_sky
        self.centres = centres
        self.grid_type = grid_type
        # TODO: add a check on `tile_shape_sky`.

        # Compute the intersection between the sky footprints of the
        # WCSs of all bands.

        self.wcs_polygon_list = [sh.Polygon(self.wcs_dict[band].calc_footprint()) for band in self.wcs_dict.keys()]

        self.wcs_intersection = sh.intersection_all(self.wcs_polygon_list)

        if self.wcs_intersection.is_empty:
                raise ValueError("Empty intersection between datasets.")

    @property
    def header_dict(self):
        """The FITS header of the full image of every band."""
        return self._header_dict
    
    @header_dict.setter
    def header_dict(self, header_dict):
        """Set the headers and rebuild the WCS of every band."""
        self._header_dict = header_dict
        self._wcs_dict = {band: WCS(header) for band, header in header_dict.items()}

    @property
    def wcs_dict(self):
        """The WCS of the full image of every band."""
        return self._wcs_dict
    
    # @wcs_dict.setter
    # def wcs_dict(self):
    #     self._wcs_dict = {band: WCS(header) for band, header in self.header_dict.items()}

    @property
    def tile_shape_sky(self):
        """The tile sides in degrees, as plain floats."""
        return self._tile_shape_sky
    
    @tile_shape_sky.setter
    def tile_shape_sky(self, tile_shape_sky):
        """
        Set the tile shape, converting the sides to degrees.

        A tuple of two angular quantities is stored as a tuple of floats
        in degrees. A list is checked to contain only tuples and is then
        stored as given, without conversion. Any other type leaves the
        attribute unset.

        Parameters
        ----------
        tile_shape_sky : tuple or list of tuple
            The two tile sides as angular `astropy.units.Quantity`.

        Raises
        ------
        ValueError
            If a list is given and one of its elements is not a tuple.
        """
        if type(tile_shape_sky) is tuple:
            
            self._tile_shape_sky = tuple([elem.to(u.deg).value for elem in tile_shape_sky])

        elif type(tile_shape_sky) is list:
            for tile_shape_sky_i in tile_shape_sky:
                if type(tile_shape_sky_i) is not tuple:
                    raise ValueError("tile_shape_sky must be a list of tuples of two angular astropy.units.Quantity")
                
                else:
                    self._tile_shape_sky = tuple([elem.to(u.deg).value for elem in tile_shape_sky])
                    
            self._tile_shape_sky = tile_shape_sky

    @property
    def overlap_sky(self):
        """The overlap between neighbouring tiles in degrees."""
        return self._overlap_sky
    
    @overlap_sky.setter
    def overlap_sky(self, overlap_sky):
        """Set the overlap, converting the quantity to degrees."""
        self._overlap_sky = overlap_sky.to(u.deg).value

    @property
    def grid_type(self):
        """The type of tiling, `"grid"` or `"sparse"`."""
        return self._grid_type
    
    @grid_type.setter
    def grid_type(self, grid_type):
        """
        Set the type of tiling, checking it is one of `grid_types`.
        """
        if grid_type in grid_types:
            self._grid_type = grid_type
        else:
            raise ValueError(f"grid_type must be one of {grid_types}")

    def add_band(self, band: str, header: fits.Header):
        """
        Return a new tiling that also includes an additional band.

        A new `Tiling` is created with the header of the new band added
        to a copy of `header_dict` and with the same tile shape,
        overlap, grid type and centres. Its tiles are then made again,
        so that they refer to the footprint common to all bands
        including the new one. The coverage masks are not passed on. The
        current object is not modified.

        Parameters
        ----------
        band
            The name of the band to add.
        header
            The FITS header of the full image in that band.

        Returns
        -------
        Tiling
            The new tiling, with `make_tiles` already called.
        """
        new_header_dict = self.header_dict.copy()
        new_header_dict[band] = header

        tile_shape_sky = tuple([tsp * u.deg for tsp in self.tile_shape_sky])
        overlap_sky = self.overlap_sky * u.deg

        new_tiling = Tiling(new_header_dict, tile_shape_sky, overlap_sky, self.grid_type, self.centres)

        new_tiling.make_tiles()

        return new_tiling

    def make_tiles(
            self, 
            coverage_mask_dict: Optional[Dict[str, npt.ArrayLike]] = None,
        ) -> None:
        """
        Compute the centres and footprints of the tiles.

        With `grid_type="grid"` a regular grid of overlapping tiles is
        laid over `wcs_intersection`; with `grid_type="sparse"` one tile
        is built around each of the `centres` given to the constructor.
        The results are stored in the attributes `centres` (list of
        `SkyCoord`) and `polygons` (list of `shapely.Polygon`).

        If coverage masks are given, now or in a previous call, the
        tiles are then selected with `_check_nan_in_data`.

        Parameters
        ----------
        coverage_mask_dict
            For every band, a full-image array that is non-zero where
            the band has valid data. If None, the masks stored by a
            previous call, if any, are used.

        Raises
        ------
        ValueError
            If `tile_shape_sky` is not a tuple.
        """

        if coverage_mask_dict is not None:
            self.coverage_mask_dict=coverage_mask_dict

        if type(self.tile_shape_sky) is not tuple:
            raise ValueError("get_tiling works only if tile_shape_sky is a tuple. Use make_tiling if you want tilees of different sizes")

        # if self.grid_type != "grid":
        #     raise ValueError("A tiling only work with a 'grid' grid type. If 'sparse' provide the centres")
        
        if self.grid_type == 'grid':
            self.centres, self.polygons = _make_grid_tiling(self.wcs_intersection, self.tile_shape_sky, self.overlap_sky)

        if self.grid_type == 'sparse':
            self.centres, self.polygons = _make_sparse_tiling(self.wcs_intersection, self.tile_shape_sky, self.centres)
            
        if hasattr(self, "coverage_mask_dict"):
            self.centres, self.polygons = _check_nan_in_data(self.centres, self.polygons, self.coverage_mask_dict, self.tile_shape_sky, self.wcs_dict)

    def get_tile_data(
            self, 
            index: int,
            Tile: BaseTile, 
            input_dict: Dict[str, Dict[str, npt.ArrayLike]],
        ) -> None:
        """
        Extract the data of one tile from full-image arrays.

        A tile object is created with the sky shape and overlap of the
        tiling (both in degrees) and, for every band and every named
        array of `input_dict`, the region of the tile is cut out with
        `Cutout2D` (`mode="partial"`, so that tiles crossing the image
        border are padded) and stored in the tile together with the WCS
        of the cutout.

        Parameters
        ----------
        index
            The index of the tile in `centres`.
        Tile
            The tile class to instantiate, `BaseTile` or a subclass.
        input_dict
            The full-image arrays as `input_dict[band][name]`. The
            arrays of a band must share the WCS of that band.

        Returns
        -------
        BaseTile
            The tile, with `data2D[band][name]` and `reg[band][name]`
            filled.

        Raises
        ------
        ValueError
            If the tile centres have not been computed or provided.
        """

        # warnings.warn("Bands in data_dicts should match the length of wcs_dict. Each array in the propoer band should have the same wcs.")

        if self.centres is None:
            raise ValueError("The centres of the tilees must be provided. Either use 'get_tiling' if grid_type is 'grid' or provide the 'centres' attribute")

        bands = list(input_dict.keys())

        centre = self.centres[index]

        tile = Tile(
            shape = self.tile_shape_sky, 
            overlap = self.overlap_sky,
            bands = bands,
            index = index
        )

        for band in bands:
            wcs = self.wcs_dict[band]

            data_names = list(input_dict[band].keys())

            for name in data_names:
                data = input_dict[band][name]

                data_cutout = Cutout2D(data, centre, _add_quantity_tuple(self.tile_shape_sky, u.deg), wcs=wcs, mode='partial')

                tile.add_data(band, name, data_cutout.data, data_cutout.wcs)

                # tile.add_reg(band, name, data_cutout.wcs)

        return tile
    
    def tile_to_mosaic(
            self, 
            tile: BaseTile,
            band: str,
            output_dict: Dict[str, Dict[str, npt.ArrayLike]],
            combine_func=None
            ) -> None:
        """
        Paste the arrays of a tile back into the full-image arrays.

        For every name in `output_dict[band]` the tile array
        `tile.data2D[band][name]` is placed in the full-image array at
        the pixel position of the tile centre. With `combine_func=None`
        the tile values are added to the existing ones
        (`astropy.nddata.utils.add_array`); otherwise the overlapping
        region is replaced by `combine_func(large_region,
        small_region)`. The arrays in `output_dict` are modified in
        place.

        Parameters
        ----------
        tile
            The tile holding the arrays to paste. Its `index` selects
            the tile centre.
        band
            The band whose arrays are pasted and whose WCS converts the
            tile centre to pixels.
        output_dict
            The full-image arrays as `output_dict[band][name]`. Every
            name must also be present in `tile.data2D[band]`.
        combine_func : callable, optional
            The function combining the existing and the tile values
            where they overlap. If None the values are summed.
        """
        
        # warnings.warn("The mosaic should have the same shape as the tile")
        
        large_wcs = self.wcs_dict[band]

        for key in output_dict[band].keys():
            
            position = self.centres[tile.index].to_pixel(large_wcs)[::-1] 
            # The [::-1] swaps the (x, y) pixel position into the
            # (y, x) order required by `add_array`.

            large_data = output_dict[band][key]
            small_data = tile.data2D[band][key]

            # Disabled: mask the pixels of the tile that fall outside
            # the intersection of the footprints.
            # Create a mask for the small_data array
            # small_wcs = tile.reg[band][key]
            # ny, nx = small_data.shape
            # y, x = np.mgrid[:ny, :nx]
            # ra, dec = small_wcs.wcs_pix2world(x, y, 0)
            # mask = np.array([self.wcs_intersection.contains(sh.Point(r, d)) for r, d in zip(ra.flat, dec.flat)])
            # mask = mask.reshape(small_data.shape)

            # # Set pixels outside the intersection to 0
            # small_data[~mask] = 0

            if combine_func is None:
                add_array(large_data, small_data, position)
            else:
                _add_array_func(large_data, small_data, position, combine_func)

    def plot_tiles(self, figsize: Tuple[int, int] = (10, 10)) -> None:
        """
        Plot the tiles over the common footprint in a WCS projection.

        The intersection of the band footprints and the footprint of
        every tile are drawn as filled polygons, and the index of each
        tile is written at its centre. The WCS of the first band of
        `wcs_dict` is used for the projection.

        Parameters
        ----------
        figsize
            The figure size `(width, height)` in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure.
        ax : astropy.visualization.wcsaxes.WCSAxes
            The axes with the plot.
        """

        band = list(self.wcs_dict.keys())[0]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': self.wcs_dict[band]})

        plot_polygon(self.wcs_intersection, ax=ax, fc='C0', ec='k', alpha=0.75, add_points=False, transform=ax.get_transform('world'))

        for polygon in self.polygons:
            plot_polygon(polygon, ax=ax, fc='C1', ec='k', alpha=0.25, add_points=False, transform=ax.get_transform('world'))

        ra_list = [centre.ra.deg for centre in self.centres]
        dec_list = [centre.dec.deg for centre in self.centres]

        for i, (ra, dec) in enumerate(zip(ra_list, dec_list)):
            ax.text(ra, dec, str(i), color='C1', fontsize=10, transform=ax.get_transform('world'))

        ax.set_aspect('equal')

        return fig, ax

    def load_tile(
            self,
            tiling_table: pd.DataFrame, 
            index: int,
            loader: Callable[[str, Dict[str, Union[str, int, Tuple[int, int]]]], BaseTile]
        ): 
        """
        Load a stored tile through a user-provided loader.

        The tile properties are read from `tiling_table` with the
        selection `tiling_table.loc["index" == index]`, converted to a
        dictionary and passed to `loader` together with the file name
        `"<index>.pkl"`.

        Parameters
        ----------
        tiling_table
            The table with the properties of the tiles.
        index
            The index of the tile to load.
        loader
            The function `loader(filename, tile_properties)` that
            returns the tile.

        Returns
        -------
        BaseTile
            The object returned by `loader`.
        """

        tile_properties = tiling_table.loc["index" == index].to_dict()

        return loader(f"{index}.pkl", tile_properties)

    def save(self, filename: str) -> None:
        """
        Pickle the tiling to `filename + ".pkl"`.

        Parameters
        ----------
        filename
            The output path without the `.pkl` extension.
        """

        file_path = path.join(filename + '.pkl')

        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        """
        Load a tiling pickled with `save`.

        Parameters
        ----------
        filename
            The path of the file without the `.pkl` extension.

        Returns
        -------
        Tiling
            The unpickled object.
        """

        file_path = path.join(filename + '.pkl')

        with open(file_path, 'rb') as f:
            return pickle.load(f)


def guess_shape(
        wcs: WCS, 
        area_target: Quantity,
        tolerance: float = 0.1, 
        overlap: Quantity = 45 * u.arcsec,
    ) -> Dict[str, Union[int, Quantity]]:
    """
    Find a tile size and grid layout matching a target tile area.

    The footprint of the image is taken from its WCS and its extent in
    right ascension and declination is computed (the RA extent handles
    the wrap at 360 deg by excluding the largest gap between corners; no
    `cos(Dec)` factor is applied). The image is then split in `nx` by
    `ny` tiles overlapping by `overlap`, with tile sides

        w = (ra_span + (nx - 1) * overlap) / nx
        h = (dec_span + (ny - 1) * overlap) / ny

    so that the tiles exactly cover the two extents.

    The total number of tiles `N = nx * ny` is increased from the
    minimum allowed by the upper area limit. For every `N`, all the
    factorisations are tried and those with a tile area strictly inside
    `area_target * (1 -/+ tolerance)` are kept. The search stops at the
    first `N` with at least one valid layout, and among those the one
    with the aspect ratio closest to 1 is returned (ties are broken by
    the smallest logarithmic distance of the area from the geometric
    centre of the allowed interval).

    Parameters
    ----------
    wcs
        The WCS of the full image.
    area_target
        The desired tile area, as an angular area quantity.
    tolerance
        The allowed fractional deviation of the tile area from
        `area_target`.
    overlap
        The overlap between neighbouring tiles, as an angular quantity.

    Returns
    -------
    dict
        The chosen layout: `nx` and `ny` (number of tiles along RA and
        Dec), `w` and `h` (tile sides along RA and Dec, in arcmin), `A`
        (tile area in arcmin^2), `ar` (aspect ratio, >= 1), `area_err`
        (logarithmic area error) and `N` (total number of tiles).

    Raises
    ------
    ValueError
        If the RA or Dec extent is not larger than the overlap.
    RuntimeError
        If no layout with a tile area in the allowed interval is found.
    """

    footprint = wcs.calc_footprint()



    # Extent of the footprint: the RA span is 360 deg minus the
    # largest gap between sorted corners, to handle the RA wrap.
    ra = np.mod(footprint[:, 0], 360.0)
    dec = footprint[:, 1]

    dec_span = (dec.max() - dec.min()) * u.deg

    ra_sorted = np.sort(ra)
    gaps = np.diff(np.r_[ra_sorted, ra_sorted[0] + 360.0])
    max_gap = gaps.max()
    ra_span = (360.0 - max_gap) * u.deg

        #dec_mean = np.deg2rad(dec.mean())
        #ra_width_true = (ra_span.to_value(u.deg) * np.cos(dec_mean)) * u.deg

    ra_span = ra_span.to(u.arcmin)
    dec_span = dec_span.to(u.arcmin)
    o = overlap.to(u.arcmin)

    A_min = (1 - tolerance) * area_target
    A_max = (1 + tolerance) * area_target

    A_min = A_min.to(u.arcmin*u.arcmin)
    A_max = A_max.to(u.arcmin*u.arcmin)

    A_tot = (ra_span * dec_span).to(u.arcmin*u.arcmin)

    if ra_span <= o or dec_span <= o:
        raise ValueError("Span troppo piccolo rispetto all'overlap.")

    # Smallest and largest number of tiles to try. The extra 200 is
    # a safety margin on the upper end of the search.
    N0 = max(1, int(np.ceil((A_tot/A_max).to_value(1))))

    Nmax = max(N0, int(np.ceil((A_tot/A_min).to_value(1))) + 200)

    best = None
    for N in range(N0, Nmax+1):
        candidates = []

        for nx in range(1, N+1):
            if N % nx != 0:
                continue
            ny = N // nx

            # Tile sides such that nx (ny) tiles overlapping by `o`
            # cover the RA (Dec) span exactly.
            w = (ra_span + (nx-1)*o)/nx
            h = (dec_span + (ny-1)*o)/ny

            if w <= o or h <= o:
                continue

            A = (w * h).to(u.arcmin * u.arcmin)
            if not (A > A_min and A < A_max):  
                continue

            # Aspect ratio (>= 1) and logarithmic distance of the
            # area from the geometric centre of [A_min, A_max].
            ar = max((w/h).to_value(1), (h/w).to_value(1)) 
            A_center = (A_min*A_max)**0.5
            area_err = abs(np.log((A/A_center).to_value(1)))  

            candidates.append(dict(nx=nx, ny=ny, w=w, h=h, A=A, ar=ar, area_err=area_err))

        if candidates:

            # Prefer the squarest tile, then the best area match.
            best = min(candidates, key=lambda d: (d['ar'], d['area_err']))
            break

    if best is None:
        raise RuntimeError("Nessuna soluzione trovata nell'intervallo di area. Allarga [A_min, A_max] o modifica overlap.")
    best['N'] = best['nx'] * best['ny']

    return best

        

                    
    