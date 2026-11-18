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

tiling_types = ['intersection']

grid_types = ['grid','sparse']

"""
LABELS

write a function to make a 'label-like' combination of overlapping values
this kind of combination is defined for segmentation maps and similar integer
arrays where the background, i.e. not source region is identified with a 0 
whereas sources with integers.

If the two overlapping labels are equal: do nothing.
If one is 0 but not both: pick the larger one.
If they are both different and non-zero: pick the second.

The latter condition is in line with how the matching will be done:

    match 1 --> 2; combine 1 and 2; match 2 --> 1
"""

def _combine_labels(label1, label2):

    if label1 == label2:
        return label2
    
    elif (label1 != label2) & (label1 * label2 == 0):
        return max([label1, label2])
    
    elif (label1 != label2) & (label1 * label2 != 0):
        return max([label1, label2])

def _add_array_func(array_large, array_small, position, func):

    """Add a smaller array at a given position in a larger array.

    Parameters
    ----------
    array_large : ndarray
        Large array.
    array_small : ndarray done in-place.

    -----
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
    return tuple([elem * unit for elem in t])

def _make_grid_tiling(
        polygon: sh.Polygon,
        shape: Union[Tuple[float, float], List[Tuple[float, float]]],
        overlap: float,
        ):
            
    #Initialize the tile parameters
    ra_min,dec_min,ra_max,dec_max=polygon.bounds
    
    Deltara =(ra_max-ra_min)
    Deltadec=(dec_max-dec_min)

    side_ra=shape[0]
    side_dec=shape[1]
    
    Nmaxtile=(Deltara/(side_ra-overlap))*(Deltadec/(side_dec-overlap))
    print('Creating {} tilees'.format(Nmaxtile))
    
    lskycoord=[]
    lpolygons=[]

    #Create the tilees 
    while dec_min<dec_max:
        ra_min=polygon.bounds[0]
        while ra_min<ra_max:
            poly_tile=_make_tile_polygon((ra_min,dec_min),(side_ra,side_dec))
            #if sh.overlaps(intersection,poly_tile):
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
    
    polygon=sh.Polygon([(corner[0],corner[1]),
                        (corner[0],corner[1]+shape[1]),
                        (corner[0]+shape[0],corner[1]+shape[1]),
                        (corner[0]+shape[0],corner[1]),
                        (corner[0],corner[1])])
    return polygon

def _check_nan_in_data(centres, polygons, coverage_mask_dict, shape, wcs_dict):
    """
    Check if there are nans in the data. If there are, remove the tilees.
    """
    bands = list(coverage_mask_dict.keys())

    indexes = []

    for i, centre in enumerate(centres):

        for band in bands:
            wcs = wcs_dict[band]

            coverage_mask = coverage_mask_dict[band]
            coverage_cutout = Cutout2D(coverage_mask, centre, _add_quantity_tuple(shape, u.deg), wcs=wcs, mode='partial')

            # if not coverage_cutout.data.all():
            if np.count_nonzero(coverage_cutout.data)/coverage_cutout.data.size <= 0.9:
                indexes.append(i)
          
    
    unique_indexes, N = np.unique(indexes, return_counts=True)
   
    idx = np.where(N == len(bands))[0]

    intersection_unique_indexes = unique_indexes[idx]

    updated_centres = np.asarray(centres, dtype=object)[intersection_unique_indexes]
    updated_polygons = np.asarray(polygons, dtype=object)[intersection_unique_indexes]

    return updated_centres.tolist(), updated_polygons.tolist()     

class Tiling:

    def __init__(
            self, 
            header_dict: Dict[str, fits.Header],
            tile_shape_sky: Union[Tuple[Quantity, Quantity], List[Tuple[Quantity, Quantity]]], #or List(Tuple[Quantity, Quantity])
            overlap_sky: Quantity = 0 * u.arcsec, 
            grid_type: Literal[grid_types] = "grid", 
            centres: SkyCoord = None, 
        ) -> None:
        
        self.header_dict = header_dict
        self.tile_shape_sky = tile_shape_sky
        self.overlap_sky = overlap_sky
        self.centres = centres
        self.grid_type = grid_type
    #aggiungere check tile_shape_sky

        """
        compute the intersection between the WCSs
        """

        self.wcs_polygon_list = [sh.Polygon(self.wcs_dict[band].calc_footprint()) for band in self.wcs_dict.keys()]

        self.wcs_intersection = sh.intersection_all(self.wcs_polygon_list)

        if self.wcs_intersection.is_empty:
                raise ValueError("Empty intersection between datasets.")

    @property
    def header_dict(self):
        return self._header_dict
    
    @header_dict.setter
    def header_dict(self, header_dict):
        self._header_dict = header_dict
        self._wcs_dict = {band: WCS(header) for band, header in header_dict.items()}

    @property
    def wcs_dict(self):
        return self._wcs_dict
    
    # @wcs_dict.setter
    # def wcs_dict(self):
    #     self._wcs_dict = {band: WCS(header) for band, header in self.header_dict.items()}

    @property
    def tile_shape_sky(self):
        return self._tile_shape_sky
    
    @tile_shape_sky.setter
    def tile_shape_sky(self, tile_shape_sky):
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
        return self._overlap_sky
    
    @overlap_sky.setter
    def overlap_sky(self, overlap_sky):
        self._overlap_sky = overlap_sky.to(u.deg).value

    @property
    def grid_type(self):
        return self._grid_type
    
    @grid_type.setter
    def grid_type(self, grid_type):
        if grid_type in grid_types:
            self._grid_type = grid_type
        else:
            raise ValueError(f"grid_type must be one of {grid_types}")

    def add_band(self, band: str, header: fits.Header):
        """
        Add a new band to the tiling, update the header_dict and wcs_dict, and remake the tiles.
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
        mode=='intersection': attribute intersection is created
        mode=='fill': attribute union is created
        attribute:
        polygons=shapely.geometry.multipolygon.MultiPolygon
        centres=List[astropy.coordinates.sky_coordinate.SkyCoord]
    
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

        #warnings.warn("Bands in data_dicts should match the length of wcs_dict. Each array in the propoer band should have the same wcs.")

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
        
        #warnings.warn("The mosaic should have the same shape as the tile")
        
        large_wcs = self.wcs_dict[band]

        for key in output_dict[band].keys():
            
            position = self.centres[tile.index].to_pixel(large_wcs)[::-1] 
            #[::-1] is because we want to transpose (x,y) into (y,x) as required by add_array

            large_data = output_dict[band][key]
            small_data = tile.data2D[band][key]

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
        Metodo per visualizzare i tiles creati con la proiezione WCS.
        Parametri:
        - band: La banda WCS da usare per la proiezione.
        - figsize: La dimensione della figura, come una tupla (width, height).
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

        tile_properties = tiling_table.loc["index" == index].to_dict()

        return loader(f"{index}.pkl", tile_properties)

    def save(self, filename: str) -> None:

        file_path = path.join(filename + '.pkl')

        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):

        file_path = path.join(filename + '.pkl')

        with open(file_path, 'rb') as f:
            return pickle.load(f)


def guess_shape(
        wcs: WCS, 
        area_target: Quantity,
        tolerance: float = 0.1, 
        overlap: Quantity = 45 * u.arcsec,
    ) -> Dict[str, Union[int, Quantity]]:

    footprint = wcs.calc_footprint()



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

    N0 = max(1, int(np.ceil((A_tot/A_max).to_value(1))))

    Nmax = max(N0, int(np.ceil((A_tot/A_min).to_value(1))) + 200)

    best = None
    for N in range(N0, Nmax+1):
        candidates = []

        for nx in range(1, N+1):
            if N % nx != 0:
                continue
            ny = N // nx

            w = (ra_span + (nx-1)*o)/nx
            h = (dec_span + (ny-1)*o)/ny

            if w <= o or h <= o:
                continue

            A = (w * h).to(u.arcmin * u.arcmin)
            if not (A > A_min and A < A_max):  
                continue

            ar = max((w/h).to_value(1), (h/w).to_value(1)) 
            A_center = (A_min*A_max)**0.5
            area_err = abs(np.log((A/A_center).to_value(1)))  

            candidates.append(dict(nx=nx, ny=ny, w=w, h=h, A=A, ar=ar, area_err=area_err))

        if candidates:

            best = min(candidates, key=lambda d: (d['ar'], d['area_err']))
            break

    if best is None:
        raise RuntimeError("Nessuna soluzione trovata nell'intervallo di area. Allarga [A_min, A_max] o modifica overlap.")
    best['N'] = best['nx'] * best['ny']

    return best

        

                    
    