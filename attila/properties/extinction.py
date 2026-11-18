


from dustmaps.sfd import SFDQuery

from astropy.coordinates import SkyCoord

from attila.field.field import Field
from attila.tiling.sources import Sources

from dust_extinction.parameter_averages import CCM89, F99, G23
from synphot import units, config
from synphot import SourceSpectrum,SpectralElement,Observation,ExtinctionModel1D
from synphot.models import BlackBodyNorm1D, ConstFlux1D
from synphot.spectrum import BaseUnitlessSpectrum
from synphot.reddening import ExtinctionCurve

from astropy import units as u

import numpy as np
from tqdm import tqdm

import shapely as sh

from os import path

class Extinction:
    def __init__(self, field:Field, extinction_map_name, filters_folder=None, extinction_maps_folder=None, bands=None):
        self.field = field
        self.extinction_map_name = extinction_map_name

        if filters_folder is None:
            self.filters_folder = path.join('..', '..', 'synphot_filters')
        else:
            self.filters_folder = filters_folder

        if extinction_maps_folder is None:
            self.extinction_maps_folder = path.join('..', '..', 'dustmaps')
        else:
            self.extinction_maps_folder = extinction_maps_folder

        if extinction_map_name == 'sfd':
            self.extinction_map = SFDQuery(path.join(self.extinction_maps_folder, 'sfd'))
        else:
            raise NotImplementedError("Only SFD extinction map is implemented")
        
        if bands is None:
            self.bands = self.field.bands
        else:
            self.bands = bands

        filter_paths = [path.join(self.filters_folder, f'{band}.dat') for band in self.bands]
        self.filters = [SpectralElement.from_file(filter_path) for filter_path in filter_paths]

        wav_min = np.min([filter.waverange.min().value for filter in self.filters])
        wav_max = np.max([filter.waverange.max().value for filter in self.filters])

        self.wavs = np.linspace(wav_min, wav_max, 1000) * u.AA

        self.sp = SourceSpectrum(ConstFlux1D, amplitude=1)

        self.ext = G23(Rv=3.1)


    def compute_extinction_from_sources(self, sources:Sources):
        
        self.A = np.zeros(len(sources), (len(self.bands)))

        centres = [source.centre for source in sources]

        wcs = self.sources.wcs

        ra_dec = np.asarray([wcs.all_pix2world(x, y, 0) for x, y in centres])
        
        coords = SkyCoord(ra_dec[:, 0], ra_dec[:, 1], unit='deg')

        Ebvs = self.extinction_map(coords)

        for i, Ebv in tqdm(enumerate(Ebvs), total=len(Ebvs)):

            frac_ext = self.ext.extinguish(self.wavs, Ebv=Ebv)

            ext_model = ExtinctionCurve(
                ExtinctionModel1D, 
                points=self.wavs, 
                lookup_table=frac_ext
            )

            self.sp_ext = self.sp * ext_model

            for filter in self.filters:

                mag = Observation(self.sp, filter).effstim().value
                mag_ext = Observation(self.sp_ext, filter).effstim().value

                self.A[i, self.bands.index(filter.name)] = abs(mag_ext - mag)
               

    def compute_extinction_from_tiling(self, coord_list, regions):

        self.regions = regions

        coords = SkyCoord(
            ra=[coord.ra for coord in coord_list],
            dec=[coord.dec for coord in coord_list]
        )

        Ebvs = self.extinction_map(coords)
        self.Ebvs = Ebvs

        self.A_grid = np.zeros((len(coords), len(self.bands)))

        for i, Ebv in tqdm(enumerate(Ebvs), total=len(Ebvs)):

            frac_ext = self.ext.extinguish(self.wavs, Ebv=Ebv)

            ext_model = ExtinctionCurve(
                ExtinctionModel1D, 
                points=self.wavs, 
                lookup_table=frac_ext
            )

            self.sp_ext = self.sp * ext_model

            for filter, band in zip(self.filters, self.bands):

                mag = Observation(self.sp, filter).effstim().value
                mag_ext = Observation(self.sp_ext, filter).effstim().value

                self.A_grid[i, self.bands.index(band)] = abs(mag - mag_ext)

        # return A_grid

        """
        Add the lookup four each source position
        """

    def apply_grid_to_sources(self, sources):

        # A = np.zeros((len(sources), len(self.bands)))

        sources_coords = np.asarray([source.centre for source in sources])

        wcs = sources.wcs

        ra_dec = np.asarray([wcs.all_pix2world(x, y, 0) for x, y in sources_coords])

        points = [sh.geometry.Point(ra, dec) for ra, dec in ra_dec]
        tree = sh.strtree.STRtree(self.regions)

        idxs = tree.query(points, predicate='within')

        return self.A_grid[idxs[1]]

            




            
