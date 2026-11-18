"""
Galactic extinction of the sources in the bands of the field.

The `Extinction` class reads the reddening E(B-V) from the
Schlegel, Finkbeiner & Davis (1998, SFD) dust map through `dustmaps`
and turns it into a band extinction with the Gordon et al. (2023, G23)
Milky Way extinction curve at R_V = 3.1. The band value is obtained
with synthetic photometry (`synphot`): a flat spectrum is observed
through the filter transmission curve with and without the extinction,
and the two results are compared.

Since E(B-V) varies slowly over the sky, the pipeline does not query
the map at every source: `compute_extinction_from_tiling` evaluates the
extinction at the centres of a set of sky tiles and
`apply_grid_to_sources` gives every source the value of the tile that
contains it. The resulting array is assigned to
`attila.properties.properties.Properties.A`.
"""

from dustmaps.sfd import SFDQuery

from astropy.coordinates import SkyCoord

from attila.field.field import Field
from attila.tiling.sources import Sources

from dust_extinction.parameter_averages import G23
from synphot import SourceSpectrum,SpectralElement,Observation,ExtinctionModel1D
from synphot.models import ConstFlux1D
from synphot.reddening import ExtinctionCurve

from astropy import units as u

import numpy as np
from tqdm import tqdm

import shapely as sh

from os import path

# Root of the repository, which holds the filter curves and dust maps.
_REPO_ROOT = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))

class Extinction:
    """
    The Galactic extinction in a set of photometric bands.

    The object holds the dust map, the filter transmission curves, the
    reference spectrum and the extinction curve, and computes the
    extinction either on a grid of sky positions
    (`compute_extinction_from_tiling`, then `apply_grid_to_sources`) or
    at the position of every source
    (`compute_extinction_from_sources`).

    Attributes
    ----------
    field : Field
        The field the extinction refers to.
    extinction_map_name : str
        The name of the dust map; only `'sfd'` is available.
    filters_folder : str
        The folder with the filter transmission curves.
    extinction_maps_folder : str
        The folder with the dust maps.
    extinction_map : dustmaps.sfd.SFDQuery
        The dust map; calling it with a `SkyCoord` returns E(B-V) in
        magnitudes.
    bands : list of str
        The names of the filters, which are also the names of their
        files without the `.dat` extension.
    filters : list of synphot.SpectralElement
        The transmission curve of each band.
    wavs : astropy.units.Quantity
        The 1000 wavelengths, linearly spaced over the range covered
        by all the filters, on which the extinction curve is sampled.
    sp : synphot.SourceSpectrum
        The reference spectrum, constant with wavelength, with
        amplitude 1 in the default flux unit of `synphot`.
    ext : dust_extinction.parameter_averages.G23
        The extinction curve, with R_V = 3.1.
    regions : list of shapely.Polygon
        The sky footprints of the grid, set by
        `compute_extinction_from_tiling`.
    Ebvs : numpy.ndarray
        The E(B-V) at the centres of the grid, in magnitudes, shape
        `(n_regions,)`; set by `compute_extinction_from_tiling`.
    A_grid : numpy.ndarray
        The extinction at the centres of the grid in each band, shape
        `(n_regions, n_bands)`; set by
        `compute_extinction_from_tiling`.
    sp_ext : synphot.SourceSpectrum
        The reference spectrum with the extinction of the last
        processed position applied.
    """
    def __init__(self, field:Field, extinction_map_name, filters_folder=None, extinction_maps_folder=None, bands=None):
        """
        Load the dust map and the filters and set up the spectra.

        The SFD map is looked for in the sub-folder `sfd` of
        `extinction_maps_folder`, and the transmission curve of each
        band in the file `<band>.dat` of `filters_folder`, which is
        read with `synphot.SpectralElement.from_file`.

        Parameters
        ----------
        field
            The field the extinction refers to. It is only used to
            provide the default bands.
        extinction_map_name : str
            The dust map to use. Only `'sfd'` is implemented.
        filters_folder : str
            The folder with the filter transmission curves. `None`
            means the `synphot_filters` folder of the repository.
        extinction_maps_folder : str
            The folder with the dust maps. `None` means the `dustmaps`
            folder of the repository.
        bands : list of str
            The names of the filter files, without extension (for
            example `'SLOAN_SDSS.g'`), in the order wanted for the
            columns of the output. `None` means the bands of the field.

        Raises
        ------
        NotImplementedError
            If `extinction_map_name` is not `'sfd'`.
        """
        self.field = field
        self.extinction_map_name = extinction_map_name

        if filters_folder is None:
            self.filters_folder = path.join(_REPO_ROOT, 'synphot_filters')
        else:
            self.filters_folder = filters_folder

        if extinction_maps_folder is None:
            self.extinction_maps_folder = path.join(_REPO_ROOT, 'dustmaps')
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

        # Common wavelength grid spanning all the filters, in Angstrom.
        wav_min = np.min([filter.waverange.min().value for filter in self.filters])
        wav_max = np.max([filter.waverange.max().value for filter in self.filters])

        self.wavs = np.linspace(wav_min, wav_max, 1000) * u.AA

        # Flat reference spectrum the extinction is applied to.
        self.sp = SourceSpectrum(ConstFlux1D, amplitude=1)

        self.ext = G23(Rv=3.1)


    def _band_extinction(self, Ebv):
        """
        Return the extinction in every band for one reddening value.

        The G23 curve gives the fraction of flux transmitted by the
        dust at the wavelengths `wavs`. This is applied to the flat
        reference spectrum, and the spectrum is observed through each
        filter with and without the extinction
        (`synphot.Observation.effstim`, in photons). The extinction is
        the magnitude difference of the two,
        `A = -2.5 log10(F_ext / F)`.

        Parameters
        ----------
        Ebv : float
            The reddening E(B-V) in magnitudes.

        Returns
        -------
        numpy.ndarray
            The extinction in magnitudes, one value per band, in the
            order of `bands`.
        """
        # Fraction of flux transmitted by the dust at each wavelength.
        frac_ext = self.ext.extinguish(self.wavs, Ebv=Ebv)

        ext_model = ExtinctionCurve(
            ExtinctionModel1D,
            points=self.wavs,
            lookup_table=frac_ext
        )

        self.sp_ext = self.sp * ext_model

        A = np.zeros(len(self.bands))
        for j, filter in enumerate(self.filters):

            flux = Observation(self.sp, filter).effstim().value
            flux_ext = Observation(self.sp_ext, filter).effstim().value

            A[j] = -2.5 * np.log10(flux_ext / flux)

        return A

    def compute_extinction_from_sources(self, sources:Sources):
        """
        Compute the extinction at the position of every source.

        This is the per-source alternative to the grid of
        `compute_extinction_from_tiling`; the pipeline in
        `attila.pipelines` uses the grid. The pixel centre of every
        source is converted to sky coordinates with the WCS of the
        sources, E(B-V) is read from the dust map at each position and
        the band values are derived with `_band_extinction`. They are
        stored in the attribute `A`, with one row per source and one
        column per band; nothing is returned.

        Parameters
        ----------
        sources
            The sources whose pixel centres are looked up.
        """

        self.A = np.zeros((len(sources), len(self.bands)))

        centres = [source.centre for source in sources]

        wcs = sources.wcs

        ra_dec = np.asarray([wcs.all_pix2world(x, y, 0) for x, y in centres])

        coords = SkyCoord(ra_dec[:, 0], ra_dec[:, 1], unit='deg')

        Ebvs = self.extinction_map(coords)

        for i, Ebv in tqdm(enumerate(Ebvs), total=len(Ebvs)):
            self.A[i] = self._band_extinction(Ebv)

    def compute_extinction_from_tiling(self, coord_list, regions):
        """
        Compute the extinction on a grid of sky positions.

        E(B-V) is read from the dust map at every position of
        `coord_list` and converted to the extinction in every band with
        `_band_extinction`.

        The results are stored in the attributes `regions`, `Ebvs` and
        `A_grid`; nothing is returned.

        Parameters
        ----------
        coord_list : list of astropy.coordinates.SkyCoord
            The sky positions where the extinction is evaluated, for
            example the centres of the tiles of a `Tiling`.
        regions : list of shapely.Polygon
            The footprint of each position, with vertices in
            (RA, Dec) in deg, in the same order as `coord_list`; for
            example the `polygons` of a `Tiling`. They are only stored
            here and are used by `apply_grid_to_sources`.
        """

        self.regions = regions

        coords = SkyCoord(
            ra=[coord.ra for coord in coord_list],
            dec=[coord.dec for coord in coord_list]
        )

        Ebvs = self.extinction_map(coords)
        self.Ebvs = Ebvs

        self.A_grid = np.zeros((len(coords), len(self.bands)))

        for i, Ebv in tqdm(enumerate(Ebvs), total=len(Ebvs)):
            self.A_grid[i] = self._band_extinction(Ebv)

    def apply_grid_to_sources(self, sources):
        """
        Assign to the sources the extinction of the region they lie in.

        The pixel centre of every source (the centre of its bounding
        box) is converted to (RA, Dec) with the WCS of the sources and
        searched among the `regions` stored by
        `compute_extinction_from_tiling`, using a spatial index
        (`shapely.STRtree`). A source inside several overlapping
        regions takes the one with the lowest index; a source outside
        every region takes the nearest one. The test is planar in
        (RA, Dec).

        Parameters
        ----------
        sources : Sources
            The sources to which the extinction is assigned.

        Returns
        -------
        numpy.ndarray
            The extinction in magnitudes, shape
            `(n_sources, n_bands)`, in the order of `sources`.
        """

        sources_coords = np.asarray([source.centre for source in sources])

        wcs = sources.wcs

        ra_dec = np.asarray([wcs.all_pix2world(x, y, 0) for x, y in sources_coords])

        points = [sh.geometry.Point(ra, dec) for ra, dec in ra_dec]
        tree = sh.strtree.STRtree(self.regions)

        # Sources outside every region fall back on the nearest one.
        region_idx = np.asarray(tree.nearest(points), dtype=int)

        # `query` returns one (point, region) pair per match; keep the
        # lowest region index for points inside several regions.
        point_idx, matched_idx = tree.query(points, predicate='within')
        order = np.lexsort((matched_idx, point_idx))
        unique_points, first = np.unique(point_idx[order], return_index=True)
        region_idx[unique_points] = matched_idx[order][first]

        return self.A_grid[region_idx]
