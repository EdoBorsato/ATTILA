"""
Profile the ATTILA detection pipeline stage by stage.

The production pipeline lives outside the repository, in a driver
script such as
`/media/eborsato/T7/sunrise_Hydra_mosaic/S/core/detection_pipeline_A_mag_correct_time.py`,
which defines three stage functions (`combine`, `detect`, `deblend`)
and two per-source deblending callables. This harness imports that
script as a module (its `__main__` guard keeps it from running), then
executes the same stages in the same order, wrapping each one in
`cProfile` and recording its wall time and the peak resident memory.

Three things are measured that the driver's own timers cannot give:

1. Function-level attribution inside each stage (`*.prof`, `*.txt`).
2. The serial cost of deblending as a function of source size
   (`deblend_per_source.csv`). The pipeline deblends in a
   `multiprocessing.Pool`, where `cProfile` only sees the parent
   process waiting, so the per-source cost is measured separately.
3. The cost of pickling the sources to and from the worker processes
   (`deblend_pickle_s` in `summary.json`).

Nothing is written next to the pipeline script: all results go to
`--outdir` and the process runs from there, because `attila_logger`
opens `example.log` in the current working directory at import time.

Examples
--------
Quick run on the central 4000 x 4000 pixels (a few minutes)::

    python profiling/profile_detection_pipeline.py --crop 6000 \\
        --tile-arcmin 8

Full mosaic, wall times only (cProfile adds a large overhead to the
stages that make hundreds of millions of Python-level calls)::

    python profiling/profile_detection_pipeline.py --no-cprofile

Inspect a stage interactively afterwards::

    python -m pstats profiling/results/<run>/detect.prof
"""
import argparse
import cProfile
import csv
import importlib.util
import io
import json
import os
import pickle
import pstats
import resource
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PIPELINE = (
    "/media/eborsato/T7/sunrise_Hydra_mosaic/S/core/"
    "detection_pipeline_A_mag_correct_time.py"
)
DEFAULT_DATA = "/media/eborsato/T7/sunrise_Hydra_mosaic/S/data/S_core_cutout"
DEFAULT_TILING = (
    "/media/eborsato/T7/sunrise_Hydra_mosaic/S/core/output/tiling/"
    "tiling_[17_17]"
)


def parse_args():
    """Return the parsed command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pipeline", default=DEFAULT_PIPELINE,
        help="Driver script defining combine, detect and deblend.",
    )
    parser.add_argument(
        "--data-prefix", default=DEFAULT_DATA,
        help="Path prefix of the images; '_<band>_data.fits' and "
             "'_<band>_noise.fits' are appended.",
    )
    parser.add_argument(
        "--bands", default="g,r", help="Comma-separated band names.",
    )
    parser.add_argument(
        "--zpt", default="30.85,30.82",
        help="Comma-separated zero points, one per band.",
    )
    parser.add_argument(
        "--distance-mpc", type=float, default=51.0,
        help="Distance of the field in Mpc.",
    )
    parser.add_argument(
        "--tiling", default=DEFAULT_TILING,
        help="Pickled Tiling to load (path without the '.pkl' suffix).",
    )
    parser.add_argument(
        "--crop", type=int, default=0,
        help="Side in pixels of a central square crop of the images; "
             "0 profiles the full mosaic.",
    )
    parser.add_argument(
        "--tile-arcmin", type=float, default=0.0,
        help="Tile side in arcmin used with --crop; 0 keeps the side "
             "of the pickled tiling. A tile larger than the cropped "
             "field produces no tiles.",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output directory; defaults to profiling/results/<time>.",
    )
    parser.add_argument(
        "--no-cprofile", action="store_true",
        help="Record wall time and memory only.",
    )
    parser.add_argument(
        "--deblend-funcs",
        default="photutils_deblend_source_1,photutils_deblend_source_2",
        help="Comma-separated names of the per-source deblending "
             "callables defined in the driver script.",
    )
    parser.add_argument(
        "--deblend-sample", type=int, default=300,
        help="Number of sources timed one by one in the serial "
             "deblending test (the largest ones are always included); "
             "0 skips the test.",
    )
    parser.add_argument(
        "--skip-pool-deblend", action="store_true",
        help="Skip the driver's own multiprocessing deblend stage.",
    )
    parser.add_argument(
        "--top", type=int, default=35,
        help="Number of functions listed in each text report.",
    )
    return parser.parse_args()


def load_pipeline_module(pipeline_path):
    """
    Import the pipeline driver script as a module without running it.

    Parameters
    ----------
    pipeline_path : str
        The path of the driver script.

    Returns
    -------
    module
        The imported module, exposing `combine`, `detect`, `deblend`
        and the per-source deblending callables.
    """
    spec = importlib.util.spec_from_file_location(
        "attila_pipeline_under_test", pipeline_path
    )
    module = importlib.util.module_from_spec(spec)
    # The deblending callables are sent to worker processes by
    # reference, so the module has to be importable by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_field(data_prefix, bands, zpts, distance_mpc, crop):
    """
    Build the `Field`, optionally restricted to a central square crop.

    The crop slices the memory-mapped arrays and rewrites the WCS
    keywords of the header, so that the tiling, which is built from
    the headers, covers the cropped area only.

    Parameters
    ----------
    data_prefix : str
        The path prefix of the images, as expected by `Field`.
    bands : list of str
        The band names.
    zpts : list of float
        The photometric zero point of each band in magnitudes.
    distance_mpc : float
        The distance of the field in Mpc.
    crop : int
        The side of the central crop in pixels; 0 keeps the full image.

    Returns
    -------
    attila.field.field.Field
        The field, with `zpt` set and the physical scales computed.
    """
    from astropy import units as u
    from astropy.cosmology.realizations import Planck18
    from astropy.wcs import WCS

    from attila.field.field import Field

    field = Field(
        obs_path_dict={band: f"{data_prefix}_{band}" for band in bands}
    )
    field.compute_physical(
        cosmology=Planck18, distance=distance_mpc * u.Mpc
    )
    field.zpt = dict(zip(bands, zpts))

    if crop:
        for image in field.multibandimages:
            ny, nx = image.data.shape
            y0 = max((ny - crop) // 2, 0)
            x0 = max((nx - crop) // 2, 0)
            window = (slice(y0, y0 + crop), slice(x0, x0 + crop))

            wcs = WCS(image.header)[window]
            header = deepcopy(image.header)
            header.update(wcs.to_header())

            image.data = image.data[window]
            image.noise = image.noise[window]
            header["NAXIS1"] = image.data.shape[1]
            header["NAXIS2"] = image.data.shape[0]
            image.header = header
            image.wcs = WCS(header)

    return field


def build_tiling(tiling_path, field, crop, tile_arcmin=0.0):
    """
    Load the pickled tiling, or rebuild it on a cropped field.

    Parameters
    ----------
    tiling_path : str
        The path of the pickled `Tiling`, without the `.pkl` suffix.
    field : attila.field.field.Field
        The field the tiles are cut from.
    crop : int
        The side of the central crop in pixels. When non-zero a new
        tiling with the same tile shape and overlap as the pickled one
        is built from the cropped headers.
    tile_arcmin : float
        The tile side in arcmin of the rebuilt tiling; 0 keeps the side
        of the pickled one.

    Returns
    -------
    attila.tiling.tilingrid.Tiling
        The tiling passed to the `combine` stage.
    """
    from astropy import units as u

    from attila.tiling.tilingrid import Tiling

    tiling = Tiling.load(tiling_path)
    if not crop:
        return tiling

    if tile_arcmin:
        tile_shape_sky = (tile_arcmin * u.arcmin, tile_arcmin * u.arcmin)
    else:
        tile_shape_sky = tuple(s * u.deg for s in tiling.tile_shape_sky)

    cropped = Tiling(
        header_dict=field.multibandimages.property_dict("header"),
        tile_shape_sky=tile_shape_sky,
        overlap_sky=tiling.overlap_sky * u.deg,
        grid_type=tiling.grid_type,
    )
    cropped.make_tiles()
    return cropped


class StageProfiler:
    """
    Run pipeline stages under `cProfile` and collect a summary.

    Attributes
    ----------
    outdir : pathlib.Path
        The directory receiving the `.prof` and `.txt` reports.
    use_cprofile : bool
        Whether `cProfile` is enabled; wall time and peak memory are
        recorded in both cases.
    top : int
        The number of functions listed in each text report.
    summary : list of dict
        One entry per stage: name, wall time in seconds and peak
        resident set size in GiB reached by the end of the stage.
    """

    def __init__(self, outdir, use_cprofile=True, top=35):
        """
        Parameters
        ----------
        outdir : pathlib.Path
            The directory receiving the reports.
        use_cprofile : bool
            Whether to enable `cProfile` around each stage.
        top : int
            The number of functions listed in each text report.
        """
        self.outdir = outdir
        self.use_cprofile = use_cprofile
        self.top = top
        self.summary = []

    def run(self, name, func, *args, **kwargs):
        """
        Call `func(*args, **kwargs)` as the stage `name`.

        Parameters
        ----------
        name : str
            The stage name, used for the report file names.
        func : callable
            The stage function.

        Returns
        -------
        object
            Whatever `func` returns.
        """
        profiler = cProfile.Profile() if self.use_cprofile else None
        print(f"[profile] start {name}", flush=True)
        start = time.perf_counter()

        if profiler is not None:
            profiler.enable()
        try:
            result = func(*args, **kwargs)
        finally:
            if profiler is not None:
                profiler.disable()
            elapsed = time.perf_counter() - start
            # `ru_maxrss` is in kilobytes on Linux; children are the
            # multiprocessing workers.
            peak_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_children = resource.getrusage(
                resource.RUSAGE_CHILDREN
            ).ru_maxrss
            self.summary.append({
                "stage": name,
                "wall_s": round(elapsed, 3),
                "peak_rss_gib": round(peak_self / 1024**2, 3),
                "peak_child_rss_gib": round(peak_children / 1024**2, 3),
            })
            print(f"[profile] end   {name}: {elapsed:.1f} s", flush=True)
            if profiler is not None:
                self._write_report(name, profiler)

        return result

    def _write_report(self, name, profiler):
        """
        Dump the binary profile and a text report of the stage `name`.

        Parameters
        ----------
        name : str
            The stage name.
        profiler : cProfile.Profile
            The disabled profiler holding the stage statistics.
        """
        profiler.dump_stats(self.outdir / f"{name}.prof")

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stream.write("==== sorted by cumulative time ====\n")
        stats.sort_stats("cumulative").print_stats(self.top)
        stream.write("\n==== sorted by internal time ====\n")
        stats.sort_stats("tottime").print_stats(self.top)
        (self.outdir / f"{name}.txt").write_text(stream.getvalue())


def time_deblend_per_source(sources, debfunc, n_sample, csv_path):
    """
    Time the deblending of individual sources in the parent process.

    The `n_sample` sources are the `n_sample // 10` largest ones plus
    an evenly spaced selection of the rest in order of area, so that
    the scaling of the cost with source size is sampled over the whole
    range. Each source is deep-copied first, leaving `sources` intact.

    Parameters
    ----------
    sources : Sources
        The detected sources.
    debfunc : callable
        The per-source deblending callable of the driver script.
    n_sample : int
        The number of sources to time.
    csv_path : pathlib.Path
        The CSV file written, one row per timed source.

    Returns
    -------
    dict
        The number of timed sources, their total time in seconds, the
        time spent in the 10 slowest ones and the ratio between the
        summed bounding-box pixels of all sources and of the sample.
    """
    ordered = sorted(sources.sources, key=lambda s: s.area)
    n_sample = min(n_sample, len(ordered))
    n_top = max(n_sample // 10, 1)
    largest = ordered[-n_top:]
    rest = ordered[:-n_top]
    n_rest = n_sample - n_top
    if n_rest > 0 and rest:
        step = max(len(rest) // n_rest, 1)
        rest = rest[::step][:n_rest]
    else:
        rest = []
    sample = rest + largest

    rows = []
    for source in sample:
        candidate = deepcopy(source)
        start = time.perf_counter()
        candidate = debfunc(candidate)
        elapsed = time.perf_counter() - start
        rows.append({
            "label": source.label,
            "area_pix": source.area,
            "bbox_ny": source.segment.shape[0],
            "bbox_nx": source.segment.shape[1],
            "n_children": candidate.deblend.nlabels,
            "elapsed_s": round(elapsed, 5),
        })

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    times = sorted(row["elapsed_s"] for row in rows)
    all_pix = sum(s.segment.size for s in sources.sources)
    sample_pix = sum(row["bbox_ny"] * row["bbox_nx"] for row in rows)
    return {
        "n_timed": len(rows),
        "total_s": round(sum(times), 3),
        "slowest_10_s": round(sum(times[-10:]), 3),
        "bbox_pixels_all_over_sample": round(all_pix / sample_pix, 3),
    }


def main():
    """Run the profiling session described in the module docstring."""
    args = parse_args()

    bands = args.bands.split(",")
    zpts = [float(z) for z in args.zpt.split(",")]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"crop{args.crop}" if args.crop else "full"
    outdir = Path(
        args.outdir or REPO_ROOT / "profiling" / "results" / f"{stamp}_{tag}"
    ).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Run from the output directory: `attila_logger` creates
    # `example.log` in the working directory when first imported.
    os.chdir(outdir)
    sys.path.insert(0, str(REPO_ROOT))
    warnings.simplefilter("ignore")

    pipeline = load_pipeline_module(args.pipeline)
    profiler = StageProfiler(
        outdir, use_cprofile=not args.no_cprofile, top=args.top
    )
    extra = {}

    field = profiler.run(
        "load_field", build_field,
        args.data_prefix, bands, zpts, args.distance_mpc, args.crop,
    )
    tiling = profiler.run(
        "load_tiling", build_tiling,
        args.tiling, field, args.crop, args.tile_arcmin,
    )
    extra["image_shape"] = list(field.multibandimages[bands[0]].shape)

    det_dict, field, tiling = profiler.run(
        "combine", pipeline.combine, field, tiling
    )
    extra["n_tiles"] = len(tiling.centres)

    sources, ext, field, tiling, threshold, nconn = profiler.run(
        "detect", pipeline.detect, det_dict, field, tiling
    )
    extra["n_sources"] = len(sources)

    debfuncs = [
        getattr(pipeline, name)
        for name in args.deblend_funcs.split(",") if name
    ]

    if debfuncs and len(sources):
        # Cost of shipping every source to a worker and back.
        start = time.perf_counter()
        payload = [pickle.dumps(source) for source in sources.sources]
        _ = [pickle.loads(blob) for blob in payload]
        extra["deblend_pickle_s"] = round(time.perf_counter() - start, 3)
        extra["deblend_pickle_gib"] = round(
            sum(len(blob) for blob in payload) / 1024**3, 3
        )
        del payload, _

    if debfuncs and args.deblend_sample and len(sources):
        debfunc = debfuncs[0]
        extra["deblend_per_source"] = profiler.run(
            f"deblend_serial_sample_{debfunc.__name__}",
            time_deblend_per_source,
            sources, debfunc, args.deblend_sample,
            outdir / "deblend_per_source.csv",
        )

    if debfuncs and not args.skip_pool_deblend:
        for debfunc in debfuncs:
            profiler.run(
                f"deblend_pool_{debfunc.__name__}", pipeline.deblend,
                sources, ext, field, tiling, threshold, nconn,
                debfunc.__name__, debfunc=debfunc,
            )

    summary = {
        "pipeline": args.pipeline,
        "crop": args.crop,
        "cprofile": not args.no_cprofile,
        **extra,
        "stages": profiler.summary,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=4))

    print(f"\n{'stage':<52}{'wall [s]':>10}{'peak RSS [GiB]':>16}")
    for row in profiler.summary:
        print(
            f"{row['stage']:<52}{row['wall_s']:>10.1f}"
            f"{row['peak_rss_gib']:>16.2f}"
        )
    print(f"\nResults written to {outdir}")


if __name__ == "__main__":
    main()
