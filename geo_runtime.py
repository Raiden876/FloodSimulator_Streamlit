"""Use the matching GDAL/PROJ resources in each Streamlit worker thread."""

from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import rasterio
from rasterio.env import GDALDataFinder, PROJDataFinder, set_proj_data_search_path


def configure_rasterio_data():
    """Restore Rasterio's own resource paths, including on hot-reload threads.

    Rasterio initializes these during its first import. Streamlit executes
    subsequent runs in new threads, and GDAL versions with thread-local PROJ
    settings may no longer see the import thread's search path. Prefer the
    wheel's matching proj.db instead of a different GIS installation's DB.
    """
    package = Path(rasterio.__file__).resolve().parent
    proj_directory = package / "proj_data"
    if not (proj_directory / "proj.db").is_file():
        discovered = PROJDataFinder().search()
        proj_directory = Path(discovered) if discovered else None
    if proj_directory is not None and (proj_directory / "proj.db").is_file():
        set_proj_data_search_path(str(proj_directory))
    gdal_directory = package / "gdal_data"
    if not gdal_directory.is_dir():
        discovered = GDALDataFinder().search()
        gdal_directory = Path(discovered) if discovered else None
    if gdal_directory is not None and gdal_directory.is_dir():
        rasterio.env.set_gdal_config("GDAL_DATA", str(gdal_directory))
    try:
        rasterio.crs.CRS.from_epsg(4326)
        rasterio.crs.CRS.from_epsg(32651)
    except rasterio.errors.CRSError as exc:
        raise RuntimeError("Rasterio cannot read its PROJ database. Reinstall rasterio in this project's .venv; its matching proj_data/proj.db must be readable.") from exc
    return str(proj_directory) if proj_directory else "system PROJ resources"


@contextmanager
def rasterio_runtime():
    with rasterio.Env():
        configure_rasterio_data()
        yield


def with_rasterio_runtime(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with rasterio_runtime():
            return function(*args, **kwargs)
    return wrapped
