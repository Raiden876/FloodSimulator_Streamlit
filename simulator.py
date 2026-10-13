"""Regional flood simulator for Metro Manila and selected Philippine provinces.

The source hazard layers are the UP NOAH shapefiles stored in this directory as
ZIP archives. This module keeps data preparation and the hydrologic model in
plain functions so they can be reused from ``main.ipynb`` as well as the
Streamlit application. The bundled population raster is used for density and
flood-exposure layers.
"""

from pathlib import Path
import hashlib
import json
from dataclasses import replace

from geo_runtime import with_rasterio_runtime

from flood_model import (
    DISPLAY_DEPTH_THRESHOLD_M, EXPOSURE_DEPTH_THRESHOLD_M,
    EXPOSURE_SENSITIVITY_THRESHOLDS_M, UNMODELED_EXPOSURE_LABEL,
    LAND_COVER_LABELS, RainfallEvent, adjust_cn_amc, get_effective_cn,
    effective_cn_values, empirical_depth_m, make_rainfall_event, route_runoff,
    validation_metrics, normalize_hsg, rainfall_runoff_analysis,
)
from terrain_model import (
    PhysicalInputs, enrich_runoff_inputs, load_dem_grid, load_physical_grid,
    build_flow_network, simulate_terrain, terrain_to_polygons, terrain_depth_on_grid,
    read_aligned_raster, depth_geotiff_bytes, RasterGrid,
)
from zipfile import ZipFile

import altair as alt
import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import requests
import rasterio
import streamlit as st
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.mask import mask as raster_mask
from rasterio.transform import array_bounds
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds, transform as transform_coordinates, reproject, calculate_default_transform
from shapely.geometry import mapping
from shapely.ops import unary_union
from streamlit_folium import st_folium
from site_config import SITE_MAPPING, site_key


DATA_CRS = "EPSG:4326"
AREA_CRS = "EPSG:32651"  # UTM zone 51N covers all configured sites.
DEFAULT_SIMPLIFY_TOLERANCE_M = 150.0
DEFAULT_BOUNDARY_SIMPLIFY_TOLERANCE_M = 100.0
DEFAULT_POPULATION_DOT_COUNT = 2_500
ALL_SITES_LABEL = "All covered sites"
POPULATION_RASTER_RELATIVE_PATH = Path(
    "Philippine_population/philippines_population_2026.tif"
)
BOUNDARY_SHP_RELATIVE_PATH = Path(
    "ph_boundaries/extracted/PH_Adm3_MuniCities.shp.shp"
)
BOUNDARY_LOOKUP_RELATIVE_PATH = Path("ph_boundaries/PH_Adm2_ProvDists.csv")

# The population raster stores estimated people per source cell. These
# thresholds convert that count to people/km2 so the map remains meaningful
# when the raster is displayed at a coarser resolution.
POPULATION_DENSITY_THRESHOLDS_PER_SQKM = (1_000.0, 5_000.0)
POPULATION_DENSITY_LABELS = {
    0: "No risk / no population",
    1: "Low population density",
    2: "Moderate population density",
    3: "High population density",
}
POPULATION_DENSITY_COLORS = {
    0: "#D9D9D9",
    1: "#FFF7BC",
    2: "#FEC44F",
    3: "#D95F0E",
}
POPULATION_HEATMAP_COLORMAP = mcolors.LinearSegmentedColormap.from_list(
    "population_yellow_red",
    ["#FFFFE5", "#FFF7BC", "#FEC44F", "#FD8D3C", "#E31A1C", "#800026"],
    N=256,
)

# Flood depth uses a blue-only palette so it cannot be confused with the
# yellow/orange/red UP NOAH hazard classification or the terrain basemap.
FLOOD_DEPTH_COLORMAP = mcolors.LinearSegmentedColormap.from_list(
    "flood_depth_blue",
    ["#E0F3F8", "#ABD9E9", "#74ADD1", "#4575B4", "#313695", "#081D58"],
    N=256,
)

# Population risk is assigned from the spatial overlap with UP NOAH hazard
# polygons. Population density is not used as a substitute for hazard risk.
POPULATION_RISK_ORDER = ["No risk", "Low risk", "Moderate risk", "High risk"]
POPULATION_RISK_LABELS = {
    0: "No risk",
    1: "Low risk",
    2: "Moderate risk",
    3: "High risk",
}

# These are the population-impact bins requested for the simulated flood.
# Shallow modeled depths below the exposure threshold remain visible on the
# flood map. Unknown cells outside terrain coverage are reported separately.
FLOOD_EXPOSURE_ORDER = ["Below exposure threshold", "0.1–1 m", "1–2 m", ">2 m", UNMODELED_EXPOSURE_LABEL]

# Single source of truth for the archives shown in the app. Centers are used
# for SoilGrids queries and map navigation.
HAZARD_LABELS = {
    1: "Low Hazard (0–0.5 m)",
    2: "Medium Hazard (>0.5–1.5 m)",
    3: "High Hazard (>1.5 m)",
}

BASE_DEPTH_BY_VAR = {1: 0.25, 2: 1.00, 3: 2.00}
LEGACY_CN_BY_VAR = {1: 78, 2: 88, 3: 98}
CN_BY_VAR = LEGACY_CN_BY_VAR  # Compatibility for existing notebooks; baseline only.


def _find_shapefile_member(archive_path: Path) -> str:
    """Return the shapefile member inside a ZIP, including nested archives."""

    with ZipFile(archive_path) as archive:
        shapefiles = [
            name for name in archive.namelist() if name.lower().endswith(".shp")
        ]

    if not shapefiles:
        raise FileNotFoundError(f"No .shp file found inside {archive_path.name}")

    # A source archive should contain one layer. Sorting keeps selection
    # deterministic if a future archive includes auxiliary shapefiles.
    return sorted(shapefiles)[0]


def _read_zip_layer(archive_path: Path) -> gpd.GeoDataFrame:
    """Read the first shapefile from a ZIP using GDAL's virtual file system."""

    member = _find_shapefile_member(archive_path)
    archive_uri = archive_path.resolve().as_posix()
    return gpd.read_file(f"zip://{archive_uri}!/{member}")


def _polygonal_geometry(geometry):
    """Keep polygon parts when validation returns a GeometryCollection."""

    if geometry is None or geometry.is_empty:
        return geometry
    if geometry.geom_type != "GeometryCollection":
        return geometry

    polygon_parts = [
        part
        for part in geometry.geoms
        if part.geom_type in {"Polygon", "MultiPolygon"} and not part.is_empty
    ]
    return unary_union(polygon_parts) if polygon_parts else None


def _prepare_layer(
    layer: gpd.GeoDataFrame,
    site_name: str,
    site_config: dict,
    simplify_tolerance_m: float,
) -> gpd.GeoDataFrame:
    """Repair, standardize, enrich, and simplify one source hazard layer."""

    if layer.crs is None:
        layer = layer.set_crs(DATA_CRS)
    else:
        layer = layer.to_crs(DATA_CRS)

    required_columns = {"Var", "geometry"}
    missing_columns = required_columns.difference(layer.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{site_name} is missing required field(s): {missing}")

    layer = layer[["Var", "geometry"]].copy()
    layer["Var"] = pd.to_numeric(layer["Var"], errors="coerce")
    layer = layer.dropna(subset=["Var", "geometry"])
    layer["Var"] = layer["Var"].astype(int)
    layer = layer[layer["Var"].isin(HAZARD_LABELS)]
    layer = layer[~layer.geometry.is_empty].copy()

    # The source multipolygons are very detailed and invalid. A fast first-pass
    # simplification keeps preprocessing practical; validation after that pass
    # is both much faster and sufficient for browser rendering.
    layer = layer.to_crs(AREA_CRS)
    source_area_sqkm = layer.geometry.area / 1_000_000.0
    layer["geometry"] = layer.geometry.simplify(
        tolerance=float(simplify_tolerance_m), preserve_topology=False
    )
    layer["geometry"] = layer.geometry.make_valid()
    layer["geometry"] = layer.geometry.apply(_polygonal_geometry)
    layer["area_sqkm"] = source_area_sqkm
    layer = layer[layer.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    layer = layer[~layer.geometry.is_empty].copy()
    if layer.empty:
        raise ValueError(f"{site_name} has no usable polygon geometries")
    layer = layer.to_crs(DATA_CRS)
    # A final repair after reprojection avoids rare self-intersections created
    # by coordinate transformation near multipart polygon boundaries.
    layer["geometry"] = layer.geometry.make_valid()
    layer["geometry"] = layer.geometry.apply(_polygonal_geometry)
    layer = layer[layer.geometry.notna() & ~layer.geometry.is_empty].copy()

    layer["site_name"] = site_name
    layer["hazard_level"] = layer["Var"].map(HAZARD_LABELS)
    layer["base_depth_m"] = layer["Var"].map(BASE_DEPTH_BY_VAR)
    layer["CN_base"] = layer["Var"].map(LEGACY_CN_BY_VAR)
    layer["CN_source"] = "Legacy hazard-based baseline"
    layer["source_return_period_yr"] = int(site_config["return_period_yr"])
    layer["source_file"] = site_config["archive"]

    return layer[
        [
            "site_name",
            "Var",
            "hazard_level",
            "base_depth_m",
            "CN_base",
            "CN_source",
            "area_sqkm",
            "source_return_period_yr",
            "source_file",
            "geometry",
        ]
    ]


def preprocess_hazard_data(
    data_dir: str | Path = ".",
    simplify_tolerance_m: float = DEFAULT_SIMPLIFY_TOLERANCE_M,
) -> gpd.GeoDataFrame:
    """Load and preprocess all configured Metro Manila/province archives.

    The returned layer is in WGS 84 for Folium, with area retained in square
    kilometres from the source UTM geometries. Missing archives are skipped so
    the app can still start with a partial dataset; their names are available
    in ``result.attrs["missing_sites"]``. Read errors are exposed through
    ``result.attrs["source_errors"]``.
    """

    base_dir = Path(data_dir)
    prepared_layers = []
    missing_sites = []
    source_errors = []

    for site_name, site_config in SITE_MAPPING.items():
        archive_path = base_dir / site_config["archive"]
        if not archive_path.exists():
            missing_sites.append(site_name)
            continue

        try:
            source_layer = _read_zip_layer(archive_path)
            prepared_layers.append(
                _prepare_layer(
                    source_layer,
                    site_name=site_name,
                    site_config=site_config,
                    simplify_tolerance_m=simplify_tolerance_m,
                )
            )
        except Exception as exc:
            source_errors.append(f"{site_name}: {exc}")

    # Retain compatibility with the original QC-only project if no regional
    # archive is available in a deployment environment.
    if not prepared_layers:
        legacy_path = base_dir / "qc_noah_100yr_clipped.geojson"
        if legacy_path.exists():
            legacy_config = {"archive": legacy_path.name, "return_period_yr": 100}
            try:
                legacy_layer = gpd.read_file(legacy_path)
                prepared_layers.append(
                    _prepare_layer(
                        legacy_layer,
                        site_name="Quezon City (legacy layer)",
                        site_config=legacy_config,
                        simplify_tolerance_m=simplify_tolerance_m,
                    )
                )
            except Exception as exc:
                source_errors.append(f"Quezon City legacy layer: {exc}")

    if not prepared_layers:
        details = "; ".join(source_errors) or "no configured archives were found"
        raise FileNotFoundError(f"Unable to load a usable flood layer: {details}")

    combined = gpd.GeoDataFrame(
        pd.concat(prepared_layers, ignore_index=True),
        crs=DATA_CRS,
    )
    combined.attrs["missing_sites"] = missing_sites
    combined.attrs["source_errors"] = source_errors
    return combined


def find_population_raster(data_dir: str | Path = ".") -> Path:
    """Return the bundled population raster used by the simulator."""

    raster_path = Path(data_dir) / POPULATION_RASTER_RELATIVE_PATH
    if raster_path.exists():
        return raster_path
    raise FileNotFoundError(
        "Population raster was not found at "
        f"{raster_path}. Expected {POPULATION_RASTER_RELATIVE_PATH}."
    )


def find_boundary_shapefile(data_dir: str | Path = ".") -> Path:
    """Return the municipal boundary shapefile bundled with the project."""

    base_dir = Path(data_dir)
    candidates = [
        base_dir / BOUNDARY_SHP_RELATIVE_PATH,
        base_dir / "ph_boundaries/extracted/Municities.shp.shp",
    ]
    for boundary_path in candidates:
        if boundary_path.exists():
            return boundary_path
    raise FileNotFoundError(
        "Philippine boundary shapefile was not found. Expected it under "
        f"{base_dir / BOUNDARY_SHP_RELATIVE_PATH}."
    )


def preprocess_boundary_data(
    data_dir: str | Path = ".",
) -> gpd.GeoDataFrame:
    """Dissolve municipal boundaries into the sites covered by hazard maps.

    The boundary source is municipal-level, so the province/district lookup
    table is used to select the configured provinces. Metro Manila is selected
    by its NCR administrative code. Only boundaries corresponding to sites in
    SITE_MAPPING are returned; unrelated Philippine municipalities are omitted.
    """

    base_dir = Path(data_dir)
    boundary_path = find_boundary_shapefile(base_dir)
    lookup_path = base_dir / BOUNDARY_LOOKUP_RELATIVE_PATH
    if not lookup_path.exists():
        raise FileNotFoundError(f"Boundary lookup table not found: {lookup_path}")

    lookup = pd.read_csv(lookup_path)
    lookup["adm2_psgc"] = pd.to_numeric(lookup["adm2_psgc"], errors="coerce")
    province_lookup = lookup[lookup["geo_level"].eq("Prov")].dropna(
        subset=["adm2_en", "adm2_psgc"]
    )
    site_to_code = {
        str(row["adm2_en"]): int(row["adm2_psgc"])
        for _, row in province_lookup.iterrows()
    }

    if boundary_path.name == "PH_Adm3_MuniCities.shp.shp":
        boundary_source = gpd.read_file(
            boundary_path,
            columns=["adm1_psgc", "adm2_psgc", "geometry"],
        )
        boundary_source["adm1_code"] = pd.to_numeric(
            boundary_source["adm1_psgc"], errors="coerce"
        )
        boundary_source["adm2_code"] = pd.to_numeric(
            boundary_source["adm2_psgc"], errors="coerce"
        )
    else:
        # Fallback for the alternate extracted WGS84 copy in the same folder.
        boundary_source = gpd.read_file(
            boundary_path,
            columns=["adm1_pcode", "adm2_pcode", "geometry"],
        )
        boundary_source["adm1_code"] = boundary_source["adm1_pcode"].map(
            lambda value: int(str(value).replace("PH", "")) * 100_000_000
            if pd.notna(value)
            else np.nan
        )
        boundary_source["adm2_code"] = boundary_source["adm2_pcode"].map(
            lambda value: int(str(value).replace("PH", "")) * 100_000
            if pd.notna(value)
            else np.nan
        )

    boundary_source = boundary_source[
        boundary_source.geometry.notna() & ~boundary_source.geometry.is_empty
    ].copy()
    if boundary_source.crs is None:
        raise ValueError(f"Boundary shapefile has no CRS: {boundary_path}")

    prepared_boundaries = []
    missing_sites = []
    ncr_code = 1_300_000_000
    for site_name in SITE_MAPPING:
        if site_name == "Metro Manila":
            site_boundary = boundary_source[
                boundary_source["adm1_code"] == ncr_code
            ]
        else:
            province_code = site_to_code.get(site_name)
            site_boundary = boundary_source[
                boundary_source["adm2_code"] == province_code
            ]

        if site_boundary.empty:
            missing_sites.append(site_name)
            continue

        geometry = unary_union(site_boundary.geometry.tolist())
        if geometry is None or geometry.is_empty:
            missing_sites.append(site_name)
            continue
        prepared_boundaries.append(
            {
                "site_name": site_name,
                "boundary_name": site_name,
                "geometry": geometry,
            }
        )

    if not prepared_boundaries:
        raise ValueError("No configured site boundaries could be matched")

    result = gpd.GeoDataFrame(prepared_boundaries, crs=boundary_source.crs)
    result = result.to_crs(AREA_CRS)
    result["geometry"] = result.geometry.simplify(
        tolerance=DEFAULT_BOUNDARY_SIMPLIFY_TOLERANCE_M,
        preserve_topology=True,
    )
    result["geometry"] = result.geometry.make_valid()
    result = result.to_crs(DATA_CRS)
    result.attrs["missing_sites"] = missing_sites
    result.attrs["source_file"] = str(boundary_path)
    return result


@st.cache_data(show_spinner="Loading site boundaries...")
def load_boundary_data(data_dir: str | Path = ".") -> gpd.GeoDataFrame:
    """Cached Streamlit wrapper around :func:`preprocess_boundary_data`."""

    return preprocess_boundary_data(data_dir)


def _population_density_class_code(
    population_density_per_sqkm: np.ndarray,
) -> np.ndarray:
    """Map density values to 0/1/2/3 population-density classes."""

    density = np.asarray(population_density_per_sqkm, dtype=float)
    codes = np.zeros(density.shape, dtype=np.uint8)
    valid = np.isfinite(density) & (density > 0.0)
    low_limit, high_limit = POPULATION_DENSITY_THRESHOLDS_PER_SQKM
    codes[valid & (density < low_limit)] = 1
    codes[valid & (density >= low_limit) & (density < high_limit)] = 2
    codes[valid & (density >= high_limit)] = 3
    return codes


def classify_population_density(population_density_per_sqkm: float) -> str:
    """Return the display class for a people/km2 value."""

    code = int(
        _population_density_class_code(
            np.asarray([population_density_per_sqkm], dtype=float)
        )[0]
    )
    return POPULATION_DENSITY_LABELS[code]


def depth_to_flood_class(depth: float, threshold_m: float = EXPOSURE_DEPTH_THRESHOLD_M) -> str:
    """Classify simulated flood depth using the population-impact bands."""

    depth = float(depth)
    population_threshold = max(EXPOSURE_DEPTH_THRESHOLD_M, float(threshold_m))
    if not np.isfinite(depth) or depth < population_threshold:
        return FLOOD_EXPOSURE_ORDER[0]
    if depth < 1.0:
        return FLOOD_EXPOSURE_ORDER[1]
    if depth <= 2.0:
        return FLOOD_EXPOSURE_ORDER[2]
    return FLOOD_EXPOSURE_ORDER[3]


def _population_sum_for_geometry(source, geometry) -> float:
    """Sum raster population values inside one polygon."""

    if geometry is None or geometry.is_empty:
        return 0.0

    try:
        values, _ = raster_mask(
            source,
            [mapping(geometry)],
            crop=True,
            all_touched=True,
            filled=False,
        )
    except (ValueError, rasterio.errors.RasterioIOError):
        # A polygon outside the raster extent has no population value in the
        # bundled source and is valid for the regional app.
        return 0.0

    population = np.asarray(values[0].compressed(), dtype=float)
    population = population[np.isfinite(population) & (population > 0.0)]
    return float(population.sum()) if population.size else 0.0


def add_population_exposure(
    hazard_data: gpd.GeoDataFrame,
    population_raster_path: str | Path,
) -> gpd.GeoDataFrame:
    """Attach population estimates and density classes to hazard polygons.

    The raster values are summed for each polygon. The resulting population
    estimate is intentionally kept as a float because the source contains
    fractional estimates for sparsely populated cells.
    """

    raster_path = Path(population_raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Population raster not found: {raster_path}")
    if hazard_data.empty:
        enriched = hazard_data.copy()
        enriched["population_estimate"] = pd.Series(dtype=float)
        enriched["population_density_per_sqkm"] = pd.Series(dtype=float)
        enriched["population_density_class"] = pd.Series(dtype=str)
        return enriched

    enriched = hazard_data.copy()
    with rasterio.open(raster_path) as source:
        if source.crs is None:
            raise ValueError(f"Population raster has no CRS: {raster_path}")
        geometries = enriched.to_crs(source.crs).geometry
        population_values = [
            _population_sum_for_geometry(source, geometry) for geometry in geometries
        ]

    enriched["population_estimate"] = np.round(population_values, 2)

    if "area_sqkm" in enriched.columns:
        area_sqkm = pd.to_numeric(enriched["area_sqkm"], errors="coerce")
    else:
        area_sqkm = enriched.to_crs(AREA_CRS).geometry.area / 1_000_000.0
    area_sqkm = area_sqkm.replace(0.0, np.nan)
    density = np.divide(
        enriched["population_estimate"].to_numpy(dtype=float),
        area_sqkm.to_numpy(dtype=float),
        out=np.zeros(len(enriched), dtype=float),
        where=area_sqkm.notna().to_numpy() & (area_sqkm.to_numpy() > 0.0),
    )
    enriched["population_density_per_sqkm"] = np.round(density, 2)
    enriched["population_density_class"] = [
        classify_population_density(value) for value in density
    ]
    enriched.attrs = dict(hazard_data.attrs)
    enriched.attrs["population_raster"] = str(raster_path)
    return enriched


def _population_values_for_boundary(source, geometry):
    """Read positive population cells whose centers fall inside a boundary."""

    if geometry is None or geometry.is_empty:
        return None, None
    try:
        values, transform = raster_mask(
            source,
            [mapping(geometry)],
            crop=True,
            all_touched=False,
            filled=False,
        )
    except (ValueError, rasterio.errors.RasterioIOError):
        return None, None
    return np.asarray(values[0].filled(np.nan), dtype=float), transform


def _rasterize_hazard_risk(
    hazard_data: gpd.GeoDataFrame,
    source_crs,
    out_shape: tuple[int, int],
    transform,
) -> np.ndarray:
    """Rasterize UP NOAH risk codes, with high risk taking precedence."""

    if hazard_data.empty:
        return np.zeros(out_shape, dtype=np.uint8)

    source_hazards = hazard_data.to_crs(source_crs)
    shapes = []
    for code in (1, 2, 3):
        for geometry in source_hazards.loc[
            source_hazards["Var"].eq(code), "geometry"
        ]:
            if geometry is not None and not geometry.is_empty:
                shapes.append((mapping(geometry), code))
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)
    return rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )


def _rasterize_simulated_depth(
    simulated_data: gpd.GeoDataFrame,
    source_crs,
    out_shape: tuple[int, int],
    transform,
) -> np.ndarray:
    """Rasterize maximum simulated depth for each population cell."""

    if simulated_data.empty or "sim_depth_m" not in simulated_data.columns:
        return np.zeros(out_shape, dtype=np.float32)

    terrain = simulated_data.attrs.get("terrain_result")
    if terrain is not None:
        return terrain_depth_on_grid(terrain, out_shape, transform, source_crs)

    source_simulation = simulated_data.to_crs(source_crs).copy()
    source_simulation["sim_depth_m"] = pd.to_numeric(
        source_simulation["sim_depth_m"], errors="coerce"
    ).fillna(0.0)
    source_simulation = source_simulation.sort_values("sim_depth_m")
    shapes = [
        (mapping(row.geometry), float(row.sim_depth_m))
        for row in source_simulation.itertuples()
        if row.geometry is not None
        and not row.geometry.is_empty
        and float(row.sim_depth_m) > 0.0
    ]
    if not shapes:
        return np.zeros(out_shape, dtype=np.float32)
    # Rasterize replaces previous values, so ascending depth order leaves the
    # maximum simulated depth when polygons overlap.
    return rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0.0,
        all_touched=False,
        dtype="float32",
    )


def _empty_population_risk_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=["site_name", "risk_class", "people", "share_pct"])


def _population_risk_summary_uncached(
    hazard_data: gpd.GeoDataFrame,
    population_raster_path: str | Path,
    boundary_data: gpd.GeoDataFrame | None,
) -> pd.DataFrame:
    """Count population by UP NOAH risk, including people in no-risk areas.

    Each population raster cell is counted once per site. Its risk class is
    determined by the UP NOAH polygon containing the cell center; cells outside
    all UP NOAH polygons but inside the administrative site boundary are ``No
    risk``. This avoids assigning every population cell a hazard merely
    because it lies somewhere in the same province.
    """

    if hazard_data.empty or boundary_data is None or boundary_data.empty:
        return _empty_population_risk_summary()

    selected_boundaries = _selected_boundary_data(hazard_data, boundary_data)
    if selected_boundaries.empty:
        return _empty_population_risk_summary()

    rows = []
    with rasterio.open(population_raster_path) as source:
        source_boundaries = selected_boundaries.to_crs(source.crs)
        for boundary in source_boundaries.itertuples():
            population, transform = _population_values_for_boundary(
                source, boundary.geometry
            )
            if population is None:
                continue
            site_hazards = hazard_data[
                hazard_data["site_name"].eq(boundary.site_name)
            ]
            risk_codes = _rasterize_hazard_risk(
                site_hazards,
                source.crs,
                population.shape,
                transform,
            )
            valid = np.isfinite(population) & (population > 0.0)
            people_by_code = np.bincount(
                risk_codes[valid],
                weights=population[valid],
                minlength=len(POPULATION_RISK_ORDER),
            )
            site_total = float(people_by_code.sum())
            for code, label in POPULATION_RISK_LABELS.items():
                people = float(people_by_code[code])
                rows.append(
                    {
                        "site_name": str(boundary.site_name),
                        "risk_class": label,
                        "people": people,
                        "share_pct": 100.0 * people / site_total
                        if site_total > 0.0
                        else 0.0,
                    }
                )

    return pd.DataFrame(rows, columns=["site_name", "risk_class", "people", "share_pct"])


@st.cache_data(show_spinner="Classifying population by UP NOAH risk...")
def _population_risk_summary_cached(
    population_raster_path: str,
    hazard_cache_key: str,
    boundary_cache_key: tuple,
    _hazard_data: gpd.GeoDataFrame,
    _boundary_data: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Cached wrapper for the population/UP NOAH spatial classification."""

    return _population_risk_summary_uncached(
        _hazard_data,
        population_raster_path,
        _boundary_data,
    )


def population_risk_summary(
    hazard_data: gpd.GeoDataFrame,
    population_raster_path: str | Path,
    boundary_data: gpd.GeoDataFrame | None,
) -> pd.DataFrame:
    """Return cached population counts by UP NOAH risk class."""

    if boundary_data is None:
        return _empty_population_risk_summary()
    return _population_risk_summary_cached(
        str(population_raster_path),
        _population_data_cache_key(hazard_data),
        _boundary_geometry_cache_key(boundary_data),
        hazard_data,
        boundary_data,
    )


def _population_flood_summary_uncached(
    simulated_data: gpd.GeoDataFrame,
    population_raster_path: str | Path,
    boundary_data: gpd.GeoDataFrame | None,
) -> pd.DataFrame:
    """Count people by simulated flood-depth band without double counting."""

    columns = ["site_name", "flood_depth_band", "people_in_band", "people_affected"]
    if simulated_data.empty or boundary_data is None or boundary_data.empty:
        return pd.DataFrame(columns=columns)

    selected_boundaries = _selected_boundary_data(simulated_data, boundary_data)
    if selected_boundaries.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    with rasterio.open(population_raster_path) as source:
        source_boundaries = selected_boundaries.to_crs(source.crs)
        for boundary in source_boundaries.itertuples():
            population, transform = _population_values_for_boundary(
                source, boundary.geometry
            )
            if population is None:
                continue
            site_simulation = simulated_data[
                simulated_data["site_name"].eq(boundary.site_name)
            ]
            depths = _rasterize_simulated_depth(
                site_simulation,
                source.crs,
                population.shape,
                transform,
            )
            valid = np.isfinite(population) & (population > 0.0)
            depth_codes = np.zeros(population.shape, dtype=np.uint8)
            depth_codes[(depths >= EXPOSURE_DEPTH_THRESHOLD_M) & (depths < 1.0)] = 1
            depth_codes[(depths >= 1.0) & (depths <= 2.0)] = 2
            depth_codes[depths > 2.0] = 3
            depth_codes[~np.isfinite(depths)] = 4
            people_by_band = np.bincount(
                depth_codes[valid],
                weights=population[valid],
                minlength=len(FLOOD_EXPOSURE_ORDER),
            )
            for code, label in enumerate(FLOOD_EXPOSURE_ORDER):
                people = float(people_by_band[code])
                rows.append(
                    {
                        "site_name": str(boundary.site_name),
                        "flood_depth_band": label,
                        "people_in_band": people,
                        "people_affected": people if 0 < code < 4 else 0.0,
                    }
                )

    return pd.DataFrame(rows, columns=columns)


def _simulation_population_cache_key(
    simulated_data: gpd.GeoDataFrame,
) -> str:
    """Build a cache key that changes when simulated depths change."""

    digest = hashlib.sha256()
    digest.update(pd.to_numeric(simulated_data.get("sim_depth_m", pd.Series(dtype=float)),
                                errors="coerce").to_numpy(dtype=float).tobytes())
    digest.update(repr(tuple(simulated_data.get("site_name", []))).encode())
    digest.update(repr(simulated_data.crs).encode())
    for geometry in simulated_data.geometry:
        digest.update(geometry.wkb if geometry is not None else b"missing")
    terrain = simulated_data.attrs.get("terrain_result")
    if terrain is not None:
        digest.update(repr((terrain.grid.shape, terrain.grid.transform, terrain.grid.crs)).encode())
        digest.update(terrain.max_depth_m.tobytes())
        if terrain.report_mask is not None:
            digest.update(terrain.report_mask.tobytes())
    return digest.hexdigest()


@st.cache_data(show_spinner="Calculating population flood exposure...")
def _population_flood_summary_cached(
    population_raster_path: str,
    simulation_cache_key: str,
    boundary_cache_key: tuple,
    _simulated_data: gpd.GeoDataFrame,
    _boundary_data: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Cached wrapper for rasterized flood/population exposure."""

    return _population_flood_summary_uncached(
        _simulated_data,
        population_raster_path,
        _boundary_data,
    )


def population_flood_summary(
    simulated_data: gpd.GeoDataFrame,
    population_raster_path: str | Path,
    boundary_data: gpd.GeoDataFrame | None,
) -> pd.DataFrame:
    """Return cached population counts by simulated flood-depth band."""

    columns = ["site_name", "flood_depth_band", "people_in_band", "people_affected"]
    if boundary_data is None:
        return pd.DataFrame(columns=columns)
    return _population_flood_summary_cached(
        str(population_raster_path),
        _simulation_population_cache_key(simulated_data),
        _boundary_geometry_cache_key(boundary_data),
        simulated_data,
        boundary_data,
    )


def _population_data_cache_key(hazard_data: gpd.GeoDataFrame) -> str:
    """Build a small cache key without asking Streamlit to hash geometries."""

    digest = hashlib.sha256()
    digest.update(repr(hazard_data.crs).encode())
    for name in ("site_name", "source_file", "Var", "area_sqkm"):
        digest.update(repr(tuple(hazard_data.get(name, []))).encode())
    for geometry in hazard_data.geometry:
        digest.update(geometry.wkb if geometry is not None else b"missing")
    return digest.hexdigest()


@st.cache_data(show_spinner="Estimating population exposure...")
def _load_population_exposure_cached(
    population_raster_path: str,
    hazard_cache_key: str,
    _hazard_data: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Cached implementation; the underscore avoids hashing geometries."""

    return add_population_exposure(_hazard_data, population_raster_path)


def load_population_exposure(
    hazard_data: gpd.GeoDataFrame,
    population_raster_path: str | Path,
) -> gpd.GeoDataFrame:
    """Return cached population enrichment for a prepared hazard layer."""

    return _load_population_exposure_cached(
        str(population_raster_path),
        _population_data_cache_key(hazard_data),
        hazard_data,
    )


@st.cache_data(show_spinner="Preparing regional flood hazard layers...")
def load_data(data_dir: str | Path = ".") -> gpd.GeoDataFrame:
    """Cached Streamlit wrapper around :func:`preprocess_hazard_data`."""

    return preprocess_hazard_data(data_dir)


@st.cache_data(show_spinner=False)
def fetch_soil_properties(lat: float, lon: float) -> tuple[float, float, str]:
    """Legacy optional texture lookup; texture is insufficient to assign HSG."""

    try:
        response = requests.get(
            "https://rest.isric.org/soilgrids/v2.0/properties/query",
            params={"lon": lon, "lat": lat, "property": ["clay", "sand"],
                    "depth": "0-5cm", "value": "mean"}, timeout=5)
        response.raise_for_status()
        values = {layer["name"]: layer["depths"][0]["values"]["mean"] / 10.
                  for layer in response.json()["properties"]["layers"]
                  if layer.get("name") in {"clay", "sand"}}
        return float(values.get("clay", np.nan)), float(values.get("sand", np.nan)), "Texture only; HSG requires conductivity and subsurface diagnostics"
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return np.nan, np.nan, "SoilGrids unavailable; no measured soil profile"


def compute_polygon_depth(
    row: pd.Series, rain_mm: float, duration_h: float,
    prior_rain_mm: float = 20.0, benchmark_rain_mm: float = 180.0,
) -> float:
    """Corrected scalar baseline, using the same CN/depth logic as simulate_flood."""
    cn = get_effective_cn(row, prior_rain_mm)
    return float(empirical_depth_m(cn, row["base_depth_m"], rain_mm, duration_h, benchmark_rain_mm))


def simulate_flood(
    hazard_data: gpd.GeoDataFrame, rain_mm: float, duration_h: float,
    prior_rain_mm: float, threshold_m: float = DISPLAY_DEPTH_THRESHOLD_M,
    benchmark_rain_mm: float = 180.0,
    population_raster_path: str | Path | None = None,
    rainfall_event: RainfallEvent | None = None,
    routing_time_h: float = 1.0, use_routed_depth: bool = False,
) -> gpd.GeoDataFrame:
    """Corrected baseline or routed empirical scenario with shared event runoff.

    A supplied rainfall series is authoritative for total and duration. Routed
    depth remains an empirical NOAH-depth scaling; terrain inundation uses a
    separate model that does not use NOAH hazard as a physical predictor.
    """
    if not np.isfinite(prior_rain_mm) or prior_rain_mm < 0:
        raise ValueError("prior_rain_mm must be finite and non-negative")
    if not np.isfinite(threshold_m) or threshold_m < 0:
        raise ValueError("threshold_m must be finite and non-negative")
    event = rainfall_event or make_rainfall_event(rain_mm, duration_h)
    rain_mm, duration_h = event.total_mm, event.duration_h
    simulated = hazard_data.copy()
    if population_raster_path is not None and "population_estimate" not in simulated:
        simulated = add_population_exposure(simulated, population_raster_path)
    cn = effective_cn_values(simulated, prior_rain_mm)
    simulated["CN_amc"] = cn
    depths = empirical_depth_m(cn, simulated["base_depth_m"].to_numpy(dtype=float),
                               rain_mm, duration_h, benchmark_rain_mm)
    routed = route_runoff(event, cn, routing_time_h)
    if use_routed_depth:
        reference = route_runoff(make_rainfall_event(benchmark_rain_mm, duration_h), cn, routing_time_h)
        ratio = np.divide(routed["peak_storage_mm"], reference["peak_storage_mm"],
                          out=np.zeros_like(cn), where=reference["peak_storage_mm"] > 0)
        depths = simulated["base_depth_m"].to_numpy(dtype=float) * ratio * (1. - np.exp(-.75 * duration_h))
    simulated["sim_depth_m"] = depths
    simulated["flood_status"] = np.where(depths >= threshold_m, "Flooded", "Below threshold")
    simulated["flood_exposure_class"] = [depth_to_flood_class(depth) for depth in depths]
    population = pd.to_numeric(simulated.get("population_estimate", pd.Series(0., index=simulated.index)),
                               errors="coerce").fillna(0.)
    simulated["affected_population"] = np.where(depths >= EXPOSURE_DEPTH_THRESHOLD_M, population, 0.)
    area_m2 = (simulated["area_sqkm"].to_numpy(dtype=float) * 1e6 if "area_sqkm" in simulated
               else simulated.to_crs(AREA_CRS).geometry.area.to_numpy())
    simulated["runoff_mm"] = routed["runoff_mm"]
    simulated["runoff_volume_m3"] = routed["runoff_mm"] * area_m2 / 1000.
    simulated["peak_runoff_m3_s"] = routed["peak_mm_h"] * area_m2 / 3.6e6
    simulated["peak_runoff_time_h"] = routed["peak_time_h"]
    simulated["model_method"] = "Routed empirical depth" if use_routed_depth else "Corrected empirical baseline"
    volumes = area_m2 / 1000.
    simulated.attrs["runoff_hydrograph"] = pd.DataFrame({
        "time_h": routed["time_h"], "rain_mm": routed["rain_mm"],
        "runoff_generated_m3": routed["incremental_runoff_mm"] @ volumes,
        "outflow_m3_s": routed["outlet_mm_h"] @ volumes / 3600.,
        "outflow_interval_mean_m3_s": (routed["outflow_mm"] @ volumes) / (routed["interval_h"] * 3600.),
        "stored_water_m3": routed["storage_mm"] @ volumes,
    })
    simulated.attrs["runoff_balance_error_m3"] = float(routed["balance_error_mm"] @ volumes)
    return simulated


def flooded_area_sqkm(simulated_data: gpd.GeoDataFrame, threshold_m: float = DISPLAY_DEPTH_THRESHOLD_M) -> float:
    """Return flooded area using the pre-simplification area field."""

    active = simulated_data[simulated_data["sim_depth_m"] >= float(threshold_m)]
    if active.empty:
        return 0.0
    if "area_sqkm" in active.columns:
        return float(active["area_sqkm"].sum())
    return float(active.to_crs(AREA_CRS).geometry.area.sum() / 1_000_000.0)


def depth_to_spectrum_hex(depth: float, max_depth_m: float = 2.5) -> str:
    """Convert a simulated depth to a Folium-compatible color."""

    norm = min(max(float(depth) / max_depth_m, 0.0), 1.0)
    return mcolors.to_hex(FLOOD_DEPTH_COLORMAP(norm))


def risk_to_hex(var_code: int) -> str:
    """Convert UP NOAH's 1/2/3 hazard code to its display color."""

    return {1: "#FFFF00", 2: "#FFA500", 3: "#FF0000"}.get(
        int(var_code), "#CCCCCC"
    )


@st.cache_data(show_spinner=False)
@with_rasterio_runtime
def _population_dot_data(
    bounds: tuple[float, float, float, float],
    population_raster_path: str,
    max_dots: int = DEFAULT_POPULATION_DOT_COUNT,
) -> pd.DataFrame:
    """Aggregate population cells into a manageable set of dot locations."""

    with rasterio.open(population_raster_path) as source:
        if source.crs is None:
            raise ValueError("Population raster has no CRS")
        source_bounds = transform_bounds(DATA_CRS, source.crs, *bounds)
        requested_window = from_bounds(*source_bounds, transform=source.transform)
        try:
            window = requested_window.intersection(
                Window(0, 0, source.width, source.height)
            )
        except (ValueError, rasterio.errors.WindowError) as exc:
            raise ValueError("Selected map extent does not overlap the population raster") from exc

        source_pixel_count = max(window.width * window.height, 1.0)
        scale = min(1.0, np.sqrt(float(max_dots) / source_pixel_count))
        output_width = max(1, int(round(window.width * scale)))
        output_height = max(1, int(round(window.height * scale)))

        population_average = source.read(
            1,
            window=window,
            out_shape=(output_height, output_width),
            resampling=Resampling.average,
            masked=True,
        )
        valid_fraction = source.read_masks(
            1,
            window=window,
            out_shape=(output_height, output_width),
            resampling=Resampling.average,
        ).astype(float) / 255.0
        population_average = np.asarray(
            population_average.filled(np.nan), dtype=float
        )
        output_transform = source.window_transform(window)
        output_transform = output_transform * output_transform.scale(
            window.width / output_width,
            window.height / output_height,
        )

        # Each dot represents a small aggregated block of source cells.
        source_cells_per_dot = (window.width / output_width) * (
            window.height / output_height
        )
        population = population_average * source_cells_per_dot * valid_fraction
        rows, cols = np.indices((output_height, output_width), dtype=float)
        xs = output_transform.c + (cols + .5) * output_transform.a
        ys = output_transform.f + (rows + .5) * output_transform.e
        longitudes, latitudes = transform_coordinates(source.crs, DATA_CRS, xs.ravel(), ys.ravel())
        longitudes = np.asarray(longitudes).reshape(xs.shape)
        latitudes = np.asarray(latitudes).reshape(ys.shape)
        if source.crs.is_projected:
            unit_m = source.crs.linear_units_factor[1]
            cell_area = abs(source.transform.a * source.transform.e) * unit_m ** 2 / 1e6
        else:
            cell_area = abs(source.transform.a * source.transform.e) * 111.32 ** 2 * np.cos(np.deg2rad(latitudes))
        density = np.divide(population_average, cell_area,
                            out=np.full(population_average.shape, np.nan), where=np.asarray(cell_area) > 0.)
        valid = np.isfinite(population) & (population > 0.0)
        if not valid.any():
            return pd.DataFrame(
                columns=[
                    "longitude",
                    "latitude",
                    "population_estimate",
                    "population_density_per_sqkm",
                    "population_density_class",
                    "population_class_code",
                    "dot_radius",
                ]
            )

        values = population[valid]
        log_values = np.log1p(values)
        log_span = float(log_values.max() - log_values.min())
        relative_size = (
            np.zeros(log_values.shape)
            if log_span == 0.0
            else (log_values - log_values.min()) / log_span
        )
        density_values = density[valid]
        class_codes = _population_density_class_code(density_values)
        return pd.DataFrame(
            {
                "longitude": longitudes[valid],
                "latitude": latitudes[valid],
                "population_estimate": np.round(values, 2),
                "population_density_per_sqkm": np.round(density_values, 2),
                "population_density_class": [
                    POPULATION_DENSITY_LABELS[int(code)] for code in class_codes
                ],
                "population_class_code": class_codes,
                "dot_radius": np.round(2.5 + 4.5 * np.sqrt(relative_size), 2),
            }
        )


def _population_dots_in_boundaries(
    population_dots: pd.DataFrame,
    selected_boundaries: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Remove dots outside the selected administrative site boundaries."""

    if population_dots.empty or selected_boundaries.empty:
        return population_dots.copy()

    points = gpd.GeoDataFrame(
        population_dots.copy(),
        geometry=gpd.points_from_xy(
            population_dots["longitude"], population_dots["latitude"]
        ),
        crs=DATA_CRS,
    )
    joined = gpd.sjoin(
        points,
        selected_boundaries[["site_name", "geometry"]],
        how="inner",
        predicate="within",
    )
    if joined.empty:
        return population_dots.iloc[0:0].copy()
    return population_dots.loc[joined.index.unique()].copy()


def _population_dots_for_map(
    selected_data: gpd.GeoDataFrame,
    selected_boundaries: gpd.GeoDataFrame,
    population_raster_path: str,
) -> pd.DataFrame:
    """Build bounded dot data without reading the gaps between distant sites."""

    if selected_data.empty:
        return pd.DataFrame()
    if selected_boundaries.empty:
        bounds = tuple(float(value) for value in selected_data.total_bounds)
        return _population_dot_data(
            bounds, population_raster_path, DEFAULT_POPULATION_DOT_COUNT
        )

    metric_boundaries = selected_boundaries.to_crs(AREA_CRS)
    areas = metric_boundaries.geometry.area.to_numpy(dtype=float)
    area_total = float(areas.sum())
    if area_total <= 0.0:
        targets = np.full(len(selected_boundaries), 100, dtype=int)
    else:
        targets = np.maximum(
            50,
            np.round(
                DEFAULT_POPULATION_DOT_COUNT * areas / area_total
            ).astype(int),
        )

    dot_frames = []
    for boundary, target in zip(selected_boundaries.itertuples(), targets):
        bounds = tuple(float(value) for value in boundary.geometry.bounds)
        dot_frames.append(
            _population_dot_data(bounds, population_raster_path, int(target))
        )
    dots = pd.concat(dot_frames, ignore_index=True) if dot_frames else pd.DataFrame()
    if not dots.empty:
        log_values = np.log1p(dots["population_estimate"].to_numpy(dtype=float))
        log_span = float(log_values.max() - log_values.min())
        relative_size = (
            np.zeros(log_values.shape)
            if log_span == 0.0
            else (log_values - log_values.min()) / log_span
        )
        dots["dot_radius"] = np.round(
            2.5 + 4.5 * np.sqrt(relative_size), 2
        )
    return _population_dots_in_boundaries(dots, selected_boundaries)


def _selected_boundary_data(
    selected_data: gpd.GeoDataFrame,
    boundary_data: gpd.GeoDataFrame | None,
) -> gpd.GeoDataFrame:
    """Return only administrative boundaries represented in selected hazards."""

    empty = gpd.GeoDataFrame(columns=["site_name", "geometry"], crs=DATA_CRS)
    if boundary_data is None or boundary_data.empty or selected_data.empty:
        return empty
    selected_sites = set(selected_data["site_name"].dropna().unique())
    selected_boundaries = boundary_data[
        boundary_data["site_name"].isin(selected_sites)
    ].copy()
    if selected_boundaries.crs != DATA_CRS:
        selected_boundaries = selected_boundaries.to_crs(DATA_CRS)
    return selected_boundaries


def _map_extent(
    selected_data: gpd.GeoDataFrame,
    selected_boundaries: gpd.GeoDataFrame,
) -> tuple[list[float], list[list[float]], int]:
    """Return map center, bounds, and selected site count."""

    if selected_data.empty:
        return [14.6091, 121.0223], None, 0

    extent_bounds = [selected_data.total_bounds]
    if not selected_boundaries.empty:
        extent_bounds.append(selected_boundaries.total_bounds)
    extent_bounds = np.asarray(extent_bounds)
    min_lon = extent_bounds[:, 0].min()
    min_lat = extent_bounds[:, 1].min()
    max_lon = extent_bounds[:, 2].max()
    max_lat = extent_bounds[:, 3].max()
    center = [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0]
    bounds = [[min_lat, min_lon], [max_lat, max_lon]]
    return center, bounds, selected_data["site_name"].nunique()


def _add_boundary_layer(
    map_object: folium.Map,
    selected_boundaries: gpd.GeoDataFrame,
    boundary_color: str = "#333333",
) -> None:
    """Add selected site boundaries as outlines without filling them."""

    if selected_boundaries.empty:
        return
    folium.GeoJson(
        selected_boundaries,
        name="Administrative boundaries for hazard sites",
        show=True,
        style_function=lambda feature: {
            "fillColor": "#FFFFFF",
            "color": boundary_color,
            "weight": 1.2,
            "dashArray": "5, 4",
            "fillOpacity": 0.0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["boundary_name"],
            aliases=["Hazard site boundary:"],
            localize=True,
        ),
    ).add_to(map_object)


def _boundary_geometry_cache_key(
    selected_boundaries: gpd.GeoDataFrame,
) -> tuple[tuple[str, tuple[float, float, float, float], float], ...]:
    """Build a stable cache key for the selected boundary geometries."""

    if selected_boundaries is None:
        return ()
    return tuple((str(row.site_name), str(selected_boundaries.crs),
                  hashlib.sha256(row.geometry.wkb).hexdigest())
                 for row in selected_boundaries.itertuples())


@st.cache_data(show_spinner=False)
@with_rasterio_runtime
def _population_heatmap_data(
    population_raster_path: str,
    boundary_cache_key: tuple[tuple[str, tuple[float, float, float, float], float], ...],
    _selected_boundaries: gpd.GeoDataFrame,
) -> tuple[list[dict], float]:
    """Prepare transparent RGBA population heatmap tiles per hazard boundary."""

    del boundary_cache_key  # The key controls caching; geometry is passed by underscore.
    prepared_tiles = []
    maximum_population = 0.0

    with rasterio.open(population_raster_path) as source:
        source_boundaries = _selected_boundaries.to_crs(source.crs)
        for boundary in source_boundaries.itertuples():
            try:
                masked_values, transform = raster_mask(
                    source,
                    [mapping(boundary.geometry)],
                    crop=True,
                    all_touched=False,
                    filled=False,
                )
            except (ValueError, rasterio.errors.RasterioIOError):
                continue

            values = np.asarray(masked_values[0].filled(np.nan), dtype=float)
            if source.crs != rasterio.crs.CRS.from_string(DATA_CRS):
                source_bounds = array_bounds(*values.shape, transform)
                geographic_transform, width, height = calculate_default_transform(
                    source.crs, DATA_CRS, values.shape[1], values.shape[0], *source_bounds)
                geographic_values = np.full((height, width), np.nan, dtype=float)
                reproject(values, geographic_values, src_transform=transform, src_crs=source.crs,
                          src_nodata=np.nan, dst_transform=geographic_transform, dst_crs=DATA_CRS,
                          dst_nodata=np.nan, resampling=Resampling.nearest)
                values, transform = geographic_values, geographic_transform
            valid = np.isfinite(values) & (values > 0.0)
            if not valid.any():
                continue
            maximum_population = max(
                maximum_population, float(np.nanmax(values[valid]))
            )
            prepared_tiles.append(
                {
                    "site_name": str(boundary.site_name),
                    "values": values,
                    "valid": valid,
                    "transform": transform,
                }
            )

    if not prepared_tiles:
        return [], 0.0

    normalizer = mcolors.LogNorm(vmin=1.0, vmax=max(10.0, maximum_population))
    rendered_tiles = []
    for tile in prepared_tiles:
        safe_values = np.where(tile["valid"], np.maximum(tile["values"], 1.0), 1.0)
        rgba = POPULATION_HEATMAP_COLORMAP(normalizer(safe_values))
        rgba[..., 3] = np.where(tile["valid"], 0.90, 0.0)
        height, width = tile["values"].shape
        west, south, east, north = array_bounds(
            height, width, tile["transform"]
        )
        rendered_tiles.append(
            {
                "site_name": tile["site_name"],
                "image": rgba,
                "bounds": [[south, west], [north, east]],
            }
        )
    return rendered_tiles, maximum_population


def _population_heatmap_legend_html(maximum_population: float) -> str:
    """Build a logarithmic population-per-pixel legend."""

    vmax = max(10.0, float(maximum_population))
    labels = [1.0, 10.0, 100.0, 1_000.0, 10_000.0, vmax]
    labels = sorted({min(value, vmax) for value in labels})
    label_html = "".join(
        f"<span>{value:,.0f}</span>" for value in labels
    )
    return f"""
    <div style="position: fixed; bottom: 30px; left: 30px; width: 285px; z-index:9999;
                background:#fff; color:#000; padding:10px; border-radius:8px;
                border:2px solid grey; font:12px sans-serif;">
        <b>Estimated population per raster pixel (2026)</b><br>
        <div style="background:linear-gradient(to right,#FFFFE5,#FFF7BC,#FEC44F,#FD8D3C,#E31A1C,#800026);
                    height:16px; width:100%; border-radius:3px; margin:6px 0;"></div>
        <div style="display:flex; justify-content:space-between; gap:4px;">
            {label_html}
        </div>
        <div style="margin-top:6px;">All positive population estimates are treated as vulnerable population.</div>
    </div>
    """


def _population_dot_legend_html(population_mode: str) -> str:
    """Build the legend for one of the separate population dot maps."""

    if population_mode == "Affected population by flood":
        legend_body = f"""
            <b>Affected population by flood depth</b><br>
            <div style="background:linear-gradient(to right,#ffffd9,#41b6c4,#225ea8,#081d58);
                        height:16px; width:100%; border-radius:3px; margin:6px 0;"></div>
            <div style="display:flex; justify-content:space-between;">
                <span>{EXPOSURE_DEPTH_THRESHOLD_M:.2f} m</span><span>1 m</span><span>2 m+</span>
            </div>
            <div style="margin-top:6px;">Only dots at simulated depths of {EXPOSURE_DEPTH_THRESHOLD_M:.2f} m or more are shown.</div>
        """
    else:
        legend_body = """
            <b>Population density dots</b><br>
            <div style="line-height:1.6; margin-top:5px;">
                <i style="background:#FFF7BC; width:12px; height:12px; display:inline-block;"></i> Low (&lt;1,000 people/km²)<br>
                <i style="background:#FEC44F; width:12px; height:12px; display:inline-block;"></i> Moderate (1,000–5,000)<br>
                <i style="background:#D95F0E; width:12px; height:12px; display:inline-block;"></i> High (≥5,000)
            </div>
            <div style="margin-top:6px;">Dot size represents the estimated population in its aggregated cell. Areas with no dot have no positive raster population estimate.</div>
        """

    return f"""
    <div style="position: fixed; bottom: 30px; left: 30px; width: 285px; z-index:9999;
                background:#fff; color:#000; padding:10px; border-radius:8px;
                border:2px solid grey; font:12px sans-serif;">
        {legend_body}
    </div>
    """


def _population_dots_in_flood(
    population_dots: pd.DataFrame,
    simulated_data: gpd.GeoDataFrame,
    threshold_m: float = EXPOSURE_DEPTH_THRESHOLD_M,
) -> pd.DataFrame:
    """Keep population dots whose centers fall inside simulated flood areas."""

    if population_dots.empty or simulated_data.empty:
        return population_dots.iloc[0:0].copy()

    points = gpd.GeoDataFrame(
        population_dots.copy(),
        geometry=gpd.points_from_xy(
            population_dots["longitude"], population_dots["latitude"]
        ),
        crs=DATA_CRS,
    )
    terrain = simulated_data.attrs.get("terrain_result")
    if terrain is not None:
        projected = points.to_crs(terrain.grid.crs)
        columns, rows = ~terrain.grid.transform * (projected.geometry.x.to_numpy(), projected.geometry.y.to_numpy())
        columns, rows = np.floor(columns).astype(int), np.floor(rows).astype(int)
        inside = (rows >= 0) & (columns >= 0) & (rows < terrain.grid.shape[0]) & (columns < terrain.grid.shape[1])
        depths = np.full(len(points), np.nan)
        depths[inside] = terrain.max_depth_m[rows[inside], columns[inside]]
        if terrain.report_mask is not None:
            modeled = np.zeros(len(points), bool)
            modeled[inside] = terrain.report_mask[rows[inside], columns[inside]]
            depths[~modeled] = np.nan
        points["sim_depth_m"] = depths
    else:
        flood_polygons = simulated_data[simulated_data["sim_depth_m"] >= float(threshold_m)][["sim_depth_m", "geometry"]].copy()
        if flood_polygons.empty:
            return population_dots.iloc[0:0].copy()
        if flood_polygons.crs != points.crs:
            flood_polygons = flood_polygons.to_crs(points.crs)
        joined = gpd.sjoin(points, flood_polygons, how="left", predicate="within")
        max_depth_by_point = joined.groupby(level=0)["sim_depth_m"].max()
        points["sim_depth_m"] = points.index.to_series().map(max_depth_by_point).fillna(0.).to_numpy()
    points = points[points["sim_depth_m"] >= float(threshold_m)].copy()
    if points.empty:
        return population_dots.iloc[0:0].copy()
    points["flood_exposure_class"] = points["sim_depth_m"].apply(
        depth_to_flood_class, threshold_m=threshold_m
    )
    points["affected_population"] = points["population_estimate"]
    return pd.DataFrame(points.drop(columns="geometry"))


def build_population_density_map(
    selected_data: gpd.GeoDataFrame,
    population_raster_path: str | Path,
    boundary_data: gpd.GeoDataFrame | None = None,
) -> folium.Map:
    """Build the reference-style logarithmic population-density heatmap."""

    selected_boundaries = _selected_boundary_data(selected_data, boundary_data)
    center, bounds, site_count = _map_extent(selected_data, selected_boundaries)
    map_object = folium.Map(
        location=center,
        zoom_start=7 if site_count > 1 else 11,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )
    folium.TileLayer(
        "OpenStreetMap", name="Street Names Map", show=False
    ).add_to(map_object)
    if bounds:
        map_object.fit_bounds(bounds, padding=(10, 10))
    _add_boundary_layer(
        map_object,
        selected_boundaries,
        boundary_color="#FFFFFF",
    )

    heatmap_tiles, maximum_population = _population_heatmap_data(
        str(population_raster_path),
        _boundary_geometry_cache_key(selected_boundaries),
        selected_boundaries,
    )
    for tile in heatmap_tiles:
        folium.raster_layers.ImageOverlay(
            image=tile["image"],
            bounds=tile["bounds"],
            origin="upper",
            mercator_project=False,
            pixelated=False,
            opacity=1.0,
            name=f"Population density heatmap — {tile['site_name']}",
        ).add_to(map_object)

    map_object.get_root().html.add_child(
        folium.Element(_population_heatmap_legend_html(maximum_population))
    )
    folium.LayerControl(collapsed=False).add_to(map_object)
    return map_object


def build_population_map(
    selected_data: gpd.GeoDataFrame,
    population_raster_path: str | Path,
    population_mode: str = "Population density",
    boundary_data: gpd.GeoDataFrame | None = None,
    simulated_data: gpd.GeoDataFrame | None = None,
    threshold_m: float = EXPOSURE_DEPTH_THRESHOLD_M,
) -> folium.Map:
    """Build the affected-population dot map or the density heatmap facade."""

    if population_mode == "Population density":
        return build_population_density_map(
            selected_data,
            population_raster_path,
            boundary_data=boundary_data,
        )

    selected_boundaries = _selected_boundary_data(selected_data, boundary_data)
    center, bounds, site_count = _map_extent(selected_data, selected_boundaries)
    map_object = folium.Map(
        location=center,
        zoom_start=7 if site_count > 1 else 11,
        tiles="CartoDB positron",
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="Street Names Map").add_to(map_object)
    if bounds:
        map_object.fit_bounds(bounds, padding=(10, 10))
    _add_boundary_layer(map_object, selected_boundaries)

    population_dots = _population_dots_for_map(
        selected_data,
        selected_boundaries,
        str(population_raster_path),
    )
    if population_mode == "Affected population by flood":
        if simulated_data is None:
            raise ValueError(
                "simulated_data is required for the affected population map"
            )
        population_dots = _population_dots_in_flood(
            population_dots, simulated_data, threshold_m=threshold_m
        )

    dot_layer_name = (
        "Affected population dots"
        if population_mode == "Affected population by flood"
        else "Population density dots"
    )
    dot_layer = folium.FeatureGroup(name=dot_layer_name, show=True)
    for row in population_dots.itertuples(index=False):
        if population_mode == "Affected population by flood":
            color = depth_to_spectrum_hex(row.sim_depth_m)
            tooltip = (
                f"{row.flood_exposure_class} | Flood depth: {row.sim_depth_m:.2f} m "
                f"| People affected: {row.affected_population:,.0f}"
            )
            popup = (
                f"Flood exposure: {row.flood_exposure_class}<br>"
                f"Flood depth: {row.sim_depth_m:.2f} m<br>"
                f"People affected: {row.affected_population:,.0f}"
            )
        else:
            color = POPULATION_DENSITY_COLORS[int(row.population_class_code)]
            tooltip = (
                f"{row.population_density_class} | Population: "
                f"{row.population_estimate:,.0f} | Density: "
                f"{row.population_density_per_sqkm:,.0f} people/km²"
            )
            popup = (
                f"Population density: {row.population_density_class}<br>"
                f"Estimated population: {row.population_estimate:,.0f}<br>"
                f"Density: {row.population_density_per_sqkm:,.0f} people/km²"
            )
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=float(row.dot_radius),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=0.35,
            tooltip=tooltip,
            popup=folium.Popup(popup, max_width=280),
        ).add_to(dot_layer)
    dot_layer.add_to(map_object)

    map_object.get_root().html.add_child(
        folium.Element(_population_dot_legend_html(population_mode))
    )
    folium.LayerControl().add_to(map_object)
    return map_object


def _legend_html(map_mode: str) -> str:
    """Build the legend for the flood-hazard map only."""

    if map_mode == "Continuous depth spectrum":
        legend_body = f"""
            <b>Simulated water depth</b><br>
            <div style="background:linear-gradient(to right,#E0F3F8,#ABD9E9,#74ADD1,#4575B4,#313695,#081D58);
                        height:16px; width:100%; border-radius:3px; margin:6px 0;"></div>
            <div style="display:flex; justify-content:space-between;">
                <span>{DISPLAY_DEPTH_THRESHOLD_M:.2f} m</span><span>1.0 m</span><span>2.5 m+</span>
            </div>
            <div style="margin-top:6px;">Blue tones show simulated flood depth; hazard classes remain yellow, orange, and red.</div>
        """
    else:
        legend_body = """
            <b>UP NOAH baseline hazard</b><br>
            <div style="margin-top:5px; line-height:1.6;">
                <i style="background:#FFFF00; width:12px; height:12px; display:inline-block;"></i> Low<br>
                <i style="background:#FFA500; width:12px; height:12px; display:inline-block;"></i> Moderate<br>
                <i style="background:#FF0000; width:12px; height:12px; display:inline-block;"></i> High
            </div>
        """

    return f"""
    <div style="position: fixed; bottom: 30px; left: 30px; width: 250px; z-index:9999;
                background:#fff; color:#000; padding:10px; border-radius:8px;
                border:2px solid grey; font:12px sans-serif;">
        {legend_body}
    </div>
    """


def build_flood_map(
    selected_data: gpd.GeoDataFrame,
    active_data: gpd.GeoDataFrame,
    map_mode: str,
    boundary_data: gpd.GeoDataFrame | None = None,
) -> folium.Map:
    """Build a Folium map focused on selected hazards and their boundaries."""

    selected_boundaries = _selected_boundary_data(selected_data, boundary_data)
    center, bounds, site_count = _map_extent(selected_data, selected_boundaries)

    map_object = folium.Map(
        location=center,
        zoom_start=7 if site_count > 1 else 11,
        tiles="CartoDB positron",
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="Street Names Map").add_to(map_object)
    if bounds:
        map_object.fit_bounds(bounds, padding=(10, 10))

    _add_boundary_layer(map_object, selected_boundaries)

    if not active_data.empty:
        active_data = active_data.copy()
        if map_mode == "Continuous depth spectrum":
            active_data["fill_color"] = active_data["sim_depth_m"].apply(
                depth_to_spectrum_hex
            )
        else:
            active_data["fill_color"] = active_data["Var"].apply(risk_to_hex)

        flood_fields = [
            "site_name",
            "hazard_level",
            "sim_depth_m",
            "CN_amc",
            "source_return_period_yr",
            "fill_color",
            "geometry",
        ]
        active_data = active_data[
            [field for field in flood_fields if field in active_data.columns]
        ].copy()

        folium.GeoJson(
            active_data,
            name="Simulated flood polygons",
            style_function=lambda feature: {
                "fillColor": feature["properties"].get("fill_color", "#CCCCCC"),
                "color": feature["properties"].get("fill_color", "#CCCCCC"),
                "weight": 0.3,
                "fillOpacity": 0.70,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    field
                    for field in [
                        "site_name",
                        "hazard_level",
                        "sim_depth_m",
                        "CN_amc",
                        "source_return_period_yr",
                    ]
                    if field in active_data.columns
                ],
                aliases=[
                    alias
                    for field, alias in [
                        ("site_name", "Site:"),
                        ("hazard_level", "NOAH hazard:"),
                        ("sim_depth_m", "Calculated depth (m):"),
                        ("CN_amc", "Adjusted CN:"),
                        ("source_return_period_yr", "Source return period (yr):"),
                    ]
                    if field in active_data.columns
                ],
                localize=True,
            ),
        ).add_to(map_object)

    map_object.get_root().html.add_child(
        folium.Element(_legend_html(map_mode))
    )
    folium.LayerControl().add_to(map_object)
    return map_object


def _site_summary(
    simulated_data: gpd.GeoDataFrame,
    risk_summary: pd.DataFrame | None = None,
    flood_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize mapped/flooded area and non-overlapping population by site."""

    summary = (
        simulated_data.groupby("site_name", sort=False)
        .agg(
            polygons=("Var", "size"),
            mapped_area_sqkm=("area_sqkm", "sum"),
            max_depth_m=("sim_depth_m", "max"),
        )
        .reset_index()
    )
    active_area = (
        simulated_data[simulated_data["sim_depth_m"] >= DISPLAY_DEPTH_THRESHOLD_M]
        .groupby("site_name")["area_sqkm"]
        .sum()
    )
    summary["flooded_area_sqkm"] = summary["site_name"].map(active_area).fillna(0.0)
    if risk_summary is not None and not risk_summary.empty:
        population_by_site = risk_summary.groupby("site_name")["people"].sum()
    elif "population_estimate" in simulated_data.columns:
        population_by_site = simulated_data.groupby("site_name")[
            "population_estimate"
        ].sum(min_count=1)
    else:
        population_by_site = pd.Series(dtype=float)
    if flood_summary is not None and not flood_summary.empty:
        affected_by_site = flood_summary.groupby("site_name")[
            "people_affected"
        ].sum()
    elif "affected_population" in simulated_data.columns:
        affected_by_site = simulated_data.groupby("site_name")[
            "affected_population"
        ].sum(min_count=1)
    else:
        affected_by_site = pd.Series(dtype=float)
    if not population_by_site.empty or not affected_by_site.empty:
        summary["mapped_population"] = (
            summary["site_name"].map(population_by_site).fillna(0.0)
        )
        summary["affected_population"] = (
            summary["site_name"].map(affected_by_site).fillna(0.0)
        )
    return summary


def population_impact_summary(
    simulated_data: gpd.GeoDataFrame,
    population_raster_path: str | Path | None = None,
    boundary_data: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """Summarize people in the requested simulated flood-depth bands."""

    if population_raster_path is not None and boundary_data is not None:
        return population_flood_summary(
            simulated_data,
            population_raster_path,
            boundary_data,
        )

    if "flood_exposure_class" not in simulated_data.columns:
        return pd.DataFrame(
            {
                "flood_depth_band": FLOOD_EXPOSURE_ORDER,
                "people_in_band": [0.0] * len(FLOOD_EXPOSURE_ORDER),
                "people_affected": [0.0] * len(FLOOD_EXPOSURE_ORDER),
            }
        )

    exposure_data = simulated_data[["flood_exposure_class"]].copy()
    exposure_data["people_in_band"] = pd.to_numeric(
        simulated_data.get(
            "population_estimate", pd.Series(0.0, index=simulated_data.index)
        ),
        errors="coerce",
    ).fillna(0.0)
    exposure_data["affected_population"] = pd.to_numeric(
        simulated_data.get(
            "affected_population", pd.Series(0.0, index=simulated_data.index)
        ),
        errors="coerce",
    ).fillna(0.0)
    summary = (
        exposure_data.groupby("flood_exposure_class", sort=False)
        .agg(
            people_in_band=("people_in_band", "sum"),
            people_affected=("affected_population", "sum"),
        )
        .reindex(FLOOD_EXPOSURE_ORDER, fill_value=0.0)
        .rename_axis("flood_depth_band")
        .reset_index()
    )
    return summary


def _population_exposure_sensitivity_uncached(
    simulated_data, population_raster_path, boundary_data,
    thresholds_m=EXPOSURE_SENSITIVITY_THRESHOLDS_M,
):
    """Threshold sensitivity on disjoint population cells within site boundaries."""
    thresholds = np.asarray(thresholds_m, dtype=float)
    if np.any(~np.isfinite(thresholds)) or np.any(thresholds < 0):
        raise ValueError("Exposure thresholds must be finite and non-negative")
    rows = []
    boundaries = _selected_boundary_data(simulated_data, boundary_data)
    if boundaries is None or boundaries.empty:
        return pd.DataFrame(columns=["site_name", "threshold_m", "people_affected", "people_modeled", "people_unmodeled"])
    with rasterio.open(population_raster_path) as source:
        for boundary in boundaries.to_crs(source.crs).itertuples():
            population, transform = _population_values_for_boundary(source, boundary.geometry)
            if population is None:
                continue
            site = simulated_data[simulated_data["site_name"].eq(boundary.site_name)]
            depths = _rasterize_simulated_depth(site, source.crs, population.shape, transform)
            populated = np.isfinite(population) & (population > 0.)
            modeled = populated & np.isfinite(depths)
            for threshold in thresholds:
                rows.append({"site_name": boundary.site_name, "threshold_m": float(threshold),
                             "people_affected": float(population[modeled & (depths >= threshold)].sum()),
                             "people_modeled": float(population[modeled].sum()),
                             "people_unmodeled": float(population[populated & ~np.isfinite(depths)].sum())})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, max_entries=16, show_spinner=False)
def _exposure_sensitivity_cached(population_path, simulation_key, boundary_key, thresholds_m,
                                 _simulated_data, _boundary_data):
    return _population_exposure_sensitivity_uncached(_simulated_data, population_path, _boundary_data, thresholds_m)


def population_exposure_sensitivity(simulated_data, population_raster_path, boundary_data,
                                    thresholds_m=EXPOSURE_SENSITIVITY_THRESHOLDS_M):
    if boundary_data is None or boundary_data.empty:
        return pd.DataFrame(columns=["site_name", "threshold_m", "people_affected", "people_modeled", "people_unmodeled"])
    return _exposure_sensitivity_cached(str(population_raster_path),
                                       _simulation_population_cache_key(simulated_data),
                                       _boundary_geometry_cache_key(boundary_data), tuple(thresholds_m),
                                       simulated_data, boundary_data)


def _input_file_key(inputs, dem_path=""):
    paths = [dem_path] + [str(getattr(inputs, name)) for name in inputs.__dataclass_fields__ if name.endswith("_path")]
    return tuple((p, Path(p).stat().st_mtime_ns, Path(p).stat().st_size) for p in paths if p)


@st.cache_data(ttl=3600, max_entries=8, show_spinner="Preparing spatial runoff inputs...")
def _runoff_inputs_cached(polygon_key, inputs, file_key, resolution_m, _polygons):
    return enrich_runoff_inputs(_polygons, inputs, resolution_m)


@st.cache_data(ttl=3600, max_entries=4, show_spinner="Preparing terrain connectivity...")
def _terrain_inputs_cached(dem_path, resolution_m, inputs, file_key):
    grid, dem = load_dem_grid(dem_path, resolution_m)
    cn, drainage, metadata = load_physical_grid(grid, inputs, np.isfinite(dem))
    network = build_flow_network(dem, grid)
    return grid, dem, cn, drainage, metadata, network


@st.cache_data(ttl=3600, max_entries=8, show_spinner="Routing water over terrain...")
def _terrain_event_cached(dem_path, resolution_m, inputs, file_key, event, prior_rain_mm, manning_n, recession_h):
    grid, dem, cn, drainage, metadata, network = _terrain_inputs_cached(dem_path, resolution_m, inputs, file_key)
    result = simulate_terrain(dem, grid, cn, event, prior_rain_mm, drainage,
                              manning_n, recession_h, network=network)
    result.diagnostics["runoff_inputs"] = metadata
    return result


def _physical_inputs_controls(data_dir, site_name):
    config_path = data_dir / "data/hydrology/inputs.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    global_settings = dict(config.get("physical_inputs", {}))
    all_site_settings = config.get("physical_inputs_by_site", {})
    if not isinstance(all_site_settings, dict):
        raise ValueError("physical_inputs_by_site must be an object in data/hydrology/inputs.json")
    site_settings = dict(all_site_settings.get(site_name, {}))
    settings = {**global_settings, **site_settings}
    if set(settings).difference(PhysicalInputs.__dataclass_fields__):
        raise ValueError("Unknown physical_inputs field(s) in data/hydrology/inputs.json")
    defaults = {"land_cover_path": "worldcover.tif", "hsg_path": "hsg.tif",
                "ksat_0_50_path": "ksat_0_50_um_s.tif", "ksat_0_100_path": "ksat_0_100_um_s.tif",
                "restrictive_depth_path": "restrictive_depth_cm.tif", "water_table_depth_path": "water_table_depth_cm.tif",
                "impervious_fraction_path": "impervious_fraction.tif", "drainage_capacity_path": "drainage_mm_h.tif"}
    hydrology_dir = data_dir / "data/hydrology"
    site_dir = hydrology_dir / "sites" / site_key(site_name)
    for key, name in defaults.items():
        site_default = site_dir / name
        shared_default = hydrology_dir / name
        if key in site_settings:
            value = site_settings[key]
        elif site_default.exists():
            value = str(site_default)
        else:
            value = settings.get(key, str(shared_default) if shared_default.exists() else "")
        if value and not Path(value).is_absolute():
            value = str(data_dir / value)
        settings[key] = value
    initial = PhysicalInputs(**settings)
    with st.expander("Soil and land-cover inputs", expanded=False):
        st.caption(f"Inputs are matched to {site_name}; missing cells use the scenario profile below.")
        hsg = st.selectbox("Assumed HSG for missing soil data", ["A", "B", "C", "D"],
                           index="ABCD".index(normalize_hsg(initial.assumed_hsg)))
        cover_codes = list(LAND_COVER_LABELS)
        cover = st.selectbox("Assumed land cover for missing data", cover_codes,
                             index=cover_codes.index(initial.assumed_land_cover), format_func=LAND_COVER_LABELS.get)
        fraction = st.slider("Impervious fraction of built-up cells", 0., 1., initial.built_up_fraction, .05)
        for key, label in [("land_cover_path", "ESA WorldCover raster"), ("hsg_path", "HSG raster (1=A .. 4=D)")]:
            settings[key] = st.text_input(label, value=settings[key], key=f"{site_key(site_name)}::{key}").strip()
        with st.expander("Soil diagnostics and drainage rasters"):
            labels = {"ksat_0_50_path": "Minimum Ksat, 0–50 cm (µm/s)",
                      "ksat_0_100_path": "Minimum Ksat, 0–100 cm (µm/s)",
                      "restrictive_depth_path": "Restrictive-layer depth (cm)",
                      "water_table_depth_path": "Seasonal water-table depth (cm)",
                      "impervious_fraction_path": "Impervious fraction raster (0–1)",
                      "drainage_capacity_path": "Drainage capacity raster (mm/h)"}
            for key, label in labels.items():
                settings[key] = st.text_input(label, value=settings[key], key=f"{site_key(site_name)}::{key}").strip()
            settings["dual_hsg_drained"] = st.checkbox("Adequate drainage documented for dual HSG cells", value=initial.dual_hsg_drained)
        settings["assumed_drainage_mm_h"] = st.number_input("Assumed drainage where no data (mm/h)", min_value=0., max_value=100., value=float(initial.assumed_drainage_mm_h))
    settings.update(assumed_hsg=hsg, assumed_land_cover=cover, built_up_fraction=fraction)
    # Typed relative paths are resolved against the application, consistently.
    for key in defaults:
        if settings[key] and not Path(settings[key]).is_absolute():
            settings[key] = str(data_dir / settings[key])
    dem_default = config.get("dem_path", "qc_dem_srtm.tif")
    if dem_default and not Path(dem_default).is_absolute():
        dem_default = str(data_dir / dem_default)
    return PhysicalInputs(**settings), dem_default


@with_rasterio_runtime
def build_terrain_flood_map(simulated_data, reference_data, map_mode, boundary_data=None):
    """Render a raster water-depth surface with an independent NOAH reference."""
    terrain = simulated_data.attrs["terrain_result"]
    boundaries = _selected_boundary_data(simulated_data, boundary_data)
    center, bounds, site_count = _map_extent(simulated_data, boundaries)
    map_object = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron", control_scale=True)
    if bounds:
        map_object.fit_bounds(bounds)
    _add_boundary_layer(map_object, boundaries)
    source_bounds = array_bounds(*terrain.grid.shape, terrain.grid.transform)
    transform, width, height = rasterio.warp.calculate_default_transform(
        terrain.grid.crs, DATA_CRS, terrain.grid.shape[1], terrain.grid.shape[0], *source_bounds)
    depths = terrain_depth_on_grid(terrain, (height, width), transform, DATA_CRS)
    rgba = FLOOD_DEPTH_COLORMAP(np.clip(np.nan_to_num(depths) / 2.5, 0., 1.))
    rgba[:, :, 3] = np.where(np.isfinite(depths) & (depths >= DISPLAY_DEPTH_THRESHOLD_M), .75, 0.)
    west, south, east, north = array_bounds(height, width, transform)
    folium.raster_layers.ImageOverlay(rgba, bounds=[[south, west], [north, east]],
                                      name="Terrain event maximum flood depth", origin="upper", opacity=1.).add_to(map_object)
    # A visible outline distinguishes coverage from dry areas.
    from rasterio.features import shapes
    coverage = terrain.report_mask if terrain.report_mask is not None else terrain.valid
    polygons = [geometry for geometry, value in shapes(coverage.astype("uint8"), mask=coverage,
                                                        transform=terrain.grid.transform) if value == 1]
    from shapely.geometry import shape
    domain = gpd.GeoDataFrame(geometry=[shape(g) for g in polygons], crs=terrain.grid.crs).to_crs(DATA_CRS)
    folium.GeoJson(domain, name="DEM model coverage", style_function=lambda f: {
        "fillOpacity": 0., "color": "#2B8CBE", "weight": 2.}).add_to(map_object)
    if reference_data is not None and not reference_data.empty:
        reference = reference_data[["Var", "hazard_level", "geometry"]].copy()
        folium.GeoJson(reference, name="UP NOAH hazard reference", show=map_mode == "UP NOAH risk classification",
                       style_function=lambda f: {"fillColor": risk_to_hex(f["properties"]["Var"]),
                                                 "fillOpacity": .3, "weight": .4, "color": "#666"},
                       tooltip=folium.GeoJsonTooltip(fields=["hazard_level"], aliases=["NOAH reference:"])).add_to(map_object)
    map_object.get_root().html.add_child(folium.Element(_legend_html("Continuous depth spectrum")))
    folium.LayerControl().add_to(map_object)
    return map_object


def _render_rainfall_runoff_analysis(simulated_data, event):
    hydrograph = (simulated_data.attrs["terrain_result"].hydrograph if "terrain_result" in simulated_data.attrs
                  else simulated_data.attrs["runoff_hydrograph"])
    rainfall, runoff, timing = rainfall_runoff_analysis(event, hydrograph)
    domain = [0., timing["axis_end_h"]]
    storm_end = alt.Chart(pd.DataFrame({"time_h": [event.duration_h]})).mark_rule(
        color="#888888", strokeDash=[4, 4]).encode(x="time_h:Q")
    st.subheader("Rainfall timing")
    rain_chart = alt.Chart(rainfall).mark_rect(color="#4575B4").encode(
        x=alt.X("start_h:Q", title="Hours from storm start", scale=alt.Scale(domain=domain)),
        x2="time_h:Q", y=alt.Y("intensity_mm_h:Q", title="Rainfall intensity (mm/h)"), y2=alt.datum(0),
        tooltip=[alt.Tooltip("start_h:Q", title="Interval start (h)"),
                 alt.Tooltip("time_h:Q", title="Interval end (h)"),
                 alt.Tooltip("intensity_mm_h:Q", title="Intensity (mm/h)", format=".2f")])
    st.altair_chart((rain_chart + storm_end).properties(height=210), width="stretch")
    st.subheader("Runoff response")
    runoff_chart = alt.Chart(runoff).mark_line(color="#238B45").encode(
        x=alt.X("time_h:Q", title="Hours from storm start", scale=alt.Scale(domain=domain)),
        y=alt.Y("outflow_m3_s:Q", title="Outlet discharge (m³/s)"),
        tooltip=[alt.Tooltip("time_h:Q", title="Time (h)", format=".2f"),
                 alt.Tooltip("outflow_m3_s:Q", title="Outlet flow (m³/s)", format=".2f")])
    peak = alt.Chart(pd.DataFrame({"time_h": [timing["peak_runoff_time_h"]],
                                   "outflow_m3_s": [timing["peak_runoff_m3_s"]]})).mark_point(
        filled=True, color="#D95F0E", size=70).encode(x="time_h:Q", y="outflow_m3_s:Q")
    st.altair_chart((runoff_chart + storm_end + peak).properties(height=250), width="stretch")
    if timing["apparent_lag_h"] is not None:
        st.caption(f"Peak rainfall interval: {timing['peak_rain_start_h']:.2f}–{timing['peak_rain_end_h']:.2f} h. "
                   f"Runoff peak: {timing['peak_runoff_time_h']:.2f} h. "
                   f"Apparent peak lag: {timing['apparent_lag_h']:.2f} h "
                   f"(interval-based range {timing['lag_min_h']:.2f}–{timing['lag_max_h']:.2f} h).")
    elif timing["peak_runoff_m3_s"] > 0.:
        st.caption(f"Runoff peak: {timing['peak_runoff_time_h']:.2f} h. Rainfall has flat or tied intensity peaks, so a unique peak lag is undefined.")
    else:
        st.caption("No modeled outlet runoff; peak lag is undefined.")
    st.caption("Both graphs use the same time axis. The dashed line marks the end of rainfall; runoff continues through the recession. Rainfall bars show interval averages.")
    if "terrain_result" not in simulated_data.attrs:
        st.caption("Polygon-mode runoff uses mapped polygon areas; overlapping polygons may inflate the aggregated flow.")
    st.download_button("Download rainfall intervals", rainfall.to_csv(index=False), "rainfall_intervals.csv", "text/csv")
    st.download_button("Download runoff hydrograph", hydrograph.to_csv(index=False), "runoff_hydrograph.csv", "text/csv")


def _render_model_evaluation(simulated_data, baseline_data, population_path, boundary_data, observed_path=""):
    st.subheader("Exposure sensitivity")
    sensitivity = population_exposure_sensitivity(simulated_data, population_path, boundary_data)
    if not sensitivity.empty:
        totals = sensitivity.groupby("threshold_m", as_index=False)[["people_affected", "people_modeled", "people_unmodeled"]].sum()
        st.dataframe(totals.round(2), hide_index=True, width="stretch")
        st.caption("0.05, 0.10, 0.30 and 0.50 m are illustrative exposure thresholds, not universal damage cutoffs. People outside model coverage are excluded from affected counts.")
        st.download_button("Download exposure sensitivity", sensitivity.to_csv(index=False), "exposure_sensitivity.csv", "text/csv")
    st.subheader("Comparison with the preserved baseline")
    comparison_baseline = baseline_data
    if "terrain_result" in simulated_data.attrs:
        terrain = simulated_data.attrs["terrain_result"]
        baseline_grid = _rasterize_simulated_depth(baseline_data, terrain.grid.crs, terrain.grid.shape, terrain.grid.transform)
        comparison_baseline = simulated_data.copy()
        comparison_baseline["sim_depth_m"] = baseline_grid[terrain.report_mask]
        comparison_baseline.attrs = {"terrain_result": replace(terrain, max_depth_m=baseline_grid)}
        st.caption("Both comparison rows use the same DEM-covered cells. The preserved baseline uses original hazard CN values and the original vectorized depth relationship.")
    rows = []
    for name, data in [("Corrected baseline", comparison_baseline), ("Selected model", simulated_data)]:
        exposure = population_impact_summary(data, population_path, boundary_data)
        rows.append({"model": name, "flooded_area_sqkm": flooded_area_sqkm(data),
                     "maximum_depth_m": float(data["sim_depth_m"].max()) if len(data) else 0.,
                     "affected_population": float(exposure["people_affected"].sum())})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    if "terrain_result" in simulated_data.attrs:
        terrain = simulated_data.attrs["terrain_result"]
        st.download_button("Download maximum flood depth GeoTIFF", depth_geotiff_bytes(terrain), "terrain_max_depth_m.tif", "image/tiff")
        with st.expander("Water balance and input coverage"):
            st.json(terrain.diagnostics)
        if observed_path:
            try:
                observed = read_aligned_raster(observed_path, terrain.grid, Resampling.nearest)
                rows = []
                for name, predictions in [("Corrected baseline", baseline_grid), ("Terrain model", terrain.max_depth_m)]:
                    metrics = validation_metrics(predictions, observed, valid_mask=terrain.report_mask)
                    rows.append({"model": name, **metrics})
                st.subheader("Independent observed-depth validation")
                st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
                st.caption("Validation uses only common valid observation/model cells. IoU and CSI are the same binary inundation score; undefined scores remain blank. Use observations from the same event.")
            except (ValueError, OSError, rasterio.errors.RasterioError) as exc:
                st.error(f"Unable to validate the observed depth raster: {exc}")
        else:
            st.caption("Independent flood observations have not been supplied; model accuracy has not been established.")
    else:
        metadata = simulated_data.attrs.get("runoff_inputs", {})
        if metadata:
            with st.expander("Runoff input coverage"):
                st.json(metadata)


def _downloaded_input_sources(inputs, data_dir, site_name):
    """Only report provenance when the selected file matches the download."""
    try:
        manifest = json.loads((data_dir / "data/hydrology/sources.json").read_text())
    except (OSError, ValueError):
        return {}
    selected = {}
    site_sources = manifest.get("sites", {}).get(site_name, {})
    for name, field in (("land_cover", "land_cover_path"), ("soil", "hsg_path")):
        source = site_sources.get(name, {})
        if not source and site_name == "Metro Manila":
            source = manifest.get("sources", {}).get(name, {})
        if source.get("local_path") and getattr(inputs, field):
            if Path(getattr(inputs, field)).resolve() == (data_dir / source["local_path"]).resolve():
                selected[name] = source
    return selected


def build_hazard_reference_map(selected_data, boundary_data=None):
    boundaries = _selected_boundary_data(selected_data, boundary_data)
    center, bounds, site_count = _map_extent(selected_data, boundaries)
    map_object = folium.Map(location=center, zoom_start=7 if site_count > 1 else 11,
                            tiles="CartoDB positron", control_scale=True)
    if bounds:
        map_object.fit_bounds(bounds)
    _add_boundary_layer(map_object, boundaries)
    fields = [name for name in ("site_name", "hazard_level", "source_return_period_yr", "Var", "geometry") if name in selected_data]
    reference = selected_data[fields].copy()
    folium.GeoJson(reference, name="UP NOAH hazard reference",
                   style_function=lambda f: {"fillColor": risk_to_hex(f["properties"]["Var"]),
                                             "color": "#666", "fillOpacity": .55, "weight": .3},
                   tooltip=folium.GeoJsonTooltip(fields=["site_name", "hazard_level"], aliases=["Site:", "Reference hazard:"])).add_to(map_object)
    map_object.get_root().html.add_child(folium.Element(_legend_html("UP NOAH risk classification")))
    folium.LayerControl().add_to(map_object)
    return map_object


@with_rasterio_runtime
def run_app() -> None:
    """One map, four main results, and optional model/population details."""
    st.set_page_config(page_title="Flood simulator", layout="wide")
    st.title("Flood simulator")
    st.caption("Metro Manila and selected Philippine provinces")
    data_dir = Path(__file__).resolve().parent
    try:
        hazard_data = load_data(str(data_dir))
        population_raster_path = find_population_raster(data_dir)
    except (ValueError, OSError, rasterio.errors.RasterioError) as exc:
        st.error(f"Unable to load flood or population data: {exc}")
        st.stop()
    try:
        boundary_data = load_boundary_data(str(data_dir))
    except (ValueError, OSError) as exc:
        boundary_data = None
        st.warning(f"Site boundaries could not be loaded: {exc}")
    if hazard_data.attrs.get("missing_sites"):
        st.warning("Missing coverage: " + ", ".join(hazard_data.attrs["missing_sites"]))
    if hazard_data.attrs.get("source_errors"):
        st.warning("Some hazard layers could not be read: " + " | ".join(hazard_data.attrs["source_errors"]))
    available = set(hazard_data["site_name"])
    site_options = [ALL_SITES_LABEL] + [name for name in SITE_MAPPING if name in available]
    site_options += sorted(available.difference(site_options))
    default_site = site_options.index("Metro Manila") if "Metro Manila" in site_options else 1
    st.sidebar.header("Rain scenario")
    selected_site = st.sidebar.selectbox("Site / province", site_options, index=default_site)
    selected_data = (hazard_data if selected_site == ALL_SITES_LABEL
                     else hazard_data[hazard_data["site_name"].eq(selected_site)].copy())
    rain_mm = st.sidebar.slider("Current rain event (mm)", 0, 500, 150, 10)
    duration_h = st.sidebar.slider("Event duration (hours)", .5, 48., 3., .5)
    prior_rain_mm = st.sidebar.slider("Prior 5-day rainfall (mm)", 0, 200, 20, 5)
    with st.sidebar.expander("Rainfall timing / CSV"):
        pattern = st.selectbox("Rainfall timing", ["Uniform", "Early peak", "Middle peak", "Late peak", "CSV time series"])
        if pattern == "CSV time series":
            uploaded_rain = st.file_uploader("Rainfall intervals CSV", type="csv", max_upload_size=5,
                                             help="Consecutive rain_mm and interval_h columns. CSV totals override sliders.")
            if uploaded_rain is None:
                st.info("Choose a rainfall CSV containing rain_mm and interval_h columns.")
                st.stop()
            try:
                event = RainfallEvent.from_frame(pd.read_csv(uploaded_rain))
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(f"Invalid rainfall time series: {exc}")
                st.stop()
            rain_mm, duration_h = event.total_mm, event.duration_h
            st.caption(f"CSV storm: {rain_mm:.1f} mm over {duration_h:.2f} hours")
        else:
            event = make_rainfall_event(rain_mm, duration_h, pattern)
    inputs = PhysicalInputs()
    dem_path = observed_path = ""
    routing_time_h, resolution_m, manning_n, recession_h = 1., 120., .06, 6.
    with st.sidebar.expander("Advanced model settings"):
        model_mode = st.selectbox("Simulation method", ["Corrected empirical baseline", "Soil and routed runoff (empirical depth)", "Terrain inundation (screening)"], index=1)
        if model_mode != "Corrected empirical baseline":
            try:
                if selected_site == ALL_SITES_LABEL:
                    st.info("Choose a site to use its matching soil and land-cover maps.")
                    st.stop()
                inputs, dem_default = _physical_inputs_controls(data_dir, selected_site)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                st.error(f"Unable to read runoff settings: {exc}")
                st.stop()
        if model_mode != "Terrain inundation (screening)":
            routing_time_h = st.slider("Runoff routing time (hours)", .1, 6., 1., .1)
        else:
            dem_path = st.text_input("DEM raster (elevation in metres)", dem_default).strip()
            resolution_m = st.select_slider("Terrain cell size (m)", [60., 90., 120., 180., 240.], value=120.)
            manning_n = st.number_input("Manning roughness (scenario assumption)", min_value=.01, max_value=.3, value=.06, step=.01)
            recession_h = st.slider("Post-rainfall recession (hours)", 0., 24., 6., .5)
            observed_path = st.text_input("Independent observed depth raster (optional, metres)").strip()
            if dem_path and not Path(dem_path).is_absolute():
                dem_path = str(data_dir / dem_path)
            if observed_path and not Path(observed_path).is_absolute():
                observed_path = str(data_dir / observed_path)
    try:
        if model_mode == "Terrain inundation (screening)":
            if selected_site == ALL_SITES_LABEL:
                st.info("Choose one coverage area for terrain inundation and a DEM covering its catchment.")
                st.stop()
            boundaries = _selected_boundary_data(selected_data, boundary_data)
            if boundaries.empty:
                st.error("Site boundaries are required to report terrain coverage.")
                st.stop()
            terrain = _terrain_event_cached(dem_path, resolution_m, inputs, _input_file_key(inputs, dem_path),
                                            event, prior_rain_mm, manning_n, recession_h)
            selected_simulated = terrain_to_polygons(terrain, boundaries, selected_data, selected_site)
            selected_simulated["flood_exposure_class"] = selected_simulated["sim_depth_m"].apply(depth_to_flood_class)
            selected_simulated["affected_population"] = 0.  # Raster exposure is authoritative.
        elif model_mode == "Soil and routed runoff (empirical depth)":
            runoff_data = _runoff_inputs_cached(_population_data_cache_key(selected_data), inputs,
                                                _input_file_key(inputs), resolution_m, selected_data)
            selected_simulated = simulate_flood(runoff_data, rain_mm, duration_h, prior_rain_mm,
                                                rainfall_event=event, routing_time_h=routing_time_h, use_routed_depth=True)
        else:
            selected_simulated = simulate_flood(selected_data, rain_mm, duration_h, prior_rain_mm,
                                                rainfall_event=event, routing_time_h=routing_time_h)
    except (ValueError, OSError, rasterio.errors.RasterioError) as exc:
        st.error(f"Unable to run the selected model: {exc}")
        st.stop()
    flood_summary = population_impact_summary(selected_simulated, population_raster_path, boundary_data)
    affected_population = float(flood_summary["people_affected"].sum())
    area_sqkm = flooded_area_sqkm(selected_simulated)
    maximum_depth = float(selected_simulated["sim_depth_m"].max())
    columns = st.columns(4)
    columns[0].metric("Flooded area", f"{area_sqkm:.2f} km²")
    columns[1].metric("Affected population", f"{affected_population:,.0f}")
    columns[2].metric("Maximum depth", f"{maximum_depth:.2f} m")
    columns[3].metric("Rainfall", f"{rain_mm:.0f} mm", help=f"{duration_h:g} hours; {rain_mm / duration_h:.1f} mm/h average")
    st.caption(f"Map and area: depth ≥{DISPLAY_DEPTH_THRESHOLD_M:.2f} m. Affected population: depth ≥{EXPOSURE_DEPTH_THRESHOLD_M:.2f} m. Storm duration: {duration_h:g} h.")
    metadata = (selected_simulated.attrs["terrain_result"].diagnostics["runoff_inputs"] if "terrain_result" in selected_simulated.attrs
                else selected_simulated.attrs.get("runoff_inputs", {}))
    if metadata.get("runoff_grid_resolution_m", 0.) > metadata.get("requested_runoff_grid_resolution_m", 0.) * 1.001:
        st.caption(f"Large-area runoff inputs summarized at {metadata['runoff_grid_resolution_m']:.0f} m (requested {metadata['requested_runoff_grid_resolution_m']:.0f} m) to keep the polygon analysis within its processing limit.")
    if metadata and (metadata.get("soil_assumed_pct", 0.) or metadata.get("land_cover_assumed_pct", 0.)):
        st.caption(f"Scenario assumptions: HSG {inputs.assumed_hsg}, {LAND_COVER_LABELS[inputs.assumed_land_cover].lower()}; soil missing in {metadata.get('soil_assumed_pct', 0.):.0f}% and cover missing in {metadata.get('land_cover_assumed_pct', 0.):.0f}% of model cells. Local data can be set under Advanced model settings.")
    outside = float(flood_summary.loc[flood_summary["flood_depth_band"].eq(UNMODELED_EXPOSURE_LABEL), "people_in_band"].sum())
    if outside > 0:
        st.warning(f"{outside:,.0f} people are outside DEM coverage. Their exposure is unknown.")
    source_metadata = _downloaded_input_sources(inputs, data_dir, selected_site) if model_mode != "Corrected empirical baseline" else {}
    source_labels = []
    if "land_cover" in source_metadata:
        source_labels.append("Land cover: ESA WorldCover 2021")
    if "soil" in source_metadata:
        source_labels.append("Soil: HiHydroSoil v2.0 global estimate")
    if source_labels:
        st.caption(" · ".join(source_labels))
    map_layer = st.selectbox("Map layer", ["Flood depth", "Affected population", "Population density", "UP NOAH reference"], key="map_layer", width=320)
    active = selected_simulated[selected_simulated["sim_depth_m"] >= DISPLAY_DEPTH_THRESHOLD_M].copy()
    if map_layer == "Population density":
        map_object = build_population_density_map(selected_data, population_raster_path, boundary_data)
        st.caption("Population density within the selected site boundaries.")
    elif map_layer == "Affected population":
        map_object = build_population_map(selected_data, population_raster_path,
                                          population_mode="Affected population by flood",
                                          boundary_data=boundary_data, simulated_data=selected_simulated)
        st.caption(f"Population dots at modeled depths of {EXPOSURE_DEPTH_THRESHOLD_M:.2f} m or more.")
    elif map_layer == "UP NOAH reference":
        map_object = build_hazard_reference_map(selected_data, boundary_data)
        st.caption("UP NOAH reference hazard classes; this layer is independent of the current storm.")
    elif "terrain_result" in selected_simulated.attrs:
        map_object = build_terrain_flood_map(selected_simulated, selected_data, "Continuous depth spectrum", boundary_data)
        st.caption("Maximum depth during the storm and recession. The blue outline marks DEM coverage.")
    else:
        map_object = build_flood_map(selected_simulated, active, "Continuous depth spectrum", boundary_data)
    st_folium(map_object, width="100%", height=620, returned_objects=[], key="scenario_map")
    if active.empty and map_layer == "Flood depth":
        st.caption("No modeled flooding reaches the display threshold for this scenario.")
    population_details = st.expander("Population and site details", key="population_details", on_change="rerun")
    if population_details.open:
        with population_details:
            risk_summary = population_risk_summary(selected_data, population_raster_path, boundary_data)
            st.dataframe(_site_summary(selected_simulated, risk_summary, flood_summary).round(2), hide_index=True, width="stretch")
            bands = flood_summary.groupby("flood_depth_band", sort=False)[["people_in_band", "people_affected"]].sum().reindex(FLOOD_EXPOSURE_ORDER, fill_value=0.)
            st.dataframe(bands.round(2), width="stretch")
            st.caption("Population is counted once per source raster cell. Outside model coverage means unknown, not dry.")
            risk = risk_summary.groupby("risk_class", sort=False)["people"].sum().reindex(POPULATION_RISK_ORDER, fill_value=0.)
            st.dataframe(risk.reset_index().round(2), hide_index=True, width="stretch")
    analysis = st.expander("Rainfall and runoff analysis", key="storm_analysis", on_change="rerun")
    if analysis.open:
        with analysis:
            _render_rainfall_runoff_analysis(selected_simulated, event)
    diagnostics = st.expander("Model comparison and validation", key="model_diagnostics", on_change="rerun")
    if diagnostics.open:
        with diagnostics:
            baseline = (selected_simulated if model_mode == "Corrected empirical baseline" else
                        simulate_flood(selected_data, rain_mm, duration_h, prior_rain_mm,
                                       rainfall_event=event, routing_time_h=routing_time_h))
            _render_model_evaluation(selected_simulated, baseline, population_raster_path, boundary_data, observed_path)
    with st.expander("Model assumptions"):
        st.write(f"Selected method: {model_mode}.")
        if model_mode == "Terrain inundation (screening)":
            st.write("Terrain routing uses local depression storage, open boundaries, and the available DEM. It does not yet model river stages, tides, backwater, sewer networks, or inflow from outside the DEM. Depths are screening estimates and require independent event observations for validation.")
        elif model_mode == "Soil and routed runoff (empirical depth)":
            st.write("Soil and land cover determine runoff CN. Rainfall timing changes runoff peaks. Inundation depth still scales from NOAH reference depths and remains an empirical estimate.")
        else:
            st.write("The preserved baseline uses the original hazard-based CN values and depth relationship. Its depth depends on total rainfall and duration; timing changes only its runoff hydrograph.")
        if source_metadata:
            for source in source_metadata.values():
                st.markdown(f"[{source['title']}]({source['documentation']}) — {source['native_resolution']}; {source['type']}.")
                st.caption(source["citation"])
        st.write("Missing input cells use the explicit scenario assumptions shown above. Curve Number moisture adjustment is applied once. Model complexity alone does not demonstrate improved accuracy.")


if __name__ == "__main__":
    run_app()
