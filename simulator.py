"""Regional flood simulator for Metro Manila and selected Philippine provinces.

The source hazard layers are the UP NOAH shapefiles stored in this directory as
ZIP archives. This module keeps data preparation and the hydrologic model in
plain functions so they can be reused from ``main.ipynb`` as well as the
Streamlit application. The bundled population raster is used for density and
flood-exposure layers.
"""

from pathlib import Path
from zipfile import ZipFile

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
from shapely.geometry import mapping
from shapely.ops import unary_union
from streamlit_folium import st_folium


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
# Values below 0.1 m are reported as 0 m/no flood, including the simulator's
# shallow display threshold between 0.05 m and 0.1 m.
FLOOD_EXPOSURE_ORDER = ["0 m (no flood)", "0.1–1 m", "1–2 m", ">2 m"]

# Single source of truth for the archives shown in the app. Centers are used
# for SoilGrids queries and map navigation.
SITE_MAPPING = {
    "Metro Manila": {
        "archive": "MetroManila.zip",
        "center": (14.6091, 121.0223),
        "return_period_yr": 25,
    },
    "Nueva Ecija": {
        "archive": "NuevaEcija.zip",
        "center": (15.50, 121.00),
        "return_period_yr": 25,
    },
    "Pampanga": {
        "archive": "Pampanga.zip",
        "center": (15.08, 120.62),
        "return_period_yr": 25,
    },
    "Bulacan": {
        "archive": "Bulacan.zip",
        "center": (14.90, 120.90),
        "return_period_yr": 25,
    },
    "Tarlac": {
        "archive": "Tarlac.zip",
        "center": (15.48, 120.60),
        "return_period_yr": 25,
    },
    "Aurora": {
        "archive": "Aurora.zip",
        "center": (15.75, 121.40),
        "return_period_yr": 25,
    },
    "Zambales": {
        "archive": "Zambales.zip",
        "center": (15.17, 120.30),
        "return_period_yr": 25,
    },
    "Bataan": {
        "archive": "Bataan.zip",
        "center": (14.75, 120.45),
        "return_period_yr": 25,
    },
    "Pangasinan": {
        "archive": "Pangasinan.zip",
        "center": (15.90, 120.40),
        "return_period_yr": 25,
    },
    "Nueva Vizcaya": {
        "archive": "NuevaVizcaya.zip",
        "center": (16.30, 121.20),
        "return_period_yr": 25,
    },
    "Rizal": {
        "archive": "Rizal.zip",
        "center": (14.60, 121.20),
        "return_period_yr": 25,
    },
    "Quezon": {
        "archive": "Quezon.zip",
        "center": (14.00, 121.60),
        "return_period_yr": 25,
    },
    "South Cotabato": {
        "archive": "SouthCotabato.zip",
        "center": (6.30, 125.00),
        "return_period_yr": 25,
    },
    "Sarangani": {
        "archive": "Sarangani.zip",
        "center": (6.20, 125.20),
        "return_period_yr": 25,
    },
    "Davao del Sur": {
        "archive": "DavaoDelSur.zip",
        "center": (6.70, 125.30),
        "return_period_yr": 25,
    },
    "Davao Occidental": {
        "archive": "DavaoOccidental.zip",
        "center": (6.00, 125.50),
        "return_period_yr": 100,
    },
}

HAZARD_LABELS = {
    1: "Low Hazard (0–0.5 m)",
    2: "Medium Hazard (>0.5–1.5 m)",
    3: "High Hazard (>1.5 m)",
}

BASE_DEPTH_BY_VAR = {1: 0.25, 2: 1.00, 3: 2.00}
CN_BY_VAR = {1: 78, 2: 88, 3: 98}


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
    layer["CN_base"] = layer["Var"].map(CN_BY_VAR)
    layer["source_return_period_yr"] = int(site_config["return_period_yr"])
    layer["source_file"] = site_config["archive"]

    return layer[
        [
            "site_name",
            "Var",
            "hazard_level",
            "base_depth_m",
            "CN_base",
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


def depth_to_flood_class(depth: float, threshold_m: float = 0.05) -> str:
    """Classify simulated flood depth using the population-impact bands."""

    depth = float(depth)
    population_threshold = max(0.1, float(threshold_m))
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
            depth_codes[(depths >= 0.1) & (depths < 1.0)] = 1
            depth_codes[(depths >= 1.0) & (depths <= 2.0)] = 2
            depth_codes[depths > 2.0] = 3
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
                        "people_affected": people if code > 0 else 0.0,
                    }
                )

    return pd.DataFrame(rows, columns=columns)


def _simulation_population_cache_key(
    simulated_data: gpd.GeoDataFrame,
) -> str:
    """Build a cache key that changes when simulated depths change."""

    depths = tuple(
        round(float(value), 4)
        for value in pd.to_numeric(
            simulated_data.get("sim_depth_m", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
    )
    return repr((tuple(simulated_data.get("site_name", [])), depths))


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

    source_files = tuple(hazard_data.get("source_file", pd.Series(dtype=str)).astype(str))
    area_total = float(
        pd.to_numeric(hazard_data.get("area_sqkm", pd.Series(dtype=float)), errors="coerce")
        .sum()
    )
    bounds = tuple(round(float(value), 6) for value in hazard_data.total_bounds)
    return repr((len(hazard_data), source_files, round(area_total, 6), bounds))


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
    """Fetch clay/sand percentages and a simple hydrologic soil group."""

    try:
        url = (
            "https://rest.isric.org/soilgrids/v2.0/properties/query"
            f"?lon={lon}&lat={lat}&property=clay&property=sand"
            "&depth=0-5cm&value=mean"
        )
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        layers = response.json()["properties"]["layers"]

        values = {
            layer["name"]: layer["depths"][0]["values"]["mean"] / 10.0
            for layer in layers
            if layer.get("name") in {"clay", "sand"}
        }
        clay_pct = float(values.get("clay", 0.0))
        sand_pct = float(values.get("sand", 0.0))

        if clay_pct > 40:
            hsg = "D (Very Slow / Clay)"
        elif clay_pct > 27:
            hsg = "C (Slow / Clay Loam)"
        elif sand_pct > 50:
            hsg = "A (High Infiltration / Sandy)"
        else:
            hsg = "B (Moderate / Loam)"

        return round(clay_pct, 1), round(sand_pct, 1), hsg
    except Exception:
        # A stable fallback keeps the simulator usable when SoilGrids is down.
        return 48.0, 18.0, "D (Very Slow / Default profile)"


def adjust_cn_amc(cn_ii: float, prior_rain_mm: float) -> float:
    """Adjust Curve Number II for antecedent moisture conditions."""

    cn_ii = float(cn_ii)
    prior_rain_mm = float(prior_rain_mm)
    if prior_rain_mm < 35.0:
        return (4.2 * cn_ii) / (10.0 - (0.058 * cn_ii))
    if prior_rain_mm > 52.5:
        return (23.0 * cn_ii) / (10.0 + (0.13 * cn_ii))
    return cn_ii


def compute_polygon_depth(
    row: pd.Series,
    rain_mm: float,
    duration_h: float,
    prior_rain_mm: float = 20.0,
    benchmark_rain_mm: float = 180.0,
) -> float:
    """Calculate simulated depth for one hazard polygon."""

    cn_value = row["CN_amc"] if "CN_amc" in row else row["CN_base"]
    cn = adjust_cn_amc(cn_value, prior_rain_mm)
    base_depth = float(row["base_depth_m"])
    retention_s = (25400.0 / cn) - 254.0
    initial_abstraction = 0.2 * retention_s

    def runoff(rainfall: float) -> float:
        if rainfall <= initial_abstraction:
            return 0.0
        return ((rainfall - initial_abstraction) ** 2) / (
            rainfall + 0.8 * retention_s
        )

    actual_runoff = runoff(float(rain_mm))
    benchmark_runoff = runoff(float(benchmark_rain_mm))
    time_factor = 1.0 - np.exp(-1.5 * (float(duration_h) / 2.0))
    ratio = 0.0 if benchmark_runoff <= 0 else actual_runoff / benchmark_runoff
    return round(max(0.0, base_depth * ratio * time_factor), 2)


def simulate_flood(
    hazard_data: gpd.GeoDataFrame,
    rain_mm: float,
    duration_h: float,
    prior_rain_mm: float,
    threshold_m: float = 0.05,
    benchmark_rain_mm: float = 180.0,
    population_raster_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Apply the AMC-adjusted SCS-CN model to every configured site polygon.

    If a population raster is supplied and the input has not already been
    enriched, each result also includes the estimated population in the
    polygon and the number of people affected by the simulated flood.
    """

    rain_mm = float(rain_mm)
    duration_h = float(duration_h)
    prior_rain_mm = float(prior_rain_mm)
    if rain_mm < 0:
        raise ValueError("rain_mm must be non-negative")
    if duration_h <= 0:
        raise ValueError("duration_h must be greater than zero")

    simulated = hazard_data.copy()
    if population_raster_path is not None and "population_estimate" not in simulated:
        simulated = add_population_exposure(simulated, population_raster_path)
    simulated["CN_amc"] = simulated["CN_base"].apply(
        lambda cn: adjust_cn_amc(cn, prior_rain_mm)
    )

    cn = simulated["CN_amc"].astype(float).to_numpy()
    base_depth = simulated["base_depth_m"].astype(float).to_numpy()
    retention_s = (25400.0 / cn) - 254.0
    initial_abstraction = 0.2 * retention_s

    actual_runoff = np.where(
        rain_mm <= initial_abstraction,
        0.0,
        ((rain_mm - initial_abstraction) ** 2)
        / (rain_mm + 0.8 * retention_s),
    )
    benchmark_rain_mm = float(benchmark_rain_mm)
    benchmark_runoff = np.where(
        benchmark_rain_mm <= initial_abstraction,
        0.0,
        ((benchmark_rain_mm - initial_abstraction) ** 2)
        / (benchmark_rain_mm + 0.8 * retention_s),
    )
    time_factor = 1.0 - np.exp(-1.5 * (duration_h / 2.0))
    scaling_ratio = np.divide(
        actual_runoff * time_factor,
        benchmark_runoff,
        out=np.zeros_like(actual_runoff, dtype=float),
        where=benchmark_runoff > 0,
    )

    simulated["sim_depth_m"] = np.round(
        np.maximum(0.0, base_depth * scaling_ratio), 2
    )
    simulated["flood_status"] = np.where(
        simulated["sim_depth_m"] >= float(threshold_m), "Flooded", "Below threshold"
    )
    simulated["flood_exposure_class"] = [
        depth_to_flood_class(depth, threshold_m=threshold_m)
        for depth in simulated["sim_depth_m"]
    ]

    if "population_estimate" in simulated.columns:
        population = pd.to_numeric(
            simulated["population_estimate"], errors="coerce"
        ).fillna(0.0)
    else:
        # Keep the simulator usable when no population raster is available.
        population = pd.Series(0.0, index=simulated.index)
    simulated["affected_population"] = np.where(
        simulated["sim_depth_m"] >= 0.1, population, 0.0
    )
    return simulated


def flooded_area_sqkm(simulated_data: gpd.GeoDataFrame, threshold_m: float = 0.05) -> float:
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
def _population_dot_data(
    bounds: tuple[float, float, float, float],
    population_raster_path: str,
    max_dots: int = DEFAULT_POPULATION_DOT_COUNT,
) -> pd.DataFrame:
    """Aggregate population cells into a manageable set of dot locations."""

    with rasterio.open(population_raster_path) as source:
        requested_window = from_bounds(*bounds, transform=source.transform)
        try:
            window = requested_window.intersection(
                Window(0, 0, source.width, source.height)
            )
        except ValueError as exc:
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
        source_cell_area_sqkm = abs(source.transform.a * source.transform.e) * (
            111.32**2
        ) * np.cos(
            np.deg2rad(
                output_transform.f
                + ((np.arange(output_height, dtype=float) + 0.5) * output_transform.e)
            )
        )
        density = np.divide(
            population_average,
            source_cell_area_sqkm[:, None],
            out=np.full(population_average.shape, np.nan, dtype=float),
            where=source_cell_area_sqkm[:, None] > 0.0,
        )

        rows, cols = np.indices((output_height, output_width), dtype=float)
        longitudes = output_transform.c + (cols + 0.5) * output_transform.a
        latitudes = output_transform.f + (rows + 0.5) * output_transform.e
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

    metric_boundaries = selected_boundaries.to_crs(AREA_CRS)
    return tuple(
        (
            str(row.site_name),
            tuple(round(float(value), 2) for value in row.geometry.bounds),
            round(float(row.geometry.area), 2),
        )
        for row in metric_boundaries.itertuples()
    )


@st.cache_data(show_spinner=False)
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
        legend_body = """
            <b>Affected population by flood depth</b><br>
            <div style="background:linear-gradient(to right,#ffffd9,#41b6c4,#225ea8,#081d58);
                        height:16px; width:100%; border-radius:3px; margin:6px 0;"></div>
            <div style="display:flex; justify-content:space-between;">
                <span>0.1 m</span><span>1 m</span><span>2 m+</span>
            </div>
            <div style="margin-top:6px;">Only dots at simulated depths of 0.1 m or more are shown.</div>
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
    threshold_m: float = 0.1,
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
    flood_polygons = simulated_data[
        simulated_data["sim_depth_m"] >= float(threshold_m)
    ][["sim_depth_m", "geometry"]].copy()
    if flood_polygons.empty:
        return population_dots.iloc[0:0].copy()
    if flood_polygons.crs != points.crs:
        flood_polygons = flood_polygons.to_crs(points.crs)

    joined = gpd.sjoin(points, flood_polygons, how="left", predicate="within")
    max_depth_by_point = joined.groupby(level=0)["sim_depth_m"].max()
    points["sim_depth_m"] = (
        points.index.to_series().map(max_depth_by_point).fillna(0.0).to_numpy()
    )
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
    threshold_m: float = 0.1,
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
        legend_body = """
            <b>Simulated water depth</b><br>
            <div style="background:linear-gradient(to right,#E0F3F8,#ABD9E9,#74ADD1,#4575B4,#313695,#081D58);
                        height:16px; width:100%; border-radius:3px; margin:6px 0;"></div>
            <div style="display:flex; justify-content:space-between;">
                <span>0.1 m</span><span>1.0 m</span><span>2.5 m+</span>
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
        simulated_data[simulated_data["sim_depth_m"] >= 0.05]
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


def run_app() -> None:
    """Render the regional Streamlit application."""

    st.set_page_config(layout="wide")
    st.title("Regional Flood Simulator — Metro Manila and Philippine Provinces")
    st.caption(
        "UP NOAH hazard polygons are shown inside the configured administrative boundaries, "
        "with population density and affected-population layers. "
        "Select a site or view all covered sites."
    )

    data_dir = Path(__file__).resolve().parent
    try:
        hazard_data = load_data(str(data_dir))
        population_raster_path = find_population_raster(data_dir)
        hazard_data = load_population_exposure(
            hazard_data, str(population_raster_path)
        )
    except Exception as exc:
        st.error(f"Unable to load the flood or population layers: {exc}")
        st.stop()

    try:
        boundary_data = load_boundary_data(str(data_dir))
    except Exception as exc:
        boundary_data = None
        boundary_error = str(exc)
    else:
        boundary_error = ""

    missing_sites = hazard_data.attrs.get("missing_sites", [])
    source_errors = hazard_data.attrs.get("source_errors", [])
    if missing_sites:
        st.warning("Missing source archive(s): " + ", ".join(missing_sites))
    if source_errors:
        st.warning("Some source layers could not be read: " + " | ".join(source_errors))
    if boundary_error:
        st.warning("Site boundary layer could not be loaded: " + boundary_error)

    available_site_set = set(hazard_data["site_name"])
    available_sites = [name for name in SITE_MAPPING if name in available_site_set]
    site_options = [ALL_SITES_LABEL] + available_sites

    st.sidebar.header("1. Coverage area")
    selected_site = st.sidebar.selectbox("Site / province", site_options)
    if selected_site == ALL_SITES_LABEL:
        selected_data = hazard_data
        soil_site = "Metro Manila" if "Metro Manila" in available_sites else available_sites[0]
    else:
        selected_data = hazard_data[hazard_data["site_name"] == selected_site].copy()
        soil_site = selected_site

    st.sidebar.header("2. Rain event parameters")
    rain_mm = st.sidebar.slider(
        "Current rain event (mm)", min_value=10, max_value=500, value=150, step=10
    )
    duration_h = st.sidebar.slider(
        "Event duration (hours)", min_value=0.5, max_value=12.0, value=3.0, step=0.5
    )

    st.sidebar.header("3. Antecedent moisture")
    prior_rain_mm = st.sidebar.slider(
        "Prior 5-day rainfall (mm)", min_value=0, max_value=200, value=20, step=5
    )

    st.sidebar.header("4. Display mode")
    map_mode = st.sidebar.radio(
        "Color polygons by", ["Continuous depth spectrum", "UP NOAH risk classification"]
    )

    simulated = simulate_flood(
        hazard_data,
        rain_mm=rain_mm,
        duration_h=duration_h,
        prior_rain_mm=prior_rain_mm,
    )
    selected_simulated = simulated[
        simulated["site_name"].isin(selected_data["site_name"].unique())
    ].copy()
    active = selected_simulated[selected_simulated["sim_depth_m"] >= 0.05].copy()
    risk_summary = population_risk_summary(
        selected_simulated,
        population_raster_path,
        boundary_data,
    )
    flood_population_summary = population_impact_summary(
        selected_simulated,
        population_raster_path=population_raster_path,
        boundary_data=boundary_data,
    )

    soil_lat, soil_lon = SITE_MAPPING[soil_site]["center"]
    clay_pct, sand_pct, hsg_class = fetch_soil_properties(soil_lat, soil_lon)
    intensity = rain_mm / duration_h
    area_sqkm = flooded_area_sqkm(selected_simulated)
    mapped_population = risk_summary["people"].sum() if not risk_summary.empty else 0.0
    affected_population = (
        flood_population_summary["people_affected"].sum()
        if not flood_population_summary.empty
        else 0.0
    )

    if prior_rain_mm < 35.0:
        amc_status = "AMC I (Dry ground — higher infiltration)"
    elif prior_rain_mm > 52.5:
        amc_status = "AMC III (Saturated ground — lower infiltration)"
    else:
        amc_status = "AMC II (Normal ground)"

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Rainfall intensity", f"{intensity:.1f} mm/hr")
    col2.metric("ISRIC soil profile", f"Clay {clay_pct}% / Sand {sand_pct}%")
    col3.metric("Hydrologic soil group", hsg_class)
    col4.metric("Flooded area", f"{area_sqkm:.2f} km²")
    col5.metric("Population in boundaries", f"{mapped_population:,.0f}")
    col6.metric("Affected by simulated flood", f"{affected_population:,.0f}")
    st.info(
        f"{amc_status}. Prior 5-day accumulation: {prior_rain_mm} mm. "
        f"Mapped polygons: {len(selected_simulated):,}."
    )

    st.subheader("Flooded area and population by site")
    summary = _site_summary(
        selected_simulated,
        risk_summary=risk_summary,
        flood_summary=flood_population_summary,
    )
    st.dataframe(
        summary.rename(
            columns={
                "site_name": "Site",
                "polygons": "Polygons",
                "mapped_area_sqkm": "Mapped area (km²)",
                "flooded_area_sqkm": "Flooded area (km²)",
                "max_depth_m": "Maximum depth (m)",
                "mapped_population": "Population in boundaries",
                "affected_population": "Affected by flood",
            }
        ).round(2),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Population by UP NOAH hazard risk")
    risk_by_class = (
        risk_summary.groupby("risk_class", sort=False)["people"]
        .sum()
        .reindex(POPULATION_RISK_ORDER, fill_value=0.0)
        .rename("people")
        .reset_index()
    )
    risk_total = float(risk_by_class["people"].sum())
    risk_by_class["share_pct"] = np.where(
        risk_total > 0.0,
        risk_by_class["people"] * 100.0 / risk_total,
        0.0,
    )
    st.caption(
        "No risk means the population cell is inside the site boundary but outside "
        "all UP NOAH hazard polygons. Low, moderate, and high risk come from the "
        "UP NOAH hazard classification at that cell."
    )
    st.dataframe(
        risk_by_class.rename(
            columns={
                "risk_class": "UP NOAH population risk",
                "people": "People",
                "share_pct": "Share of population (%)",
            }
        ).round(2),
        width="stretch",
        hide_index=True,
    )

    st.subheader("People by simulated flood depth")
    flood_by_band = (
        flood_population_summary.groupby("flood_depth_band", sort=False)
        .agg(
            people_in_band=("people_in_band", "sum"),
            people_affected=("people_affected", "sum"),
        )
        .reindex(FLOOD_EXPOSURE_ORDER, fill_value=0.0)
        .rename_axis("flood_depth_band")
        .reset_index()
    )
    st.caption(
        "Population is counted once per raster cell. The 0 m band includes cells "
        "with simulated depth below 0.1 m; affected population starts at 0.1 m."
    )
    st.dataframe(
        flood_by_band.rename(
            columns={
                "flood_depth_band": "Simulated flood depth",
                "people_in_band": "People in band",
                "people_affected": "People affected",
            }
        ).round(2),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Separate map views")
    hazard_tab, density_tab, affected_tab = st.tabs(
        ["Flood hazards", "Population density heatmap", "Affected population dots"]
    )

    with hazard_tab:
        if active.empty:
            st.warning("No polygons meet the 0.05 m display threshold for this event.")
        hazard_map = build_flood_map(
            selected_simulated,
            active,
            map_mode,
            boundary_data=boundary_data,
        )
        st_folium(
            hazard_map,
            width="100%",
            height=650,
            returned_objects=[],
            key="hazard_interactive_map",
        )

    with density_tab:
        st.caption(
            "Population density is shown as a separate logarithmic yellow-to-red heatmap "
            "on a dark basemap. All positive population estimates within the hazard-site "
            "boundaries are treated as vulnerable population."
        )
        density_map = build_population_density_map(
            selected_data,
            population_raster_path,
            boundary_data=boundary_data,
        )
        st_folium(
            density_map,
            width="100%",
            height=650,
            returned_objects=[],
            key="population_density_map",
        )

    with affected_tab:
        st.caption(
            "Only population dots located inside simulated flooded polygons are shown. "
            "Dot color follows the flood-depth spectrum; affected dots start at 0.1 m."
        )
        affected_map = build_population_map(
            selected_data,
            population_raster_path,
            population_mode="Affected population by flood",
            boundary_data=boundary_data,
            simulated_data=selected_simulated,
        )
        st_folium(
            affected_map,
            width="100%",
            height=650,
            returned_objects=[],
            key="affected_population_map",
        )


if __name__ == "__main__":
    run_app()
