"""Terrain-informed runoff screening with explicit storage and a water balance.

This is a D8 routing and local depression-storage model, not a hydraulic solver.
It has open edges and no river stage, tides, backwater, or engineered sewer network.
"""

from dataclasses import dataclass
import heapq
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject
from shapely import box
from shapely.geometry import mapping

from geo_runtime import with_rasterio_runtime

from flood_model import (CN_BY_LAND_COVER, RainfallEvent, adjust_cn_amc,
                         curve_numbers, hsg_from_properties, incremental_runoff_mm,
                         normalize_hsg)

MODEL_CRS = "EPSG:32651"


@dataclass(frozen=True)
class RasterGrid:
    shape: tuple
    transform: object
    crs: object = MODEL_CRS

    @property
    def cell_area_m2(self):
        return abs(self.transform.a * self.transform.e - self.transform.b * self.transform.d)


@dataclass(frozen=True)
class PhysicalInputs:
    land_cover_path: str = ""
    hsg_path: str = ""
    ksat_0_50_path: str = ""
    ksat_0_100_path: str = ""
    restrictive_depth_path: str = ""
    water_table_depth_path: str = ""
    impervious_fraction_path: str = ""
    drainage_capacity_path: str = ""
    assumed_hsg: str = "D"
    assumed_land_cover: int = 50
    built_up_fraction: float = 0.65
    assumed_drainage_mm_h: float = 0.0
    dual_hsg_drained: bool = False


def grid_for_geometries(geometries, resolution_m=120.0, max_cells=150_000):
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolution_m must be finite and greater than zero")
    bounds = geometries.to_crs(MODEL_CRS).total_bounds
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Study area has no usable bounds")
    left, bottom, right, top = bounds
    width = max(1, int(np.ceil((right - left) / resolution_m)))
    height = max(1, int(np.ceil((top - bottom) / resolution_m)))
    if width * height > max_cells:
        raise ValueError("Study grid exceeds 150,000 cells; choose a smaller site or coarser resolution")
    return RasterGrid((height, width), from_origin(left, top, resolution_m, resolution_m))


def runoff_grid_for_geometries(geometries, resolution_m=120.0, max_cells=150_000):
    """Build a bounded grid for polygon-level CN summaries.

    Large provincial areas need a coarser composite grid than local terrain
    routing. Increase the requested resolution only as much as needed to keep
    the raster analysis within its memory budget, and return that resolution.
    """
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolution_m must be finite and greater than zero")
    if int(max_cells) < 1:
        raise ValueError("max_cells must be a positive integer")
    projected = geometries.to_crs(MODEL_CRS)
    bounds = projected.total_bounds
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Study area has no usable bounds")
    left, bottom, right, top = bounds
    if right <= left or top <= bottom:
        raise ValueError("Study area bounds must have positive width and height")
    effective_resolution = float(resolution_m)
    while True:
        width = max(1, int(np.ceil((right - left) / effective_resolution)))
        height = max(1, int(np.ceil((top - bottom) / effective_resolution)))
        if width * height <= int(max_cells):
            break
        effective_resolution *= max(1.01, math.sqrt((width * height) / int(max_cells)) * 1.001)
    grid = RasterGrid((height, width), from_origin(left, top, effective_resolution, effective_resolution))
    return grid, effective_resolution


@with_rasterio_runtime
def read_aligned_raster(raster_path, grid, resampling=Resampling.nearest):
    """Respect CRS and nodata; cells outside the source remain NaN."""
    with rasterio.open(raster_path) as source:
        if source.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        if source.count != 1:
            raise ValueError(f"Expected a single-band raster: {raster_path}")
        aligned = np.full(grid.shape, np.nan, dtype=np.float64)
        reproject(rasterio.band(source, 1), aligned, src_transform=source.transform, src_crs=source.crs,
                  src_nodata=source.nodata, dst_transform=grid.transform, dst_crs=grid.crs,
                  dst_nodata=np.nan, resampling=resampling)
    return aligned


@with_rasterio_runtime
def load_dem_grid(dem_path, resolution_m=120.0, max_cells=150_000):
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolution_m must be finite and greater than zero")
    with rasterio.open(dem_path) as source:
        if source.crs is None or source.count != 1:
            raise ValueError("DEM must have a CRS and one elevation band in metres")
        transform, width, height = calculate_default_transform(
            source.crs, MODEL_CRS, source.width, source.height, *source.bounds,
            resolution=float(resolution_m))
    if width * height > max_cells:
        raise ValueError("DEM exceeds 150,000 model cells; clip it or choose a coarser resolution")
    grid = RasterGrid((height, width), transform)
    dem = read_aligned_raster(dem_path, grid, Resampling.average)
    if not np.isfinite(dem).any():
        raise ValueError("DEM has no valid elevations")
    return grid, dem


def _read_optional(raster_path, grid, method):
    if not raster_path:
        return np.full(grid.shape, np.nan)
    if not Path(raster_path).is_file():
        raise FileNotFoundError(f"Configured raster does not exist: {raster_path}")
    return read_aligned_raster(raster_path, grid, method)


def load_physical_grid(grid, inputs, valid_mask=None):
    """Combine real aligned rasters with explicit, counted scenario assumptions."""
    valid = np.ones(grid.shape, bool) if valid_mask is None else np.asarray(valid_mask, bool)
    if valid.shape != grid.shape or not valid.any():
        raise ValueError("Physical input mask is empty or has the wrong shape")
    cover = _read_optional(inputs.land_cover_path, grid, Resampling.mode)
    known_cover = np.isfinite(cover)
    if np.any(known_cover & ~np.isin(cover, list(CN_BY_LAND_COVER))):
        raise ValueError("Land-cover raster must use ESA WorldCover class codes; encode missing values as nodata")
    cover[~known_cover] = int(inputs.assumed_land_cover)
    group = _read_optional(inputs.hsg_path, grid, Resampling.mode)
    known_group = np.isfinite(group)
    if np.any(known_group & ~np.isin(group, [1, 2, 3, 4, 5, 6, 7])):
        raise ValueError("HSG codes must be 1..4 (A..D), or 5..7 (A/D, B/D, C/D)")
    dual = known_group & (group >= 5)
    group[dual] = group[dual] - 4 if inputs.dual_hsg_drained else 4
    # Direct HSG observations take precedence. Both Ksat profile minima are
    # supported, but only the appropriate profile is required per cell.
    properties = [_read_optional(p, grid, Resampling.min) for p in (
        inputs.ksat_0_50_path, inputs.ksat_0_100_path,
        inputs.restrictive_depth_path, inputs.water_table_depth_path)]
    inferred = hsg_from_properties(*properties)
    from_properties = ~known_group & np.isfinite(inferred)
    group[from_properties] = inferred[from_properties]
    known_soil = known_group | from_properties
    assumed_group = "ABCD".index(normalize_hsg(inputs.assumed_hsg)) + 1
    group[~known_soil] = assumed_group
    impervious = _read_optional(inputs.impervious_fraction_path, grid, Resampling.average)
    known_impervious = np.isfinite(impervious)
    default_fraction = np.where(cover == 50, inputs.built_up_fraction, 0.)
    impervious[~known_impervious] = default_fraction[~known_impervious]
    cn = curve_numbers(cover, group, impervious, inputs.built_up_fraction)
    drainage = _read_optional(inputs.drainage_capacity_path, grid, Resampling.average)
    known_drainage = np.isfinite(drainage)
    drainage[~known_drainage] = inputs.assumed_drainage_mm_h
    if np.any(~np.isfinite(drainage)) or np.any(drainage < 0):
        raise ValueError("Drainage capacity must be finite and non-negative in mm/h")
    count = int(valid.sum())
    metadata = {
        "model_cells": count,
        "land_cover_supplied_cells": int((known_cover & valid).sum()),
        "soil_supplied_cells": int((known_soil & valid).sum()),
        "impervious_supplied_cells": int((known_impervious & valid).sum()),
        "drainage_supplied_cells": int((known_drainage & valid).sum()),
        "land_cover_assumed_pct": float((~known_cover & valid).sum() * 100. / count),
        "soil_assumed_pct": float((~known_soil & valid).sum() * 100. / count),
        "impervious_assumed_pct": float((~known_impervious & valid).sum() * 100. / count),
        "assumed_hsg": inputs.assumed_hsg,
        "assumed_land_cover": inputs.assumed_land_cover,
        "built_up_impervious_fraction": inputs.built_up_fraction,
        "assumed_drainage_mm_h": inputs.assumed_drainage_mm_h,
        "source_paths": {name: str(getattr(inputs, name)) for name in inputs.__dataclass_fields__ if name.endswith("_path")},
    }
    return cn, drainage, metadata


def enrich_runoff_inputs(polygons, inputs, resolution_m=120.0):
    """Area-weighted composite CN per polygon, never derived from NOAH hazard."""
    enriched = polygons.copy()
    enriched["CN_legacy"] = enriched.get("CN_legacy", enriched["CN_base"])
    enriched = enriched.drop(columns=["CN_amc"], errors="ignore")
    has_rasters = any(getattr(inputs, name) for name in inputs.__dataclass_fields__ if name.endswith("_path"))
    if not has_rasters:
        group = "ABCD".index(normalize_hsg(inputs.assumed_hsg)) + 1
        cn = float(curve_numbers(inputs.assumed_land_cover, group, built_up_fraction=inputs.built_up_fraction))
        enriched["CN_base"] = cn
        enriched["CN_source"] = "Explicit assumed soil / land-cover profile"
        enriched.attrs["runoff_inputs"] = {"soil_assumed_pct": 100., "land_cover_assumed_pct": 100.,
                                           "assumed_hsg": inputs.assumed_hsg,
                                           "assumed_land_cover": inputs.assumed_land_cover}
        return enriched
    grid, effective_resolution = runoff_grid_for_geometries(polygons.geometry, resolution_m)
    geometries = polygons.to_crs(grid.crs).geometry
    valid_geometries = [mapping(g) for g in geometries if g is not None and not g.is_empty]
    if not valid_geometries:
        raise ValueError("Study area has no usable polygon geometries")
    if len(valid_geometries) != len(geometries):
        raise ValueError("Study area contains an empty polygon geometry")
    union = rasterize([(geometry, 1) for geometry in valid_geometries], out_shape=grid.shape, transform=grid.transform,
                      fill=0, all_touched=False, dtype="uint8").astype(bool)
    if not union.any():
        raise ValueError("Study polygons do not cover any cells at the runoff input resolution")
    cn, _, metadata = load_physical_grid(grid, inputs, union)
    means = []
    for geom in geometries:
        minx, miny, maxx, maxy = geom.bounds
        cell_x, cell_y = abs(grid.transform.a), abs(grid.transform.e)
        col0 = max(0, int(np.floor((minx - grid.transform.c) / cell_x)))
        col1 = min(grid.shape[1], int(np.ceil((maxx - grid.transform.c) / cell_x)))
        row0 = max(0, int(np.floor((grid.transform.f - maxy) / cell_y)))
        row1 = min(grid.shape[0], int(np.ceil((grid.transform.f - miny) / cell_y)))
        if col1 <= col0 or row1 <= row0:
            mask = np.zeros((0, 0), dtype=bool)
        else:
            local_transform = grid.transform * Affine.translation(col0, row0)
            mask = geometry_mask([mapping(geom)], out_shape=(row1 - row0, col1 - col0),
                                 transform=local_transform, invert=True, all_touched=False)
        if not mask.any():
            # A sub-cell polygon gets its representative point; this prevents
            # disappearing polygons but does not imply sub-grid data precision.
            point = geom.representative_point()
            col, row = ~grid.transform * (point.x, point.y)
            row = min(max(int(row), 0), grid.shape[0] - 1)
            col = min(max(int(col), 0), grid.shape[1] - 1)
            means.append(float(cn[row, col]))
        else:
            means.append(float(cn[row0:row1, col0:col1][mask].mean()))
    enriched["CN_base"] = means
    enriched["CN_source"] = "Soil / land-cover composite; see coverage diagnostics"
    metadata["runoff_grid_resolution_m"] = float(effective_resolution)
    metadata["requested_runoff_grid_resolution_m"] = float(resolution_m)
    metadata["runoff_grid_shape"] = list(grid.shape)
    metadata["runoff_grid_total_cells"] = int(grid.shape[0] * grid.shape[1])
    enriched.attrs["runoff_inputs"] = metadata
    return enriched


@dataclass
class FlowNetwork:
    valid: np.ndarray
    filled_dem_m: np.ndarray
    receiver: np.ndarray
    link_length_m: np.ndarray
    slope: np.ndarray
    accumulation_cells: np.ndarray
    depression_capacity_m: np.ndarray


def build_flow_network(dem_m, grid):
    """Priority-flood conditioning and deterministic acyclic D8 connectivity.

    Filled elevations only establish spill connectivity and local depression
    capacity. Original elevations remain the depth datum. Raster/nodata edges
    are open outlets; flat ties use priority-flood order to prevent cycles.
    """
    dem = np.asarray(dem_m, float)
    if dem.shape != grid.shape:
        raise ValueError("DEM must match the model grid")
    valid = np.isfinite(dem)
    if not valid.any():
        raise ValueError("DEM has no valid elevations")
    h, w = dem.shape
    neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    padded = np.pad(valid, 1)
    interior = valid.copy()
    for dr, dc in neighbors:
        interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    edge = valid & ~interior
    filled = dem.copy()
    visited = ~valid.copy()
    parent = np.full(dem.shape, -1, dtype=np.int64)
    rank = np.full(dem.shape, -1, dtype=np.int64)
    heap = []
    for row, col in np.argwhere(edge):
        index = int(row * w + col)
        heapq.heappush(heap, (dem[row, col], index))
        visited[row, col] = True
    order = []
    while heap:
        elevation, index = heapq.heappop(heap)
        row, col = divmod(index, w)
        rank[row, col] = len(order)
        order.append(index)
        for dr, dc in neighbors:
            rr, cc = row + dr, col + dc
            if 0 <= rr < h and 0 <= cc < w and not visited[rr, cc]:
                visited[rr, cc] = True
                filled[rr, cc] = max(elevation, dem[rr, cc])
                parent[rr, cc] = index
                heapq.heappush(heap, (filled[rr, cc], int(rr * w + cc)))
    receiver = np.full(dem.shape, -1, dtype=np.int64)
    length = np.full(dem.shape, abs(grid.transform.a))
    slope = np.zeros(dem.shape)
    best_score = np.full(dem.shape, -np.inf)
    cell_x, cell_y = abs(grid.transform.a), abs(grid.transform.e)
    indices = np.arange(h * w).reshape(dem.shape)
    for dr, dc in neighbors:
        rows = slice(max(0, -dr), min(h, h - dr))
        cols = slice(max(0, -dc), min(w, w - dc))
        dst_rows = slice(max(0, dr), min(h, h + dr))
        dst_cols = slice(max(0, dc), min(w, w + dc))
        distance = np.hypot(dr * cell_y, dc * cell_x)
        drop = filled[rows, cols] - filled[dst_rows, dst_cols]
        allowed = (interior[rows, cols] & valid[dst_rows, dst_cols]
                   & (rank[dst_rows, dst_cols] < rank[rows, cols]))
        score = drop / distance
        better = allowed & (score >= 0) & (score > best_score[rows, cols])
        receiver[rows, cols][better] = indices[dst_rows, dst_cols][better]
        length[rows, cols][better] = distance
        slope[rows, cols][better] = score[better]
        best_score[rows, cols][better] = score[better]
    # Every interior cell has an earlier parent even on completely flat DEMs.
    unresolved = interior & (receiver < 0)
    receiver[unresolved] = parent[unresolved]
    accumulation = valid.astype(float).ravel()
    for index in reversed(order):
        dest = receiver.ravel()[index]
        if dest >= 0:
            accumulation[dest] += accumulation[index]
    capacity = np.where(valid, np.maximum(filled - dem, 0.), 0.)
    return FlowNetwork(valid, filled, receiver, length, slope,
                       accumulation.reshape(dem.shape), capacity)


@dataclass
class TerrainResult:
    grid: RasterGrid
    valid: np.ndarray
    max_depth_m: np.ndarray
    end_depth_m: np.ndarray
    effective_cn: np.ndarray
    network: FlowNetwork
    hydrograph: pd.DataFrame
    diagnostics: dict
    report_mask: np.ndarray | None = None


def simulate_terrain(dem_m, grid, cn_base, event, prior_rain_mm=20.,
                     drainage_mm_h=0., manning_n=.06, recession_h=6.,
                     max_step_s=30., network=None):
    """Route event excess through D8 cells, conserving volume at each substep.

    Movable water exceeds local depression capacity. Its transport fraction is
    1-exp(-v*dt/link_length), with a Manning velocity proxy limited to 0.02..3 m/s.
    Open outlets and drainage remove water; depression storage remains explicit.
    """
    if not isinstance(event, RainfallEvent):
        raise ValueError("event must be a RainfallEvent")
    if not np.isfinite(manning_n) or manning_n <= 0:
        raise ValueError("manning_n must be finite and greater than zero")
    if not np.isfinite(recession_h) or not 0 <= recession_h <= 168:
        raise ValueError("recession_h must be in [0, 168]")
    if not np.isfinite(max_step_s) or max_step_s <= 0:
        raise ValueError("max_step_s must be finite and greater than zero")
    network = build_flow_network(dem_m, grid) if network is None else network
    valid = network.valid
    if valid.shape != grid.shape:
        raise ValueError("Network must match the model grid")
    selected = np.flatnonzero(valid.ravel())
    # Compact indices keep nodata outside all volume computations.
    inverse = np.full(valid.size, -1, dtype=np.int64)
    inverse[selected] = np.arange(len(selected))
    raw_receiver = network.receiver.ravel()[selected]
    receiver = np.full(len(selected), -1, dtype=np.int64)
    connected = raw_receiver >= 0
    receiver[connected] = inverse[raw_receiver[connected]]
    cn = adjust_cn_amc(np.broadcast_to(np.asarray(cn_base, float), grid.shape).ravel()[selected], prior_rain_mm)
    drainage = np.broadcast_to(np.asarray(drainage_mm_h, float), grid.shape).ravel()[selected]
    if np.any(~np.isfinite(drainage)) or np.any(drainage < 0):
        raise ValueError("Drainage must be finite and non-negative")
    area = grid.cell_area_m2
    capacity = network.depression_capacity_m.ravel()[selected] * area
    lengths = network.link_length_m.ravel()[selected]
    slopes = np.maximum(network.slope.ravel()[selected], 1e-4)
    increments = incremental_runoff_mm(event, cn) * area / 1000.
    times = list(event.interval_h)
    rain = list(event.rain_mm)
    tail_count = int(np.ceil(recession_h / .5))
    times.extend(min(.5, recession_h - i * .5) for i in range(tail_count))
    rain.extend([0.] * tail_count)
    increments = np.concatenate([increments, np.zeros((tail_count, len(selected)))])
    # At the velocity cap a parcel traverses at most half a link per substep.
    step_limit = min(max_step_s, float(lengths.min()) / 6.)
    if sum(int(np.ceil(dt * 3600. / step_limit)) for dt in times) > 60_000:
        raise ValueError("Simulation exceeds 60,000 substeps; shorten the event or coarsen the grid")
    storage = np.zeros(len(selected))
    maximum = np.zeros(len(selected))
    generated_total = outflow_total = drained_total = 0.
    elapsed = 0.
    records = [{"time_h": 0., "rain_mm": 0., "runoff_generated_m3": 0.,
                "outflow_m3_s": 0., "outflow_interval_mean_m3_s": 0.,
                "stored_water_m3": 0., "drained_water_m3": 0., "balance_error_m3": 0.,
                "peak_outflow_m3_s": 0., "peak_outflow_time_h": 0.}]
    peak_flow = peak_time = 0.
    for duration, runoff_volume, rainfall in zip(times, increments, rain):
        substeps = max(1, int(np.ceil(duration * 3600. / step_limit)))
        dt = duration * 3600. / substeps
        increment = runoff_volume / substeps
        interval_out = interval_drain = 0.
        interval_peak_flow, interval_peak_time = 0., elapsed
        last_rate = 0.
        for _ in range(substeps):
            storage += increment
            drained = np.minimum(storage, drainage * area / 1000. * dt / 3600.)
            storage -= drained
            movable = np.maximum(storage - capacity, 0.)
            moving_depth = movable / area
            velocity = np.clip(moving_depth ** (2. / 3.) * np.sqrt(slopes) / manning_n, .02, 3.)
            outgoing = movable * (-np.expm1(-velocity * dt / lengths))
            storage -= outgoing
            incoming = np.bincount(receiver[connected], weights=outgoing[connected], minlength=len(storage))
            storage += incoming
            maximum = np.maximum(maximum, storage / area)
            outlet = float(outgoing[~connected].sum())
            interval_out += outlet
            interval_drain += float(drained.sum())
            last_rate = outlet / dt
            elapsed += dt / 3600.
            if last_rate > interval_peak_flow:
                interval_peak_flow, interval_peak_time = last_rate, elapsed
            if last_rate > peak_flow:
                peak_flow, peak_time = last_rate, elapsed
        generated_total += float(runoff_volume.sum())
        outflow_total += interval_out
        drained_total += interval_drain
        error = generated_total - outflow_total - drained_total - float(storage.sum())
        records.append({"time_h": elapsed, "rain_mm": rainfall,
                        "runoff_generated_m3": float(runoff_volume.sum()),
                        "outflow_m3_s": last_rate,
                        "outflow_interval_mean_m3_s": interval_out / (duration * 3600.),
                        "stored_water_m3": float(storage.sum()),
                        "drained_water_m3": interval_drain, "balance_error_m3": error,
                        "peak_outflow_m3_s": interval_peak_flow, "peak_outflow_time_h": interval_peak_time})
    def expand(values):
        full = np.full(grid.shape, np.nan)
        full.ravel()[selected] = values
        return full
    diagnostics = {"generated_runoff_m3": generated_total,
                   "boundary_outflow_m3": outflow_total, "drained_volume_m3": drained_total,
                   "end_storage_m3": float(storage.sum()),
                   "balance_error_m3": generated_total - outflow_total - drained_total - float(storage.sum()),
                   "peak_boundary_outflow_m3_s": peak_flow, "peak_outflow_time_h": peak_time,
                   "model_area_sqkm": len(selected) * area / 1e6,
                   "cell_resolution_m": abs(grid.transform.a), "recession_h": recession_h,
                   "manning_n": manning_n, "open_outlet_cells": int((~connected).sum())}
    return TerrainResult(grid, valid.copy(), expand(maximum), expand(storage / area),
                         expand(cn), network, pd.DataFrame(records), diagnostics)


def terrain_to_polygons(result, boundary_data, hazard_data, site_name):
    """Reporting cells are independent of NOAH; hazard values are reference only."""
    grid = result.grid
    report = result.valid.copy()
    if boundary_data is not None and not boundary_data.empty:
        boundaries = boundary_data.to_crs(grid.crs)
        report &= geometry_mask([mapping(g) for g in boundaries.geometry], out_shape=grid.shape,
                                transform=grid.transform, invert=True, all_touched=False)
    if not report.any():
        raise ValueError("Selected site does not overlap the DEM at this resolution")
    result.report_mask = report
    hazard_codes = np.zeros(grid.shape, dtype=np.uint8)
    if hazard_data is not None and not hazard_data.empty:
        hazards = hazard_data.to_crs(grid.crs).sort_values("Var")
        shapes = [(mapping(r.geometry), int(r.Var)) for r in hazards.itertuples()]
        hazard_codes = rasterize(shapes, out_shape=grid.shape, transform=grid.transform,
                                 fill=0, all_touched=False, dtype="uint8")
    rows, cols = np.where(report)
    x0 = grid.transform.c + cols * grid.transform.a
    y0 = grid.transform.f + rows * grid.transform.e
    geometries = box(x0, y0 + grid.transform.e, x0 + grid.transform.a, y0)
    cells = gpd.GeoDataFrame({"site_name": site_name, "Var": hazard_codes[report],
                              "sim_depth_m": result.max_depth_m[report],
                              "CN_amc": result.effective_cn[report],
                              "area_sqkm": grid.cell_area_m2 / 1e6,
                              "model_method": "D8 runoff and depression storage"},
                             geometry=geometries, crs=grid.crs).to_crs("EPSG:4326")
    cells["hazard_level"] = cells["Var"].map({0: "Outside NOAH hazard reference", 1: "Low reference hazard",
                                              2: "Moderate reference hazard", 3: "High reference hazard"})
    cells.attrs["terrain_result"] = result
    return cells


@with_rasterio_runtime
def terrain_depth_on_grid(result, target_shape, target_transform, target_crs):
    depth = result.max_depth_m.copy()
    if result.report_mask is not None:
        depth[~result.report_mask] = np.nan
    target = np.full(target_shape, np.nan, dtype=np.float32)
    reproject(depth, target, src_transform=result.grid.transform, src_crs=result.grid.crs,
              src_nodata=np.nan, dst_transform=target_transform, dst_crs=target_crs,
              dst_nodata=np.nan, resampling=Resampling.nearest)
    return target


@with_rasterio_runtime
def depth_geotiff_bytes(result):
    """Peak depth with -9999 outside reporting coverage; units are metres."""
    values = result.max_depth_m.copy()
    if result.report_mask is not None:
        values[~result.report_mask] = np.nan
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=values.shape[1], height=values.shape[0], count=1,
                         dtype="float32", crs=result.grid.crs, transform=result.grid.transform,
                         nodata=-9999., compress="deflate") as dest:
            dest.write(np.where(np.isfinite(values), values, -9999.).astype("float32"), 1)
            dest.update_tags(units="m", statistic="maximum over event and recession",
                             model="D8 runoff and local depression storage screening")
        return memory.read()
