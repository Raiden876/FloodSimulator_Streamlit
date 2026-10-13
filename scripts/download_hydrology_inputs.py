"""Download site-specific WorldCover and HiHydroSoil subsets for every configured area.

The inputs cover each UP NOAH flood-polygon bounding envelope; the app samples only the hazard polygons. To keep them
usable by the simulator, WorldCover is resampled to 30 m and HSG to 250 m in
UTM zone 51N. Reads use the providers' public Cloud-Optimized GeoTIFFs; no
credentials or full global downloads are required.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from shapely.geometry import mapping
from shapely.ops import unary_union

from geo_runtime import rasterio_runtime
from site_config import SITE_MAPPING, site_key

TARGET_CRS = "EPSG:32651"
NODATA = 0
MAX_OUTPUT_CELLS = 50_000_000
WORLD_COVER_RESOLUTION_M = 30
HSG_RESOLUTION_M = 250
WORLD_COVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif"
SOURCES = {
    "land_cover": {
        "title": "ESA WorldCover 2021 v200",
        "documentation": "https://esa-worldcover.org/en/data-access",
        "citation": "Zanaga et al. (2022), ESA WorldCover 10 m 2021 v200, doi:10.5281/zenodo.7254221",
        "license": "CC BY 4.0",
        "native_resolution": "10 m",
        "output_resolution_m": WORLD_COVER_RESOLUTION_M,
        "epoch": "2021",
        "type": "Satellite-derived land-cover classification, nearest-neighbour resampled",
        "codes": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
        "filename": "worldcover.tif",
    },
    "soil": {
        "url": "https://data.naturalcapitalalliance.stanford.edu/download/global/HiHydroSoilv2/HiHydroSoilV2_Soil_Groups_250m_reclassified.tif",
        "title": "HiHydroSoil v2.0 hydrologic soil groups (NatCap reclassification)",
        "documentation": "https://data.naturalcapitalalliance.stanford.edu/dataset/sts-cdd01bf6db49adf846783f7d47d9ca1186c9b3143df283e9539a2d8969c07ecb",
        "citation": "Simons, G.W.H., R. Koster, P. Droogers (2020), HiHydroSoil v2.0, FutureWater Report 213; Natural Capital Alliance reclassified derivative",
        "license": "CC BY 4.0",
        "native_resolution": "250 m",
        "output_resolution_m": HSG_RESOLUTION_M,
        "epoch": "Global model estimate; HiHydroSoil v2.0 (2020)",
        "type": "Model-derived HSG from soil hydraulic properties and depth; not field observations",
        "codes": [1, 2, 3, 4],
        "filename": "hsg.tif",
    },
}


def find_shapefile_member(archive_path):
    with ZipFile(archive_path) as archive:
        members = sorted(name for name in archive.namelist() if name.lower().endswith(".shp"))
    if not members:
        raise FileNotFoundError(f"No shapefile found in {archive_path.name}")
    return members[0]


def site_footprint(site_name):
    archive_path = ROOT / SITE_MAPPING[site_name]["archive"]
    if not archive_path.exists():
        raise FileNotFoundError(f"Configured hazard archive is missing: {archive_path}")
    member = find_shapefile_member(archive_path)
    layer = gpd.read_file(f"zip://{archive_path.resolve().as_posix()}!/{member}")
    if layer.crs is None:
        raise ValueError(f"The hazard layer for {site_name} has no CRS")
    if layer.empty:
        raise ValueError(f"The hazard layer for {site_name} has no features")
    # Use transformed layer bounds rather than repairing all source vertices.
    # The app samples these rasters only where its polygons exist, so this
    # bounded rectangular envelope is sufficient and keeps large archives fast.
    source_bounds = layer.total_bounds
    bounds_wgs84 = transform_bounds(layer.crs, "EPSG:4326", *source_bounds, densify_pts=21)
    bounds_projected = transform_bounds(layer.crs, TARGET_CRS, *source_bounds, densify_pts=21)
    from shapely.geometry import box
    footprint_envelope = box(*bounds_projected)
    return footprint_envelope, [float(value) for value in bounds_wgs84], member


def aligned_grid(geometry, resolution_m):
    min_x, min_y, max_x, max_y = geometry.bounds
    left = math.floor(min_x / resolution_m) * resolution_m
    bottom = math.floor(min_y / resolution_m) * resolution_m
    right = math.ceil(max_x / resolution_m) * resolution_m
    top = math.ceil(max_y / resolution_m) * resolution_m
    width = max(1, int(round((right - left) / resolution_m)))
    height = max(1, int(round((top - bottom) / resolution_m)))
    if width * height > MAX_OUTPUT_CELLS:
        raise ValueError(f"Site raster grid would have {width * height:,} cells; split the configured site first")
    transform = from_origin(left, top, resolution_m, resolution_m)
    inside = geometry_mask([mapping(geometry)], out_shape=(height, width), transform=transform,
                           invert=True, all_touched=True)
    return (height, width), transform, inside


def worldcover_tile_urls(bounds_wgs84):
    west, south, east, north = bounds_wgs84
    epsilon = 1e-10
    min_lon, max_lon = math.floor(west / 3) * 3, math.floor((east - epsilon) / 3) * 3
    min_lat, max_lat = math.floor(south / 3) * 3, math.floor((north - epsilon) / 3) * 3
    urls = []
    for lat in range(min_lat, max_lat + 1, 3):
        for lon in range(min_lon, max_lon + 1, 3):
            lat_id = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            lon_id = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            tile_id = f"{lat_id}{lon_id}"
            urls.append((tile_id, WORLD_COVER_BASE.format(tile_id)))
    return urls


def project_source(url, shape, transform, inside, codes, *, categorical_nodata=0):
    destination = np.full(shape, NODATA, dtype=np.uint8)
    with rasterio.open(url) as source:
        scratch = np.full(shape, NODATA, dtype=np.uint8)
        reproject(source=rasterio.band(source, 1), destination=scratch,
                  src_transform=source.transform, src_crs=source.crs,
                  src_nodata=source.nodata if source.nodata is not None else categorical_nodata,
                  dst_transform=transform, dst_crs=TARGET_CRS, dst_nodata=NODATA,
                  resampling=Resampling.nearest, init_dest_nodata=True)
        valid = inside & (scratch != NODATA)
        destination[valid] = scratch[valid]
    unexpected = sorted(set(np.unique(destination[inside]).astype(int)).difference(codes).difference({NODATA}))
    if unexpected:
        raise ValueError(f"Unexpected raster classes in {url}: {unexpected}")
    return destination


def write_site_raster(output, values, transform, source_info, urls):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.tif")
    height, width = values.shape
    profile = dict(driver="GTiff", height=height, width=width, count=1, dtype="uint8",
                   crs=TARGET_CRS, transform=transform, nodata=NODATA,
                   compress="deflate", predictor=2, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(values, 1)
        destination.update_tags(source_title=source_info["title"], citation=source_info["citation"],
                                license=source_info["license"], data_type=source_info["type"],
                                source_urls=";".join(urls))
    temporary.replace(output)
    digest = hashlib.sha256()
    with output.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"local_path": output.relative_to(ROOT).as_posix(), "shape": [height, width],
            "crs": TARGET_CRS, "output_resolution_m": source_info["output_resolution_m"],
            "classes_present": sorted(int(value) for value in np.unique(values) if value != NODATA),
            "valid_footprint_cells": int(np.count_nonzero(values)), "sha256": digest.hexdigest(),
            "bytes": output.stat().st_size}


def download_one(site_name, selected_sources, force, manifest):
    footprint, bounds_wgs84, hazard_member = site_footprint(site_name)
    site_folder = ROOT / "data/hydrology/sites" / site_key(site_name)
    site_entry = manifest.setdefault("sites", {}).setdefault(site_name, {})
    site_entry.update({"coverage": "bounding envelope of UP NOAH hazard polygons", "hazard_archive": SITE_MAPPING[site_name]["archive"],
                       "hazard_layer": hazard_member, "bounds_wgs84": bounds_wgs84,
                       "retrieved_utc": datetime.now(timezone.utc).isoformat()})
    if "land_cover" in selected_sources:
        output = site_folder / SOURCES["land_cover"]["filename"]
        old = site_entry.get("land_cover", {})
        if force or not output.exists() or not old.get("sha256"):
            urls = worldcover_tile_urls(bounds_wgs84)
            shape, transform, inside = aligned_grid(footprint, WORLD_COVER_RESOLUTION_M)
            values = np.full(shape, NODATA, dtype=np.uint8)
            for tile_id, url in urls:
                print(f"  {site_name}: reading WorldCover tile {tile_id}", flush=True)
                values = np.where(inside & (values == NODATA),
                                  project_source(url, shape, transform, inside, SOURCES["land_cover"]["codes"]),
                                  values).astype(np.uint8)
            if not np.count_nonzero(values):
                raise ValueError(f"No WorldCover cells were available within {site_name}")
            info = dict(SOURCES["land_cover"])
            info.update({"source_urls": [url for _, url in urls],
                         "documentation": SOURCES["land_cover"]["documentation"],
                         "native_resolution": SOURCES["land_cover"]["native_resolution"],
                         "output_resolution_m": WORLD_COVER_RESOLUTION_M,
                         "type": SOURCES["land_cover"]["type"] + "; clipped to the UP NOAH footprint bounding envelope"})
            record = write_site_raster(output, values, transform, info, info["source_urls"])
            record.update({key: value for key, value in info.items() if key not in {"filename", "codes"}})
            record["tile_ids"] = [tile_id for tile_id, _ in urls]
            site_entry["land_cover"] = record
            manifest["sites"][site_name] = site_entry
            (ROOT / "data/hydrology/sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"  Saved {output.relative_to(ROOT)} ({record['bytes']:,} bytes)", flush=True)
        else:
            print(f"  {site_name}: keeping existing {output.relative_to(ROOT)}", flush=True)
    if "soil" in selected_sources:
        output = site_folder / SOURCES["soil"]["filename"]
        old = site_entry.get("soil", {})
        if force or not output.exists() or not old.get("sha256"):
            shape, transform, inside = aligned_grid(footprint, HSG_RESOLUTION_M)
            print(f"  {site_name}: reading HiHydroSoil subset", flush=True)
            values = project_source(SOURCES["soil"]["url"], shape, transform, inside, SOURCES["soil"]["codes"])
            if not np.count_nonzero(values):
                raise ValueError(f"No HiHydroSoil cells were available within {site_name}")
            info = dict(SOURCES["soil"])
            info["source_urls"] = [SOURCES["soil"]["url"]]
            info["type"] += "; clipped to the UP NOAH footprint bounding envelope"
            record = write_site_raster(output, values, transform, info, info["source_urls"])
            record.update({key: value for key, value in info.items() if key not in {"filename", "codes"}})
            site_entry["soil"] = record
            manifest["sites"][site_name] = site_entry
            (ROOT / "data/hydrology/sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"  Saved {output.relative_to(ROOT)} ({record['bytes']:,} bytes)", flush=True)
        else:
            print(f"  {site_name}: keeping existing {output.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=list(SITE_MAPPING), action="append",
                        help="download for this site; repeat to choose several (default: all configured sites)")
    parser.add_argument("--only", choices=["land_cover", "soil", "both"], default="both")
    parser.add_argument("--force", action="store_true", help="replace existing site rasters")
    args = parser.parse_args()
    sites = args.site or list(SITE_MAPPING)
    selected_sources = {"land_cover", "soil"} if args.only == "both" else {args.only}
    folder = ROOT / "data/hydrology"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "sources.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sources": {}}
    manifest.setdefault("sources", {})
    with rasterio_runtime(), rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT=45,
                                          GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=2,
                                          GDAL_CACHEMAX=64):
        for site_name in sites:
            print(f"Preparing hydrology inputs for {site_name}...", flush=True)
            download_one(site_name, selected_sources, args.force, manifest)
    print(f"Prepared {len(sites)} sites. The app selects rasters automatically by site.")


if __name__ == "__main__":
    main()
