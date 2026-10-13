"""Run the real bundled DEM and check conservation; this is not accuracy validation."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from flood_model import make_rainfall_event
from terrain_model import load_dem_grid, load_physical_grid, PhysicalInputs, build_flow_network, simulate_terrain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution-m", type=float, default=120.)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assumed-inputs", action="store_true", help="Ignore local downloaded rasters to reproduce the assumption-only smoke check")
    args = parser.parse_args()
    started = time.perf_counter()
    grid, dem = load_dem_grid(ROOT / "qc_dem_srtm.tif", args.resolution_m)
    cover_path, hsg_path = ROOT / "data/hydrology/worldcover.tif", ROOT / "data/hydrology/hsg.tif"
    use_local = not args.assumed_inputs and cover_path.exists() and hsg_path.exists()
    inputs = PhysicalInputs(land_cover_path=str(cover_path) if use_local else "",
                            hsg_path=str(hsg_path) if use_local else "")
    cn, drainage, metadata = load_physical_grid(grid, inputs, np.isfinite(dem))
    network = build_flow_network(dem, grid)
    result = simulate_terrain(dem, grid, cn, make_rainfall_event(150., 3.),
                              prior_rain_mm=20., drainage_mm_h=drainage,
                              recession_h=6., network=network)
    d = result.diagnostics
    scale = max(d["generated_runoff_m3"], 1.)
    assert abs(d["balance_error_m3"]) / scale < 1e-9
    assert np.nanmin(result.max_depth_m) >= 0.
    report = {"purpose": "Bundled DEM conservation smoke check; no observed flood data, no accuracy claim",
              "dem": "qc_dem_srtm.tif", "rain_mm": 150., "duration_h": 3., "prior_rain_mm": 20.,
              "soil_and_cover": ("ESA WorldCover 2021 and HiHydroSoil v2.0 local subsets; missing cells use explicit assumptions"
                                 if use_local else "Explicit assumed HSG D, built-up cover with 65% imperviousness"),
              "grid_shape": grid.shape, "diagnostics": d, "input_coverage": metadata,
              "maximum_depth_m": float(np.nanmax(result.max_depth_m)),
              "relative_balance_error": abs(d["balance_error_m3"]) / scale,
              "elapsed_seconds": time.perf_counter() - started}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
