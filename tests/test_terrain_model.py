import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "flood-simulator-mpl"))
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box

from flood_model import RainfallEvent, make_rainfall_event
import simulator
from terrain_model import (RasterGrid, PhysicalInputs, build_flow_network, simulate_terrain,
                           read_aligned_raster, load_physical_grid, enrich_runoff_inputs,
                           runoff_grid_for_geometries, terrain_to_polygons, TerrainResult,
                           terrain_depth_on_grid, depth_geotiff_bytes)


def write_raster(path, values, grid, nodata=-9999.):
    values = np.asarray(values, float)
    with rasterio.open(path, "w", driver="GTiff", height=grid.shape[0], width=grid.shape[1], count=1,
                       dtype="float32", crs=grid.crs, transform=grid.transform, nodata=nodata) as dest:
        dest.write(np.where(np.isfinite(values), values, nodata).astype("float32"), 1)


class TerrainTests(unittest.TestCase):
    def setUp(self):
        self.grid = RasterGrid((5, 7), from_origin(300000., 1620000., 120., 120.))
        self.dem = np.tile(np.arange(7., 0., -1.) * 3., (5, 1))

    def test_connections_are_acyclic_and_accumulate_upstream_cells(self):
        network = build_flow_network(self.dem, self.grid)
        for start in np.flatnonzero(network.valid):
            visited = set()
            index = int(start)
            while index >= 0:
                self.assertNotIn(index, visited)
                visited.add(index)
                index = int(network.receiver.ravel()[index])
        outlets = network.receiver < 0
        self.assertEqual(float(network.accumulation_cells[outlets].sum()), network.valid.sum())
        self.assertGreater(network.accumulation_cells.max(), 1.)

    def test_flat_cells_and_nodata_have_no_cycles(self):
        dem = np.zeros(self.grid.shape)
        dem[2, 3] = np.nan
        network = build_flow_network(dem, self.grid)
        for start in np.flatnonzero(network.valid):
            visited, index = set(), int(start)
            while index >= 0:
                self.assertNotIn(index, visited)
                visited.add(index)
                index = int(network.receiver.ravel()[index])
        result = simulate_terrain(dem, self.grid, 100., make_rainfall_event(100., 1.), 40., recession_h=.5)
        self.assertTrue(np.isnan(result.max_depth_m[2, 3]))
        self.assertAlmostEqual(result.diagnostics["generated_runoff_m3"], 34 * .1 * 14400.)
        self.assertLess(abs(result.diagnostics["balance_error_m3"]), 1e-6)

    def test_zero_rain_generates_no_water(self):
        result = simulate_terrain(self.dem, self.grid, 88., make_rainfall_event(0., 1.), recession_h=1.)
        np.testing.assert_array_equal(result.max_depth_m, 0.)
        self.assertEqual(result.diagnostics["generated_runoff_m3"], 0.)
        self.assertEqual(result.diagnostics["boundary_outflow_m3"], 0.)

    def test_routing_drainage_and_residual_storage_conserve_water(self):
        result = simulate_terrain(self.dem, self.grid, 88., make_rainfall_event(150., 3.), 40.,
                                  drainage_mm_h=2., recession_h=1.)
        d = result.diagnostics
        self.assertGreater(d["boundary_outflow_m3"], 0.)
        self.assertGreater(d["drained_volume_m3"], 0.)
        self.assertGreaterEqual(d["end_storage_m3"], 0.)
        self.assertAlmostEqual(d["generated_runoff_m3"], d["boundary_outflow_m3"] + d["drained_volume_m3"] + d["end_storage_m3"], places=6)
        self.assertLess(result.hydrograph.balance_error_m3.abs().max(), 1e-6)
        self.assertTrue((result.max_depth_m >= 0).all())

    def test_timing_changes_terrain_peak_with_same_runoff_volume(self):
        uniform = RainfallEvent((25.,) * 6, (1.,) * 6)
        burst = RainfallEvent((0., 0., 0., 150., 0., 0.), (1.,) * 6)
        one = simulate_terrain(self.dem, self.grid, 88., uniform, 40., recession_h=1.)
        two = simulate_terrain(self.dem, self.grid, 88., burst, 40., recession_h=1.)
        self.assertAlmostEqual(one.diagnostics["generated_runoff_m3"], two.diagnostics["generated_runoff_m3"])
        self.assertGreater(two.diagnostics["peak_boundary_outflow_m3_s"], one.diagnostics["peak_boundary_outflow_m3_s"])
        self.assertGreater(np.nanmax(two.max_depth_m), np.nanmax(one.max_depth_m))

    def test_local_depression_storage_preserves_original_datum(self):
        dem = np.full(self.grid.shape, 10.)
        dem[2, 3] = 0.
        result = simulate_terrain(dem, self.grid, 100., make_rainfall_event(150., 1.), 40., recession_h=1.)
        self.assertEqual(result.network.depression_capacity_m[2, 3], 10.)
        self.assertGreater(result.end_depth_m[2, 3], .15)  # Receives upstream water as well.
        local_cn = np.full(self.grid.shape, 30.)
        local_cn[2, 3] = 100.
        isolated = simulate_terrain(dem, self.grid, local_cn, make_rainfall_event(50., 1.), 40., recession_h=1.)
        self.assertAlmostEqual(isolated.end_depth_m[2, 3], .05)
        self.assertAlmostEqual(result.network.filled_dem_m[2, 3], 10.)
        self.assertLess(abs(result.diagnostics["balance_error_m3"]), 1e-6)


class SpatialInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.grid = RasterGrid((2, 2), from_origin(300000., 1620000., 120., 120.))

    def test_real_soil_and_cover_rasters_control_cn(self):
        cover_path, hsg_path = self.folder / "cover.tif", self.folder / "soil.tif"
        write_raster(cover_path, [[10, 10], [50, np.nan]], self.grid)
        write_raster(hsg_path, [[1, 4], [1, np.nan]], self.grid)
        cn, drainage, metadata = load_physical_grid(self.grid, PhysicalInputs(land_cover_path=str(cover_path), hsg_path=str(hsg_path)))
        np.testing.assert_allclose(cn, [[30., 77.], [77.35, 91.7]])
        self.assertEqual(metadata["soil_assumed_pct"], 25.)
        self.assertEqual(metadata["land_cover_assumed_pct"], 25.)
        np.testing.assert_array_equal(drainage, 0.)

    def test_soil_diagnostics_and_dual_group_are_used(self):
        paths = []
        for name, values in [("k50", [[41, 15], [3, .5]]), ("k100", [[11, 6], [2, .3]]),
                             ("restriction", np.full((2, 2), 150)), ("water", [[150, 150], [150, 30]])]:
            p = self.folder / (name + ".tif")
            write_raster(p, values, self.grid)
            paths.append(str(p))
        inputs = PhysicalInputs(ksat_0_50_path=paths[0], ksat_0_100_path=paths[1],
                                restrictive_depth_path=paths[2], water_table_depth_path=paths[3], assumed_land_cover=10)
        cn, _, metadata = load_physical_grid(self.grid, inputs)
        np.testing.assert_allclose(cn, [[30, 55], [70, 77]])
        self.assertEqual(metadata["soil_assumed_pct"], 0.)
        hsg_path = self.folder / "dual.tif"
        write_raster(hsg_path, [[5, 6], [7, 4]], self.grid)
        wet, _, _ = load_physical_grid(self.grid, PhysicalInputs(hsg_path=str(hsg_path), assumed_land_cover=10))
        drained, _, _ = load_physical_grid(self.grid, PhysicalInputs(hsg_path=str(hsg_path), assumed_land_cover=10, dual_hsg_drained=True))
        np.testing.assert_array_equal(wet, 77.)
        np.testing.assert_array_equal(drained, [[30, 55], [70, 77]])

    def test_hazard_class_does_not_determine_physical_cn(self):
        polygons = gpd.GeoDataFrame({"Var": [1, 3], "CN_base": [78., 98.]},
                                   geometry=[box(300000., 1619760., 300120., 1620000.), box(300120., 1619760., 300240., 1620000.)],
                                   crs=self.grid.crs)
        inputs = PhysicalInputs(assumed_hsg="B", assumed_land_cover=10)
        result = enrich_runoff_inputs(polygons, inputs)
        np.testing.assert_array_equal(result.CN_base, [55., 55.])
        np.testing.assert_array_equal(result.CN_legacy, [78., 98.])
        cover_path, hsg_path = self.folder / "cover.tif", self.folder / "soil.tif"
        write_raster(cover_path, [[10, 60], [10, 60]], self.grid)
        write_raster(hsg_path, [[1, 4], [1, 4]], self.grid)
        result = enrich_runoff_inputs(polygons, PhysicalInputs(land_cover_path=str(cover_path), hsg_path=str(hsg_path)))
        np.testing.assert_allclose(result.CN_base, [30., 94.])

    def test_large_polygon_runoff_inputs_adapt_resolution_and_report_it(self):
        polygon = gpd.GeoDataFrame({"CN_base": [80.]},
                                   geometry=[box(300000., 1520000., 400000., 1620000.)],
                                   crs=self.grid.crs)
        grid, resolution = runoff_grid_for_geometries(polygon.geometry, 120.)
        self.assertGreater(resolution, 120.)
        self.assertLessEqual(grid.shape[0] * grid.shape[1], 150_000)
        cover_path, hsg_path = self.folder / "cover.tif", self.folder / "soil.tif"
        write_raster(cover_path, [[10, 50], [10, 50]], self.grid)
        write_raster(hsg_path, [[1, 4], [1, 4]], self.grid)
        result = enrich_runoff_inputs(polygon, PhysicalInputs(land_cover_path=str(cover_path),
                                                              hsg_path=str(hsg_path)))
        metadata = result.attrs["runoff_inputs"]
        self.assertGreater(metadata["runoff_grid_resolution_m"], 120.)
        self.assertLessEqual(metadata["runoff_grid_total_cells"], 150_000)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isfinite(result.CN_base.iloc[0]))

    def test_nodata_outside_extent_is_unknown(self):
        path = self.folder / "observed.tif"
        write_raster(path, [[0., .2], [.3, np.nan]], self.grid)
        shifted = RasterGrid((2, 3), self.grid.transform, self.grid.crs)
        aligned = read_aligned_raster(path, shifted)
        self.assertEqual(aligned[0, 0], 0.)
        self.assertTrue(np.isnan(aligned[1, 1]))
        self.assertTrue(np.isnan(aligned[:, 2]).all())

    def test_invalid_units_codes_and_missing_paths_fail(self):
        path = self.folder / "invalid.tif"
        write_raster(path, np.full((2, 2), 65.), self.grid)
        with self.assertRaises(ValueError):
            load_physical_grid(self.grid, PhysicalInputs(impervious_fraction_path=str(path)))
        with self.assertRaises(ValueError):
            load_physical_grid(self.grid, PhysicalInputs(land_cover_path=str(path)))
        with self.assertRaises(FileNotFoundError):
            load_physical_grid(self.grid, PhysicalInputs(hsg_path=str(self.folder / "absent.tif")))


class PopulationCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.grid = RasterGrid((2, 3), from_origin(300000., 1620000., 120., 120.))
        self.depths = np.array([[.04, .05, .1], [.3, .5, np.nan]])
        self.pop_path = Path(self.temp.name) / "population.tif"
        write_raster(self.pop_path, [[10, 20, 30], [40, 50, 60]], self.grid)
        network = build_flow_network(np.where(np.isfinite(self.depths), 0., np.nan), self.grid)
        self.result = TerrainResult(self.grid, network.valid, self.depths, self.depths.copy(),
                                    np.full(self.grid.shape, 88.), network, pd.DataFrame(), {})
        self.boundaries = gpd.GeoDataFrame({"site_name": ["Fixture"]},
                                           geometry=[box(300000., 1619760., 300360., 1620000.)], crs=self.grid.crs)
        self.cells = terrain_to_polygons(self.result, self.boundaries, None, "Fixture")

    def test_population_is_counted_once_with_unknown_coverage_separate(self):
        summary = simulator._population_flood_summary_uncached(self.cells, self.pop_path, self.boundaries)
        self.assertEqual(summary.people_in_band.sum(), 210.)
        self.assertEqual(summary.people_affected.sum(), 120.)
        unmodeled = summary[summary.flood_depth_band == "Outside model coverage"]
        self.assertEqual(unmodeled.people_in_band.sum(), 60.)
        self.assertEqual(unmodeled.people_affected.sum(), 0.)
        sensitivity = simulator._population_exposure_sensitivity_uncached(self.cells, self.pop_path, self.boundaries)
        np.testing.assert_array_equal(sensitivity.people_affected, [140., 120., 90., 50.])
        np.testing.assert_array_equal(sensitivity.people_unmodeled, 60.)

    def test_geotiff_preserves_depth_and_coverage(self):
        data = depth_geotiff_bytes(self.result)
        with MemoryFile(data) as memory, memory.open() as source:
            values = source.read(1, masked=True)
            self.assertTrue(values.mask[1, 2])
            np.testing.assert_allclose(values.compressed(), self.depths[np.isfinite(self.depths)], rtol=1e-6)
            self.assertEqual(source.crs.to_epsg(), 32651)
            self.assertEqual(source.tags()["units"], "m")

    def test_population_dot_sampling_uses_grid_depth_and_ignores_nodata(self):
        coordinates = gpd.GeoSeries([box(300000 + col * 120, 1620000 - (row + 1) * 120,
                                              300120 + col * 120, 1620000 - row * 120).centroid
                                    for row in range(2) for col in range(3)], crs=self.grid.crs).to_crs(simulator.DATA_CRS)
        dots = pd.DataFrame({"longitude": coordinates.x, "latitude": coordinates.y, "population_estimate": [1.] * 6})
        affected = simulator._population_dots_in_flood(dots, self.cells)
        self.assertEqual(len(affected), 3)
        np.testing.assert_allclose(sorted(affected.sim_depth_m), [.1, .3, .5])


if __name__ == "__main__":
    unittest.main()
