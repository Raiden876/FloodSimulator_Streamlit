import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "flood-simulator-mpl"))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

import flood_model as model
import simulator


def fixture_polygons():
    return gpd.GeoDataFrame({"site_name": ["Fixture"] * 3, "Var": [1, 2, 3],
                            "CN_base": [78., 88., 98.], "base_depth_m": [.25, 1., 2.],
                            "area_sqkm": [1., 1., 1.], "population_estimate": [10., 20., 30.]},
                           geometry=[box(121 + i * .01, 14.6, 121 + (i + 1) * .01, 14.61) for i in range(3)],
                           crs=simulator.DATA_CRS)


class BaselineTests(unittest.TestCase):
    def test_captured_original_vectorized_outputs(self):
        snapshot = json.loads((Path(__file__).parent / "fixtures/legacy_baseline.json").read_text())
        for prior, expected in snapshot["depths_by_prior_rain_mm"].items():
            result = simulator.simulate_flood(fixture_polygons(), snapshot["rain_mm"], snapshot["duration_h"], float(prior))
            np.testing.assert_array_equal(result.sim_depth_m, expected)

    def test_scalar_vector_agree_with_raw_and_effective_cn(self):
        for prior in [0., 35., 40., 52.5, 80.]:
            raw = fixture_polygons()
            for frame in [raw, raw.assign(CN_amc=[np.nan, 82., model.adjust_cn_amc(98., prior)])]:
                with self.subTest(prior=prior, supplied="CN_amc" in frame):
                    result = simulator.simulate_flood(frame, 150., 3., prior)
                    scalar = frame.apply(lambda row: simulator.compute_polygon_depth(row, 150., 3., prior), axis=1)
                    np.testing.assert_array_equal(result.sim_depth_m, scalar)
                    # Passing an already simulated row cannot apply AMC again.
                    repeated = result.apply(lambda row: simulator.compute_polygon_depth(row, 150., 3., prior), axis=1)
                    np.testing.assert_array_equal(result.sim_depth_m, repeated)

    def test_thresholds_are_distinct_and_do_not_zero_shallow_depth(self):
        self.assertEqual(simulator.depth_to_flood_class(.05), simulator.FLOOD_EXPOSURE_ORDER[0])
        self.assertEqual(simulator.depth_to_flood_class(.1), simulator.FLOOD_EXPOSURE_ORDER[1])
        self.assertEqual(simulator.depth_to_flood_class(1.), simulator.FLOOD_EXPOSURE_ORDER[2])
        self.assertEqual(simulator.depth_to_flood_class(2.), simulator.FLOOD_EXPOSURE_ORDER[2])
        self.assertEqual(simulator.depth_to_flood_class(2.01), simulator.FLOOD_EXPOSURE_ORDER[3])
        frame = fixture_polygons().assign(sim_depth_m=[.049, .05, .099])
        self.assertEqual(simulator.flooded_area_sqkm(frame), 2.)
        self.assertEqual(simulator.flooded_area_sqkm(frame, model.EXPOSURE_DEPTH_THRESHOLD_M), 0.)

    def test_cache_key_changes_across_threshold_and_geometry(self):
        one = fixture_polygons().assign(sim_depth_m=[.09999, 0., 0.])
        two = one.assign(sim_depth_m=[.10001, 0., 0.])
        self.assertNotEqual(simulator._simulation_population_cache_key(one), simulator._simulation_population_cache_key(two))
        two = one.copy()
        two.loc[0, "geometry"] = box(121.01, 14.6, 121.02, 14.61)
        self.assertNotEqual(simulator._simulation_population_cache_key(one), simulator._simulation_population_cache_key(two))

    def test_bad_inputs_fail_explicitly(self):
        for field, value in [("rain_mm", -1), ("duration_h", 0), ("prior_rain_mm", np.nan), ("rain_mm", np.inf)]:
            kwargs = dict(rain_mm=150., duration_h=3., prior_rain_mm=20.)
            kwargs[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                simulator.simulate_flood(fixture_polygons(), **kwargs)
        with self.assertRaises(ValueError):
            simulator.simulate_flood(fixture_polygons().assign(CN_base=0.), 150., 3., 20.)


class HydrologyTests(unittest.TestCase):
    def test_known_scs_values_and_extremes(self):
        self.assertAlmostEqual(model.scs_runoff_mm(100., 80.), 50.539058, places=5)
        self.assertEqual(model.scs_runoff_mm(0., 100.), 0.)
        self.assertEqual(model.scs_runoff_mm(150., 100.), 150.)
        for prior in [20., 40., 80.]:
            self.assertEqual(model.adjust_cn_amc(100., prior), 100.)
        self.assertEqual(model.scs_runoff_mm(5., 30.), 0.)

    def test_incremental_runoff_telescope_without_resetting_losses(self):
        event = model.RainfallEvent((10., 20., 0., 30., 90.), (.5, .5, .5, 1., 1.))
        cn = np.array([30., 78., 88., 98., 100.])
        increments = model.incremental_runoff_mm(event, cn)
        self.assertTrue((increments >= 0).all())
        np.testing.assert_allclose(increments.sum(axis=0), model.scs_runoff_mm(150., cn))
        self.assertTrue((increments[2] == 0).all())

    def test_same_total_different_timing_changes_peak_not_total(self):
        uniform = model.RainfallEvent((25.,) * 6, (1.,) * 6)
        burst = model.RainfallEvent((0., 0., 0., 150., 0., 0.), (1.,) * 6)
        one = model.route_runoff(uniform, np.array([80., 95.]), 1.)
        two = model.route_runoff(burst, np.array([80., 95.]), 1.)
        np.testing.assert_allclose(one["runoff_mm"], two["runoff_mm"])
        self.assertTrue((two["peak_mm_h"] > one["peak_mm_h"]).all())
        self.assertFalse(np.array_equal(one["peak_time_h"], two["peak_time_h"]))
        for result in [one, two]:
            np.testing.assert_allclose(result["outflow_mm"].sum(axis=0) + result["residual_storage_mm"], result["runoff_mm"], atol=1e-11)
            self.assertTrue((result["outflow_mm"] >= -1e-12).all())

    def test_exact_reservoir_solution(self):
        event = model.RainfallEvent((10.,), (1.,))
        routed = model.route_runoff(event, 100., 2., recession_h=0.)
        expected = 20. * (1. - np.exp(-.5))
        self.assertAlmostEqual(float(routed["residual_storage_mm"]), expected)
        self.assertAlmostEqual(float(routed["outflow_mm"].sum()), 10. - expected)

    def test_csv_and_presets_have_authoritative_valid_totals(self):
        event = model.RainfallEvent.from_frame(pd.DataFrame({"rain_mm": [20., 50.], "interval_h": [.5, 2.]}))
        self.assertEqual(event.total_mm, 70.)
        self.assertEqual(event.duration_h, 2.5)
        result = simulator.simulate_flood(fixture_polygons(), 1., 1., 40., rainfall_event=event)
        expected = simulator.simulate_flood(fixture_polygons(), 70., 2.5, 40.)
        np.testing.assert_array_equal(result.sim_depth_m, expected.sim_depth_m)
        for pattern in ["Uniform", "Early peak", "Middle peak", "Late peak"]:
            event = model.make_rainfall_event(150., 3.2, pattern)
            self.assertAlmostEqual(event.total_mm, 150.)
            self.assertAlmostEqual(event.duration_h, 3.2)

    def test_invalid_rainfall_intervals(self):
        for rain, durations in [([], []), ([10.], [0.]), ([np.nan], [1.]), ([-1.], [1.]), ([10., 2.], [1.])]:
            with self.subTest(rain=rain), self.assertRaises(ValueError):
                model.RainfallEvent(tuple(rain), tuple(durations))

    def test_hsg_uses_conductivity_restriction_and_water_table(self):
        np.testing.assert_array_equal(model.hsg_from_properties([41., 40., 10., 1.], [np.nan] * 4, [75.] * 4, [75.] * 4), [1., 2., 3., 4.])
        np.testing.assert_array_equal(model.hsg_from_properties([np.nan] * 4, [11., 10., 4., .4], [150.] * 4, [150.] * 4), [1., 2., 3., 4.])
        self.assertEqual(float(model.hsg_from_properties(50., 50., 40., 150.)), 4.)
        self.assertEqual(float(model.hsg_from_properties(50., 50., 150., 30.)), 4.)
        self.assertTrue(np.isnan(model.hsg_from_properties(np.nan, np.nan, 150., 150.)))
        self.assertEqual(model.normalize_hsg("A/D"), "D")
        self.assertEqual(model.normalize_hsg("A/D", drained=True), "A")

    def test_cn_responds_to_soil_cover_and_imperviousness(self):
        np.testing.assert_allclose(model.curve_numbers([10, 10, 60, 50, 80], [1, 4, 4, 4, 1]), [30., 77., 94., 91.7, 100.])
        self.assertEqual(float(model.curve_numbers(50, 1, impervious_fraction=1.)), 98.)
        self.assertEqual(float(model.curve_numbers(50, 1, impervious_fraction=0.)), 39.)
        with self.assertRaises(ValueError):
            model.curve_numbers(50, 1, impervious_fraction=65.)

    def test_routed_empirical_depth_changes_with_storm_timing(self):
        frame = fixture_polygons()
        one = simulator.simulate_flood(frame, 150., 6., 40., rainfall_event=model.RainfallEvent((25.,) * 6, (1.,) * 6), use_routed_depth=True)
        two = simulator.simulate_flood(frame, 150., 6., 40., rainfall_event=model.RainfallEvent((0., 0., 0., 150., 0., 0.), (1.,) * 6), use_routed_depth=True)
        np.testing.assert_allclose(one.runoff_mm, two.runoff_mm)
        self.assertTrue((two.sim_depth_m > one.sim_depth_m).all())

    def test_validation_masks_nodata_and_calculates_known_metrics(self):
        metrics = model.validation_metrics([0., .2, .3, np.nan], [0., .1, 0., 1.])
        self.assertEqual(metrics["valid_cells"], 3)
        self.assertEqual(metrics["inundation_iou"], .5)
        self.assertEqual(metrics["critical_success_index"], .5)
        self.assertAlmostEqual(metrics["depth_mae_m"], .4 / 3.)
        self.assertAlmostEqual(metrics["depth_rmse_m"], np.sqrt(.1 / 3.))
        self.assertTrue(np.isnan(model.validation_metrics([0.], [0.])["inundation_iou"]))
        self.assertEqual(model.validation_metrics([.2], [np.nan])["valid_cells"], 0)


class AnalysisTests(unittest.TestCase):
    def test_rain_and_runoff_keep_a_shared_clock_and_show_lag(self):
        event = model.RainfallEvent((0., 0., 0., 150., 0., 0.), (1.,) * 6)
        hydrograph = pd.DataFrame({"time_h": [1., 2., 3., 4., 5., 6., 7.],
                                   "outflow_m3_s": [0., 0., 0., 8., 4., 2., 1.]})
        rain, runoff, timing = model.rainfall_runoff_analysis(event, hydrograph)
        self.assertEqual(timing["axis_end_h"], 7.)
        self.assertEqual(timing["peak_rain_start_h"], 3.)
        self.assertEqual(timing["peak_rain_end_h"], 4.)
        self.assertEqual(timing["peak_runoff_time_h"], 4.)
        self.assertEqual(timing["apparent_lag_h"], .5)
        self.assertEqual(timing["lag_min_h"], 0.)
        self.assertEqual(timing["lag_max_h"], 1.)
        self.assertEqual(runoff.time_h.iloc[0], 0.)
        self.assertEqual(rain.intensity_mm_h.max(), 150.)

    def test_flat_rainfall_and_zero_flow_do_not_invent_lag(self):
        event = model.make_rainfall_event(150., 3.)
        hydrograph = pd.DataFrame({"time_h": [1., 2., 3., 4.], "outflow_m3_s": [1., 3., 5., 2.]})
        _, _, timing = model.rainfall_runoff_analysis(event, hydrograph)
        self.assertIsNone(timing["apparent_lag_h"])
        event = model.make_rainfall_event(0., 3.)
        _, _, timing = model.rainfall_runoff_analysis(event, hydrograph.assign(outflow_m3_s=0.))
        self.assertIsNone(timing["apparent_lag_h"])

    def test_terrain_substep_peak_is_preserved_in_plot_samples(self):
        event = model.RainfallEvent((100., 0.), (1., 1.))
        trace = pd.DataFrame({"time_h": [0., 1., 2.], "outflow_m3_s": [0., 5., 9.],
                              "peak_outflow_time_h": [0., 1., 1.5],
                              "peak_outflow_m3_s": [0., 5., 11.]})
        _, flow, timing = model.rainfall_runoff_analysis(event, trace)
        self.assertEqual(timing["peak_runoff_time_h"], 1.5)
        self.assertEqual(timing["peak_runoff_m3_s"], 11.)
        self.assertEqual(timing["apparent_lag_h"], 1.)


if __name__ == "__main__":
    unittest.main()
