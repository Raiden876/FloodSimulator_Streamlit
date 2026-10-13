"""Exercise app branches with small real raster fixtures; no server or browser."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "flood-simulator-mpl"))
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from streamlit.testing.v1 import AppTest

import simulator
from terrain_model import RasterGrid, load_physical_grid, build_flow_network
from test_terrain_model import write_raster


class AppTests(unittest.TestCase):
    def test_model_choices_and_terrain_exports_render(self):
        with tempfile.TemporaryDirectory() as folder:
            grid = RasterGrid((5, 7), from_origin(300000., 1620000., 120., 120.))
            dem = np.tile(np.arange(7., 0., -1.) * 3., (5, 1))
            population_path = Path(folder) / "population.tif"
            write_raster(population_path, np.full(grid.shape, 10.), grid)
            geometries = [box(300000., 1619400., 300360., 1620000.), box(300360., 1619400., 300840., 1620000.)]
            hazards = gpd.GeoDataFrame({"site_name": ["Metro Manila"] * 2, "Var": [1, 3],
                                        "hazard_level": ["Low", "High"], "base_depth_m": [.25, 2.],
                                        "CN_base": [78., 98.], "area_sqkm": [.216, .288],
                                        "population_estimate": [150., 200.], "source_return_period_yr": [25, 25]},
                                       geometry=geometries, crs=grid.crs).to_crs(simulator.DATA_CRS)
            boundaries = gpd.GeoDataFrame({"site_name": ["Metro Manila"]},
                                          geometry=[box(300000., 1619400., 300840., 1620000.)],
                                          crs=grid.crs).to_crs(simulator.DATA_CRS)
            def terrain_inputs(dem_path, resolution_m, inputs, file_key):
                cn, drainage, metadata = load_physical_grid(grid, inputs)
                return grid, dem, cn, drainage, metadata, build_flow_network(dem, grid)
            def select(app, label, choice):
                next(widget for widget in app.selectbox if widget.label == label).select(choice)
            with patch.object(simulator, "load_data", return_value=hazards), \
                 patch.object(simulator, "find_population_raster", return_value=population_path), \
                 patch.object(simulator, "load_population_exposure", side_effect=lambda data, _: data), \
                 patch.object(simulator, "load_boundary_data", return_value=boundaries), \
                 patch.object(simulator, "_terrain_inputs_cached", side_effect=terrain_inputs), \
                 patch.object(simulator, "st_folium", return_value=None):
                app = AppTest.from_string("import simulator; simulator.run_app()", default_timeout=30).run()
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertEqual(len(app.error), 0, str(app.error))
                self.assertEqual(len(app.metric), 4)
                self.assertEqual(len(app.tabs), 0)
                self.assertEqual(len(app.dataframe), 0)  # Details are lazy and collapsed.
                for layer in ["Population density", "Affected population", "UP NOAH reference", "Flood depth"]:
                    select(app, "Map layer", layer)
                    app.run()
                    self.assertEqual(len(app.exception), 0, str(app.exception))
                    self.assertEqual(len(app.error), 0, str(app.error))
                app.session_state["storm_analysis"] = True
                app.session_state["population_details"] = True
                app.session_state["model_diagnostics"] = True
                app.run()
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertGreaterEqual(len(app.dataframe), 5)
                self.assertTrue(any(s.value == "Rainfall timing" for s in app.subheader))
                self.assertTrue(any(s.value == "Runoff response" for s in app.subheader))
                select(app, "Simulation method", "Corrected empirical baseline")
                app.run()
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertEqual(len(app.error), 0, str(app.error))
                select(app, "Site / province", "Metro Manila")
                select(app, "Simulation method", "Terrain inundation (screening)")
                app.session_state["model_diagnostics"] = True
                app.session_state["storm_analysis"] = True
                app.run()
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertEqual(len(app.error), 0, str(app.error))
                # Exercise independent validation with a local observation raster.
                observed_path = Path(folder) / "observed.tif"
                write_raster(observed_path, np.full(grid.shape, .1), grid)
                next(w for w in app.text_input if w.label.startswith("Independent observed depth")).set_value(str(observed_path))
                app.session_state["model_diagnostics"] = True
                app.run()
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertEqual(len(app.error), 0, str(app.error))
                self.assertTrue(any(s.value == "Independent observed-depth validation" for s in app.subheader))
                # Missing CSV is a clear request for input, not an app exception.
                select(app, "Rainfall timing", "CSV time series")
                app.run()
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertTrue(any("rainfall CSV" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
