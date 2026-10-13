import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import unittest

import rasterio
from rasterio.env import set_proj_data_search_path
from geo_runtime import rasterio_runtime, configure_rasterio_data


class GeoRuntimeTests(unittest.TestCase):
    def test_worker_recovers_missing_proj_search_path(self):
        def worker():
            set_proj_data_search_path(str(Path("missing-proj-database").resolve()))
            with rasterio_runtime():
                self.assertEqual(rasterio.crs.CRS.from_epsg(4326).to_epsg(), 4326)
                self.assertEqual(rasterio.crs.CRS.from_epsg(32651).to_epsg(), 32651)
                return configure_rasterio_data()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            for _ in range(2):
                location = pool.submit(worker).result()
                self.assertTrue((Path(location) / "proj.db").is_file())

    def test_invalid_inherited_proj_environment_is_repaired(self):
        code = "import os; os.environ['PROJ_DATA']='missing-proj-database'; os.environ['PROJ_LIB']='missing-proj-database'; import rasterio; from geo_runtime import rasterio_runtime; ctx=rasterio_runtime(); ctx.__enter__(); print(rasterio.crs.CRS.from_epsg(4326).to_epsg()); ctx.__exit__(None,None,None)"
        completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "4326")


if __name__ == "__main__":
    unittest.main()
