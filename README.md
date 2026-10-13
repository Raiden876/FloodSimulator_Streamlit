# Flood simulator

The Streamlit app models event runoff and flood exposure for Metro Manila and the configured Philippine provinces. Run it from this directory:

~~~powershell
.\.venv\Scripts\python.exe -m streamlit run simulator.py
~~~

If the running app reports that it cannot import `rainfall_runoff_analysis`, stop its terminal process with Ctrl+C and restart it with the command above. A browser refresh keeps the current Python process alive. The app also restores Rasterio's bundled PROJ database at runtime; start it with this project's `.venv` interpreter.

## Using the app

Choose the site and rainfall amount, duration, and prior five-day rainfall in the sidebar. The main page shows four results and one map. **Map layer** switches among flood depth, affected population, population density, and the independent UP NOAH reference. There are no map tabs.

Optional sections below the map show population details, rainfall/runoff analysis, baseline comparisons, and model assumptions. Advanced model settings and soil inputs stay collapsed in the sidebar.

**Rainfall and runoff analysis** contains two separate plots: rainfall intensity bars (mm/h) and outlet discharge (m³/s). Both use the same elapsed-time axis and mark the end of rainfall. The runoff graph includes the recession. Peak lag is estimated from the midpoint of a unique peak-rainfall interval, with the interval's timing uncertainty reported as a range. Uniform/tied rainfall peaks and zero-flow cases do not receive an invented lag.

## Model choices

| Method | Runoff inputs | Flood depth |
| --- | --- | --- |
| Corrected empirical baseline | Original hazard-associated CN 78/88/98; moisture adjustment once | Original vectorized depth relationship, preserved for comparison |
| Soil and routed runoff (default) | Joint soil group and land cover; cumulative SCS-CN excess; linear reservoir routing | Empirical scaling of NOAH reference depths using peak runoff storage |
| Terrain inundation | Joint soil group and land cover on a projected grid; time-dependent excess | D8 terrain transport and local depression storage, independent of NOAH hazard class |

Polygon runoff inputs summarize soil and land cover into an area-weighted curve number for each hazard polygon. The default 120 m composite grid automatically coarsens for large study areas to stay within a 150,000-cell processing limit; the effective resolution appears in the results. This composite drives the empirical routed-depth model and is not a cell-by-cell flood-depth map.

The corrected scalar and vectorized baseline share the same effective-CN function. An explicit nonmissing CN_amc is already adjusted and is used directly. Missing CN_amc values fall back to an adjustment of CN_base. The original vectorized depths are captured in tests/fixtures/legacy_baseline.json.

The terrain model uses priority-flood conditioning for acyclic drainage connectivity while retaining original DEM elevations as the depth datum. Water above local depression capacity travels to the receiving cell using a bounded Manning velocity approximation. Drainage removes at most the available water. Every run reports generated runoff, boundary outflow, removed drainage, final storage, and balance error.

This is an inundation **screening model**, not a calibrated hydraulic forecast. It does not solve water-surface gradients, backwater, river/coastal stages, sewer hydraulics, floodplain spreading, or inflow beyond the DEM. Depression storage is local to each cell rather than an equilibrated water surface over an entire basin. Raw DEM pits, coarse terrain, roughness, imperviousness, and drainage assumptions can strongly affect depth. A full hydraulic solver and independent event validation remain later work.

## Downloaded spatial inputs

Site-specific ESA WorldCover and HiHydroSoil subsets are stored under `data/hydrology/sites/<site>/` for all 16 configured areas. The app selects them automatically when you choose a site and use a soil or terrain model. The subsets cover each site's UP NOAH flood-polygon bounding envelope; they do not cover every part of each province. The rectangular envelope may include gaps between polygons, but the simulator samples only its hazard geometries.

| Input | Source and resolution | Coverage |
| --- | --- | --- |
| `data/hydrology/sites/<site>/worldcover.tif` | [ESA WorldCover 2021 v200](https://esa-worldcover.org/en/data-access), native 10 m, resampled to 30 m | Each hazard-polygon bounding envelope |
| `data/hydrology/sites/<site>/hsg.tif` | [NatCap HiHydroSoil v2.0 derivative](https://data.naturalcapitalalliance.stanford.edu/dataset/sts-cdd01bf6db49adf846783f7d47d9ca1186c9b3143df283e9539a2d8969c07ecb), about 250 m | Each hazard-polygon bounding envelope; gaps remain nodata |
| `qc_dem_srtm.tif` | Local SRTM-derived DEM, about 30 m | Partial Metro Manila coverage only |

The HSG groups are global model estimates, not measured local soil profiles. [FutureWater's source description](https://www.futurewater.eu/solutions/hihydrosoil/) describes the soil hydraulic properties behind HiHydroSoil. Both downloaded products use CC BY 4.0. The source manifest records citations, URLs, retrieval times, bounds, tile IDs, classes, output resolution, and SHA-256 hashes.

Citations: Zanaga et al. (2022), ESA WorldCover 10 m 2021 v200, doi:10.5281/zenodo.7254221. Simons, G.W.H., R. Koster, P. Droogers (2020), HiHydroSoil v2.0, FutureWater Report 213; the HSG raster is Natural Capital Alliance's reclassified derivative.

The soil and cover inputs now follow all 16 configured sites, including the Central Luzon provinces and the four South Cotabato/Sarangani/Davao areas. The current DEM still covers only part of Metro Manila, so terrain inundation outside that raster remains unavailable until matching elevation data is supplied. Missing soil or cover cells use the explicit fallback profile and are counted in coverage diagnostics. The default profile is HSG D, built-up cover, 65% built-up imperviousness, and zero assumed drainage. Buildings, engineered drainage capacity, and flood observations have not been supplied.

## Downloading or refreshing site inputs

Run this from the project root to prepare or refresh all configured sites:

~~~powershell
.venv/Scripts/python.exe scripts/download_hydrology_inputs.py
~~~

To download only one area, add --site Quezon; repeat --site to choose several. Add --only land_cover or --only soil for one dataset, or --force to replace existing subsets. The downloader reads each site's hazard archive, crops to its mapped bounding envelope, uses public Cloud-Optimized GeoTIFF range reads, and does not fetch whole global rasters. WorldCover's 10 m data is resampled to 30 m; the 250 m soil groups remain 250 m. The per-site CRS is EPSG:32651.

## Soil and land-cover configuration

Use the controls under Advanced model settings, or optionally copy examples/hydrology_inputs.json to data/hydrology/inputs.json. Relative paths resolve against the project directory. Every raster must have a valid CRS, one band, and explicit nodata for missing cells.

| Input | Encoding/units |
| --- | --- |
| Land cover | ESA WorldCover codes 10/20/30/40/50/60/70/80/90/95/100 |
| HSG | 1=A, 2=B, 3=C, 4=D; optional 5=A/D, 6=B/D, 7=C/D |
| Ksat 0–50 and 0–100 cm | Minimum saturated conductivity over the indicated profile, µm/s |
| Restrictive layer | Depth in cm |
| Seasonal water table | Depth in cm |
| Imperviousness | Fraction 0–1, not percentage |
| Drainage capacity | mm/h over a model cell |
| DEM and observed depths | Metres |

A supplied HSG takes precedence. Where HSG is missing, conductivity, restrictive-depth, and water-table inputs can classify undrained HSG using [NRCS NEH 630 chapter 7](https://directives.nrcs.usda.gov/sites/default/files2/1712930597/11905.pdf). The relevant shallow/deep conductivity limits are selected by profile depth. Texture alone is not used to assign HSG. Dual HSG defaults to undrained D unless adequate drainage is explicitly documented.

The joint land-cover/HSG lookup uses [TR-55 CN tables published by HEC-HMS](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/cn-tables). WorldCover categories are mapped to representative woods, brush, pasture, crops, bare soil, and built-up cover. Built-up CN blends pervious open space with paved CN=98 using the selected impervious fraction. Wetlands/mangroves and moss/ice categories use documented screening proxies; the crosswalk and cover condition require local calibration. In polygon mode, joint cell CN values are averaged within each polygon as an approximate composite CN.

## Time-dependent rainfall

Choose a uniform or early/middle/late peak scenario, or upload a CSV with consecutive rain_mm and interval_h columns. rain_mm is depth during the interval, not intensity or cumulative rainfall. CSV totals and duration override the amount/duration sliders. Irregular interval lengths are supported; invalid, missing, negative, or nonfinite inputs are rejected.

examples/rainfall_uniform_150mm_6h.csv and examples/rainfall_burst_150mm_6h.csv both contain 150 mm over six hours. Fixed-CN event runoff has the same total volume, but its timing and routed peak differ. Cumulative excess differences follow the [HEC-HMS SCS-CN event formulation](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/canopy-surface-infiltration-and-runoff-volume/infiltration/scs-curve-number-loss-model); losses are not reset each interval. SCS-CN itself remains a cumulative event-loss model, not intensity-sensitive Green–Ampt infiltration.

## Exposure and independent validation

flood_model.py centralizes the 0.05 m map/area threshold and the 0.10 m population-exposure threshold. Affected population and exposure sensitivity use the same depth surface and disjoint population raster cells. Shallow modeled cells remain distinguishable from cells outside terrain coverage. Exposure is unknown outside DEM coverage; those people are not labeled dry or affected.

Sensitivity is reported at 0.05/0.10/0.30/0.50 m as illustrative thresholds, not universal damage cutoffs. The population source is a 2026 estimate, so counts should not be interpreted as observed disaster impacts. Existing baseline area statistics retain source polygon areas and may count overlaps; terrain area uses nonoverlapping reporting cells.

For terrain mode, provide a single-band independent observed-depth GeoTIFF for the same event under Advanced model settings. Validation excludes nodata/negative/missing observations and uses the common valid DEM/reporting footprint. IoU/CSI, depth MAE, and RMSE are reported for both the upgraded surface and preserved baseline. IoU and CSI are the same binary inundation score; scores with no wet union remain undefined. Nearest-neighbor alignment is used, so observation resolution and grid alignment limit comparisons.

No observed flood event has been downloaded or used to establish predictive accuracy. Terrain-to-baseline comparisons use a common coverage mask rather than comparing whole-region baseline population with a partial DEM domain.

Downloads in the app include rainfall intervals, the runoff hydrograph, sensitivity counts, and a peak-depth GeoTIFF with -9999 nodata outside terrain coverage. A maximum-depth map summarizes different peak times across cells, not one simultaneous water surface.

## PROJ database repair

geo_runtime.py restores Rasterio's matching bundled proj.db and GDAL data paths in each Streamlit worker thread and during raster operations. This repairs a missing search path after hot reload or an invalid inherited PROJ environment without changing global Windows environment variables. See [Rasterio's PROJ FAQ](https://rasterio.readthedocs.io/en/stable/faq.html) and [PROJ's resource search paths](https://proj.org/en/stable/resource_files.html). If the actual proj.db file is missing from the installed package, the app reports that Rasterio must be reinstalled in the project environment.

## Verification

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_bundled_terrain.py --output reports/bundled_terrain_check.json
~~~

The suite checks the captured baseline, scalar/vector agreement, partial effective CN input, AMC boundaries, rainfall totals, storm timing, reservoir conservation, actual raster alignment and encodings, D8 cycles, terrain conservation, depression storage, exposure thresholds, missing coverage, GeoTIFF exports, PROJ path recovery, all map choices, collapsed/lazy app sections, independent validation, and separate rainfall/runoff plots with peak-lag uncertainty.

The bundled DEM smoke check uses local downloaded inputs when available. --assumed-inputs reproduces the assumption-only run. reports/bundled_terrain_check.json is a conservation check, not an observed accuracy assessment.
