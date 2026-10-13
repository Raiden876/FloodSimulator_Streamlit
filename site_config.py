"""Names and geographic reference points for configured hazard sites."""

import re

SITE_MAPPING = {
    "Metro Manila": {"archive": "MetroManila.zip", "center": (14.6091, 121.0223), "return_period_yr": 25},
    "Nueva Ecija": {"archive": "NuevaEcija.zip", "center": (15.50, 121.00), "return_period_yr": 25},
    "Pampanga": {"archive": "Pampanga.zip", "center": (15.08, 120.62), "return_period_yr": 25},
    "Bulacan": {"archive": "Bulacan.zip", "center": (14.90, 120.90), "return_period_yr": 25},
    "Tarlac": {"archive": "Tarlac.zip", "center": (15.48, 120.60), "return_period_yr": 25},
    "Aurora": {"archive": "Aurora.zip", "center": (15.75, 121.40), "return_period_yr": 25},
    "Zambales": {"archive": "Zambales.zip", "center": (15.17, 120.30), "return_period_yr": 25},
    "Bataan": {"archive": "Bataan.zip", "center": (14.75, 120.45), "return_period_yr": 25},
    "Pangasinan": {"archive": "Pangasinan.zip", "center": (15.90, 120.40), "return_period_yr": 25},
    "Nueva Vizcaya": {"archive": "NuevaVizcaya.zip", "center": (16.30, 121.20), "return_period_yr": 25},
    "Rizal": {"archive": "Rizal.zip", "center": (14.60, 121.20), "return_period_yr": 25},
    "Quezon": {"archive": "Quezon.zip", "center": (14.00, 121.60), "return_period_yr": 25},
    "South Cotabato": {"archive": "SouthCotabato.zip", "center": (6.30, 125.00), "return_period_yr": 25},
    "Sarangani": {"archive": "Sarangani.zip", "center": (6.20, 125.20), "return_period_yr": 25},
    "Davao del Sur": {"archive": "DavaoDelSur.zip", "center": (6.70, 125.30), "return_period_yr": 25},
    "Davao Occidental": {"archive": "DavaoOccidental.zip", "center": (6.00, 125.50), "return_period_yr": 100},
}


def site_key(site_name: str) -> str:
    """Return a stable lowercase folder name for a configured site."""
    return re.sub(r"[^a-z0-9]+", "_", site_name.lower()).strip("_")
