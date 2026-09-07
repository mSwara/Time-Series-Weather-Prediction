"""Fetch the large-scale (~14,610-day) Kolkata daily weather dataset used in
the scaled-up extension of this project (notebooks 08+).

Source: Open-Meteo Historical Weather API (https://open-meteo.com/en/docs/historical-weather-api),
an ERA5/ERA5-Land reanalysis-backed archive. Free, no API key required.

Why this source, and how it differs from the original 300-row dataset:
The original dataset's `specific_humidity_500hPa`, `geopotential_height_500hPa`,
`Wind_500hPa_ms`, and explicit heat-flux terms come from ERA5's pressure-level
product via the Copernicus Climate Data Store (CDS), which requires a
registered API key. This script instead uses Open-Meteo's free surface-level
archive, which provides a different (though related) set of daily variables:
2m temperature (max/min/mean), 2m wet-bulb temperature, 2m relative humidity,
10m wind speed, cloud cover, shortwave radiation, sunshine duration,
precipitation, surface pressure, vapour pressure deficit, and reference
evapotranspiration. This is a deliberate, documented substitution — not a
silent one — made to get real, reproducible data at scale without requiring a
CDS account. See reports/large_scale_extension.md for the full rationale.

Usage: python -m src.fetch_large_data
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

LAT, LON = 22.57, 88.36  # Kolkata
START_DATE, END_DATE = "1980-01-01", "2019-12-31"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "wet_bulb_temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "cloud_cover_mean",
    "shortwave_radiation_sum",
    "surface_pressure_mean",
    "et0_fao_evapotranspiration",
    "precipitation_sum",
    "vapour_pressure_deficit_max",
    "sunshine_duration",
]

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_large"
OUT_JSON = OUT_DIR / "kolkata_1980_2019.json"
OUT_CSV = OUT_DIR / "kolkata_1980_2019.csv"


def fetch() -> dict:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = fetch()
    OUT_JSON.write_text(json.dumps(data), encoding="utf-8")

    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "time": "Date",
        "temperature_2m_mean": "dbt",
        "wet_bulb_temperature_2m_mean": "wbt",
    })

    gaps = df["Date"].diff().dt.days.dropna()
    assert (gaps == 1).all(), "Fetched series is not strictly continuous - investigate before use"

    df.to_csv(OUT_CSV, index=False)
    print(f"Fetched {len(df)} continuous daily rows ({df.Date.min().date()} to {df.Date.max().date()})")
    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
