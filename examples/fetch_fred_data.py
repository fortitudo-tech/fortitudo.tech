import os
import pandas as pd
from fortitudo.tech.fred_data import fetch_series_from_map, save_spreadsheet

MAPPING_CSV = "examples/fred_mapping_template.csv"  # replace with your populated mapping
OUT_CSV = "data/fred_timeseries.csv"
OUT_XLSX = "data/fred_timeseries.xlsx"
START = "2016-01-01"
END = None  # to latest

if __name__ == "__main__":
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        # The user shared a key; prefer env var to avoid hardcoding
        # Set FRED_API_KEY in your environment before running this script
        raise SystemExit("Please set FRED_API_KEY in your environment.")

    mapping = pd.read_csv(MAPPING_CSV)
    df = fetch_series_from_map(mapping, api_key=api_key, start=START, end=END)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    save_spreadsheet(df, OUT_CSV, OUT_XLSX)
    print("Saved:", OUT_CSV)
    if os.path.exists(OUT_XLSX):
        print("Saved:", OUT_XLSX)
