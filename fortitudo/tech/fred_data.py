# fortitudo.tech - Novel Investment Technologies.
# GPL-3.0-or-later

from __future__ import annotations

import os
from typing import Optional, List, Dict
import requests
import pandas as pd

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED API key is required. Set FRED_API_KEY env var or pass api_key.")
        self.timeout = timeout

    def get_series(self, series_id: str, start: Optional[str] = None, end: Optional[str] = None,
                   frequency: Optional[str] = None, aggregation_method: Optional[str] = None) -> pd.Series:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if start:
            params["observation_start"] = start
        if end:
            params["observation_end"] = end
        # Optionally allow frequency/aggregation if user wants resampling server-side
        if frequency:
            params["frequency"] = frequency
        if aggregation_method:
            params["aggregation_method"] = aggregation_method

        r = requests.get(FRED_API_URL, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # observations: list of dicts with 'date' and 'value' (string; '.' for missing)
        obs = data.get("observations", [])
        if not obs:
            return pd.Series(dtype=float)
        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"])
        # Convert value to float; '.' becomes NaN
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("date")["value"].sort_index()
        s.name = series_id
        return s


def fetch_series_from_map(map_df: pd.DataFrame, api_key: Optional[str], start: str, end: Optional[str]) -> pd.DataFrame:
    """Fetch multiple FRED series given a mapping DataFrame.

    Expected columns (case-insensitive):
        - symbol: your instrument or alias name (output column name)
        - fred_series_id: the FRED series id to fetch
        - frequency (optional): FRED frequency override
        - aggregation_method (optional): FRED aggregation method

    Returns: wide DataFrame indexed by date, one column per symbol.
    """
    # Normalize columns
    colmap = {c.lower(): c for c in map_df.columns}
    def col(name: str) -> Optional[str]:
        return colmap.get(name)

    required = ["symbol", "fred_series_id"]
    for req in required:
        if col(req) is None:
            raise ValueError(f"Mapping is missing required column: {req}")

    client = FredClient(api_key=api_key)
    series: Dict[str, pd.Series] = {}
    for _, row in map_df.iterrows():
        symbol = row[col("symbol")]
        fred_id = row[col("fred_series_id")]
        freq = row[col("frequency")] if col("frequency") in map_df.columns else None
        aggr = row[col("aggregation_method")] if col("aggregation_method") in map_df.columns else None
        s = client.get_series(str(fred_id), start=start, end=end, frequency=freq, aggregation_method=aggr)
        if s.empty:
            # keep empty to signal missing; but still add column
            series[str(symbol)] = pd.Series(dtype=float, name=str(symbol))
        else:
            s = s.rename(str(symbol))
            series[str(symbol)] = s

    # Align on union of dates
    if not series:
        return pd.DataFrame()
    df = pd.concat(series.values(), axis=1)
    # Keep as-is; downstream can decide how to handle NaNs
    return df


def save_spreadsheet(df: pd.DataFrame, csv_path: str, xlsx_path: Optional[str] = None) -> None:
    df.to_csv(csv_path, index=True)
    if xlsx_path:
        try:
            import xlsxwriter  # noqa: F401
            with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="FRED_Data")
        except Exception:
            # Fallback: try openpyxl if available, else skip xlsx
            try:
                with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="FRED_Data")
            except Exception:
                pass
