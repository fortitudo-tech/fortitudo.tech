"""
Monthly total return modeling for government bond series using yields.

Approach (constant-maturity approximation):
  TR_t ≈ carry_t + price_t
       ≈ (y_t / 12) - D * (Δy_t)

Where y is yield in decimal (e.g., 0.025 for 2.5%), D is an assumed Macaulay/modified duration proxy per tenor,
and Δy_t = y_t - y_{t-1} at monthly frequency. Convexity and roll are ignored for simplicity.

This is suitable for long-term government benchmark yields (e.g., FRED DGS* for US, IRLTLT01* for non-US).
Non-USD series are converted to the base currency via monthly FX returns using yfinance pairs like 'EURUSD=X'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import re
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class TRModelConfig:
    base_currency: str = "USD"
    # Duration assumptions by tenor in years
    duration_map: Dict[int, float] = None  # type: ignore

    def __post_init__(self):
        if self.duration_map is None:
            object.__setattr__(
                self,
                "duration_map",
                {
                    1: 1.0,
                    2: 2.0,
                    3: 2.8,
                    5: 4.5,
                    7: 6.5,
                    10: 9.0,
                    20: 17.0,
                    30: 21.0,
                },
            )


def _infer_tenor_from_name(col: str) -> Optional[int]:
    # Match DGS\d+ to capture tenor in years
    m = re.search(r"DGS(\d+)", col)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    # Long-term government yields (IRLTLT01*) – assume 10y
    if "IRLTLT01" in col:
        return 10
    return None


def _infer_country_currency(col: str) -> Tuple[str, str]:
    # Return (country_code, currency)
    # United States -> USD
    if "United_States" in col or "DGS" in col:
        return ("US", "USD")
    # United Kingdom -> GBP
    if "United_Kingdom" in col or "GBM" in col:
        return ("UK", "GBP")
    # Germany/Deutschland -> EUR
    if "Deutschland" in col or "DEM" in col:
        return ("DE", "EUR")
    # France -> EUR
    if "Trésor" in col or "FRM" in col:
        return ("FR", "EUR")
    # Italy -> EUR
    if "Tesoro" in col or "ITM" in col:
        return ("IT", "EUR")
    # Netherlands/Dutch -> EUR
    if "Dutch" in col or "NLM" in col:
        return ("NL", "EUR")
    # Fallback unknown -> assume USD
    return ("UNK", "USD")


def _fx_pair(local: str, base: str) -> Optional[str]:
    local = local.upper()
    base = base.upper()
    if local == base:
        return None
    return f"{local}{base}=X"


def _import_yfinance():
    try:
        import yfinance as yf  # type: ignore
        return yf
    except Exception as e:
        raise RuntimeError(
            "yfinance is required for FX conversion; install via pip install yfinance"
        ) from e


def build_monthly_tr_from_yields(yield_df: pd.DataFrame, config: Optional[TRModelConfig] = None) -> pd.DataFrame:
    if config is None:
        config = TRModelConfig()
    df = yield_df.copy()
    # Normalize and resample to month-end, taking last observation in month
    df.index = pd.to_datetime(df.index)
    m = df.resample("M").last()

    # Compute yields in decimals
    y = m.astype(float) / 100.0

    # Build per-column tenor and duration
    tenors = {c: _infer_tenor_from_name(c) for c in y.columns}
    durations = {c: config.duration_map.get(tenors[c], 9.0) for c in y.columns}

    # Δy per column
    dy = y.diff()

    # carry and price components
    carry = y / 12.0
    price = pd.DataFrame({c: -durations[c] * dy[c] for c in y.columns})

    tr_local = carry + price
    tr_local = tr_local.dropna(how="all")

    # FX conversion for non-USD series
    base = config.base_currency.upper()
    cols = list(tr_local.columns)
    fx_need: Dict[str, str] = {}
    col_ccy: Dict[str, str] = {}
    for c in cols:
        _, ccy = _infer_country_currency(c)
        col_ccy[c] = ccy
        pair = _fx_pair(ccy, base)
        if pair:
            fx_need[pair] = c

    if fx_need:
        yf = _import_yfinance()
        fx_pairs = sorted(set(fx_need.keys()))
        fx_px = yf.download(fx_pairs, auto_adjust=True, progress=False, group_by="ticker")
        # Extract Close for each pair and resample monthly last
        fx_frames = []
        for pair in fx_pairs:
            if isinstance(fx_px.columns, pd.MultiIndex):
                if pair in fx_px.columns.get_level_values(0):
                    s = fx_px[(pair, "Close")].rename(pair)
                else:
                    continue
            else:
                if "Close" in fx_px.columns:
                    s = fx_px["Close"].rename(pair)
                else:
                    continue
            fx_frames.append(s)
        if fx_frames:
            fx = pd.concat(fx_frames, axis=1)
            fx_m = fx.resample("M").last()
            fx_ret = fx_m.pct_change().dropna(how="all")
        else:
            fx_ret = pd.DataFrame(index=tr_local.index)
    else:
        fx_ret = pd.DataFrame(index=tr_local.index)

    # Merge FX returns onto TR and convert
    tr_usd = tr_local.copy()
    for c in cols:
        ccy = col_ccy.get(c, base)
        pair = _fx_pair(ccy, base)
        if not pair or pair not in fx_ret.columns:
            continue
        # align indexes
        aligned = tr_usd[[c]].join(fx_ret[[pair]], how="left").dropna()
        rl = aligned[c]
        rf = aligned[pair]
        tr_usd.loc[aligned.index, c] = (1.0 + rl) * (1.0 + rf) - 1.0

    return tr_usd
