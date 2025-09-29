# fortitudo.tech - Novel Investment Technologies.
# GPL-3.0-or-later

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def _import_yfinance():
    try:
        import yfinance as yf  # type: ignore
        return yf
    except Exception as e:
        raise RuntimeError(
            "yfinance is required to fetch market data. Please install it via pip install yfinance"
        ) from e


def fetch_prices(symbols: List[str], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    yf = _import_yfinance()
    data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False, group_by='ticker')
    # Normalize to a simple wide DF of Adjusted Close (already auto_adjust=True makes 'Close' total return-like)
    if isinstance(symbols, str):
        symbols = [symbols]
    frames = []
    for sym in symbols:
        if sym in data.columns.get_level_values(0):
            s = data[(sym, 'Close')].rename(sym)
        else:
            # Single symbol case yields a flat column index
            if 'Close' in data.columns:
                s = data['Close'].rename(sym)
            else:
                raise ValueError(f"Close price not found for {sym}")
        frames.append(s)
    prices = pd.concat(frames, axis=1)
    prices = prices.dropna(how='all')
    return prices


def fetch_fx_pairs(pairs: List[str], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame()
    yf = _import_yfinance()
    data = yf.download(pairs, start=start, end=end, auto_adjust=True, progress=False, group_by='ticker')
    frames = []
    for pair in pairs:
        if pair in data.columns.get_level_values(0):
            s = data[(pair, 'Close')].rename(pair)
        else:
            if 'Close' in data.columns:
                s = data['Close'].rename(pair)
            else:
                raise ValueError(f"Close price not found for FX pair {pair}")
        frames.append(s)
    fx = pd.concat(frames, axis=1).dropna(how='all')
    return fx


def convert_to_base_currency(prices: pd.DataFrame, instrument_currencies: Dict[str, str], base_currency: str,
                             start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    base = base_currency.upper()
    needed_pairs: List[str] = []
    # yfinance FX tickers are like 'EURUSD=X'
    for sym in prices.columns:
        cur = instrument_currencies.get(sym, base)
        if cur is None or str(cur).upper() == base:
            continue
        pair = f"{cur.upper()}{base}=X"
        needed_pairs.append(pair)
    needed_pairs = sorted(list(set(needed_pairs)))
    if needed_pairs:
        fx = fetch_fx_pairs(needed_pairs, start=start, end=end)
        # Align on dates
        prices = prices.copy()
        for sym in prices.columns:
            cur = instrument_currencies.get(sym, base)
            if cur is None or str(cur).upper() == base:
                continue
            pair = f"{cur.upper()}{base}=X"
            if pair not in fx.columns:
                raise ValueError(f"Missing FX rate for {sym} {cur}->{base}: expected {pair}")
            # Multiply local price by FX to get base currency
            prices[sym] = (prices[sym]).reindex(fx.index).fillna(method='ffill') * fx[pair]
        # Drop leading rows potentially introduced by reindex
        prices = prices.dropna(how='all')
    return prices


def compute_returns(prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
    if method == 'simple':
        ret = prices.pct_change().dropna(how='any')
    elif method == 'log':
        ret = np.log(prices).diff().dropna(how='any')
    else:
        raise ValueError("method must be 'simple' or 'log'")
    return ret


def align_and_filter(prices: pd.DataFrame, min_coverage: float = 0.95) -> pd.DataFrame:
    # Keep only dates where most instruments have data, and drop columns with too many NaNs
    col_nonnull = prices.notna().mean(axis=0)
    keep_cols = col_nonnull[col_nonnull >= min_coverage].index.tolist()
    filtered = prices[keep_cols]
    filtered = filtered.dropna(how='any')
    return filtered
