# fortitudo.tech - Novel Investment Technologies.
# GPL-3.0-or-later

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .universe import load_universe_from_csv, load_universe_from_dataframe
from .market_data import fetch_prices, convert_to_base_currency, align_and_filter, compute_returns


def prepare_inputs_from_universe_df(
    universe_df: pd.DataFrame,
    base_currency: str = 'USD',
    start: Optional[str] = None,
    end: Optional[str] = None,
    returns: str = 'simple',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Prepare R (S×I), p (S×1), v (I,), and a normalized metadata DataFrame.

    This function fetches prices, converts to the base currency, aligns and computes returns.
    It does not run any optimization or equations beyond return computation.
    """
    _, meta = load_universe_from_dataframe(universe_df)
    symbols = meta['tradable_symbol'].tolist()
    # currency map keyed by tradable symbol
    instrument_currencies: Dict[str, str] = {}
    for _, row in meta.iterrows():
        instrument_currencies[row['tradable_symbol']] = (row['currency'] or base_currency)

    prices = fetch_prices(symbols, start=start, end=end)
    prices = convert_to_base_currency(prices, instrument_currencies, base_currency, start=start, end=end)
    prices = align_and_filter(prices)
    rets = compute_returns(prices, method=returns)

    # Scenario matrix R: S×I, in the column order of meta['tradable_symbol']
    R = rets[meta['tradable_symbol']].values
    S = R.shape[0]
    p = np.ones((S, 1)) / S
    v = np.ones((R.shape[1],))
    return R, p, v, meta


def prepare_inputs_from_universe_csv(
    csv_path: str,
    base_currency: str = 'USD',
    start: Optional[str] = None,
    end: Optional[str] = None,
    returns: str = 'simple',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    return prepare_inputs_from_universe_df(df, base_currency=base_currency, start=start, end=end, returns=returns)
