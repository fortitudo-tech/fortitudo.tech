# fortitudo.tech - Novel Investment Technologies.
# GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd


@dataclass
class UniverseItem:
    symbol: Optional[str]
    name: Optional[str]
    type: Optional[str]  # 'ETF' | 'BOND'
    currency: Optional[str]  # e.g., 'USD', 'EUR'
    group: Optional[str]  # e.g., 'Equity', 'Treasury', 'IG', 'HY'
    proxy_symbol: Optional[str] = None
    weight_cap: Optional[float] = None

    @property
    def tradable_symbol(self) -> str:
        if self.symbol and isinstance(self.symbol, str) and len(self.symbol) > 0:
            return self.symbol.strip()
        if self.proxy_symbol and isinstance(self.proxy_symbol, str) and len(self.proxy_symbol) > 0:
            return self.proxy_symbol.strip()
        raise ValueError("Universe item missing both symbol and proxy_symbol: " + str(self))


def load_universe_from_dataframe(df: pd.DataFrame) -> Tuple[List[UniverseItem], pd.DataFrame]:
    required_cols = {"symbol", "type", "currency"}
    missing = required_cols - set(c.lower() for c in df.columns)
    # allow case-insensitive and optional fields; we will normalize
    # Build a mapping of lower->original columns
    colmap = {c.lower(): c for c in df.columns}

    def col(name: str) -> Optional[str]:
        return colmap.get(name)

    items: List[UniverseItem] = []
    for _, row in df.iterrows():
        items.append(
            UniverseItem(
                symbol=(row[col("symbol")] if col("symbol") in df.columns else None),
                name=(row[col("name")] if col("name") in df.columns else None),
                type=(row[col("type")] if col("type") in df.columns else None),
                currency=(row[col("currency")] if col("currency") in df.columns else None),
                group=(row[col("group")] if col("group") in df.columns else None),
                proxy_symbol=(row[col("proxy_symbol")] if col("proxy_symbol") in df.columns else None),
                weight_cap=(row[col("weight_cap")] if col("weight_cap") in df.columns else None),
            )
        )

    # Normalize to a clean DataFrame with expected casing
    norm_df = pd.DataFrame(
        [
            {
                "symbol": it.symbol,
                "name": it.name,
                "type": (str(it.type).upper() if pd.notna(it.type) and it.type else None),
                "currency": (str(it.currency).upper() if pd.notna(it.currency) and it.currency else None),
                "group": it.group,
                "proxy_symbol": it.proxy_symbol,
                "weight_cap": it.weight_cap,
                "tradable_symbol": it.tradable_symbol,
            }
            for it in items
        ]
    )
    return items, norm_df


def load_universe_from_csv(path: str) -> Tuple[List[UniverseItem], pd.DataFrame]:
    df = pd.read_csv(path)
    return load_universe_from_dataframe(df)
