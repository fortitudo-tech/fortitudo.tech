"""
Backfill missing ETF returns using yfinance and merge them into data/etfs_returns.csv.

This script intentionally avoids conda; run it with any Python 3 venv:
  - python3 -m venv .venv
  - .venv/bin/python -m pip install --upgrade pip
  - .venv/bin/python -m pip install pandas yfinance
  - .venv/bin/python examples/backfill_missing_etfs.py

It will:
  - Read the authoritative 100-ETF list from data/etfs_invalid.csv (name is legacy; contents are the 100 approved tickers)
  - Compare against columns in data/etfs_returns.csv
  - Download adjusted close for missing tickers with yfinance
  - Compute simple returns, save them to data/etfs_returns_backfill.csv
  - Merge into data/etfs_returns.csv (back up old file to data/etfs_returns.backup.csv)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception as e:  # pragma: no cover - runtime check
    raise SystemExit(
        "yfinance is required. Install with: .venv/bin/python -m pip install yfinance"
    ) from e


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RETURNS_PATH = DATA_DIR / "etfs_returns.csv"
BACKUP_PATH = DATA_DIR / "etfs_returns.backup.csv"
BACKFILL_PATH = DATA_DIR / "etfs_returns_backfill.csv"
LIST_PATH = DATA_DIR / "etfs_invalid.csv"  # legacy filename; contains the 100 approved tickers


def load_authoritative_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing list file: {path}")
    s = pd.read_csv(path).iloc[:, 0].astype(str).str.strip().str.upper()
    s = s[s.ne("") & s.ne("NAN")]
    return s.tolist()


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RETURNS_PATH.exists():
        raise FileNotFoundError(f"Missing returns file: {RETURNS_PATH}")

    # Load canonical 100 list and current returns
    approved = load_authoritative_list(LIST_PATH)
    ret = pd.read_csv(RETURNS_PATH)
    if "Date" not in ret.columns:
        raise ValueError("etfs_returns.csv must have a 'Date' column")
    current_cols = {c.upper() for c in ret.columns if c != "Date"}

    missing = [t for t in approved if t not in current_cols]
    present = [t for t in approved if t in current_cols]
    print(f"Approved tickers: {len(approved)} | Present: {len(present)} | Missing: {len(missing)}")
    if not missing:
        print("No backfill needed.")
        return 0

    # Download adjusted close for missing tickers
    start = "2016-01-01"
    print(f"Downloading adjusted close for {len(missing)} tickers starting {start}...")
    px = yf.download(missing, start=start, auto_adjust=False, progress=False)  # type: ignore
    # yfinance returns a column MultiIndex for multi-ticker downloads
    if isinstance(px.columns, pd.MultiIndex):
        if ("Adj Close" in px.columns.get_level_values(0)):
            px = px["Adj Close"].copy()
        else:
            # Fallback: try Close if Adj Close not available
            level0 = px.columns.get_level_values(0)
            use = "Close" if ("Close" in level0) else level0[0]
            print(f"Adj Close not found, using '{use}' level for returns computation.")
            px = px[use].copy()
    else:
        # Single-ticker case: ensure DataFrame
        px = px.to_frame(name=missing[0])

    # Keep only requested tickers (yfinance may drop unknowns)
    available = [c for c in px.columns if isinstance(c, str)]
    missing_resolved = [t for t in missing if t in available]
    unresolved = [t for t in missing if t not in available]
    if unresolved:
        print(f"Warning: {len(unresolved)} tickers not returned by yfinance and will be skipped: {unresolved}")

    px = px[missing_resolved].sort_index()
    ret_backfill = px.pct_change().dropna(how="all")
    ret_backfill.index.name = "Date"

    # Save the backfill-only file
    BACKFILL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ret_backfill.to_csv(BACKFILL_PATH)
    print(f"Saved backfill-only returns to {BACKFILL_PATH} with shape {ret_backfill.shape}")

    # Merge into the main returns file by Date
    ret["Date"] = pd.to_datetime(ret["Date"])
    merged = ret.set_index("Date")

    # Ensure backfill index is datetime
    ret_backfill = ret_backfill.copy()
    ret_backfill.index = pd.to_datetime(ret_backfill.index)

    # Align on union of dates, preserving existing data
    merged = merged.join(ret_backfill, how="outer")
    merged = merged.sort_index()

    # Reorder columns: existing first, then newly added
    new_cols = [c for c in ret_backfill.columns if c not in ret.columns]
    ordered_cols = [c for c in ret.columns if c != "Date"] + new_cols
    merged = merged[ordered_cols]

    # Backup old file and write new
    RETURNS_PATH.replace(BACKUP_PATH) if RETURNS_PATH.exists() else None
    merged.reset_index().to_csv(RETURNS_PATH, index=False)
    print(f"Backed up old returns to {BACKUP_PATH}")
    print(f"Wrote merged returns to {RETURNS_PATH} with shape {merged.shape}")

    # Final coverage report
    final_cols = {c.upper() for c in merged.columns}
    final_present = [t for t in approved if t in final_cols]
    still_missing = [t for t in approved if t not in final_cols]
    print(
        f"Coverage after merge — Present: {len(final_present)}/100 | Still missing: {len(still_missing)}"
    )
    if still_missing:
        print(f"Still missing (no data from source): {still_missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
