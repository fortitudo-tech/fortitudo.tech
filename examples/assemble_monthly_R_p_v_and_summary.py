from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


DATA = Path(__file__).resolve().parents[1] / "data"


def _clean_daily_returns(ret_daily: pd.DataFrame) -> pd.DataFrame:
    # Remove obvious sentinel/outliers (e.g., 999, -0.999) and extreme daily moves
    r = ret_daily.copy()
    mask_extreme = r.abs() > 0.5  # daily |r| > 50% -> NaN
    r[mask_extreme] = np.nan
    return r


def monthly_from_daily_returns(ret_daily: pd.DataFrame) -> pd.DataFrame:
    # Compound daily simple returns into month-end total return, ignoring NaNs
    ret_daily.index = pd.to_datetime(ret_daily.index)
    r = _clean_daily_returns(ret_daily)
    gross = (1.0 + r).where(r.notna(), 1.0)
    g = gross.resample("ME").prod() - 1.0
    return g.dropna(how="all")


def main() -> int:
    etf_path = DATA / "etfs_returns.csv"
    treas_path = DATA / "treasuries_returns_monthly.csv"
    if not etf_path.exists():
        raise SystemExit(f"Missing {etf_path}")
    if not treas_path.exists():
        raise SystemExit(f"Missing {treas_path}. Run examples/build_monthly_treasury_tr.py first.")

    etf = pd.read_csv(etf_path)
    etf["Date"] = pd.to_datetime(etf["Date"]) 
    etf = etf.set_index("Date")
    etf_m = monthly_from_daily_returns(etf)
    # Restrict to user window
    etf_m = etf_m.loc[etf_m.index >= pd.Timestamp("2016-01-01")]

    tre_m = pd.read_csv(treas_path, index_col=0, parse_dates=True)
    tre_m = tre_m.loc[tre_m.index >= pd.Timestamp("2016-01-01")]

    # Align monthly indexes, inner join to keep months where both sides exist
    Rdf = etf_m.join(tre_m, how="inner").dropna(how="any")

    # Create R, p, v
    R = Rdf.values
    S = R.shape[0]
    p = np.ones((S, 1)) / S
    v = np.ones((R.shape[1],))

    # Basic summary
    mu = Rdf.mean()
    vol = Rdf.std()
    print(f"Monthly scenarios S={S}, instruments I={R.shape[1]}")
    print("Top 5 by mean monthly return:")
    print(mu.sort_values(ascending=False).head(5))
    print("Top 5 by volatility:")
    print(vol.sort_values(ascending=False).head(5))

    # Save artifacts
    R_out = DATA / "R_monthly.csv"
    p_out = DATA / "p_monthly.csv"
    v_out = DATA / "v_monthly.csv"
    meta_out = DATA / "meta_monthly_columns.txt"
    pd.DataFrame(R, index=Rdf.index, columns=Rdf.columns).to_csv(R_out)
    pd.DataFrame(p, index=Rdf.index, columns=["p"]).to_csv(p_out)
    pd.DataFrame(v, index=Rdf.columns, columns=["v"]).to_csv(v_out)
    meta_out.write_text("\n".join(Rdf.columns))
    print(f"Saved R to {R_out}, p to {p_out}, v to {v_out}, columns to {meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
