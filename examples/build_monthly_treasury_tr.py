from __future__ import annotations

from pathlib import Path
import pandas as pd

try:
    # Prefer normal import when package is installed/configured
    from fortitudo.tech.treasury_tr_model import build_monthly_tr_from_yields, TRModelConfig  # type: ignore
except Exception:
    # Fallback: import module directly from source to avoid heavy package __init__ side-effects
    import importlib.util
    import sys
    MOD_PATH = Path(__file__).resolve().parents[1] / "fortitudo" / "tech" / "treasury_tr_model.py"
    spec = importlib.util.spec_from_file_location("treasury_tr_model", str(MOD_PATH))
    if spec is None or spec.loader is None:  # pragma: no cover
        raise
    mod = importlib.util.module_from_spec(spec)
    sys.modules["treasury_tr_model"] = mod
    spec.loader.exec_module(mod)
    build_monthly_tr_from_yields = mod.build_monthly_tr_from_yields  # type: ignore
    TRModelConfig = mod.TRModelConfig  # type: ignore


DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> int:
    src = DATA / "fred_treasuries_timeseries.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}")
    df = pd.read_csv(src)
    # normalize index
    if "date" not in df.columns:
        raise SystemExit("fred_treasuries_timeseries.csv must have 'date' column")
    df["date"] = pd.to_datetime(df["date"]) 
    df = df.set_index("date")

    config = TRModelConfig(base_currency="USD")
    tr = build_monthly_tr_from_yields(df, config)
    out = DATA / "treasuries_returns_monthly.csv"
    tr.to_csv(out)
    print(f"Wrote {out} shape={tr.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
