# Example: Prepare inputs from a user-provided universe CSV without running optimization

import pandas as pd
import fortitudo.tech as ft

CSV_PATH = "data/universe.csv"  # replace with your uploaded CSV path

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    R, p, v, meta = ft.prepare_inputs_from_universe_df(
        df,
        base_currency="USD",
        start="2016-01-01",
        end=None,  # up to latest available
        returns="simple",  # or "log"
    )
    print("Prepared inputs (no optimization run):")
    print("R shape:", R.shape)
    print("p shape:", p.shape)
    print("v shape:", v.shape)
    print("Universe metadata columns:", list(meta.columns))
    print(meta.head())
