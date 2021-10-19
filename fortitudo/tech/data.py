from pkgutil import get_data
from io import StringIO
import pandas as pd


def load_pnl() -> pd.DataFrame:
    pnl_string = StringIO(get_data('fortitudo.tech', 'data/pnl.csv').decode())
    return pd.read_csv(pnl_string)
