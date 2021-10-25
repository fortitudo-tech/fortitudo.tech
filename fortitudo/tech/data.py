# fortitudo.tech investment and risk technologies
# Copyright (C) 2021 Fortitudo Technologies ApS

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from pkgutil import get_data
from io import StringIO
import pandas as pd


def load_pnl() -> pd.DataFrame:
    """Function for loading the P&L simulation from Vorobets (2021).

    Returns:
        P&L simulation.
    """
    pnl_string = StringIO(get_data('fortitudo.tech', 'data/pnl.csv').decode())
    return pd.read_csv(pnl_string)
