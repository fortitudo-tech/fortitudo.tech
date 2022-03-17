# fortitudo.tech - Novel Investment Technologies.
# Copyright (C) 2021-2022 Fortitudo Technologies.

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

import numpy as np
from scipy.stats import norm
from typing import Tuple


def _d1_d2(F: float, K: float, sigma: float, T: float) -> Tuple[float, float]:
    """Function for computing the parameters d1 and d2 in Black's model.

    Args:
        F: Forward price for maturity T.
        K: Strike value.
        sigma: Implied volatility for maturity T and strike K.
        T: Time to maturity.

    Returns:
        d1 and d2.
    """
    d1 = (np.log(F / K) + sigma**2 * T / 2) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_option(F: float, K: float, sigma: float, r: float, T: float) -> float:
    """Function for computing European call option price using Black's formula.

    Args:
        F: Forward price for maturity T.
        K: Strike value.
        sigma: Implied volatility for maturity T and strike K.
        r: Interest rate for maturity T.
        T: Time to maturity.

    Returns:
        European call option price.
    """
    d1, d2 = _d1_d2(F, K, sigma, T)
    call_price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return call_price


def put_option(F: float, K: float, sigma: float, r: float, T: float) -> float:
    """Function for computing European put option price using Black's formula.

    Args:
        F: Forward price for maturity T.
        K: Strike value.
        sigma: Implied volatility for maturity T and strike K.
        r: Interest rate for maturity T.
        T: Time to maturity.

    Returns:
        European put option price.
    """
    d1, d2 = _d1_d2(F, K, sigma, T)
    put_price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return put_price
