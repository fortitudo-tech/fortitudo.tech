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

from context import np, pytest, time_series, call_option, put_option

tol = 1e-8

i = np.random.randint(0, len(time_series))
r = time_series['1y'][i] / 100
T = 1
S_0 = time_series['Equity Index'][i]
disc_factor = np.exp(-r * T)
F = 1 / disc_factor * S_0
sigma = time_series['1y100'][i] / 100
sigma_105 = time_series['1y105'][i] / 100

call = call_option(F, F, sigma, r, T)
put = put_option(F, F, sigma, r, T)
call_105 = call_option(F, 1.05 * F, sigma_105, r, T)
put_105 = put_option(F, 1.05 * F, sigma_105, r, T)


@pytest.mark.parametrize("call_price, strike", [(call, 1), (call_105, 1.05)])
def test_call_bounds(call_price, strike):
    assert max(S_0 - disc_factor * strike * F, 0) <= call_price <= S_0


@pytest.mark.parametrize("put_price, strike", [(put, 1), (put_105, 1.05)])
def test_put_bounds(put_price, strike):
    strike_pv = disc_factor * strike * F
    assert max(strike_pv - S_0, 0) <= put_price <= strike_pv


@pytest.mark.parametrize(
    "put_price, call_price, strike", [(put, call, 1), (put_105, call_105, 1.05)])
def test_parity(put_price, call_price, strike):
    call_parity = put_price + S_0 - disc_factor * strike * F
    assert np.abs(call_parity - call_price) <= tol
