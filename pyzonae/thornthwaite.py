# -*- coding: utf-8 -*-
# Copyright 2026 the pyZonae authors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# without warranties or conditions of any kind. See the License for the
# specific language governing permissions and limitations under the License.
"""
Thornthwaite potential evapotranspiration and the Willmott-Feddema moisture index.

These feed the Thornthwaite-Feddema classification (Feddema 2005). PE is the
temperature-based Thornthwaite (1948) formula with the Willmott et al. (1985)
high-temperature correction, which removes the discontinuity of the original
table above 26 degC. Day length comes from :mod:`pyzonae.orbital`, so the whole
thing is palaeo-ready: pass orbital parameters and PE follows.

Thornthwaite PE (mm/month), with T the monthly mean temperature (degC):

    T < 0        : PE = 0
    0 <= T <= 26 : PE = 16 * (N/360) * (10 T / I) ** a
    T > 26       : PE = (N/360) * (-415.85 + 32.24 T - 0.43 T**2)   [Willmott 1985]

where
    I = sum_months (max(0, T_m) / 5) ** 1.514          (annual heat index)
    a = 6.75e-7 I**3 - 7.71e-5 I**2 + 1.792e-2 I + 0.49239
    N = monthly mean day length in hours, times the number of days, folded into
        the (N/360) factor (N/12 for the hours, days/30 for the month length).

Coefficients cross-checked against Wikipedia, Aschonitis et al. (2022) and the
WSL fire-weather reference; the T>26 branch joins the T<=26 branch continuously
(verified in the tests).
"""

import numpy as np

from .orbital import monthly_daylength, OrbitalParameters

_DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                          dtype=float)


def heat_index(monthly_t):
    """Annual heat index I from 12 monthly mean temperatures (degC)."""
    t = np.asarray(monthly_t, dtype=float)
    contrib = np.where(t > 0.0, (np.maximum(t, 0.0) / 5.0) ** 1.514, 0.0)
    return np.sum(contrib, axis=0)


def _exponent_a(I):
    return (6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239)


def potential_evapotranspiration(monthly_t, lats, orb=None):
    """Monthly Thornthwaite PE (mm/month) for a (12, ...) temperature stack.

    Parameters
    ----------
    monthly_t : array (12, ...) of monthly mean temperature (degC)
    lats : array broadcastable to the trailing shape, latitudes in degrees
    orb : OrbitalParameters or None
        Orbit used for day length. None -> present day.

    Returns
    -------
    array (12, ...) of PE in mm/month.
    """
    if orb is None:
        orb = OrbitalParameters()
    t = np.asarray(monthly_t, dtype=float)

    I = heat_index(t)
    a = _exponent_a(I)

    # Day length per month (12, ...), matching the temperature grid.
    N = monthly_daylength(lats, orb)          # hours, shape (12, ...)
    N = np.broadcast_to(N, t.shape)

    days = _DAYS_IN_MONTH.reshape((12,) + (1,) * (t.ndim - 1))
    # Thornthwaite's correction factor: (N / 12) * (days / 30).
    corr = (N / 12.0) * (days / 30.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        # Base branch, 0 <= T <= 26.
        base = 16.0 * corr * (10.0 * np.clip(t, 0, None) / I) ** a
        # High-temperature branch, T > 26 (Willmott 1985).
        hot = corr * (-415.85 + 32.24 * t - 0.43 * t**2)

    pe = np.where(t < 0.0, 0.0, np.where(t > 26.0, hot, base))
    # I == 0 (everywhere at or below freezing) -> PE is 0, not NaN.
    pe = np.where(np.isfinite(pe), pe, 0.0)
    return pe


def annual_pe(monthly_t, lats, orb=None):
    """Annual total PE (mm/yr): the Thornthwaite thermal factor."""
    return np.sum(potential_evapotranspiration(monthly_t, lats, orb), axis=0)


def moisture_index(P_ann, PE_ann):
    """Willmott & Feddema (1992) moisture index, in [-1, 1].

        Im = 1 - PE/P   if P > PE
        Im = 0          if P == PE
        Im = P/PE - 1   if P <= PE
    """
    P = np.asarray(P_ann, dtype=float)
    PE = np.asarray(PE_ann, dtype=float)
    out = np.full(np.broadcast(P, PE).shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        wet = P > PE
        out = np.where((P > PE) & (P > 0), 1.0 - PE / P, out)
        out = np.where(P == PE, 0.0, out)
        out = np.where((P <= PE) & (PE > 0), P / PE - 1.0, out)
    # PE == 0 and P == 0 -> undefined; leave as 0 (no demand, no supply).
    out = np.where((P == 0) & (PE == 0), 0.0, out)
    return out


def monthly_moisture_index(monthly_p, monthly_t, lats, orb=None):
    """Per-month Im, needed later for the seasonality factor."""
    pe = potential_evapotranspiration(monthly_t, lats, orb)
    p = np.asarray(monthly_p, dtype=float)
    return moisture_index(p, pe)
