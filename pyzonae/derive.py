# -*- coding: utf-8 -*-
# Copyright 2026 the pyzonae authors
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
Derive the climate indices needed by every classifier from monthly T and P.

The output is a stack of 15 fields (the "arguments" vector), laid out so that
indices 0-12 match the pyKoeppen convention and indices 13-14 add the two extra
variables the Defaut scheme needs:

    0: T_min    coldest-month mean T2m             (C)
    1: T_max    warmest-month mean T2m             (C)
    2: T_mon    number of months with T >= 10 C    (count)
    3: T_ann    annual mean T2m                    (C)
    4: P_min    precipitation of driest month      (mm/month)
    5: P_ann    annual precipitation               (mm/year)
    6: P_smin   min summer-half-year precip        (mm/month)
    7: P_smax   max summer-half-year precip        (mm/month)
    8: P_wmin   min winter-half-year precip        (mm/month)
    9: P_wmax   max winter-half-year precip        (mm/month)
   10: P_th     dryness threshold                  (mm)
   11: P_wpro   winter fraction of annual precip   (-)
   12: P_dry    number of dry months (P <= 60)     (count)
   13: P_ann_mm annual precipitation               (mm)  [alias of 5, explicit]
   14: P_sec3   driest 3 consecutive months (Gaussen, mm)

Summer/winter half-years follow pyKoeppen: Apr-Sep is "summer" in the northern
hemisphere and is swapped with the winter slice south of the equator.
"""

import numpy as np
from numpy import ma

N_ARGS = 15


def _maxminsum_slice(mon, lo, hi):
    """Max, min, sum over a wrapped month slice; mon has month as axis 0."""
    doubled = np.concatenate([mon, mon], axis=0)
    sl = doubled[12 + lo:12 + hi, ...]
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmax(sl, axis=0), np.nanmin(sl, axis=0), np.nansum(sl, axis=0)


def _gaussen_driest3(pr, tas_kelvin):
    """Cumulative precip of the 3 consecutive months with lowest P/T (Gaussen).

    The Gaussen aridity ratio S = P / T must use temperature on an *absolute*
    scale (Kelvin): S is only meaningful for T > 0, and Kelvin guarantees that
    everywhere on Earth, so no ad-hoc masking of cold months is needed. Using
    Celsius here would make S negative below 0 C and wrongly flag cold months as
    "driest".

    The driest 3-consecutive-month window is identified and *that same window*
    is summed (Defaut's intent: total precipitation of the three driest months).
    The search wraps around the year, so windows spanning December-January
    (e.g. a boreal-winter dry season) are considered too.

    Parameters
    ----------
    pr : ndarray (month, lat, lon)
        Monthly precipitation (mm/month).
    tas_kelvin : ndarray (month, lat, lon)
        Monthly mean temperature in Kelvin.

    Returns
    -------
    ndarray (lat, lon)
        Precipitation accumulated over the driest 3-month window.
    """
    nt = pr.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        S = pr / tas_kelvin           # Kelvin -> always positive, no guard needed
    best_start = np.zeros(pr.shape[1:], dtype=int)
    running_best = np.full(pr.shape[1:], np.inf)
    for start in range(nt):
        idx = [(start + k) % nt for k in range(3)]
        s3 = S[idx[0]] + S[idx[1]] + S[idx[2]]
        better = s3 < running_best
        running_best = np.where(better, s3, running_best)
        best_start = np.where(better, start, best_start)
    # Accumulate precipitation over the identified driest window.
    acc = np.zeros(pr.shape[1:])
    for k in range(3):
        m = (best_start + k) % nt
        acc += np.take_along_axis(pr, m[None, ...], axis=0)[0]
    return acc


def build_arguments(fields):
    """Compute the (N_ARGS, lat, lon) stack of derived indices.

    Parameters
    ----------
    fields : pyzonae.io.ClimateFields

    Returns
    -------
    args : numpy.ma.MaskedArray, shape (15, nlat, nlon)
    shape : tuple, the (nlat, nlon) grid shape
    lats, lons : coordinate arrays
    """
    tas = fields.tas
    pr = fields.pr

    mon_TAS = ma.masked_invalid(np.asarray(tas.values, dtype=float))
    mon_PRC = ma.masked_invalid(np.asarray(pr.values, dtype=float))
    lats = np.asarray(fields.lats.values)
    lons = np.asarray(fields.lons.values)
    grid_shape = mon_TAS.shape[1:]

    T_min = ma.min(mon_TAS, axis=0)
    T_max = ma.max(mon_TAS, axis=0)
    T_ann = ma.mean(mon_TAS, axis=0)
    T_mon = ma.sum(ma.masked_greater_equal(mon_TAS, 10.0).mask, axis=0)

    P_min = ma.min(mon_PRC, axis=0)
    P_ann = ma.sum(mon_PRC, axis=0)
    P_dry = ma.sum(ma.masked_less_equal(mon_PRC, 60.0).mask, axis=0)

    # Summer (Apr-Sep) / winter (Oct-Mar) half-years, hemisphere-aware.
    filled_PRC = mon_PRC.filled(np.nan)
    P_smax, P_smin, P_ssum = _maxminsum_slice(filled_PRC, 3, 9)
    P_wmax, P_wmin, P_wsum = _maxminsum_slice(filled_PRC, -3, 3)

    # Swap S/W south of the equator so "summer" always means local summer.
    equ = int(np.argmin(np.abs(lats)))
    ascending = lats[0] < lats[-1]
    sh = slice(0, equ) if ascending else slice(equ, None)
    for a, b in ((P_smax, P_wmax), (P_smin, P_wmin), (P_ssum, P_wsum)):
        a[sh, :], b[sh, :] = b[sh, :].copy(), a[sh, :].copy()

    # Dryness threshold P_th (Koeppen), from winter/summer precip fractions.
    with np.errstate(invalid="ignore", divide="ignore"):
        P_wpro = P_wsum / P_ann.filled(np.nan)
        P_spro = P_ssum / P_ann.filled(np.nan)
    Tann_f = T_ann.filled(np.nan)
    var_1 = np.where(P_wpro >= 2.0 / 3.0, 2.0 * Tann_f, 0.0)
    var_2 = np.where(P_spro >= 2.0 / 3.0, 2.0 * Tann_f + 28.0, 0.0)
    P_th = np.where((var_1 + var_2) <= 0.0, 2.0 * Tann_f + 14.0, var_1 + var_2)

    # Defaut extras
    # Gaussen ratio requires an absolute temperature scale: convert the (Celsius)
    # monthly field back to Kelvin just for this computation.
    P_sec3 = _gaussen_driest3(filled_PRC, mon_TAS.filled(np.nan) + 273.15)

    args = ma.zeros((N_ARGS,) + grid_shape)
    args[0] = T_min
    args[1] = T_max
    args[2] = T_mon
    args[3] = T_ann
    args[4] = P_min
    args[5] = P_ann
    args[6] = ma.masked_invalid(P_smin)
    args[7] = ma.masked_invalid(P_smax)
    args[8] = ma.masked_invalid(P_wmin)
    args[9] = ma.masked_invalid(P_wmax)
    args[10] = ma.masked_invalid(P_th)
    args[11] = ma.masked_invalid(P_wpro)
    args[12] = P_dry
    args[13] = P_ann                       # explicit mm alias for Defaut
    args[14] = ma.masked_invalid(P_sec3)

    # Propagate the core temperature/precip mask to all layers.
    core_mask = ma.getmaskarray(T_min) | ma.getmaskarray(P_min)
    args.mask = np.broadcast_to(core_mask, args.shape).copy()

    return args, grid_shape, lats, lons
