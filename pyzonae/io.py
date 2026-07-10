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
Input/output for climate classification, using xarray throughout.

This replaces the old ``lcm_utils`` / netCDF4 loading of pyKoeppen with a single
xarray-based path, so the package has no dependency on private LSCE modules.

The loader returns monthly climatologies of near-surface air temperature (in
degrees Celsius) and precipitation (in mm/month), on a common lat/lon grid, plus
an optional land fraction used to mask out ocean cells.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass
class ClimateFields:
    """Container for the monthly climatology used by all classifiers."""
    tas: xr.DataArray   # (time, lat, lon) monthly mean T2m, degrees C
    pr: xr.DataArray    # (time, lat, lon) monthly precipitation, mm/month
    lats: xr.DataArray
    lons: xr.DataArray
    landmask: xr.DataArray | None = None  # (lat, lon) boolean, True over land


def _guess_name(ds, candidates):
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(f"None of {candidates} found in dataset (have: {list(ds.variables)})")


def _scale_by_month(pr, per_month_factor):
    """Multiply a (time, lat, lon) DataArray by a length-12 per-month factor.

    The factor is applied along the leading time axis. Falls back gracefully if
    the number of time steps is not 12 (multiplies by the mean factor).
    """
    nt = pr.sizes.get("time", pr.shape[0])
    if nt == len(per_month_factor):
        factor = xr.DataArray(per_month_factor, dims="time")
        return pr * factor
    return pr * float(np.mean(per_month_factor))


def load_climatology(
    tas_file,
    pr_file,
    tas_var=None,
    pr_var=None,
    sftlf_file=None,
    sftlf_var=None,
    land_threshold=0.5,
    tas_units="auto",
    pr_units="mm/month",
    pr_scale=1.0,
    days_per_month=None,
    decode_times=False,
):
    """Load and harmonize monthly temperature and precipitation climatologies.

    Parameters
    ----------
    tas_file, pr_file : str
        Paths to NetCDF files holding monthly-mean T2m and precipitation.
    tas_var, pr_var : str, optional
        Variable names. If omitted, common names are auto-detected
        (``tas``/``t2m``/``tmp``/``temp`` and ``pr``/``prcp``/``tp``/``precip``).
    sftlf_file, sftlf_var : str, optional
        Land-fraction file/variable. If given, cells with land fraction below
        ``land_threshold`` are masked out.
    land_threshold : float
        Minimum land fraction (0-1) to keep a cell. If the land field looks like
        a percentage (max > 1.5), it is rescaled to 0-1 automatically.
    tas_units : {"auto", "C", "K"}
        Temperature unit handling. "auto" converts to Celsius when the field
        looks like Kelvin (mean > 100).
    pr_units : str
        Units of the precipitation variable, converted to mm/month. One of
        ``"mm/month"`` (default, no change), ``"mm/day"``, ``"m/month"``,
        ``"kg/m2/s"`` (the CMIP native flux, = mm/s), or ``"kg/m2/month"``
        (= mm/month). Per-second and per-day units are integrated to monthly
        totals using standard days-per-month (see ``days_per_month``).
    pr_scale : float
        Extra multiplicative factor applied *after* ``pr_units`` conversion, for
        any unit not covered above. Leave at 1.0 in the common case.
    days_per_month : sequence of 12 floats, optional
        Days in each month, used only for per-second / per-day conversions.
        Defaults to the standard (non-leap) calendar.
    decode_times : bool
        Passed to :func:`xarray.open_dataset`.

    Returns
    -------
    ClimateFields
    """
    ds_t = xr.open_dataset(tas_file, decode_times=decode_times)
    ds_p = xr.open_dataset(pr_file, decode_times=decode_times)

    tvar = tas_var or _guess_name(ds_t, ["tas", "t2m", "tmp", "temp", "tem"])
    pvar = pr_var or _guess_name(ds_p, ["pr", "prcp", "tp", "precip", "prc"])

    tas = ds_t[tvar]
    pr = ds_p[pvar]

    # --- Precipitation units -> mm/month ---------------------------------
    # 1 kg m-2 of water == 1 mm depth, so kg/m2/* and mm/* differ only by the
    # time base. Per-second and per-day fluxes are integrated to monthly totals.
    if days_per_month is None:
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dpm = np.asarray(days_per_month, dtype=float)

    unit = pr_units.lower().replace(" ", "").replace("kgm-2", "kg/m2")
    if unit in ("mm/month", "kg/m2/month", "kg/m2/mon"):
        pass  # already mm/month
    elif unit in ("mm/day", "mm/d", "kg/m2/day", "kg/m2/d"):
        pr = _scale_by_month(pr, dpm)                 # mm/day * days
    elif unit in ("m/month", "m/mon"):
        pr = pr * 1000.0                              # m -> mm
    elif unit in ("kg/m2/s", "mm/s"):
        pr = _scale_by_month(pr, dpm * 86400.0)       # mm/s * seconds-in-month
    else:
        raise ValueError(
            f"Unknown pr_units '{pr_units}'. Known: mm/month, mm/day, m/month, "
            f"kg/m2/s, kg/m2/month."
        )
    if pr_scale != 1.0:
        pr = pr * pr_scale

    # Temperature to Celsius
    if tas_units == "K" or (tas_units == "auto" and float(tas.mean()) > 100.0):
        tas = tas - 273.15

    # Align precipitation coords onto the temperature grid names if needed
    lat_name = "lat" if "lat" in pr.coords else ("latitude" if "latitude" in pr.coords else "y")
    lon_name = "lon" if "lon" in pr.coords else ("longitude" if "longitude" in pr.coords else "x")
    lats = pr[lat_name]
    lons = pr[lon_name]

    landmask = None
    if sftlf_file is not None:
        ds_m = xr.open_dataset(sftlf_file, decode_times=decode_times)
        svar = sftlf_var or _guess_name(ds_m, ["sftlf", "lsm", "land", "mask"])
        land = ds_m[svar]
        if float(land.max()) > 1.5:   # percentage -> fraction
            land = land / 100.0
        land = land.assign_coords({lat_name: lats, lon_name: lons})
        landmask = land >= land_threshold
        tas = tas.where(landmask)
        pr = pr.where(landmask)

    return ClimateFields(tas=tas, pr=pr, lats=lats, lons=lons, landmask=landmask)
