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

# Defer evaluation of annotations, so modern typing syntax in this module can
# never break older interpreters at import time.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import xarray as xr


@dataclass
class ClimateFields:
    """Container for the monthly climatology used by all classifiers."""
    tas: xr.DataArray   # (time, lat, lon) monthly mean T2m, degrees C
    pr: xr.DataArray    # (time, lat, lon) monthly precipitation, mm/month
    lats: xr.DataArray
    lons: xr.DataArray
    landmask: Optional[xr.DataArray] = None  # (lat, lon) boolean, True over land
    orog: Optional[xr.DataArray] = None      # (lat, lon) surface elevation, m
    frost_free: Optional[xr.DataArray] = None  # (lat, lon) True where frost-free


def _guess_name(ds, candidates):
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(f"None of {candidates} found in dataset (have: {list(ds.variables)})")


def _guess_time_dim(da):
    """Name of the time dimension of a (time, lat, lon) DataArray.

    Climate files disagree on this name: 'time', 't', 'month', 'time_counter'
    (NEMO/IPSL), 'Time' (WRF), and so on. We look, in order, for a coordinate
    flagged as time by its metadata, then a common name, then fall back to the
    leading dimension -- which for these monthly climatologies is the time axis.
    """
    # 1) a coordinate whose CF metadata marks it as time
    for name in da.dims:
        coord = da.coords.get(name)
        if coord is None:
            continue
        attrs = {k.lower(): str(v).lower() for k, v in coord.attrs.items()}
        if attrs.get("axis") == "t" or attrs.get("standard_name") == "time":
            return name
        units = attrs.get("units", "")
        if " since " in units:            # "days since 1850-01-01" etc.
            return name
    # 2) a common name
    for name in ("time", "t", "month", "months", "time_counter", "Time", "TIME"):
        if name in da.dims:
            return name
    # 3) fall back to the leading dimension
    return da.dims[0]


def _scale_by_month(pr, per_month_factor):
    """Multiply a (time, lat, lon) DataArray by a length-12 per-month factor.

    The factor is applied along the leading time axis. Falls back gracefully if
    the number of time steps is not 12 (multiplies by the mean factor).
    """
    tdim = _guess_time_dim(pr)
    nt = pr.sizes.get(tdim, pr.shape[0])
    if nt == len(per_month_factor):
        factor = xr.DataArray(per_month_factor, dims=tdim)
        return pr * factor
    return pr * float(np.mean(per_month_factor))


def load_climatology(
    tas_file,
    pr_file,
    tas_var=None,
    pr_var=None,
    sftlf_file=None,
    sftlf_var=None,
    orog_file=None,
    orog_var=None,
    frost_line=None,
    frost_threshold=0.0,
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

    # --- Orography (required by Holdridge, ignored by the other schemes) ---
    orog = None
    if orog_file is not None:
        ds_o = xr.open_dataset(orog_file, decode_times=decode_times)
        ovar = orog_var or _guess_name(ds_o, ["orog", "elevation", "elev", "z", "topo", "hgt"])
        orog = ds_o[ovar].assign_coords({lat_name: lats, lon_name: lons})
        if landmask is not None:
            orog = orog.where(landmask)

    # --- Frost line -------------------------------------------------------
    # Only ever used to split WarmTemperate from Subtropical. Four strategies:
    #
    #   None                  -> unknown. The two regions are reported merged
    #                            rather than inventing a boundary.
    #   "coldest_month"       -> frost-free where the coldest monthly mean tas
    #                            exceeds `frost_threshold` (degC). Crude, but
    #                            computable from a monthly climatology.
    #   "tasmin_threshold"    -> same, but applied to a monthly tasmin field,
    #                            which must be supplied as `frost_line=<DataArray>`
    #                            of monthly minima (closer to the daily criterion).
    #   <DataArray> or <path> -> a ready-made boolean mask, True = frost-free.
    #                            This is the hook for a properly calibrated frost
    #                            line derived from daily data (Lugo et al. use
    #                            "fewer than 0.5 frost days per year", where a
    #                            frost day has daily Tmin < 0 degC).
    frost_free = None
    if isinstance(frost_line, str) and frost_line == "coldest_month":
        frost_free = tas.min(dim=_guess_time_dim(tas)) > frost_threshold
    elif isinstance(frost_line, str) and frost_line == "tasmin_threshold":
        raise ValueError(
            "frost_line='tasmin_threshold' requires the monthly tasmin field; "
            "pass it directly as frost_line=<DataArray of monthly minima>."
        )
    elif isinstance(frost_line, str):
        # treat as a path to a NetCDF holding a boolean/0-1 frost-free mask
        ds_f = xr.open_dataset(frost_line, decode_times=decode_times)
        fvar = _guess_name(ds_f, ["frost_free", "frostfree", "nofrost", "mask"])
        frost_free = ds_f[fvar].assign_coords({lat_name: lats, lon_name: lons}) > 0.5
    elif frost_line is not None:
        fl = frost_line
        # A supplied frost field is either a ready 2-D (lat, lon) mask, or a
        # 3-D stack of monthly minima to be reduced. Distinguish by rank, not by
        # a dimension name -- the time axis may be called anything.
        if getattr(fl, "ndim", 2) >= 3:
            fl = fl.min(dim=_guess_time_dim(fl)) > frost_threshold
        frost_free = fl.assign_coords({lat_name: lats, lon_name: lons})
        if frost_free.dtype != bool:
            frost_free = frost_free > 0.5
    if frost_free is not None and landmask is not None:
        frost_free = frost_free.where(landmask)

    return ClimateFields(tas=tas, pr=pr, lats=lats, lons=lons, landmask=landmask,
                         orog=orog, frost_free=frost_free)
