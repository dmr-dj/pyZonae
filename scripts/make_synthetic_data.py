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
Generate a small, physically-plausible synthetic monthly climatology.

This makes the repository self-contained: the test-data directory ships with
NetCDF files that reproduce the broad structure of Earth's climate (latitudinal
temperature gradient, seasonal cycle with a hemispheric phase flip, an ITCZ rain
band, subtropical dry zones, and midlatitude storm tracks) without depending on
any private or copyrighted dataset.

Run::

    python scripts/make_synthetic_data.py --outdir test-data

It writes:
    synthetic_tas_monClim.nc    near-surface air temperature (degrees C)
    synthetic_pr_monClim.nc     precipitation (mm/month)
    synthetic_sftlf.nc          land fraction (0-1)
"""

import argparse
import os

import numpy as np
import xarray as xr


def build(nlat=48, nlon=96, seed=0):
    rng = np.random.default_rng(seed)
    lats = np.linspace(-87.5, 87.5, nlat)
    lons = np.linspace(1.875, 358.125, nlon)
    months = np.arange(1, 13)
    latg = lats[:, None]

    tas = np.empty((12, nlat, nlon))
    pr = np.empty((12, nlat, nlon))

    for mi, m in enumerate(months):
        # Seasonal insolation phase: peak NH summer near July.
        phase = np.cos(2 * np.pi * (m - 7) / 12.0)

        # Temperature: warm equator, cold poles, seasonal swing growing with |lat|.
        base_T = 30.0 - 0.55 * np.abs(latg)
        seasonal_amp = 0.18 * np.abs(latg) + 2.0
        # NH and SH seasons are opposite.
        seasonal = seasonal_amp * phase * np.sign(latg)
        T = base_T + seasonal
        T = np.repeat(T, nlon, axis=1)
        T += rng.normal(0, 0.4, size=(nlat, nlon))
        tas[mi] = T

        # Precipitation: ITCZ band that migrates with the season, subtropical
        # dry belts near +/-25 deg, and midlatitude storm tracks near +/-50 deg.
        itcz_center = 8.0 * phase   # migrates north in NH summer
        itcz = 180.0 * np.exp(-((latg - itcz_center) ** 2) / (2 * 8.0 ** 2))
        subtropical = -60.0 * np.exp(-((np.abs(latg) - 25.0) ** 2) / (2 * 7.0 ** 2))
        storm = 70.0 * np.exp(-((np.abs(latg) - 50.0) ** 2) / (2 * 10.0 ** 2))
        # Winter-enhanced storm tracks (more rain in local winter).
        storm = storm * (1.0 + 0.4 * (-phase) * np.sign(latg))
        P = np.clip(itcz + subtropical + storm + 20.0, 2.0, None)
        P = np.repeat(P, nlon, axis=1)
        # Add zonal structure: a "monsoon" longitude sector and a dry sector.
        zon = 1.0 + 0.5 * np.sin(2 * np.pi * lons / 360.0)[None, :]
        P = P * zon
        P += rng.normal(0, 4.0, size=(nlat, nlon))
        pr[mi] = np.clip(P, 1.0, None)

    # Simple land fraction: a couple of idealized continents + polar land.
    lon2d, lat2d = np.meshgrid(lons, lats)
    land = np.zeros((nlat, nlon))
    # "Continent A": lon 20-140, lat -40..70
    land += ((lon2d > 20) & (lon2d < 140) & (lat2d > -40) & (lat2d < 70)).astype(float)
    # "Continent B": lon 200-320, lat -55..75
    land += ((lon2d > 200) & (lon2d < 320) & (lat2d > -55) & (lat2d < 75)).astype(float)
    # Antarctica-like polar land
    land += (lat2d < -70).astype(float)
    land = np.clip(land, 0, 1)

    def da(data, name, units, long_name):
        return xr.DataArray(
            data, dims=("time", "lat", "lon"),
            coords={"time": months, "lat": lats, "lon": lons},
            name=name, attrs={"units": units, "long_name": long_name},
        )

    # Ship temperature in KELVIN and precipitation in mm/month, matching the
    # PMIP4/AWIESM convention. This lets code that assumes Kelvin (e.g. the
    # original Bonneroy notebook, which unconditionally does tas - 273.15) run on
    # this synthetic data unchanged. The pyzonae loader auto-detects Kelvin.
    tas_K = tas + 273.15
    ds_t = da(tas_K, "tas", "K", "near-surface air temperature").to_dataset()
    ds_p = da(pr, "pr", "mm/month", "precipitation").to_dataset()
    # sftlf as a percentage (0-100), as in AWIESM/PMIP4. The pyzonae loader
    # auto-rescales to a 0-1 fraction; AWIESM-style code using ">= 0.5" also
    # works because land cells are 100 and ocean cells are 0.
    ds_m = xr.DataArray(
        land * 100.0, dims=("lat", "lon"), coords={"lat": lats, "lon": lons},
        name="sftlf", attrs={"units": "%", "long_name": "land area fraction"},
    ).to_dataset()
    return ds_t, ds_p, ds_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="test-data")
    ap.add_argument("--nlat", type=int, default=48)
    ap.add_argument("--nlon", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ds_t, ds_p, ds_m = build(a.nlat, a.nlon, a.seed)
    ds_t.to_netcdf(os.path.join(a.outdir, "synthetic_tas_monClim.nc"))
    ds_p.to_netcdf(os.path.join(a.outdir, "synthetic_pr_monClim.nc"))
    ds_m.to_netcdf(os.path.join(a.outdir, "synthetic_sftlf.nc"))
    print("wrote synthetic_{tas,pr,sftlf} to", a.outdir)


if __name__ == "__main__":
    main()
