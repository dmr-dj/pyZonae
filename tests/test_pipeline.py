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
Non-regression tests for pyzonae.

These build a tiny synthetic climatology on the fly (no external data needed)
and check that every classification runs end to end and returns sensible,
in-vocabulary results. Run with ``pytest`` from the repo root.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyzonae import run_classification, classify_cell, CLASSIFICATIONS, get_cmap  # noqa: E402
from scripts.make_synthetic_data import build  # noqa: E402


@pytest.fixture(scope="module")
def data_dir():
    d = tempfile.mkdtemp()
    ds_t, ds_p, ds_m = build(nlat=24, nlon=48, seed=1)
    ds_t.to_netcdf(os.path.join(d, "tas.nc"))
    ds_p.to_netcdf(os.path.join(d, "pr.nc"))
    ds_m.to_netcdf(os.path.join(d, "sftlf.nc"))
    return d


@pytest.mark.parametrize("typ", CLASSIFICATIONS)
def test_runs_and_labels_in_vocabulary(data_dir, typ):
    m, labels, cmap, lons, lats = run_classification(
        typ_classification=typ,
        tas_file=os.path.join(data_dir, "tas.nc"),
        pr_file=os.path.join(data_dir, "pr.nc"),
        sftlf_file=os.path.join(data_dir, "sftlf.nc"),
    )
    valid = set(labels.values())
    produced = set(int(v) for v in m.compressed())
    assert produced, "no cells were classified"
    assert produced.issubset(valid), f"{typ}: out-of-vocabulary indices {produced - valid}"


def test_cell_contract_all_classifiers():
    # A single mild temperate cell vector (15 elements) must classify under every
    # scheme and return a key present in that scheme's dictionary.
    args = [4.0, 20.0, 6, 11.0, 45.0, 900.0, 45.0, 70.0, 55.0, 95.0,
            300.0, 0.5, 0, 900.0, 120.0]
    for typ in CLASSIFICATIONS:
        labels, _ = get_cmap(typ)
        key = classify_cell(typ, args)
        assert key in labels, f"{typ}: key {key!r} not in label dict"


def test_defaut_sentinel_on_nan():
    args = [np.nan] * 15
    key = classify_cell("Defaut96", args)
    labels, _ = get_cmap("Defaut96")
    assert labels[key] == 10000


def test_landmask_reduces_cells(data_dir):
    m_land, *_ = run_classification(
        "peel",
        tas_file=os.path.join(data_dir, "tas.nc"),
        pr_file=os.path.join(data_dir, "pr.nc"),
        sftlf_file=os.path.join(data_dir, "sftlf.nc"),
    )
    m_all, *_ = run_classification(
        "peel",
        tas_file=os.path.join(data_dir, "tas.nc"),
        pr_file=os.path.join(data_dir, "pr.nc"),
        sftlf_file=None,
    )
    assert m_land.count() < m_all.count()


def test_pr_units_roundtrip(tmp_path):
    """Every supported pr_units expresses the same physical amount as mm/month."""
    import numpy as np
    import xarray as xr
    from pyzonae.io import load_climatology

    lats = np.array([-30., 0., 30.]); lons = np.array([0., 120., 240.])
    months = np.arange(1, 13)
    dpm = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31.])
    base = np.full((12, 3, 3), 90.0)  # 90 mm/month

    def w(path, data, units):
        xr.DataArray(data, dims=("time", "lat", "lon"),
                     coords={"time": months, "lat": lats, "lon": lons},
                     name="pr", attrs={"units": units}).to_dataset().to_netcdf(path)

    tfile = tmp_path / "t.nc"
    xr.DataArray(np.full((12, 3, 3), 288.0), dims=("time", "lat", "lon"),
                 coords={"time": months, "lat": lats, "lon": lons},
                 name="tas").to_dataset().to_netcdf(tfile)

    cases = {
        "mm/month": base,
        "mm/day": base / dpm[:, None, None],
        "m/month": base / 1000.0,
        "kg/m2/s": base / (dpm[:, None, None] * 86400.0),
        "kg/m2/month": base,
    }
    for unit, data in cases.items():
        pfile = tmp_path / f"p_{unit.replace('/', '_')}.nc"
        w(pfile, data, unit)
        fld = load_climatology(str(tfile), str(pfile), pr_units=unit)
        assert np.allclose(fld.pr.values, 90.0, atol=1e-6), f"{unit} did not round-trip"


def test_custom_variable_names(tmp_path):
    import numpy as np
    import xarray as xr
    from pyzonae.io import load_climatology

    lats = np.array([0., 30.]); lons = np.array([0., 180.]); months = np.arange(1, 13)
    xr.DataArray(np.full((12, 2, 2), 285.0), dims=("time", "lat", "lon"),
                 coords={"time": months, "lat": lats, "lon": lons},
                 name="temperature_2m").to_dataset().to_netcdf(tmp_path / "t.nc")
    xr.DataArray(np.full((12, 2, 2), 80.0), dims=("time", "lat", "lon"),
                 coords={"time": months, "lat": lats, "lon": lons},
                 name="total_precip").to_dataset().to_netcdf(tmp_path / "p.nc")
    fld = load_climatology(str(tmp_path / "t.nc"), str(tmp_path / "p.nc"),
                           tas_var="temperature_2m", pr_var="total_precip")
    assert abs(float(fld.pr.mean()) - 80.0) < 1e-6
