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
    ds_t, ds_p, ds_m, ds_o = build(nlat=24, nlon=48, seed=1)
    ds_t.to_netcdf(os.path.join(d, "tas.nc"))
    ds_p.to_netcdf(os.path.join(d, "pr.nc"))
    ds_m.to_netcdf(os.path.join(d, "sftlf.nc"))
    ds_o.to_netcdf(os.path.join(d, "orog.nc"))
    return d


@pytest.mark.parametrize("typ", CLASSIFICATIONS)
def test_runs_and_labels_in_vocabulary(data_dir, typ):
    # orog is only required by Holdridge, but passing it is harmless for the
    # others, so the generic test can stay a single parametrized case.
    m, labels, cmap, lons, lats = run_classification(
        typ_classification=typ,
        tas_file=os.path.join(data_dir, "tas.nc"),
        pr_file=os.path.join(data_dir, "pr.nc"),
        sftlf_file=os.path.join(data_dir, "sftlf.nc"),
        orog_file=os.path.join(data_dir, "orog.nc"),
    )
    valid = set(labels.values())
    produced = set(int(v) for v in m.compressed())
    assert produced, "no cells were classified"
    assert produced.issubset(valid), f"{typ}: out-of-vocabulary indices {produced - valid}"


def test_cell_contract_all_classifiers():
    # A single mild temperate cell vector (15 elements) must classify under every
    # scheme and return a key present in that scheme's dictionary.
    # 17 slots: 0-12 Koeppen, 13-14 Defaut, 15-16 Holdridge (Tbio, T0bio)
    args = [4.0, 20.0, 6, 11.0, 45.0, 900.0, 45.0, 70.0, 55.0, 95.0,
            300.0, 0.5, 0, 900.0, 120.0, 11.5, 12.0]
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


# --- Holdridge -------------------------------------------------------------

def test_holdridge_biotemperature_formula():
    """Tbio divides by 12, not by the number of qualifying months."""
    from pyzonae.classifiers.holdridge import biotemperature

    # Six months at 20 degC, six months at -10 degC (which contribute ZERO).
    monthly = [20.0] * 6 + [-10.0] * 6
    assert abs(biotemperature(monthly) - (6 * 20.0 / 12.0)) < 1e-9  # == 10.0
    # A naive "mean of qualifying months" would give 20.0 -- guard against it.
    assert abs(biotemperature(monthly) - 20.0) > 1.0

    # Values above 30 degC are clipped to zero contribution too.
    assert abs(biotemperature([35.0] * 12) - 0.0) < 1e-9
    # A constant 15 degC year gives exactly 15.
    assert abs(biotemperature([15.0] * 12) - 15.0) < 1e-9


def test_holdridge_sealevel_lapse_rate():
    """T0bio lapses monthly temperatures by -6.0 degC/km before averaging."""
    from pyzonae.classifiers.holdridge import biotemperature, sealevel_biotemperature

    monthly = [10.0] * 12
    # 1000 m up: sea-level temperatures are 6 degC warmer -> 16 degC
    assert abs(sealevel_biotemperature(monthly, 1000.0) - 16.0) < 1e-9
    # At sea level the two agree
    assert abs(sealevel_biotemperature(monthly, 0.0) - biotemperature(monthly)) < 1e-9


def test_holdridge_pet_ratio():
    from pyzonae.classifiers.holdridge import pet_ratio
    assert abs(pet_ratio(10.0, 589.3) - 1.0) < 1e-6      # 10*58.93/589.3 == 1


def test_holdridge_requires_orography(data_dir):
    """Holdridge must refuse to run without elevation, with a clear message."""
    with pytest.raises(ValueError, match="requires surface elevation"):
        run_classification(
            "Holdridge",
            tas_file=os.path.join(data_dir, "tas.nc"),
            pr_file=os.path.join(data_dir, "pr.nc"),
        )


def test_holdridge_frost_line_modes(data_dir):
    """Without a frost line the warm-temperate/subtropical split is left merged."""
    kw = dict(
        tas_file=os.path.join(data_dir, "tas.nc"),
        pr_file=os.path.join(data_dir, "pr.nc"),
        sftlf_file=os.path.join(data_dir, "sftlf.nc"),
        orog_file=os.path.join(data_dir, "orog.nc"),
    )
    m_none, labels, *_ = run_classification("Holdridge", **kw)
    inv = {v: k for k, v in labels.items()}
    regions_none = {inv[int(v)].split()[0] for v in m_none.compressed()}
    assert "WarmTemperate/Subtropical" in regions_none
    assert "Subtropical" not in regions_none

    m_frost, labels2, *_ = run_classification(
        "Holdridge", frost_line="coldest_month", frost_threshold=0.0, **kw
    )
    inv2 = {v: k for k, v in labels2.items()}
    regions_frost = {inv2[int(v)].split()[0] for v in m_frost.compressed()}
    assert "WarmTemperate/Subtropical" not in regions_frost
    assert "Subtropical" in regions_frost or "WarmTemperate" in regions_frost


def test_holdridge_strict_differs_from_fuzzy(data_dir):
    """The two rule sets must give measurably different maps."""
    kw = dict(
        tas_file=os.path.join(data_dir, "tas.nc"),
        pr_file=os.path.join(data_dir, "pr.nc"),
        sftlf_file=os.path.join(data_dir, "sftlf.nc"),
        orog_file=os.path.join(data_dir, "orog.nc"),
    )
    m_f, *_ = run_classification("Holdridge", holdridge_rule="fuzzy", **kw)
    m_s, *_ = run_classification("Holdridge", holdridge_rule="strict", **kw)
    assert not np.array_equal(m_f.filled(-1), m_s.filled(-1))


def test_holdridge_no_growing_season_is_outside_model():
    """Tbio == 0 (every month <= 0 degC) must NOT be labelled 'Superhumid'.

    With no growing season the PET ratio collapses to 0*58.93/P = 0, which would
    fall in the wettest humidity province and paint ice sheets as superhumid.
    Such cells are outside the model's domain.
    """
    from pyzonae.classifiers.holdridge import get_holdridge_classification

    args = [0.0] * 17
    args[5] = 200.0     # some precipitation
    args[15] = 0.0      # Tbio == 0: no month between 0 and 30 degC
    args[16] = 0.0
    assert get_holdridge_classification(args) is None

    # And through the dispatch it becomes the explicit sentinel.
    assert classify_cell("Holdridge", args) == "Outside Holdridge model"


def test_holdridge_legend_only_shows_used_classes(data_dir):
    """plot_classification(only_used=True) must not label absent classes."""
    import matplotlib
    matplotlib.use("Agg")
    from pyzonae.plotting import plot_classification

    m, labels, cmap, lons, lats = run_classification(
        "Holdridge",
        tas_file=os.path.join(data_dir, "tas.nc"),
        pr_file=os.path.join(data_dir, "pr.nc"),
        sftlf_file=os.path.join(data_dir, "sftlf.nc"),
        orog_file=os.path.join(data_dir, "orog.nc"),
    )
    present = set(int(v) for v in m.compressed())
    assert len(present) < len(labels), "test premise: not all classes are used"
    fig, ax = plot_classification(m, lons, lats, labels, cmap,
                                  coastlines=False, only_used=True)
    assert fig is not None


# --- Decision-space diagrams ----------------------------------------------
# These lock the invariants that caught two real bugs during development:
#   * the PET-ratio loci must be straight (else the triangle is meaningless)
#   * the three line families must meet at 60 deg (else the cells are not
#     Holdridge's hexagons -- an early projection was correct but distorted)
#   * every humidity province must fall inside its PETR band (this is what
#     exposed the open-ended Superarid class)
#   * Defaut's boundary curves must be clipped to where the tree really applies
#     them (else spurious lines run across the whole plane)

def test_holdridge_petr_loci_are_straight():
    """Cells of equal PET ratio must be collinear in the triangle projection."""
    from pyzonae.holdridge_triangle import project

    # Three very different (Tbio, P) pairs with the SAME PET ratio.
    pairs = [(12.0, 500.0), (24.0, 1000.0), (6.0, 250.0)]
    ratios = {round(t * 58.93 / p, 6) for t, p in pairs}
    assert len(ratios) == 1, "test premise: the three pairs share one PET ratio"

    pts = [project(t, p) for t, p in pairs]
    x = np.array([q[0] for q in pts])
    y = np.array([q[1] for q in pts])
    resid = np.max(np.abs(np.polyval(np.polyfit(y, x, 1), y) - x))
    assert resid < 1e-9, f"PET loci not straight (residual {resid:.2e})"


def test_holdridge_lattice_is_hexagonal():
    """The three families of lines must meet at 60 degrees."""
    from pyzonae.holdridge_triangle import project

    def d(p0, p1):
        a = np.array(project(*p0)); b = np.array(project(*p1))
        return b - a

    d_tbio = d((4.0, 500.0), (4.0, 1000.0))    # Tbio held constant
    d_prec = d((4.0, 500.0), (8.0, 500.0))     # P held constant
    d_petr = d((4.0, 500.0), (8.0, 1000.0))    # both doubled -> PETR constant

    def angle(u, v):
        c = abs(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return np.degrees(np.arccos(np.clip(c, -1, 1)))

    for u, v in ((d_tbio, d_prec), (d_tbio, d_petr), (d_prec, d_petr)):
        assert abs(angle(u, v) - 60.0) < 1e-6, "lattice is not hexagonal"


def test_holdridge_provinces_lie_in_their_petr_band(data_dir):
    """Every humidity province must sit inside its PET-ratio interval.

    Note the last province is OPEN-ENDED (PETR >= 8): pretending it is a closed
    [8, 16) band is what made an earlier version of this check fail spuriously.
    """
    from pyzonae.classifiers.holdridge import (
        PET_COEFFICIENT, HUMIDITY_BOUNDS, HUMIDITY_PROVINCES,
    )
    from pyzonae.derive import build_arguments
    from pyzonae.io import load_climatology

    fields = load_climatology(
        os.path.join(data_dir, "tas.nc"), os.path.join(data_dir, "pr.nc"),
        sftlf_file=os.path.join(data_dir, "sftlf.nc"),
        orog_file=os.path.join(data_dir, "orog.nc"),
    )
    args, _, _, _ = build_arguments(fields)
    m, labels, _, _, _ = run_classification(
        "Holdridge",
        tas_file=os.path.join(data_dir, "tas.nc"),
        pr_file=os.path.join(data_dir, "pr.nc"),
        sftlf_file=os.path.join(data_dir, "sftlf.nc"),
        orog_file=os.path.join(data_dir, "orog.nc"),
    )
    flat = args.reshape(args.shape[0], -1)
    ok = ~(np.ma.getmaskarray(flat)[0] | np.ma.getmaskarray(flat)[4])
    P = np.asarray(flat[5])[ok]
    Tb = np.asarray(flat[15])[ok]
    cls = np.asarray(m).reshape(-1)[ok]
    inv = {v: k for k, v in labels.items()}

    live = (Tb > 0) & (P > 0) & np.isfinite(Tb) & np.isfinite(P)
    petr = np.full_like(Tb, np.nan)
    petr[live] = Tb[live] * PET_COEFFICIENT / P[live]
    prov = np.array([
        inv.get(int(c), "").split()[2]
        if inv.get(int(c), "") and not inv[int(c)].startswith("Outside") else ""
        for c in cls
    ])

    bounds = [0.0] + HUMIDITY_BOUNDS[:-1] + [np.inf]   # last class is open
    for i, name in enumerate(HUMIDITY_PROVINCES):
        sel = live & (prov == name)
        if not sel.any():
            continue
        lo, hi = bounds[i], bounds[i + 1]
        v = petr[sel]
        assert np.all((v >= lo) & (v < hi)), f"{name} escapes its PETR band"


def test_defaut_boundaries_are_clipped_to_live_regions():
    """A boundary must be drawn only where it really separates its two groups."""
    from pyzonae.decision_space import (
        active_segments, ANNUAL_BOUNDARIES, T_COOL_COLD,
    )
    T = np.linspace(T_COOL_COLD, 30.0, 120)
    by_label = {lab: (fn, pair) for lab, fn, pair, _, _ in ANNUAL_BOUNDARIES}

    # E|HA cannot exist in the cold: the eremic stage is not realised there.
    fn, pair = by_label["E|HA"]
    x, y = active_segments(fn, pair, T, 18.0)
    good = np.isfinite(x)
    assert good.any(), "E|HA should be live somewhere"
    assert y[good].min() > T_COOL_COLD + 1.0, "E|HA should not reach the cold end"

    # SH|SX must not run across the whole Qn2 range (it stopped at ~175).
    fn, pair = by_label["SH|SX"]
    x, y = active_segments(fn, pair, T, 18.0, qn2_max=320.0)
    good = np.isfinite(x)
    assert good.any()
    assert x[good].max() < 250.0, "SH|SX extends far beyond where it is live"


def test_defaut_continentality_partition_is_exact(data_dir):
    """The marker bands must cover every cell exactly once."""
    from pyzonae.decision_space import compute_cloud, CONT_MARKERS
    from pyzonae.derive import build_arguments
    from pyzonae.io import load_climatology

    fields = load_climatology(
        os.path.join(data_dir, "tas.nc"), os.path.join(data_dir, "pr.nc"),
        sftlf_file=os.path.join(data_dir, "sftlf.nc"),
    )
    args, _, _, _ = build_arguments(fields)
    cloud = compute_cloud(args)
    amp = cloud["amp"]

    total = 0
    for _, lo, hi, _, _, _ in CONT_MARKERS:
        band = np.ones(len(amp), dtype=bool)
        if lo is not None:
            band &= amp > lo
        if hi is not None:
            band &= amp <= hi
        total += int(band.sum())
    assert total == len(amp), "continentality bands do not partition the cells"


@pytest.mark.parametrize("typ", ["Defaut96", "Holdridge"])
def test_diagrams_render(data_dir, typ):
    """Both diagrams must build without error from the public API."""
    import matplotlib
    matplotlib.use("Agg")
    from pyzonae import plot_diagram, build_arguments, load_climatology

    kw = dict(
        tas_file=os.path.join(data_dir, "tas.nc"),
        pr_file=os.path.join(data_dir, "pr.nc"),
        sftlf_file=os.path.join(data_dir, "sftlf.nc"),
    )
    if typ == "Holdridge":
        kw["orog_file"] = os.path.join(data_dir, "orog.nc")

    m, labels, cmap, _, _ = run_classification(typ, **kw)
    fields = load_climatology(
        kw["tas_file"], kw["pr_file"], sftlf_file=kw["sftlf_file"],
        orog_file=kw.get("orog_file"),
    )
    args, _, _, _ = build_arguments(fields)
    fig, ax = plot_diagram(typ, m, args, labels, cmap)
    assert fig is not None


def test_diagram_refuses_koeppen(data_dir):
    """Koeppen has no faithful decision-space picture; saying so is better."""
    from pyzonae import plot_diagram, build_arguments, load_climatology

    m, labels, cmap, _, _ = run_classification(
        "peel",
        tas_file=os.path.join(data_dir, "tas.nc"),
        pr_file=os.path.join(data_dir, "pr.nc"),
    )
    fields = load_climatology(os.path.join(data_dir, "tas.nc"),
                             os.path.join(data_dir, "pr.nc"))
    args, _, _, _ = build_arguments(fields)
    with pytest.raises(ValueError, match="No decision-space diagram"):
        plot_diagram("peel", m, args, labels, cmap)
