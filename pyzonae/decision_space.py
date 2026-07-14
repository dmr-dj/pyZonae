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
Defaut (1996) in its own decision space: Qn2 (aridity) against temperature.

Why this beats a grid legend
----------------------------
A grid tabulates the *outcome* of the classification. This plots the
classification itself: every pixel sits at its (Qn2, T) coordinates and the
decision boundaries are drawn on top. Defaut's aridity boundaries are polynomials
in temperature, not constants, so their curvature becomes visible rather than
merely tabulated -- and the density of the cloud shows which parts of climate
space are actually populated, which a grid of equal-area cells hides.

Two vertical axes
-----------------
Defaut's tree changes variable partway down: thermal degrees 1-4 test the
**annual mean** temperature, the colder degrees 5-7 test the **warmest month**
(tc). The figure is therefore two stacked panels sharing the Qn2 axis: the lower
one in annual mean, the upper one in tc, so every rule is drawn against the
variable it truly uses.

Boundaries are clipped automatically
------------------------------------
A boundary is only meaningful where it actually separates two groups. The E|HA
curve stops existing below about 10 degC, because the eremic stage does not exist
there; drawing the polynomial anyway produces the spurious lines one sees in a
naive plot. Rather than hard-coding a temperature window per curve (a list that
silently rots the moment a threshold changes), this module **probes the decision
tree**: at each point of a candidate curve it asks the real classifier which
group sits on either side, and keeps the point only if both groups of the pair
are realised. The boundaries drawn are, by construction, the boundaries the code
applies.

The third axis
--------------
Continentality (a...e) is decided on the thermal amplitude tc - tf, which appears
on neither axis. Within one (group, degree) family the variants occupy the *same*
region of the plane -- SH3a,b and SH3d,e both span Qn2 75-160 and T 10-16 -- so a
single panel would merely overplot them. They are faceted instead.
"""

import re

import numpy as np
import matplotlib.pyplot as plt

from .classifiers.defaut import (
    E_HA, HA_A, A_SA, SA_SH, SH_SX, SH_G, G_O, SX_CBM, SX_BSAA, Qn2,
    etage_climatique_Bonneroy,
)
from .cmaps import Defaut_cmap_1996

# Thresholds, mirroring the tree's own named constants.
T_HOT_WARM = 23.0        # treschaud_chaud
T_WARM_TEMP = 16.5       # chaud_tempere
T_TEMP_COOL = 10.0       # tempere_frais  (C_BM)
T_COOL_COLD = 4.5        # frais_froid    (BM_BS)
TC_COOL_COLD = 20.0      # frais_froid_tc (BM_BS_tc)
TC_COLD_VCOLD = 10.5     # froid_tresfroid (BS_AA)
TC_VCOLD_NIVAL = 2.0     # tresfroid_nival (AA_N)

AMP_HYPEROC_OC = 9.0
AMP_OC_SUBOC = 16.0

FACETS = [
    ("a — hyperoceanic\n(tc−tf ≤ 9)", None, AMP_HYPEROC_OC),
    ("b — oceanic\n(9 < tc−tf ≤ 16)", AMP_HYPEROC_OC, AMP_OC_SUBOC),
    ("c,d,e — sub- to continental\n(tc−tf > 16)", AMP_OC_SUBOC, None),
]

# Continentality bands rendered as marker SHAPES rather than as separate panels.
# This frees the third axis without splitting the figure, but it only works if
# the draw order respects the populations: the continental band holds ~80% of all
# cells, so drawing it last would bury the 12% of hyperoceanic ones underneath.
# They are therefore drawn most-numerous first, rarest last (i.e. on top), and
# the rare ones get a thin edge so they stay legible against the mass below.
#   (label, low, high, marker, zorder, edge_width)
CONT_MARKERS = [
    ("c,d,e — sub- to continental (tc−tf > 16)", AMP_OC_SUBOC, None, "o", 1, 0.0),
    ("b — oceanic (9 < tc−tf ≤ 16)", AMP_HYPEROC_OC, AMP_OC_SUBOC, "^", 2, 0.25),
    ("a — hyperoceanic (tc−tf ≤ 9)", None, AMP_HYPEROC_OC, "s", 3, 0.25),
]

_GRP = re.compile(r"^([A-Z]+)")
_DEFAUT_DICT, _ = Defaut_cmap_1996()
_INV = {v: k for k, v in _DEFAUT_DICT.items()}

# Aridity boundaries in the ANNUAL MEAN, with the pair of groups each separates.
# The pair is what allows the curve to be clipped to where it is live.
ANNUAL_BOUNDARIES = [
    ("E|HA", E_HA, ("E", "HA"), "tab:red", "--"),
    ("HA|A", HA_A, ("HA", "A"), "tab:red", "-"),
    ("A|SA", A_SA, ("A", "SA"), "tab:red", "-"),
    ("SA|SH", SA_SH, ("SA", "SH"), "tab:red", "-"),
    ("SH|SX", SH_SX, ("SH", "SX"), "olive", "-"),
    ("SX|AX", SX_CBM, ("SX", "AX"), "olive", "-"),
    ("SH|G", SH_G, ("SH", "G"), "tab:red", "--"),
    ("G|O", G_O, ("G", "O"), "olive", "--"),
]

# The one aridity boundary that is a function of the WARMEST MONTH.
TC_BOUNDARIES = [
    ("SX|AX (cold)", SX_BSAA, ("SX", "AX"), "tab:blue", "-"),
]


def _group_of(qn2, T, tc, tf):
    """Run the real classifier and return its aridity group, or None."""
    e = etage_climatique_Bonneroy(T, tc, tf, qn2)
    if e is None or (isinstance(e, float) and np.isnan(e)):
        return None
    key = _INV.get(int(e))
    if not key or key.startswith("Doesn"):
        return None
    m = _GRP.match(key)
    return m.group(1) if m else None


def active_segments(fn, pair, y_values, amp, use_tc=False,
                    qn2_max=320.0, pad=2.5):
    """Keep only the portion of a boundary that really separates ``pair``.

    At each y we evaluate the curve, then ask the *actual decision tree* which
    group lies a little to the left and a little to the right of it. The point is
    kept only if the two groups of ``pair`` are what we find. Elsewhere the
    polynomial is a meaningless extrapolation and is masked out with NaN, so
    matplotlib breaks the line rather than drawing a phantom boundary.
    """
    lo_g, hi_g = pair
    xs, ys = [], []
    for y in y_values:
        try:
            x = float(fn(y))
        except Exception:
            xs.append(np.nan); ys.append(np.nan); continue
        if not np.isfinite(x) or x <= 0 or x > qn2_max:
            xs.append(np.nan); ys.append(np.nan); continue

        if use_tc:
            tc = y
            T = y - amp / 2.0
            tf = tc - amp
        else:
            T = y
            tc = T + amp / 2.0
            tf = tc - amp

        g_lo = _group_of(max(x - pad, 0.05), T, tc, tf)
        g_hi = _group_of(x + pad, T, tc, tf)
        live = ((g_lo == lo_g and g_hi == hi_g) or
                (g_lo == hi_g and g_hi == lo_g))
        if live:
            xs.append(x); ys.append(y)
        else:
            xs.append(np.nan); ys.append(np.nan)
    return np.array(xs), np.array(ys)


def compute_cloud(fields_args):
    """Extract (Qn2, T, tc, amplitude) for every land cell."""
    a = fields_args
    flat = a.reshape(a.shape[0], -1)
    mask = np.ma.getmaskarray(flat)
    ok = ~(mask[0] | mask[4])
    tf = np.asarray(flat[0])[ok]
    tc = np.asarray(flat[1])[ok]
    T = np.asarray(flat[3])[ok]
    P = np.asarray(flat[13])[ok]
    Psec = np.asarray(flat[14])[ok]
    with np.errstate(invalid="ignore", divide="ignore"):
        q = Qn2(P, Psec, T, tc, tf)
    return {"qn2": q, "T": T, "tc": tc, "amp": tc - tf, "index": np.where(ok)[0]}


def plot_defaut_space(class_map, fields_args, label_dict, cmap,
                      facet=False, markers=True, qn2_max=300.0, figsize=None,
                      point_size=12.0, alpha=0.75,
                      title="Defaut (1996) decision space"):
    """Scatter every pixel in (Qn2, temperature) with live decision boundaries.

    Parameters
    ----------
    point_size : float
        Marker area. Generous by default: with a few thousand cells the cloud
        should read as a density, not as sparse dust.
    facet : bool
        One column per continentality band. Off by default now that ``markers``
        can carry that axis on a single panel.
    markers : bool
        Encode continentality as marker shape (square = hyperoceanic, triangle =
        oceanic, circle = sub- to continental) instead of splitting the figure.
        Only applies when ``facet`` is False.
    """
    cloud = compute_cloud(fields_args)
    values = sorted(label_dict.values())
    slot = {v: i for i, v in enumerate(values)}
    flat_classes = np.asarray(class_map).reshape(-1)
    cls = flat_classes[cloud["index"]]
    colours = np.array([cmap(slot.get(int(c), 0)) for c in cls])

    panels = FACETS if facet else [("all cells", None, None)]
    n = len(panels)
    if figsize is None:
        figsize = (6.2 * n, 8.4)

    fig, axes = plt.subplots(
        2, n, figsize=figsize, sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [2, 5], "hspace": 0.03},
    )

    T_lo = np.linspace(T_COOL_COLD, 30.0, 240)
    tc_hi = np.linspace(-20.0, TC_COOL_COLD, 240)

    for j, (name, lo, hi) in enumerate(panels):
        ax_hi, ax_lo = axes[0, j], axes[1, j]

        sel = np.ones(len(cloud["qn2"]), dtype=bool)
        if lo is not None:
            sel &= cloud["amp"] > lo
        if hi is not None:
            sel &= cloud["amp"] <= hi
        amp_mid = float(np.nanmedian(cloud["amp"][sel])) if sel.any() else 18.0

        # Warm stages read against the annual mean, cold stages against tc.
        warm = sel & (cloud["T"] >= T_COOL_COLD)
        cold = sel & (cloud["T"] < T_COOL_COLD)

        if markers and not facet:
            # One marker shape per continentality band. Drawn most-numerous
            # first so the rare bands land on top rather than under the mass.
            for lab, blo, bhi, mk, zo, ew in CONT_MARKERS:
                band = np.ones(len(cloud["amp"]), dtype=bool)
                if blo is not None:
                    band &= cloud["amp"] > blo
                if bhi is not None:
                    band &= cloud["amp"] <= bhi
                for ax_, ysel, yvals in ((ax_lo, warm & band, cloud["T"]),
                                         (ax_hi, cold & band, cloud["tc"])):
                    if not ysel.any():
                        continue
                    ax_.scatter(cloud["qn2"][ysel], yvals[ysel], s=point_size,
                                c=colours[ysel], marker=mk, zorder=zo,
                                linewidths=ew, edgecolors="0.25", alpha=alpha)
        else:
            ax_lo.scatter(cloud["qn2"][warm], cloud["T"][warm], s=point_size,
                          c=colours[warm], linewidths=0, alpha=alpha)
            ax_hi.scatter(cloud["qn2"][cold], cloud["tc"][cold], s=point_size,
                          c=colours[cold], linewidths=0, alpha=alpha)

        for lab, fn, pair, col, ls in ANNUAL_BOUNDARIES:
            x, y = active_segments(fn, pair, T_lo, amp_mid, use_tc=False,
                                   qn2_max=qn2_max)
            ax_lo.plot(x, y, color=col, ls=ls, lw=1.2, alpha=0.95)
        for lab, fn, pair, col, ls in TC_BOUNDARIES:
            x, y = active_segments(fn, pair, tc_hi, amp_mid, use_tc=True,
                                   qn2_max=qn2_max)
            ax_hi.plot(x, y, color=col, ls=ls, lw=1.2, alpha=0.95)

        for t in (T_HOT_WARM, T_WARM_TEMP, T_TEMP_COOL):
            ax_lo.axhline(t, color="tab:red", lw=0.8, ls="--", alpha=0.5)
        for t in (TC_COLD_VCOLD, TC_VCOLD_NIVAL):
            ax_hi.axhline(t, color="tab:blue", lw=0.8, ls="-", alpha=0.5)

        ax_hi.set_ylim(TC_COOL_COLD, -20.0)
        ax_lo.set_ylim(30.0, T_COOL_COLD)
        ax_hi.set_xlim(0, qn2_max)
        ax_hi.spines["bottom"].set_visible(False)
        ax_lo.spines["top"].set_visible(False)
        ax_hi.set_title(f"{name}\n{int(sel.sum())} cells", fontsize=9)
        ax_lo.set_xlabel("Qn2 (aridity index)")
        for a_ in (ax_hi, ax_lo):
            a_.grid(alpha=0.25, lw=0.4)

    axes[0, 0].set_ylabel("Warmest month (°C)\ndegrees 5–7", fontsize=8,
                          color="tab:blue")
    axes[1, 0].set_ylabel("Annual mean temperature (°C)\ndegrees 1–4", fontsize=8,
                          color="tab:red")

    if markers and not facet:
        # The shapes mean nothing without a key. Colour still encodes the class,
        # so this legend is deliberately colourless: it explains shape only.
        from matplotlib.lines import Line2D
        handles = [
            Line2D([], [], linestyle="none", marker=mk, markersize=6,
                   markerfacecolor="0.75", markeredgecolor="0.25", label=lab)
            for lab, _, _, mk, _, _ in reversed(CONT_MARKERS)
        ]
        axes[1, 0].legend(handles=handles, loc="lower right", fontsize=7,
                          frameon=True, framealpha=0.9,
                          title="Continentality (tc − tf)", title_fontsize=7.5)

    fig.suptitle(title, fontsize=12, weight="bold")
    return fig, axes
