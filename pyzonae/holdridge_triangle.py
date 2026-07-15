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
Holdridge's triangular life-zone diagram (his Fig. 1), with the data on top.

Why a triangle is legitimate here -- and was not for Defaut
-----------------------------------------------------------
Holdridge's three axes are bound by an exact relation::

    PETR = Tbio * 58.93 / P

In logarithms this is linear::

    log(PETR) + log(P) - log(Tbio) = log(58.93)

so the three log-coordinates lie on a *plane*: there are only two degrees of
freedom, and the classification can be drawn without loss in 2-D. That plane,
cut by the axis ranges, is Holdridge's triangle. Lines of constant PETR come out
straight, which is what makes the diagram readable.

Defaut has no such relation (its aridity index depends on precipitation terms
that are independent of the other two axes), which is why a triangle would be
*wrong* there and a decision-space scatter is used instead. See
:mod:`pyzonae.decision_space`.

What the third dimension becomes
--------------------------------
The altitudinal belt is not a fourth variable: it is the *offset* between the
latitudinal region implied by the sea-level biotemperature and the one implied by
the actual biotemperature. Each cell therefore has two positions in the triangle.
We plot the cell at its **actual** Tbio (that is where its vegetation lives) and
encode the belt as marker shape, in the same spirit as continentality in the
Defaut diagram.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .classifiers.holdridge import (
    LATITUDINAL_REGIONS, ALTITUDINAL_BELTS, HUMIDITY_PROVINCES,
    THRESHOLDS, HUMIDITY_BOUNDS, PET_COEFFICIENT,
    biotemperature, sealevel_biotemperature,
)

# Axis ranges of the classic diagram.
TBIO_MIN, TBIO_MAX = 1.5, 24.0          # degC, doubling each step
P_MIN, P_MAX = 62.5, 8000.0             # mm/yr, doubling each step
PETR_MIN, PETR_MAX = 0.125, 32.0        # dimensionless

# Marker per altitudinal belt (see module docstring).
BELT_MARKERS = {
    "Basal": "o",
    "LowerMontane": "^",
    "Montane": "s",
    "Subalpine": "D",
    "Alpine": "v",
    "Nival": "*",
}


SQRT3_2 = np.sqrt(3.0) / 2.0


def project(tbio, p_ann):
    """Place a cell in Holdridge's triangle.

    The projection is the one that makes the diagram *hexagonal*. Working in
    log2 coordinates u = log2(Tbio), v = log2(P), the three families of lines are

        constant Tbio  (u fixed)
        constant P     (v fixed)
        constant PETR  (u - v fixed, from PETR = Tbio*58.93/P)

    A naive axial projection keeps the PET loci straight but leaves those three
    families at 63/34/30 degrees, so the cells come out as squashed
    parallelograms. The triangular-lattice map below sends them to 60/60/60,
    which is what tiles the plane with regular hexagons -- Holdridge's cells::

        x = log2(P) - 0.5 * log2(Tbio)
        y = (sqrt(3)/2) * log2(Tbio)

    Lines of constant PET ratio remain exactly straight (verified).
    """
    u = np.log2(np.clip(tbio, TBIO_MIN / 2.0, TBIO_MAX * 2.0))
    v = np.log2(np.clip(p_ann, P_MIN / 2.0, P_MAX * 2.0))
    x = v - 0.5 * u
    y = SQRT3_2 * u
    return x, y


def _petr_line(petr, n=60):
    """The straight locus of constant PET ratio, as (x, y) in triangle space."""
    tb = np.linspace(TBIO_MIN / 2.0, TBIO_MAX * 2.0, n)
    p = tb * PET_COEFFICIENT / petr
    return project(tb, p)


def draw_frame(ax, rule="fuzzy", label_axes=True):
    """Draw the triangle: its three graduated axes and the guide lines."""
    col = 1 if rule == "fuzzy" else 0

    # --- Biotemperature guides: horizontal lines (constant Tbio) ---------
    # Each threshold is the lower edge of a latitudinal region, so we can name
    # the bands rather than leaving bare numbers.
    named = [(THRESHOLDS[r][col], r) for r in LATITUDINAL_REGIONS
             if THRESHOLDS[r][col] > 0]
    for t, region in named:
        p = np.array([P_MIN / 2.0, P_MAX * 2.0])
        x, y = project(np.full_like(p, t), p)
        ax.plot(x, y, color="0.75", lw=0.6, zorder=0)
        if label_axes:
            ax.text(x[0] - 0.12, y[0], f"{t:g}°C  {region}", ha="right",
                    va="center", fontsize=6.5, color="0.35")

    # --- Precipitation guides: constant P ---------------------------------
    p_ticks = [62.5, 125, 250, 500, 1000, 2000, 4000, 8000]
    for p in p_ticks:
        tb = np.array([TBIO_MIN / 2.0, TBIO_MAX * 2.0])
        x, y = project(tb, np.full_like(tb, p))
        ax.plot(x, y, color="0.75", lw=0.6, zorder=0)
        if label_axes:
            ax.text(x[0], y[0] - 0.18, f"{p:g}", ha="center", va="top",
                    fontsize=6.5, color="0.35", rotation=45)

    # --- PET ratio guides: straight oblique lines -------------------------
    for q in HUMIDITY_BOUNDS:
        x, y = _petr_line(q)
        ax.plot(x, y, color="tab:brown", lw=0.7, ls="--", alpha=0.8, zorder=0)
        if label_axes:
            ax.text(x[-1] + 0.1, y[-1], f"{q:g}", ha="left", va="center",
                    fontsize=6.5, color="tab:brown")

    if label_axes:
        ax.set_xlabel("annual precipitation (mm, log₂)  →", fontsize=8)
        ax.set_ylabel("biotemperature (°C, log₂)\ncold at top, as in Holdridge's figure",
                      fontsize=8)
        ax.text(0.99, 0.02, "dashed brown: PET ratio", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="tab:brown")


def plot_holdridge_triangle(class_map, fields_args, label_dict, cmap,
                            rule="fuzzy", point_size=14.0, alpha=0.8,
                            markers=True, hexagons=False, hex_alpha=0.30,
                            figsize=(11, 9),
                            title="Holdridge life zones — decision space"):
    """Scatter every cell in Holdridge's triangle, over the zone framework.

    Parameters
    ----------
    class_map : masked 2-D array of class indices
    fields_args : derived-index stack (needs slots 5, 15, 16)
    markers : bool
        Encode the altitudinal belt as marker shape. Drawn most-numerous first so
        the rare high belts are not buried under the basal mass.
    hexagons : bool
        Underlay Holdridge's hexagonal life-zone cells (his Fig. 1). Off by
        default: with a few thousand points on top, the cells tend to compete
        with the data rather than orient the reader. Turn on to compare against
        the published figure.
    """
    a = fields_args
    flat = a.reshape(a.shape[0], -1)
    mask = np.ma.getmaskarray(flat)
    ok = ~(mask[0] | mask[4])

    p_ann = np.asarray(flat[5])[ok]
    tbio = np.asarray(flat[15])[ok]
    idx = np.where(ok)[0]

    cls = np.asarray(class_map).reshape(-1)[idx]
    values = sorted(label_dict.values())
    slot = {v: i for i, v in enumerate(values)}
    inv = {v: k for k, v in label_dict.items()}

    # Cells with no growing season (Tbio == 0) are outside the model and have no
    # place in the diagram: PETR is undefined for them.
    live = np.isfinite(tbio) & (tbio > 0) & np.isfinite(p_ann) & (p_ann > 0)

    colours = np.array([cmap(slot.get(int(c), 0)) for c in cls])
    keys = [inv.get(int(c), "") for c in cls]
    # The belt is read out of the key, but in a *cross* diagram the colouring
    # comes from another classification entirely, whose keys have a different
    # shape ("SH3c", "Cfb"). Fall back to no belt rather than crashing: the
    # marker axis simply carries no information in that case.
    def _belt(k):
        if not k or k.startswith("Outside"):
            return ""
        parts = k.split()
        return parts[1] if len(parts) >= 3 else ""

    belts = np.array([_belt(k) for k in keys])

    fig, ax = plt.subplots(figsize=figsize)
    if hexagons:
        draw_hexagons(ax, label_dict, cmap, rule=rule, alpha=hex_alpha)
    draw_frame(ax, rule=rule)

    x, y = project(tbio, p_ann)

    if markers and (belts != "").any():
        # Most-numerous first, so rare belts land on top rather than underneath.
        order = sorted(BELT_MARKERS, key=lambda b: -(belts == b).sum())
        for b in order:
            sel = live & (belts == b)
            if not sel.any():
                continue
            ew = 0.0 if b == "Basal" else 0.3
            ax.scatter(x[sel], y[sel], s=point_size, c=colours[sel],
                       marker=BELT_MARKERS[b], linewidths=ew,
                       edgecolors="0.25", alpha=alpha, zorder=2)
        handles = [
            Line2D([], [], linestyle="none", marker=BELT_MARKERS[b],
                   markersize=6, markerfacecolor="0.75",
                   markeredgecolor="0.25", label=b)
            for b in ALTITUDINAL_BELTS if (belts == b).any()
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=True,
                  framealpha=0.9, title="Altitudinal belt", title_fontsize=7.5)
    else:
        # No belts to show: either markers were switched off, or the colours come
        # from a classification that has no altitudinal axis (a cross diagram).
        ax.scatter(x[live], y[live], s=point_size, c=colours[live],
                   linewidths=0, alpha=alpha, zorder=2)

    n_out = int((~live).sum())
    if n_out:
        ax.text(0.99, 0.99,
                f"{n_out} cells outside the model\n(Tbio = 0: no growing season)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                style="italic", color="0.35")

    # Holdridge draws his diagram with the nival/polar end at the apex and the
    # tropical end along the base. Our y coordinate grows with warmth, so the
    # axis is flipped here. (This does not affect the straightness of the
    # PET-ratio loci -- verified.)
    ax.invert_yaxis()

    ax.set_title(f"{title}  [{rule} thresholds]", fontsize=11, weight="bold")
    ax.grid(False)
    fig.tight_layout()
    return fig, ax


def draw_hexagons(ax, label_dict, cmap, rule="fuzzy", alpha=0.30,
                  edge="0.55", lw=0.5):
    """Underlay Holdridge's hexagonal life-zone cells.

    Each cell of the diagram is the region closer to one lattice node than to any
    other, which -- because the three line families meet at 60 degrees under
    :func:`project` -- is a regular hexagon. Nodes sit where a biotemperature
    step, a precipitation step and a PET-ratio step coincide, i.e. on the
    doubling ladders of the three axes.

    The hexagons are drawn *behind* the data (low zorder) and washed out with
    alpha, so the cloud stays readable on top. Set ``alpha=0`` to suppress them.
    """
    from matplotlib.patches import RegularPolygon

    # Lattice step: one doubling of Tbio, i.e. one unit of u.
    # In (x, y): a unit step in u moves (-0.5, sqrt3/2); a unit step in v moves
    # (1, 0). The hexagon circumradius that tiles this lattice is 1/sqrt(3).
    R = 1.0 / np.sqrt(3.0)

    inv = {v: k for k, v in label_dict.items()}
    values = sorted(label_dict.values())
    slot = {v: i for i, v in enumerate(values)}

    # Walk the doubling ladders of biotemperature and precipitation.
    u_lo = np.log2(TBIO_MIN / 2.0)
    u_hi = np.log2(TBIO_MAX * 2.0)
    v_lo = np.log2(P_MIN / 2.0)
    v_hi = np.log2(P_MAX * 2.0)

    n = 0
    for u in np.arange(np.floor(u_lo), np.ceil(u_hi) + 1e-9, 1.0):
        for v in np.arange(np.floor(v_lo), np.ceil(v_hi) + 1e-9, 1.0):
            tb = 2.0 ** u
            p = 2.0 ** v
            if not (TBIO_MIN / 2 <= tb <= TBIO_MAX * 2):
                continue
            if not (P_MIN / 2 <= p <= P_MAX * 2):
                continue
            # Colour the cell by the class its centre falls in.
            args = [0.0] * 17
            args[5] = p
            args[15] = tb
            args[16] = tb          # centre of the diagram: sea-level == actual
            from .classifiers.holdridge import get_holdridge_classification
            key = get_holdridge_classification(args, rule=rule, frost_free=None)
            if key is None or key not in label_dict:
                continue
            colour = cmap(slot[label_dict[key]])
            x, y = project(tb, p)
            ax.add_patch(RegularPolygon(
                (x, y), numVertices=6, radius=R, orientation=0.0,
                facecolor=colour, edgecolor=edge, lw=lw, alpha=alpha, zorder=0))
            n += 1
    return n
