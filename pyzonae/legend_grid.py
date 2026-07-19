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
Grid legend for Defaut (1996), in the style used by Botti.

The problem this solves: a classification with three independent axes cannot be
read from a flat colorbar, and colour alone cannot encode three dimensions (hue,
lightness and saturation are not perceived independently).

Botti's answer, reproduced here: let colour be a mere *identifier*, and let the
**position in a grid** carry the structure. Each cell of the grid holds the class
code, painted in the colour used on the map. Combinations that do not exist are
left blank -- which is itself informative.

Defaut's codes are self-describing, which makes the decomposition exact::

    SA3d,e
    ^^ ^ ^^^
    |  |  +-- continentality: a (hyperoceanic) ... e (continental)
    |  +----- thermal degree: 1 (hot) ... 7 (very cold)
    +-------- aridity group:  E, HA, A, SA, SH, SX, AX, G, O

Note that a code may carry a *merged* continentality token (``a,b`` or ``c,d,e``):
Defaut does not always subdivide that axis, so such a class legitimately spans
several columns of the grid. Botti's figure shows exactly the same merged cells.
"""

import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Aridity groups ordered by INCREASING moisture (i.e. increasing qn2), which is
# the order the classification itself implies. Measured qn2 medians on a global
# pre-industrial climatology:
#
#   E 4.5 | HA 15 | A 42 | SA 65 | SX 76 | SH 115 | AX 147 | G 219 | O 601
#
# Note that SX (subxeric) and AX (axeric) are not "cold sidebars": they sit on
# the moisture gradient like the others. What makes them look out of place if you
# only read colours is that they are also the COLD stages (median annual T of
# -5 and -16 degC, against +23 degC for G and O), and the grid does not show
# temperature in its column order.
ARIDITY_GROUPS = ["E", "HA", "A", "SA", "SX", "SH", "AX", "G", "O"]
ARIDITY_LABELS = {
    "E": "E\nEremic",
    "HA": "HA\nHyperarid",
    "A": "A\nArid",
    "SA": "SA\nSemiarid",
    "SH": "SH\nSubhumid",
    "SX": "SX\nSubxeric",
    "AX": "AX\nAxeric",
    "G": "G\nGuinean",
    "O": "O\nOmbrophilous",
}

# Continentality, from the most oceanic to the most continental.
CONTINENTALITY = ["a", "b", "c", "d", "e"]
CONTINENTALITY_LABELS = {
    "a": "a. Hyperoceanic",
    "b": "b. Oceanic",
    "c": "c. Suboceanic",
    "d": "d. Subcontinental",
    "e": "e. Continental",
}



# --------------------------------------------------------------------------
# Quantitative ranges
# --------------------------------------------------------------------------
# Defaut's aridity boundaries are NOT constants: they are polynomials in the
# annual mean temperature (E_HA(T), HA_A(T), ...). A single qn2 range per group
# would therefore be wrong. What is exact is the range *at a given thermal
# degree*, which is precisely what one column of the grid represents -- so the
# range is computed per (group, degree) cell.
#
# Thermal degrees, from the decision tree's own named thresholds:
#   1: T >= 23        2: 16.5 <= T < 23     3: 10 <= T < 16.5   4: 4.5 <= T < 10
#   5,6,7: colder stages, keyed on the warmest month (tc), not on T.
THERMAL_RANGES = {
    1: (23.0, None),      # hot
    2: (16.5, 23.0),      # temperate
    3: (10.0, 16.5),      # cool
    4: (4.5, 10.0),       # cold
    5: (None, 4.5),       # very cold  (tc-based below)
    6: (None, None),
    7: (None, None),
}
THERMAL_LABELS = {
    1: "1. Hot\n(T>=23)", 2: "2. Temp.\n(16.5-23)", 3: "3. Cool\n(10-16.5)",
    4: "4. Cold\n(4.5-10)", 5: "5. V.cold", 6: "6. Subnival", 7: "7. Nival",
}

# Continentality is decided on the annual thermal amplitude (tc - tf):
#   a,b : amplitude <= 9      (hyperoceanic / oceanic)
#   c,d,e: amplitude > 16     (suboceanic / subcontinental / continental)
CONTINENTALITY_RANGES = {
    "a": "amp<=9", "b": "amp<=16", "c": "amp>16", "d": "amp>16", "e": "amp>16",
}


def qn2_range_for(group, degree):
    """Exact qn2 interval of a (group, degree) cell, evaluated at that degree.

    Returns (low, high) with None for an open end, or None if not computable
    (degrees 5-7 branch on the warmest month rather than the annual mean).
    """
    from .classifiers.defaut import E_HA, HA_A, A_SA, SA_SH, SH_SX, SH_G, G_O

    rng = THERMAL_RANGES.get(degree)
    if rng is None or degree >= 5:
        return None
    lo_T, hi_T = rng
    # Evaluate the boundary curves at a representative temperature for this
    # degree: the midpoint of its interval (or a little above the lower bound
    # for the open-ended hot degree).
    if hi_T is None:
        T = lo_T + 3.0
    elif lo_T is None:
        T = hi_T - 3.0
    else:
        T = 0.5 * (lo_T + hi_T)

    edges = {
        "E":  (None,      E_HA(T)),
        "HA": (E_HA(T),   HA_A(T)),
        "A":  (HA_A(T),   A_SA(T)),
        "SA": (A_SA(T),   SA_SH(T)),
        "SH": (SA_SH(T),  SH_G(T)),
        "G":  (SH_G(T),   G_O(T)),
        "O":  (G_O(T),    None),
        "SX": (SH_SX(T),  None),
        "AX": (None,      None),
    }
    return edges.get(group)


_KEY = re.compile(r"^([A-Z]+)(\d*)([a-z,]*)$")


def decompose(key):
    """Split a Defaut code into (group, thermal_degree, [continentality...]).

    Returns ``None`` for the sentinel or anything unparseable.
    ``thermal_degree`` is ``None`` for groups that have none (E, G, O), and the
    continentality list is empty for those too.
    """
    m = _KEY.match(key)
    if not m:
        return None
    group, digit, letters = m.groups()
    if group not in ARIDITY_GROUPS:
        return None
    degree = int(digit) if digit else None
    cols = [c for c in letters.split(",") if c] if letters else []
    return group, degree, cols


def plot_defaut_grid(label_dict, cmap, present=None, ax=None,
                     title="Key to phytoclimatic units (Defaut 1996)",
                     show_codes=True, show_qn2=True):
    """Draw the Botti-style grid legend for Defaut.

    Parameters
    ----------
    label_dict : dict[str, int]
        The Defaut vocabulary, as returned by :mod:`pyzonae.cmaps`.
    cmap : matplotlib colormap
        Colours indexed by position in the sorted label values.
    present : set of int, optional
        If given, only these class indices are drawn (the rest of the grid stays
        blank). Pass the classes actually on your map to reproduce Botti's habit
        of showing only what occurs.
    show_codes : bool
        Write the class code inside each cell (as Botti does).

    Returns
    -------
    fig, ax
    """
    values = sorted(label_dict.values())
    slot = {v: i for i, v in enumerate(values)}

    # Collect the cells: (group, degree, continentality) -> (key, colour)
    cells = {}
    degrees_by_group = {g: set() for g in ARIDITY_GROUPS}
    for key, val in label_dict.items():
        if present is not None and val not in present:
            continue
        d = decompose(key)
        if d is None:
            continue
        group, degree, cols = d
        colour = cmap(slot[val])
        if not cols:
            # E, G, O: no continentality axis. They occupy the whole column.
            cols = [None]
        degrees_by_group[group].add(degree)
        for c in cols:
            cells[(group, degree, c)] = (key, colour)

    # Column layout: each aridity group is a block of its own thermal degrees.
    columns = []          # list of (group, degree)
    for g in ARIDITY_GROUPS:
        for deg in sorted(x for x in degrees_by_group[g] if x is not None):
            columns.append((g, deg))
        if None in degrees_by_group[g]:      # E, G, O -> single column
            columns.append((g, None))
    if not columns:
        raise ValueError("no Defaut classes to draw")

    col_index = {c: i for i, c in enumerate(columns)}
    n_col = len(columns)
    n_row = len(CONTINENTALITY)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, 0.42 * n_col), 3.4))
    else:
        fig = ax.figure

    ax.set_xlim(0, n_col)
    ax.set_ylim(0, n_row)
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Draw the cells.
    for (group, degree, cont), (key, colour) in cells.items():
        ci = col_index.get((group, degree))
        if ci is None:
            continue
        if cont is None:
            rows = range(n_row)          # groups without a continentality axis
        else:
            rows = [CONTINENTALITY.index(cont)]
        for ri in rows:
            ax.add_patch(mpatches.Rectangle(
                (ci, ri), 1, 1, facecolor=colour, edgecolor="white", lw=0.8))
            if show_codes and (cont is None or ri == CONTINENTALITY.index(cont)):
                # Contrast-aware text colour.
                r, g_, b = colour[:3]
                lum = 0.299 * r + 0.587 * g_ + 0.114 * b
                ax.text(ci + 0.5, ri + 0.5, key, ha="center", va="center",
                        fontsize=5.2, weight="bold",
                        color="white" if lum < 0.5 else "black")

    # Row labels (continentality).
    for ri, c in enumerate(CONTINENTALITY):
        ax.text(-0.15, ri + 0.5, CONTINENTALITY_LABELS[c], ha="right", va="center",
                fontsize=6.5)

    # Column headers: the thermal degree, plus a group band above.
    for (group, degree), ci in col_index.items():
        lab = str(degree) if degree is not None else "-"
        if show_qn2:
            rng = qn2_range_for(group, degree)
            if rng:
                lo_q, hi_q = rng
                if lo_q is not None and hi_q is not None:
                    lab += f"\n{lo_q:.0f}-{hi_q:.0f}"
                elif hi_q is not None:
                    lab += f"\n<{hi_q:.0f}"
                elif lo_q is not None:
                    lab += f"\n>{lo_q:.0f}"
        ax.text(ci + 0.5, -0.10, lab, ha="center", va="bottom", fontsize=5.2)
    # Group bands. Deliberately WITHOUT a qn2 range: Defaut's aridity boundaries
    # are polynomials in temperature, so a single interval per group does not
    # exist. Quoting one (by pooling evaluations across thermal degrees) would
    # invent a range whose endpoints never co-occur. The exact interval belongs
    # to a (group, degree) CELL, and is written under each column instead.
    for g in ARIDITY_GROUPS:
        idx = [i for (gr, _), i in col_index.items() if gr == g]
        if not idx:
            continue
        lo, hi = min(idx), max(idx) + 1
        ax.plot([lo + 0.05, hi - 0.05], [-0.95, -0.95], lw=1.2, color="0.3",
                clip_on=False)
        ax.text((lo + hi) / 2.0, -1.05, ARIDITY_LABELS[g], ha="center",
                va="bottom", fontsize=6.5, weight="bold")

    ax.text(-0.15, -1.05, "Aridity group", ha="right", va="bottom",
            fontsize=6.5, weight="bold")
    ax.text(-0.15, -0.10, "Heat degree\nQn2 range", ha="right", va="bottom",
            fontsize=5.5, weight="bold")
    if title:
        ax.set_title(title, fontsize=9, weight="bold", pad=26)
    return fig, ax


def plot_defaut_with_grid(class_map, lons, lats, label_dict, cmap,
                          title="Defaut (1996) phytoclimatic units",
                          figsize=(16, 11), coastlines=True):
    """Map + Botti-style grid legend in one figure.

    Only the classes present on the map are shown in the grid, as Botti does.
    """
    try:
        import cartopy.crs as ccrs
        has_cartopy = True
    except Exception:
        has_cartopy = False

    present = set(int(v) for v in np.ma.compressed(class_map))
    values = sorted(label_dict.values())
    slot = {v: i for i, v in enumerate(values)}

    # Reduce the colormap to the classes on the map, keeping their colours.
    from matplotlib.colors import ListedColormap
    used_vals = [v for v in values if v in present]
    small = ListedColormap([cmap(slot[v]) for v in used_vals])
    remap = {v: i for i, v in enumerate(used_vals)}
    m2 = np.ma.masked_all(class_map.shape, dtype=float)
    m2.data[...] = 0.0                  # see plotting.py: avoid garbage in .data
    for v, i in remap.items():
        m2[class_map == v] = i
    m2.mask = np.ma.getmaskarray(class_map)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 1.0], hspace=0.10)

    if has_cartopy:
        ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        ax_map.pcolormesh(lons, lats, m2, cmap=small,
                          transform=ccrs.PlateCarree(),
                          vmin=-0.5, vmax=len(used_vals) - 0.5)
        if coastlines:
            ax_map.coastlines(linewidth=0.5)
        ax_map.gridlines(linewidth=0.3)
    else:
        ax_map = fig.add_subplot(gs[0])
        ax_map.pcolormesh(lons, lats, m2, cmap=small,
                          vmin=-0.5, vmax=len(used_vals) - 0.5)
    ax_map.set_title(title, size=13, weight="bold")

    ax_key = fig.add_subplot(gs[1])
    plot_defaut_grid(label_dict, cmap, present=present, ax=ax_key,
                     title="KEY TO PHYTOCLIMATIC UNITS")
    return fig, ax_map
