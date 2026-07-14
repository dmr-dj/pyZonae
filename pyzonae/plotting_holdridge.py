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
Structured plotting for Holdridge life zones.

A Holdridge life zone is not an atomic category: it is the intersection of three
independent axes (latitudinal region x altitudinal belt x humidity province).
Flattening several hundred combinations into one colorbar throws that structure
away and produces an unreadable strip of labels.

:func:`plot_holdridge` instead draws one panel per axis, which is how published
Holdridge maps are read: you look up a colour's hue to get the humidity, its
lightness to get the thermal region, and consult the belt panel for relief.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import cartopy.crs as ccrs
    _HAS_CARTOPY = True
except Exception:      # pragma: no cover
    _HAS_CARTOPY = False

from .classifiers.holdridge import (
    LATITUDINAL_REGIONS, ALTITUDINAL_BELTS, HUMIDITY_PROVINCES,
    SUBTROPICAL, WARM_TEMPERATE_MERGED,
)

SENTINEL_KEY = "Outside Holdridge model"


def _decompose(key):
    """Split a life-zone key into (region, belt, province)."""
    if key == SENTINEL_KEY:
        return None
    parts = key.split()
    # region may itself contain no space (WarmTemperate/Subtropical is one token)
    return parts[0], parts[1], parts[2]


def plot_holdridge(class_map, lons, lats, label_dict, cmap,
                   title="Holdridge life zones", figsize=(13, 8),
                   coastlines=True):
    """Draw a Holdridge map with a three-panel, axis-wise legend.

    Only the classes actually present are described, and they are described by
    axis rather than as a flat list.

    Returns
    -------
    fig, ax
    """
    inv = {v: k for k, v in label_dict.items()}
    present_vals = sorted(set(int(v) for v in np.ma.compressed(class_map)))
    present_keys = [inv[v] for v in present_vals if v in inv]

    # Which value along each axis actually occurs?
    used_regions, used_belts, used_provs = [], [], []
    for k in present_keys:
        d = _decompose(k)
        if d is None:
            continue
        r, b, p = d
        if r not in used_regions:
            used_regions.append(r)
        if b not in used_belts:
            used_belts.append(b)
        if p not in used_provs:
            used_provs.append(p)

    # Keep each axis in its natural (thermal / altitudinal / moisture) order.
    region_order = LATITUDINAL_REGIONS[:4] + [
        "WarmTemperate", SUBTROPICAL, WARM_TEMPERATE_MERGED, "Tropical"]
    used_regions = [r for r in region_order if r in used_regions]
    used_belts = [b for b in ALTITUDINAL_BELTS if b in used_belts]
    used_provs = [p for p in HUMIDITY_PROVINCES if p in used_provs]

    # Remap to contiguous colour slots, exactly as plot_classification does.
    full_values = sorted(label_dict.values())
    full_slot = {v: i for i, v in enumerate(full_values)}
    value_to_slot = {v: i for i, v in enumerate(present_vals)}
    picked = [cmap(full_slot[v]) for v in present_vals]
    from matplotlib.colors import ListedColormap
    small_cmap = ListedColormap(picked)

    remapped = np.ma.masked_all(class_map.shape, dtype=float)
    for v, slot in value_to_slot.items():
        remapped[class_map == v] = slot
    remapped.mask = np.ma.getmaskarray(class_map)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1], hspace=0.25, wspace=0.25)

    if _HAS_CARTOPY:
        ax = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
        ax.pcolormesh(lons, lats, remapped, cmap=small_cmap,
                      transform=ccrs.PlateCarree(),
                      vmin=-0.5, vmax=len(present_vals) - 0.5)
        if coastlines:
            ax.coastlines(linewidth=0.5)
        ax.gridlines(linewidth=0.3)
    else:
        ax = fig.add_subplot(gs[0, :])
        ax.pcolormesh(lons, lats, remapped, cmap=small_cmap,
                      vmin=-0.5, vmax=len(present_vals) - 0.5)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    ax.set_title(title, size=13)

    # --- Three legend panels, one per axis ---------------------------------
    def _repr_colour(axis, value):
        """A representative colour for one value of one axis: average the colours
        of every present zone sharing that value."""
        cols = []
        for v in present_vals:
            k = inv.get(v)
            d = _decompose(k) if k else None
            if d is None:
                continue
            if d[axis] == value:
                cols.append(np.array(cmap(full_slot[v])[:3]))
        if not cols:
            return (0.5, 0.5, 0.5)
        return tuple(np.mean(cols, axis=0))

    panels = [
        (0, "Latitudinal region\n(sea-level biotemperature)", used_regions),
        (1, "Altitudinal belt\n(actual vs sea-level Tbio)", used_belts),
        (2, "Humidity province\n(PET ratio)", used_provs),
    ]
    for col, (axis, heading, values) in enumerate(panels):
        lax = fig.add_subplot(gs[1, col])
        lax.axis("off")
        handles = [mpatches.Patch(facecolor=_repr_colour(axis, v),
                                  edgecolor="0.3", label=v) for v in values]
        lax.legend(handles=handles, loc="upper center", frameon=False,
                   fontsize=7, ncol=2 if len(values) > 4 else 1,
                   title=heading, title_fontsize=8)

    # The sentinel, if present, deserves an explicit mention: those cells are
    # not a life zone, they are outside the model (no growing season).
    if any(inv.get(v) == SENTINEL_KEY for v in present_vals):
        fig.text(0.5, 0.02,
                 "Black: outside the Holdridge model "
                 "(biotemperature = 0, i.e. no growing season)",
                 ha="center", fontsize=7, style="italic")

    return fig, ax
