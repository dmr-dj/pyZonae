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
Shared plotting for any classification map.

The same routine renders Koeppen-Geiger and Defaut maps: it takes the integer
class map, the label dictionary and the colormap, and draws a categorical map
with a discrete colorbar. Cartopy is used when available; otherwise it falls
back to a plain matplotlib axes so the package still runs headless.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    _HAS_CARTOPY = True
except Exception:      # pragma: no cover - cartopy optional
    _HAS_CARTOPY = False


def plot_classification(class_map, lons, lats, label_dict, cmap,
                        title="", figsize=(12, 7), coastlines=True):
    """Draw a categorical classification map.

    Parameters
    ----------
    class_map : 2-D array (lat, lon)
        Integer class indices (masked where undefined).
    lons, lats : 1-D arrays
    label_dict : dict[str, int]
        Key -> integer index, as returned by :mod:`pyzonae.cmaps`.
    cmap : matplotlib colormap
    title : str
    """
    # Map each label's integer value to a contiguous color slot 0..N-1, so an
    # out-of-range sentinel value (e.g. Defaut's 10000) cannot stretch the color
    # scale. We remap the class_map through this ordering before plotting.
    ordered_values = sorted(label_dict.values())
    value_to_slot = {v: i for i, v in enumerate(ordered_values)}
    n_slots = len(ordered_values)

    remapped = np.ma.masked_all(class_map.shape, dtype=float)
    for v, slot in value_to_slot.items():
        remapped[class_map == v] = slot
    remapped.mask = np.ma.getmaskarray(class_map)
    class_map = remapped

    vmin, vmax = -0.5, n_slots - 0.5
    tick_positions = list(range(n_slots))

    fig = plt.figure(figsize=figsize)
    if _HAS_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        mesh = ax.pcolormesh(lons, lats, class_map, cmap=cmap,
                             transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
        ax.gridlines()
        if coastlines:
            ax.coastlines()
    else:
        ax = plt.axes()
        mesh = ax.pcolormesh(lons, lats, class_map, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")

    # Labels ordered to match the color slots.
    slot_to_key = {value_to_slot[v]: k for k, v in label_dict.items()}
    ordered_keys = [slot_to_key[i] for i in tick_positions]

    cbar = plt.colorbar(mesh, orientation="horizontal", shrink=0.9, pad=0.05)
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(ordered_keys, fontsize=6, weight="bold")
    for lab in cbar.ax.get_xticklabels():
        lab.set_rotation(90)
    if title:
        ax.set_title(title, size=14)
    fig.tight_layout()
    return fig, ax
