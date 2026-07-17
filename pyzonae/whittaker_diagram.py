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
The Whittaker biome diagram: the classification drawn in its own two axes.

Whittaker's scheme is unusual among the pyZonae classifications in that its
decision space *is* the raw data plane -- mean annual temperature against annual
precipitation -- with no index to derive and no projection to apply. The diagram
is therefore the most faithful of the three: the coloured polygons are exactly
the biome boundaries the classifier tests, and each plotted cell sits at the very
coordinates the classifier reads. This reproduces the familiar Whittaker diagram
(e.g. Ricklefs 2008, Fig. 5.5), with the dataset's own cells scattered on top.

Framing
-------
A global model reaches temperatures far colder than any vegetated biome (ice
sheets down to tens of degrees below zero). By default the axes are framed on
Whittaker's envelope, so the diagram stays legible and matches the published
figure; cells colder or wetter than the envelope are counted and reported rather
than silently dropped. Pass ``clip_to_biomes=False`` to widen the axes to the
data instead.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from .classifiers.whittaker_data import BIOME_POLYGONS, BIOME_NAMES, BIOME_COLORS
from .classifiers.whittaker import biome_of, OUTSIDE, DEFAULT_FILL_TOLERANCE


def _polygon_extent():
    allv = np.vstack([np.asarray(v) for v in BIOME_POLYGONS.values()])
    return (allv[:, 0].min(), allv[:, 0].max(),
            allv[:, 1].min(), allv[:, 1].max())


def plot_whittaker_diagram(class_map, fields_args, label_dict, cmap,
                           colour_points_by_biome=False, point_size=6.0,
                           point_alpha=0.45, polygon_alpha=None,
                           clip_to_biomes=True,
                           fill_tolerance=DEFAULT_FILL_TOLERANCE,
                           figsize=(9, 7),
                           title="Whittaker biome diagram"):
    """Draw the Whittaker diagram with the dataset's cells scattered on it.

    Parameters
    ----------
    class_map : masked 2-D array of class indices (unused for placement; the
        points are placed from ``fields_args`` so the diagram matches exactly
        what the classifier saw). Accepted for a uniform diagram signature.
    fields_args : the derived-index stack; slot 3 is temperature (degC), slot 5
        is precipitation (mm/yr, converted to cm here).
    label_dict, cmap : the Whittaker vocabulary and colours (for the point
        colouring option; the polygons use the official biome colours).
    colour_points_by_biome : bool
        If True each point takes its biome's colour; if False all points are a
        neutral dark, which reads better as a density over the coloured polygons.
    polygon_alpha : float or None
        Opacity of the background biome polygons. Defaults to solid (1.0) for the
        neutral-density view, and to a lightened 0.35 when points are coloured by
        biome, so the points read clearly on top rather than merging into a
        same-colour background (the same treatment as the Holdridge diagram).
    clip_to_biomes : bool
        Frame the axes on Whittaker's envelope (default) or on the data.
    fill_tolerance : float
        Gap-filling tolerance used when colouring points by biome; see
        :func:`pyzonae.classifiers.whittaker.biome_of`.

    Returns
    -------
    fig, ax
    """
    a = fields_args
    flat = a.reshape(a.shape[0], -1)
    mask = np.ma.getmaskarray(flat)
    ok = ~(mask[0] | mask[4])
    T = np.asarray(flat[3])[ok]
    Pcm = np.asarray(flat[5])[ok] / 10.0        # mm/yr -> cm/yr

    # When points carry the biome colours, a solid background of the same colours
    # swallows them; lighten it and let the points sit clearly on top.
    if polygon_alpha is None:
        polygon_alpha = 0.35 if colour_points_by_biome else 1.0

    fig, ax = plt.subplots(figsize=figsize)

    # --- Polygons, in the official Ricklefs colours -----------------------
    # Drawn back to front is irrelevant here since they do not overlap, but we
    # keep the canonical order for a stable legend.
    for name in BIOME_NAMES:
        verts = np.asarray(BIOME_POLYGONS[name])
        ax.add_patch(MplPolygon(verts, closed=True,
                                facecolor=BIOME_COLORS[name], edgecolor="white",
                                linewidth=1.0, alpha=polygon_alpha, zorder=1,
                                label=name))

    # --- The dataset's cells, on top --------------------------------------
    if colour_points_by_biome:
        colours = []
        for t, p in zip(T, Pcm):
            b = biome_of(t, p, fill_tolerance=fill_tolerance)
            colours.append(BIOME_COLORS.get(b, "0.2"))
        # Points opaque and edged so each biome colour stands out against the
        # lightened polygon of the same hue.
        ax.scatter(T, Pcm, s=point_size * 1.4, c=colours, alpha=0.9,
                   linewidths=0.3, edgecolors="0.25", zorder=3)
    else:
        ax.scatter(T, Pcm, s=point_size, c="0.15", alpha=point_alpha,
                   linewidths=0, zorder=3)

    tmin, tmax, pmin, pmax = _polygon_extent()
    n_out = 0
    if clip_to_biomes:
        ax.set_xlim(tmin - 1.0, tmax + 1.0)
        ax.set_ylim(pmin - 5.0, pmax + 10.0)
        n_out = int(((T < tmin) | (T > tmax) | (Pcm < pmin) | (Pcm > pmax)).sum())
    else:
        ax.set_xlim(T.min() - 2.0, max(T.max(), tmax) + 2.0)
        ax.set_ylim(min(Pcm.min(), pmin) - 5.0, max(Pcm.max(), pmax) + 10.0)

    ax.set_xlabel("Mean annual temperature (°C)")
    ax.set_ylabel("Annual precipitation (cm)")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.9,
              title="Biome", title_fontsize=8)

    if n_out:
        ax.text(0.99, 0.02,
                f"{n_out} cells outside the diagram\n"
                f"(colder or wetter than any biome)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                style="italic", color="0.35")

    fig.tight_layout()
    return fig, ax
