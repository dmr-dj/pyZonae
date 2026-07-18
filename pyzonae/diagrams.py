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
Decision-space diagrams: plot a classification in the space of its own variables.

A map shows *where* each class falls on the globe. A decision-space diagram shows
*why*: every grid cell is placed at its coordinates in the classifying variables,
and the decision boundaries are drawn on top, straight from the classifier's own
functions. Boundaries that are curves (as Defaut's are) look like curves;
densities of climate space that are empty look empty.

The right diagram depends on the *structure* of the classification, not on taste:

* **Defaut (1996)** has three genuinely independent axes (aridity Qn2, thermal
  degree, continentality). Its aridity index depends on precipitation terms that
  are independent of the other two, so no projection collapses it: a triangular
  diagram would invent a constraint that does not exist. It is plotted as a
  scatter in (Qn2, temperature), with continentality carried by marker shape.

* **Holdridge (1967)** has an exact constraint, PETR = Tbio * 58.93 / P, which is
  linear in logarithms. The three axes therefore lie on a plane -- two degrees of
  freedom -- and the classification projects without loss onto his triangular,
  hexagon-tiled diagram.

Same idea, two geometries, each forced by the mathematics of the scheme.
"""

from .decision_space import plot_defaut_space
from .holdridge_triangle import plot_holdridge_triangle
from .whittaker_diagram import plot_whittaker_diagram

# Which classifications have a decision-space diagram.
DIAGRAMS = ("Defaut96", "Holdridge", "Whittaker")


def plot_diagram(typ_classification, class_map, fields_args, label_dict, cmap,
                 **kwargs):
    """Draw the decision-space diagram appropriate to ``typ_classification``.

    Parameters
    ----------
    typ_classification : {"Defaut96", "Holdridge", "Whittaker"}
    class_map : masked 2-D array of class indices (from ``run_classification``)
    fields_args : the derived-index stack (from ``build_arguments``)
    label_dict, cmap : the vocabulary and colours for that classification
    **kwargs
        Passed through to the underlying plotter. Useful ones:

        Defaut96  -- ``markers`` (bool), ``facet`` (bool), ``point_size``,
                     ``qn2_max``
        Holdridge -- ``rule`` ({"fuzzy", "strict"}), ``hexagons`` (bool),
                     ``markers`` (bool), ``point_size``
        Whittaker -- ``colour_points_by_biome`` (bool), ``clip_to_biomes``
                     (bool), ``point_size``

    Returns
    -------
    fig, ax

    Raises
    ------
    ValueError
        If the classification has no decision-space diagram. The Koeppen-Geiger
        variants do not: their rules are a nest of conditionals on many variables
        rather than boundaries in a low-dimensional space, so there is no honest
        2-D picture of them.
    """
    if typ_classification == "Defaut96":
        return plot_defaut_space(class_map, fields_args, label_dict, cmap,
                                 **kwargs)
    if typ_classification == "Holdridge":
        return plot_holdridge_triangle(class_map, fields_args, label_dict, cmap,
                                       **kwargs)
    if typ_classification == "Whittaker":
        return plot_whittaker_diagram(class_map, fields_args, label_dict, cmap,
                                      **kwargs)
    raise ValueError(
        f"No decision-space diagram for '{typ_classification}'. "
        f"Available: {DIAGRAMS}. The Koeppen-Geiger variants classify on many "
        f"interacting criteria and have no faithful low-dimensional picture."
    )


def plot_cross_diagram(space, colour_by, class_maps, label_dicts, cmaps,
                       fields_args, legend_max=30, **kwargs):
    """Draw one classification's *space* coloured by another's *classes*.

    This is a comparison of taxonomies: the axes and decision boundaries come
    from ``space``, while the colour of each cell comes from ``colour_by``. It
    answers "where do Köppen's classes fall in Defaut's decision space, and where
    do the two schemes disagree?".

    It works because the two are already decoupled: the diagram takes its
    geometry from the shared derived-index stack, and its colours from whichever
    class map it is handed. Nothing special is needed beyond labelling the figure
    honestly, which is what this wrapper adds.

    Parameters
    ----------
    space : str
        Classification whose decision space is drawn (one of :data:`DIAGRAMS`).
    colour_by : str
        Classification whose classes colour the points. May be any classification,
        including ``space`` itself (which reproduces the ordinary diagram).
    class_maps, label_dicts, cmaps : dict
        Keyed by classification name; must contain at least ``colour_by``.
    fields_args : the derived-index stack.
    legend_max : int
        Above this many classes, the colour legend is dropped as unreadable.

    Returns
    -------
    fig, ax
    """
    import numpy as np
    from matplotlib.lines import Line2D

    if space not in DIAGRAMS:
        raise ValueError(f"'{space}' has no decision space. Available: {DIAGRAMS}")
    if colour_by not in class_maps:
        raise KeyError(f"no class map supplied for '{colour_by}'")

    cmap_used = class_maps[colour_by]
    labels_used = label_dicts[colour_by]
    colours_used = cmaps[colour_by]

    title = kwargs.pop(
        "title",
        f"{colour_by} classes in {space}'s decision space"
        if colour_by != space else f"{space} decision space",
    )
    fig, ax = plot_diagram(space, cmap_used, fields_args, labels_used,
                           colours_used, title=title, **kwargs)

    # A cross diagram is unreadable without saying which colours mean what.
    present = sorted(set(int(v) for v in np.ma.compressed(cmap_used)))
    inv = {v: k for k, v in labels_used.items()}
    values = sorted(labels_used.values())
    slot = {v: i for i, v in enumerate(values)}
    if len(present) <= legend_max:
        handles = [
            Line2D([], [], linestyle="none", marker="o", markersize=5,
                   markerfacecolor=colours_used(slot[v]),
                   markeredgecolor="none", label=inv.get(v, str(v)))
            for v in present
        ]
        axes = fig.axes
        target = axes[0] if len(axes) == 1 else axes[min(1, len(axes) - 1)]
        target.legend(handles=handles, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), fontsize=6.5, frameon=False,
                      ncol=1 if len(handles) <= 16 else 2,
                      title=f"{colour_by} class", title_fontsize=7.5)
        fig.subplots_adjust(right=0.82)
    return fig, ax
