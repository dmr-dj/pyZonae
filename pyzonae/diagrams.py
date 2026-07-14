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

# Which classifications have a decision-space diagram.
DIAGRAMS = ("Defaut96", "Holdridge")


def plot_diagram(typ_classification, class_map, fields_args, label_dict, cmap,
                 **kwargs):
    """Draw the decision-space diagram appropriate to ``typ_classification``.

    Parameters
    ----------
    typ_classification : {"Defaut96", "Holdridge"}
    class_map : masked 2-D array of class indices (from ``run_classification``)
    fields_args : the derived-index stack (from ``build_arguments``)
    label_dict, cmap : the vocabulary and colours for that classification
    **kwargs
        Passed through to the underlying plotter. Useful ones:

        Defaut96  -- ``markers`` (bool), ``facet`` (bool), ``point_size``,
                     ``qn2_max``
        Holdridge -- ``rule`` ({"fuzzy", "strict"}), ``hexagons`` (bool),
                     ``markers`` (bool), ``point_size``

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
    raise ValueError(
        f"No decision-space diagram for '{typ_classification}'. "
        f"Available: {DIAGRAMS}. The Koeppen-Geiger variants classify on many "
        f"interacting criteria and have no faithful low-dimensional picture."
    )
