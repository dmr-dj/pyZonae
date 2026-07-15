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
pyzonae: gridded climate classification with pluggable rule sets.

Supported classifications (pass as ``typ_classification``):

    "kottek"     Koeppen-Geiger, Kottek et al. 2006
    "peel"       Koeppen-Geiger, Peel et al. 2007
    "cannon"     Koeppen-Geiger, Cannon 2012
    "trewartha"  Trewartha, Belda et al. 2014
    "Defaut96"   Defaut (1996) bioclimatic stages

High-level usage::

    from pyzonae import run_classification
    class_map, label_dict, cmap, lons, lats = run_classification(
        typ_classification="Defaut96",
        tas_file="test-data/tas.nc",
        pr_file="test-data/pr.nc",
    )

Besides the map, Defaut96 and Holdridge can be drawn in their own *decision
space* -- every cell placed at its coordinates in the classifying variables, with
the decision boundaries on top::

    from pyzonae import load_climatology, build_arguments, plot_diagram
    fields = load_climatology(tas_file, pr_file, sftlf_file=..., orog_file=...)
    args, shape, lats, lons = build_arguments(fields)
    fig, ax = plot_diagram("Holdridge", class_map, args, label_dict, cmap)
"""

from .classify import classify_cell, CLASSIFICATIONS
from .cmaps import get_cmap
from .run import run_classification
from .derive import build_arguments
from .io import load_climatology
from .diagrams import plot_diagram, plot_cross_diagram, DIAGRAMS

__all__ = [
    "classify_cell", "CLASSIFICATIONS", "get_cmap", "run_classification",
    "build_arguments", "load_climatology", "plot_diagram",
    "plot_cross_diagram", "DIAGRAMS",
]
__version__ = "0.1.0"
