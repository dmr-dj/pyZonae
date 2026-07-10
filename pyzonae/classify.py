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
Unified classification dispatch.

A single entry point, :func:`classify_cell`, maps a classification *name* to the
right classifier and returns a classification *key* (str). This is where the new
``"Defaut96"`` option lives, side by side with the Koeppen-Geiger variants.

All classifiers share the same calling convention: they take the derived
``arguments`` vector for one grid cell (see :mod:`pyzonae.derive`) and return a
key that :mod:`pyzonae.cmaps` knows how to color.
"""

from .classifiers import koeppen as _kg
from .classifiers import defaut as _df

# Names accepted for typ_classification.
CLASSIFICATIONS = ("kottek", "peel", "cannon", "trewartha", "Defaut96")


def classify_cell(typ_classification, arguments):
    """Return the classification key for one grid cell.

    Parameters
    ----------
    typ_classification : str
        One of :data:`CLASSIFICATIONS`.
    arguments : sequence of float
        The derived index vector for a single cell.

    Returns
    -------
    str
        Classification key (e.g. ``"Cfb"`` or ``"HA1a,b"``).
    """
    if typ_classification in ("kottek", "peel"):
        return _kg.get_kg_classification(arguments, vers=typ_classification)
    if typ_classification == "cannon":
        return _kg.get_kg_classification_Cannon(arguments)
    if typ_classification == "trewartha":
        return _kg.get_kg_classification_Trewartha(arguments)
    if typ_classification == "Defaut96":
        return _df.get_defaut_classification(arguments)
    raise ValueError(
        f"Unknown classification '{typ_classification}'. "
        f"Known: {CLASSIFICATIONS}"
    )
