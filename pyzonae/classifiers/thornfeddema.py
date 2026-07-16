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
Thornthwaite-Feddema (2005) global climate classification -- two main factors.

This is the *revised* Thornthwaite-type classification of Feddema (2005), not the
1948 original. It is built entirely on potential evapotranspiration and a water
balance, which is what makes it a genuine water-availability scheme rather than a
temperature/precipitation threshold scheme.

This first version implements the two primary factors (Feddema's Tables 5 and 6):

* **Moisture** from the Willmott-Feddema index Im, six classes symmetric about 0.
* **Thermal** from the annual PE, six equal 300 mm classes.

The two seasonality factors (Tables 7-8) are deliberately left for a later
version; the class key here is ``"<Moisture> <Thermal>"``, e.g. ``"Moist Warm"``.

Reference
---------
Feddema, J. J. (2005). A revised Thornthwaite-type global climate classification.
Physical Geography 26(6), 442-466.
"""

import numpy as np

# Table 5: moisture types from the annual moisture index Im, driest to wettest.
# Each entry is the LOWER bound; the class holds Im in [lower, next_lower).
MOISTURE_TYPES = [
    ("Arid", -1.00),
    ("Semiarid", -0.66),
    ("Dry", -0.33),
    ("Moist", 0.00),
    ("Wet", 0.33),
    ("Saturated", 0.66),
]

# Table 6: thermal types from annual PE (mm), cold to hot. Lower bound each.
THERMAL_TYPES = [
    ("Frost", 0.0),
    ("Cold", 300.0),
    ("Cool", 600.0),
    ("Warm", 900.0),
    ("Hot", 1200.0),
    ("Torrid", 1500.0),
]


def _bucket(value, table):
    """Name of the class whose lower bound is the greatest not exceeding value."""
    name = table[0][0]
    for label, lo in table:
        if value >= lo:
            name = label
        else:
            break
    return name


def moisture_type(im):
    """Feddema moisture class name for a moisture index in [-1, 1]."""
    if im is None or (isinstance(im, float) and np.isnan(im)):
        return None
    return _bucket(im, MOISTURE_TYPES)


def thermal_type(pe_ann):
    """Feddema thermal class name for an annual PE in mm."""
    if pe_ann is None or (isinstance(pe_ann, float) and np.isnan(pe_ann)):
        return None
    return _bucket(pe_ann, THERMAL_TYPES)


def get_thornfeddema_classification(arguments, factors=2):
    """Classify one cell into a Thornthwaite-Feddema type.

    Parameters
    ----------
    arguments : sequence
        Derived indices. Uses slot 17 (annual PE) and 18 (moisture index Im).
    factors : {2, 4}
        Number of Feddema factors to apply.

        * ``2`` (default) -- the two primary factors, moisture x thermal
          (Feddema Tables 5-6). Key ``"<Moisture> <Thermal>"``, e.g.
          ``"Moist Warm"``.
        * ``4`` -- adds the two seasonality factors (Tables 7-8). Not yet
          implemented; requested here so the interface is stable, but it raises
          rather than silently returning a two-factor key.

    Returns
    -------
    str or None
        The class key, or ``None`` if a factor is undefined.
    """
    if factors not in (2, 4):
        raise ValueError(f"factors must be 2 or 4, got {factors!r}")

    pe_ann = arguments[17]
    im = arguments[18]
    m = moisture_type(im)
    t = thermal_type(pe_ann)
    if m is None or t is None:
        return None

    if factors == 2:
        return f"{m} {t}"

    # factors == 4: the seasonality and cause factors (Tables 7-8) still to come.
    raise NotImplementedError(
        "The four-factor Thornthwaite-Feddema classification (adding seasonality "
        "and its cause, Feddema Tables 7-8) is not implemented yet. Use factors=2 "
        "for the moisture x thermal classification."
    )
