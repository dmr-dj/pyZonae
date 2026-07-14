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
Holdridge life zones.

Implements the classification as operationalised by Lugo et al. (1999), which
is the clearest published algorithmic description of Holdridge (1967).

Key property (and the reason this scheme fits pyZonae's constraints): despite
their names, the "latitudinal regions" (polar ... tropical) are **not**
geographic. They are labels for intervals of *sea-level biotemperature*. A cell
in Florida is classified subtropical because its climate says so, not because of
where it sits. The only non-climatic input is elevation, and that is a derivable
field, not a coordinate.

Algorithm
---------
1. Biotemperature (Holdridge 1967, eq. 2 in Lugo et al.)::

       Tbio = sum( Tavg[i] if 0 < Tavg[i] < 30 else 0 ) / 12

   Note the denominator is **12**, not the count of qualifying months. Months
   outside (0, 30) contribute zero, they are not dropped.

2. Sea-level biotemperature: lapse the monthly temperatures to sea level with
   -6.0 degC/km, then apply the same formula::

       T_sealevel[i] = Tavg[i] + 6.0 * elevation_km
       T0bio         = sum( T_sealevel[i] if 0 < . < 30 else 0 ) / 12

3. PET ratio::

       PETR = (Tbio * 58.93) / P_ann

4. Latitudinal region  <- from T0bio (plus the frost line, if available).
5. Altitudinal belt    <- from the *offset* between the region implied by T0bio
   and the region implied by the actual Tbio. If they agree, the belt is basal.
   This works because Holdridge put latitudinal regions and altitudinal belts on
   the same logarithmic biotemperature scale.
6. Humidity province   <- from PETR on a log2 progression.

Thresholds
----------
Two rule sets are supported, selected by ``rule``:

* ``"fuzzy"``   (default) - Lugo et al.'s adjusted thresholds. These exist to
  stop a trivial elevation difference from flipping the life zone: their worked
  example has a site whose actual and sea-level Tbio differ by only 0.1 degC
  (i.e. 17 m of elevation) yet which the strict rule pushes into a different
  altitudinal belt.
* ``"strict"``  - Holdridge's original thresholds, artefacts included. Useful
  for reproducing the original system and for quantifying what the fuzzy rule
  actually changes.

Reference
---------
Lugo, A. E., Brown, S. L., Dodson, R., Smith, T. S., Shugart, H. H. (1999).
The Holdridge life zones of the conterminous United States in relation to
ecosystem mapping. Journal of Biogeography 26, 1025-1038.
Holdridge, L. R. (1967). Life Zone Ecology. Tropical Science Center, San Jose.
"""

import numpy as np

# Table 1 of Lugo et al. (1999): biotemperature thresholds (degC) separating the
# latitudinal regions, in both the original and the "fuzzy" variant.
#
# The value is the *lower* bound of the named region, i.e. a cell belongs to
# region k when Tbio >= THRESHOLDS[k].
LATITUDINAL_REGIONS = ["Polar", "Subpolar", "Boreal", "CoolTemperate",
                       "WarmTemperate", "Tropical"]

THRESHOLDS = {
    # region ->      (strict, fuzzy)
    "Polar":         (0.0,   0.0),    # everything below the subpolar bound
    "Subpolar":      (1.5,   1.68),
    "Boreal":        (3.0,   3.36),
    "CoolTemperate": (6.0,   6.72),
    "WarmTemperate": (12.0,  13.44),  # warm temperate / subtropical
    "Tropical":      (24.0,  26.89),
}

# Altitudinal belts. Holdridge put latitudinal regions and altitudinal belts on
# the SAME logarithmic biotemperature scale ("a logarithmic progression from
# 1.5 degC (nival and polar) to 24 degC (basal and tropical)", Lugo et al.).
# There are therefore exactly as many belts as regions, and the belt is read off
# as the number of steps between the region implied by the sea-level Tbio and
# the region implied by the actual Tbio:
#
#     offset 0 -> Basal        (actual and sea-level agree)
#     offset 1 -> LowerMontane
#     offset 2 -> Montane
#     offset 3 -> Subalpine
#     offset 4 -> Alpine
#     offset 5 -> Nival        (maximum: Tropical at sea level, Polar in fact)
ALTITUDINAL_BELTS = ["Basal", "LowerMontane", "Montane",
                     "Subalpine", "Alpine", "Nival"]

# Humidity provinces from the PET ratio, on Holdridge's log2 progression.
# Boundaries: 0.125, 0.25, 0.5, 1, 2, 4, 8, 16 -> eight provinces.
HUMIDITY_BOUNDS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
HUMIDITY_PROVINCES = ["Superhumid", "Perhumid", "Humid", "Subhumid",
                      "Semiarid", "Arid", "Perarid", "Superarid"]

LAPSE_RATE_C_PER_KM = 6.0
PET_COEFFICIENT = 58.93

# When the frost line is unavailable, warm temperate and subtropical cannot be
# told apart, so they are reported as a single, explicit class.
WARM_TEMPERATE_MERGED = "WarmTemperate/Subtropical"
SUBTROPICAL = "Subtropical"


def biotemperature(monthly_t):
    """Holdridge biotemperature from 12 monthly mean temperatures (degC).

    Values outside (0, 30) degC contribute zero; the denominator stays 12.
    """
    t = np.asarray(monthly_t, dtype=float)
    contrib = np.where((t > 0.0) & (t < 30.0), t, 0.0)
    return float(np.sum(contrib) / 12.0)


def sealevel_biotemperature(monthly_t, elevation_m):
    """Biotemperature after lapsing the monthly temperatures to sea level."""
    t = np.asarray(monthly_t, dtype=float) + LAPSE_RATE_C_PER_KM * (elevation_m / 1000.0)
    return biotemperature(t)


def pet_ratio(tbio, p_ann):
    """Potential-evapotranspiration ratio, PETR = (Tbio * 58.93) / P_ann."""
    if p_ann is None or p_ann <= 0 or np.isnan(p_ann):
        return np.nan
    return (tbio * PET_COEFFICIENT) / p_ann


def _region_index(tbio, rule):
    """Index into LATITUDINAL_REGIONS for a given biotemperature."""
    col = 1 if rule == "fuzzy" else 0
    idx = 0
    for k, name in enumerate(LATITUDINAL_REGIONS):
        if tbio >= THRESHOLDS[name][col]:
            idx = k
    return idx


def _humidity_province(petr):
    """Name of the humidity province for a PET ratio."""
    if np.isnan(petr):
        return None
    for bound, name in zip(HUMIDITY_BOUNDS, HUMIDITY_PROVINCES):
        if petr < bound:
            return name
    return HUMIDITY_PROVINCES[-1]  # Superarid, beyond the last bound


def get_holdridge_classification(arguments, rule="fuzzy", frost_free=None):
    """Classify one grid cell into a Holdridge life zone.

    Parameters
    ----------
    arguments : sequence
        Derived indices for the cell. This classifier uses the extended slots
        (see :mod:`pyzonae.derive`)::

            5:  P_ann     annual precipitation                (mm)
           15:  Tbio      biotemperature                      (degC)
           16:  T0bio     sea-level biotemperature            (degC)

    rule : {"fuzzy", "strict"}
        Threshold set. ``"fuzzy"`` follows Lugo et al. (1999); ``"strict"``
        uses Holdridge's original thresholds.
    frost_free : bool or None
        True if the cell is frost-free, False if frost-prone, None if unknown.
        This only ever splits WarmTemperate from Subtropical. When None, the two
        are reported merged, rather than guessing a boundary.

    Returns
    -------
    str
        A life-zone key, e.g. ``"WarmTemperate Basal Humid"``.
    """
    p_ann = arguments[5]
    tbio = arguments[15]
    t0bio = arguments[16]

    if np.isnan(tbio) or np.isnan(t0bio) or np.isnan(p_ann):
        return None

    # Tbio == 0 means every month is at or below freezing (or above 30 degC):
    # there is no growing season at all. Holdridge's PET ratio, PETR = Tbio*58.93/P,
    # then collapses to 0, which would fall in the wettest humidity province and
    # label an ice sheet "Superhumid". That is meaningless: the humidity axis is
    # undefined without any biologically active month. Such cells sit outside the
    # model's domain and are reported as such, rather than silently mis-assigned.
    if tbio <= 0.0:
        return None

    # Latitudinal region from the SEA-LEVEL biotemperature.
    region_idx = _region_index(t0bio, rule)
    region = LATITUDINAL_REGIONS[region_idx]

    # The warm-temperate / subtropical split is decided by frost, not by
    # temperature. Without a frost line we refuse to guess.
    if region == "WarmTemperate":
        if frost_free is True:
            region = SUBTROPICAL
        elif frost_free is None:
            region = WARM_TEMPERATE_MERGED
        # frost_free is False -> stays "WarmTemperate"

    # Altitudinal belt from the offset between sea-level and actual Tbio.
    # Same logarithmic scale, so a colder actual Tbio means a higher belt.
    actual_idx = _region_index(tbio, rule)
    belt_idx = max(0, region_idx - actual_idx)
    belt_idx = min(belt_idx, len(ALTITUDINAL_BELTS) - 1)
    belt = ALTITUDINAL_BELTS[belt_idx]

    # Humidity province from the PET ratio.
    petr = pet_ratio(tbio, p_ann)
    province = _humidity_province(petr)
    if province is None:
        return None

    return f"{region} {belt} {province}"
