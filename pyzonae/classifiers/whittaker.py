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
Whittaker (1970) biome classification -- nine biomes in (temperature, precipitation).

Whittaker placed the major terrestrial biomes directly in the plane of mean
annual temperature and annual precipitation, drawing their boundaries by hand
from observed vegetation rather than from derived thresholds. Unlike Koeppen or
Feddema, whose classes come from bucketing each axis independently, Whittaker's
boundaries are *oblique polygons*: classification is therefore a point-in-polygon
test, not a comparison against bounds.

The polygons are the plotbiomes digitization of Figure 5.5 in Ricklefs (2008),
itself a modified version of Whittaker's Figure 4.10 (see :mod:`whittaker_data`).
That digitization is a clean partition -- the nine polygons do not overlap -- so
no priority rule is needed. Points outside every polygon (very hot-very dry, very
cold) fall outside the diagram, which Whittaker never claimed to cover in full;
they are returned as a sentinel rather than forced into a biome.

Whittaker himself cautioned that "the pattern is a considerable simplification"
and that the boundaries are "approximate", shiftable by maritime vs continental
effects, soil, and fire. The classification is deliberately coarse; that is a
property of the scheme, not of this implementation.
"""

import numpy as np
from matplotlib.path import Path

from ..whittaker_data import BIOME_POLYGONS, BIOME_NAMES

OUTSIDE = "Outside Whittaker diagram"

# Precompute one matplotlib Path per biome polygon (cheap, done once at import).
_PATHS = {name: Path(np.asarray(verts, dtype=float))
          for name, verts in BIOME_POLYGONS.items()}


def biome_of(temp_c, precip_cm):
    """Return the Whittaker biome for a temperature (degC) and precip (cm/yr).

    Returns the biome name, or :data:`OUTSIDE` if the point lies outside every
    polygon. NaN inputs return ``None``.
    """
    if temp_c is None or precip_cm is None:
        return None
    if (isinstance(temp_c, float) and np.isnan(temp_c)) or \
       (isinstance(precip_cm, float) and np.isnan(precip_cm)):
        return None
    point = (float(temp_c), float(precip_cm))
    for name in BIOME_NAMES:
        if _PATHS[name].contains_point(point):
            return name
    return OUTSIDE


def get_whittaker_classification(arguments):
    """Classify one cell into a Whittaker biome.

    Uses derived slot 3 (annual mean temperature, degC) and slot 5 (annual
    precipitation, mm/yr). Whittaker's precipitation axis is in **cm/yr**, so the
    millimetre value is divided by ten -- the one unit trap here.

    Returns the biome name, :data:`OUTSIDE`, or ``None`` if either input is
    undefined.
    """
    t_ann = arguments[3]
    p_ann_mm = arguments[5]
    if t_ann is None or p_ann_mm is None:
        return None
    if (isinstance(t_ann, float) and np.isnan(t_ann)) or \
       (isinstance(p_ann_mm, float) and np.isnan(p_ann_mm)):
        return None
    return biome_of(t_ann, p_ann_mm / 10.0)     # mm/yr -> cm/yr
