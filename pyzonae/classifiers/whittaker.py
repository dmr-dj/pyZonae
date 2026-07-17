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
itself a modified version of Whittaker's Figure 4.10 (see :mod:`pyzonae.classifiers.whittaker_data`).
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

from .whittaker_data import BIOME_POLYGONS, BIOME_NAMES

OUTSIDE = "Outside Whittaker diagram"

# Precompute one matplotlib Path per biome polygon (cheap, done once at import).
_PATHS = {name: Path(np.asarray(verts, dtype=float))
          for name, verts in BIOME_POLYGONS.items()}

# Precompute polygon vertex arrays for the nearest-edge distance used to fill
# digitization gaps (see biome_of).
_VERTS = {name: np.asarray(verts, dtype=float)
          for name, verts in BIOME_POLYGONS.items()}

# Default cap (in the mixed degC/cm plane) below which a point that misses every
# polygon is snapped to the nearest biome rather than called OUTSIDE. The
# plotbiomes polygons do not perfectly tile the plane: narrow gaps exist between
# adjacent biomes (e.g. between Tundra, Boreal forest and Woodland near -12 degC),
# and real cells land in them. Those gaps are digitization artifacts, not places
# Whittaker meant to leave blank, so points that fall a short distance from a
# polygon are assigned to it. Genuinely out-of-envelope points (too cold, too
# hot) sit far from every polygon and stay OUTSIDE.
#
# The distance mixes degC and cm, so it is a pragmatic snapping tolerance, not a
# physically meaningful metric; the default is deliberately small.
DEFAULT_FILL_TOLERANCE = 5.0


def _point_segment_distance(p, a, b):
    ap = p - a
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom == 0.0 else float(np.clip(np.dot(ap, ab) / denom, 0.0, 1.0))
    return float(np.hypot(*(a + t * ab - p)))


def _nearest_biome(temp_c, precip_cm):
    """Nearest biome polygon and the point's distance to its boundary."""
    pt = np.array([temp_c, precip_cm], dtype=float)
    best, best_d = None, np.inf
    for name in BIOME_NAMES:
        v = _VERTS[name]
        d = min(_point_segment_distance(pt, v[i], v[(i + 1) % len(v)])
                for i in range(len(v)))
        if d < best_d:
            best_d, best = d, name
    return best, best_d


def biome_of(temp_c, precip_cm, fill_tolerance=DEFAULT_FILL_TOLERANCE):
    """Return the Whittaker biome for a temperature (degC) and precip (cm/yr).

    A point inside a polygon returns that biome. A point that misses every
    polygon but lies within ``fill_tolerance`` of one is snapped to the nearest
    biome, which fills the narrow gaps left by the digitization (see
    :data:`DEFAULT_FILL_TOLERANCE`). Set ``fill_tolerance=0`` to disable snapping
    and return :data:`OUTSIDE` for any point outside the polygons.

    Returns the biome name, :data:`OUTSIDE`, or ``None`` for NaN inputs.
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
    if fill_tolerance and fill_tolerance > 0:
        name, dist = _nearest_biome(point[0], point[1])
        if dist <= fill_tolerance:
            return name
    return OUTSIDE


def get_whittaker_classification(arguments, fill_tolerance=DEFAULT_FILL_TOLERANCE):
    """Classify one cell into a Whittaker biome.

    Uses derived slot 3 (annual mean temperature, degC) and slot 5 (annual
    precipitation, mm/yr). Whittaker's precipitation axis is in **cm/yr**, so the
    millimetre value is divided by ten -- the one unit trap here.

    ``fill_tolerance`` controls gap-filling; see :func:`biome_of`.

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
    return biome_of(t_ann, p_ann_mm / 10.0, fill_tolerance=fill_tolerance)
