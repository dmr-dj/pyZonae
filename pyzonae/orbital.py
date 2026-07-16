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
Orbital parameters and day length.

Thornthwaite's potential evapotranspiration needs the length of daylight, which
depends on latitude and on the Earth's orbit. For present-day work the orbit is
fixed and this is pure astronomy. For palaeoclimate work the three Milankovitch
parameters change, and the day length with them, so they are exposed here as a
small container that the caller fills from whatever orbital solution they already
use (Berger 1978, Laskar 2004, ...). pyZonae does not compute the parameters from
a date -- that is deliberately left to the established tools.

Calendar convention
--------------------
Day length is evaluated per *calendar month of fixed length* (the mid-month day),
matching climatologies whose months are ordinary ~30-day months. This is the
right choice when the source model wrote fixed-length months; a "celestial"
(fixed solar-longitude) convention would need a different mapping and is not used
here. See Joussaume & Braconnot (1997) on why the two must not be mixed.

Only obliquity enters the *amplitude* of the solar declination. Eccentricity and
the longitude of perihelion enter through the time<->orbital-position relation:
the Earth moves faster near perihelion, so a fixed calendar day maps to a shifted
orbital position. Both effects are included below.
"""

from dataclasses import dataclass

import numpy as np

# Present-day values (Berger 1978 conventions), used as defaults.
OBLIQUITY_TODAY = 23.44          # degrees
ECCENTRICITY_TODAY = 0.0167
PERIHELION_LON_TODAY = 282.9     # degrees, longitude of perihelion (from vernal equinox)

DAYS_IN_YEAR = 365.0
# Mid-month day-of-year for 12 fixed-length calendar months.
_MID_MONTH_DOY = np.array(
    [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float
)


@dataclass
class OrbitalParameters:
    """The three Milankovitch parameters, plus the calendar anchor.

    Fill these from your own orbital solution (Berger 1978, Laskar 2004). The
    defaults are present-day, so leaving this untouched reproduces the modern
    calculation exactly.

    Attributes
    ----------
    obliquity : float
        Axial tilt, degrees. Present-day ~23.44.
    eccentricity : float
        Orbital eccentricity. Present-day ~0.0167.
    perihelion_longitude : float
        Longitude of perihelion measured from the moving vernal equinox, degrees.
        Present-day ~282.9 (perihelion in early January).
    vernal_equinox_doy : float
        Day-of-year of the vernal equinox on the fixed-length calendar. This is
        the calendar anchor tying orbital position to calendar date; ~80 (Mar 21)
        by convention, which is the usual reference for palaeo calendars.
    """

    obliquity: float = OBLIQUITY_TODAY
    eccentricity: float = ECCENTRICITY_TODAY
    perihelion_longitude: float = PERIHELION_LON_TODAY
    vernal_equinox_doy: float = 80.0


def _true_longitude(doy, orb):
    """Solar longitude (deg from vernal equinox) for a calendar day-of-year.

    Inverts Kepler's equation: calendar day -> mean anomaly -> eccentric anomaly
    -> true anomaly -> solar longitude. With e = 0 this reduces to a uniform
    sweep, i.e. the simple present-day approximation.
    """
    e = orb.eccentricity
    # Longitude of the Sun at the vernal equinox is 0 by definition; the perihelion
    # sits at (360 - perihelion_longitude) ahead in true anomaly terms.
    # Mean anomaly advances uniformly in TIME; we tie time to calendar day.
    # Angle of perihelion measured from vernal equinox:
    lon_peri = np.radians(orb.perihelion_longitude)

    # True anomaly of the vernal equinox = -lon_peri (equinox is lon_peri before
    # perihelion along the orbit). Convert to mean anomaly at the equinox.
    def true_to_mean(nu):
        E = np.arctan2(np.sqrt(1 - e**2) * np.sin(nu), e + np.cos(nu))
        return E - e * np.sin(E)

    M_equinox = true_to_mean(-lon_peri)

    # Mean anomaly advances 2*pi per year in calendar time.
    frac = (np.asarray(doy) - orb.vernal_equinox_doy) / DAYS_IN_YEAR
    M = M_equinox + 2 * np.pi * frac

    # Solve Kepler M = E - e sin E for E (Newton; a few iterations suffice).
    E = M.copy() if np.ndim(M) else np.array(M, dtype=float)
    for _ in range(50):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))

    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                        np.sqrt(1 - e) * np.cos(E / 2))
    # Solar longitude measured from the vernal equinox:
    return nu + lon_peri


def declination(doy, orb=None):
    """Solar declination (radians) for a calendar day-of-year."""
    if orb is None:
        orb = OrbitalParameters()
    lam = _true_longitude(doy, orb)
    return np.arcsin(np.sin(np.radians(orb.obliquity)) * np.sin(lam))


def daylength_hours(lat_deg, doy, orb=None):
    """Length of daylight (hours) at a latitude and calendar day-of-year."""
    if orb is None:
        orb = OrbitalParameters()
    phi = np.radians(lat_deg)
    d = declination(doy, orb)
    x = -np.tan(phi) * np.tan(d)
    x = np.clip(x, -1.0, 1.0)          # polar day / night
    ws = np.arccos(x)
    return 24.0 / np.pi * ws


def monthly_daylength(lat_deg, orb=None):
    """Day length (hours) for each of the 12 fixed-length calendar months.

    Returns an array of shape (12,) if ``lat_deg`` is scalar, or (12, ...) with
    the latitude dimensions trailing.
    """
    lat = np.asarray(lat_deg, dtype=float)
    out = np.stack([daylength_hours(lat, d, orb) for d in _MID_MONTH_DOY], axis=0)
    return out
