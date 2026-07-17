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
End-to-end orchestration: load data, derive indices, classify, return a map.

This is the common driver for every classification, including ``"Defaut96"``.
It replaces the bespoke ``__main__`` blocks of the original pyKoeppen and Defaut
scripts.
"""

import numpy as np
from numpy import ma

from .io import load_climatology
from .derive import build_arguments
from .classify import classify_cell
from .cmaps import get_cmap


def run_classification(
    typ_classification,
    tas_file,
    pr_file,
    sftlf_file=None,
    orog_file=None,
    orog_var=None,
    frost_line=None,
    frost_threshold=0.0,
    holdridge_rule="fuzzy",
    tf_factors=2,
    orbital=None,
    tas_var=None,
    pr_var=None,
    sftlf_var=None,
    pr_scale=1.0,
    pr_units="mm/month",
    tas_units="auto",
    progress=False,
):
    """Compute a classification map.

    Parameters
    ----------
    typ_classification : str
        One of the supported classification names.
    tas_file, pr_file : str
        NetCDF inputs (monthly climatology).
    sftlf_file : str, optional
        Land-fraction file for masking ocean.
    tas_var, pr_var, sftlf_var : str, optional
        Override variable-name auto-detection.
    pr_scale : float
        Precipitation unit factor (e.g. multiply to reach mm/month).
    tas_units : {"auto", "C", "K"}
    progress : bool
        Print a simple progress percentage.

    Returns
    -------
    class_map : numpy.ma.MaskedArray (lat, lon)
        Integer class indices, masked where undefined.
    label_dict : dict[str, int]
    cmap : matplotlib colormap
    lons, lats : coordinate arrays
    """
    label_dict, cmap = get_cmap(typ_classification, factors=tf_factors)

    if typ_classification == "Holdridge" and orog_file is None:
        raise ValueError(
            "The Holdridge classification requires surface elevation. "
            "Pass orog_file=... (CLI: --orog). It is used to compute the "
            "sea-level biotemperature, which sets the latitudinal region, and "
            "the altitudinal belt. The other classifications do not need it."
        )

    fields = load_climatology(
        tas_file, pr_file,
        tas_var=tas_var, pr_var=pr_var,
        sftlf_file=sftlf_file, sftlf_var=sftlf_var,
        orog_file=orog_file, orog_var=orog_var,
        frost_line=frost_line, frost_threshold=frost_threshold,
        pr_scale=pr_scale, pr_units=pr_units, tas_units=tas_units,
    )
    args, grid_shape, lats, lons = build_arguments(fields, orbital=orbital)

    n = int(np.prod(grid_shape))
    flat = args.reshape(args.shape[0], n)
    class_flat = ma.masked_all(n, dtype=int)

    mask = ma.getmaskarray(flat)
    ff_flat = None
    if fields.frost_free is not None:
        ff_flat = np.asarray(fields.frost_free.values, dtype=float).reshape(-1)

    for i in range(n):
        # Skip cells with missing temperature or precipitation.
        if mask[0, i] or mask[4, i]:
            continue
        opts = {}
        if typ_classification == "Holdridge":
            opts["holdridge_rule"] = holdridge_rule
            if ff_flat is not None:
                v = ff_flat[i]
                opts["frost_free"] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else bool(v)
        elif typ_classification == "ThornFeddema05":
            opts["tf_factors"] = tf_factors
        key = classify_cell(typ_classification, flat[:, i], **opts)
        class_flat[i] = label_dict.get(key, max(label_dict.values()))
        if progress and (i % max(1, n // 20) == 0):
            print(f"  {100 * i // n:3d}%", end="\r")
    if progress:
        print("  100%")

    class_map = class_flat.reshape(grid_shape)
    return class_map, label_dict, cmap, lons, lats
