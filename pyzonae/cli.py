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
"""Command-line interface for pyZonae (installed as ``pyzonae-classify``)."""

import argparse
import sys


from pyzonae import run_classification, CLASSIFICATIONS
from pyzonae.plotting import plot_classification


def main():
    ap = argparse.ArgumentParser(description="Gridded climate classification.")
    ap.add_argument("--classification", required=True, choices=CLASSIFICATIONS)
    ap.add_argument("--tas", required=True, help="monthly temperature NetCDF")
    ap.add_argument("--pr", required=True, help="monthly precipitation NetCDF")
    ap.add_argument("--sftlf", default=None, help="land-fraction NetCDF (optional)")
    ap.add_argument("--tas-var", default=None,
                    help="temperature variable name (default: auto-detect tas/t2m/tmp/...)")
    ap.add_argument("--pr-var", default=None,
                    help="precipitation variable name (default: auto-detect pr/prcp/tp/...)")
    ap.add_argument("--sftlf-var", default=None,
                    help="land-fraction variable name (default: auto-detect sftlf/lsm/...)")
    ap.add_argument("--tas-units", default="auto", choices=["auto", "C", "K"])
    ap.add_argument("--pr-units", default="mm/month",
                    choices=["mm/month", "mm/day", "m/month", "kg/m2/s", "kg/m2/month"],
                    help="precipitation units, converted to mm/month (default: mm/month)")
    ap.add_argument("--pr-scale", type=float, default=1.0,
                    help="extra multiplicative factor applied after --pr-units")
    ap.add_argument("--save", default=None, help="output image path")
    ap.add_argument("--no-coastlines", action="store_true")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()

    m, labels, cmap, lons, lats = run_classification(
        typ_classification=a.classification,
        tas_file=a.tas, pr_file=a.pr, sftlf_file=a.sftlf,
        tas_var=a.tas_var, pr_var=a.pr_var, sftlf_var=a.sftlf_var,
        tas_units=a.tas_units, pr_scale=a.pr_scale, pr_units=a.pr_units,
        progress=a.progress,
    )
    print(f"Classified {m.count()} cells with '{a.classification}'.")

    fig, _ = plot_classification(
        m, lons, lats, labels, cmap,
        title=f"{a.classification} classification",
        coastlines=not a.no_coastlines,
    )
    if a.save:
        fig.savefig(a.save, dpi=120, bbox_inches="tight")
        print("saved", a.save)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
