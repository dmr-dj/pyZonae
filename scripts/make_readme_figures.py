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
"""Regenerate the example figures embedded in the README.

The synthetic climatology must exist first::

    python scripts/make_synthetic_data.py --outdir test-data
    python scripts/make_readme_figures.py

Writes ``docs/images/synthetic_<classification>.png``.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyzonae import run_classification
from pyzonae.plotting import plot_classification

FIGURES = [
    ("peel", "Köppen-Geiger (Peel et al. 2007)"),
    ("Defaut96", "Defaut (1996) bioclimatic stages"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="test-data")
    ap.add_argument("--outdir", default="docs/images")
    ap.add_argument("--dpi", type=int, default=110)
    a = ap.parse_args()

    tas = os.path.join(a.data_dir, "synthetic_tas_monClim.nc")
    pr = os.path.join(a.data_dir, "synthetic_pr_monClim.nc")
    sftlf = os.path.join(a.data_dir, "synthetic_sftlf.nc")
    for f in (tas, pr, sftlf):
        if not os.path.exists(f):
            sys.exit(
                f"missing {f}\n"
                "Run: python scripts/make_synthetic_data.py --outdir test-data"
            )

    os.makedirs(a.outdir, exist_ok=True)
    for typ, title in FIGURES:
        m, labels, cmap, lons, lats = run_classification(
            typ, tas, pr, sftlf_file=sftlf
        )
        fig, _ = plot_classification(
            m, lons, lats, labels, cmap,
            title=f"{title} — synthetic test data",
            coastlines=False,
        )
        out = os.path.join(a.outdir, f"synthetic_{typ}.png")
        fig.savefig(out, dpi=a.dpi, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
