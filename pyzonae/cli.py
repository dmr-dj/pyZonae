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
from pyzonae.plotting_holdridge import plot_holdridge
from pyzonae.diagrams import plot_diagram, DIAGRAMS
from pyzonae.io import load_climatology
from pyzonae.derive import build_arguments


def main():
    ap = argparse.ArgumentParser(description="Gridded climate classification.")
    ap.add_argument("--classification", required=True,
                    choices=tuple(CLASSIFICATIONS) + ("all",),
                    help="classification to run, or 'all' to run every one and "
                         "write one output per scheme (see --save)")
    ap.add_argument("--tas", required=True, help="monthly temperature NetCDF")
    ap.add_argument("--pr", required=True, help="monthly precipitation NetCDF")
    ap.add_argument("--sftlf", default=None, help="land-fraction NetCDF (optional)")
    ap.add_argument("--orog", default=None,
                    help="surface elevation NetCDF (REQUIRED for --classification Holdridge)")
    ap.add_argument("--orog-var", default=None,
                    help="elevation variable name (default: auto-detect orog/elevation/...)")
    ap.add_argument("--holdridge-rule", default="fuzzy", choices=["fuzzy", "strict"],
                    help="Holdridge thresholds: 'fuzzy' (Lugo et al. 1999, default) "
                         "or 'strict' (Holdridge's originals)")
    ap.add_argument("--frost-line", default=None,
                    help="frost line for the WarmTemperate/Subtropical split: "
                         "omit to leave them merged (default), 'coldest_month' for a "
                         "threshold on the coldest monthly tas, or a path to a NetCDF "
                         "holding a frost-free mask")
    ap.add_argument("--frost-threshold", type=float, default=0.0,
                    help="threshold in degC used with --frost-line coldest_month")
    ap.add_argument("--obliquity", type=float, default=None,
                    help="ThornFeddema05: orbital obliquity in degrees "
                         "(palaeo runs; default present-day 23.44)")
    ap.add_argument("--eccentricity", type=float, default=None,
                    help="ThornFeddema05: orbital eccentricity (default 0.0167)")
    ap.add_argument("--perihelion-longitude", type=float, default=None,
                    help="ThornFeddema05: longitude of perihelion in degrees "
                         "(default 282.9)")
    ap.add_argument("--tf-factors", type=int, default=2, choices=[2, 4],
                    help="ThornFeddema05: 2 factors (moisture x thermal, "
                         "default) or 4 (adds seasonality and its cause)")
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
    ap.add_argument("--diagram", action="store_true",
                    help="draw the decision-space diagram instead of the map "
                         f"(available for: {', '.join(DIAGRAMS)})")
    ap.add_argument("--hexagons", action="store_true",
                    help="Holdridge diagram: underlay the hexagonal life-zone cells")
    ap.add_argument("--no-markers", action="store_true",
                    help="diagram: do not encode the third axis as marker shape")
    ap.add_argument("--facet", action="store_true",
                    help="Defaut diagram: one panel per continentality band")
    ap.add_argument("--colour-points", action="store_true",
                    help="Whittaker diagram: colour each point by its biome "
                         "instead of a neutral density")
    ap.add_argument("--no-clip", action="store_true",
                    help="Whittaker diagram: widen the axes to the data instead "
                         "of framing on Whittaker's envelope (shows cells colder "
                         "or wetter than any biome)")
    ap.add_argument("--point-size", type=float, default=None,
                    help="diagram: marker area (default 12 for Defaut, 14 for Holdridge)")
    ap.add_argument("--no-coastlines", action="store_true")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()

    orbital = None
    if any(v is not None for v in (a.obliquity, a.eccentricity,
                                   a.perihelion_longitude)):
        from pyzonae.orbital import OrbitalParameters
        defaults = OrbitalParameters()
        orbital = OrbitalParameters(
            obliquity=a.obliquity if a.obliquity is not None else defaults.obliquity,
            eccentricity=a.eccentricity if a.eccentricity is not None else defaults.eccentricity,
            perihelion_longitude=a.perihelion_longitude if a.perihelion_longitude is not None else defaults.perihelion_longitude,
        )

    if a.classification == "all":
        return _run_all(a, orbital)

    m, labels, cmap, lons, lats = run_classification(
        typ_classification=a.classification,
        tas_file=a.tas, pr_file=a.pr, sftlf_file=a.sftlf,
        orog_file=a.orog, orog_var=a.orog_var,
        holdridge_rule=a.holdridge_rule,
        tf_factors=a.tf_factors,
        orbital=orbital,
        frost_line=a.frost_line, frost_threshold=a.frost_threshold,
        tas_var=a.tas_var, pr_var=a.pr_var, sftlf_var=a.sftlf_var,
        tas_units=a.tas_units, pr_scale=a.pr_scale, pr_units=a.pr_units,
        progress=a.progress,
    )
    print(f"Classified {m.count()} cells with '{a.classification}'.")

    if a.diagram:
        if a.classification not in DIAGRAMS:
            sys.exit(
                f"--diagram is not available for '{a.classification}'.\n"
                f"Available: {', '.join(DIAGRAMS)}. The Koeppen-Geiger variants "
                f"classify on many interacting criteria and have no faithful "
                f"low-dimensional picture."
            )
        # The diagram needs the derived indices, not just the class map.
        fields = load_climatology(
            a.tas, a.pr, tas_var=a.tas_var, pr_var=a.pr_var,
            sftlf_file=a.sftlf, sftlf_var=a.sftlf_var,
            orog_file=a.orog, orog_var=a.orog_var,
            tas_units=a.tas_units, pr_units=a.pr_units, pr_scale=a.pr_scale,
        )
        fields_args, _, _, _ = build_arguments(fields)
        kw = {}
        if a.point_size is not None:
            kw["point_size"] = a.point_size
        if a.classification == "Holdridge":
            kw["markers"] = not a.no_markers
            kw["rule"] = a.holdridge_rule
            kw["hexagons"] = a.hexagons
        elif a.classification == "Defaut96":
            kw["markers"] = not a.no_markers
            kw["facet"] = a.facet
        elif a.classification == "Whittaker":
            kw["colour_points_by_biome"] = a.colour_points
            kw["clip_to_biomes"] = not a.no_clip
        fig, _ = plot_diagram(a.classification, m, fields_args, labels, cmap, **kw)
    elif a.classification == "Holdridge":
        # Holdridge zones are composite (region x belt x province), so a flat
        # colorbar of several hundred entries is useless. Use the axis-wise legend.
        fig, _ = plot_holdridge(
            m, lons, lats, labels, cmap,
            title="Holdridge life zones",
            coastlines=not a.no_coastlines,
        )
    else:
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


def _run_all(a, orbital):
    """Run every classification on one dataset and write one output per scheme.

    Loading and deriving happen once (see
    :func:`pyzonae.run.run_all_classifications`), which is where nearly all the
    time goes; classifying eight times on the shared index stack is cheap.

    Output naming: ``--save`` is treated as a template. ``maps/out.png`` becomes
    ``maps/out_peel.png``, ``maps/out_Defaut96.png`` and so on; without an
    extension the classification name is simply appended. Without ``--save``
    nothing is written and only a summary is printed.
    """
    import os
    import matplotlib.pyplot as plt
    from pyzonae.run import run_all_classifications

    results = run_all_classifications(
        a.tas, a.pr,
        sftlf_file=a.sftlf, sftlf_var=a.sftlf_var,
        tas_var=a.tas_var, pr_var=a.pr_var,
        orog_file=a.orog, orog_var=a.orog_var,
        tas_units=a.tas_units, pr_units=a.pr_units, pr_scale=a.pr_scale,
        frost_line=a.frost_line, frost_threshold=a.frost_threshold,
        holdridge_rule=a.holdridge_rule, tf_factors=a.tf_factors,
        orbital=orbital,
    )
    lons, lats = results.pop("_lons"), results.pop("_lats")

    skipped = [n for n in CLASSIFICATIONS if n not in results]
    if skipped:
        print(f"skipped (missing inputs): {', '.join(skipped)}")

    root, ext = os.path.splitext(a.save) if a.save else (None, "")
    for typ, (m, labels, cmap) in results.items():
        n_classes = len(set(int(v) for v in m.compressed()))
        print(f"{typ:16s} {m.count():7d} cells, {n_classes:3d} classes")
        if not a.save:
            continue
        if a.diagram:
            if typ not in DIAGRAMS:
                continue
            fields = load_climatology(
                a.tas, a.pr, tas_var=a.tas_var, pr_var=a.pr_var,
                sftlf_file=a.sftlf, sftlf_var=a.sftlf_var,
                orog_file=a.orog, orog_var=a.orog_var,
                tas_units=a.tas_units, pr_units=a.pr_units, pr_scale=a.pr_scale,
            )
            fields_args, _, _, _ = build_arguments(fields, orbital=orbital)
            fig, _ = plot_diagram(typ, m, fields_args, labels, cmap)
        elif typ == "Holdridge":
            fig, _ = plot_holdridge(m, lons, lats, labels, cmap,
                                    title=f"{typ} classification",
                                    coastlines=not a.no_coastlines)
        else:
            fig, _ = plot_classification(m, lons, lats, labels, cmap,
                                         title=f"{typ} classification",
                                         coastlines=not a.no_coastlines)
        out = f"{root}_{typ}{ext}"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")
    return 0
