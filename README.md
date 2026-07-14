<!-- README shields loosely based on: https://github.com/othneildrew/Best-README-Template -->

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Commits][commits-shield]][commits-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache-2.0 License][license-shield]][license-url]

# pyZonae

Gridded climate classification with pluggable rule sets. This is a modernized
merge of two codebases:

* **pyKoeppen** (Didier M. Roche, Didier Paillard) — Köppen-Geiger classifications
  in the Kottek (2006), Peel (2007), Cannon (2012) and Trewartha/Belda (2014)
  variants.
* **Defaut (1996)** bioclimatic stages (notebook by G. Bonneroy, 2026).
* **Holdridge (1967)** Life's zones (implementation by D.M. Roche and C. Opus, 2026).

The two schemes now share one pipeline. You pick the rule with a single option,
`typ_classification`, whose accepted values are:

| value        | scheme                                   |
|--------------|------------------------------------------|
| `kottek`     | Köppen-Geiger, Kottek et al. 2006        |
| `peel`       | Köppen-Geiger, Peel et al. 2007          |
| `cannon`     | Köppen-Geiger, Cannon 2012               |
| `trewartha`  | Trewartha, Belda et al. 2014             |
| **`Defaut96`** | **Defaut (1996) bioclimatic stages**   |
| **`Holdridge`** | **Holdridge life zones (Lugo et al. 1999)** |

## Install

```bash
pip install -r requirements.txt
```

Or install the package itself (which also provides a `pyzonae-classify`
command-line tool):

```bash
pip install .
```

`cartopy` is optional (used only to draw coastlines); everything runs without it.

## Quick start

No external or private data is required: pyZonae ships a **generator** for a
synthetic global climatology rather than the data files themselves (they are
reproducible, so there is no reason to version binaries). Generate them first —
this is a required first step, the `test-data/` directory starts out empty:

```bash
python scripts/make_synthetic_data.py --outdir test-data
```

This writes `synthetic_tas_monClim.nc`, `synthetic_pr_monClim.nc` and
`synthetic_sftlf.nc`. The generator is deterministic (fixed seed), so everyone
gets identical files. Then classify and plot:

```bash
# Köppen-Geiger (Peel)
python scripts/classify_map.py --classification peel \
    --tas test-data/synthetic_tas_monClim.nc \
    --pr  test-data/synthetic_pr_monClim.nc \
    --sftlf test-data/synthetic_sftlf.nc --save map_peel.png

# Defaut (1996)
python scripts/classify_map.py --classification Defaut96 \
    --tas test-data/synthetic_tas_monClim.nc \
    --pr  test-data/synthetic_pr_monClim.nc \
    --sftlf test-data/synthetic_sftlf.nc --save map_defaut.png
```

### What you should get

The two commands above produce these maps. The synthetic world has two
idealized continents, so the climate bands are symmetric about the equator —
which makes it easy to check at a glance that a classification is behaving
sensibly.

| Köppen-Geiger (Peel et al. 2007) | Defaut (1996) |
|---|---|
| ![Köppen-Geiger classification of the synthetic climatology](docs/images/synthetic_peel.png) | ![Defaut 1996 classification of the synthetic climatology](docs/images/synthetic_Defaut96.png) |

Köppen-Geiger shows the familiar A→B→C→D→E progression from equator to pole;
Defaut shows the analogous eremic → arid → … → axeric-nival gradient over the
same fields.

### Data conventions

The synthetic files follow the PMIP4/AWIESM convention so they work with both
this package **and** the original Bonneroy notebook, unmodified:

* `tas` in **Kelvin** (the notebook does an unconditional `tas - 273.15`);
* `pr` in **mm/month**;
* `sftlf` as a **percentage** (0–100).

**Variable names.** By default the loader auto-detects common names (`tas`,
`t2m`, `tmp`, … for temperature; `pr`, `prcp`, `tp`, … for precipitation;
`sftlf`, `lsm`, … for the mask). If your file uses something else, name it
explicitly:

```bash
python scripts/classify_map.py --classification peel \
    --tas my_file.nc --tas-var temperature_2m \
    --pr  my_file2.nc --pr-var total_precip ...
```

**Precipitation units.** Precipitation is converted to mm/month via `--pr-units`
(default `mm/month`, a no-op). Supported values: `mm/month`, `mm/day`,
`m/month`, `kg/m2/s` (the CMIP native flux) and `kg/m2/month`. Per-second and
per-day units are integrated to monthly totals using standard days-per-month.
For anything not covered, `--pr-scale F` applies an extra multiplicative factor
*after* the unit conversion. Examples:

```bash
# CMIP flux in kg m-2 s-1
python scripts/classify_map.py ... --pr-units kg/m2/s

# precipitation given in metres per month
python scripts/classify_map.py ... --pr-units m/month
```

**Temperature units.** `--tas-units auto` (default) converts to °C when the
field looks like Kelvin; force it with `--tas-units K` or `--tas-units C`.

On your own AWIESM (PMIP4) files, temperature is in Kelvin, so `auto` handles it;
you may also force it explicitly with `--tas-units K`:

```bash
python scripts/classify_map.py --classification Defaut96 \
    --tas PMIP4_tas_Amon_AWIESM1_piControl_monClim.nc \
    --pr  PMIP4_pr_Amon_AWIESM1_piControl_monClim.nc \
    --sftlf PMIP4_sftlf_fx_AWIESM1_piControl.nc \
    --tas-units K --save map_awiesm_defaut.png
```

## Library use

(Assumes you have run `make_synthetic_data.py` first, or substitute your own
NetCDF paths.)

```python
from pyzonae import run_classification
from pyzonae.plotting import plot_classification

class_map, labels, cmap, lons, lats = run_classification(
    typ_classification="Defaut96",
    tas_file="test-data/synthetic_tas_monClim.nc",
    pr_file="test-data/synthetic_pr_monClim.nc",
    sftlf_file="test-data/synthetic_sftlf.nc",
)
fig, ax = plot_classification(class_map, lons, lats, labels, cmap)
```

## Architecture

```
pyzonae/
├── io.py                   # xarray loading (replaces lcm_utils/netCDF4);
│                           #   units, land mask, orography, frost line
├── derive.py               # the 17 derived indices, shared by every scheme
│                           #   (Koeppen 0-12, Defaut 13-14, Holdridge 15-16)
├── classify.py             # dispatch: name -> classifier
├── cmaps.py                # colours + label dicts, one registry
├── run.py                  # load -> derive -> classify -> map
├── cli.py                  # argparse CLI (installed as `pyzonae-classify`)
│
├── classifiers/            # the rules themselves — one module per scheme
│   ├── koeppen.py          #   KG/Trewartha logic (verbatim, NumPy-2 compatible)
│   ├── defaut.py           #   Defaut tree; Qn2 and its polynomial boundaries
│   └── holdridge.py        #   biotemperature, PET ratio, latitudinal regions,
│                           #     altitudinal belts; fuzzy vs strict thresholds
│
├── plotting.py             # shared categorical map (cartopy optional);
│                           #   only labels the classes actually present
├── plotting_holdridge.py   # Holdridge MAP: three-panel legend, since its zones
│                           #   are composite (region x belt x province)
│
├── diagrams.py             # decision-space dispatch: plot_diagram(typ, ...)
├── decision_space.py       #   Defaut in (Qn2, T): boundary curves clipped by
│                           #     probing the tree; two panels (annual mean / tc)
├── holdridge_triangle.py   #   Holdridge's triangular, hexagon-tiled diagram
│
└── legend_grid.py          # UNUSED: a Botti-style grid legend for Defaut, kept
                            #   as an alternative to the decision-space diagram
scripts/
├── make_synthetic_data.py  # writes the synthetic NetCDFs into test-data/
├── make_readme_figures.py  # regenerates the example maps in docs/images/
└── classify_map.py         # thin wrapper around pyzonae.cli
test-data/                  # empty in git; populated by make_synthetic_data.py
docs/images/                # example maps shown in this README
tests/
└── test_pipeline.py        # 27 tests; builds its own data, needs no files on disk
```

Two plotting families, and the distinction matters:

* **Maps** (`plotting.py`, `plotting_holdridge.py`) — where each class falls on
  the globe. Every classification has one.
* **Decision-space diagrams** (`diagrams.py` and the two modules under it) — why:
  each cell placed at its coordinates in the classifying variables, with the
  boundaries drawn from the classifier's own functions. Only `Defaut96` and
  `Holdridge` have one; see the section above for why Köppen-Geiger cannot.

### How a classifier plugs in

Every classifier takes the same per-cell index vector (see `derive.py`) and
returns a **key string** (e.g. `"Cfb"`, `"HA1a,b"`, `"Boreal Basal Humid"`).
`cmaps.py` maps that key to an integer and a colour. In the common case, adding a
scheme means writing one `get_*` function that returns keys and one `*_cmap_*`
function returning `(dict, colormap)`, then registering both — the loading,
plotting and main loop are untouched.

Holdridge showed what the two extra cases look like:

* **It needed new derived indices** (biotemperature, sea-level biotemperature).
  These were appended to the shared stack in `derive.py` as slots 15–16, which
  keeps the existing slots stable — Köppen and Defaut were unaffected.
* **It needed a new input** (surface elevation). `run.py` raises an explanatory
  error if `Holdridge` is requested without `--orog`, while the other schemes
  keep working without it. A scheme may add an input without imposing it on
  everyone.

If a scheme also has a low-dimensional decision space, add a plotter and register
it in `diagrams.py`. Not every scheme does — see above.

## Notes on modernization

Relative to the original scripts:

* Removed the private `lcm_utils` / `progressbar` dependencies and hard-coded
  `/tertiaire/...` paths; all I/O is now xarray.
* Fixed `np.int` and `numpy.core._multiarray_umath` usage (broken on NumPy ≥ 1.24).
* Unified the two output conventions: the Defaut tree previously returned bare
  integers and now returns keys like the Köppen functions.
* Fixed a latent sentinel typo in the Defaut tree (`1000` → `10000`) and a
  missing `#` in one Defaut hex color.
* Summer/winter half-year precipitation is computed hemisphere-aware, and the
  Gaussen "three driest consecutive months" is vectorized.

## License

pyZonae is licensed under the **Apache License, Version 2.0** in its entirety.
See the [`LICENSE`](LICENSE) file for the full text and [`NOTICE`](NOTICE) for
attributions. Every source file carries an SPDX `Apache-2.0` header.

Attributions (all Apache-2.0):

* Köppen-Geiger classifications and colormaps (originally pyKoeppen) —
  © 2019-2022 Didier M. Roche; authors Didier M. Roche and Didier Paillard.
* Defaut (1996) bioclimatic classification, decision tree and colormap —
  © 2026 G. Bonneroy.
* Package architecture, xarray I/O, derived indices, dispatch, plotting,
  synthetic test-data generator and tests — © 2026 the pyZonae authors.

The Defaut scheme implements the method described in Defaut, B. (1996),
*La biogéographie des Orthoptères et la classification bioclimatique*,
Matériaux Entomocénotiques (cited for scientific attribution; the code is an
original Apache-2.0 implementation).

## Holdridge life zones

`Holdridge` implements the life-zone system as operationalised by Lugo et al.
(1999). Despite their names, Holdridge's *latitudinal regions* (polar ...
tropical) are **not geographic**: they are labels for intervals of sea-level
biotemperature. The classification depends only on climate plus elevation.

It needs one extra input, surface elevation (`--orog`), which the other schemes
do not use. Without it, pyZonae raises a clear error rather than guessing.

```bash
python scripts/classify_map.py --classification Holdridge \
    --tas test-data/synthetic_tas_monClim.nc \
    --pr  test-data/synthetic_pr_monClim.nc \
    --sftlf test-data/synthetic_sftlf.nc \
    --orog  test-data/synthetic_orog.nc --save map_holdridge.png
```

### Two rule sets: `--holdridge-rule`

* `fuzzy` (default) — Lugo et al.'s adjusted thresholds. They exist because,
  under the strict rule, a trivial elevation difference can flip a cell into a
  different altitudinal belt (their example: 17 m of relief is enough).
* `strict` — Holdridge's original thresholds, artefacts included. Useful for
  reproducing the original system and for measuring what the fuzzy rule changes.

### The frost line: `--frost-line`

The frost line splits *warm temperate* from *subtropical*, and **only** that.
Lugo et al. derive it from **daily** minima ("fewer than 0.5 frost days per
year", a frost day being one with daily Tmin < 0 °C) — data a monthly
climatology does not have. pyZonae therefore makes it pluggable:

| value | behaviour |
|---|---|
| *(omitted, default)* | the two regions are reported **merged**, as `WarmTemperate/Subtropical`. No boundary is invented, and the merge is explicit in the legend. |
| `coldest_month` | frost-free where the coldest monthly mean `tas` exceeds `--frost-threshold` (°C). Crude, but computable from monthly data. |
| *path to a NetCDF* | a ready-made frost-free mask. **This is the hook for a properly calibrated frost line derived from daily data.** |

Everything else in the system (biotemperature, precipitation, PET ratio,
altitudinal belts) is computable from a monthly climatology, so the default mode
is fully usable — it simply declines to draw one boundary it cannot justify.

### Known limitation

Holdridge ignores climatic **seasonality** at the life-zone level. Two regions
with the same annual biotemperature and precipitation classify identically even
if one has a wet winter and a dry summer. This is structural to the system (Lugo
et al. call it the third and only valid criticism), not an implementation
choice — unlike Köppen and Defaut, which are seasonality-aware.


## Decision-space diagrams

Besides the map, `Defaut96` and `Holdridge` can be drawn in the space of their
own classifying variables. A map shows *where* each class lands; the diagram
shows *why*: every grid cell sits at its coordinates in the classifying
variables, and the decision boundaries are drawn on top — generated directly from
the classifier's functions, so they cannot drift out of step with the code.

```bash
# Defaut in (Qn2, temperature)
python scripts/classify_map.py --classification Defaut96 \
    --tas t.nc --pr p.nc --sftlf m.nc --diagram --save defaut_space.png

# Holdridge in his triangular diagram, with the hexagonal cells underlaid
python scripts/classify_map.py --classification Holdridge \
    --tas t.nc --pr p.nc --sftlf m.nc --orog o.nc \
    --diagram --hexagons --save holdridge_triangle.png
```

Or from Python:

```python
from pyzonae import run_classification, load_climatology, build_arguments, plot_diagram

m, labels, cmap, lons, lats = run_classification("Holdridge", tas, pr,
                                                 sftlf_file=msk, orog_file=oro)
fields = load_climatology(tas, pr, sftlf_file=msk, orog_file=oro)
args, _, _, _ = build_arguments(fields)
fig, ax = plot_diagram("Holdridge", m, args, labels, cmap, hexagons=True)
```

### Why the two diagrams have different geometries

This is forced by the mathematics of each scheme, not by taste.

**Holdridge** has an exact constraint, `PETR = Tbio × 58.93 / P`, which is linear
in logarithms: `log(PETR) + log(P) − log(Tbio) = log(58.93)`. The three axes
therefore lie on a *plane* — two degrees of freedom — so the classification
projects into 2-D without loss. That plane, cut by the axis ranges, is his
triangle. Under the projection used here the three families of lines meet at
60°, which is what tiles the plane with the regular hexagons of his Fig. 1.
Cold sits at the apex and tropical along the base, as he draws it.

**Defaut** has no such constraint. Its aridity index `Qn2` depends on
precipitation terms that are independent of the other two axes: regressing Qn2 on
temperature and continentality explains only ~9 % of its variance. The three axes
have three genuine degrees of freedom, so a triangular projection would *invent*
a constraint that does not exist and would overlay distinct classes. Defaut is
therefore plotted as a scatter in (Qn2, temperature).

Two further points of fidelity in the Defaut diagram:

* Its decision tree **changes variable** partway down — degrees 1–4 test the
  annual mean temperature, degrees 5–7 the warmest month — so the figure is split
  into two stacked panels, each drawn against the variable it actually uses.
* A boundary is drawn **only where it is live**. The `E|HA` curve, for instance,
  ceases to exist below ~10 °C because the eremic stage is not realised there.
  Rather than hard-coding a temperature window per curve, the module *probes the
  decision tree*: at each point it asks the real classifier which group lies on
  either side, and keeps the point only if the expected pair appears.

The third axis — continentality for Defaut, altitudinal belt for Holdridge — is
carried by **marker shape** in both cases, since it appears on neither axis.
Bands are drawn most-numerous first so the rare ones are not buried.

The Köppen-Geiger variants have **no** decision-space diagram: they classify on
many interacting criteria, with no faithful low-dimensional picture. Asking for
one raises an explanatory error rather than producing a misleading figure.

## Gaussen driest-3-months (a note on temperature units)

The Defaut scheme needs the total precipitation of the three consecutive driest
months, "driest" in the Gaussen sense (lowest P/T). This ratio must use
temperature on an **absolute scale (Kelvin)**: P/T is only meaningful for T > 0,
which Kelvin guarantees everywhere. Using Celsius makes the ratio negative below
0 C and wrongly selects cold months as "driest". `pyzonae` computes this ratio
in Kelvin, identifies the driest 3-month window, and sums that same window,
wrapping around the December-January boundary so boreal-winter dry seasons are
also considered.

## Contact

Didier M. Roche - [@dja_rosh](https://x.com/dja_rosh) - didier.roche@lsce.ipsl.fr

Project Link: [https://github.com/dmr-dj/pyZonae](https://github.com/dmr-dj/pyZonae)

## Acknowledgements

Special thanks to [jypeter](https://github.com/jypeter) for development support on the Defaut96 classification


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Other stuff taken to get the shields correctly -->

[contributors-shield]: https://img.shields.io/github/contributors/dmr-dj/pyZonae
[contributors-url]: https://github.com/dmr-dj/pyZonae/graphs/contributors
[commits-shield]: https://img.shields.io/github/commit-activity/y/dmr-dj/pyZonae
[commits-url]: https://github.com/dmr-dj/pyZonae/graphs/commit-activity
[stars-shield]: https://img.shields.io/github/stars/dmr-dj/pyZonae
[stars-url]: https://github.com/dmr-dj/pyZonae/stargazers
[issues-shield]: https://img.shields.io/github/issues/dmr-dj/pyZonae
[issues-url]: https://github.com/dmr-dj/pyZonae/issues
[license-shield]: https://img.shields.io/github/license/dmr-dj/pyZonae
[license-url]: https://github.com/dmr-dj/pyZonae/blob/main/LICENSE

