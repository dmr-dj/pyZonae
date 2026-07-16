# -*- coding: utf-8 -*-
# Copyright 2019-2022 Didier M. Roche <didier.roche@lsce.ipsl.fr>
# Copyright 2026 G. Bonneroy
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
Colormaps and label dictionaries for the climate classifications.

Each ``*_cmap_*`` function returns a tuple ``(label_dict, colormap)`` where:

* ``label_dict`` maps a classification key (str, e.g. ``"Cfb"`` or ``"HA1a,b"``)
  to an integer index (1-based) used as the plotted value.
* ``colormap`` is a :class:`matplotlib.colors.ListedColormap` whose colors are
  ordered to match those indices.

Original Koeppen-Geiger colormaps by Didier M. Roche (dmr), from pyKoeppen.
Defaut/Bonneroy colormap by G. Bonneroy, 2026.

The registry at the bottom of this module maps a classification *name* (the value
of ``typ_classification``) to the function that provides its colors/labels.
"""

import matplotlib as mpl
import numpy as np


# --------------------------------------------------------------------------
# Koeppen-Geiger colormaps (ported verbatim from pyKoeppen/create_KG_cmap.py)
# --------------------------------------------------------------------------
def KG_cmap_2006() :

    # Color scheme used by Kottek et al., 2006

    # Hex colors
    # Af    960000
    # Am    ff0000
    # As    ff9999
    # Aw    ffcccc  ##
    # BWk   ffff64
    # BWh   ffcc00
    # BSk   ccaa54
    # BSh   cc8d14
    # Cfa   003200
    # Cfb   005000
    # Cfc   007800
    # Csa   00ff00
    # Csb   96ff00
    # Csc   c8ff00
    # Cwa   b46400
    # Cwb   966400
    # Cwc   5a3c00
    # Dfa   320032
    # Dfb   640064
    # Dfc   c800c8
    # Dfd   c71585
    # Dsa   ff6dff
    # Dsb   ffb4ff
    # Dsc   e6c8ff
    # Dsd   c8c8c8
    # Dwa   c8b4ff
    # Dwb   9a7fb3
    # Dwc   8859b3
    # Dwd   6f24b3 ##
    # EF    6496ff
    # ET    64ffff

    List_Colors = ['#960000','#ff0000','#ff9999','#ffcccc','#ffff64','#ffcc00','#ccaa54','#cc8d14',
                   '#003200','#005000','#007800','#00ff00','#96ff00','#c8ff00','#b46400',
                   '#966400','#5a3c00','#320032','#640064','#c800c8','#c71585','#ff6dff',
                   '#ffb4ff','#e6c8ff','#c8c8c8','#c8b4ff','#9a7fb3','#8859b3','#6f24b3','#6496ff',
                   '#64ffff']

    KG_dict = {
              "Af"  :  1,
              "Am"  :  2,
              "As"  :  3,
              "Aw"  :  4,
              "BWk" :  5,
              "BWh" :  6,
              "BSk" :  7,
              "BSh" :  8,
              "Cfa" :  9,
              "Cfb" : 10,
              "Cfc" : 11,
              "Csa" : 12,
              "Csb" : 13,
              "Csc" : 14,
              "Cwa" : 15,
              "Cwb" : 16,
              "Cwc" : 17,
              "Dfa" : 18,
              "Dfb" : 19,
              "Dfc" : 20,
              "Dfd" : 21,
              "Dsa" : 22,
              "Dsb" : 23,
              "Dsc" : 24,
              "Dsd" : 25,
              "Dwa" : 26,
              "Dwb" : 27,
              "Dwc" : 28,
              "Dwd" : 29,
              "EF"  : 30,
              "ET"  : 31
              }


    return KG_dict, mpl.colors.ListedColormap(List_Colors)
#end def KG_cmap_2006

def KG_cmap_2007() :

    # Color scheme used by Peel et al., 2007

    # Hex colors
    # Af    010feb
    # Am    0e72f9
    # As    ??????	Does not exist in Peel et al., 2007
    # Aw    42a8fd
    # BWk   f29790
    # BWh   f60000
    # BSk   ffda65
    # BSh   f2a406
    # Cfa   cbf74c
    # Cfb   66ec2e
    # Cfc   4bca23
    # Csa   f3fa00
    # Csb   c7c804
    # Csc   ??????	Does not occur in Peel et al., 2007 || Proposed: a7a900
    # Cwa   98d995
    # Cwb   65ca5e
    # Cwc   399a36
    # Dfa   15fffe
    # Dfb   39c7f8
    # Dfc   04767c
    # Dfd   0b4560
    # Dsa   f500fe
    # Dsb   de00c7
    # Dsc   932e95
    # Dsd   956492
    # Dwa   adacd5
    # Dwb   597bda
    # Dwc   5054b0
    # Dwd   2b0d7c
    # EF    65696c
    # ET    b3afb0

    List_Colors = ['#010feb','#0e72f9','#42a8fd','#f29790','#f60000','#ffda65','#f2a406',
                   '#cbf74c','#66ec2e','#4bca23','#f3fa00','#c7c804','#a7a900','#98d995',
                   '#65ca5e','#399a36','#15fffe','#39c7f8','#04767c','#0b4560','#f500fe',
                   '#de00c7','#932e95','#956492','#adacd5','#597bda','#5054b0','#2b0d7c','#65696c',
                   '#b3afb0']


    KG_dict = {
              "Af"  :  1,
              "Am"  :  2,
              "Aw"  :  3,
              "BWk" :  4,
              "BWh" :  5,
              "BSk" :  6,
              "BSh" :  7,
              "Cfa" :  8,
              "Cfb" :  9,
              "Cfc" : 10,
              "Csa" : 11,
              "Csb" : 12,
              "Csc" : 13,
              "Cwa" : 14,
              "Cwb" : 15,
              "Cwc" : 16,
              "Dfa" : 17,
              "Dfb" : 18,
              "Dfc" : 19,
              "Dfd" : 20,
              "Dsa" : 21,
              "Dsb" : 22,
              "Dsc" : 23,
              "Dsd" : 24,
              "Dwa" : 25,
              "Dwb" : 26,
              "Dwc" : 27,
              "Dwd" : 28,
              "EF"  : 29,
              "ET"  : 30
              }

    return KG_dict, mpl.colors.ListedColormap(List_Colors)
#end def KG_cmap_2007

def KG_cmap_2012() :

    # Color scheme used by Cannon, 2012

    # Hex colors
    # 1WW   fd0001
    # 1WD   fe5400
    # 1DW   fcaa00
    # 1DD   fefe00
    # 2Ww   8fed8f
    # 2Wd   72d374
    # 2Dh   54b554

    # 2Dc   3a9b3a
    # 2Dw   1c7d1c
    # 2Dd   006301
    # 3hh   890101
    # 3hW   941818
    # 3hD   9b3332
    # 3cw   a4504e

    # 3cd   b0686b
    # 3Hw   b78481
    # 3Hd   be9f9d
    # 3CH   cbb7b8
    # 3CC   d2d2d2
    # 4Ww   fc69b0
    # 4Wd   eb59be

    # 4HW   db4acd
    # 4HD   c53ed8
    # 4Ch   b32de2
    # 4Cc   9f20ed
    # 5hh   010088
    # 5hc   003fa7
    # 5hC   017ec1
    # 5cw   02bee3

    # 5cd   00fefc


    List_Colors = ['#fd0001','#fe5400','#fcaa00','#fefe00','#8fed8f','#72d374','#54b554',
                   '#3a9b3a','#1c7d1c','#006301','#890101','#941818','#9b3332','#a4504e',
                   '#b0686b','#b78481','#be9f9d','#cbb7b8','#d2d2d2','#fc69b0','#eb59be',
                   '#db4acd','#c53ed8','#b32de2','#9f20ed','#010088','#003fa7','#017ec1','#02bee3',
                   '#00fefc']

    KG_dict = {
              "1WW" :  1,
              "1WD" :  2,
              "1DW" :  3,
              "1DD" :  4,
              "2Ww" :  5,
              "2Wd" :  6,
              "2Dh" :  7,
              "2Dc" :  8,
              "2Dw" :  9,
              "2Dd" : 10,
              "3hh" : 11,
              "3hW" : 12,
              "3hD" : 13,
              "3cw" : 14,
              "3cd" : 15,
              "3Hw" : 16,
              "3Hd" : 17,
              "3CH" : 18,
              "3CC" : 19,
              "4Ww" : 20,
              "4Wd" : 21,
              "4HW" : 22,
              "4HD" : 23,
              "4Ch" : 24,
              "4Cc" : 25,
              "5hh" : 26,
              "5hc" : 27,
              "5hC" : 28,
              "5cw" : 29,
              "5cd" : 30
              }

    return KG_dict, mpl.colors.ListedColormap(List_Colors)
#end def KG_cmap_2012

def KG_cmap_2014() :

    # Color scheme used by Belda et al., 2014

    # Hex colors
    # Ar    84070b
    # Aw    cd1c0a
    # As    b64f04
    # BW    ffde49
    # BS    f09137
    # Cs    9fc301
    # Cw    2c8a29
    # Cf    009736
    # Do    00add7
    # Dc    b2559c
    # Eo    1451a1
    # Ec    0c356b
    # Ft    c0c0c0
    # Fi    8c8c8c

    List_Colors = ['#84070b','#cd1c0a','#b64f04','#ffde49','#f09137','#9fc301','#2c8a29','#009736',
                   '#00add7','#b2559c','#1451a1','#0c356b','#c0c0c0','#8c8c8c']

    KG_dict = {
              "Ar"  :  1,
              "Aw"  :  2,
              "As"  :  3,
              "BW"  :  4,
              "BS"  :  5,
              "Cs"  :  6,
              "Cw"  :  7,
              "Cr"  :  8,
              "Do"  :  9,
              "Dc"  : 10,
              "Eo"  : 11,
              "Ec"  : 12,
              "Ft"  : 13,
              "Fi"  : 14
              }


    return KG_dict, mpl.colors.ListedColormap(List_Colors, name="belda14", N=14)
#end def KG_cmap_2006

# -
# Defaut (1996) / Bonneroy (2026) colormap
# -

def Bonneroy_cmap_2026() :

    # Color scheme transformed by Bonneroy, 2026
    # Hex colors:

    # E1       #541111
    # E2       #5E1313
    # E3       #6E1616
    # HA1a,b   #752222
    # HA1c,d   #852727
    # HA2a,b   #872F2B
    # HA2c,d   #8F322D
    # HA3a,b   #963530
    # HA3c,d   #9C3631
    # HA4a,b   #9E3C37
    # HA4c,d   #A33E39
    # A1a,b    #A8403B
    # A1c,d    #B5453F
    # A2a,b    #BD4B40    ###
    # A2c,d    #BF4646
    # A3a,b    #C24747
    # A3c,d    #C25340    ###
    # A4a,b    #C64949
    # A4c,d    #C85C40    ###
    # SA1a,b   #C9504D
    # SA1c,d   #C95E4C
    # SA2a,b   #CE6540    ###
    # SA2c,d   #D26F40    ###
    # SA3a,b   #D77A40    ###
    # SA3c     #DB8340    #####
    # SA3d     #DF8F40    #####
    # SA4a,b   #E29B40    ###          '#E29B40', 27
    # SA4c,d   #E5A740    ###          '#E5A740', 28
    # SH1a,b   #E5AA40
    # SH1c,d   #E6AF40
    # SH2a,b   #E8B340    ###
    # SH2c,d   #EDBC40    ###
    # SH3a,b   #F2C540    ###
    # SH3c     #F7CE40    #####
    # SH3d     #F5D745    #####
    # SH4a,b   #F2E04B    ###
    # SH4c     #EFE852    #####
    # SH4d     #E6E957    #####
    # G1       #1E6E15
    # G2       #1A5E12
    # O1       #165210
    # O2       #154D0F
    # SX3a     #DAE65C    #####
    # SX3b     #CDE360    #####
    # SX3c     #BFDE65    #####
    # SX3d     #B3D96D    #####
    # SX4a,b   #A7D374    ###        
    # SX4c     #9BCD7C    #####
    # SX4d     #7FC792    #####
    # SX5a,b   #63C0A9    ###
    # SX5c     #47BABF    #####
    # SX5d     #40AFC7    #####
    # SX6a,b   #3FA7C7
    # SX6c     #40A3C8    #####
    # SX6d     #4095C9    #####
    # AX3a     #4189CA    #####
    # AX3b     #447ACA    #####
    # AX3c     #476BC9    #####
    # AX3d     #4A5DC8    #####
    # AX4a     #4D53C1    #####
    # AX4b     #504AB9    #####
    # AX4c     #5241B2    #####
    # AX4d     #6240B4    #####
    # AX5a     #7340B9    #####
    # AX5b     #8540BE    #####
    # AX5c     #9640C2    #####
    # AX5d     #A840C7    #####
    # AX6a     #BA40CC    #####
    # AX6b     #CC4FD3    #####
    # AX6c     #DD85E0    #####
    # AX6d     #EFBBEE    #####
    # AX7a     #FFF0FB    #####
    # AX7b     #FFF0FB    #####
    # AX7c     #FFF0FB    #####
    # AX7d     #FFF0FB    #####
    

    List_Colors = ['#541111','#5E1313','#6E1616','#752222','#852727','#872F2B','#8F322D',
                   '#963530','#9C3631','#9E3C37','#A33E39','#A8403B','#B5453F','#BD4B40',
                   '#BF4646','#C24747','#C25340','#C64949','#C85C40','#C9504D','#C95E4C',
                   '#CE6540','#D26F40','#D77A40','#DB8340','#DF8F40','#E29B40','#E5A740',
                   '#E5AA40','#E6AF40','#E8B340','#EDBC40','#F2C540','#F7CE40','#F5D745',
                   '#F2E04B','#EFE852','#E6E957','#1E6E15','#1A5E12','#165210','#154D0F',
                   '#DAE65C','#CDE360','#BFDE65','#B3D96D','#A7D374','#9BCD7C','#7FC792',
                   '#63C0A9','#47BABF','#40AFC7','#3FA7C7','#40A3C8','#4095C9','#4189CA',
                   '#447ACA','#476BC9','#4A5DC8','#4D53C1','#504AB9','#5241B2','#6240B4',
                   '#7340B9','#8540BE','#9640C2','#A840C7','#BA40CC','#CC4FD3','#DD85E0',
                   '#EFBBEE','#FFF0FB','#FFF0FB','#FFF0FB','#FFF0FB', '#000000']

    Defaut_dict = {
             "E1"       :   1,
             "E2"       :   2,
             "E3"       :   3,
             "HA1a,b"   :   4,
             "HA1c,d,e" :   5,
             "HA2a,b"   :   6,
             "HA2c,d,e" :   7,
             "HA3a,b"   :   8,
             "HA3c,d,e" :   9,
             "HA4a,b"   :   10,
             "HA4c,d,e" :   11,
             "A1a,b"    :   12,
             "A1c,d,e"  :   13,
             "A2a,b"    :   14,
             "A2c,d,e"  :   15,
             "A3a,b"    :   16,
             "A3c,d,e"  :   17,
             "A4a,b"    :   18,
             "A4c,d,e"  :   19,
             "SA1a,b"   :   20,
             "SA1c,d,e" :   21,
             "SA2a,b"   :   22,
             "SA2c,d,e" :   23,
             "SA3a,b"   :   24,
             "SA3c"     :   25,
             "SA3d,e"   :   26,
             "SA4a,b"   :   27,
             "SA4c,d,e" :   28,
             "SH1a,b"   :   29,
             "SH1c,d,e" :   30,
             "SH2a,b"   :   31,
             "SH2c,d,e" :   32,
             "SH3a,b"   :   33,
             "SH3c"     :   34,
             "SH3d,e"   :   35,
             "SH4a,b"   :   36,
             "SH4c"     :   37,
             "SH4d,e"   :   38,
             "G1"       :   39,
             "G2"       :   40,
             "O1"       :   41,
             "O2"       :   42,
             "SX3a"     :   43,
             "SX3b"     :   44,
             "SX3c"     :   45,
             "SX3d,e"   :   46,
             "SX4a,b"   :   47,
             "SX4c"     :   48,
             "SX4d,e"   :   49,
             "SX5a,b"   :   50,
             "SX5c"     :   51,
             "SX5d,e"   :   52,
             "SX6a,b"   :   53,
             "SX6c"     :   54,
             "SX6d,e"   :   55,
             "AX3a"     :   56,
             "AX3b"     :   57,
             "AX3c"     :   58,
             "AX3d,e"   :   59,
             "AX4a"     :   60,
             "AX4b"     :   61,
             "AX4c"     :   62,
             "AX4d,e"   :   63,
             "AX5a"     :   64,
             "AX5b"     :   65,
             "AX5c"     :   66,
             "AX5d,e"   :   67,
             "AX6a"     :   68,
             "AX6b"     :   69,
             "AX6c"     :   70,
             "AX6d,e"   :   71,
             "AX7a"     :   72,
             "AX7b"     :   73,
             "AX7c"     :   74,
             "AX7d,e"   :   75,
             "Doesn't exist in Defaut's model" : 10000
              
              }

    
    return Defaut_dict, mpl.colors.ListedColormap(List_Colors)

# Friendly alias so the public name matches the classification key "Defaut96".
def Defaut_cmap_1996():
    """Alias of :func:`Bonneroy_cmap_2026` (labels/colors for Defaut, 1996)."""
    return Bonneroy_cmap_2026()


# --------------------------------------------------------------------------
# Holdridge life zones
# --------------------------------------------------------------------------
def Holdridge_cmap():
    """Labels and colours for the Holdridge life zones.

    There are several hundred possible (region, belt, province) combinations, so
    the colormap is generated rather than hand-tabulated. The encoding is chosen
    for readability:

    * **hue**       <- humidity province (blue = humid ... red/brown = arid)
    * **lightness** <- latitudinal region (light = cold ... dark = warm)
    * **saturation**<- altitudinal belt (muted at high belts)

    Returns ``(label_dict, colormap)`` like every other ``*_cmap*`` here.
    """
    import colorsys

    from .classifiers.holdridge import (
        LATITUDINAL_REGIONS, ALTITUDINAL_BELTS, HUMIDITY_PROVINCES,
        SUBTROPICAL, WARM_TEMPERATE_MERGED,
    )

    # Regions in thermal order, including both frost-resolved variants and the
    # merged class used when no frost line is supplied.
    regions = ["Polar", "Subpolar", "Boreal", "CoolTemperate",
               "WarmTemperate", SUBTROPICAL, WARM_TEMPERATE_MERGED, "Tropical"]

    # Hue per humidity province: humid (blue ~0.58) -> arid (red ~0.02)
    n_p = len(HUMIDITY_PROVINCES)
    hues = np.linspace(0.58, 0.02, n_p)

    labels = {}
    colors = []
    idx = 1
    for r_i, region in enumerate(regions):
        # light for cold regions, dark for warm ones
        light = 0.80 - 0.42 * (r_i / max(1, len(regions) - 1))
        for b_i, belt in enumerate(ALTITUDINAL_BELTS):
            sat_scale = 1.0 - 0.45 * (b_i / max(1, len(ALTITUDINAL_BELTS) - 1))
            for p_i, prov in enumerate(HUMIDITY_PROVINCES):
                h = hues[p_i]
                s = 0.75 * sat_scale
                v_l = light
                r, g, b = colorsys.hls_to_rgb(h, v_l, s)
                labels[f"{region} {belt} {prov}"] = idx
                colors.append(mpl.colors.to_hex((r, g, b)))
                idx += 1

    # Sentinel for cells the model cannot classify (e.g. missing inputs).
    labels["Outside Holdridge model"] = 10000
    colors.append("#000000")

    return labels, mpl.colors.ListedColormap(colors)


# --------------------------------------------------------------------------
# Thornthwaite-Feddema (2005), two main factors
# --------------------------------------------------------------------------
def ThornFeddema_cmap():
    """Labels and colours for the two-factor Thornthwaite-Feddema classification.

    Following Feddema's Fig. 10: the moisture class sets the HUE (violet-blue for
    wet through green, yellow, to red for arid), and the thermal class sets the
    LIGHTNESS (dark = hot/torrid, light = frost). 6 x 6 = 36 classes.
    """
    import colorsys
    from .classifiers.thornfeddema import MOISTURE_TYPES, THERMAL_TYPES

    # Hue per moisture class, wettest -> driest (violet ~0.78 down to red ~0.0).
    moist_names = [m for m, _ in MOISTURE_TYPES]           # Arid..Saturated
    hues = {
        "Saturated": 0.78, "Wet": 0.60, "Moist": 0.36,
        "Dry": 0.17, "Semiarid": 0.09, "Arid": 0.01,
    }
    therm_names = [t for t, _ in THERMAL_TYPES]            # Frost..Torrid

    labels, colors = {}, []
    idx = 1
    for m in moist_names:
        for j, t in enumerate(therm_names):
            # light for cold (Frost), dark for hot (Torrid)
            light = 0.82 - 0.46 * (j / (len(therm_names) - 1))
            r, g, b = colorsys.hls_to_rgb(hues[m], light, 0.72)
            labels[f"{m} {t}"] = idx
            colors.append(mpl.colors.to_hex((r, g, b)))
            idx += 1
    return labels, mpl.colors.ListedColormap(colors)


# --------------------------------------------------------------------------
# Registry: classification name -> colormap/label provider
# --------------------------------------------------------------------------
CMAP_REGISTRY = {
    "kottek":    KG_cmap_2006,
    "peel":      KG_cmap_2007,
    "cannon":    KG_cmap_2012,
    "trewartha": KG_cmap_2014,
    "Defaut96":  Defaut_cmap_1996,
    "Holdridge": Holdridge_cmap,
    "ThornFeddema05": ThornFeddema_cmap,
}


def get_cmap(typ_classification):
    """Return ``(label_dict, colormap)`` for a classification name.

    Raises
    ------
    KeyError
        If ``typ_classification`` is not a known classification.
    """
    try:
        return CMAP_REGISTRY[typ_classification]()
    except KeyError:
        raise KeyError(
            f"Unknown classification '{typ_classification}'. "
            f"Known: {sorted(CMAP_REGISTRY)}"
        )
