# -*- coding: utf-8 -*-
# Copyright 2019-2022 Didier M. Roche <didier.roche@lsce.ipsl.fr>
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
Koeppen-Geiger climate classification.

Pure classification logic ported from pyKoeppen (Didier M. Roche, dmr;
Didier Paillard). Each ``get_kg_classification*`` function takes a 13-element
``arguments`` vector of derived climate indices and returns a classification
*key* (str) such as ``"Cfb"``. The mapping from key to plotted integer and to
color lives in :mod:`pyzonae.cmaps`.

The ``arguments`` vector layout (see :mod:`pyzonae.derive`):

    0: T_min    coldest-month mean T2m           (C)
    1: T_max    warmest-month mean T2m           (C)
    2: T_mon    number of months with T >= 10 C  (count)
    3: T_ann    annual mean T2m                  (C)
    4: P_min    precipitation of driest month    (mm/month)
    5: P_ann    annual precipitation             (mm/year)
    6: P_smin   min summer-half-year precip      (mm/month)
    7: P_smax   max summer-half-year precip      (mm/month)
    8: P_wmin   min winter-half-year precip      (mm/month)
    9: P_wmax   max winter-half-year precip      (mm/month)
   10: P_th     dryness threshold                (mm)
   11: P_wpro   winter fraction of annual precip (-)
   12: P_dry    number of dry months (P <= 60)   (count)

Version history (from the original module):
  0.0-0.75  see pyKoeppen changelog
  0.80      match/case dispatch of classifications
  0.81      packaged into pyzonae; removed lcm_utils / hard-coded paths;
            NumPy 2.x compatible.
"""

__version__ = "0.81"


def get_Equatorial_Climates_Kottek(P_min,P_ann,P_smin,P_wmin,classification="") :

    if (P_min > 60.0):
        classification += "f"  # Equatorial rainforest, fully humid
    elif (P_ann >= 25.0 * (100.0 - P_min)):
        classification += "m"  # Equatorial monsoon
    elif (P_smin <= 60.0):
        classification += "s"  # Equatorial Savannah with dry summer
    elif (P_wmin <= 60.0):
        classification += "w"  # Equatorial Savannah with dry winter
    # endif
    return classification
#enddef get_Equatorial_Climates_Kottek

def get_Equatorial_Climates_Peel(P_min,P_ann,classification="") :

    if (P_min > 60.0):
        classification += "f"  # Equatorial rainforest, fully humid
    elif (P_ann >= 25.0 * (100.0 - P_min)):
        classification += "m"  #  Equatorial monsoon
    elif (P_min <= 60.0):
        classification += "w"  # Equatorial Savannah
    # endif
    return classification
#enddef get_Equatorial_Climates_Peel

def get_Arid_Climates(P_ann,P_th,T_ann):

    if P_ann > 5 * P_th:
        classification = "S"  # Steppe climate
    else:
        classification = "W"  # Desert climate
    # endif
    if T_ann >= 18.0:
        classification += "h"  # Hot
    else:
        classification += "k"  # cold
    # endif
    return classification
#enddef get_Arid_Climates

def get_second_letter(P_smin,P_wmin,P_wmax,P_smax):
    if P_smin < P_wmin and P_wmax > 3.0 * P_smin and P_smin < 40.0:
        classification = "s"  # WTC, dry summers
    elif P_wmin < P_smin and P_smax > 10.0 * P_wmin:
        classification = "w"  # WTC, dry winters
    else:
        classification = "f"  # WTC, fully humid
    # endif
    return classification
# end def get_second_letter

def get_third_letter(T_max,T_mon,T_min):

    if T_max >= 22.0:
        classification = "a"  # Hot summer
    elif T_mon >= 4:  # Modification done
        classification = "b"  # Warm summer
    elif T_min > -38.0:
        classification = "c"  # Cool summer and cold winter
    else:
        classification = "d"  # extremely continental
    # endif

    return classification
# end def get_third_letter

def get_Second_Third_Letter(P_smin,P_wmin,P_wmax,P_smax, T_max, T_min, T_mon):

    classification = get_second_letter(P_smin,P_wmin,P_wmax,P_smax)

    classification += get_third_letter(T_max,T_mon,T_min)

    return classification
# enddef get_Second_Third_Letter

def get_Polar_Climates(T_max):

    if T_max >= 0.0:
        classification = "T"  # Tundra
    else:
        classification = "F"  #  Frost
    # endif

    return classification
# enddef get_Snow_Climates

def get_kg_classification(arguments, vers="peel"):

    T_min = arguments[0]
    T_max = arguments[1]
    T_mon = arguments[2]
    T_ann = arguments[3]
    P_min = arguments[4]
    P_ann = arguments[5]
    P_smin= arguments[6]
    P_smax= arguments[7]
    P_wmin= arguments[8]
    P_wmax= arguments[9]
    P_th  = arguments[10]

    kg_classification=""

    if T_max < 10.0:
        kg_classification += "E"
        kg_classification += get_Polar_Climates(T_max)
    elif P_ann < 10 * P_th:
        kg_classification += "B"
        kg_classification += get_Arid_Climates(P_ann, P_th, T_ann)
    elif T_min >= 18.0:
        kg_classification += "A"
        if vers == "kottek":
            kg_classification += get_Equatorial_Climates_Kottek(P_min, P_ann, P_smin, P_wmin)
        else:
            kg_classification += get_Equatorial_Climates_Peel(P_min, P_ann)
        # endif
    elif T_min > -3.0 and T_min < 18.0:
        kg_classification += "C"
        kg_classification += get_Second_Third_Letter(P_smin, P_wmin, P_wmax, P_smax, T_max, T_min, T_mon)
    elif T_min <= -3.0 and T_max >= 10.0:
        kg_classification += "D"
        kg_classification += get_Second_Third_Letter(P_smin, P_wmin, P_wmax, P_smax, T_max, T_min, T_mon)
    else:
        kg_classification = "F"
    # endif T_min

    return kg_classification
# enddef get_kg_classification

def get_kg_classification_Trewartha(arguments, vers="trewartha"):
    T_min = arguments[0]
    T_max = arguments[1]
    T_mon = arguments[2]
    T_ann = arguments[3]
    P_min = arguments[4]
    P_ann = arguments[5]
    P_smin = arguments[6]
    P_smax = arguments[7]
    P_wmin = arguments[8]
    P_wmax = arguments[9]
    P_wpro = arguments[11]
    P_dry = arguments[12]

    kg_classification = ""

    A = 2.3*T_ann-0.64*P_wpro+41

    if T_max < 0.0:
        kg_classification += "Fi"
    elif T_max < 10.0:
        kg_classification += "Ft"
    elif T_mon <= 3:
        if T_min <= -10.0:
            kg_classification += "Ec"
        else:
            kg_classification += "Eo"
        # endif on T_min
    elif T_mon <= 7:
        if T_min <= 0.0:
            kg_classification += "Dc"
        else:
            kg_classification += "Do"
        # endif on T_min
    elif T_min > 18.0 and P_ann/10.0 > A :
        # Climates Ar and Aw, with or without dry season
        if P_dry < 3:
            kg_classification += "Ar"
        else:
            kg_classification += "Aw"
        pass
    elif T_mon >= 8.0 and P_ann/10.0 > A : # and P_ann < 890.0 
        if P_smin < 30.0 and P_min < 1. / 3. * P_wmax:
            kg_classification += "Cs"
        elif P_smax > 10.0 * P_wmin:
            kg_classification += "Cw"
        else:
            kg_classification += "Cr"
        # endif on precipitation
    else:
        if P_ann/10.0 <= 0.5 * A :
            kg_classification += "BW"
        else:
            kg_classification += "BS"
    # endif T_min

    return kg_classification


# enddef get_kg_classification_Trewartha


def get_kg_classification_Cannon(arguments):

    T_min = arguments[0]
    T_max = arguments[1]
    T_mon = arguments[2]
    T_ann = arguments[3]
    P_min = arguments[4]
    P_ann = arguments[5]
    P_smin= arguments[6]
    P_smax= arguments[7]
    P_wmin= arguments[8]
    P_wmax= arguments[9]
    P_th  = arguments[10]

    kg_classification=""

    if T_ann >= 12.0:
        if P_ann >= 1400.0:
            if P_min >= 70.0:
                if P_ann >= 2800.0:
                    if P_ann >= 3700.0:
                        kg_classification = "1WW"
                    else:
                        kg_classification = "1WD"
                    # endif
                else:  # P_ann < 2800
                    if P_ann > 2000.0:
                        kg_classification = "1DW"
                    else:
                        kg_classification = "1DD"
                    # endif
                # endif on P_ann > 2800
            else:  # P_min < 70
                if P_ann >= 2200.0:

                    if P_smax >= 590.0:
                        kg_classification = "2Ww"
                    else:
                        kg_classification = "2Wd"
                    # endif

                else:  # P_ann < 2200
                    if P_min > 30.0:
                        if T_min >= 16.0:
                            kg_classification = "2Dh"
                        else:
                            kg_classification = "2Dc"
                        # endif
                    else:  # P_min < 30
                        if P_wmax >= 170.0:
                            kg_classification = "2Dw"
                        else:
                            kg_classification = "2Dd"
                        # endif
                    # endif on P_min > 30
                # endif on P_ann > 2200
            # endif on P_min
        else:  # P_ann < 1400
            if P_ann >= 600.0:
                if T_min >= 14.0:
                    if T_max >= 30.0:
                        kg_classification="3hh"
                    else:
                        if P_ann >= 1000.0:
                            kg_classification="3hW"
                        else:
                            kg_classification="3hD"
                        # endif
                    # endif on T_max >= 30
                else:  # T_min < 14
                    if P_wmin >= 40.0:
                        kg_classification = "3cw"
                    else:
                        kg_classification = "3cd"
                    # endif
                # endif
            else:  # P_ann < 600.0
                if T_ann >= 22:
                    if P_smax >= 60.0:
                        kg_classification = "3Hw"
                    else:
                        kg_classification = "3Hd"
                    # endif P_smax >= 60
                else:  # T_ann < 22
                    if T_ann >= 18.0:
                        kg_classification = "3CH"
                    else:  # T_ann < 18
                        kg_classification = "3CC"
                    # endif
                # endif on T_ann >= 22
            # endif on P_ann >= 600
        # endif on P_ann >= 1400
    else:  # T_ann < 12
        if T_ann >= -2.0:
            if P_wmax >= 110.0:
                if P_wmax >= 230:
                    kg_classification = "4Ww"
                else:
                    kg_classification = "4Wd"
                # endif
            else:  # P_wmax < 110
                if T_ann >= 5:
                    if P_ann >= 500:
                        kg_classification = "4HW"
                    else:
                        kg_classification = "4HD"
                    # endif
                else:  # T_ann < 5
                    if T_max >= 16.0:
                        kg_classification = "4Ch"
                    else:
                        kg_classification = "4Cc"
                    # endif
                # endif on T_ann >= 5
            # endif P_wmax >= 110
        else:  # T_ann < -2.0
            if T_max >= 5:
                if T_ann >= -9:
                    if T_max >= 13:
                        kg_classification = "5hh"
                    else:
                        kg_classification = "5hc"
                    # endif on T_max >= 13
                else:  # T_ann < -9
                    kg_classification = "5hC"
                # endif T_ann >= -9
            else:  # T_max < 5
                if P_min >= 50:
                    kg_classification = "5cw"
                else:
                    kg_classification = "5cd"
                # endif
            # endif on T_max >= 5
        # endif on T_ann >= -2
    # endif on T_ann >= 12
    return kg_classification
# enddef get_kg_classification_Cannon
