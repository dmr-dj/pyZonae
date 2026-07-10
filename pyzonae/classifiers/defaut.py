# -*- coding: utf-8 -*-
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
Defaut (1996) bioclimatic classification.

Ported from the AWIESM notebook by G. Bonneroy (2026). The decision tree
originally returned an integer "etage" index; here it returns a classification
*key* (str, e.g. ``"HA1a,b"``) so that it shares the same contract as the
Koeppen-Geiger classifiers. The mapping key -> plotted integer -> color lives
in :mod:`pyzonae.cmaps` (``Bonneroy_cmap_2026`` / ``Defaut_cmap_1996``).

Reference:
    Defaut, B. (1996). La biogeographie des Orthopteres et la classification
    bioclimatique. Materiaux Entomocenotiques.

Aridity index (Qn2, "indice d'aridite de Gaussen") and the polynomial stage
boundaries are reproduced verbatim from the original notebook.
"""

import numpy as np

from ..cmaps import Defaut_cmap_1996

# Build the integer -> key inverse map once, from the label dictionary, so the
# decision tree can translate its numeric "etage" into a classification key.
_DEFAUT_DICT, _ = Defaut_cmap_1996()
_INT_TO_KEY = {v: k for k, v in _DEFAUT_DICT.items()}
_SENTINEL_KEY = "Doesn't exist in Defaut's model"  # maps to 10000


# --------------------------------------------------------------------------
# Aridity index and stage boundary functions (verbatim from the notebook)
# --------------------------------------------------------------------------
def Qn2(P,Psec,T,tc,tf):
    # The radicand is negative for cells outside Defaut's model (annual mean
    # T < -30 C, i.e. ice-sheet climates) or with no seasonal cycle (tc == tf).
    # Those legitimately yield NaN and are mapped to the sentinel downstream, so
    # we silence the expected warning here.
    with np.errstate(invalid="ignore", divide="ignore"):
        indice_daridite=10*np.sqrt( (50*(P+10*Psec))/((T+30)*(tc-tf)) )
    return indice_daridite

def Q2_EMBERGER(P,tc,tf):    
    indice_daridite=1000*P/(((tc+tf)/2)*(tc-tf))
    return indice_daridite

#P = Précipitations moyennes annuelles, en mm
#Psec = Précipitations cumulées des 3 mois consécutifs les plus secs, en mm
#T = Température moyenne annuelle, en °Celsius
#tc = Température moyenne du mois le plus chaud, en °Celsius
#tf = Température moyenne du mois le plus froid, en °Celsius


### étages xériques :
a_E_HA , b_E_HA = -0.208325, 9.89552
def E_HA(T):
    Qn2=a_E_HA*T + b_E_HA
    return Qn2

a_HA_A , b_HA_A , c_HA_A = -0.0321309, 1.15537, 23.2553
def HA_A(T):
    Qn2=a_HA_A*T**2 + b_HA_A*T + c_HA_A
    return Qn2

a_A_SA , b_A_SA , c_A_SA, d_A_SA = 0.000672872, -0.065408, 1.5689, 42.8432
def A_SA(T):
    Qn2=a_A_SA*T**3 + b_A_SA*T**2 + c_A_SA*T + d_A_SA
    return Qn2

a_SA_SH , b_SA_SH = 0.127449, 73.6908
def SA_SH(T):
    Qn2=a_SA_SH*T + b_SA_SH
    return Qn2

    

### étages subxériques :
a_SH_SX, b_SH_SX, c_SH_SX, d_SH_SX, e_SH_SX, f_SH_SX, g_SH_SX, h_SH_SX, i_SH_SX = 0.0000118089, - 0.00107858, + 0.0412537 , - 0.85959 , + 10.6349, - 79.6916 , + 351.591, - 826.978, + 867.334
def SH_SX(T):
    Qn2=a_SH_SX*T**8 + b_SH_SX*T**7 + c_SH_SX*T**6 + d_SH_SX*T**5 + e_SH_SX*T**4 + f_SH_SX*T**3 + g_SH_SX*T**2 + h_SH_SX*T + i_SH_SX
    return Qn2


a_SX_CBM, b_SX_CBM, c_SX_CBM, d_SX_CBM , e_SX_CBM, f_SX_CBM, g_SX_CBM, h_SX_CBM=  0.00027178 , - 0.0190579 , + 0.549051 , - 8.39751 , + 73.5899 , - 369.641 , + 990.044 , - 992.56
def SX_CBM(T):
    Qn2=a_SX_CBM*T**7 + b_SX_CBM*T**6 + c_SX_CBM*T**5 + d_SX_CBM*T**4 + e_SX_CBM*T**3 + f_SX_CBM*T**2 + g_SX_CBM*T + h_SX_CBM
    return Qn2
    
a_SX_AA, b_SX_AA, c_SX_AA, d_SX_AA, e_SX_AA, f_SX_AA, g_SX_AA, h_SX_AA, i_SX_AA, j_SX_AA = 1.53691*10**-7, - 0.000020412, + 0.00117214, - 0.0381088, + 0.771216, - 10.0507, + 84.183, - 436.618, + 1275.93, - 1529.25   # Pour tc !!!
def SX_BSAA(tc):
    Qn2= a_SX_AA*tc**9 +b_SX_AA*tc**8 +c_SX_AA*tc**7 +d_SX_AA*tc**6 +e_SX_AA*tc**5 +f_SX_AA*tc**4 + g_SX_AA*tc**3 + h_SX_AA*tc**2 + i_SX_AA*tc + j_SX_AA
    return Qn2


### étages axériques et limites thermiques
treschaud_chaud = float(23) #Pour T
chaud_tempere = float(16.5) #Pour T
tempere_frais = C_BM = float(10) #Pour T
frais_froid = BM_BS = float(4.5) #Pour T
frais_froid_tc = BM_BS_tc = float(20) #Pour tc
froid_tresfroid = BS_AA = float(10.5) #Pour tc
tresfroid_nival = AA_N =  float(2) #Pour tc


### étages éthiopiens    
a_SH_G , b_SH_G, c_SH_G, d_SH_G, e_SH_G = 0.00353774 , - 0.480321 , + 23.6988 , - 507.809 , + 4168.4
def SH_G(T):
    Qn2 = a_SH_G*T**4 + b_SH_G*T**3 + c_SH_G*T**2 + d_SH_G*T +e_SH_G
    return Qn2

a_G_O , b_G_O , c_G_O, d_G_O , e_G_O = 0.0608647 , - 6.81867 , + 286.409 , - 5345.98 , + 37699.8
def G_O(T):
    Qn2 = a_G_O*T**4 + b_G_O*T**3 + c_G_O*T**2 + d_G_O*T + e_G_O
    return Qn2


### Continentalité thermique (tc-tf)
hypercontinental_continental = continental_hypercontinental = float(42)
continental_subcontinental = continental_suboceanique = subcontinental_continental = suboceanique_continental = float(22)
subcontinental_oceanique = suboceanique_oceanique = oceanique_suboceanique = oceanique_subcontinental =float(16)
oceanique_hyperoceanique = hyperoceanique_oceanique = float(9)


# -
# Decision tree
# -

def etage_climatique_Bonneroy(Tmoy,tc,tf,qn2):
    if np.isnan(Tmoy) or np.isnan(tc) or np.isnan(qn2):
        return np.nan
    
    Etage=np.nan
    if (Tmoy>=frais_froid):                                                                              ########## Tmean >= 4.5°C : graph du bas
        if (qn2 <E_HA(Tmoy)):                                                                            ### E (Eremique)
            if (Tmoy>=treschaud_chaud):                                                                  # E1
                Etage=1
            elif (Tmoy>=chaud_tempere) and (Tmoy<treschaud_chaud):                                       # E2
                Etage=2
            elif (Tmoy>=tempere_frais) and (Tmoy<chaud_tempere):                                         # E3
                Etage=3
            else:
                Etage=10000
        elif (qn2 >=E_HA(Tmoy)) and (qn2 <HA_A(Tmoy)):                                                   ### HA (Hyper Aride)
            if (Tmoy>=treschaud_chaud):
                if (tc-tf)<=oceanique_suboceanique:                                                      # HA1a,b
                    Etage=4
                elif (tc-tf)>oceanique_suboceanique:                                                     # HA1c,d,e
                    Etage=5
                else:
                    Etage=10000
            elif (Tmoy>=chaud_tempere) and (Tmoy<treschaud_chaud):
                if (tc-tf)<=oceanique_suboceanique:                                                      # HA2a,b
                    Etage=6
                elif (tc-tf)>oceanique_suboceanique:                                                     # HA2c,d,e
                    Etage=7
                else:
                    Etage=10000
            elif (Tmoy>=tempere_frais) and (Tmoy<chaud_tempere): 
                if (tc-tf)<=oceanique_suboceanique:                                                      # HA3a,b
                    Etage=8
                elif (tc-tf)>oceanique_suboceanique:                                                     # HA3c,d,e
                    Etage=9
                else:
                    Etage=10000
            elif (Tmoy>=frais_froid) and (Tmoy<tempere_frais):
                if (tc-tf)<=oceanique_suboceanique:                                                      # HA4a,b
                    Etage=10
                elif (tc-tf)>oceanique_suboceanique:                                                     # HA4c,d,e
                    Etage=11
                else:
                    Etage=10000
            else:
                Etage=10000
        elif (qn2>=HA_A(Tmoy)) and (qn2<A_SA(Tmoy)):                                                     ### A (Aride)
            if(Tmoy>=treschaud_chaud):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # A1a,b
                    Etage = 12
                elif((tc-tf)>oceanique_suboceanique):                                                    # A1c,d,e
                    Etage = 13
                else:
                    Etage=10000
            elif(Tmoy>=chaud_tempere) and (Tmoy<treschaud_chaud):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # A2a,b
                    Etage = 14
                elif((tc-tf)>oceanique_suboceanique):                                                    # A2c,d,e
                    Etage = 15
                else:                                  
                    Etage = 10000
            elif (Tmoy>=tempere_frais) and (Tmoy<chaud_tempere):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # A3a,b
                    Etage = 16
                elif ((tc-tf)>oceanique_suboceanique):                                                   # A3c,d,e
                    Etage = 17
                else : 
                    Etage = 10000
            elif (Tmoy>=frais_froid) and (Tmoy<tempere_frais):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # A4a,b
                    Etage = 18
                elif ((tc-tf)>oceanique_suboceanique):                                                   # A4c,d,e
                    Etage = 19
                else : 
                    Etage = 10000
            else :
                Etage=10000
        elif (qn2>=A_SA(Tmoy)) and (qn2<SA_SH(Tmoy)):                                                    ### SA (Semi-Aride)
            if(Tmoy>=treschaud_chaud):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # SA1a,b
                    Etage = 20
                elif((tc-tf)>oceanique_suboceanique):                                                    # SA1c,d,e
                    Etage = 21
                else:
                    Etage=10000
            elif(Tmoy>=chaud_tempere) and (Tmoy<treschaud_chaud):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # SA2a,b
                    Etage = 22
                elif((tc-tf)>oceanique_suboceanique):                                                    # SA2c,d,e
                    Etage = 23
                else:                                  
                    Etage = 10000
            elif (Tmoy>=tempere_frais) and (Tmoy<chaud_tempere):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # SA3a,b
                    Etage = 24
                elif ((tc-tf)>oceanique_suboceanique) and ((tc-tf)<=subcontinental_continental):         # SA3c
                    Etage = 25
                elif ((tc-tf)>subcontinental_continental):                                               # SA3d,e
                    Etage = 26
                else : 
                    Etage=10000
            elif (Tmoy>=frais_froid) and (Tmoy<tempere_frais):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # SA4a,b
                    Etage = 27
                elif((tc-tf)>oceanique_suboceanique):                                                    # SA4c,d,e
                    Etage = 28
                else:                                  
                    Etage = 10000
            else:
                Etage=10000
        elif (qn2>=SA_SH(Tmoy)) and (qn2<SH_SX(Tmoy)) and (qn2<SH_G(Tmoy)):                              ### SH (Subhumide)
            if(Tmoy>=treschaud_chaud):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # SH1a,b
                    Etage = 29
                elif((tc-tf)>oceanique_suboceanique):                                                    # SH1c,d
                    Etage = 30
                else:
                    Etage=10000
            elif (Tmoy>=chaud_tempere) and (Tmoy<treschaud_chaud):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # SH2a,b
                    Etage = 31
                elif ((tc-tf)>oceanique_suboceanique):                                                   # SH2c,d
                    Etage = 32
                else : 
                    Etage=10000
            elif (Tmoy>=tempere_frais) and (Tmoy<chaud_tempere):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # SH3a,b
                    Etage = 33
                elif ((tc-tf)>oceanique_suboceanique) and ((tc-tf)<=subcontinental_continental):         # SH3c
                    Etage = 34
                elif ((tc-tf)>subcontinental_continental):                                               # SH3d,e
                    Etage = 35
                else : 
                    Etage=10000
            elif (Tmoy>=frais_froid) and (Tmoy<tempere_frais):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # SH4a,b
                    Etage = 36
                elif ((tc-tf)>oceanique_suboceanique) and ((tc-tf)<=subcontinental_continental):         # SH4c
                    Etage = 37
                elif ((tc-tf)>subcontinental_continental):                                               # SH4d,e
                    Etage = 38
                else : 
                    Etage=10000
            else : 
                Etage = 10000
        elif (qn2>=SH_G(Tmoy)) and (qn2<G_O(Tmoy)) and (qn2<SH_SX(Tmoy)):                                ### G (Guinéen)
            if(Tmoy>=treschaud_chaud):                                                                   # G1
                Etage = 39
            elif (Tmoy>=chaud_tempere) and (Tmoy<treschaud_chaud):                                       # G2
                Etage = 40
            else:
                Etage=10000
        elif (qn2>=G_O(Tmoy)):                                                                           ### O (Ombrophile)
            if(Tmoy>=treschaud_chaud):                                                                   # O1
                Etage = 41
            elif (Tmoy>=chaud_tempere) and (Tmoy<treschaud_chaud):                                       # O2
                Etage = 42
            else:
                Etage=10000
        elif (qn2>=SH_SX(Tmoy)) and (qn2<SX_CBM(Tmoy)):                                                  ### SX3,4 (Subxérique tempéré à frais)
            if (Tmoy>=tempere_frais) and (Tmoy<chaud_tempere):
                if ((tc-tf)<=hyperoceanique_oceanique):                                                  # SX3a
                    Etage = 43
                elif ((tc-tf)<=suboceanique_oceanique) and ((tc-tf)>hyperoceanique_oceanique):           # SX3b
                    Etage = 44
                elif ((tc-tf)>oceanique_suboceanique) and ((tc-tf)<=subcontinental_continental):         # SX3c
                    Etage = 45
                elif ((tc-tf)>subcontinental_continental):                                               # SX3d,e
                    Etage = 46
                else : 
                    Etage=10000
            elif (Tmoy>=frais_froid) and (Tmoy<tempere_frais):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # SX4a,b
                    Etage = 47
                elif ((tc-tf)>oceanique_suboceanique) and ((tc-tf)<=subcontinental_continental):         # SX4c
                    Etage = 48
                elif ((tc-tf)>subcontinental_continental):                                               # SX4d,e
                    Etage = 49
                else : 
                    Etage=10000
            else: 
                Etage=10000
        elif qn2>=SX_CBM(Tmoy):                                                                          ### AX3,4 (Axérique tempéré à frais)
            if (Tmoy>=tempere_frais) and (Tmoy<chaud_tempere):                                           
                if ((tc-tf)<=hyperoceanique_oceanique):                                                  # AX3a
                    Etage = 56
                elif ((tc-tf)<=suboceanique_oceanique) and ((tc-tf)>hyperoceanique_oceanique):           # AX3b
                    Etage = 57
                elif ((tc-tf)>oceanique_suboceanique) and ((tc-tf)<=subcontinental_continental):         # AX3c
                    Etage = 58
                elif ((tc-tf)>subcontinental_continental):                                               # AX3d,e
                    Etage = 59
                else : 
                    Etage=10000
            elif (Tmoy>=frais_froid) and (Tmoy<tempere_frais):                                           
                if ((tc-tf)<=hyperoceanique_oceanique):                                                  # AX4a
                    Etage = 60
                elif ((tc-tf)>hyperoceanique_oceanique) and ((tc-tf)<=oceanique_suboceanique):           # AX4b
                    Etage = 61
                elif ((tc-tf)>oceanique_suboceanique) and ((tc-tf)<=subcontinental_continental):         # AX4c
                    Etage = 62
                elif ((tc-tf)>subcontinental_continental):                                               # AX4d,e
                    Etage = 63
                else : 
                    Etage=10000
            else: 
                Etage=10000
        else:
            Etage=10000    
    elif (Tmoy<frais_froid):                                                                             ########## Tmean < 4.5°C : graph du dessus
        if (qn2<SX_BSAA(tc)):                                                                            # SX5,6 (Subxérique froid à très froid)
            if (tc>=froid_tresfroid):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # SX5a,b
                    Etage = 50
                elif ((tc-tf)>oceanique_suboceanique) and ((tc-tf)<=subcontinental_continental):         # SX5c
                    Etage = 51
                elif ((tc-tf)>subcontinental_continental):                                               # SX5d,e
                    Etage = 52
                else : 
                    Etage=10000
            elif (tc>=tresfroid_nival) and (tc<froid_tresfroid):
                if ((tc-tf)<=oceanique_suboceanique):                                                    # SX6a,b
                    Etage = 53
                elif ((tc-tf)>oceanique_suboceanique) and ((tc-tf)<=subcontinental_continental):         # SX6c
                    Etage = 54
                elif ((tc-tf)>subcontinental_continental):                                               # SX6d,e
                    Etage = 55
                else : 
                    Etage=10000
            else : 
                Etage=10000
        
        elif (qn2>=SX_BSAA(tc)):                                                                         ### AX5,6,7 (Axérique froid à nival)
            if (tc>=froid_tresfroid):
                if ((tc-tf)<=hyperoceanique_oceanique):                                                  # AX5a
                    Etage = 64
                elif ((tc-tf)>hyperoceanique_oceanique) and ((tc-tf)<=oceanique_suboceanique):           # AX5b
                    Etage = 65
                elif ((tc-tf)>oceanique_suboceanique) and ((tc-tf)<=subcontinental_continental):         # AX5c
                    Etage = 66
                elif ((tc-tf)>subcontinental_continental):                                               # AX5d,e
                    Etage = 67
                else : 
                    Etage = 10000
            elif (tc>=tresfroid_nival) and (tc<froid_tresfroid):
                if ((tc-tf)<=hyperoceanique_oceanique):                                                  # AX6a
                    Etage = 68
                elif ((tc-tf)>hyperoceanique_oceanique) and ((tc-tf)<=oceanique_suboceanique):           # AX6b
                    Etage = 69
                elif ((tc-tf)>oceanique_suboceanique) and ((tc-tf)<=subcontinental_continental):         # AX6c
                    Etage = 70
                elif ((tc-tf)>subcontinental_continental):                                               # AX6d,e
                    Etage = 71
                else : 
                    Etage = 10000
            elif (tc<tresfroid_nival):
                if ((tc-tf)<=hyperoceanique_oceanique):                                                  # AX7a
                    Etage = 72
                elif ((tc-tf)>hyperoceanique_oceanique) and ((tc-tf)<=oceanique_suboceanique):           # AX7b
                    Etage = 73
                elif ((tc-tf)>oceanique_suboceanique) and ((tc-tf)<=subcontinental_continental):         # AX7c
                    Etage = 74
                elif ((tc-tf)>subcontinental_continental):                                               # AX7d,e
                    Etage = 75
                else : 
                    Etage = 10000
            else:
                Etage=10000
        else:
            Etage=10000
    else :
        Etage=10000
    return Etage


# --------------------------------------------------------------------------
# Public wrapper: same calling convention as the Koeppen classifiers
# --------------------------------------------------------------------------
def get_defaut_classification(arguments):
    """Classify one grid cell with the Defaut (1996) scheme.

    Parameters
    ----------
    arguments : sequence of float, length >= 15
        Derived climate indices. The Defaut scheme uses a superset of the
        Koeppen vector; the extra trailing entries carry the variables Defaut
        needs (see :mod:`pyzonae.derive`)::

            3:  T_ann   annual mean temperature       (C)   -> Tmoy
            1:  T_max   warmest-month mean T           (C)   -> tc
            0:  T_min   coldest-month mean T           (C)   -> tf
           13:  P_ann_mm  annual precipitation         (mm)  -> P
           14:  P_sec3    driest 3 consecutive months  (mm)  -> Psec

    Returns
    -------
    str
        A classification key such as ``"HA1a,b"`` or the sentinel
        ``"Doesn't exist in Defaut's model"``.
    """
    T_ann = arguments[3]
    tc = arguments[1]
    tf = arguments[0]
    P = arguments[13]
    Psec = arguments[14]

    qn2 = Qn2(P, Psec, T_ann, tc, tf)
    etage = etage_climatique_Bonneroy(T_ann, tc, tf, qn2)

    if etage is None or (isinstance(etage, float) and np.isnan(etage)):
        return _SENTINEL_KEY
    return _INT_TO_KEY.get(int(etage), _SENTINEL_KEY)
