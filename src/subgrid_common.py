# -*- coding: utf-8 -*-
"""
subgrid_common.py
=================
Metrique de VARIANCE SOUS-MAILLE (sigma-bar) partagee par tous les scripts
(SOTA + ablation + methode adoptee) du papier.

Definition (sous-section "Sub-grid Variance" du papier)
-------------------------------------------------------
Pour chaque cellule grossiere C (0.25 deg) contenant N sous-pixels fins
(0.05 deg) valides :

        sigma_C = sqrt( (1/N) * sum_{p in C} ( yhat_p - mean_C )^2 )

et la valeur rapportee est la moyenne de sigma_C sur toutes les cellules
grossieres et toutes les dates de test :

        sigma_bar = moyenne( sigma_C )

Un modele qui recopie trivialement la valeur grossiere dans chacun de ses
sous-pixels donnerait sigma_C = 0 partout ; une valeur sigma_bar non nulle
indique donc que le modele genere une reelle structure sous-maille.

Coherence avec le protocole du papier
-------------------------------------
Le decoupage 0.05 -> 0.25 utilise EXACTEMENT la meme grille grossiere que
le protocole d'evaluation (§5.1) : on lit la reference GLDAS 0.25 deg native
(via tws_dict) pour recuperer son profil, on identifie a quelle cellule
grossiere appartient chaque pixel fin, puis on calcule l'ecart-type des
pixels fins predits a l'interieur de chaque cellule. On reutilise les
fonctions de GRSL_v12.py (load_raster, to_float_with_nan, reproject_to_profile)
afin de ne rien reimplementer.

Ce module ne calcule RIEN de nouveau cote modele : il lit uniquement les
cartes fines SR_TWS_YYYYMMDD.tif deja exportees par chaque script, donc il
n'exige aucun re-entrainement.

Placer ce fichier a cote de GRSL_v12.py.
"""

import os
import glob
import numpy as np

import GRSL_v12 as base


# Ratio grossier / fin : 0.25 / 0.05 = 5. Redefinissable si besoin.
COARSE_FINE_FACTOR = int(round(0.25 / getattr(base, "TARGET_RES_DEG", 0.05)))
# Nombre minimal de sous-pixels fins valides pour qu'une cellule 0.25 deg compte.
MIN_VALID_SUBPIXELS = 2


def _coarse_index_grid(fine_profile, coarse_profile):
    """Renvoie, pour chaque pixel fin, l'indice lineaire de sa cellule grossiere.

    On projette une grille d'indices grossiers (0..Hc*Wc-1) sur la grille fine
    par plus-proche-voisin. Deux pixels fins tombant dans la meme cellule 0.25
    deg partagent alors le meme indice, ce qui definit la partition exacte
    utilisee pour agreger, identique au protocole d'evaluation du papier.
    """
    Hc = coarse_profile["height"]
    Wc = coarse_profile["width"]
    coarse_ids = np.arange(Hc * Wc, dtype=np.float32).reshape(Hc, Wc)

    # Plus-proche-voisin : chaque pixel fin herite de l'ID de sa cellule mere.
    idx_fine = base.reproject_to_profile(
        coarse_ids,
        coarse_profile,          # src = grille grossiere
        None,                    # pas de nodata sur les indices
        fine_profile,            # dst = grille fine
        dst_nodata=-1.0,
        resampling=base.Resampling.nearest,
    )
    return idx_fine  # 2D (Hf, Wf), valeurs = ID de cellule grossiere (ou -1)


def sigma_bar_for_map_dir(map_dir, test_dates, tws_dict, fine_profile,
                          factor=COARSE_FINE_FACTOR,
                          min_valid=MIN_VALID_SUBPIXELS):
    """Calcule sigma_bar sur les cartes fines SR_TWS d'un dossier.

    Parametres
    ----------
    map_dir : str
        Dossier contenant les cartes fines SR_TWS_YYYYMMDD.tif (0.05 deg).
    test_dates : liste de datetime
        Dates a scorer (typiquement data["test_dates"]).
    tws_dict : dict
        data["tws_dict"] : permet de recuperer le profil GLDAS 0.25 deg par date.
    fine_profile : dict
        data["fine_profile"] : profil de la grille fine 0.05 deg.
    factor : int
        Ratio grossier/fin (5 par defaut) ; utilise seulement en repli.
    min_valid : int
        Nombre minimal de sous-pixels valides par cellule grossiere.

    Retour
    ------
    (sigma_bar, n_cells, n_dates) : float, int, int
        sigma_bar en mm, nombre total de cellules grossieres comptees,
        nombre de dates effectivement scorees. sigma_bar vaut NaN si rien
        n'a pu etre calcule.
    """
    total_sigma = 0.0
    total_cells = 0
    n_dates = 0

    cached_index = None
    cached_coarse_key = None

    for d in test_dates:
        map_path = os.path.join(map_dir, f"SR_TWS_{d.strftime('%Y%m%d')}.tif")
        if not os.path.exists(map_path):
            continue
        if d not in tws_dict:
            continue

        # 1) Carte fine predite (0.05 deg).
        try:
            pred_fine, _, pred_nod = base.load_raster(map_path)
        except Exception:
            continue
        pred_fine = base.to_float_with_nan(pred_fine, pred_nod).astype(np.float32)

        # 2) Profil de la reference grossiere 0.25 deg native pour cette date.
        try:
            _, coarse_profile, _ = base.load_raster(tws_dict[d])
        except Exception:
            continue

        # 3) Grille d'appartenance pixel-fin -> cellule 0.25 deg (mise en cache).
        coarse_key = (coarse_profile["height"], coarse_profile["width"],
                      tuple(np.round(np.array(
                          coarse_profile["transform"])[:6], 8)))
        if coarse_key != cached_coarse_key:
            cached_index = _coarse_index_grid(fine_profile, coarse_profile)
            cached_coarse_key = coarse_key
        idx_fine = cached_index

        if idx_fine.shape != pred_fine.shape:
            # Securite : si dimensions incoherentes, on saute la date.
            continue

        ids = idx_fine.ravel()
        vals = pred_fine.ravel()
        good = np.isfinite(vals) & (ids >= 0)
        if not good.any():
            continue
        ids = ids[good].astype(np.int64)
        vals = vals[good].astype(np.float64)

        # 4) Ecart-type intra-cellule, vectorise via sommes par groupe.
        #    var_C = E[x^2] - (E[x])^2 ; sigma_C = sqrt(var_C).
        n_groups = ids.max() + 1
        count = np.bincount(ids, minlength=n_groups)
        sum_x = np.bincount(ids, weights=vals, minlength=n_groups)
        sum_x2 = np.bincount(ids, weights=vals * vals, minlength=n_groups)

        valid_c = count >= min_valid
        if not valid_c.any():
            continue
        cnt = count[valid_c]
        mean_c = sum_x[valid_c] / cnt
        var_c = np.maximum(sum_x2[valid_c] / cnt - mean_c * mean_c, 0.0)
        sigma_c = np.sqrt(var_c)

        total_sigma += float(sigma_c.sum())
        total_cells += int(valid_c.sum())
        n_dates += 1

    if total_cells == 0:
        return float("nan"), 0, 0
    return total_sigma / total_cells, total_cells, n_dates


def report_sigma_bar(name, map_dir, data,
                     factor=COARSE_FINE_FACTOR,
                     min_valid=MIN_VALID_SUBPIXELS):
    """Calcule et affiche sigma_bar pour un modele, a partir de son dossier de
    cartes exportees et du dict `data` (issu de prepare_data()).

    Retourne sigma_bar (float, en mm).
    """
    sigma_bar, n_cells, n_dates = sigma_bar_for_map_dir(
        map_dir,
        data["test_dates"],
        data["tws_dict"],
        data["fine_profile"],
        factor=factor,
        min_valid=min_valid,
    )
    print("\n" + "=" * 60)
    print(f" {name} — VARIANCE SOUS-MAILLE")
    if np.isfinite(sigma_bar):
        print(f"   sigma_bar = {sigma_bar:.4f} mm "
              f"(sur {n_cells} cellules 0.25 deg, {n_dates} dates)")
    else:
        print("   sigma_bar = NaN (aucune carte fine trouvee dans le dossier)")
        print(f"   dossier attendu : {map_dir}")
    print("=" * 60)
    return sigma_bar
