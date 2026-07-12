# -*- coding: utf-8 -*-
"""
sota_common.py — VERSION CORRIGEE
==================================
Pipeline de donnees PARTAGE pour tous les modeles de comparaison (SOTA) du
papier : RFR, XGBoost, LightGBM, SVR, TFT standalone et TFT+RFR (uniforme).

CORRECTION MAJEURE (par rapport a la version precedente)
--------------------------------------------------------
Le papier (paragraphe 5.1) definit le protocole de validation ainsi :
   « for every test date the spatial prediction at 0.05 deg is first
     area-weighted averaged and aggregated to the 0.25 deg grid, and
     then for each spatial coarse-grid pair we compare the pair »

Autrement dit, TOUTES les baselines doivent etre evaluees APRES agregation
vers la grille GLDAS 0.25 deg, exactement comme l'approche adoptee.

L'ancienne fonction `report()` comparait des paires pixel-fin (0.05 deg) contre
la reference 0.25 deg simplement rasterisee sur la grille fine. La variance
intra-cellule des predictions gonflait alors le RMSE des SOTA sans affecter
l'approche adoptee, ce qui rendait la comparaison inequitable.

Cette version ajoute :
  * `evaluate_at_coarse_scale(dates, data, predict_fn)` pour les modeles
    tabulaires (RFR, XGBoost, LightGBM, SVR) ;
  * `evaluate_exported_maps_at_coarse_scale(...)` pour les modeles dont les
    predictions sont deja ecrites sur disque (TFT standalone, TFT+RFR fusion) ;
  * l'ancienne `report()` reste disponible pour retro-compatibilite mais un
    warning s'affiche : ne l'utilise plus pour le Tableau 3 du papier.

Placer ce fichier dans le MEME repertoire que GRSL_v12.py.
"""

import os
import warnings
import numpy as np

# Import du pipeline de base. GRSL_v12.py doit etre dans le meme dossier.
import GRSL_v12 as base


# =============================================================================
# 1) CHARGEMENT + PREPARATION DES DONNEES (INCHANGE)
# =============================================================================
def prepare_data():
    """Reproduit la phase de preparation de donnees de GRSL_v12.main(),
    et renvoie tout ce qu'il faut pour entrainer/exporter un modele SOTA."""
    base.ensure_name_file()
    sections  = base.parse_name_file(base.NAME_FILE)
    pr_dict   = base.build_date_dict(sections, base.SECTION_PR)
    tws_dict  = base.build_date_dict(sections, base.SECTION_TWS)
    ndvi_dict = base.build_date_dict(sections, base.SECTION_NDVI)
    et_dict   = base.build_date_dict(sections, base.SECTION_ET)

    dates_all = sorted(set(pr_dict.keys()) & set(tws_dict.keys()))
    if not dates_all:
        raise RuntimeError("Aucune date commune Pr/TWS.")
    print(f" Dates disponibles : {len(dates_all)} "
          f"({dates_all[0]} -> {dates_all[-1]})")

    fine_profile = base.get_fine_profile_from_pr(pr_dict)

    # Slope
    slope_path = base.get_static_path(sections, base.SECTION_SLOPE,
                                      base.SLOPE_ALIASES)
    slope_map = None
    if slope_path is not None:
        slope_map = base.load_var_on_fine(slope_path, fine_profile)
        print(f" Slope : moyenne bassin = {float(np.nanmean(slope_map)):.3f}")
    else:
        print(" [WARN] Slope introuvable -> features a 3 canaux.")

    # Masque de domaine (couverture TWS)
    Hf, Wf = fine_profile["height"], fine_profile["width"]
    domain_mask = np.zeros((Hf, Wf), dtype=bool)
    for d in sorted(tws_dict.keys()):
        try:
            tws_c, gprof, gnod = base.load_raster(tws_dict[d])
            tws_c = base.to_float_with_nan(tws_c, gnod)
            tws_f = base.reproject_to_profile(
                tws_c, gprof, gnod, fine_profile, dst_nodata=np.nan,
                resampling=base.Resampling.nearest)
            domain_mask |= np.isfinite(tws_f)
        except Exception:
            continue
    if not domain_mask.any():
        domain_mask = None
    else:
        print(f" Domaine TWS : {int(domain_mask.sum())} pixels "
              f"({100.0 * domain_mask.sum() / (Hf * Wf):.1f}% de la grille).")

    # API (decay-slope) + gap-fill
    print(" Calcul API [decay spatial slope]...")
    api_maps = base.compute_api_series(pr_dict, fine_profile, dates_all,
                                       slope_map=slope_map)
    if domain_mask is not None:
        for d in list(api_maps.keys()):
            api_maps[d] = base.spatial_fill_nearest(api_maps[d],
                                                    domain_mask=domain_mask)

    # NDVI / ET interpolation + gap-fill
    print(" Interpolation + gap-fill NDVI / ET...")
    ndvi_maps = base.build_interp_series(ndvi_dict, fine_profile, dates_all,
                                         domain_mask=domain_mask,
                                         fill_residual=True)
    et_maps   = base.build_interp_series(et_dict, fine_profile, dates_all,
                                         domain_mask=domain_mask,
                                         fill_residual=True)
    dates_all = [d for d in dates_all if d in ndvi_maps and d in et_maps]
    if not dates_all:
        raise RuntimeError("Aucune date avec NDVI+ET.")

    # Split chronologique 80/20 (identique au pipeline de base)
    idx = np.arange(len(dates_all))
    if base.CHRONOLOGICAL_SPLIT:
        n_test = max(1, int(round(len(dates_all) * base.TEST_FRACTION)))
        cut    = len(dates_all) - n_test
        train_dates = [dates_all[i] for i in idx[:cut]]
        test_dates  = [dates_all[i] for i in idx[cut:]]
        print(f" Train: {len(train_dates)} | Test: {len(test_dates)} "
              f"[chronologique : test = {test_dates[0]} -> {test_dates[-1]}]")
    else:
        step = int(round(1.0 / base.TEST_FRACTION))
        tmask = (idx % step == 0)
        test_dates  = [dates_all[i] for i in idx[tmask]]
        train_dates = [dates_all[i] for i in idx[~tmask]]

    # Normaliseur ajuste sur le train
    _, _, _, normalizer, _ = base.assemble_samples(
        train_dates, api_maps, ndvi_maps, et_maps, tws_dict,
        fine_profile, normalizer=None, fit_normalizer=True,
        slope_map=slope_map)

    return dict(
        api_maps=api_maps, ndvi_maps=ndvi_maps, et_maps=et_maps,
        tws_dict=tws_dict, fine_profile=fine_profile, slope_map=slope_map,
        domain_mask=domain_mask, dates_all=dates_all,
        train_dates=train_dates, test_dates=test_dates, normalizer=normalizer)


# =============================================================================
# 2) MATRICES DE FEATURES TABULAIRES (INCHANGE)
# =============================================================================
def build_xy(dates, data):
    """Construit (X, y) tabulaires via base.assemble_samples pour ces dates."""
    X, y, _, _, _ = base.assemble_samples(
        dates, data["api_maps"], data["ndvi_maps"], data["et_maps"],
        data["tws_dict"], data["fine_profile"],
        normalizer=data["normalizer"], fit_normalizer=False,
        slope_map=data["slope_map"])
    return X, y


# =============================================================================
# 3) EXPORT DES CARTES DOWNSCALEES POUR UN MODELE SOTA (INCHANGE)
# =============================================================================
def export_maps_generic(predict_fn, data, map_dir):
    """Exporte une carte TWS SR par date (intersection domaine x clip x erosion)."""
    fine_profile = data["fine_profile"]
    api_maps = data["api_maps"]; ndvi_maps = data["ndvi_maps"]
    et_maps  = data["et_maps"];  slope_map = data["slope_map"]
    Hf, Wf = fine_profile["height"], fine_profile["width"]
    os.makedirs(map_dir, exist_ok=True)

    clip_mask = base.build_clip_mask_from_shapefile(base.TENSIFT_SHP, fine_profile)
    base_mask = None
    if data["domain_mask"] is not None:
        base_mask = np.asarray(data["domain_mask"], dtype=bool)
    if clip_mask is not None:
        clip_mask = np.asarray(clip_mask, dtype=bool)
        base_mask = clip_mask if base_mask is None else (base_mask & clip_mask)
    export_mask = base_mask
    if base_mask is not None and base.EDGE_EROSION_PIXELS > 0:
        export_mask = base.erode_mask(base_mask, base.EDGE_EROSION_PIXELS)
        print(f" [EXPORT] {int(export_mask.sum())} pixels conserves "
              f"(clip + erosion {base.EDGE_EROSION_PIXELS}px).")
    export_mask_flat = export_mask.ravel() if export_mask is not None else None

    slope_flat = slope_map.ravel() if slope_map is not None else None
    n_written = 0
    for d in data["dates_all"]:
        if d not in api_maps or d not in ndvi_maps or d not in et_maps:
            continue
        pr = api_maps[d].ravel(); nd = ndvi_maps[d].ravel(); et = et_maps[d].ravel()
        if slope_flat is not None:
            stack = np.stack([pr, nd, et, slope_flat], axis=1)
        else:
            stack = np.stack([pr, nd, et], axis=1)
        finite = np.isfinite(stack).all(axis=1)
        if export_mask_flat is not None:
            finite &= export_mask_flat
        if not finite.any():
            continue
        Xpix = stack[finite]
        pred = predict_fn(Xpix)
        full = np.full(Hf * Wf, np.nan, dtype=np.float32)
        full[np.flatnonzero(finite)] = pred.astype(np.float32)
        out_path = os.path.join(map_dir, f"SR_TWS_{d.strftime('%Y%m%d')}.tif")
        base.write_geotiff(out_path, full.reshape(Hf, Wf), fine_profile,
                           nodata_value=base.OUT_NODATA)
        n_written += 1
    print(f" [EXPORT] {n_written} cartes ecrites dans {map_dir}")


# =============================================================================
# 4) NOUVEAU : EVALUATION A L'ECHELLE 0.25 deg (PROTOCOLE DU PAPIER)
# =============================================================================
def _aggregate_fine_to_coarse(pred_fine_2d, fine_profile,
                               coarse_profile, coarse_nodata=None):
    """Agrege une prediction fine (0.05 deg) vers la grille grossiere via
    moyenne ponderee par aire (rasterio.warp.reproject en Resampling.average).
    """
    # Utilise le wrapper deja disponible dans le pipeline de base.
    pred_coarse = base.reproject_to_profile(
        pred_fine_2d.astype(np.float32),
        fine_profile,        # src profile (fine)
        np.nan,              # src nodata
        coarse_profile,      # dst profile (coarse)
        dst_nodata=np.nan,
        resampling=base.Resampling.average,
    )
    return pred_coarse


def evaluate_at_coarse_scale(dates, data, predict_fn):
    """
    Evaluation cohérente avec le protocole du papier :
    1) predire au fin (0.05 deg) sur tout le domaine pour chaque date test ;
    2) agreger vers la grille GLDAS 0.25 deg native par moyenne ponderee par aire ;
    3) comparer, cellule par cellule, aux valeurs GLDAS 0.25 deg brutes.

    Cette fonction reproduit exactement le protocole applique a l'approche
    adoptee (§5.1 du papier), pour que la comparaison SOTA soit equitable.

    Parametres
    ----------
    dates : liste de datetime
        Les dates test (typiquement data["test_dates"]).
    data : dict
        Le dictionnaire retourne par prepare_data().
    predict_fn : callable
        Fonction predict_fn(X) -> y qui accepte une matrice tabulaire
        [Pr, NDVI, ET, Slope] et renvoie une prediction TWS scalaire par ligne.
        Correspond a `model.predict` pour les modeles sklearn/xgboost/lightgbm.

    Retour
    ------
    (y_true_coarse, y_pred_coarse) : deux vecteurs numpy 1D prets pour
    `base.metrics(y_true, y_pred)`.
    """
    api_maps  = data["api_maps"]
    ndvi_maps = data["ndvi_maps"]
    et_maps   = data["et_maps"]
    tws_dict  = data["tws_dict"]
    slope_map = data["slope_map"]
    fine_profile = data["fine_profile"]
    domain_mask  = data["domain_mask"]

    Hf, Wf = fine_profile["height"], fine_profile["width"]
    slope_flat = slope_map.ravel() if slope_map is not None else None

    y_true_all, y_pred_all = [], []
    n_dates_scored = 0

    for d in dates:
        if d not in api_maps or d not in ndvi_maps or d not in et_maps:
            continue
        if d not in tws_dict:
            continue

        # ---- 1) Prediction fine (0.05 deg) sur tous les pixels valides ----
        pr = api_maps[d].ravel()
        nd = ndvi_maps[d].ravel()
        et = et_maps[d].ravel()
        if slope_flat is not None:
            stack = np.stack([pr, nd, et, slope_flat], axis=1)
        else:
            stack = np.stack([pr, nd, et], axis=1)
        finite = np.isfinite(stack).all(axis=1)
        if domain_mask is not None:
            finite &= domain_mask.ravel()
        if not finite.any():
            continue

        pred_fine = np.full(Hf * Wf, np.nan, dtype=np.float32)
        pred_fine[np.flatnonzero(finite)] = \
            np.asarray(predict_fn(stack[finite]), dtype=np.float32)
        pred_fine = pred_fine.reshape(Hf, Wf)

        # ---- 2) Chargement de la reference GLDAS 0.25 deg native ----
        try:
            tws_c, coarse_profile, coarse_nod = base.load_raster(tws_dict[d])
        except Exception:
            continue
        tws_ref = base.to_float_with_nan(tws_c, coarse_nod).astype(np.float32)

        # ---- 3) Agregation de la prediction fine vers 0.25 deg ----
        pred_coarse = _aggregate_fine_to_coarse(
            pred_fine, fine_profile, coarse_profile, coarse_nodata=coarse_nod)

        # ---- 4) Paires valides ----
        valid = np.isfinite(tws_ref) & np.isfinite(pred_coarse)
        if not valid.any():
            continue

        y_true_all.append(tws_ref[valid])
        y_pred_all.append(pred_coarse[valid])
        n_dates_scored += 1

    if not y_true_all:
        raise RuntimeError(
            "evaluate_at_coarse_scale : aucune paire (ref, pred) valide.")

    print(f" [EVAL 0.25 deg] {n_dates_scored} dates scorees ; "
          f"{sum(a.size for a in y_true_all)} cellules 0.25 deg au total.")
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


def evaluate_exported_maps_at_coarse_scale(map_dir, dates, tws_dict,
                                            fine_profile):
    """
    Meme protocole que `evaluate_at_coarse_scale`, mais en lisant les cartes
    fines deja ecrites sur disque (utile pour les modeles a base de sequences
    comme le TFT standalone, ou pour la fusion TFT+RFR passe par
    base.export_downscaled_maps).

    Parametres
    ----------
    map_dir : str
        Dossier contenant les fichiers SR_TWS_YYYYMMDD.tif.
    dates : liste de datetime
        Les dates test.
    tws_dict : dict
        data["tws_dict"], pour retrouver les references 0.25 deg.
    fine_profile : dict
        data["fine_profile"], profil de la grille 0.05 deg.

    Retour
    ------
    (y_true_coarse, y_pred_coarse) : deux vecteurs numpy 1D prets pour metrics.
    """
    y_true_all, y_pred_all = [], []
    n_dates_scored = 0

    for d in dates:
        map_path = os.path.join(map_dir, f"SR_TWS_{d.strftime('%Y%m%d')}.tif")
        if not os.path.exists(map_path):
            continue
        if d not in tws_dict:
            continue

        # Prediction fine sur disque
        try:
            pred_fine, _, pred_nod = base.load_raster(map_path)
        except Exception:
            continue
        pred_fine = base.to_float_with_nan(pred_fine, pred_nod).astype(np.float32)

        # Reference grossiere native
        try:
            tws_c, coarse_profile, coarse_nod = base.load_raster(tws_dict[d])
        except Exception:
            continue
        tws_ref = base.to_float_with_nan(tws_c, coarse_nod).astype(np.float32)

        # Agregation
        pred_coarse = _aggregate_fine_to_coarse(
            pred_fine, fine_profile, coarse_profile, coarse_nodata=coarse_nod)

        valid = np.isfinite(tws_ref) & np.isfinite(pred_coarse)
        if not valid.any():
            continue
        y_true_all.append(tws_ref[valid])
        y_pred_all.append(pred_coarse[valid])
        n_dates_scored += 1

    if not y_true_all:
        raise RuntimeError(
            "evaluate_exported_maps_at_coarse_scale : aucune carte scoree.")
    print(f" [EVAL 0.25 deg | cartes disque] {n_dates_scored} dates scorees ; "
          f"{sum(a.size for a in y_true_all)} cellules 0.25 deg au total.")
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


# =============================================================================
# 5) AFFICHAGE DES METRIQUES
# =============================================================================
def report(name, y_true, y_pred):
    """Affiche RMSE, R^2, r. Utilise base.metrics."""
    rmse, r2, r = base.metrics(y_true, y_pred)
    print("\n" + "=" * 60)
    print(f" {name} — PERFORMANCES TEST")
    print(f"   RMSE = {rmse:.3f} mm | R2 = {r2*100:.1f}% | r = {r*100:.1f}%")
    print("=" * 60)
    return rmse, r2, r


def report_pixel_deprecated(name, y_true, y_pred):
    """Ancienne API pixel-a-pixel : conservee pour retro-compatibilite mais
    affiche un warning. Ne PAS utiliser pour le Tableau 3 du papier."""
    warnings.warn(
        "report_pixel_deprecated: evaluation pixel a 0.05 deg. "
        "Pour le Tableau 3, utilise evaluate_at_coarse_scale + report.",
        stacklevel=2)
    return report(name + " [pixel 0.05 deg]", y_true, y_pred)