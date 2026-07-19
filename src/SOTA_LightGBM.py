# -*- coding: utf-8 -*-
"""
SOTA_LightGBM.py — Modele de comparaison : LightGBM Regressor.

CORRECTION : evaluation a 0.25 deg (protocole §5.1 du papier).

Hyperparametres FIXES (aucune optimisation bayesienne), pour rester coherent
avec l'approche adoptee (causal_tft_pipeline), qui fixe elle aussi ses hyperparametres.

Protocole : memes features [Pr, NDVI, ET, Slope], meme split chronologique
80/20, memes metriques (agregation 0.25 deg), meme export (clip Tensift +
erosion de bord).

AJOUT : calcul de la variance sous-maille (sigma-bar) sur les cartes fines
exportees (metrique de structure fine, sous-section "Sub-grid Variance").

Dependances : lightgbm, scikit-learn.
Placer ce fichier a cote de causal_tft_pipeline.py, sota_common.py et subgrid_common.py.
"""

import causal_tft_pipeline as base
import sota_common as sc
import subgrid_common as sg

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

SEED = base.RANDOM_SEED

# Hyperparametres FIXES de LightGBM (regression tabulaire standard).
LGBM_PARAMS = {
    "n_estimators":      800,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "max_depth":         -1,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "min_child_samples": 20,
    "reg_lambda":        1.0,
    "reg_alpha":         0.0,
}


def main():
    print("=" * 70)
    print(" SOTA — LIGHTGBM REGRESSOR (hyperparametres fixes)")
    print(" Evaluation : agregation vers 0.25 deg (protocole du papier §5.1)")
    print("=" * 70)
    if not _HAS_LGBM:
        raise SystemExit("[ERREUR] lightgbm non installe. -> pip install lightgbm")

    data = sc.prepare_data()
    X_tr, y_tr = sc.build_xy(data["train_dates"], data)
    print(f" Echantillons train (0.05 deg) : {X_tr.shape}")

    print(f" Hyperparametres LightGBM (fixes) : {LGBM_PARAMS}")
    model = LGBMRegressor(
        objective="regression",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        **LGBM_PARAMS
    )
    model.fit(X_tr, y_tr)

    # --------- EVALUATION EQUITABLE A 0.25 deg ---------
    y_te_coarse, y_pred_coarse = sc.evaluate_at_coarse_scale(
        data["test_dates"], data, model.predict)
    sc.report("LightGBM (evaluation 0.25 deg)", y_te_coarse, y_pred_coarse)

    out_dir = base.os.path.join(base.OUT_DIR, "SOTA_LIGHTGBM_MAPS")
    sc.export_maps_generic(model.predict, data, out_dir)

    # --------- VARIANCE SOUS-MAILLE (structure fine) ---------
    sg.report_sigma_bar("LightGBM", out_dir, data)

    print("\n Termine (LightGBM).")


if __name__ == "__main__":
    main()
