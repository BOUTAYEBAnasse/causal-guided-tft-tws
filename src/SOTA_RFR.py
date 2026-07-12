# -*- coding: utf-8 -*-
"""
SOTA_RFR.py — Modele de comparaison : Random Forest Regressor (baseline).

CORRECTION : evaluation a 0.25 deg (protocole §5.1 du papier), au lieu de la
comparaison pixel-a-pixel (qui gonflait artificiellement le RMSE des SOTA).

Hyperparametres FIXES, identiques a ceux de l'approche adoptee (GRSL_v12,
RFR_PARAMS) et au Tableau II du papier. AUCUNE optimisation bayesienne :
l'approche adoptee n'en utilise pas non plus, donc pour une comparaison
coherente les modeles SOTA emploient des hyperparametres fixes.

Protocole : memes features [Pr, NDVI, ET, Slope], meme split chronologique
80/20, meme export (clip bassin Tensift + erosion de bord), et desormais
memes metriques (a 0.25 deg apres agregation).

AJOUT : calcul de la variance sous-maille (sigma-bar) sur les cartes fines
exportees (metrique de structure fine, sous-section "Sub-grid Variance").

Placer ce fichier a cote de GRSL_v12.py, sota_common.py et subgrid_common.py.
"""

from sklearn.ensemble import RandomForestRegressor

import GRSL_v12 as base
import sota_common as sc
import subgrid_common as sg

SEED = base.RANDOM_SEED


def main():
    print("=" * 70)
    print(" SOTA — RANDOM FOREST REGRESSOR (baseline, hyperparametres fixes)")
    print(" Evaluation : agregation vers 0.25 deg (protocole du papier §5.1)")
    print("=" * 70)

    data = sc.prepare_data()
    X_tr, y_tr = sc.build_xy(data["train_dates"], data)
    print(f" Echantillons train (0.05 deg) : {X_tr.shape}")

    # Hyperparametres FIXES : exactement ceux de GRSL_v12 (Tableau II du papier).
    print(f" Hyperparametres RFR (fixes) : {base.RFR_PARAMS} | "
          f"max_samples={base.RFR_MAX_SAMPLES}")
    model = RandomForestRegressor(
        bootstrap=True, max_samples=base.RFR_MAX_SAMPLES, n_jobs=-1,
        random_state=SEED, **base.RFR_PARAMS)
    model.fit(X_tr, y_tr)

    # --------- EVALUATION EQUITABLE A 0.25 deg ---------
    y_te_coarse, y_pred_coarse = sc.evaluate_at_coarse_scale(
        data["test_dates"], data, model.predict)
    sc.report("RFR (evaluation 0.25 deg)", y_te_coarse, y_pred_coarse)

    # Export des cartes downscalees (utile pour illustrations, meme protocole).
    out_dir = base.os.path.join(base.OUT_DIR, "SOTA_RFR_MAPS")
    sc.export_maps_generic(model.predict, data, out_dir)

    # --------- VARIANCE SOUS-MAILLE (structure fine) ---------
    sg.report_sigma_bar("RFR", out_dir, data)

    print("\n Termine (RFR).")


if __name__ == "__main__":
    main()
