# -*- coding: utf-8 -*-
"""
ABLATION_TFT_RFR_corr.py
========================
Ligne 3 du Tableau 4 : TFT + RFR + correlation matrix.

Logique identique au pipeline GRSL_v12.py, sauf que la matrice de Granger
est remplacee par M_Corr = Corr(Pr_norm, NDVI_norm, 1-ET_norm).
C1, C2 et C3 restent actives afin d'isoler uniquement la nature de la matrice.

AJOUT : calcul de la variance sous-maille (sigma-bar) sur les cartes fines
exportees (metrique de structure fine, sous-section "Sub-grid Variance").
"""

import os

import ablation_common as ac
import GRSL_v12 as base
import subgrid_common as sg


def main():
    cfg = ac.AblationConfig(
        name="TFT_RFR_correlation",
        causal_matrix="correlation",
        use_C1=True,
        use_C2=True,
        use_C3=True,
        fusion="nnls",
        affine_recal=True,
        residual_beta=True,
        kl_lambda=ac.KL_LAMBDA_DEFAULT,
        ensemble_members=ac.ENSEMBLE_MEMBERS,
        n_zones=ac.N_HYDROZONES,
    )
    result = ac.run_ablation(cfg)

    out_subdir = "ABLATION_TFT_RFR_CORR_MAPS"
    ac.export_fused_maps(result, out_subdir=out_subdir)
    ac.save_summary(result)

    # --------- VARIANCE SOUS-MAILLE (structure fine) ---------
    out_dir = os.path.join(base.OUT_DIR, out_subdir)
    sg.report_sigma_bar("TFT + RFR (correlation matrix)", out_dir, result["data"])

    print("\nTermine (Ablation : TFT + RFR + correlation matrix).")


if __name__ == "__main__":
    main()
