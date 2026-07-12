# -*- coding: utf-8 -*-
"""
ABLATION_TFT_RFR_C1_C2.py
=========================
Ligne "+ C1 + C2 (causal prior)" du Tableau d'ablation.

Configuration :
  * matrice causale : GRANGER
  * C1 = ON   : regularisation KL dans la loss TFT
  * C2 = ON   : prior causal dans la variable selection (Eq.VS')
  * C3 = OFF  : pas d'attention causale comme 4e canal (TFT a 3 canaux)
  * Fusion NNLS + recalibration affine + beta residuel

Isolation : par rapport a la ligne "+ C1 (KL loss)", on active en plus C2
pour mesurer l'apport propre du prior causal dans la selection des variables.

AJOUT : calcul de la variance sous-maille (sigma-bar) sur les cartes fines
exportees (metrique de structure fine, sous-section "Sub-grid Variance").
"""

import os

import ablation_common as ac
import GRSL_v12 as base
import subgrid_common as sg


def main():
    cfg = ac.AblationConfig(
        name="TFT_RFR_C1_C2",
        causal_matrix="granger",
        use_C1=True,
        use_C2=True,
        use_C3=False,
        fusion="nnls",
        affine_recal=True,
        residual_beta=True,
        kl_lambda=ac.KL_LAMBDA_DEFAULT,
        ensemble_members=ac.ENSEMBLE_MEMBERS,
        n_zones=ac.N_HYDROZONES,
    )
    result = ac.run_ablation(cfg)

    out_subdir = "ABLATION_TFT_RFR_C1_C2_MAPS"
    ac.export_fused_maps(result, out_subdir=out_subdir)
    ac.save_summary(result)

    # --------- VARIANCE SOUS-MAILLE (structure fine) ---------
    out_dir = os.path.join(base.OUT_DIR, out_subdir)
    sg.report_sigma_bar("TFT + RFR + C1 + C2 (causal prior)", out_dir, result["data"])

    print("\nTermine (Ablation : + C1 + C2 causal prior).")


if __name__ == "__main__":
    main()
