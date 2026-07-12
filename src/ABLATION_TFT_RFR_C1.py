# -*- coding: utf-8 -*-
"""
ABLATION_TFT_RFR_C1.py
======================
Ligne "+ C1 (KL loss)" du Tableau d'ablation.

Configuration :
  * matrice causale : GRANGER (comme la methode adoptee)
  * C1 = ON   : regularisation KL (var_weights || p_granger) dans la loss TFT
  * C2 = OFF  : pas de prior causal dans la variable selection
  * C3 = OFF  : pas d'attention causale comme 4e canal (TFT a 3 canaux)
  * Fusion NNLS + recalibration affine + beta residuel (comme corr / adoptee)

Isolation : par rapport a "TFT + RFR (correlation matrix)" ligne au-dessus,
on remplace la matrice correlation par Granger ET on desactive C2 et C3,
afin d'isoler l'apport propre de C1 (KL loss guidee par Granger).

AJOUT : calcul de la variance sous-maille (sigma-bar) sur les cartes fines
exportees (metrique de structure fine, sous-section "Sub-grid Variance").
"""

import os

import ablation_common as ac
import GRSL_v12 as base
import subgrid_common as sg


def main():
    cfg = ac.AblationConfig(
        name="TFT_RFR_C1_only",
        causal_matrix="granger",
        use_C1=True,
        use_C2=False,
        use_C3=False,
        fusion="nnls",
        affine_recal=True,
        residual_beta=True,
        kl_lambda=ac.KL_LAMBDA_DEFAULT,
        ensemble_members=ac.ENSEMBLE_MEMBERS,
        n_zones=ac.N_HYDROZONES,
    )
    result = ac.run_ablation(cfg)

    out_subdir = "ABLATION_TFT_RFR_C1_MAPS"
    ac.export_fused_maps(result, out_subdir=out_subdir)
    ac.save_summary(result)

    # --------- VARIANCE SOUS-MAILLE (structure fine) ---------
    out_dir = os.path.join(base.OUT_DIR, out_subdir)
    sg.report_sigma_bar("TFT + RFR + C1 (KL loss)", out_dir, result["data"])

    print("\nTermine (Ablation : + C1 KL loss).")


if __name__ == "__main__":
    main()
