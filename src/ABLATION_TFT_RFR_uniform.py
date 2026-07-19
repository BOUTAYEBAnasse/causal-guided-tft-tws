# -*- coding: utf-8 -*-
"""
ABLATION_TFT_RFR_uniform.py
===========================
Ligne 2 du Tableau 4 : TFT + RFR + uniform weight.

Logique de donnees et d'export identique a causal_tft_pipeline.py, mais C1, C2 et C3 sont
desactives et la fusion TFT/RFR est fixee a w1=w2=0.5, sans recalibration affine.

AJOUT : calcul de la variance sous-maille (sigma-bar) sur les cartes fines
exportees (metrique de structure fine, sous-section "Sub-grid Variance").
"""

import os

import ablation_common as ac
import causal_tft_pipeline as base
import subgrid_common as sg


def main():
    cfg = ac.AblationConfig(
        name="TFT_RFR_uniform",
        causal_matrix="none",
        use_C1=False,
        use_C2=False,
        use_C3=False,
        fusion="uniform",
        affine_recal=False,
        residual_beta=False,
        ensemble_members=ac.ENSEMBLE_MEMBERS,
    )
    result = ac.run_ablation(cfg)

    out_subdir = "ABLATION_TFT_RFR_UNIFORM_MAPS"
    ac.export_fused_maps(result, out_subdir=out_subdir)
    ac.save_summary(result)

    # --------- VARIANCE SOUS-MAILLE (structure fine) ---------
    out_dir = os.path.join(base.OUT_DIR, out_subdir)
    sg.report_sigma_bar("TFT + RFR (uniform weight)", out_dir, result["data"])

    print("\nTermine (Ablation : TFT + RFR + uniform weight).")


if __name__ == "__main__":
    main()
