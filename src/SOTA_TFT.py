# -*- coding: utf-8 -*-
"""
SOTA_TFT.py — Benchmark SOTA : Temporal Fusion Transformer standalone.

Objectif
--------
Ce script suit la meme logique d'execution que SOTA_RFR.py :
  1) il reutilise le pipeline de donnees commun dans sota_common.py ;
  2) il entraine un seul type de modele SOTA ;
  3) il evalue les performances sur le split chronologique test ;
  4) il exporte les cartes SR_TWS_YYYYMMDD.tif.

Difference avec causal_tft_pipeline.py / methode proposee
-----------------------------------------------
Ici, le TFT est volontairement "standalone" :
  * pas de Granger ;
  * pas de matrice de correlation ;
  * pas de C1 / KL causal dans la loss ;
  * pas de C2 / prior causal dans la variable selection ;
  * pas de C3 / canal d'attention causale ;
  * pas de RFR ;
  * pas de fusion NNLS ;
  * pas de recalibration affine.

Les sequences TFT contiennent seulement :
    [Pr_norm, NDVI_norm, 1 - ET_norm]
avec Slope comme contexte statique du GRN final, comme dans causal_tft_pipeline.py.

AJOUT : calcul de la variance sous-maille (sigma-bar) sur les cartes fines
exportees (metrique de structure fine, sous-section "Sub-grid Variance").

Version CPU-friendly inscrite directement dans le fichier :
    TFT_PROFILE=cpu
    TFT_SEQ_LEN=30
    TFT_EPOCHS=15
    ENSEMBLE_MEMBERS=3

Placer ce fichier a cote de :
    causal_tft_pipeline.py
    sota_common.py
    subgrid_common.py

Dependances : torch, scikit-learn, rasterio, scipy, numpy.
"""

import os
import numpy as np

# -----------------------------------------------------------------------------
# Profil CPU inscrit directement dans le fichier telechargeable.
# Les variables d'environnement restent surchargeables si besoin, mais vous
# n'avez pas besoin de faire "set ..." avant l'execution.
# -----------------------------------------------------------------------------
os.environ.setdefault("TFT_PROFILE", "cpu")
os.environ.setdefault("TFT_SEQ_LEN", "30")
os.environ.setdefault("TFT_EPOCHS", "15")
os.environ.setdefault("ENSEMBLE_MEMBERS", "3")
os.environ.setdefault("TFT_BATCH_SIZE", "64")

import causal_tft_pipeline as base
import sota_common as sc
import subgrid_common as sg

SEED = base.RANDOM_SEED


def _envi(name, default):
    v = os.environ.get(name, "")
    if v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


# Valeurs CPU effectives.
TFT_PROFILE = os.environ.get("TFT_PROFILE", "cpu").lower()
TFT_SEQ_LEN = _envi("TFT_SEQ_LEN", 30)
TFT_EPOCHS = _envi("TFT_EPOCHS", 15)
ENSEMBLE_MEMBERS = _envi("ENSEMBLE_MEMBERS", 3)
TFT_BATCH_SIZE = _envi("TFT_BATCH_SIZE", 64)

# Propagation dans causal_tft_pipeline.py, car on reutilise ses classes/fonctions TFT.
base.TFT_SEQ_LEN = TFT_SEQ_LEN
base.TFT_MAX_EPOCHS = TFT_EPOCHS
base.TFT_BATCH_SIZE = TFT_BATCH_SIZE
base.N_ENSEMBLE_SEEDS = ENSEMBLE_MEMBERS

# Standalone TFT = aucune guidance causale.
base.USE_ATT_AS_FEATURE = False       # pas de 4e canal Att_Granger
base.GAMMA_GRANGER_PRIOR = 0.0        # C2 off
base.LAMBDA_GRANGER_DEFAULT = 0.0     # C1 off

# Threads CPU raisonnables.
if getattr(base, "_HAS_TFT", False):
    try:
        base.torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    except Exception:
        pass


class _ZeroRegressor:
    """Modele muet utilise seulement pour reutiliser export_downscaled_maps.

    Dans l'export causal_tft_pipeline, la fonction attend un RFR et des poids de fusion.
    Ici, on donne w_TFT=1 et w_RFR=0 ; ce predicteur retourne donc zero et
    n'influence jamais la sortie.
    """
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=np.float32)


def build_standalone_tft_sequences(data):
    """Construit les sequences TFT sans C1/C2/C3 a partir du pipeline SOTA."""
    slope_mean = None
    if data["slope_map"] is not None:
        slope_mean = float(np.nanmean(data["slope_map"]))

    # L_granger=None + USE_ATT_AS_FEATURE=False => 3 canaux uniquement.
    X_all, y_all, seq_dates, ctx_all, n_feat = base.build_sequences(
        data["dates_all"],
        data["api_maps"],
        data["ndvi_maps"],
        data["et_maps"],
        data["tws_dict"],
        data["fine_profile"],
        data["normalizer"],
        seq_len=TFT_SEQ_LEN,
        slope_mean=slope_mean,
        L_granger=None,
        labels_2d=None,
        L_dict=None,
    )

    if n_feat != 3:
        raise RuntimeError(f"TFT standalone attendu avec 3 canaux, recu n_feat={n_feat}.")

    train_set = set(data["train_dates"])
    test_set = set(data["test_dates"])
    tr_mask = np.array([d in train_set for d in seq_dates], dtype=bool)
    te_mask = np.array([d in test_set for d in seq_dates], dtype=bool)

    X_tr, y_tr, ctx_tr = X_all[tr_mask], y_all[tr_mask], ctx_all[tr_mask]
    X_te, y_te, ctx_te = X_all[te_mask], y_all[te_mask], ctx_all[te_mask]
    seq_dates_te = [seq_dates[i] for i in range(len(seq_dates)) if te_mask[i]]

    if len(X_tr) < 2:
        raise RuntimeError(
            "Pas assez de sequences train. Reduisez TFT_SEQ_LEN ou verifiez les dates."
        )
    if len(X_te) < 1:
        raise RuntimeError(
            "Pas assez de sequences test. Reduisez TFT_SEQ_LEN ou verifiez le split."
        )

    print(f" Sequences TFT : train={len(X_tr)} | test={len(X_te)} | "
          f"seq_len={TFT_SEQ_LEN} | canaux={n_feat}")
    if seq_dates_te:
        print(f" Periode test sequence : {seq_dates_te[0]} -> {seq_dates_te[-1]}")

    return X_tr, y_tr, ctx_tr, X_te, y_te, ctx_te, seq_dates_te, n_feat


def train_tft_standalone(X_tr, y_tr, ctx_tr):
    """Entraine un TFT standalone, sans regularisation ni prior causal."""
    if not getattr(base, "_HAS_TFT", False):
        raise SystemExit("[ERREUR] PyTorch non installe. -> pip install torch")

    prior_uniform = np.ones(3, dtype=np.float32) / 3.0
    seeds = base.ENSEMBLE_SEEDS[:max(1, int(ENSEMBLE_MEMBERS))]

    print(" Hyperparametres TFT CPU :")
    print(f"   profile={TFT_PROFILE} | seq_len={TFT_SEQ_LEN} | "
          f"epochs={TFT_EPOCHS} | ensemble={len(seeds)} | batch={TFT_BATCH_SIZE}")
    print("   C1=off lambda=0 | C2=off gamma=0 | C3=off | fusion=aucune")

    # lambda_granger=0.0 + gamma=0.0 => entrainement purement HuberLoss.
    model, tws_scaler, device = base.train_tft_ensemble(
        X_tr,
        y_tr,
        ctx_tr,
        granger_prior_vec=prior_uniform,
        lambda_granger=0.0,
        seeds=seeds,
    )
    return model, tws_scaler, device


def export_tft_maps(model, tws_scaler, device, data, n_feat=3):
    """Exporte les cartes SR_TWS avec uniquement les predictions TFT.

    Retourne le dossier d'export pour permettre le calcul de sigma-bar.
    """
    out_dir = base.os.path.join(base.OUT_DIR, "SOTA_TFT_MAPS")

    # Masque clip Tensift identique a causal_tft_pipeline.py.
    clip_mask = base.build_clip_mask_from_shapefile(base.TENSIFT_SHP,
                                                   data["fine_profile"])

    # Matrice nulle uniquement pour satisfaire l'interface d'export ; beta=0.
    zero_L = np.zeros((3, 3), dtype=float)
    zero_rfr = _ZeroRegressor()

    base.export_downscaled_maps(
        tft_model=model,
        tws_scaler=tws_scaler,
        device=device,
        rfr=zero_rfr,
        weights=(1.0, 0.0),              # TFT uniquement
        L_dict=zero_L,
        beta=0.0,
        tws_std_train=1.0,
        normalizer=data["normalizer"],
        api_maps=data["api_maps"],
        ndvi_maps=data["ndvi_maps"],
        et_maps=data["et_maps"],
        fine_profile=data["fine_profile"],
        all_dates=data["dates_all"],
        map_dir=out_dir,
        labels_2d=None,
        slope_map=data["slope_map"],
        n_feat=n_feat,
        L_granger=zero_L,
        domain_mask=data["domain_mask"],
        clip_mask=clip_mask,
    )
    print(f" [EXPORT] Cartes TFT standalone ecrites dans : {out_dir}")
    return out_dir


def main():
    print("=" * 70)
    print(" SOTA — TEMPORAL FUSION TRANSFORMER STANDALONE (CPU-friendly)")
    print("=" * 70)

    data = sc.prepare_data()

    X_tr, y_tr, ctx_tr, X_te, y_te, ctx_te, seq_dates_te, n_feat = \
        build_standalone_tft_sequences(data)

    model, tws_scaler, device = train_tft_standalone(X_tr, y_tr, ctx_tr)

    y_pred = base.predict_tft_sequence(model, X_te, tws_scaler, device,
                                       slope_context=ctx_te)
    sc.report("TFT standalone", y_te, y_pred)

    out_dir = export_tft_maps(model, tws_scaler, device, data, n_feat=n_feat)

    # --------- VARIANCE SOUS-MAILLE (structure fine) ---------
    sg.report_sigma_bar("TFT standalone", out_dir, data)

    print("\n Termine (TFT standalone).")


if __name__ == "__main__":
    main()
