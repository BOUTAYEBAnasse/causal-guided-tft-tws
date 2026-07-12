# -*- coding: utf-8 -*-
"""
SOTA_SVR_FAST_v2.py — Version allegée et corrigée du modele SVR.

Correction :
  LinearSVR(loss="squared_epsilon_insensitive", dual=False)

Cette correction evite :
  ValueError: Unsupported set of arguments:
  penalty='l2', loss='epsilon_insensitive', dual=False

Modes disponibles :
  FAST_MODE = "nystroem"  : recommande, approximation RBF rapide
  FAST_MODE = "linear"    : tres rapide, mais lineaire
  FAST_MODE = "rbf_light" : RBF exact allege, peut rester lent

AJOUT : calcul de la variance sous-maille (sigma-bar) sur les cartes fines
exportees (metrique de structure fine, sous-section "Sub-grid Variance").
"""

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_approximation import Nystroem

import GRSL_v12 as base
import sota_common as sc
import subgrid_common as sg

SEED = base.RANDOM_SEED

FAST_MODE = "nystroem"

USE_TRAIN_SUBSAMPLING = True
MAX_TRAIN_SAMPLES = 60000

NYSTROEM_COMPONENTS = 300
NYSTROEM_GAMMA = 0.25

LINEAR_SVR_PARAMS = {
    "C": 10.0,
    "epsilon": 0.05,
    "loss": "squared_epsilon_insensitive",
    "dual": False,
    "max_iter": 10000,
    "tol": 1e-4,
    "random_state": SEED,
}


def maybe_subsample(X, y, max_samples=60000, seed=42):
    """Sous-echantillonnage aleatoire reproductible."""
    n = X.shape[0]
    if n <= max_samples:
        return X, y

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    idx.sort()
    print(f" [FAST] Sous-echantillonnage train : {n} -> {max_samples} echantillons")
    return X[idx], y[idx]


def make_model():
    """Construit le modele selon FAST_MODE."""
    if FAST_MODE == "nystroem":
        regressor = Pipeline([
            ("x_scaler", StandardScaler()),
            ("rbf_approx", Nystroem(
                kernel="rbf",
                gamma=NYSTROEM_GAMMA,
                n_components=NYSTROEM_COMPONENTS,
                random_state=SEED
            )),
            ("z_scaler", StandardScaler()),
            ("linear_svr", LinearSVR(**LINEAR_SVR_PARAMS)),
        ])

    elif FAST_MODE == "linear":
        regressor = Pipeline([
            ("x_scaler", StandardScaler()),
            ("linear_svr", LinearSVR(**LINEAR_SVR_PARAMS)),
        ])

    elif FAST_MODE == "rbf_light":
        regressor = Pipeline([
            ("x_scaler", StandardScaler()),
            ("svr", SVR(
                kernel="rbf",
                C=10.0,
                epsilon=0.10,
                gamma="scale",
                cache_size=1000
            )),
        ])

    else:
        raise ValueError(
            "FAST_MODE invalide. Utiliser : 'nystroem', 'linear' ou 'rbf_light'."
        )

    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler()
    )


def main():
    print("=" * 70)
    print(" SOTA — SUPPORT VECTOR REGRESSOR FAST v2")
    print(f" Mode : {FAST_MODE}")
    print("=" * 70)

    data = sc.prepare_data()

    X_tr, y_tr = sc.build_xy(data["train_dates"], data)
    X_te, y_te = sc.build_xy(data["test_dates"],  data)

    print(f" Echantillons train: {X_tr.shape} | test: {X_te.shape}")

    if USE_TRAIN_SUBSAMPLING:
        X_tr, y_tr = maybe_subsample(
            X_tr,
            y_tr,
            max_samples=MAX_TRAIN_SAMPLES,
            seed=SEED
        )

    print(f" Parametres LinearSVR : {LINEAR_SVR_PARAMS}")

    if FAST_MODE == "nystroem":
        print(
            f" Approximation RBF : Nystroem("
            f"n_components={NYSTROEM_COMPONENTS}, gamma={NYSTROEM_GAMMA})"
        )

    model = make_model()

    print(" Apprentissage du modele FAST...")
    model.fit(X_tr, y_tr)

    print(" Prediction test...")
    y_pred = model.predict(X_te)

    sc.report(f"SVR_FAST_{FAST_MODE}", y_te, y_pred)

    out_dir = base.os.path.join(base.OUT_DIR, f"SOTA_SVR_FAST_v2_{FAST_MODE}_MAPS")

    print(" Export des cartes...")
    sc.export_maps_generic(model.predict, data, out_dir)

    # --------- VARIANCE SOUS-MAILLE (structure fine) ---------
    sg.report_sigma_bar(f"SVR_{FAST_MODE}", out_dir, data)

    print("\n Termine (SVR FAST v2).")


if __name__ == "__main__":
    main()
