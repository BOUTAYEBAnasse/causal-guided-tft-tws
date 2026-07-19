# -*- coding: utf-8 -*-
"""
ablation_common.py (v3 — logique causal_tft_pipeline + CPU local)
=======================================================
Module partage pour l'etude d'ablation du papier.

Cette version ne depend plus de ``sota_common.py`` : elle reutilise directement
la chaine de donnees et les fonctions de ``causal_tft_pipeline.py`` afin que les scripts
ABLATION_TFT_RFR_uniform.py et ABLATION_TFT_RFR_corr.py s'executent dans le
meme environnement que le pipeline principal.

Correspondance avec causal_tft_pipeline.py :
  * creation/lecture de name_files_GRSL_TWS.txt ;
  * API avec decay module par la pente [B4] ;
  * interpolation + cascade gap-fill NDVI/ET ;
  * split 80/20 chronologique ;
  * normalisation GlobalMinMax ;
  * K-means hydro-climatique par silhouette [B5] ;
  * TFT causal-guide : [C1] KL, [C2] prior, [C3] canal d'attention ;
  * RFR sur [Pr, NDVI, ET, Slope] ;
  * fusion NNLS ou uniforme ;
  * beta residuel optionnel et recalibration affine OLS optionnelle ;
  * export des cartes via le meme clip Tensift + erosion que causal_tft_pipeline.

Execution CPU :
  Par defaut, si CUDA est absent, le profil ``cpu`` est active :
    seq_len=30, epochs=15, ensemble=3, batch=64.
  Pour reproduire le protocole papier/GRSL complet :
    set TFT_PROFILE=paper   (Windows CMD)
    export TFT_PROFILE=paper (Linux/macOS)

Variables d'environnement utiles :
  TFT_PROFILE=cpu|paper
  TFT_SEQ_LEN=30|60
  TFT_EPOCHS=15|60
  ENSEMBLE_MEMBERS=3|9
  TFT_BATCH_SIZE=64
  TFT_LAMBDA_MODE=fixed|grid
  MATRIX_DATE_SOURCE=all|train
  RFR_N_ESTIMATORS=300|800
"""

import os
import time
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import nnls

import causal_tft_pipeline as base


# =============================================================================
# 0) PROFIL CPU / PAPER + PROPAGATION DANS causal_tft_pipeline
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ON_CPU = DEVICE.type == "cpu"


def _envi(name, default):
    v = os.environ.get(name, "")
    if v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _envf(name, default):
    v = os.environ.get(name, "")
    if v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


PROFILE = os.environ.get("TFT_PROFILE", "cpu" if _ON_CPU else "paper").lower()

if PROFILE == "paper":
    _P_SEQ = 60
    _P_EPOCHS = 60
    _P_ENSEMBLE = 9
    _P_BATCH = 64
    _P_RFR_EST = int(base.RFR_PARAMS.get("n_estimators", 800))
else:
    _P_SEQ = 30
    _P_EPOCHS = 15
    _P_ENSEMBLE = 3
    _P_BATCH = 64
    _P_RFR_EST = 300

TFT_SEQ_LEN = _envi("TFT_SEQ_LEN", _P_SEQ)
TFT_EPOCHS = _envi("TFT_EPOCHS", _P_EPOCHS)
TFT_BATCH_SIZE = _envi("TFT_BATCH_SIZE", _P_BATCH)
ENSEMBLE_MEMBERS = _envi("ENSEMBLE_MEMBERS", _P_ENSEMBLE)
RFR_N_ESTIMATORS = _envi("RFR_N_ESTIMATORS", _P_RFR_EST)

# Constantes scientifiques reprises de causal_tft_pipeline / papier.
KL_LAMBDA_DEFAULT = _envf("KL_LAMBDA", getattr(base, "LAMBDA_GRANGER_DEFAULT", 0.01))
CAUSAL_PRIOR_GAMMA = _envf("GAMMA_PRIOR", getattr(base, "GAMMA_GRANGER_PRIOR", 0.5))
N_HYDROZONES = 3
MATRIX_DATE_SOURCE = os.environ.get("MATRIX_DATE_SOURCE", "all").lower()  # all = comme causal_tft_pipeline
TFT_LAMBDA_MODE = os.environ.get("TFT_LAMBDA_MODE", "fixed" if PROFILE == "cpu" else "grid").lower()

# Forcer causal_tft_pipeline a utiliser le meme profil lorsque ses fonctions sont appelees.
base.TFT_SEQ_LEN = TFT_SEQ_LEN
base.TFT_MAX_EPOCHS = TFT_EPOCHS
base.TFT_BATCH_SIZE = TFT_BATCH_SIZE
base.N_ENSEMBLE_SEEDS = ENSEMBLE_MEMBERS
base.GAMMA_GRANGER_PRIOR = CAUSAL_PRIOR_GAMMA
base.USE_ATT_AS_FEATURE = True

if _ON_CPU:
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    except Exception:
        pass

print(f"[ABLATION] device={DEVICE} | profile={PROFILE} | seq_len={TFT_SEQ_LEN} "
      f"epochs={TFT_EPOCHS} ensemble={ENSEMBLE_MEMBERS} batch={TFT_BATCH_SIZE}")
print(f"[ABLATION] lambda_mode={TFT_LAMBDA_MODE} | matrix_dates={MATRIX_DATE_SOURCE} "
      f"| RFR trees={RFR_N_ESTIMATORS}")


# =============================================================================
# 1) CONFIGURATION
# =============================================================================
@dataclass
class AblationConfig:
    name: str = "ablation"
    causal_matrix: str = "granger"       # granger | correlation | none
    use_C1: bool = True                  # KL loss
    use_C2: bool = True                  # prior dans la variable selection
    use_C3: bool = True                  # attention comme 4e canal
    fusion: str = "nnls"                 # nnls | uniform
    affine_recal: bool = True
    residual_beta: bool = True
    kl_lambda: float = KL_LAMBDA_DEFAULT
    ensemble_members: int = ENSEMBLE_MEMBERS
    n_zones: int = N_HYDROZONES


# =============================================================================
# 2) PREPARATION DES DONNEES — meme logique que causal_tft_pipeline.main()
# =============================================================================
def prepare_data():
    print("\n[DATA] Preparation identique a causal_tft_pipeline.py ...")
    base.ensure_name_file()

    sections = base.parse_name_file(base.NAME_FILE)
    pr_dict = base.build_date_dict(sections, base.SECTION_PR)
    tws_dict = base.build_date_dict(sections, base.SECTION_TWS)
    ndvi_dict = base.build_date_dict(sections, base.SECTION_NDVI)
    et_dict = base.build_date_dict(sections, base.SECTION_ET)

    dates_all = sorted(set(pr_dict.keys()) & set(tws_dict.keys()))
    if not dates_all:
        raise RuntimeError("Aucune date commune Pr/TWS. Verifiez name_files_GRSL_TWS.txt.")
    print(f"[DATA] dates={len(dates_all)} ({dates_all[0]} -> {dates_all[-1]})")
    print(f"[DATA] composites NDVI={len(ndvi_dict)} | ET={len(et_dict)}")

    fine_profile = base.get_fine_profile_from_pr(pr_dict)

    slope_path = base.get_static_path(sections, base.SECTION_SLOPE, base.SLOPE_ALIASES)
    slope_map, slope_mean = None, None
    if slope_path is not None:
        print(f"[DATA] Slope : {slope_path}")
        slope_map = base.load_var_on_fine(slope_path, fine_profile)
        slope_mean = float(np.nanmean(slope_map))
        print(f"[DATA] slope_mean={slope_mean:.3f}")
    else:
        print("[DATA][WARN] Slope introuvable : contexte statique mis a 0.")

    Hf, Wf = fine_profile["height"], fine_profile["width"]
    print("[DATA] Construction du masque de domaine TWS...")
    domain_mask = np.zeros((Hf, Wf), dtype=bool)
    for d in sorted(tws_dict.keys()):
        try:
            tws_c, gprof, gnod = base.load_raster(tws_dict[d])
            tws_c = base.to_float_with_nan(tws_c, gnod)
            tws_f = base.reproject_to_profile(
                tws_c, gprof, gnod, fine_profile,
                dst_nodata=np.nan, resampling=base.Resampling.nearest)
            domain_mask |= np.isfinite(tws_f)
        except Exception:
            continue
    if not domain_mask.any():
        print("[DATA][WARN] Domaine TWS vide -> masque ignore.")
        domain_mask = None
    else:
        print(f"[DATA] domaine={int(domain_mask.sum())} pixels "
              f"({100.0 * domain_mask.sum() / (Hf * Wf):.1f}% grille)")

    print("[DATA] API avec decay spatial par slope [B4]...")
    api_maps = base.compute_api_series(pr_dict, fine_profile, dates_all, slope_map=slope_map)
    if domain_mask is not None:
        for d in list(api_maps.keys()):
            api_maps[d] = base.spatial_fill_nearest(api_maps[d], domain_mask=domain_mask)

    print("[DATA] Interpolation + gap-fill NDVI/ET...")
    ndvi_maps = base.build_interp_series(ndvi_dict, fine_profile, dates_all,
                                         domain_mask=domain_mask, fill_residual=True)
    et_maps = base.build_interp_series(et_dict, fine_profile, dates_all,
                                       domain_mask=domain_mask, fill_residual=True)
    dates_all = [d for d in dates_all if d in ndvi_maps and d in et_maps]
    if not dates_all:
        raise RuntimeError("Aucune date avec NDVI+ET apres interpolation.")

    idx = np.arange(len(dates_all))
    if base.CHRONOLOGICAL_SPLIT:
        n_test = max(1, int(round(len(dates_all) * base.TEST_FRACTION)))
        cut = len(dates_all) - n_test
        train_dates = [dates_all[i] for i in idx[:cut]]
        test_dates = [dates_all[i] for i in idx[cut:]]
        print(f"[DATA] train={len(train_dates)} | test={len(test_dates)} "
              f"[chronologique : {test_dates[0]} -> {test_dates[-1]}]")
    else:
        step = int(round(1.0 / base.TEST_FRACTION))
        test_mask = (idx % step == 0)
        test_dates = [dates_all[i] for i in idx[test_mask]]
        train_dates = [dates_all[i] for i in idx[~test_mask]]
        print(f"[DATA] train={len(train_dates)} | test={len(test_dates)} [1-sur-{step}]")

    _, _, _, normalizer, _ = base.assemble_samples(
        train_dates, api_maps, ndvi_maps, et_maps, tws_dict,
        fine_profile, normalizer=None, fit_normalizer=True, slope_map=slope_map)

    return dict(
        sections=sections, pr_dict=pr_dict, tws_dict=tws_dict,
        api_maps=api_maps, ndvi_maps=ndvi_maps, et_maps=et_maps,
        fine_profile=fine_profile, slope_map=slope_map, slope_mean=slope_mean,
        domain_mask=domain_mask, dates_all=dates_all,
        train_dates=train_dates, test_dates=test_dates,
        normalizer=normalizer,
    )


# =============================================================================
# 3) MATRICES GRANGER / CORRELATION ET ZONES
# =============================================================================
def _norm_series(normalizer, name, values):
    return normalizer.transform(name, np.asarray(values, dtype=float))


def _corr_from_means(pr_ser, nd_ser, et_ser, normalizer):
    pr = _norm_series(normalizer, "PR", pr_ser)
    nd = _norm_series(normalizer, "NDVI", nd_ser)
    et = 1.0 - _norm_series(normalizer, "ET", et_ser)
    S = np.column_stack([pr, nd, et])
    S = S[np.isfinite(S).all(axis=1)]
    if S.shape[0] < 5:
        return np.zeros((3, 3), dtype=float)
    C = np.abs(np.corrcoef(S, rowvar=False))
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 0.0)
    return C.astype(float)


def build_correlation_global(api_maps, ndvi_maps, et_maps, dates, normalizer):
    pr, nd, et = [], [], []
    for d in dates:
        if d not in api_maps or d not in ndvi_maps or d not in et_maps:
            continue
        pr.append(float(np.nanmean(api_maps[d])))
        nd.append(float(np.nanmean(ndvi_maps[d])))
        et.append(float(np.nanmean(et_maps[d])))
    return _corr_from_means(pr, nd, et, normalizer)


def build_correlation_per_cluster(labels_2d, api_maps, ndvi_maps, et_maps,
                                  dates, normalizer, C_global,
                                  min_pixels=None):
    if min_pixels is None:
        min_pixels = getattr(base, "MIN_PIXELS_PER_CLUSTER", 25)
    labels_flat = labels_2d.ravel()
    C_dict = {-1: C_global}
    clusters = sorted(set(int(k) for k in np.unique(labels_flat) if k >= 0))
    for k in clusters:
        mask = labels_flat == k
        if mask.sum() < min_pixels:
            C_dict[k] = C_global
            print(f"[CORR] zone {k}: {mask.sum()} pixels < {min_pixels} -> fallback global")
            continue
        pr, nd, et = [], [], []
        for d in dates:
            if d not in api_maps or d not in ndvi_maps or d not in et_maps:
                continue
            pr.append(float(np.nanmean(api_maps[d].ravel()[mask])))
            nd.append(float(np.nanmean(ndvi_maps[d].ravel()[mask])))
            et.append(float(np.nanmean(et_maps[d].ravel()[mask])))
        Ck = _corr_from_means(pr, nd, et, normalizer)
        if not np.isfinite(Ck).all() or np.allclose(Ck, 0.0):
            Ck = C_global
            print(f"[CORR] zone {k}: correlation degeneree -> fallback global")
        C_dict[k] = Ck
    return C_dict


def build_matrices_and_zones(data, cfg: AblationConfig):
    zero = np.zeros((3, 3), dtype=float)
    if cfg.causal_matrix == "none":
        print("[MATRIX] Ablation sans matrice : C1/C2/C3 desactives.")
        return None, {-1: zero}, zero, 0

    matrix_dates = data["dates_all"] if MATRIX_DATE_SOURCE == "all" else data["train_dates"]

    print("[ZONES] Clustering hydro-climatique [B5] comme causal_tft_pipeline...")
    labels_2d, _kmeans, _cluster_scaler, best_K = base.build_pixel_clusters(
        data["api_maps"], data["ndvi_maps"], data["et_maps"], data["fine_profile"],
        matrix_dates, k_range=base.K_RANGE, seed=base.CLUSTER_SEED,
        slope_map=data["slope_map"])

    if cfg.causal_matrix == "granger":
        series = {
            "PR": np.array([np.nanmean(data["api_maps"][d]) for d in matrix_dates]),
            "NDVI": np.array([np.nanmean(data["ndvi_maps"][d]) for d in matrix_dates]),
            "ET": np.array([np.nanmean(data["et_maps"][d]) for d in matrix_dates]),
        }
        M_global = base.build_granger_matrix_multilag(series, ("PR", "NDVI", "ET"))
        print("[GRANGER] globale =\n", np.round(M_global, 4))
        M_dict = base.build_granger_matrices_per_cluster(
            labels_2d, data["api_maps"], data["ndvi_maps"], data["et_maps"],
            matrix_dates, M_global, min_pixels=base.MIN_PIXELS_PER_CLUSTER)
    elif cfg.causal_matrix == "correlation":
        M_global = build_correlation_global(
            data["api_maps"], data["ndvi_maps"], data["et_maps"],
            matrix_dates, data["normalizer"])
        print("[CORR] globale =\n", np.round(M_global, 4))
        M_dict = build_correlation_per_cluster(
            labels_2d, data["api_maps"], data["ndvi_maps"], data["et_maps"],
            matrix_dates, data["normalizer"], M_global,
            min_pixels=base.MIN_PIXELS_PER_CLUSTER)
    else:
        raise ValueError("causal_matrix doit etre 'granger', 'correlation' ou 'none'.")

    for k in sorted(M_dict.keys()):
        tag = "globale/fallback" if k == -1 else f"zone {k}"
        print(f"[MATRIX] {tag} =\n", np.round(M_dict[k], 4))
    return labels_2d, M_dict, M_global, best_K


# =============================================================================
# 4) SEQUENCES TFT — meme logique bassin-moyen que causal_tft_pipeline.build_sequences()
# =============================================================================
def make_prior_vec(M_global, cfg: AblationConfig):
    n_feat = 4 if cfg.use_C3 else 3
    if M_global is None or np.allclose(M_global, 0.0):
        prior = np.ones(n_feat, dtype=np.float32) / n_feat
    else:
        prior3 = base.granger_prior_vector(M_global).astype(np.float32)
        if cfg.use_C3:
            prior = np.append(prior3, 0.1).astype(np.float32)
            prior = prior / (prior.sum() + 1e-8)
        else:
            prior = prior3 / (prior3.sum() + 1e-8)
    print(f"[C2] prior utilise par le TFT = {np.round(prior, 4)} "
          f"(gamma={'on' if cfg.use_C2 else 'off'})")
    return prior


def build_tft_sequences(data, cfg: AblationConfig, M_global=None, M_dict=None, labels_2d=None):
    normalizer = data["normalizer"]
    api_maps, ndvi_maps, et_maps = data["api_maps"], data["ndvi_maps"], data["et_maps"]
    tws_dict, fine_profile = data["tws_dict"], data["fine_profile"]

    pr_means, nd_means, et_means, tws_means, att_means, valid_dates = [], [], [], [], [], []

    for d in data["dates_all"]:
        if d not in api_maps or d not in ndvi_maps or d not in et_maps or d not in tws_dict:
            continue
        tws_c, gprof, gnod = base.load_raster(tws_dict[d])
        tws_c = base.to_float_with_nan(tws_c, gnod)
        tws_f = base.reproject_to_profile(tws_c, gprof, gnod, fine_profile,
                                          dst_nodata=np.nan,
                                          resampling=base.Resampling.bilinear)
        tws_m = float(np.nanmean(tws_f))
        if not np.isfinite(tws_m):
            continue

        pr_m = float(np.nanmean(api_maps[d]))
        nd_m = float(np.nanmean(ndvi_maps[d]))
        et_m = float(np.nanmean(et_maps[d]))

        if cfg.use_C3 and M_global is not None and M_dict is not None:
            pr_n = float(normalizer.transform("PR", np.array([pr_m]))[0])
            nd_n = float(normalizer.transform("NDVI", np.array([nd_m]))[0])
            et_n = 1.0 - float(normalizer.transform("ET", np.array([et_m]))[0])
            V_d = np.array([[pr_n, nd_n, et_n]])
            if labels_2d is not None:
                all_labels = labels_2d.ravel()
                valid_pix = np.isfinite(api_maps[d].ravel())
                if valid_pix.any():
                    V_pix = np.tile(V_d, (valid_pix.sum(), 1))
                    lab_pix = all_labels[valid_pix]
                    att_m = float(np.mean(base.compute_attention_map_spatial(V_pix, M_dict, lab_pix)))
                else:
                    att_m = float(base.compute_attention_map(V_d, M_global)[0])
            else:
                att_m = float(base.compute_attention_map(V_d, M_global)[0])
        else:
            att_m = 0.0

        pr_means.append(pr_m); nd_means.append(nd_m); et_means.append(et_m)
        tws_means.append(tws_m); att_means.append(att_m); valid_dates.append(d)

    if len(valid_dates) <= TFT_SEQ_LEN:
        raise RuntimeError(f"Pas assez de dates ({len(valid_dates)}) pour seq_len={TFT_SEQ_LEN}.")

    pr_arr = normalizer.transform("PR", np.array(pr_means))
    nd_arr = normalizer.transform("NDVI", np.array(nd_means))
    et_arr = 1.0 - normalizer.transform("ET", np.array(et_means))
    att_arr = np.array(att_means, dtype=np.float32)

    slope_n = 0.0
    if data["slope_mean"] is not None and "SLOPE" in normalizer.vmin:
        slope_n = float(normalizer.transform("SLOPE", np.array([data["slope_mean"]]))[0])

    n_feat = 4 if cfg.use_C3 else 3
    X_seq, y_seq, seq_dates = [], [], []
    for i in range(TFT_SEQ_LEN, len(valid_dates)):
        chans = [
            pr_arr[i - TFT_SEQ_LEN:i],
            nd_arr[i - TFT_SEQ_LEN:i],
            et_arr[i - TFT_SEQ_LEN:i],
        ]
        if n_feat == 4:
            chans.append(att_arr[i - TFT_SEQ_LEN:i])
        X_seq.append(np.stack(chans, axis=1).astype(np.float32))
        y_seq.append(float(tws_means[i]))
        seq_dates.append(valid_dates[i])

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.float32)
    ctx = np.full((len(X_seq), 1), slope_n, dtype=np.float32)
    print(f"[SEQ] total={len(X_seq)} | canaux={n_feat} | seq_len={TFT_SEQ_LEN}")
    return X_seq, y_seq, seq_dates, ctx, n_feat


# =============================================================================
# 5) TFT — meme architecture CausalTFT, mais C1/C2 activables/desactivables
# =============================================================================
def causal_kl_loss(var_weights_mean, prior_tensor):
    p_obs = var_weights_mean + 1e-8
    p_obs = p_obs / p_obs.sum()
    p_ref = prior_tensor + 1e-8
    p_ref = p_ref / p_ref.sum()
    return (p_obs * torch.log(p_obs / p_ref)).sum()


def train_tft_ablation(X_tr_seq, y_tr_seq, slope_context_tr, prior_vec,
                       cfg: AblationConfig, lambda_granger=None, seed=None):
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    if lambda_granger is None:
        lambda_granger = cfg.kl_lambda

    y_min = float(np.min(y_tr_seq))
    y_max = float(np.max(y_tr_seq))
    denom = (y_max - y_min) if (y_max - y_min) > 0 else 1.0
    y_norm = (y_tr_seq - y_min) / denom

    n_feat = int(X_tr_seq.shape[2])
    gamma = CAUSAL_PRIOR_GAMMA if cfg.use_C2 else 0.0
    lam = float(lambda_granger) if cfg.use_C1 else 0.0

    dataset = base.TWSSequenceDataset(X_tr_seq, y_norm, slope_context=slope_context_tr)
    loader = DataLoader(dataset, batch_size=TFT_BATCH_SIZE, shuffle=True, num_workers=0)

    context_size = slope_context_tr.shape[1] if slope_context_tr is not None else 1
    model = base.CausalTFT(
        input_size=n_feat, context_size=context_size,
        seq_len=TFT_SEQ_LEN, hidden_size=base.TFT_HIDDEN_SIZE,
        n_heads=base.TFT_ATTENTION_HEADS, dropout=base.TFT_DROPOUT,
        hidden_cont=base.TFT_HIDDEN_CONT,
        granger_prior=prior_vec, gamma=gamma,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=base.TFT_LR,
                                 weight_decay=base.TFT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss()

    print(f"   TFT seed={seed} sur {DEVICE} | ep={TFT_EPOCHS} | canaux={n_feat} "
          f"| C1={'on' if cfg.use_C1 else 'off'} lambda={lam} "
          f"| C2={'on' if cfg.use_C2 else 'off'} gamma={gamma}")
    t0 = time.time()
    for epoch in range(TFT_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb, cb in loader:
            xb, yb, cb = xb.to(DEVICE), yb.to(DEVICE), cb.to(DEVICE)
            optimizer.zero_grad()
            pred, var_w = model(xb, context=cb)
            loss_pred = criterion(pred, yb)
            if cfg.use_C1 and lam > 0:
                loss = loss_pred + lam * causal_kl_loss(var_w, model.granger_prior)
            else:
                loss = loss_pred
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss_pred.item() * len(yb)
        avg_loss = epoch_loss / max(1, len(dataset))
        scheduler.step(avg_loss)
        print(f"      epoch {epoch + 1:03d}/{TFT_EPOCHS} | loss_pred={avg_loss:.5f}", flush=True)
    print(f"   TFT seed={seed} termine en {time.time() - t0:.1f}s")
    return model, {"min": y_min, "max": y_max, "denom": denom}, DEVICE


def predict_tft(model, X_seq, scaler, ctx):
    return base.predict_tft_sequence(model, X_seq, scaler, DEVICE, slope_context=ctx)


def train_tft_ensemble_ablation(X_tr_seq, y_tr_seq, ctx_tr, prior_vec,
                                cfg: AblationConfig, lambda_granger):
    seeds = base.ENSEMBLE_SEEDS[:max(1, cfg.ensemble_members)]
    members, scalers = [], []
    for i, sd in enumerate(seeds):
        print(f"[ENSEMBLE] membre {i + 1}/{len(seeds)} | seed={sd}")
        m, sc, _ = train_tft_ablation(
            X_tr_seq, y_tr_seq, ctx_tr, prior_vec, cfg,
            lambda_granger=lambda_granger, seed=sd)
        members.append(m); scalers.append(sc)
    return base.TFTEnsemble(members, scalers, DEVICE), scalers[0], DEVICE


# =============================================================================
# 6) FUSION, RFR, RECALIBRATION
# =============================================================================
def train_rfr(data):
    X_tr_pix, y_tr_pix, _, _, _ = base.assemble_samples(
        data["train_dates"], data["api_maps"], data["ndvi_maps"], data["et_maps"],
        data["tws_dict"], data["fine_profile"], normalizer=data["normalizer"],
        fit_normalizer=False, slope_map=data["slope_map"])
    params = dict(base.RFR_PARAMS)
    params["n_estimators"] = RFR_N_ESTIMATORS
    if "RFR_MAX_FEATURES" in os.environ:
        params["max_features"] = _envf("RFR_MAX_FEATURES", params.get("max_features", 0.4))
    max_samples = _envf("RFR_MAX_SAMPLES", getattr(base, "RFR_MAX_SAMPLES", 0.8))
    print(f"[RFR] fit X={X_tr_pix.shape} | trees={params['n_estimators']} | max_samples={max_samples}")
    rfr = RandomForestRegressor(
        bootstrap=True, max_samples=max_samples, n_jobs=-1,
        random_state=base.RANDOM_SEED, **params)
    rfr.fit(X_tr_pix, y_tr_pix)
    tws_std_train = float(np.nanstd(y_tr_pix)) if y_tr_pix.size > 1 else 1.0
    return rfr, tws_std_train


def learn_weights(tft_val, rfr_val, y_val, mode="nnls"):
    ok = np.isfinite(tft_val) & np.isfinite(rfr_val) & np.isfinite(y_val)
    if ok.sum() < 2 or mode == "uniform":
        return 0.5, 0.5
    A = np.column_stack([tft_val[ok], rfr_val[ok]])
    w, _ = nnls(A, y_val[ok])
    s = float(w.sum())
    if s <= 0:
        return 0.5, 0.5
    w = w / s
    return float(w[0]), float(w[1])


def fit_affine_if_better(y_true, y_pred, enabled=True):
    if not enabled:
        return 1.0, 0.0
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 2:
        return 1.0, 0.0
    yt, yp = y_true[ok], y_pred[ok]
    vp_mean, vt_mean = float(np.mean(yp)), float(np.mean(yt))
    var_p = float(np.mean((yp - vp_mean) ** 2))
    cov_pt = float(np.mean((yp - vp_mean) * (yt - vt_mean)))
    if var_p <= 1e-12:
        return 1.0, 0.0
    alpha = cov_pt / var_p
    b = vt_mean - alpha * vp_mean
    rmse_before = float(np.sqrt(mean_squared_error(yt, yp)))
    rmse_after = float(np.sqrt(mean_squared_error(yt, alpha * yp + b)))
    if rmse_after < rmse_before:
        print(f"[AFFINE] retenue : alpha={alpha:.4f} b={b:.4f} | "
              f"RMSE_val {rmse_before:.4f}->{rmse_after:.4f}")
        return float(alpha), float(b)
    print(f"[AFFINE] non retenue : RMSE_val {rmse_before:.4f} vs {rmse_after:.4f}")
    return 1.0, 0.0


def search_beta(y_val, base_val, X_val, M_global, tws_std_train, enabled=True):
    if (not enabled) or M_global is None or np.allclose(M_global, 0.0):
        return 0.0
    grid = getattr(base, "BETA_GRID", [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2])
    att = base.compute_attention_map(X_val[:, -1, :3], M_global)
    ok = np.isfinite(y_val) & np.isfinite(base_val) & np.isfinite(att)
    if ok.sum() < 2:
        return 0.0
    best_beta, best_rmse = 0.0, np.inf
    for beta in grid:
        pred = base_val[ok] + beta * tws_std_train * att[ok]
        rmse = float(np.sqrt(mean_squared_error(y_val[ok], pred)))
        if rmse < best_rmse:
            best_beta, best_rmse = float(beta), rmse
    print(f"[BETA] beta={best_beta} | RMSE_val={best_rmse:.4f}")
    return best_beta


def report(name, y_true, y_pred):
    rmse, r2, r = base.metrics(y_true, y_pred)
    print("\n=== RESULTAT ABLATION ===")
    print(f"{name:<34} RMSE={rmse:.4f} mm | R2={r2 * 100:.2f}% | r={r * 100:.2f}%")
    return rmse, r2, r


# =============================================================================
# 7) ORCHESTRATEUR
# =============================================================================
def run_ablation(cfg: AblationConfig):
    print("=" * 78)
    print(f"ABLATION — {cfg.name}")
    print(f"matrix={cfg.causal_matrix} | C1={cfg.use_C1} C2={cfg.use_C2} C3={cfg.use_C3} "
          f"| fusion={cfg.fusion} | affine={cfg.affine_recal} | beta={cfg.residual_beta}")
    print("=" * 78)

    data = prepare_data()
    labels_2d, M_dict, M_global, best_K = build_matrices_and_zones(data, cfg)
    prior_vec = make_prior_vec(M_global, cfg)

    X_all, y_all, seq_dates, ctx_all, n_feat = build_tft_sequences(
        data, cfg, M_global=M_global, M_dict=M_dict, labels_2d=labels_2d)

    train_set, test_set = set(data["train_dates"]), set(data["test_dates"])
    tr_mask = np.array([d in train_set for d in seq_dates])
    te_mask = np.array([d in test_set for d in seq_dates])
    X_tr_full, y_tr_full, ctx_tr_full = X_all[tr_mask], y_all[tr_mask], ctx_all[tr_mask]
    X_te, y_te, ctx_te = X_all[te_mask], y_all[te_mask], ctx_all[te_mask]
    seq_dates_tr = [seq_dates[i] for i in range(len(seq_dates)) if tr_mask[i]]
    seq_dates_te = [seq_dates[i] for i in range(len(seq_dates)) if te_mask[i]]
    if len(X_tr_full) < 3 or len(X_te) < 1:
        raise RuntimeError("Split sequence insuffisant. Reduisez TFT_SEQ_LEN ou ajoutez des dates.")
    print(f"[SEQ] train={len(X_tr_full)} | test={len(X_te)}")

    order = np.argsort(np.array(seq_dates_tr))
    n_val = max(1, int(round(len(X_tr_full) * base.ENSEMBLE_VAL_FRACTION)))
    if n_val >= len(order):
        n_val = max(1, len(order) // 3)
    sub_idx = order[:-n_val]
    val_idx = order[-n_val:]
    X_sub, y_sub, ctx_sub = X_tr_full[sub_idx], y_tr_full[sub_idx], ctx_tr_full[sub_idx]
    X_val, y_val, ctx_val = X_tr_full[val_idx], y_tr_full[val_idx], ctx_tr_full[val_idx]
    val_dates = [seq_dates_tr[i] for i in val_idx]
    print(f"[SPLIT] sous-train={len(X_sub)} | validation={len(X_val)}")

    # Selection de lambda : par defaut fixed sur CPU, grid en profil paper.
    best_lambda = 0.0 if not cfg.use_C1 else cfg.kl_lambda
    if cfg.use_C1 and TFT_LAMBDA_MODE == "grid":
        best_rmse = np.inf
        grid = getattr(base, "LAMBDA_GRANGER_GRID", [0.0, 0.001, 0.005, 0.01, 0.05, 0.1])
        print(f"[C1] Recherche lambda sur validation : {grid}")
        for lam in grid:
            m_try, sc_try, _ = train_tft_ablation(X_sub, y_sub, ctx_sub, prior_vec, cfg,
                                                  lambda_granger=lam, seed=base.RANDOM_SEED)
            p_try = predict_tft(m_try, X_val, sc_try, ctx_val)
            ok = np.isfinite(p_try) & np.isfinite(y_val)
            rmse_try = float(np.sqrt(mean_squared_error(y_val[ok], p_try[ok]))) if ok.sum() >= 2 else np.inf
            print(f"   lambda={lam} -> RMSE_val={rmse_try:.4f}")
            if rmse_try < best_rmse:
                best_rmse, best_lambda = rmse_try, float(lam)
        print(f"[C1] lambda optimal={best_lambda} | RMSE_val={best_rmse:.4f}")
    else:
        print(f"[C1] lambda fixe={best_lambda}")

    print("[TFT] Modele temporaire sur sous-train pour apprendre fusion/recalibration...")
    tft_tmp, scaler_tmp, _ = train_tft_ablation(
        X_sub, y_sub, ctx_sub, prior_vec, cfg,
        lambda_granger=best_lambda, seed=base.RANDOM_SEED)

    print("[RFR] Entrainement sur train complet pixel-wise...")
    rfr, tws_std_train = train_rfr(data)
    print(f"[RFR] std_TWS_train={tws_std_train:.4f} mm")

    tft_val = predict_tft(tft_tmp, X_val, scaler_tmp, ctx_val)
    rfr_val_dict = base.rfr_predict_per_date(
        rfr, val_dates, data["api_maps"], data["ndvi_maps"], data["et_maps"],
        data["tws_dict"], data["fine_profile"], slope_map=data["slope_map"])
    rfr_val = np.array([rfr_val_dict.get(d, np.nan) for d in val_dates])

    w1, w2 = learn_weights(tft_val, rfr_val, y_val, mode=cfg.fusion)
    print(f"[FUSION] {cfg.fusion} -> w1(TFT)={w1:.4f} | w2(RFR)={w2:.4f}")

    val_base = w1 * tft_val + w2 * rfr_val
    beta = search_beta(y_val, val_base, X_val, M_global, tws_std_train,
                       enabled=(cfg.residual_beta and cfg.causal_matrix != "none"))
    if beta != 0.0:
        val_base = val_base + beta * tws_std_train * base.compute_attention_map(X_val[:, -1, :3], M_global)
    alpha, b = fit_affine_if_better(y_val, val_base, enabled=cfg.affine_recal)

    print("[A4] Re-entrainement ensemble TFT sur 100% du train...")
    tft_model, tws_scaler, device = train_tft_ensemble_ablation(
        X_tr_full, y_tr_full, ctx_tr_full, prior_vec, cfg, lambda_granger=best_lambda)

    rfr_te_dict = base.rfr_predict_per_date(
        rfr, seq_dates_te, data["api_maps"], data["ndvi_maps"], data["et_maps"],
        data["tws_dict"], data["fine_profile"], slope_map=data["slope_map"])
    rfr_te = np.array([rfr_te_dict.get(d, np.nan) for d in seq_dates_te])
    tft_te = predict_tft(tft_model, X_te, tws_scaler, ctx_te)
    y_pred = w1 * tft_te + w2 * rfr_te
    if beta != 0.0:
        y_pred = y_pred + beta * tws_std_train * base.compute_attention_map(X_te[:, -1, :3], M_global)
    y_pred = alpha * y_pred + b

    # Statistiques par membre, utiles pour la ligne d'ablation.
    if hasattr(tft_model, "members"):
        per_member = []
        for i, m in enumerate(tft_model.members):
            p_i = base.predict_tft_sequence(m, X_te, tft_model.scalers[i], device, slope_context=ctx_te)
            yp_i = w1 * p_i + w2 * rfr_te
            if beta != 0.0:
                yp_i = yp_i + beta * tws_std_train * base.compute_attention_map(X_te[:, -1, :3], M_global)
            yp_i = alpha * yp_i + b
            ok_i = np.isfinite(yp_i) & np.isfinite(y_te)
            per_member.append(base.metrics(y_te[ok_i], yp_i[ok_i]))
        per_member = np.asarray(per_member, dtype=float)
        if per_member.size:
            mean_m, std_m = np.nanmean(per_member, axis=0), np.nanstd(per_member, axis=0)
            print(f"[ENSEMBLE] membres RMSE={mean_m[0]:.4f}+/-{std_m[0]:.4f} | "
                  f"R2={mean_m[1]*100:.2f}+/-{std_m[1]*100:.2f}% | "
                  f"r={mean_m[2]*100:.2f}+/-{std_m[2]*100:.2f}%")

    ok = np.isfinite(y_pred) & np.isfinite(y_te)
    rmse, r2, r = report(cfg.name, y_te[ok], y_pred[ok])

    return dict(
        rmse=rmse, r2=r2, r=r, cfg=cfg, data=data,
        labels_2d=labels_2d, M_dict=M_dict, M_global=M_global, best_K=best_K,
        prior_vec=prior_vec, tft_model=tft_model, tws_scaler=tws_scaler,
        device=device, rfr=rfr, w1=w1, w2=w2, beta=beta,
        tws_std_train=tws_std_train, alpha=alpha, b=b,
        n_feat=n_feat, lambda_opt=best_lambda,
        y_test=y_te[ok], y_pred=y_pred[ok], seq_dates_test=seq_dates_te,
    )


# =============================================================================
# 8) EXPORT DES CARTES — meme fonction que causal_tft_pipeline
# =============================================================================
def export_fused_maps(result, out_subdir):
    data = result["data"]
    out_dir = os.path.join(base.OUT_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    clip_mask = base.build_clip_mask_from_shapefile(base.TENSIFT_SHP, data["fine_profile"])
    M_global = result["M_global"]
    M_dict = result["M_dict"]
    if M_global is None:
        M_global = np.zeros((3, 3), dtype=float)
    if M_dict is None:
        M_dict = {-1: M_global}

    print(f"[EXPORT] cartes vers : {out_dir}")
    base.export_downscaled_maps(
        result["tft_model"], result["tws_scaler"], result["device"], result["rfr"],
        (result["w1"], result["w2"]), M_dict,
        result["beta"], result["tws_std_train"], data["normalizer"],
        data["api_maps"], data["ndvi_maps"], data["et_maps"], data["fine_profile"],
        data["dates_all"], out_dir,
        labels_2d=result["labels_2d"], slope_map=data["slope_map"],
        n_feat=result["n_feat"], L_granger=M_global,
        domain_mask=data["domain_mask"], clip_mask=clip_mask)
    print(f"[EXPORT] termine : {out_dir}")


def save_summary(result, out_name=None):
    if out_name is None:
        out_name = f"{result['cfg'].name}_summary.txt"
    path = os.path.join(base.OUT_DIR, out_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"name={result['cfg'].name}\n")
        f.write(f"rmse={result['rmse']}\nr2={result['r2']}\nr={result['r']}\n")
        f.write(f"w1={result['w1']}\nw2={result['w2']}\n")
        f.write(f"beta={result['beta']}\nalpha={result['alpha']}\nb={result['b']}\n")
        f.write(f"lambda_opt={result['lambda_opt']}\nK={result['best_K']}\n")
        f.write(f"n_feat={result['n_feat']}\nprofile={PROFILE}\n")
    print(f"[SAVE] resume : {path}")
    return path
