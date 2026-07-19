# -*- coding: utf-8 -*-
"""
==============================================================================
 Innovative Causal Attention Approach for Downscaling Terrestrial Water Storage
 (GRSL paper) - Tensift Watershed, Morocco
 *** VERSION ENSEMBLE + GRANGER SPATIALEMENT VARIABLE — v4 (CAUSAL-GUIDED) ***
 *** TFT + RFR + Spatially-Explicit Causal Attention ***
==============================================================================

DIAGNOSTIC FONDAMENTAL (v2/v3) :
  beta_opt = 0.0 systematiquement, car la causalite de Granger est appliquee
  APRES la prediction finie. Le TFT+RFR capture deja ~91% de la variance ;
  aucun residu structure ne reste a corriger en post-hoc.

SOLUTION v4 — ARCHITECTURE "CAUSAL-GUIDED DEEP LEARNING" :
  L'attention de Granger n'est plus un post-processing sur TWS_pred.
  Elle est integree PENDANT l'apprentissage du TFT par deux mecanismes :

  [C1] REGULARISATION CAUSALE DANS LA LOSS DU TFT :
       Loss_total = Loss_Huber(y_pred, y_true)
                  + lambda * Loss_Granger(var_weights, L_granger)
       Loss_Granger = KL( var_weights || p_granger )
       p_granger[i] = softmax( sum_j L[j,i] )  — influence causale recue
       var_weights   = softmax( var_select_logits ) — poids de selection TFT
       Justification : les poids de selection des variables (Variable Selection
       Network) doivent etre coherents avec la causalite de Granger. Si Granger
       indique que ET est fortement influence par Pr et NDVI, le TFT doit
       accorder plus de poids a Pr et NDVI dans sa selection. La divergence KL
       penalise toute selection de variables qui contredit la structure causale.
       lambda optimise par CV sur {0.001, 0.005, 0.01, 0.05, 0.1}.

  [C2] PRIOR GRANGER DANS LA VARIABLE SELECTION (Eq.VS') :
       logits_VS = W_select(var_cat) + gamma * granger_prior
       granger_prior[i] = sum_j L[j,i] / sum_total  (influence recue par i)
       gamma = 0.5 (hyperparametre fixe, ajustable)
       Justification : le prior biaise l'initialisation de la selection
       vers les variables causalement influentes. L'apprentissage peut
       le surpasser si les donnees le justifient, mais le prior garantit
       que la structure causale de Granger est le point de depart.
       Difference vs C1 : C2 agit sur l'architecture (initialisation),
       C1 agit sur la loss (entrainement). Ensemble ils assurent que
       Granger guide le TFT a DEUX niveaux.

  [C3] ATTENTION CAUSALE SPATIALE COMME FEATURE SUPPLEMENTAIRE DU TFT :
       La carte d'attention Att_Granger(p,d) est calculee pour chaque date
       d'entrainement et ajoutee comme 4e canal de la sequence temporelle.
       Justification : le TFT apprend directement la relation entre la
       structure causale locale (qui varie selon la zone hydro-climatique)
       et la dynamique de TWS. L'attention de Granger devient un PREDICTOR
       du TFT, non un post-hoc. Cela permet au modele de ponderer l'apport
       causal DIFFEREMMENT selon les periodes (saison seche vs humide).

HERITAGES DE v2/v3 (conserves) :
  [A2] Slope = contexte GRN du TFT (hors sequence temporelle)
  [A3] Normalisation intra-zone de l'attention causale spatiale
  [A4] TFT reentrainee sur 100% du train apres optimisation lambda
  [B2] Matrices de Granger multi-lag ponderes AIC
  [B3] Slope pixel-par-pixel a l'inference spatiale
  [B4] API spatialement variable : decay module par slope (Eq.1')
  [B5] K clusters optimal par silhouette score

ARCHITECTURE v4 :
  Sequences TFT : [Pr, NDVI, 1-ET, Att_Granger_spatial] (4 canaux) [C3]
  Variable Selection : logits += gamma * granger_prior [C2]
  Loss : Huber + lambda * KL(var_weights || p_granger) [C1]
  Contexte GRN final : Slope pixel [A2]
  RFR : [Pr, NDVI, ET, Slope] (inchange)
  Fusion : TWS_base = w1*TFT + w2*RFR (NNLS)
  Eq.11 : TWS_pred = TWS_base  (beta=0 attendu car Granger est dans TFT)
          ou correction residuelle si beta_opt != 0

Dependances :
  numpy, rasterio, scikit-learn, statsmodels, torch
  pip install torch scikit-learn statsmodels rasterio
==============================================================================
"""

import os
import re
import glob
import warnings
from datetime import datetime, date, timedelta

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_geom
from rasterio.crs import CRS
from rasterio.features import geometry_mask

try:
    import fiona
    _HAS_FIONA = True
except Exception:
    _HAS_FIONA = False

from sklearn.metrics import r2_score, mean_squared_error, silhouette_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    _HAS_TFT = True
    print(f"[INFO] PyTorch {torch.__version__} detecte — TFT natif actif.")
except ImportError:
    _HAS_TFT = False
    print("[WARN] PyTorch non disponible. pip install torch")

warnings.filterwarnings("ignore")


# =============================================================================
# 0) PARAMETRES GENERAUX
# =============================================================================
np.random.seed(42)
RANDOM_SEED = 42

# [FIX-8] Ensemble de TFT a seeds multiples (moyenne des predictions).
#   Reduit la variance -> anti-overfitting + donne moyenne +/- ecart-type.
#   Mettre a 1 pour retrouver le comportement v7 (un seul TFT).
N_ENSEMBLE_SEEDS = 9
ENSEMBLE_SEEDS   = [42, 123, 2024, 7, 99, 314, 1618, 271, 577]   # tronques a N_ENSEMBLE_SEEDS

# [FIX-11] Agregation de l'ensemble : "mean" (v10) ou "median".
#   La mediane est plus robuste aux membres aberrants (ex. seed instable),
#   sans risque d'overfitting (simple agregation). A/B : comparez les deux.
ENSEMBLE_AGG = "median"

# [FIX-9] Recalibration d'echelle (correction affine pente/biais).
#   Quand r >> R2, le modele suit la FORME mais rate l'AMPLITUDE.
#   On ajuste y_cal = alpha*y_pred + b par moindres carres SUR LA VALIDATION
#   (jamais sur le test), puis on applique alpha/b au test. 2 parametres -> anti-overfit.
#   Mettre False pour retrouver le comportement v8.
RECALIBRATE_SCALE = True

BASE_DIR  = r"C:\Users\PC\Downloads\Project_Tensift"
NAME_FILE = os.path.join(BASE_DIR, "name_files_GRSL_TWS.txt")

OUT_DIR = os.path.join(BASE_DIR, "GRSL_CAUSAL_ATTENTION_TFT_v4_TENSIFT_TWS")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_NODATA = -9999.0

# =============================================================================
# [CLIP] Decoupage des rasters TWS SR sur le bassin de Tensift
# -----------------------------------------------------------------------------
# Le raster en sortie est une grille rectangulaire. Pour ne garder que le
# bassin, on decoupe (clip) chaque carte sur le contour du shapefile Tensift.
# Tout pixel HORS du polygone devient NoData.
TENSIFT_SHP = r"C:\Users\PC\Downloads\Project_Tensift\Tensift.shp"

# =============================================================================
# [EDGE-FIX] Correction de l'artefact de bord ("anneau au minimum")
# -----------------------------------------------------------------------------
# Cause : sur l'anneau de bordure, les covariables (NDVI/ET/API) sont
# extrapolees ou comblees par plus-proche-voisin (spatial_fill_nearest) :
# elles sortent de la distribution d'entrainement et le modele leur attribue
# des valeurs extremes (= minimum de la plage), d'ou le liisere sombre.
#
# Correction : le masque d'export final est l'INTERSECTION du domaine de
# donnees, du polygone du bassin (clip shapefile), puis EROD(e) d'un anneau de
# EDGE_EROSION_PIXELS pixels pour retirer les pixels de bord peu fiables.
# Resultat : forme du bassin nette, sans liisere minimal. Mettre 0 pour
# desactiver l'erosion (clip seul).
EDGE_EROSION_PIXELS = 1

# [EDGE-FIX] Filet de securite : on masque (-> NoData) les predictions hors de
# l'intervalle [p_low, p_high] des percentiles calcules SUR LES PIXELS FIABLES
# (apres erosion) de chaque carte. Capte d'eventuels pics residuels non lies au
# bord. Mettre CLIP_PREDICTION_PERCENTILES = False pour desactiver.
CLIP_PREDICTION_PERCENTILES = False
PRED_PERCENTILE_LOW  = 0.5
PRED_PERCENTILE_HIGH = 99.5

STUDY_START   = date(2023, 3, 31)
STUDY_END     = date(2025, 10, 30)
TEST_FRACTION = 0.20

# [FIX-7] True  = test = derniere periode chronologique (generalisation honnete au futur).
#         False = ancien sous-echantillonnage 1-sur-5 (test non independant, R2 optimiste).
CHRONOLOGICAL_SPLIT = True

# --- [B4] API avec decay slope ---
API_DECAY_J           = 0.9
API_SLOPE_K           = 0.5
RECOMPUTE_API_FROM_RAW = True

# --- [B2] Granger multi-lag ---
GRANGER_MAXLAG = 5

# --- [B5] Silhouette pour K ---
K_RANGE               = [2, 3, 4]   # [FIX-1b] borne K pour petit bassin (zones peuplees)
MIN_PIXELS_PER_CLUSTER = 25          # [FIX-1a] seuil adapte a Tensift (~247 px)
CLUSTER_SEED          = RANDOM_SEED

TARGET_RES_DEG = 0.05

# =============================================================================
# [C1] HYPERPARAMETRES DE REGULARISATION CAUSALE
# =============================================================================
LAMBDA_GRANGER_GRID = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
LAMBDA_GRANGER_DEFAULT = 0.01

# =============================================================================
# [C2] PRIOR GRANGER DANS VARIABLE SELECTION
# =============================================================================
GAMMA_GRANGER_PRIOR = 0.5   # intensite du prior causal (0 = desactive)

# =============================================================================
# [C3] ATTENTION COMME 4e CANAL DE SEQUENCE
# =============================================================================
USE_ATT_AS_FEATURE = True   # ajoute Att_Granger comme 4e canal [Pr,NDVI,1-ET,Att]

# =============================================================================
# HYPERPARAMETRES TFT
# =============================================================================
TFT_SEQ_LEN         = 60
TFT_HIDDEN_SIZE     = 64
TFT_ATTENTION_HEADS = 4
TFT_DROPOUT         = 0.1
TFT_HIDDEN_CONT     = 16
TFT_BATCH_SIZE      = 64
TFT_MAX_EPOCHS      = 60
TFT_LR              = 1e-3
TFT_WEIGHT_DECAY    = 0.0    # [FIX-11] desactive : en v10 il dispersait les membres (sigma 0.3->1.6) sans gain

# =============================================================================
# HYPERPARAMETRES RFR
# =============================================================================
RFR_PARAMS = {
    "n_estimators":      800,
    "max_depth":         25,
    "min_samples_split": 20,
    "min_samples_leaf":  10,
    "max_features":      0.4,
}
RFR_MAX_SAMPLES       = 0.8
ENSEMBLE_VAL_FRACTION = 0.20

# Grille beta residuel (post-hoc, attendu = 0 si C1/C2/C3 fonctionnent)
BETA_GRID    = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
BETA_DEFAULT = 0.0

# =============================================================================
# SECTIONS
# =============================================================================
SECTION_TWS   = "Total_TWS"
SECTION_PR    = "Precipitations_API"
SECTION_NDVI  = "NDVI"
SECTION_ET    = "ET"
SECTION_SLOPE = "Slope"
SLOPE_ALIASES = ["Slope", "Pente", "SLOPE", "slope", "Slope_Tensift"]

# =============================================================================
# 0bis) FICHIER name_files_GRSL_TWS.txt — ADAPTATION TENSIFT
# =============================================================================
AUTO_CREATE_NAME_FILE = True
OVERWRITE_NAME_FILE = True

NAME_FILE_SECTIONS = {
    "Total_TWS": "TWS_GLDAS_Clipped",
    "Precipitations_API": "CHIRPS_GEE_Clipped",
    "NDVI": "MODIS_NDVI_GEE_Clipped",
    "ET": "MODIS_ET_GEE_Clipped",
    "Slope": "Slope_Tensift.tif",
}

def ensure_name_file():
    """Crée automatiquement name_files_GRSL_TWS.txt selon les chemins Tensift."""
    if not AUTO_CREATE_NAME_FILE:
        return

    if os.path.isfile(NAME_FILE) and not OVERWRITE_NAME_FILE:
        print(f"✅ name_files_GRSL_TWS.txt déjà existant : {NAME_FILE}")
        return

    print("\n====================================================")
    print("Création / mise à jour de name_files_GRSL_TWS.txt")
    print("====================================================")

    missing = []
    for section_name, rel_path in NAME_FILE_SECTIONS.items():
        full_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(full_path):
            kind = "dossier" if os.path.isdir(full_path) else "fichier"
            print(f"✅ [{section_name}] {kind} trouvé : {rel_path}")
        else:
            print(f"⚠️ [{section_name}] chemin introuvable : {rel_path}")
            missing.append((section_name, rel_path))

    with open(NAME_FILE, "w", encoding="utf-8") as f:
        for section_name, rel_path in NAME_FILE_SECTIONS.items():
            f.write(f"[{section_name}]\n")
            f.write(f"{rel_path}\n\n")

    print(f"\n✅ Fichier créé : {NAME_FILE}")

    if missing:
        print("\n⚠️ Chemins introuvables à vérifier avant l'exécution :")
        for section_name, rel_path in missing:
            print(f"   [{section_name}] -> {rel_path}")


# =============================================================================
# 1) UTILITAIRES IO / DATES
# =============================================================================
def parse_name_file(path_txt):
    sections = {}
    current = None
    with open(path_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip()
                sections[current] = []
            else:
                if current is None:
                    continue
                sections[current].append(line)
    return sections


def _norm_abs_from_entry(entry):
    entry = entry.strip().strip('"').strip("'")
    p = entry
    if not os.path.isabs(p):
        p = os.path.join(BASE_DIR, p)
    return os.path.normpath(p)


def _scan_tifs_in_dir(d):
    out = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        out.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))
    return sorted(list(set(out)))


def _expand_entry_to_files(entry):
    p = _norm_abs_from_entry(entry)
    if any(ch in p for ch in ["*", "?", "["]):
        files = glob.glob(p, recursive=True)
        return sorted(list({os.path.normpath(x) for x in files if os.path.isfile(x)}))
    if os.path.isdir(p):
        return _scan_tifs_in_dir(p)
    if os.path.isfile(p):
        return [p]
    return []


def extract_date_yyyymmdd_from_any(path_str):
    b = os.path.basename(path_str)
    m = re.search(r"(\d{8})", b)
    if not m:
        return None
    s8 = m.group(1)
    for fmt in ("%Y%m%d", "%d%m%Y"):
        try:
            d = datetime.strptime(s8, fmt).date()
            return d.strftime("%Y%m%d")
        except Exception:
            continue
    return None


def yyyymmdd_to_date(s):
    return datetime.strptime(s, "%Y%m%d").date()


def build_date_dict(sections, section_name):
    out = {}
    if section_name not in sections:
        print(f"   [WARN] section absente : {section_name}")
        return out
    for entry in sections[section_name]:
        for fp in _expand_entry_to_files(entry):
            s8 = extract_date_yyyymmdd_from_any(fp)
            if s8 is None:
                continue
            d = yyyymmdd_to_date(s8)
            if STUDY_START <= d <= STUDY_END:
                out[d] = fp
    return out


def get_static_path(sections, section_name, aliases=None):
    names = [section_name] + (aliases or [])
    sec_keys = {k.strip().lower(): k for k in sections.keys()}
    for nm in names:
        key = sec_keys.get(nm.strip().lower())
        if key is None:
            continue
        for entry in sections[key]:
            files = _expand_entry_to_files(entry)
            if files:
                return files[0]
    for nm in names:
        hits = glob.glob(os.path.join(BASE_DIR, "**", f"*{nm}*.tif"), recursive=True)
        hits += glob.glob(os.path.join(BASE_DIR, "**", f"*{nm}*.tiff"), recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


# =============================================================================
# 2) RASTER
# =============================================================================
def load_raster(path):
    with rasterio.open(path) as src:
        data    = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata  = src.nodata
        profile["_bounds"]  = tuple(src.bounds)
        profile["_crs_str"] = str(src.crs) if src.crs is not None else None
    if profile.get("crs") is None:
        profile["crs"] = CRS.from_epsg(4326)
    return data, profile, nodata


def to_float_with_nan(arr, nodata):
    a = arr.astype("float32")
    if nodata is not None:
        a = np.where(a == nodata, np.nan, a)
    return a


def reproject_to_profile(src_data, src_profile, src_nodata, dst_profile,
                          dst_nodata=np.nan, resampling=Resampling.bilinear):
    dst_data = np.full((dst_profile["height"], dst_profile["width"]),
                       dst_nodata, dtype="float32")
    src_prof = src_profile.copy()
    if src_prof.get("crs") is None:
        src_prof["crs"] = dst_profile.get("crs", CRS.from_epsg(4326))
    reproject(
        source=src_data, destination=dst_data,
        src_transform=src_prof["transform"], src_crs=src_prof["crs"],
        src_nodata=src_nodata,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile.get("crs", CRS.from_epsg(4326)),
        dst_nodata=dst_nodata, resampling=resampling,
    )
    return dst_data


def write_geotiff(path, arr2d, profile, nodata_value=OUT_NODATA, compress="DEFLATE"):
    p = {k: v for k, v in profile.items() if not str(k).startswith("_")}
    out_crs = p.get("crs", None) or CRS.from_epsg(4326)
    p.update(dtype="float32", count=1, nodata=nodata_value, compress=compress, crs=out_crs)
    out = arr2d.astype("float32")
    out = np.where(np.isfinite(out), out, nodata_value).astype("float32")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(out, 1)


# =============================================================================
# 3) NORMALISATION MIN-MAX
# =============================================================================
class GlobalMinMax:
    def __init__(self):
        self.vmin = {}
        self.vmax = {}

    def fit(self, name, values):
        v = values[np.isfinite(values)]
        if v.size == 0:
            self.vmin[name], self.vmax[name] = 0.0, 1.0
        else:
            self.vmin[name] = float(np.min(v))
            self.vmax[name] = float(np.max(v))
        return self

    def transform(self, name, values):
        vmin, vmax = self.vmin[name], self.vmax[name]
        denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
        return (values - vmin) / denom


# =============================================================================
# 4) GRANGER MULTI-LAG PONDERE AIC [B2]
# =============================================================================
def _granger_ols_per_lag(cause, effect, maxlag):
    def ols_ssr(y, X):
        X2 = np.column_stack([np.ones(len(y)), X])
        beta, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
        resid = y - X2 @ beta
        return float(np.sum(resid ** 2)), len(y), X2.shape[1]

    results = {}
    for lag in range(1, maxlag + 1):
        if len(effect) <= lag + 2:
            break
        y  = effect[lag:]
        Xy = np.column_stack([effect[lag - i - 1: len(effect) - i - 1] for i in range(lag)])
        Xc = np.column_stack([cause [lag - i - 1: len(cause)  - i - 1] for i in range(lag)])
        ssr_r, n_r, _   = ols_ssr(y, Xy)
        ssr_u, n_u, k_u = ols_ssr(y, np.column_stack([Xy, Xc]))
        aic_u = n_u * np.log(max(ssr_u / n_u, 1e-30)) + 2 * k_u if n_u > 0 else np.inf
        var_r = ssr_r / n_r
        var_u = ssr_u / n_u
        coef  = float(max(0.0, np.log(max(var_r, 1e-12) / max(var_u, 1e-12)))) if var_u > 0 else 0.0
        results[lag] = (coef, aic_u)
    return results


def granger_coefficient_multilag(cause, effect, maxlag=GRANGER_MAXLAG):
    cause  = np.asarray(cause,  dtype=float)
    effect = np.asarray(effect, dtype=float)
    m = np.isfinite(cause) & np.isfinite(effect)
    cause, effect = cause[m], effect[m]
    if cause.size < (maxlag + 5):
        return 0.0

    lag_results = {}
    if _HAS_STATSMODELS:
        try:
            data = np.column_stack([effect, cause])
            res  = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            for lag in range(1, maxlag + 1):
                ssr_r = res[lag][1][0].ssr
                ssr_u = res[lag][1][1].ssr
                aic_u = res[lag][1][1].aic
                n = data.shape[0]
                var_r = ssr_r / n
                var_u = ssr_u / n
                coef  = float(max(0.0, np.log(max(var_r, 1e-12) / max(var_u, 1e-12)))) if var_u > 0 else 0.0
                lag_results[lag] = (coef, aic_u)
        except Exception:
            lag_results = {}

    if not lag_results:
        lag_results = _granger_ols_per_lag(cause, effect, maxlag)
    if not lag_results:
        return 0.0

    lags  = sorted(lag_results.keys())
    aics  = np.array([lag_results[l][1] for l in lags], dtype=float)
    coefs = np.array([lag_results[l][0] for l in lags], dtype=float)
    aics  = np.nan_to_num(aics, nan=1e9, posinf=1e9)
    aics_s = aics - np.min(aics)
    w = np.exp(-aics_s)
    w = w / (w.sum() + 1e-30)
    return float(np.dot(w, coefs))


def build_granger_matrix_multilag(series_dict, var_order=("PR", "NDVI", "ET")):
    n = len(var_order)
    L = np.zeros((n, n), dtype=float)
    for i, vi in enumerate(var_order):
        for j, vj in enumerate(var_order):
            if i == j:
                continue
            L[i, j] = granger_coefficient_multilag(
                series_dict[vi], series_dict[vj], maxlag=GRANGER_MAXLAG)
    return L


# =============================================================================
# 5) ATTENTION CAUSALE + SPATIALE [A3][B3]
# =============================================================================
def softmax_rows(Q):
    Q = Q - np.nanmax(Q, axis=1, keepdims=True)
    e = np.exp(Q)
    s = np.sum(e, axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return e / s


def minmax_norm_vec(v):
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return v
    vmin, vmax = np.min(finite), np.max(finite)
    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    return (v - vmin) / denom


def compute_attention_map(V, L):
    L    = np.nan_to_num(np.asarray(L, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Q    = V @ L.T
    beta = softmax_rows(Q)
    w    = np.sum(beta * V, axis=1)
    Att  = minmax_norm_vec(w)
    return np.where(np.isfinite(Att), Att, 0.0)


def compute_attention_map_spatial(V, L_dict, labels):
    """[A3] Attention spatiale avec normalisation intra-zone."""
    V      = np.asarray(V, dtype=float)
    labels = np.asarray(labels)
    weighted = np.zeros(V.shape[0], dtype=float)

    for k, L in L_dict.items():
        mask = (labels == k)
        if not mask.any():
            continue
        Lk   = np.nan_to_num(np.asarray(L, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        Vk   = V[mask]
        Qk   = Vk @ Lk.T
        bk   = softmax_rows(Qk)
        w_k  = np.sum(bk * Vk, axis=1)
        w_k  = minmax_norm_vec(w_k)   # [A3]
        weighted[mask] = w_k

    Att = minmax_norm_vec(weighted)
    return np.where(np.isfinite(Att), Att, 0.0)


def granger_prior_vector(L):
    """
    [C2] Prior causal : influence recue par chaque variable.
    p[i] = sum_j L[j,i]  (somme des influences entrantes vers i)
    Normalise en probabilite pour biaser var_select.
    """
    L = np.asarray(L, dtype=float)
    influence_in = L.sum(axis=0)              # (n_vars,) influence recue
    s = influence_in.sum()
    if s <= 0:
        return np.ones(L.shape[0]) / L.shape[0]
    return influence_in / s


# =============================================================================
# 5bis) CLUSTERING [B5]
# =============================================================================
def build_pixel_clusters(api_maps, ndvi_maps, et_maps, fine_profile, dates,
                          k_range=None, seed=CLUSTER_SEED, slope_map=None):
    if k_range is None:
        k_range = K_RANGE
    Hf, Wf = fine_profile["height"], fine_profile["width"]
    N = Hf * Wf

    def stack_series(maps):
        cols = [maps[d].ravel() for d in dates if d in maps]
        return np.stack(cols, axis=1) if cols else np.zeros((N, 1))

    pr_s = stack_series(api_maps)
    nd_s = stack_series(ndvi_maps)
    et_s = stack_series(et_maps)

    with np.errstate(invalid="ignore"):
        cols = [
            np.nanmean(pr_s, axis=1), np.nanmean(nd_s, axis=1), np.nanmean(et_s, axis=1),
            np.nanstd(pr_s,  axis=1), np.nanstd(nd_s,  axis=1), np.nanstd(et_s,  axis=1),
        ]
        if slope_map is not None:
            cols.append(slope_map.ravel())
        feats = np.column_stack(cols)

    valid  = np.isfinite(feats).all(axis=1)
    labels = np.full(N, -1, dtype=int)

    if valid.sum() < max(k_range) + 1:
        print(f"   [WARN] trop peu de pixels valides ({valid.sum()}) -> zone globale (-1).")
        return labels.reshape(Hf, Wf), None, None, k_range[0]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(feats[valid])

    print(f"   [B5] Selection K optimal (silhouette) sur {k_range}...")
    best_K, best_sil = k_range[0], -1.0
    for K in k_range:
        if valid.sum() < K + 1:
            continue
        km_try  = KMeans(n_clusters=K, random_state=seed, n_init=10)
        lab_try = km_try.fit_predict(Xs)
        try:
            sil = silhouette_score(Xs, lab_try,
                                   sample_size=min(5000, Xs.shape[0]),
                                   random_state=seed)
        except Exception:
            sil = -1.0
        print(f"      K={K} -> silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil, best_K = sil, K

    print(f"   K* optimal = {best_K} (silhouette = {best_sil:.4f})")
    kmeans = KMeans(n_clusters=best_K, random_state=seed, n_init=10)
    labels[valid] = kmeans.fit_predict(Xs)

    uniq, counts = np.unique(labels[labels >= 0], return_counts=True)
    print("   Repartition pixels par zone :",
          {int(u): int(cnt) for u, cnt in zip(uniq, counts)})
    return labels.reshape(Hf, Wf), kmeans, scaler, best_K


def build_granger_matrices_per_cluster(labels_2d, api_maps, ndvi_maps, et_maps,
                                        dates, L_global,
                                        min_pixels=MIN_PIXELS_PER_CLUSTER):
    labels_flat = labels_2d.ravel()
    L_dict  = {-1: L_global}
    clusters = sorted(set(int(k) for k in np.unique(labels_flat) if k >= 0))

    for k in clusters:
        mask = (labels_flat == k)
        if mask.sum() < min_pixels:
            L_dict[k] = L_global
            print(f"   Zone {k}: {mask.sum()} pixels < {min_pixels} -> fallback global.")
            continue

        pr_ser, nd_ser, et_ser = [], [], []
        for d in dates:
            if d not in api_maps or d not in ndvi_maps or d not in et_maps:
                continue
            pr_ser.append(float(np.nanmean(api_maps[d].ravel()[mask])))
            nd_ser.append(float(np.nanmean(ndvi_maps[d].ravel()[mask])))
            et_ser.append(float(np.nanmean(et_maps[d].ravel()[mask])))

        series_k = {"PR": np.array(pr_ser), "NDVI": np.array(nd_ser),
                    "ET": np.array(et_ser)}
        Lk = build_granger_matrix_multilag(series_k, ("PR", "NDVI", "ET"))
        if not np.isfinite(Lk).all() or np.allclose(Lk, 0.0):
            Lk = L_global
            print(f"   Zone {k}: Granger degenere -> fallback global.")
        L_dict[k] = Lk

    return L_dict


# =============================================================================
# 6) DATASET [B4]
# =============================================================================
def compute_api_series(pr_dict, fine_profile, dates_sorted, slope_map=None):
    """[B4] decay(p) = API_DECAY_J * exp(-API_SLOPE_K * slope_norm(p))"""
    api_maps  = {}
    prev      = None
    decay_map = None

    if slope_map is not None and RECOMPUTE_API_FROM_RAW:
        sl        = slope_map.copy()
        sl_finite = sl[np.isfinite(sl)]
        if sl_finite.size > 0:
            sl_n = (sl - sl_finite.min()) / max(sl_finite.max() - sl_finite.min(), 1e-9)
        else:
            sl_n = np.zeros_like(sl)
        decay_map = API_DECAY_J * np.exp(-API_SLOPE_K * sl_n)
        decay_map = np.where(np.isfinite(decay_map), decay_map, API_DECAY_J)
        print(f"   [B4] Decay API spatial : min={decay_map[np.isfinite(decay_map)].min():.3f} "
              f"max={decay_map[np.isfinite(decay_map)].max():.3f}")

    for d in dates_sorted:
        pr, prof, nod = load_raster(pr_dict[d])
        pr   = to_float_with_nan(pr, nod)
        pr_f = reproject_to_profile(pr, prof, nod, fine_profile,
                                    dst_nodata=np.nan, resampling=Resampling.bilinear)
        if not RECOMPUTE_API_FROM_RAW:
            api_maps[d] = pr_f
            continue
        if prev is None:
            api = pr_f.copy()
        else:
            api = pr_f + (decay_map if decay_map is not None else API_DECAY_J) * prev
        api = np.where(np.isfinite(api), api, pr_f)
        api_maps[d] = api
        prev = np.where(np.isfinite(api), api, 0.0)
    return api_maps


def spatial_fill_nearest(arr2d, domain_mask=None):
    """
    [GAP-FILL spatial] Comble les NaN par le plus proche voisin valide
    (distance euclidienne sur grille). Justification : pour un pixel sans
    observation (nuage / ombre / saturation), la meilleure estimation, a defaut
    de serie temporelle, est la valeur biophysique du pixel valide le plus
    proche (continuite spatiale des regimes vegetation / hydrologie).
    Si domain_mask est fourni, seuls les pixels DANS le domaine sont combles ;
    le hors-domaine reste NaN (no-data legitime, ex. hors-bassin).
    """
    from scipy.ndimage import distance_transform_edt
    out = arr2d.astype(np.float32).copy()
    nan_mask = ~np.isfinite(out)
    if domain_mask is not None:
        nan_mask &= domain_mask          # ne combler qu'a l'interieur du domaine
    if not nan_mask.any():
        return out
    valid = np.isfinite(out)
    if not valid.any():
        return out                       # rien de valide -> rien a propager
    # indices du pixel valide le plus proche pour chaque cellule
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    out[nan_mask] = out[iy[nan_mask], ix[nan_mask]]
    return out


def erode_mask(mask2d, n_pixels=1):
    """[EDGE-FIX] Erode un masque booleen de n_pixels (retire l'anneau exterieur).

    Implementation sans scipy.ndimage.binary_erosion stricte : on utilise la
    transformee de distance euclidienne pour ne garder que les pixels situes a
    plus de n_pixels du bord (hors-masque). Robuste et sans dependance externe
    supplementaire. Renvoie le masque inchange si n_pixels <= 0.
    """
    if mask2d is None or n_pixels is None or n_pixels <= 0:
        return mask2d
    from scipy.ndimage import distance_transform_edt
    m = np.asarray(mask2d, dtype=bool)
    if not m.any():
        return m
    # distance de chaque pixel INTERNE au pixel hors-masque le plus proche
    dist = distance_transform_edt(m)
    # on conserve les pixels suffisamment a l'interieur (anneau exterieur retire)
    return dist > float(n_pixels)


def build_clip_mask_from_shapefile(shp_path, fine_profile):
    """[CLIP] Masque booleen (H, W) du bassin Tensift sur la grille fine.

    True  = pixel A L'INTERIEUR du polygone du bassin (a conserver).
    False = pixel HORS bassin (-> NoData a l'export).

    Le shapefile est reprojete a la volee vers le CRS de la grille si besoin.
    Renvoie None (clip ignore) si le fichier est absent, fiona indisponible,
    geometrie vide ou masque vide — avec un avertissement explicite.
    """
    if not _HAS_FIONA:
        print("[CLIP][WARN] fiona indisponible — clip shapefile ignore. "
              "pip install fiona")
        return None
    if not os.path.isfile(shp_path):
        print(f"[CLIP][WARN] Shapefile introuvable : {shp_path} — clip ignore.")
        return None

    Hf = fine_profile["height"]
    Wf = fine_profile["width"]
    transform = fine_profile["transform"]
    dst_crs = fine_profile.get("crs", None) or CRS.from_epsg(4326)

    try:
        with fiona.open(shp_path, "r") as src:
            src_crs = src.crs
            geoms = [feat["geometry"] for feat in src
                     if feat.get("geometry") is not None]
    except Exception as e:
        print(f"[CLIP][WARN] Lecture shapefile echouee ({e}) — clip ignore.")
        return None

    if not geoms:
        print("[CLIP][WARN] Aucune geometrie dans le shapefile — clip ignore.")
        return None

    # Reprojection des geometries vers le CRS du raster si necessaire.
    try:
        if src_crs and CRS.from_user_input(src_crs) != CRS.from_user_input(dst_crs):
            geoms = [transform_geom(src_crs, dst_crs, g) for g in geoms]
            print(f"[CLIP] Shapefile reprojete {CRS.from_user_input(src_crs).to_string()} "
                  f"-> {CRS.from_user_input(dst_crs).to_string()}")
    except Exception as e:
        print(f"[CLIP][WARN] Reprojection geometries echouee ({e}) — "
              "tentative sans reprojection.")

    try:
        # geometry_mask : par defaut True = HORS geometrie. On inverse.
        outside = geometry_mask(geoms, out_shape=(Hf, Wf),
                                transform=transform, invert=False,
                                all_touched=True)
        inside = ~outside
    except Exception as e:
        print(f"[CLIP][WARN] Rasterisation du masque echouee ({e}) — clip ignore.")
        return None

    n_in = int(inside.sum())
    if n_in == 0:
        print("[CLIP][WARN] Masque bassin vide (0 pixel) — clip ignore. "
              "Verifiez le CRS du shapefile et de la grille.")
        return None

    print(f"[CLIP] Masque bassin Tensift : {n_in} pixels intra-bassin "
          f"({100.0 * n_in / (Hf * Wf):.1f}% de la grille).")
    return inside


def load_var_on_fine(path, fine_profile, resampling=Resampling.bilinear,
                     fill_before_resample=True, domain_mask_coarse=None):
    """
    Charge une variable sur la grille fine.
    [GAP-FILL] Si fill_before_resample : on comble spatialement les trous SUR LA
    GRILLE GROSSIERE (resolution native de l'observation) AVANT reechantillonnage.
    Cela evite que le bilineaire propage / elargisse les NaN (un NaN grossier
    contaminerait un bloc de pixels fins), tout en gardant chaque valeur ancree
    sur l'observation native la plus proche.
    """
    arr, prof, nod = load_raster(path)
    arr = to_float_with_nan(arr, nod)
    if fill_before_resample:
        if arr.ndim == 3:
            arr = np.stack([spatial_fill_nearest(arr[b], domain_mask_coarse)
                            for b in range(arr.shape[0])], axis=0)
        else:
            arr = spatial_fill_nearest(arr, domain_mask_coarse)
    return reproject_to_profile(arr, prof, nod, fine_profile,
                                dst_nodata=np.nan, resampling=resampling)


def get_fine_profile_from_pr(pr_dict):
    any_date = sorted(pr_dict.keys())[0]
    _, prof, _ = load_raster(pr_dict[any_date])
    prof = prof.copy()
    prof["crs"] = prof.get("crs", CRS.from_epsg(4326))
    return prof


def build_interp_series(var_dict, fine_profile, target_dates,
                        domain_mask=None, fill_residual=True):
    """
    Construit la serie complete d'une variable sur les dates cibles.

    Cascade de comblement, du plus rigoureux (specifique au pixel) au plus
    generique, pour qu'AUCUN pixel exporte ne reste no-data par defaut tout en
    restant une estimation defendable :

      1. Interpolation TEMPORELLE lineaire entre composites encadrants (existant).
      2. Comblement TEMPOREL par pixel : pour chaque pixel encore NaN a une date,
         on prend la valeur observee la plus proche dans le temps SUR CE MEME
         pixel (forward / backward fill). C'est l'estimation la plus locale et la
         plus defendable (la grandeur biophysique d'un pixel varie lentement).
      3. Comblement SPATIAL des residus : pixels NaN a TOUTES les dates (nuage
         persistant, ombre permanente) combles par plus proche voisin valide.
      4. CLIMATOLOGIE : ultime filet de securite (moyenne temporelle du pixel,
         puis moyenne globale) — ne devrait quasiment jamais servir.
    """
    comp_dates  = sorted(var_dict.keys())
    if not comp_dates:
        return {}
    comp_arrays = {d: load_var_on_fine(var_dict[d], fine_profile) for d in comp_dates}
    out = {}
    # --- Etape 1 : interpolation temporelle lineaire (inchangee) ---
    for d in target_dates:
        if d in comp_arrays:
            out[d] = comp_arrays[d]
            continue
        prev_dates = [c for c in comp_dates if c <= d]
        nxt_dates  = [c for c in comp_dates if c >= d]
        if prev_dates and nxt_dates:
            d0, d1 = prev_dates[-1], nxt_dates[0]
            if d0 == d1:
                out[d] = comp_arrays[d0]
            else:
                w = (d - d0).days / float((d1 - d0).days)
                out[d] = (1.0 - w) * comp_arrays[d0] + w * comp_arrays[d1]
        elif prev_dates:
            out[d] = comp_arrays[prev_dates[-1]]
        elif nxt_dates:
            out[d] = comp_arrays[nxt_dates[0]]

    if not fill_residual or not out:
        return out

    # Empile en (T, H, W) pour traiter pixel par pixel
    sdates = sorted(out.keys())
    cube   = np.stack([out[d] for d in sdates], axis=0)   # (T, H, W)
    T, H, W = cube.shape

    # --- Etape 2 : forward/backward fill TEMPOREL par pixel ---
    flat = cube.reshape(T, H * W)
    valid = np.isfinite(flat)
    # forward fill
    idx = np.where(valid, np.arange(T)[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    for t in range(T):
        take = idx[t]
        m = take >= 0
        flat[t, m & ~valid[t]] = flat[take[m & ~valid[t]], np.where(m & ~valid[t])[0]]
    # backward fill (pour les NaN en debut de serie)
    valid = np.isfinite(flat)
    idxb = np.where(valid, np.arange(T)[:, None], T)
    idxb = np.minimum.accumulate(idxb[::-1], axis=0)[::-1]
    for t in range(T):
        take = idxb[t]
        m = take < T
        flat[t, m & ~valid[t]] = flat[take[m & ~valid[t]], np.where(m & ~valid[t])[0]]
    cube = flat.reshape(T, H, W)

    # --- Etape 3 : comblement SPATIAL des pixels NaN a toutes les dates ---
    still_nan = ~np.isfinite(cube).any(axis=0)           # (H, W) : NaN partout
    if domain_mask is not None:
        still_nan &= domain_mask
    if still_nan.any():
        for t in range(T):
            cube[t] = spatial_fill_nearest(cube[t], domain_mask=domain_mask)

    # --- Etape 4 : climatologie (filet ultime) ---
    if not np.isfinite(cube).all():
        clim = np.nanmean(cube, axis=0)                  # moyenne temporelle/pixel
        gmean = float(np.nanmean(clim)) if np.isfinite(clim).any() else 0.0
        clim = np.where(np.isfinite(clim), clim, gmean)
        for t in range(T):
            m = ~np.isfinite(cube[t])
            if domain_mask is not None:
                m &= domain_mask
            cube[t][m] = clim[m]

    # Si domain_mask fourni : on RETABLIT le no-data hors-domaine (legitime)
    if domain_mask is not None:
        outside = ~domain_mask
        for t in range(T):
            cube[t][outside] = np.nan

    for i, d in enumerate(sdates):
        out[d] = cube[i]
    return out


# =============================================================================
# 7) ASSEMBLAGE ECHANTILLONS
# =============================================================================
def assemble_samples(dates, api_maps, ndvi_maps, et_maps, tws_dict,
                     fine_profile, normalizer=None, fit_normalizer=False,
                     slope_map=None):
    Xs, ys, metas = [], [], []
    slope_flat = slope_map.ravel() if slope_map is not None else None

    for d in dates:
        if d not in api_maps or d not in tws_dict:
            continue
        if d not in ndvi_maps or d not in et_maps:
            continue
        pr   = api_maps[d]
        ndvi = ndvi_maps[d]
        et   = et_maps[d]
        tws_c, gprof, gnod = load_raster(tws_dict[d])
        tws_c = to_float_with_nan(tws_c, gnod)
        tws_f = reproject_to_profile(tws_c, gprof, gnod, fine_profile,
                                     dst_nodata=np.nan, resampling=Resampling.bilinear)
        if slope_flat is not None:
            stack = np.stack([pr.ravel(), ndvi.ravel(), et.ravel(),
                              slope_flat, tws_f.ravel()], axis=1)
        else:
            stack = np.stack([pr.ravel(), ndvi.ravel(), et.ravel(),
                              tws_f.ravel()], axis=1)
        finite = np.isfinite(stack).all(axis=1)
        if not finite.any():
            continue
        valid  = stack[finite]
        n_feat = 4 if slope_flat is not None else 3
        Xs.append(valid[:, :n_feat])
        ys.append(valid[:, n_feat])
        metas.append((d, np.flatnonzero(finite)))

    if not Xs:
        raise RuntimeError("Aucun echantillon valide assemble.")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    if normalizer is None:
        normalizer = GlobalMinMax()
    if fit_normalizer:
        normalizer.fit("PR",   X[:, 0])
        normalizer.fit("NDVI", X[:, 1])
        normalizer.fit("ET",   X[:, 2])
        if slope_flat is not None:
            normalizer.fit("SLOPE", X[:, 3])

    pr_n   = normalizer.transform("PR",   X[:, 0])
    ndvi_n = normalizer.transform("NDVI", X[:, 1])
    et_n   = normalizer.transform("ET",   X[:, 2])
    V_norm = np.stack([pr_n, ndvi_n, 1.0 - et_n], axis=1)
    return X, y, V_norm, normalizer, metas


# =============================================================================
# 8) DATASET PYTORCH [C3]
# =============================================================================
class TWSSequenceDataset(Dataset):
    """
    [C3] X_seq : (N, seq_len, n_feat) — n_feat=3 ou 4 si Att inclus
    slope_context : (N, 1) — contexte Slope pour le GRN final [A2]
    """
    def __init__(self, X_seq, y_seq, slope_context=None):
        self.X   = torch.tensor(X_seq,  dtype=torch.float32)
        self.y   = torch.tensor(y_seq,  dtype=torch.float32)
        self.ctx = (torch.tensor(slope_context, dtype=torch.float32)
                    if slope_context is not None else None)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        ctx = self.ctx[idx] if self.ctx is not None else torch.zeros(1)
        return self.X[idx], self.y[idx], ctx


# =============================================================================
# 9) ARCHITECTURE TFT CAUSAL-GUIDED [C1][C2]
# =============================================================================
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.fc1     = nn.Linear(input_size, hidden_size)
        self.fc2     = nn.Linear(hidden_size, output_size)
        self.gate    = nn.Linear(hidden_size, output_size)
        self.skip    = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.norm    = nn.LayerNorm(output_size)
        self.drop    = nn.Dropout(dropout)
        self.elu     = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h    = self.elu(self.fc1(x))
        h    = self.drop(h)
        out  = self.fc2(h)
        gate = self.sigmoid(self.gate(h))
        return self.norm(self.skip(x) + gate * out)


class CausalTFT(nn.Module):
    """
    TFT Causal-Guided v4 :

    [C2] Prior de Granger dans la variable selection :
         logits_VS += gamma * granger_prior  (biaise vers les variables causales)

    [C1] La loss inclut une regularisation KL(var_weights || p_granger).
         Le forward retourne aussi les poids de selection pour le calcul de loss.

    [A2] Slope comme contexte statique du GRN final.

    [C3] Si n_feat=4, le 4e canal est Att_Granger (feature causale temporelle).
    """
    def __init__(self, input_size=3, context_size=1,
                 seq_len=TFT_SEQ_LEN, hidden_size=TFT_HIDDEN_SIZE,
                 n_heads=TFT_ATTENTION_HEADS, dropout=TFT_DROPOUT,
                 hidden_cont=TFT_HIDDEN_CONT,
                 granger_prior=None, gamma=GAMMA_GRANGER_PRIOR):
        super().__init__()
        self.seq_len      = seq_len
        self.hidden_size  = hidden_size
        self.context_size = context_size
        self.input_size   = input_size
        self.gamma        = gamma

        # [C2] Prior causal (vecteur de taille input_size, normalise)
        if granger_prior is not None:
            gp = np.asarray(granger_prior, dtype=np.float32)
            # Adapter si input_size > len(granger_prior) (cas avec Att comme 4e canal)
            if len(gp) < input_size:
                gp = np.append(gp, [0.0] * (input_size - len(gp)))
            gp = gp[:input_size]
            gp = gp / (gp.sum() + 1e-8)
            self.register_buffer("granger_prior",
                                 torch.tensor(gp, dtype=torch.float32))
        else:
            self.register_buffer("granger_prior",
                                 torch.ones(input_size, dtype=torch.float32) / input_size)

        # Variable Selection : un GRN par variable d'entree
        self.var_grns   = nn.ModuleList([
            GatedResidualNetwork(1, hidden_cont, hidden_cont, dropout)
            for _ in range(input_size)
        ])
        self.var_select = nn.Linear(input_size * hidden_cont, input_size)
        self.softmax    = nn.Softmax(dim=-1)
        self.input_proj = nn.Linear(input_size * hidden_cont + hidden_cont, hidden_size)

        # LSTM encoder
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True,
                            num_layers=2, dropout=dropout)

        # Multi-Head Self-Attention
        self.attn      = nn.MultiheadAttention(hidden_size, n_heads,
                                               dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size)

        # GRN final : h_last + contexte Slope [A2]
        grn_input_size = hidden_size + context_size
        self.grn_out   = GatedResidualNetwork(grn_input_size, hidden_size, hidden_cont, dropout)
        self.fc_out    = nn.Linear(hidden_cont, 1)

    def forward(self, x, context=None):
        """
        x       : (B, seq_len, input_size)
        context : (B, context_size) — Slope
        Retourne (pred, var_weights_mean) pour calcul loss [C1]
        """
        B, T, F = x.shape

        # Variable Selection
        var_embs = [grn(x[:, :, i:i+1]) for i, grn in enumerate(self.var_grns)]
        var_cat  = torch.cat(var_embs, dim=-1)                        # (B,T,F*hc)

        # [C2] Prior causal injecte dans les logits de selection
        logits   = self.var_select(var_cat)                           # (B,T,F)
        prior    = self.granger_prior.unsqueeze(0).unsqueeze(0)       # (1,1,F)
        logits   = logits + self.gamma * prior
        weights  = self.softmax(logits)                               # (B,T,F)

        # Moyenne spatiale des poids pour la regularisation KL [C1]
        var_weights_mean = weights.mean(dim=(0, 1))                   # (F,)

        weighted = sum(weights[:, :, i:i+1] * var_embs[i] for i in range(F))

        # Projection + LSTM
        h = self.input_proj(torch.cat([var_cat, weighted], dim=-1))
        h, _ = self.lstm(h)

        # Multi-Head Self-Attention
        attn_out, _ = self.attn(h, h, h)
        h = self.attn_norm(h + attn_out)

        # Derniere etape + contexte Slope [A2]
        h_last = h[:, -1, :]
        if context is not None and self.context_size > 0:
            h_last = torch.cat([h_last, context], dim=-1)
        else:
            zeros  = torch.zeros(B, self.context_size, device=h_last.device)
            h_last = torch.cat([h_last, zeros], dim=-1)

        out = self.grn_out(h_last)
        return self.fc_out(out).squeeze(-1), var_weights_mean


def causal_kl_loss(var_weights_mean, granger_prior_tensor):
    """
    [C1] Divergence KL entre les poids de selection observes et le prior de Granger.
    KL(p_observed || p_granger) = sum_i p_obs[i] * log(p_obs[i] / p_granger[i])
    Penalise si le TFT ignore des variables causalement importantes.
    """
    p_obs  = var_weights_mean + 1e-8
    p_obs  = p_obs / p_obs.sum()
    p_ref  = granger_prior_tensor + 1e-8
    p_ref  = p_ref / p_ref.sum()
    kl     = (p_obs * torch.log(p_obs / p_ref)).sum()
    return kl


# =============================================================================
# 10) SEQUENCES TEMPORELLES [C3]
# =============================================================================
def build_sequences(dates_list, api_maps, ndvi_maps, et_maps, tws_dict,
                    fine_profile, normalizer, seq_len=TFT_SEQ_LEN,
                    slope_mean=None, L_granger=None, labels_2d=None,
                    L_dict=None):
    """
    [C3] Si L_granger est fourni (et USE_ATT_AS_FEATURE=True), l'attention
    causale de Granger est calculee pour chaque date et ajoutee comme 4e canal.
    Sequences : [Pr, NDVI, 1-ET] ou [Pr, NDVI, 1-ET, Att_Granger] selon C3.
    """
    pr_means, nd_means, et_means, tws_means = [], [], [], []
    att_means = []   # [C3]
    valid_dates = []

    Hf = fine_profile["height"]
    Wf = fine_profile["width"]

    for d in dates_list:
        if d not in api_maps or d not in ndvi_maps or d not in et_maps:
            continue
        if d not in tws_dict:
            continue
        tws_c, gprof, gnod = load_raster(tws_dict[d])
        tws_c = to_float_with_nan(tws_c, gnod)
        tws_f = reproject_to_profile(tws_c, gprof, gnod, fine_profile,
                                     dst_nodata=np.nan, resampling=Resampling.bilinear)
        tws_m = float(np.nanmean(tws_f))
        if not np.isfinite(tws_m):
            continue

        pr_m  = float(np.nanmean(api_maps[d]))
        nd_m  = float(np.nanmean(ndvi_maps[d]))
        et_m  = float(np.nanmean(et_maps[d]))

        # [C3] Attention moyenne du bassin a la date d
        if USE_ATT_AS_FEATURE and L_granger is not None:
            pr_n  = float(normalizer.transform("PR",   np.array([pr_m]))[0])
            nd_n  = float(normalizer.transform("NDVI", np.array([nd_m]))[0])
            et_n  = 1.0 - float(normalizer.transform("ET", np.array([et_m]))[0])
            V_d   = np.array([[pr_n, nd_n, et_n]])
            if L_dict is not None and labels_2d is not None:
                # Attention spatiale moyennee sur le bassin
                all_labels = labels_2d.ravel()
                valid_pix  = np.isfinite(api_maps[d].ravel())
                if valid_pix.any():
                    V_pix  = np.tile(V_d, (valid_pix.sum(), 1))
                    lab_pix = all_labels[valid_pix]
                    att_map = compute_attention_map_spatial(V_pix, L_dict, lab_pix)
                    att_m   = float(np.mean(att_map))
                else:
                    att_m = float(compute_attention_map(V_d, L_granger)[0])
            else:
                att_m = float(compute_attention_map(V_d, L_granger)[0])
        else:
            att_m = 0.0

        pr_means.append(pr_m)
        nd_means.append(nd_m)
        et_means.append(et_m)
        tws_means.append(tws_m)
        att_means.append(att_m)
        valid_dates.append(d)

    if len(valid_dates) <= seq_len:
        raise RuntimeError(f"Pas assez de dates ({len(valid_dates)}) pour seq={seq_len}.")

    pr_arr  = normalizer.transform("PR",   np.array(pr_means))
    nd_arr  = normalizer.transform("NDVI", np.array(nd_means))
    et_arr  = 1.0 - normalizer.transform("ET", np.array(et_means))
    att_arr = np.array(att_means, dtype=np.float32)   # deja dans [0,1]

    slope_n = 0.0
    if slope_mean is not None and "SLOPE" in normalizer.vmin:
        slope_n = float(normalizer.transform("SLOPE", np.array([slope_mean]))[0])

    n_feat = 4 if (USE_ATT_AS_FEATURE and L_granger is not None) else 3

    X_seq, y_seq, seq_dates = [], [], []
    for i in range(seq_len, len(valid_dates)):
        chans = [
            pr_arr[i - seq_len: i],
            nd_arr[i - seq_len: i],
            et_arr[i - seq_len: i],
        ]
        if n_feat == 4:
            chans.append(att_arr[i - seq_len: i])   # [C3] canal Att_Granger
        feat_seq = np.stack(chans, axis=1).astype(np.float32)
        X_seq.append(feat_seq)
        y_seq.append(float(tws_means[i]))
        seq_dates.append(valid_dates[i])

    X_seq  = np.array(X_seq)
    y_seq  = np.array(y_seq, dtype=np.float32)
    slope_context = np.full((len(X_seq), 1), slope_n, dtype=np.float32)
    return X_seq, y_seq, seq_dates, slope_context, n_feat


# =============================================================================
# 10b) ENTRAINEMENT TFT AVEC REGULARISATION CAUSALE [C1]
# =============================================================================
def train_tft(X_tr_seq, y_tr_seq, slope_context_tr,
              granger_prior_vec, lambda_granger=LAMBDA_GRANGER_DEFAULT,
              seed=None):
    """
    [C1] Loss = Huber(pred, y) + lambda * KL(var_weights || p_granger)
    [C2] granger_prior_vec biaise la variable selection dans CausalTFT
    [FIX-8] seed : initialise torch/numpy pour des membres d'ensemble distincts.
    """
    if not _HAS_TFT:
        raise ImportError("PyTorch non disponible.")

    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    y_min  = float(np.min(y_tr_seq))
    y_max  = float(np.max(y_tr_seq))
    denom  = (y_max - y_min) if (y_max - y_min) > 0 else 1.0
    y_norm = (y_tr_seq - y_min) / denom

    n_feat = int(X_tr_seq.shape[2])

    dataset = TWSSequenceDataset(X_tr_seq, y_norm, slope_context=slope_context_tr)
    loader  = DataLoader(dataset, batch_size=TFT_BATCH_SIZE, shuffle=True, num_workers=0)

    context_size = slope_context_tr.shape[1] if slope_context_tr is not None else 1

    model = CausalTFT(
        input_size=n_feat, context_size=context_size,
        seq_len=TFT_SEQ_LEN, hidden_size=TFT_HIDDEN_SIZE,
        n_heads=TFT_ATTENTION_HEADS, dropout=TFT_DROPOUT,
        hidden_cont=TFT_HIDDEN_CONT,
        granger_prior=granger_prior_vec,
        gamma=GAMMA_GRANGER_PRIOR,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=TFT_LR,
                                 weight_decay=TFT_WEIGHT_DECAY)   # [FIX-10]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"   TFT sur {device} | {TFT_MAX_EPOCHS} ep | {n_feat} canaux | "
          f"lambda={lambda_granger:.4f} | gamma={GAMMA_GRANGER_PRIOR}")

    model.train()
    for epoch in range(TFT_MAX_EPOCHS):
        epoch_loss = 0.0
        for xb, yb, cb in loader:
            xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
            optimizer.zero_grad()
            pred, var_w = model(xb, context=cb)
            # [C1] Loss = prediction + regularisation causale KL
            loss_pred = criterion(pred, yb)
            loss_kl   = causal_kl_loss(var_w, model.granger_prior)
            loss      = loss_pred + lambda_granger * loss_kl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss_pred.item() * len(yb)   # on trace uniquement la pred loss
        avg_loss = epoch_loss / len(dataset)
        scheduler.step(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1:3d}/{TFT_MAX_EPOCHS} | Loss_pred={avg_loss:.4f}")

    model.eval()
    tws_scaler = {"min": y_min, "max": y_max, "denom": denom}
    return model, tws_scaler, device


def predict_tft_sequence(model, X_seq, tws_scaler, device, slope_context=None):
    model.eval()
    if slope_context is None:
        slope_context = np.zeros((len(X_seq), 1), dtype=np.float32)
    dataset = TWSSequenceDataset(X_seq, np.zeros(len(X_seq), dtype=np.float32),
                                 slope_context=slope_context)
    loader  = DataLoader(dataset, batch_size=TFT_BATCH_SIZE * 4,
                         shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for xb, _, cb in loader:
            xb, cb = xb.to(device), cb.to(device)
            pred, _ = model(xb, context=cb)
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds)
    return preds * tws_scaler["denom"] + tws_scaler["min"]


# =============================================================================
# [FIX-8] ENSEMBLE DE TFT (seeds multiples) — moyenne des predictions
# =============================================================================
class TFTEnsemble:
    """
    Conteneur de N modeles TFT entraines avec des seeds differents.
    - predict_tft_sequence(ensemble, ...) renvoie la MOYENNE des predictions.
    - .eval() / .state_dict() delegues au 1er membre pour compat. aval
      (export cartes, sauvegarde .pt). Chaque membre partage le meme scaler.
    """
    def __init__(self, members, scalers, device):
        self.members = members          # liste de CausalTFT
        self.scalers = scalers          # liste de scalers (un par membre)
        self.device  = device
        self.granger_prior = members[0].granger_prior

    def eval(self):
        for m in self.members:
            m.eval()
        return self

    def to(self, device):
        for m in self.members:
            m.to(device)
        self.device = device
        return self

    def state_dict(self):
        # On sauvegarde le 1er membre (compat. format v7 .pt)
        return self.members[0].state_dict()

    def __call__(self, xb, context=None):
        # Agregation des sorties (espace normalise commun a tous les membres).
        # [FIX-11] "mean" ou "median" selon ENSEMBLE_AGG.
        outs, vw0 = [], None
        for m in self.members:
            pred, vw = m(xb, context=context)
            outs.append(pred)
            if vw0 is None:
                vw0 = vw
        stacked = torch.stack(outs, dim=0)
        if str(ENSEMBLE_AGG).lower() == "median":
            agg_pred = stacked.median(dim=0).values
        else:
            agg_pred = stacked.mean(dim=0)
        return agg_pred, vw0


def train_tft_ensemble(X_tr_seq, y_tr_seq, slope_context_tr,
                       granger_prior_vec, lambda_granger,
                       seeds):
    """
    [FIX-8] Entraine len(seeds) TFT et renvoie un TFTEnsemble.
    Le scaler de sortie est identique pour tous (memes y), donc on le reutilise.
    """
    members, scalers, device = [], [], None
    for i, sd in enumerate(seeds):
        print(f"   [Ensemble {i+1}/{len(seeds)}] TFT seed={sd} ...")
        m, sc, dev = train_tft(X_tr_seq, y_tr_seq, slope_context_tr,
                               granger_prior_vec, lambda_granger=lambda_granger,
                               seed=sd)
        members.append(m); scalers.append(sc); device = dev
    ens = TFTEnsemble(members, scalers, device)
    # scaler commun (identique pour tous les membres)
    return ens, scalers[0], device


# =============================================================================
# 11) PREDICTION FINALE [B1][B3]
# =============================================================================
def predict_pixel_with_attention(tft_model, tws_scaler, device, rfr, weights,
                                  X_pixel_seq, X_pixel_raw, V_norm_current,
                                  L, beta, tws_std_train,
                                  cluster_labels=None, slope_context=None):
    """
    Eq.11 residuelle : TWS_pred = TWS_base + beta * std_train * Att_spatial
    Avec Granger integre dans le TFT [C1/C2/C3], beta_opt devrait etre > 0.
    """
    w1, w2 = weights
    base_tft = predict_tft_sequence(tft_model, X_pixel_seq, tws_scaler, device,
                                    slope_context=slope_context)
    base_rfr = rfr.predict(X_pixel_raw)
    base     = w1 * base_tft + w2 * base_rfr

    if isinstance(L, dict) and cluster_labels is not None:
        att = compute_attention_map_spatial(V_norm_current, L, cluster_labels)
    else:
        L_use = L[-1] if isinstance(L, dict) else L
        att   = compute_attention_map(V_norm_current, L_use)

    return base + beta * tws_std_train * att


# =============================================================================
# 12) METRIQUES
# =============================================================================
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m  = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    if yt.size == 0:
        return float("nan"), float("nan"), float("nan")
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2   = float(r2_score(yt, yp)) if yt.size > 1 else float("nan")
    if yt.size > 1 and np.std(yt) > 0 and np.std(yp) > 0:
        r = float(np.corrcoef(yt, yp)[0, 1])
    else:
        r = float("nan")
    return rmse, r2, r


# =============================================================================
# 13a) RANDOM FOREST
# =============================================================================
def train_rfr(X_tr, y_tr):
    rfr = RandomForestRegressor(
        bootstrap=True, max_samples=RFR_MAX_SAMPLES, n_jobs=-1,
        random_state=RANDOM_SEED, **RFR_PARAMS)
    rfr.fit(X_tr, y_tr)
    return rfr


def rfr_predict_per_date(rfr, dates, api_maps, ndvi_maps, et_maps, tws_dict,
                          fine_profile, slope_map=None):
    slope_flat = slope_map.ravel() if slope_map is not None else None
    out = {}
    for d in dates:
        if d not in api_maps or d not in ndvi_maps or d not in et_maps:
            continue
        pr   = api_maps[d].ravel()
        ndvi = ndvi_maps[d].ravel()
        et   = et_maps[d].ravel()
        stack = (np.stack([pr, ndvi, et, slope_flat], axis=1) if slope_flat is not None
                 else np.stack([pr, ndvi, et], axis=1))
        finite = np.isfinite(stack).all(axis=1)
        if not finite.any():
            continue
        out[d] = float(np.mean(rfr.predict(stack[finite])))
    return out


# =============================================================================
# 13b) NNLS
# =============================================================================
def learn_ensemble_weights(pred_tft, pred_rfr, y_true):
    A = np.column_stack([pred_tft, pred_rfr])
    w, _ = nnls(A, y_true)
    s = w.sum()
    if s <= 0:
        return 0.5, 0.5
    w = w / s
    return float(w[0]), float(w[1])


# =============================================================================
# 13c) PIPELINE COMPLET v4
# =============================================================================
def train_and_evaluate_ensemble(all_dates, train_dates, test_dates,
                                api_maps, ndvi_maps, et_maps, tws_dict,
                                fine_profile, normalizer, L_granger,
                                L_dict, labels_2d,
                                slope_map=None, slope_mean=None):
    """
    Pipeline v4 — Causal-Guided TFT :
      1)  Sequences [Pr, NDVI, 1-ET, Att_Granger] [C3] + contexte Slope [A2]
      2)  Split interne sous-train / validation
      3)  Recherche lambda optimal [C1] par CV sur validation
      4)  TFT (sous-train) avec [C1] regularisation KL + [C2] prior VS
      5)  RFR sur train complet
      6)  NNLS -> w1, w2
      7)  Recherche beta residuel [B1] sur validation
      8)  [A4] TFT reentrainee sur 100% train avec lambda optimal
      9)  Evaluation finale sur test
    """
    # Prior de Granger pour les 3 variables temporelles (+ 0 pour Att si C3)
    prior_3 = granger_prior_vector(L_granger)   # (3,)
    if USE_ATT_AS_FEATURE:
        # Att_Granger est une meta-feature causale -> prior modere
        prior_vec = np.append(prior_3, 0.1)
        prior_vec = prior_vec / prior_vec.sum()
    else:
        prior_vec = prior_3

    print(f" [C2] Prior causal Granger : {np.round(prior_vec, 4)}")

    # [C3] Sequences avec Att comme 4e canal
    print(f" Construction des sequences ([C3] Att_Granger={'oui' if USE_ATT_AS_FEATURE else 'non'})...")
    X_all_seq, y_all_seq, seq_dates_all, slope_ctx_all, n_feat = build_sequences(
        all_dates, api_maps, ndvi_maps, et_maps, tws_dict, fine_profile, normalizer,
        slope_mean=slope_mean, L_granger=L_granger, labels_2d=labels_2d, L_dict=L_dict)
    print(f"   Sequences train/test | {n_feat} canaux par pas de temps")

    train_set = set(train_dates)
    test_set  = set(test_dates)
    tr_mask   = np.array([d in train_set for d in seq_dates_all])
    te_mask   = np.array([d in test_set  for d in seq_dates_all])

    X_tr_full = X_all_seq[tr_mask];  y_tr_full = y_all_seq[tr_mask]
    ctx_tr    = slope_ctx_all[tr_mask]
    X_te      = X_all_seq[te_mask];  y_te      = y_all_seq[te_mask]
    ctx_te    = slope_ctx_all[te_mask]
    seq_dates_tr = [seq_dates_all[i] for i in range(len(seq_dates_all)) if tr_mask[i]]
    seq_dates_te = [seq_dates_all[i] for i in range(len(seq_dates_all)) if te_mask[i]]
    print(f"   Sequences train: {len(X_tr_full)} | test: {len(X_te)}")

    # Split interne — [FIX-3] validation TEMPORELLE (fin de periode) au lieu d'aleatoire.
    # Un split aleatoire melange des sequences interpolees entre dates de train et
    # produit une RMSE_val optimiste non comparable au test (sous-echantillonnage regulier).
    n_tr    = len(X_tr_full)
    n_val   = max(2, int(round(n_tr * ENSEMBLE_VAL_FRACTION)))
    order   = np.argsort(np.array(seq_dates_tr))   # tri chronologique
    sub_idx = order[:-n_val]                        # sous-train = dates les plus anciennes
    val_idx = order[-n_val:]                        # validation = dates les plus recentes

    X_sub = X_tr_full[sub_idx];  y_sub = y_tr_full[sub_idx];  ctx_sub = ctx_tr[sub_idx]
    X_val = X_tr_full[val_idx];  y_val = y_tr_full[val_idx];  ctx_val = ctx_tr[val_idx]
    val_dates = [seq_dates_tr[i] for i in val_idx]
    print(f"   Split -> sous-train: {len(X_sub)} | validation: {len(X_val)}")

    # [C1] Recherche lambda optimal sur validation
    print(f" [C1] Recherche lambda optimal {LAMBDA_GRANGER_GRID}...")
    best_lambda, best_rmse_lam = LAMBDA_GRANGER_DEFAULT, np.inf
    for lam in LAMBDA_GRANGER_GRID:
        print(f"   Test lambda={lam}...")
        m_try, sc_try, dev_try = train_tft(X_sub, y_sub, ctx_sub,
                                            prior_vec, lambda_granger=lam)
        pred_try = predict_tft_sequence(m_try, X_val, sc_try, dev_try, slope_context=ctx_val)
        ok  = np.isfinite(pred_try) & np.isfinite(y_val)
        if ok.sum() < 2:
            continue
        r_try = float(np.sqrt(mean_squared_error(y_val[ok], pred_try[ok])))
        print(f"      -> RMSE_val={r_try:.4f} mm")
        if r_try < best_rmse_lam:
            best_rmse_lam, best_lambda = r_try, lam
    print(f"   Lambda optimal = {best_lambda} (RMSE_val = {best_rmse_lam:.4f} mm)")

    # TFT final sur sous-train avec lambda optimal
    print(f" [Membre 1/2] TFT (sous-train, lambda={best_lambda})...")
    tft_model, tws_scaler, device = train_tft(X_sub, y_sub, ctx_sub,
                                               prior_vec, lambda_granger=best_lambda)

    # RFR sur train complet
    print(" [Membre 2/2] RandomForest...")
    X_tr_pix, y_tr_pix, _, _, _ = assemble_samples(
        train_dates, api_maps, ndvi_maps, et_maps, tws_dict,
        fine_profile, normalizer=normalizer, fit_normalizer=False, slope_map=slope_map)
    rfr = train_rfr(X_tr_pix, y_tr_pix)

    tws_std_train = float(np.nanstd(y_tr_pix)) if y_tr_pix.size > 1 else 1.0
    print(f"   std TWS train = {tws_std_train:.3f} mm")

    # NNLS -> w1, w2
    print(" Apprentissage des poids d'ensemble (NNLS)...")
    tft_val = predict_tft_sequence(tft_model, X_val, tws_scaler, device, slope_context=ctx_val)
    rfr_val_dict = rfr_predict_per_date(rfr, val_dates, api_maps, ndvi_maps,
                                         et_maps, tws_dict, fine_profile, slope_map=slope_map)
    rfr_val = np.array([rfr_val_dict.get(d, np.nan) for d in val_dates])
    okv = np.isfinite(tft_val) & np.isfinite(rfr_val) & np.isfinite(y_val)
    w1, w2 = learn_ensemble_weights(tft_val[okv], rfr_val[okv], y_val[okv])
    print(f"   Poids -> w1(TFT)={w1:.3f} | w2(RFR)={w2:.3f}")

    # Recherche beta residuel
    print(" Recherche beta residuel post-hoc...")
    tws_base_val = w1 * tft_val + w2 * rfr_val
    V_last_val   = X_val[:, -1, :3]
    att_val      = compute_attention_map(V_last_val, L_granger)
    best_beta, best_rmse_beta = BETA_DEFAULT, np.inf
    for bt in BETA_GRID:
        y_val_p = tws_base_val[okv] + bt * tws_std_train * att_val[okv]
        ok2 = np.isfinite(y_val_p) & np.isfinite(y_val[okv])
        if ok2.sum() < 2:
            continue
        r_bt = float(np.sqrt(mean_squared_error(y_val[okv][ok2], y_val_p[ok2])))
        if r_bt < best_rmse_beta:
            best_rmse_beta, best_beta = r_bt, bt
    print(f"   Beta residuel = {best_beta} (RMSE_val = {best_rmse_beta:.4f} mm)")

    # [FIX-9] Coefficients de recalibration affine, ajustes SUR LA VALIDATION
    # (a partir du membre sous-train, qui n'a PAS vu la validation -> honnete).
    # y_cal = alpha * y_pred + b, par moindres carres ordinaires.
    cal_alpha, cal_b = 1.0, 0.0
    if RECALIBRATE_SCALE:
        y_val_final = tws_base_val[okv] + best_beta * tws_std_train * att_val[okv]
        okc = np.isfinite(y_val_final) & np.isfinite(y_val[okv])
        if okc.sum() >= 2:
            yv_true = y_val[okv][okc]
            yv_pred = y_val_final[okc]
            # OLS 1D : alpha = cov/var ; b = mean(true) - alpha*mean(pred)
            vp_mean = float(np.mean(yv_pred)); vt_mean = float(np.mean(yv_true))
            var_p   = float(np.mean((yv_pred - vp_mean) ** 2))
            cov_pt  = float(np.mean((yv_pred - vp_mean) * (yv_true - vt_mean)))
            if var_p > 1e-12:
                cal_alpha = cov_pt / var_p
                cal_b     = vt_mean - cal_alpha * vp_mean
            # Controle : RMSE val avant/apres recalibration
            rmse_before = float(np.sqrt(mean_squared_error(yv_true, yv_pred)))
            rmse_after  = float(np.sqrt(mean_squared_error(
                yv_true, cal_alpha * yv_pred + cal_b)))
            print(f"   [FIX-9] Recalibration (validation) : alpha={cal_alpha:.3f} "
                  f"b={cal_b:.2f} | RMSE_val {rmse_before:.3f} -> {rmse_after:.3f} mm")
            if rmse_after >= rmse_before:
                # garde-fou : si la recalibration n'aide pas EN VALIDATION, on l'annule
                print("   [FIX-9] Recalibration sans gain en validation -> desactivee.")
                cal_alpha, cal_b = 1.0, 0.0

    # [A4][FIX-8] Ensemble de TFT (seeds multiples) sur 100% du train
    seeds_used = ENSEMBLE_SEEDS[:max(1, N_ENSEMBLE_SEEDS)]
    print(f" [A4] Re-entrainement ENSEMBLE de {len(seeds_used)} TFT "
          f"100% train ({len(X_tr_full)} seq, lambda={best_lambda})...")
    tft_model, tws_scaler, device = train_tft_ensemble(
        X_tr_full, y_tr_full, ctx_tr, prior_vec,
        lambda_granger=best_lambda, seeds=seeds_used)

    # Predictions communes (RFR + attention, identiques pour tous les membres)
    rfr_te_dict = rfr_predict_per_date(rfr, seq_dates_te, api_maps, ndvi_maps,
                                        et_maps, tws_dict, fine_profile, slope_map=slope_map)
    rfr_te   = np.array([rfr_te_dict.get(d, np.nan) for d in seq_dates_te])
    V_last = X_te[:, -1, :3]
    att    = compute_attention_map(V_last, L_granger)

    # --- Metriques PAR MEMBRE (pour ecart-type) ---
    per_member = []
    for i, m in enumerate(tft_model.members):
        tft_te_i = predict_tft_sequence(m, X_te, tft_model.scalers[i], device,
                                        slope_context=ctx_te)
        y_pred_i = w1 * tft_te_i + w2 * rfr_te + best_beta * tws_std_train * att
        y_pred_i = cal_alpha * y_pred_i + cal_b   # [FIX-9] recalibration
        oki = np.isfinite(y_pred_i) & np.isfinite(y_te)
        per_member.append(metrics(y_te[oki], y_pred_i[oki]))
    per_member = np.array(per_member)   # shape (N, 3) -> (rmse, r2, r)
    mean_m = per_member.mean(axis=0)
    std_m  = per_member.std(axis=0)

    # --- Metrique de l'ENSEMBLE (moyenne des predictions) ---
    tft_te = predict_tft_sequence(tft_model, X_te, tws_scaler, device, slope_context=ctx_te)
    tws_base = w1 * tft_te + w2 * rfr_te
    y_pred   = tws_base + best_beta * tws_std_train * att
    y_pred   = cal_alpha * y_pred + cal_b   # [FIX-9] recalibration
    okt = np.isfinite(y_pred) & np.isfinite(y_te)
    rmse, r2, r = metrics(y_te[okt], y_pred[okt])

    # Affichage stats d'ensemble
    print(f"\n   [FIX-8] Stats sur {len(seeds_used)} seeds (membres individuels) :")
    print(f"      RMSE = {mean_m[0]:.2f} +/- {std_m[0]:.2f} mm")
    print(f"      R2   = {mean_m[1]*100:.1f} +/- {std_m[1]*100:.1f} %")
    print(f"      r    = {mean_m[2]*100:.1f} +/- {std_m[2]*100:.1f} %")
    print(f"   [FIX-8] Ensemble ({ENSEMBLE_AGG} des predictions) : "
          f"RMSE={rmse:.2f} | R2={r2*100:.1f}% | r={r*100:.1f}%")

    return ((rmse, r2, r), tft_model, tws_scaler, device, rfr,
            (w1, w2), best_beta, tws_std_train, best_lambda, n_feat, seq_dates_te,
            (cal_alpha, cal_b))


def print_result(rmse, r2, r, w1, w2, beta, lambda_g, n_feat, n_clusters):
    print("\n=== APPROCHE PROPOSEE v4 : Causal-Guided TFT + Ensemble ===")
    print(f"{'Model':<36}{'RMSE(mm)':>10}{'R2(%)':>10}{'Pearson r(%)':>14}")
    print(f"{'Causal-Guided TFT v4':<36}{rmse:>10.2f}{r2*100:>10.1f}{r*100:>14.1f}")
    print()
    print(f" TFT : {n_feat} canaux | lambda_KL={lambda_g} | gamma_prior={GAMMA_GRANGER_PRIOR}")
    print(f" [C1] Loss = Huber + {lambda_g}*KL(var_weights||p_granger)")
    print(f" [C2] Prior causal dans Variable Selection (gamma={GAMMA_GRANGER_PRIOR})")
    print(f" [C3] Att_Granger spatial comme {n_feat}e canal de sequence")
    print(f" Fusion NNLS : w1(TFT)={w1:.3f} w2(RFR)={w2:.3f} | beta_residuel={beta}")
    print(f" K*={n_clusters} zones | [B2] Multi-lag | [B3] Slope pixel | [B4] API decay")


# =============================================================================
# 14) VALIDATION PIEZOMETRIQUE
# =============================================================================
def validate_on_wells(tft_model, tws_scaler, device, rfr, weights, L_dict,
                       beta, tws_std_train, normalizer, api_maps, ndvi_maps,
                       et_maps, fine_profile, wells, all_dates, n_feat,
                       labels_2d=None, slope_map=None, slope_mean=None,
                       L_granger=None):
    from rasterio.transform import rowcol
    transform = fine_profile["transform"]
    Hf, Wf   = fine_profile["height"], fine_profile["width"]
    date_idx  = {d: i for i, d in enumerate(all_dates)}

    slope_n_basin = 0.0
    if slope_mean is not None and "SLOPE" in normalizer.vmin:
        slope_n_basin = float(normalizer.transform("SLOPE", np.array([slope_mean]))[0])

    preds, obs = [], []
    for w in wells:
        d = yyyymmdd_to_date(w["date"]) if isinstance(w["date"], str) else w["date"]
        if d not in api_maps or d not in ndvi_maps or d not in et_maps:
            continue
        try:
            row, col = rowcol(transform, w["lon"], w["lat"])
        except Exception:
            continue
        if not (0 <= row < Hf and 0 <= col < Wf):
            continue

        idx = date_idx.get(d, None)
        if idx is None or idx < TFT_SEQ_LEN:
            continue

        seq, valid_seq = [], True
        for k in range(TFT_SEQ_LEN):
            d_k = all_dates[idx - TFT_SEQ_LEN + k]
            if d_k not in api_maps or d_k not in ndvi_maps or d_k not in et_maps:
                valid_seq = False; break
            pr_k = float(api_maps[d_k][row, col])
            nd_k = float(ndvi_maps[d_k][row, col])
            et_k = float(et_maps[d_k][row, col])
            if not np.isfinite([pr_k, nd_k, et_k]).all():
                valid_seq = False; break
            pr_n = float(normalizer.transform("PR",   np.array([pr_k]))[0])
            nd_n = float(normalizer.transform("NDVI", np.array([nd_k]))[0])
            et_n = 1.0 - float(normalizer.transform("ET", np.array([et_k]))[0])
            step = [pr_n, nd_n, et_n]
            # [C3] Att comme 4e canal
            if n_feat == 4 and L_granger is not None:
                V_s = np.array([[pr_n, nd_n, et_n]])
                a_k = float(compute_attention_map(V_s, L_granger)[0])
                step.append(a_k)
            seq.append(step)
        if not valid_seq:
            continue

        slope_n_pix = slope_n_basin
        if slope_map is not None and "SLOPE" in normalizer.vmin:
            sl_pix = float(slope_map[row, col])
            if np.isfinite(sl_pix):
                slope_n_pix = float(normalizer.transform("SLOPE", np.array([sl_pix]))[0])

        X_pixel = np.array([seq], dtype=np.float32)
        ctx_pix = np.array([[slope_n_pix]], dtype=np.float32)
        V_curr  = np.array([[seq[-1][0], seq[-1][1], seq[-1][2]]])

        raw = [float(api_maps[d][row, col]),
               float(ndvi_maps[d][row, col]),
               float(et_maps[d][row, col])]
        if slope_map is not None:
            raw.append(float(slope_map[row, col]))
        X_raw = np.array([raw], dtype=float)

        clab = None
        if labels_2d is not None:
            clab = np.array([int(labels_2d[row, col])])

        p = predict_pixel_with_attention(
            tft_model, tws_scaler, device, rfr, weights,
            X_pixel, X_raw, V_curr, L_dict, beta, tws_std_train,
            cluster_labels=clab, slope_context=ctx_pix)[0]
        preds.append(p)
        obs.append(w["value"])

    if len(preds) < 2:
        print("   [WARN] validation piezometrique : pas assez de points.")
        return None
    return metrics(np.array(obs), np.array(preds))


# =============================================================================
# 15) EXPORT DES CARTES [B3][C3]
# =============================================================================
def export_downscaled_maps(tft_model, tws_scaler, device, rfr, weights, L_dict,
                            beta, tws_std_train, normalizer, api_maps, ndvi_maps,
                            et_maps, fine_profile, all_dates, map_dir,
                            labels_2d=None, slope_map=None, n_feat=3,
                            L_granger=None, domain_mask=None, clip_mask=None):
    Hf, Wf   = fine_profile["height"], fine_profile["width"]
    date_idx = {d: i for i, d in enumerate(all_dates)}
    os.makedirs(map_dir, exist_ok=True)

    # [CLIP]+[EDGE-FIX] Construction du masque d'export final :
    #   1) base = domaine de donnees (couverture TWS) si fourni,
    #   2) intersection avec le polygone du bassin (clip shapefile) si fourni,
    #   3) erosion d'un anneau de bordure pour retirer les pixels au minimum.
    # Si ni domaine ni clip ne sont disponibles, on exporte toute la grille.
    base_mask = None
    if domain_mask is not None:
        base_mask = np.asarray(domain_mask, dtype=bool)
    if clip_mask is not None:
        clip_mask = np.asarray(clip_mask, dtype=bool)
        base_mask = clip_mask if base_mask is None else (base_mask & clip_mask)

    export_mask = base_mask
    if base_mask is not None and EDGE_EROSION_PIXELS > 0:
        export_mask = erode_mask(base_mask, EDGE_EROSION_PIXELS)
        n_drop = int(base_mask.sum() - export_mask.sum())
        print(f" [EDGE-FIX] Masque export : clip={'oui' if clip_mask is not None else 'non'} "
              f"| erosion {EDGE_EROSION_PIXELS}px -> {n_drop} pixels de bord exclus "
              f"| {int(export_mask.sum())} pixels conserves.")
    elif base_mask is not None:
        print(f" [CLIP] Masque export (sans erosion) : "
              f"{int(base_mask.sum())} pixels conserves.")
    export_mask_flat = export_mask.ravel() if export_mask is not None else None

    slope_flat_norm = None
    if slope_map is not None and "SLOPE" in normalizer.vmin:
        slope_flat_norm = normalizer.transform("SLOPE", slope_map.ravel()).astype(np.float32)

    for d in all_dates:
        idx = date_idx.get(d, None)
        if idx is None or idx < TFT_SEQ_LEN:
            continue
        if d not in api_maps or d not in ndvi_maps or d not in et_maps:
            continue

        pr    = api_maps[d]
        ndvi  = ndvi_maps[d]
        et    = et_maps[d]
        stack = np.stack([pr.ravel(), ndvi.ravel(), et.ravel()], axis=1)
        finite = np.isfinite(stack).all(axis=1)
        # [EDGE-FIX] On restreint l'export au domaine erode (sans l'anneau de bord).
        if export_mask_flat is not None:
            finite &= export_mask_flat
        if not finite.any():
            continue

        X_valid = stack[finite]
        N_pix   = X_valid.shape[0]
        pr_n  = normalizer.transform("PR",   X_valid[:, 0])
        nd_n  = normalizer.transform("NDVI", X_valid[:, 1])
        et_n  = 1.0 - normalizer.transform("ET", X_valid[:, 2])
        V_curr = np.stack([pr_n, nd_n, et_n], axis=1)

        X_rfr = (np.column_stack([X_valid, slope_map.ravel()[finite]])
                 if slope_map is not None else X_valid)

        seq_feat = np.zeros((N_pix, TFT_SEQ_LEN, n_feat), dtype=np.float32)
        for k in range(TFT_SEQ_LEN):
            d_k = all_dates[idx - TFT_SEQ_LEN + k]
            if d_k not in api_maps or d_k not in ndvi_maps or d_k not in et_maps:
                continue
            pr_k = api_maps[d_k].ravel()[finite]
            nd_k = ndvi_maps[d_k].ravel()[finite]
            et_k = et_maps[d_k].ravel()[finite]
            seq_feat[:, k, 0] = normalizer.transform("PR",   pr_k).astype(np.float32)
            seq_feat[:, k, 1] = normalizer.transform("NDVI", nd_k).astype(np.float32)
            seq_feat[:, k, 2] = (1.0 - normalizer.transform("ET", et_k)).astype(np.float32)
            if n_feat == 4 and L_granger is not None:
                # [C3] Att_Granger pixel-par-pixel pour ce pas de temps
                V_k = np.stack([seq_feat[:, k, 0],
                                seq_feat[:, k, 1],
                                seq_feat[:, k, 2]], axis=1)
                if labels_2d is not None:
                    lab_k = labels_2d.ravel()[finite]
                    L_use = L_dict
                    att_k = compute_attention_map_spatial(V_k, L_use, lab_k)
                else:
                    att_k = compute_attention_map(V_k, L_granger)
                seq_feat[:, k, 3] = att_k.astype(np.float32)

        ctx_pix = (slope_flat_norm[finite].reshape(-1, 1)
                   if slope_flat_norm is not None
                   else np.zeros((N_pix, 1), dtype=np.float32))

        clab = labels_2d.ravel()[finite] if labels_2d is not None else None

        pred = predict_pixel_with_attention(
            tft_model, tws_scaler, device, rfr, weights,
            seq_feat, X_rfr, V_curr, L_dict, beta, tws_std_train,
            cluster_labels=clab, slope_context=ctx_pix)

        full = np.full(Hf * Wf, np.nan, dtype=np.float32)
        full[np.flatnonzero(finite)] = pred.astype(np.float32)

        # [EDGE-FIX] Filet de securite : ecrete les pics residuels par percentiles
        # calcules sur les pixels fiables de la carte courante.
        if CLIP_PREDICTION_PERCENTILES:
            vals = pred[np.isfinite(pred)]
            if vals.size > 10:
                lo = np.percentile(vals, PRED_PERCENTILE_LOW)
                hi = np.percentile(vals, PRED_PERCENTILE_HIGH)
                out_of_range = np.isfinite(full) & ((full < lo) | (full > hi))
                full[out_of_range] = np.nan

        out_path = os.path.join(map_dir, f"SR_TWS_{d.strftime('%Y%m%d')}.tif")
        write_geotiff(out_path, full.reshape(Hf, Wf), fine_profile, nodata_value=OUT_NODATA)


# =============================================================================
# 16) MAIN
# =============================================================================
def load_wells_if_any(sections):
    if "WELLS" not in sections:
        return None
    for entry in sections["WELLS"]:
        p = _norm_abs_from_entry(entry)
        if os.path.isfile(p) and p.lower().endswith(".csv"):
            wells = []
            with open(p, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
                hi = {h.strip().lower(): k for k, h in enumerate(header)}
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 4:
                        continue
                    try:
                        wells.append(dict(
                            lat=float(parts[hi["lat"]]),
                            lon=float(parts[hi["lon"]]),
                            date=parts[hi["date"]].strip(),
                            value=float(parts[hi["value"]]),
                        ))
                    except Exception:
                        continue
            print(f" Puits charges : {len(wells)}")
            return wells
    return None


def main():
    print("=" * 78)
    print(" GRSL - Causal-Guided TFT for TWS Downscaling — TENSIFT v4")
    print(" [C1] KL Granger loss | [C2] Prior VS | [C3] Att feature")
    print(" [B2] Multi-lag | [B3] Slope pixel | [B4] API decay | [B5] K sil.")
    print("=" * 78)

    ensure_name_file()

    sections  = parse_name_file(NAME_FILE)
    pr_dict   = build_date_dict(sections, SECTION_PR)
    tws_dict  = build_date_dict(sections, SECTION_TWS)
    ndvi_dict = build_date_dict(sections, SECTION_NDVI)
    et_dict   = build_date_dict(sections, SECTION_ET)

    dates_all = sorted(set(pr_dict.keys()) & set(tws_dict.keys()))
    if not dates_all:
        raise RuntimeError("Aucune date commune Pr/TWS.")
    print(f" Dates disponibles : {len(dates_all)} ({dates_all[0]} -> {dates_all[-1]})")
    print(f" Composites NDVI : {len(ndvi_dict)} | ET : {len(et_dict)}")

    fine_profile = get_fine_profile_from_pr(pr_dict)

    # Slope
    slope_path = get_static_path(sections, SECTION_SLOPE, SLOPE_ALIASES)
    slope_map  = None
    slope_mean = None
    if slope_path is not None:
        print(f" Slope : {slope_path}")
        slope_map  = load_var_on_fine(slope_path, fine_profile)
        slope_mean = float(np.nanmean(slope_map))
        print(f"   Moyenne bassin = {slope_mean:.3f}")
    else:
        print(" [WARN] Slope introuvable.")

    # [GAP-FILL] Masque de DOMAINE = couverture TWS (la cible n'est definie que
    # sur le bassin GRACE). On ne comble QUE l'interieur de ce domaine ; le
    # hors-domaine reste no-data legitime (ex. hors-bassin / ocean).
    print(" Construction du masque de domaine (couverture TWS)...")
    Hf, Wf = fine_profile["height"], fine_profile["width"]
    domain_mask = np.zeros((Hf, Wf), dtype=bool)
    for d in sorted(tws_dict.keys()):
        try:
            tws_c, gprof, gnod = load_raster(tws_dict[d])
            tws_c = to_float_with_nan(tws_c, gnod)
            tws_f = reproject_to_profile(tws_c, gprof, gnod, fine_profile,
                                         dst_nodata=np.nan,
                                         resampling=Resampling.nearest)
            domain_mask |= np.isfinite(tws_f)
        except Exception:
            continue
    if not domain_mask.any():
        print("   [WARN] Domaine TWS vide -> comblement sur toute la grille.")
        domain_mask = None
    else:
        print(f"   Domaine : {int(domain_mask.sum())} pixels valides "
              f"({100.0*domain_mask.sum()/(Hf*Wf):.1f}% de la grille).")

    # [B4] API
    print(" API [B4] decay spatial si slope...")
    api_maps = compute_api_series(pr_dict, fine_profile, dates_all, slope_map=slope_map)

    # [GAP-FILL] Comble les trous residuels de l'API dans le domaine, pour que
    # les 3 covariables (API, NDVI, ET) partagent la meme couverture spatiale et
    # qu'aucun pixel du bassin ne soit ecarte a l'export par un seul NaN d'API.
    if domain_mask is not None:
        for d in list(api_maps.keys()):
            api_maps[d] = spatial_fill_nearest(api_maps[d], domain_mask=domain_mask)

    print(" Interpolation + gap-fill NDVI / ET...")
    ndvi_maps = build_interp_series(ndvi_dict, fine_profile, dates_all,
                                    domain_mask=domain_mask, fill_residual=True)
    et_maps   = build_interp_series(et_dict,   fine_profile, dates_all,
                                    domain_mask=domain_mask, fill_residual=True)
    dates_all = [d for d in dates_all if d in ndvi_maps and d in et_maps]
    if not dates_all:
        raise RuntimeError("Aucune date avec NDVI+ET.")

    # [FIX-7] Split 80/20 — CHRONOLOGIQUE PUR (test = derniere periode, jamais vue).
    # Remplace le sous-echantillonnage 1-sur-5, dont les dates de test etaient
    # entourees de dates de train (a +/-1 jour) avec NDVI/ET interpoles : un tel
    # test n'est pas independant et surestime R2. Le split chronologique mesure
    # la vraie capacite de generalisation a un futur inconnu.
    idx = np.arange(len(dates_all))
    if CHRONOLOGICAL_SPLIT:
        n_test      = max(1, int(round(len(dates_all) * TEST_FRACTION)))
        cut         = len(dates_all) - n_test
        train_dates = [dates_all[i] for i in idx[:cut]]
        test_dates  = [dates_all[i] for i in idx[cut:]]
        print(f" Train: {len(train_dates)} | Test: {len(test_dates)} "
              f"[split chronologique : test = {test_dates[0]} -> {test_dates[-1]}]")
    else:
        step      = int(round(1.0 / TEST_FRACTION))
        test_mask = (idx % step == 0)
        test_dates  = [dates_all[i] for i in idx[test_mask]]
        train_dates = [dates_all[i] for i in idx[~test_mask]]
        print(f" Train: {len(train_dates)} | Test: {len(test_dates)} [sous-echantillonnage regulier]")

    # Normalisation
    _, _, _, normalizer, _ = assemble_samples(
        train_dates, api_maps, ndvi_maps, et_maps, tws_dict,
        fine_profile, normalizer=None, fit_normalizer=True, slope_map=slope_map)

    # [B2] Granger multi-lag global
    print(" Granger global [B2 multi-lag]...")
    series = {
        "PR":   np.array([np.nanmean(api_maps[d])  for d in dates_all]),
        "NDVI": np.array([np.nanmean(ndvi_maps[d]) for d in dates_all]),
        "ET":   np.array([np.nanmean(et_maps[d])   for d in dates_all]),
    }
    L_granger = build_granger_matrix_multilag(series, ("PR", "NDVI", "ET"))
    print(" L_Granger =\n", np.round(L_granger, 4))

    # [B5] Clustering K optimal
    print(" Clustering [B5] silhouette...")
    labels_2d, kmeans, cluster_scaler, best_K = build_pixel_clusters(
        api_maps, ndvi_maps, et_maps, fine_profile, dates_all,
        k_range=K_RANGE, seed=CLUSTER_SEED, slope_map=slope_map)

    # Granger par zone [B2]
    print(" Granger par zone [B2]...")
    L_dict = build_granger_matrices_per_cluster(
        labels_2d, api_maps, ndvi_maps, et_maps, dates_all, L_granger,
        min_pixels=MIN_PIXELS_PER_CLUSTER)
    for k in sorted(L_dict.keys()):
        tag = "globale (fallback)" if k == -1 else f"zone {k}"
        print(f"   L[{tag}] =\n", np.round(L_dict[k], 4))

    print(f" TFT : hidden={TFT_HIDDEN_SIZE} heads={TFT_ATTENTION_HEADS} "
          f"seq={TFT_SEQ_LEN} ep={TFT_MAX_EPOCHS}")
    print(f" [C1] lambda grid : {LAMBDA_GRANGER_GRID}")
    print(f" [C2] gamma prior : {GAMMA_GRANGER_PRIOR}")
    print(f" [C3] Att feature : {USE_ATT_AS_FEATURE}")

    # Pipeline
    out = train_and_evaluate_ensemble(
        dates_all, train_dates, test_dates,
        api_maps, ndvi_maps, et_maps, tws_dict,
        fine_profile, normalizer, L_granger, L_dict, labels_2d,
        slope_map=slope_map, slope_mean=slope_mean)

    (rmse, r2, r), tft_model, tws_scaler, device, rfr, weights, \
        beta_opt, tws_std_train, lambda_opt, n_feat, seq_dates_te, \
        recal = out
    cal_alpha, cal_b = recal
    w1, w2 = weights
    print_result(rmse, r2, r, w1, w2, beta_opt, lambda_opt, n_feat, best_K)
    if RECALIBRATE_SCALE and (abs(cal_alpha - 1.0) > 1e-6 or abs(cal_b) > 1e-6):
        print(f" [FIX-9] Recalibration d'echelle appliquee aux metriques : "
              f"y_cal = {cal_alpha:.3f}*y_pred + {cal_b:.2f} "
              f"(coeffs appris sur validation, pas sur test)")
        print(" [FIX-9] NB : cartes exportees NON recalibrees (correction validee "
              "sur la moyenne du bassin, non pixel-par-pixel).")

    # Piezometrique
    wells = load_wells_if_any(sections)
    if wells:
        res = validate_on_wells(
            tft_model, tws_scaler, device, rfr, weights, L_dict, beta_opt,
            tws_std_train, normalizer, api_maps, ndvi_maps, et_maps,
            fine_profile, wells, dates_all, n_feat,
            labels_2d=labels_2d, slope_map=slope_map, slope_mean=slope_mean,
            L_granger=L_granger)
        if res:
            rmse_w, r2_w, r_w = res
            print(f"\n=== VALIDATION PIEZOMETRIQUE ({len(wells)} mesures) ===")
            print(f" RMSE={rmse_w:.2f} mm | R2={r2_w*100:.1f}% | r={r_w*100:.1f}%")

    # Export cartes
    print("\n Export cartes TWS (0.05 deg)...")
    # [CLIP] Masque du bassin de Tensift construit sur la grille fine.
    clip_mask = build_clip_mask_from_shapefile(TENSIFT_SHP, fine_profile)
    sr_maps_dir = os.path.join(OUT_DIR, "SR_MAPS_TWS")
    export_downscaled_maps(
        tft_model, tws_scaler, device, rfr, weights, L_dict,
        beta_opt, tws_std_train, normalizer,
        api_maps, ndvi_maps, et_maps, fine_profile, dates_all,
        sr_maps_dir,
        labels_2d=labels_2d, slope_map=slope_map,
        n_feat=n_feat, L_granger=L_granger,
        domain_mask=domain_mask, clip_mask=clip_mask)

    # --------- VARIANCE SOUS-MAILLE (structure fine, methode adoptee) ---------
    # Import differe pour eviter tout import circulaire (subgrid_common importe
    # causal_tft_pipeline). Ne calcule rien de nouveau cote modele : lit les cartes fines
    # qui viennent d'etre exportees et applique la definition commune de sigma-bar.
    try:
        import subgrid_common as sg
        sg.report_sigma_bar(
            "Adopted method (GRSL causal-guided TFT)",
            sr_maps_dir,
            {"test_dates": test_dates,
             "tws_dict": tws_dict,
             "fine_profile": fine_profile},
        )
    except Exception as exc:
        print(f" [WARN] Calcul sigma-bar ignore ({exc}).")

    if labels_2d is not None:
        zone_path = os.path.join(OUT_DIR, "HYDRO_CLIMATIC_ZONES_TWS.tif")
        write_geotiff(zone_path, labels_2d.astype(np.float32),
                      fine_profile, nodata_value=-1.0)
        print(f" Zones hydro-climatiques : {zone_path}")

    # Sauvegarde
    import joblib
    model_path = os.path.join(OUT_DIR, "ensemble_tft_causal_v4_tensift_tws.pt")
    torch.save({
        "model_state_dict":   tft_model.state_dict(),
        "tws_scaler":         tws_scaler,
        "normalizer_vmin":    normalizer.vmin,
        "normalizer_vmax":    normalizer.vmax,
        "L_granger":          L_granger,
        "L_granger_per_zone": L_dict,
        "n_clusters_opt":     best_K,
        "use_slope":          slope_map is not None,
        "slope_mean":         slope_mean,
        "beta_opt":           beta_opt,
        "lambda_opt":         lambda_opt,
        "tws_std_train":      tws_std_train,
        "n_feat":             n_feat,
        "ensemble_weights":   {"w1_tft": w1, "w2_rfr": w2},
        "ameliorations_v4":   ["C1_KL_causal_loss", "C2_granger_prior_VS",
                               "C3_att_as_feature",
                               "B2_multilag", "B3_slope_pixel",
                               "B4_API_decay", "B5_K_silhouette",
                               "A2_slope_context", "A3_intrazone_norm",
                               "A4_retrain_full"],
        "hyperparams": {
            "seq_len":      TFT_SEQ_LEN,  "hidden_size":  TFT_HIDDEN_SIZE,
            "n_heads":      TFT_ATTENTION_HEADS, "dropout": TFT_DROPOUT,
            "hidden_cont":  TFT_HIDDEN_CONT, "n_feat":  n_feat,
            "context_size": 1, "gamma_prior": GAMMA_GRANGER_PRIOR,
            "lambda_granger": lambda_opt,
        }
    }, model_path)
    rfr_path = os.path.join(OUT_DIR, "ensemble_rfr_v4_tensift_tws.joblib")
    joblib.dump(rfr, rfr_path)
    if kmeans is not None:
        joblib.dump({"kmeans": kmeans, "cluster_scaler": cluster_scaler,
                     "L_dict": L_dict, "K_opt": best_K},
                    os.path.join(OUT_DIR, "spatial_granger_v4_tensift_tws.joblib"))

    print(f" TFT : {model_path}")
    print(f" RFR : {rfr_path}")
    print(f" w1={w1:.3f} w2={w2:.3f} | lambda={lambda_opt} | beta={beta_opt} | K*={best_K}")
    print("\n Termine.")


if __name__ == "__main__":
    main()
