# Causal-Guided Temporal Fusion Transformer for Downscaling Terrestrial Water Storage

Reproducibility repository for the paper:

> **Causal-Guided Temporal Fusion Transformer for Downscaling Terrestrial Water Storage data: Case of the Tensift River Basin, Morocco**
> Anasse Boutayeb, Iyad Lahsen-Cherif, Ahmed El Khadimi, Manal Abbassi
> *Submitted to the **Green MM Workshop**, 34th ACM International Conference on Multimedia (ACM MM '26), Rio de Janeiro, Brazil.*

The GLDAS TWS product is downscaled from 0.25° to 0.05° (five-fold) over the Tensift river basin (Morocco) using a Temporal Fusion Transformer (TFT) guided by a spatially-explicit Granger causality structure, fused with a Random Forest Regressor (RFR) via non-negative least squares (NNLS).

---

## Pipeline

![Causal-Guided TFT workflow](assets/workflow.jpg)

Three causal-guidance mechanisms are injected inside the TFT block (Sec. 4.3 of the paper):

- **C1 — Causal loss regularization:** KL divergence between the variable-selection weights and the Granger prior added to the Huber loss.
- **C2 — Causal prior in variable selection:** the Granger prior biases the variable-selection logits.
- **C3 — Granger attention as an extra pathway:** the spatially-variable Granger attention map is appended as a fourth temporal channel alongside `Pr`, `NDVI`, `1 − ET`.

The Granger matrices are estimated **per hydro-climatic zone** (K-means, K\* = 3 by silhouette score) with multi-lag AIC weighting (L_max = 5). The TFT is a 9-member median ensemble; slope is encoded as a static context through a Gated Residual Network. The final prediction is `w1 · TFT + w2 · RFR` with NNLS-fitted weights (`w1 = 0.854`, `w2 = 0.138`), followed by an OLS affine recalibration on the validation split.

---

## Repository layout

```
causal-guided-tft-tws/
├── README.md
├── assets/
│   └── workflow.jpg                  # Fig. 5 of the paper
├── src/                              # Python modules (importable from notebooks)
│   ├── GRSL_v12.py                   # Main pipeline (adopted method)
│   ├── ABLATION_TFT_RFR_C1.py
│   ├── ABLATION_TFT_RFR_C1_C2.py
│   ├── ABLATION_TFT_RFR_corr.py
│   ├── ABLATION_TFT_RFR_uniform.py
│   ├── SOTA_LightGBM.py
│   ├── SOTA_RFR.py
│   ├── SOTA_SVR.py
│   ├── SOTA_TFT.py
│   ├── SOTA_XGBoost.py
│   ├── ablation_common.py            # shared helper — ablation config + run + export
│   ├── sota_common.py                # shared helper — SOTA data prep + 0.25° eval + export
│   └── subgrid_common.py             # shared helper — sub-grid variance σ̄ (Eq. 24)
└── notebooks/                        # One Colab notebook per module
    ├── GRSL_v12.ipynb
    ├── ABLATION_TFT_RFR_C1.ipynb
    ├── ABLATION_TFT_RFR_C1_C2.ipynb
    ├── ABLATION_TFT_RFR_corr.ipynb
    ├── ABLATION_TFT_RFR_uniform.ipynb
    ├── SOTA_LightGBM.ipynb
    ├── SOTA_RFR.ipynb
    ├── SOTA_SVR.ipynb
    ├── SOTA_TFT.ipynb
    └── SOTA_XGBoost.ipynb
```

The three helpers `ablation_common.py`, `sota_common.py`, `subgrid_common.py` factor out the common data pipeline (train/test split, aggregation to 0.25°, raster export, sub-grid variance report) shared by all ablation and SOTA modules.

---

## Data

All inputs (see Table 1 of the paper) are downloaded via the **Google Earth Engine** API on a common WGS84 0.05° grid and clipped to the Tensift basin.

| Variable          | Native resolution | Frequency | GEE asset / Source              |
|-------------------|-------------------|-----------|---------------------------------|
| TWS (GLDAS)       | 0.25°             | Daily     | `NASA/GLDAS/V022/CLSM/G025/DA1D` |
| API precipitation | 0.05°             | Daily     | `UCSB-CHG/CHIRPS/DAILY`         |
| ET                | 500 m             | 8-day     | `MODIS/061/MOD16A2GF`           |
| NDVI              | 250 m             | 16-day    | `MODIS/061/MOD13Q1`             |
| Slope             | static            | —         | SRTM DTM                        |

**Expected folder layout** (referenced by `BASE_DIR` in `GRSL_v12.py`, override in the notebooks):

```
Project_Tensift/
├── TWS_GLDAS_Clipped/
├── CHIRPS_GEE_Clipped/
├── MODIS_NDVI_GEE_Clipped/
├── MODIS_ET_GEE_Clipped/
├── Slope_Tensift.tif
└── Tensift.shp                        # + .shx, .dbf, .prj
```

Study period: **31 March 2023 – 30 October 2025** (944 daily dates), split 80 % train / 20 % test uniformly over the whole period.

---

## Running the notebooks (Google Colab)

Each notebook follows the same skeleton:

1. `pip install` the required extras.
2. Mount Google Drive.
3. `git clone` this repository.
4. Set `DATA_ROOT` to the folder that holds your `Project_Tensift/` data.
5. Import the corresponding module and call `main()`.

For any TFT-based notebook (`GRSL_v12`, all ablations, `SOTA_TFT`), switch the Colab runtime to a **T4 GPU** (*Runtime → Change runtime type → T4 GPU*). Tree-based SOTA (`SOTA_RFR`, `SOTA_XGBoost`, `SOTA_LightGBM`, `SOTA_SVR`) run comfortably on CPU.

---

## Results

### Table 3 — Main testing results (evaluation aggregated to 0.25°, protocol Sec. 5.1)

| Model                        | RMSE (mm) | R² (%) | r (%) | σ̄ (mm) |
|------------------------------|-----------|--------|-------|---------|
| SVR                          | 59.242    | 55.3   | 74.4  | 26.84   |
| XGBoost                      | 28.336    | 89.8   | 94.8  | 35.83   |
| LightGBM                     | 30.899    | 87.8   | 93.8  | 35.46   |
| Classical RFR                | 43.295    | 76.1   | 87.6  | 26.62   |
| Standalone TFT               | 49.930    | 68.2   | 82.7  | 19.79   |
| Standard TFT + RFR fusion    | 45.341    | 73.8   | 86.2  | 15.31   |
| **Adopted method (ours)**    | **5.25**  | **99.65** | **99.82** | **4.79** |

### Table 4 — Ablation study

Results differ slightly from Table 3 because the test dataset is drawn with different random sampling.

| Resulting model                     | RMSE (mm) | R² (%) | r (%) | σ̄ (mm) |
|-------------------------------------|-----------|--------|-------|---------|
| Classical RFR                       | 41.8720   | 77.60  | 88.10 | 26.6167 |
| TFT + RFR (uniform weight)          | 11.8610   | 98.21  | 99.10 | 14.3071 |
| TFT + RFR (correlation matrix)      | 11.1534   | 98.41  | 99.20 | 8.8850  |
| + C1 (KL loss)                      | 9.2476    | 98.91  | 99.45 | 8.44    |
| + C1 + C2 (causal prior)            | 7.3189    | 99.31  | 99.66 | 6.30    |
| **Adopted method (C1 + C2 + C3)**   | **5.7340** | **99.58** | **99.79** | **4.69** |

### Table 2 — Retained hyperparameters (Gaussian Process optimization)

| Block           | Hyperparameter                       | Value             |
|-----------------|--------------------------------------|-------------------|
| RFR member      | Number of estimators                 | 800               |
|                 | Max tree depth                       | 25                |
|                 | Min samples to split                 | 20                |
|                 | Min samples per leaf                 | 10                |
|                 | Max features per split               | 0.357             |
|                 | Max samples                          | 0.75              |
| TFT model       | Attention heads                      | 4                 |
|                 | Sequence length                      | 60                |
|                 | Epochs                               | 60                |
|                 | Dropout                              | 0.1               |
| Causal prior    | Prior strength γ                     | 0.5               |
|                 | Granger max lag L_max                | 5                 |
| Ensemble        | Members M                            | 9                 |
|                 | Weights (w1, w2)                     | (0.854, 0.138)    |
| Zones           | Hydro-climatic zones K\*             | 3                 |
| Recalibration   | Residual β                           | 0.0               |

### Sub-grid variance σ̄

Following Eq. (24) of the paper, σ̄ is the mean over all 0.25° coarse cells and all test dates of the intra-cell standard deviation of the 0.05° downscaled sub-pixels. Trivial replication (all sub-pixels equal to the coarse value) gives σ̄ → 0, while a meaningful downscaling produces a strictly positive σ̄ that reflects genuine sub-grid structure.

---

## Contact

Anasse Boutayeb — `boutayebanasse@gmail.com`
Institut National des Postes et des Télécommunications (INPT), Rabat, Morocco.

