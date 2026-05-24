# Asteroid Family Classification Using Proper Orbital Elements

A data science project analyzing the structure of the main asteroid belt using proper orbital elements from the AstDyS-2 catalog. The project covers unsupervised clustering, supervised classification, and background population analysis across ~1 million asteroids.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Results](#results)
- [Key Findings](#key-findings)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Overview

Asteroid families are groups of asteroids that share a common origin — fragments of a parent body destroyed in a collisional event billions of years ago. Members of the same family have nearly identical **proper orbital elements**: semi-major axis ($a_p$), eccentricity ($e_p$), and inclination ($\sin i_p$), which are conserved quantities that remain stable over long timescales unlike osculating elements.

This project uses the AstDyS-2 proper element catalog to:

1. Reproduce and visualize the main belt structure in proper element space
2. Map chaotic regions using Lyapunov Characteristic Exponents (LCE) and RMS uncertainties
3. Recover asteroid family structure through unsupervised clustering (HCM, DBSCAN, HDBSCAN)
4. Build supervised classifiers for binary (family vs background) and multiclass (which family) problems
5. Analyze the background population for primordial vs escaped family members

---

## Dataset

Data sourced from the [AstDyS-2 catalog](https://newton.spacedys.com/~astdys2/propsynth/):

| File | Description | Records |
|---|---|---|
| `all.syn` | Proper elements: $a_p$, $e_p$, $\sin i_p$, $n$, $g$, $s$, LCE | 1,152,208 |
| `all.sig` | RMS uncertainties of proper elements | 1,152,208 |
| `all_tro.famrec` | Family membership labels (AstDyS classification) | 1,152,208 |

After filtering (RMS threshold + orbital bounds + LCE > 0):

| Population | Count | Percentage |
|---|---|---|
| Total | 1,038,241 | 100% |
| Family members | 230,607 | 22.2% |
| Background | 807,634 | 77.8% |
| Unique families | 98 | — |

---

## Project Structure
```bash
├── data/
│   ├── lvl1/                              # Raw input files
│   │   ├── proper_elements.syn
│   │   ├── RMS_main_belt.sig
│   │   └── family_classification.famrec
│   ├── lvl2/                              # Cleaned master dataframe
│   │   └── master_df_clean.pkl
│   └── results/                           # Model outputs and predictions
├── plots/                                 # All generated figures
├── tuner_logs/                            # Keras Tuner trial logs
├── 01_data_fetching.ipynb                 # Data download and acquisition
├── 02_data_parsing.ipynb                  # Stage 1 — Data ingestion & cleaning
├── 03_data_merge.ipynb                    # Stage 1 — Merging datasets
├── 04_data_visualization.ipynb            # Stage 2 & 3 — EDA & chaos mapping
├── 05a_clustering_classic_HCM.ipynb       # Stage 4 — HCM classic features
├── 05b_clustering_classic_DBSCAN.ipynb    # Stage 4 — DBSCAN/HDBSCAN classic
├── 05c_clustering_extended_HCM.ipynb      # Stage 4 — HCM extended features
├── 05d_clustering_extended_DB_HDBSCAN.ipynb # Stage 4 — DBSCAN/HDBSCAN extended
├── 06_1_binary_rf.ipynb                   # Stage 5 — Binary RF
├── 06_2_binary_svm.ipynb                  # Stage 5 — Binary SVM
├── 06_3_binary_dl_dense.ipynb             # Stage 5 — Binary DL Dense
├── 07_1_multiclass_rf.ipynb               # Stage 5 — Multiclass RF (98 classes)
├── 07_2_multiclass_dl_dense.ipynb         # Stage 5 — Multiclass DL baseline
├── 07_3_multiclass_dl_bayesian.ipynb      # Stage 5 — Multiclass DL + Keras Tuner
├── 08_background_population_analysis.ipynb # Stage 6 — Background analysis
└── 09_multiclass_99_dl_keras_tuner.ipynb  # Stage 5 — 99-class end-to-end DL
```
---

## Pipeline

```bash
Raw AstDyS-2 Files
↓
01_data_fetching
↓
02_data_parsing + 03_data_merge
Parse proper elements, RMS, family labels
Filter: RMS threshold + orbital bounds + LCE > 0
Merge into master dataframe (1,038,241 asteroids)
↓
04_data_visualization
Proper element planes (a_p vs e_p, a_p vs sin_i_p)
Kirkwood gap annotation
LCE and RMS chaos maps
Family size distributions
↓
05a–05d — Unsupervised Clustering
HCM (classic + extended features)
DBSCAN / HDBSCAN (classic + extended features)
Evaluate vs official labels (ARI, NMI)
↓
06_1–06_3 — Binary Classification
RF, SVM, DL Dense (family vs background)
↓
07_1–07_3 — Multiclass Classification (98 classes)
RF, DL Dense, DL + Keras Tuner Bayesian
↓
08 — Background Population Analysis
Halo candidate identification
Primordial vs escaped member analysis
↓
09 — 99-class End-to-End Classification
DL + Keras Tuner (full belt, single model)
```

---

## Results

### Stage 4 — Unsupervised Clustering

| Method | Features | ARI | NMI | Clusters Found |
|---|---|---|---|---|
| HCM | Classic (a, e, sin i) | 0.7945 | 0.9446 | 132 vs 98 |
| HCM | Extended (+ g, s) | 0.7945 | 0.9445 | 160 vs 98 |
| DBSCAN | Classic | ~0.017 | ~0.18 | 101 |
| DBSCAN | Extended | 0.376 | 0.717 | 291 |
| **HDBSCAN** | **Classic** | **0.9779** | **0.9785** | **160** |
| HDBSCAN | Extended | 0.9547 | 0.9747 | 160 |

### Stage 5 — Binary Classification (Family vs Background)

| Model | Features | F1 | AUC |
|---|---|---|---|
| **RF** | **Orbital** | **0.9372** | **0.9960** |
| RF | Full | 0.9259 | 0.9944 |
| DL Dense | Orbital | 0.8113 | 0.9800 |
| DL Dense | Full | 0.8000 | 0.9766 |
| SVM RBF | Orbital | 0.6164 | 0.8857 |
| SVM Linear | Orbital | ~0.0002 | ~0.61 |

### Stage 5 — Multiclass Classification (98 families, family members only)

| Model | Features | Accuracy | F1 Macro |
|---|---|---|---|
| **RF** | **Orbital** | **0.9992** | 0.9919 |
| **RF** | **Full** | **0.9992** | **0.9941** |
| DL Tuned (Keras Tuner) | Orbital | 0.9894 | 0.9467 |
| DL Dense | Orbital | 0.9564 | 0.9183 |
| DL Dense | Full | 0.9010 | 0.8010 |

### Stage 5 — 99-class End-to-End (Full Belt)

| Model | Classes | Accuracy | F1 Macro |
|---|---|---|---|
| DL Tuned (Keras Tuner) | 99 | 0.6054 | 0.3196 |

### Stage 6 — Background Population

| Metric | Value |
|---|---|
| Total background | 807,634 (77.8%) |
| Halo candidates (near families) | 350,819 (43.4% of background) |
| Likely primordial (isolated) | 19,387 (2.4% of background) |
| Background median LCE | 2.990 |
| Family median LCE | 1.970 |

---

## Key Findings

**1. Proper orbital elements alone are sufficient for family classification.**
Random Forest achieves 0.9992 accuracy on the 98-class problem using only $a_p$, $e_p$, $\sin i_p$, $g$, $s$.

**2. Chaos indicators are task-dependent.**
RMS and LCE consistently hurt binary classification but marginally help RF multiclass — chaos encodes family-specific dynamical history useful for distinguishing geometrically similar families, but not for separating families from background.

**3. The family/background boundary is highly non-linear.**
Linear SVM achieves F1 = 0.0002 while RBF SVM achieves 0.616 and RF achieves 0.937. Tree-based ensembles handle this non-linearity most effectively.

**4. HDBSCAN recovers family structure without labels.**
ARI of 0.9779 means unsupervised clustering nearly matches supervised classification performance — families are genuinely distinct structures in proper element space.

**5. The background population is not primordial.**
Only 2.4% of background asteroids are genuinely isolated. 43.4% sit near family halos — likely escaped members or resonance-scattered material.

**6. Two-stage pipeline outperforms single end-to-end model.**
99-class single model achieves F1 Macro 0.32 vs 0.95+ for the two-stage approach. Separating family detection from family assignment is both architecturally and scientifically cleaner.

**7. Precession frequencies (g, s) do not improve clustering.**
These frequencies diffuse chaotically over Gyr timescales, making them unreliable for HCM and HDBSCAN despite carrying dynamical information useful in supervised settings.

---

## Recommended Production Pipeline

New asteroid with proper elements (a_p, e_p, sin_i_p, g, s)
↓
RF Binary Classifier (06_1)
↓
┌──────────┴──────────┐
Background      Family member
│                     ↓
│        RF Multiclass Classifier (07_1)
│        (+ rms_a, rms_e, rms_sini, LCE)
│                     ↓
│           Family assignment
│           (1 of 98 families)
└─────────────────────┘

---

## Requirements

Python >= 3.10
Core
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
Deep Learning
tensorflow >= 2.10
keras-tuner
GPU Acceleration (optional)
cudf
cuml
cupy
rapids >= 25.10

Install core dependencies:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn tensorflow keras-tuner
```

For RAPIDS GPU acceleration (NVIDIA GPU required):

```bash
# Follow official RAPIDS installation guide at https://rapids.ai
```

---

## Usage

Run notebooks in order:

```bash
jupyter notebook 01_data_fetching.ipynb
jupyter notebook 02_data_parsing.ipynb
jupyter notebook 03_data_merge.ipynb
jupyter notebook 04_data_visualization.ipynb

# Clustering
jupyter notebook 05a_clustering_classic_HCM.ipynb
jupyter notebook 05b_clustering_classic_DBSCAN.ipynb
jupyter notebook 05c_clustering_extended_HCM.ipynb
jupyter notebook 05d_clustering_extended_DB_HDBSCAN.ipynb

# Binary classification
jupyter notebook 06_1_binary_rf.ipynb
jupyter notebook 06_2_binary_svm.ipynb
jupyter notebook 06_3_binary_dl_dense.ipynb

# Multiclass classification
jupyter notebook 07_1_multiclass_rf.ipynb
jupyter notebook 07_2_multiclass_dl_dense.ipynb
jupyter notebook 07_3_multiclass_dl_bayesian.ipynb

# Background analysis
jupyter notebook 08_background_population_analysis.ipynb

# 99-class end-to-end
jupyter notebook 09_multiclass_99_dl_keras_tuner.ipynb
```

Place raw data files in `data/lvl1/` before running `02_data_parsing.ipynb`.

---

## References

- Knežević, Z. & Milani, A. (2003). Proper element catalogs and asteroid families. *A&A*, 403, 1165–1173.
- Zappalà, V. et al. (1995). Asteroid families: Search of a 12,487-asteroid sample using two different clustering techniques. *Icarus*, 116, 291–314.
- Nesvorný, D. et al. (2015). Asteroid families in the first data release of the Sloan Digital Sky Survey Moving Object Catalog. *AJ*, 150, 48.
- AstDyS-2 catalog: https://newton.spacedys.com/~astdys2/
