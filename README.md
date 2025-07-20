# Home Credit – Credit Risk Model Stability  
*Kaggle Code Competition · Feb 5 → May 27 2024*

Predict borrower default **and** keep model performance **stable over time**.  
The contest mirrors a real score‑card lifecycle where **feature drift** can silently erode AUC.

---

## 1 Competition Snapshot
| Item | Detail |
|------|--------|
| **Training rows** | ≈ 1.22 M loans (`train_base`) |
| **Tables** | 100 + relational tables, three historical depths (0·1·2) |
| **Metric** | **Gini Stability** = mean weekly Gini – slope penalty – residual σ |
| **Runtime quota** | ≤ 12 h CPU/GPU · No internet |
| **Prizes** | 1st $25 k · 2nd $20 k · … · Stability track $10 k |

---

## 2 Dataset Overview

### 2.1 Depth Taxonomy  
| Depth | Grain | Example tables | Join keys |
|-------|-------|----------------|-----------|
| **0 – static** | 1 row / case | `static_0_*`, `static_cb_0` | `case_id` |
| **1 – history** | `num_group1` | `applprev_1`, `person_1`, `credit_bureau_*_1` | `case_id`, `num_group1` |
| **2 – deep** | `num_group1` + `num_group2` | `credit_bureau_*_2`, `applprev_2` | `case_id`, `num_group1`, `num_group2` |

Key columns: `date_decision`, `WEEK_NUM`, `MONTH`, and `target` (train only).

---

## 3 Evaluation – Gini Stability  

1. Compute **Gini** for every `WEEK_NUM`.  
2. Fit a regression over weekly Ginis; penalise **negative slope**.  
3. Penalise **residual variance** (oscillation).  

High AUC **and** low drift win.

---

## 4 Timeline
* **Start**  05 Feb 2024  
* **Entry / Team merge**  20 May 2024  
* **Final submit**  27 May 2024 (23:59 UTC)  

---

## 5 Feature‑Engineering Pipeline  

### 5.1 Depth 0 (static)
* Parse `date_decision` plus ~200 *_D columns to epoch‑days.  
* Cast booleans → `Int32`; drop free‑text; one‑hot `education_1103M`, `maritalst_385M`.

### 5.2 Depth 1 (history)
* **Lazy Polars** streams parquet batches (< 4 GB RAM).  
* Per‑table aggregates (mean/sum/min/max/`n_unique`) for amounts, DPDs, flags.  
* Durations such as `approval_to_activation`.

### 5.3 Depth 2 (deep history)
* Two‑stage aggregation `(case_id, num_group1)` → `case_id` for collaterals & payments.  
* Total after join: **453 numeric + categorical** features.

### 5.4 Memory Discipline
* Each depth writes parquet checkpoints to disk and reloads for the next join, staying within Kaggle RAM.

---

## 6 Modelling & Training

| Item | Setting |
|------|---------|
| **Algorithm** | LightGBM (v3.3, CPU) |
| **Objective** | `binary` |
| **Parameters** | `num_leaves 25`, `min_data_in_leaf 100`, `lr 0.10`,<br>`feature_fraction 0.8`, `bagging_fraction 0.7`, `bagging_freq 10`,<br>`reg_alpha 0.1`, `reg_lambda 10`, `n_rounds 100` |
| **CV split** | 80 / 20 chronological |
| **Validation AUC** | **0.8318** |

### 6.1 Inference
* Rebuild test features (Depth 0‑2).  
* **Chunked prediction** in blocks of 25 k rows → keeps < 12 GB RAM.  
* Unseen categorical levels mapped to `"Unknown"`.  
* Submission runtime ≈ 9 h (fits Kaggle limit).

---

## 7 Leaderboard Results

| Notebook (version) | Model | Public LB | Private LB |
|--------------------|-------|-----------|------------|
| **v27 (final)** | LightGBM 453‑feat | **0.2596** | **0.2666** |
| v02 (baseline) | early LightGBM | 0.1973 | 0.2191 |
| v17‑v26 | OOM / runtime failures | — | — |

---

## 8 Key Learnings
* **Polars lazy + parquet** is a lifesaver for multi‑table ETL under tight RAM.  
* Hierarchical aggregation compressed 2 k + raw columns to 453 features with minimal signal loss.  
* Even plain log‑loss objective can score well on Gini Stability if features are robust.  
* Careful categorical alignment & chunked inference avoid memory blow‑ups.

---

## 9 Next Steps
1. Implement custom LightGBM loss combining AUC + stability regulariser.  
2. Target encoding for high‑cardinality masks.  
3. Quarter‑by‑quarter hyper‑parameter retuning to combat drift.  
4. Blend tree and tabular‑NN models for regime diversification.

---

## 10 Acknowledgements
Thanks to **Home Credit** for releasing a production‑scale credit dataset and to the Kaggle community for memory‑saving tricks.

*Authored by **Adomas Fiseris** – last updated 20 Jul 2025*
