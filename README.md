# 🧠 EEG Emotion Classification — 4th Place Solution

> **Competition**: Kaggle EEG Emotion/Neutral Classification  
> **Public Leaderboard AUC**: **0.557** · **Rank**: 🏆 **4th Place**  
> **Approach**: Classical ML — No deep learning required

---

## Overview

This repository contains the **4th-place solution** for the Kaggle EEG Emotion/Neutral Classification Competition.

The pipeline classifies **emotional vs. neutral** memory reactivation states from 16-channel sleep-stage EEG recordings. Despite the extremely weak cross-subject signal (Cohen's d ≈ 0.02 — **10× weaker than "small"**), this approach achieves competitive performance using a **purely classical ML pipeline**:

```
Raw EEG → Theta Bandpass (4-8 Hz) → Z-Score → Covariance [70:130] → Top-75 Features → Shrinkage LDA
```

All deep learning approaches tested (EEGNet, RNN, Transformers, EEGPT) failed to learn useful representations at this signal strength. The key insight: **removing noise beats adding complexity**.

---

## ✨ Key Highlights

| Highlight | Details |
|-----------|---------|
| 🏆 **4th Place** | Among all competition participants |
| 🧪 **340+ experiments** | Systematic exploration across 63 leaderboard submissions |
| 🧬 **Science-backed window** | `[70:130]` (350–650 ms) confirmed by 5+ independent neuroscience studies |
| 🎯 **Feature stability selection** | Only modification that improved both cross-validation and leaderboard simultaneously |
| ⚡ **60-second training** | Full pipeline runs on a standard laptop CPU — no GPU needed |

---

## 📂 Project Structure

```
├── README.md                        ← You are here
├── model_card.md                    ← Model specs, evaluation, ethics
├── requirements.txt                 ← Python dependencies
│
├── _research_architecture/          ← Research documentation
│   ├── 01_pipeline_blueprint.md     ← Full mathematical pipeline walkthrough
│   ├── 02_experiment_log.md         ← 340+ experiments: what worked & what failed
│   └── 03_window_evidence.md        ← Scientific evidence for the [70:130] window
│
├── src/                             ← Source code
│   ├── config.py                    ← Hyperparameters & paths
│   ├── data.py                      ← Data loading, HDF5 parsing, bandpass filtering
│   ├── features.py                  ← Covariance extraction & stability-based selection
│   ├── model.py                     ← LDA training, LOSO validation, inference
│   ├── pipeline.py                  ← Main entrypoint — runs the full pipeline
│   ├── visualization.py             ← Plotting utilities for analysis
│   └── eeg_pipeline.py              ← Original monolithic script (kept for reference)
│
└── Reports/                         ← Auto-generated plots & figures
    ├── 01_label_distribution.png
    ├── 02_signal_preprocessing.png
    ├── 03_mean_covariance.png
    ├── 04_loso_per_subject.png
    ├── 05_feature_stability_scores.png
    ├── 06_confusion_matrix.png
    ├── 07_test_prediction_distribution.png
    ├── 08_score_heatmap.png
    └── eda_*.png                    ← Exploratory data analysis plots
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Dataset Paths

Edit [`src/config.py`](src/config.py) to point to your local dataset:

```python
TRAIN_PATH = os.path.join(BASE_DIR, 'dataset', 'training')
TEST_PATH  = os.path.join(BASE_DIR, 'dataset', 'testing')
```

### 3. Run the Pipeline

```bash
python src/pipeline.py
```

The pipeline will:
1. **Load** training data from HDF5 `.mat` files
2. **Preprocess** — bandpass filter (θ 4–8 Hz) + per-subject z-score
3. **Extract features** — covariance matrices over `[70:130]` window
4. **Select features** — top 75 by combined stability score
5. **Validate** — Leave-One-Subject-Out (LOSO) cross-validation
6. **Train** — final model on all subjects
7. **Generate** — submission CSV + diagnostic plots in `Reports/`

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **LOSO Cross-Validation AUC** | 0.534 |
| **Public Leaderboard AUC** | **0.557** |
| **Competition Rank** | **4th Place** |

---

## 🔬 Pipeline Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Raw EEG    │───▶│  Bandpass    │───▶│  Z-Score     │───▶│  Window      │
│  16ch×200tp │    │  θ 4–8 Hz   │    │  per subject │    │  [70:130]    │
└─────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                  │
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐
│  Broadcast  │◀───│  Shrinkage   │◀───│  Select      │◀───│  Covariance  │
│  200 tp     │    │  LDA         │    │  Top K=75    │    │  16×16 → 136 │
└──────┬──────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │
       ▼
  submission.csv
```

### Why Each Decision Matters

| Decision | Rationale |
|----------|-----------|
| **Theta 4–8 Hz** | Only frequency band with discriminative signal — all others at or below chance |
| **Z-score (signal)** | Makes covariance mathematically equivalent to correlation |
| **Window [70:130]** | Peak reactivation window — backed by 5+ neuroscience studies |
| **Covariance features** | Captures spatial channel synchrony patterns |
| **K=75 selection** | Only modification that improved **both** LOSO and LB |
| **Shrinkage LDA** | Optimal for correlated features with moderate sample-to-feature ratio |
| **Broadcast** | Per-timepoint predictions too weak; single prediction per trial is optimal |

---

## 📖 Deep-Dive Documentation

| Document | Description |
|----------|-------------|
| 📐 [Pipeline Blueprint](_research_architecture/01_pipeline_blueprint.md) | Complete mathematical walkthrough — every step explained with formulas and examples |
| 🧪 [Experiment Log](_research_architecture/02_experiment_log.md) | Full history of 340+ experiments — what worked, what failed, and why |
| 📄 [Window Evidence](_research_architecture/03_window_evidence.md) | Scientific literature review — 5+ papers confirming the [70:130] window |
| 🧠 [Model Card](model_card.md) | Standardized model documentation — training, evaluation, fairness, ethics |

---

## 🛠️ Technical Details

| Property | Value |
|----------|-------|
| **Algorithm** | Linear Discriminant Analysis (LDA) |
| **Solver** | `lsqr` with Ledoit-Wolf automatic shrinkage |
| **Input** | 16-channel EEG @ 200 Hz (`.mat` HDF5 format) |
| **Features** | 75 (selected from 136 covariance upper-triangle features) |
| **Training data** | 14 subjects · 10,209 trials |
| **Test data** | 3 unseen subjects (zero-shot) |
| **Training time** | ~60 seconds (CPU only) |
| **Model size** | ~47 KB |

---

## 📝 License & Citation

This project is open for research and educational purposes. For details on the underlying TMR paradigm and datasets, see the references cited in the [Window Evidence](_research_architecture/03_window_evidence.md) document.
