# 🧠 Model Card — EEG Emotion Classifier

> **Algorithm**: Shrinkage LDA · **AUC**: 0.557 (Public LB) · **Rank**: 🥈 2nd Place  
> **Task**: Binary classification — Emotional vs. Neutral memory reactivation from sleep EEG  
> **Related docs**: [Pipeline Blueprint](_research_architecture/01_pipeline_blueprint.md) · [Experiment Log](_research_architecture/02_experiment_log.md) · [Window Evidence](_research_architecture/03_window_evidence.md)

---

## 1 · Model Summary

This model classifies **emotional vs. neutral** states from 16-channel sleep-stage EEG recordings. It uses a **Linear Discriminant Analysis (LDA)** classifier with Ledoit-Wolf automatic shrinkage, trained on covariance-based features extracted from the **theta frequency band (4–8 Hz)**.

The model achieves a **public leaderboard AUC of 0.557**, placing **🥈 2nd** in the Kaggle EEG Emotion/Neutral classification competition — using a purely classical ML approach with **zero deep learning**.

### Why This Approach Works

The cross-subject emotional signal in sleep EEG is extraordinarily weak (Cohen's d ≈ 0.02–0.06 — **10× weaker than "small"**). All deep learning architectures tested (EEGNet, RNNs, Transformers, EEGPT) failed to learn useful representations at this signal strength. The winning strategy was:

1. **Isolate the right signal** — theta band only (4–8 Hz)
2. **Focus on the right window** — 350–650 ms post-cue (peak memory reactivation)
3. **Use the right representation** — spatial covariance captures channel synchrony
4. **Remove noise, don't add features** — stability-based selection from 136 → 75 features

---

## 2 · Pipeline Overview

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

---

## 3 · Usage

```python
import numpy as np, joblib
from scipy.signal import butter, filtfilt

# Load model artifacts
lda_model = joblib.load('lda_model.pkl')
best_mask = np.load('best_mask.npy')

# Bandpass filter helper (theta: 4-8 Hz)
def bandpass(data, lo, hi, fs=200, order=4):
    b, a = butter(order, [lo/(0.5*fs), hi/(0.5*fs)], btype='band')
    return filtfilt(b, a, data, axis=-1)

# Input: raw EEG array of shape (n_trials, 16, n_timepoints) at 200 Hz
# 1. Bandpass filter to theta (4-8 Hz)
# 2. Z-score normalize per subject
# 3. Extract covariance features from window [70:130] (350-650ms)
# 4. Z-score covariance features per subject
# 5. Apply best_mask to select 75 features
# 6. Predict

predictions = lda_model.predict_proba(selected_features)[:, 1]
# Output: probability scores (0 to 1), higher = emotional
```

| Property | Value |
|----------|-------|
| **Input shape** | `(n_trials, 16, n_timepoints)` — 16 EEG channels at 200 Hz |
| **Output shape** | `(n_trials,)` — probability scores between 0 (neutral) and 1 (emotional) |

> **⚠️ Known limitation:** The model expects the exact same preprocessing pipeline (theta bandpass → z-score → covariance → feature masking). Feeding raw or differently preprocessed EEG data will produce meaningless results.

---

## 4 · System Requirements

### Runtime Dependencies

This is a **standalone model** — no external APIs or downstream services required.

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8+ |
| **Core packages** | NumPy, SciPy, scikit-learn, h5py, pandas |
| **Input format** | Raw EEG in `.mat` (HDF5), 16 channels @ 200 Hz |
| **Minimum segment** | 0–1000 ms post-stimulus (model uses the 350–650 ms window) |

### Compute Requirements

| Metric | Value |
|--------|-------|
| Training time | ~60 seconds on a single CPU core |
| Inference time | <1 second per subject (~40 trials) |
| Memory | ~500 MB RAM |
| GPU/TPU | **Not required** |

---

## 5 · Model Characteristics

### Architecture

| Property | Value |
|----------|-------|
| Algorithm | Linear Discriminant Analysis (LDA) |
| Solver | `lsqr` with automatic Ledoit-Wolf shrinkage |
| Total input features | 136 (upper triangle of 16×16 covariance matrix) |
| Selected features | 75 (after stability-based feature selection) |
| Model file size | ~47 KB (`lda_model.pkl`) |
| Feature mask size | <1 KB (`best_mask.npy`) |
| Number of layers | 1 (linear classifier) |
| Inference latency | <10 ms per subject |

### Training Details

- **Trained from scratch** — no pre-trained weights or transfer learning
- **Not pruned** and **not quantized** — simple linear classifier with minimal parameters
- **No differential privacy** techniques applied
- **Regularization**: Automatic Ledoit-Wolf shrinkage on the within-class covariance estimate

### Feature Selection — Combined Stability Score

The key innovation is a **stability-based feature selection** method that was the **only modification to improve both LOSO and LB simultaneously**:

```
score = sign_agreement × mean_abs_coef / (coef_variation + ε)
```

| Component | What It Measures |
|-----------|-----------------|
| **Sign agreement** | Do all LOSO folds agree on the feature's direction? |
| **Mean abs coefficient** | How much does LDA rely on this feature? |
| **Coefficient of variation** | How much does the weight fluctuate across folds? |

Top K=75 features are kept; 61 unstable features are removed. Full details in the [Pipeline Blueprint](_research_architecture/01_pipeline_blueprint.md#step-8--feature-selection--combined-stability-score).

---

## 6 · Data Overview

### Training Data

The training data consists of sleep-stage EEG recordings from **14 subjects** performing an emotional vs. neutral Targeted Memory Reactivation (TMR) task.

| Property | Value |
|----------|-------|
| **Source** | Kaggle competition (HDF5 `.mat` format) |
| **Channels** | 16 EEG channels at 200 Hz |
| **Total trials** | 10,209 (5,038 emotional + 5,171 neutral) |
| **Subjects** | 14 (training) + 3 (test, unseen) |

### Pre-Processing Pipeline

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | Temporal cropping (t ≥ 0) | Retain post-stimulus data only |
| 2 | 4th-order Butterworth (4–8 Hz) | Isolate theta band |
| 3 | Per-subject z-score (signals) | Make covariance ≈ correlation |
| 4 | Covariance over `[70:130]` window | Capture channel synchrony in reactivation window |
| 5 | Upper triangle extraction | 136 features (symmetric matrix → no redundancy) |
| 6 | Stability-based selection | Top 75 features |
| 7 | Per-subject z-score (features) | Scale features for LDA |

### Data Splits

| Split | Details |
|-------|---------|
| **Training** | All subjects from competition training set (`sleep_neu` + `sleep_emo`) |
| **Validation** | Leave-One-Subject-Out (LOSO) across all 14 training subjects |
| **Test** | 3 held-out subjects provided by competition organizers (labels unavailable) |

> **No data leakage**: Z-score normalization is computed **independently per subject**. Feature selection is performed on training data only.

### Demographics

No demographic information (age, gender, ethnicity) was provided or used. Subject identifiers are anonymized numerical IDs.

---

## 7 · Evaluation Results

### Performance Summary

| Metric | Score |
|--------|-------|
| LOSO Cross-Validation AUC | ~0.534 |
| Public Leaderboard AUC | **0.557** |
| Competition Rank | **🥈 2nd** |

The model was evaluated using **Leave-One-Subject-Out (LOSO)** cross-validation — each subject is held out as the test set while all remaining subjects are used for training. This is the gold standard for assessing EEG cross-subject generalization.

### Per-Subject Variability

Per-subject AUC scores vary across individuals due to **high inter-subject variability** in neural signatures. Some subjects are inherently easier to classify based on:

- Strength of their theta-band emotional response
- Whether they belong to the "emo > neu" or "emo < neu" group (signal direction reversal)

### LOSO–Leaderboard Anti-Correlation

> ⚠️ **Important**: LOSO improvements **anti-correlate** with leaderboard performance in 6 out of 7 tested modifications. Only **feature removal** (K=136 → K=75) broke this pattern and improved both simultaneously.

This is documented in detail in the [Experiment Log](_research_architecture/02_experiment_log.md#7--the-loso-anti-correlation-problem).

---

## 8 · Classifier Comparison

### What Works

All linear classifiers perform **equivalently** — regularization matters, algorithm choice does not:

| Classifier | LOSO AUC | LB Score |
|------------|----------|----------|
| **Shrinkage LDA (auto) ★** | **0.5335** | **0.557** |
| Logistic Regression | 0.537 | 0.555 |
| Ridge Classifier | 0.537 | 0.555 |
| Linear SVM | 0.537 | 0.555 |

### What Failed

| Classifier | Why |
|------------|-----|
| Random Forest / XGBoost / LightGBM | Tree models memorize training subjects on d ≈ 0.02 signal |
| EEGNet / CNN / RNN / Transformers | Signal too weak for neural architectures |
| DANN / EEGPT / SupCon | Neural components contributed nothing |
| KNN / Gaussian NB | Poor performance in high-dimensional covariance space |

Full analysis of 340+ experiments in the [Experiment Log](_research_architecture/02_experiment_log.md).

---

## 9 · Limitations & Ethics

### Usage Limitations

| Limitation | Detail |
|------------|--------|
| **Domain-specific** | Trained exclusively on sleep-stage EEG data — not valid for motor imagery, P300, or other EEG paradigms |
| **Hardware-specific** | Requires 16-channel EEG at 200 Hz — different setups need pipeline modification |
| **Not for clinical use** | Developed for a research competition, not validated for diagnostics |

### Fairness

Fairness analysis was not formally conducted as no demographic attributes were available. The model treats all subjects equally through **per-subject normalization**, which mitigates individual differences in EEG signal amplitude and baseline activity.

### Ethical Considerations

- Training data was provided by competition organizers under their data use agreement
- No personally identifiable information is contained in the model weights
- **Must not** be used for emotion surveillance, deception detection, or any application infringing on individual privacy or autonomy
- EEG-based emotion classification has known limitations in reliability and generalizability — results should be interpreted with appropriate scientific caution
