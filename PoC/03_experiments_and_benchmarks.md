# Experiment Log — 340+ Experiments · 63 Submissions

> **Competition**: Kaggle EEG Emotion/Neutral Classification  
> **Final Result**: Public LB AUC = **0.557** · Rank **🥈 2nd**  
> **Related docs**: [Scientific Papers & Citations](./01_scientific_papers_and_citations.md) · [Pipeline Blueprint](./02_pipeline_blueprint.md)

---

## 1 · Data Analysis Findings

### 1.1 · Signal Characteristics

Two types of signal exist in this dataset — and only one is usable:

| Signal Type | Cohen's d | Strength | Usable? |
|-------------|-----------|----------|---------|
| **Cross-subject** | 0.02–0.06 | Tiny | ✅ Only option |
| Within-subject | 0.3–0.5 | 5–7× stronger | ❌ Direction flips across subjects |

> **Effect size reference**: d = 0.2 → small · d = 0.5 → medium · d = 0.8 → large
>
> Our cross-subject signal is **10× weaker than "small"**.

### 1.2 · Signal Direction Reversal

The within-subject signal flips direction — some subjects show emotional > neutral, others show the opposite:

| Group | Direction | Subjects | Count |
|-------|-----------|----------|-------|
| **A** | emo > neu | S0, S2, S3, S4, S6, S7, S8, S10 | 8 |
| **B** | emo < neu | S1, S5, S9, S11, S12, S13 | 6 |
| **Test** | — | S18 (unknown), S19 (unknown), S20 (unknown) | 3 |

A global model averages opposing directions → near chance. No reliable way to detect groups without labels.

---

## 2 · Frequency Band Analysis

Every major EEG frequency band was tested. **Only theta produces a signal:**

| Band | Frequency | LOSO AUC | Delta vs Chance | Verdict |
|------|-----------|----------|----------------|---------|
| **Theta** | **4–8 Hz** | **0.5250** | **+0.025** | **✅ Winner** |
| Gamma | 30+ Hz | 0.5109 | +0.011 | ❌ Marginal |
| Alpha | 8–12 Hz | 0.5040 | +0.004 | ❌ Noise level |
| Broadband | 1–30 Hz | 0.5005 | ≈ 0 | ❌ Chance |
| Spindle | 12–15 Hz | 0.4973 | −0.003 | ❌ Below chance |
| Delta | 1–4 Hz | 0.4936 | −0.006 | ❌ Below chance |

> **Key finding**: Multi-band stacking **always** hurts performance. More features = more overfitting on this weak signal.

---

## 3 · Time Window Analysis

The covariance window was systematically optimized. Wider windows dilute the signal:

| Window | Time (ms) | Samples | LOSO AUC | LB Score | Notes |
|--------|-----------|---------|----------|----------|-------|
| `[0:200]` | 0–1000 | 200 | 0.5120 | 0.517 | Full window — worst |
| `[50:150]` | 250–750 | 100 | 0.5151 | — | Wide |
| `[60:120]` | 300–600 | 60 | 0.5158 | — | |
| `[65:125]` | 325–625 | 60 | 0.5199 | — | |
| `[70:110]` | 350–550 | 40 | 0.5210 | — | |
| **`[70:130]` ★** | **350–650** | **60** | **0.5218** | **0.551** | **Best overall balance** |
| `[80:120]` | 400–600 | 40 | 0.5264 | — | Best LOSO, but LB worse |
| `[100:120]` | 500–600 | 20 | — | 0.544 | Too narrow |

> **Pattern**: Wider → monotonic decline. The narrowest windows have best LOSO but anti-correlate with LB.

> **Why `[70:130]`?** This window is not arbitrary — it captures the peak TMR reactivation response confirmed by 5+ independent neuroscience studies. See [Scientific Papers & Citations](./01_scientific_papers_and_citations.md).

---

## 4 · Feature Engineering

### 4.1 · Feature Comparison

| Feature Type | LOSO AUC | LB Score | Status |
|-------------|----------|----------|--------|
| **Theta cov K=75 (V8b)** | **0.5335** | **0.557** | **✅ Winner** |
| Theta cov (all 136) | 0.5250 | — | Baseline |
| CSP 12-component | 0.5247 | — | ❌ No improvement |
| Hilbert envelope cov | 0.4998 | — | ❌ Chance level |
| Raw (no filter) cov | 0.5083 | — | ❌ Too noisy |
| CAR + theta cov | 0.5047 | — | ❌ Re-referencing hurts |
| DE / Hjorth / Statistical | 0.49–0.51 | — | ❌ All worse |
| Wavelet / Spectral | 0.49–0.51 | — | ❌ All worse |
| Per-timepoint Hilbert | — | 0.508 | ❌ Too weak per timepoint |
| Sliding-window cov per-tp | — | 0.545 | ❌ Worse than broadcast |

### 4.2 · Feature Selection — K Value

The combined stability score with different K values:

| K (features kept) | Note |
|-------------------|------|
| 50 | Tested |
| **75** | **✅ Best** |
| 100 | Tested |
| 125 | Tested |
| 136 (no selection) | Baseline |

> Feature **removal** is the only modification that improved both LOSO and LB simultaneously.

---

## 5 · Classifier Comparison

### 5.1 · ✅ Approved Classifiers

| Classifier | LOSO AUC | LB Score | Notes |
|------------|----------|----------|-------|
| **Shrinkage LDA (auto) ★** | **0.5218–0.5335** | **0.557** | Ledoit-Wolf automatic shrinkage |
| Logistic Regression | 0.521–0.537 | 0.555 | Equivalent at LB level |
| Ridge Classifier | 0.521–0.537 | 0.555 | Equivalent at LB level |
| Linear SVM | 0.521–0.537 | 0.555 | Equivalent at LB level |

> **Takeaway**: All linear classifiers perform equivalently. Regularization is the key, not the algorithm.

### 5.2 · ❌ Rejected Classifiers

| Classifier | LOSO AUC | Why It Failed |
|------------|----------|---------------|
| Random Forest | < 0.520 | Massive overfitting — tree models memorize training subjects |
| XGBoost | < 0.520 | Same — trees overfit on d ≈ 0.02 signal |
| LightGBM | < 0.520 | Same |
| CatBoost | < 0.520 | Same |
| Gaussian Naive Bayes | 0.505 | Independence assumptions unsuited for covariance features |
| KNN / NearestCentroid | 0.515 | Degrades in high-dimensional spaces |
| DANN / SupCon | 0.509–0.521 | Neural component contributes nothing |
| EEGPT / Attention | 0.509–0.521 | Same |
| EEGNet 1D/2D (CNN) | 0.500–0.521 | Signal too weak for any neural architecture |
| RNN / Attention | 0.500–0.521 | Same |

> **Conclusion**: All deep learning architectures failed. The signal (d ≈ 0.02) is below the threshold where neural networks can learn meaningful representations.

---

## 6 · Complete Dead Ends — Don't Retry

### 6.1 · Preprocessing Methods

| Method | Category | Result | Why It Failed |
|--------|----------|--------|---------------|
| IQR | Outlier Removal | Hurt LB | Removed genuine emotional signal |
| LOF | Outlier Removal | −0.003 LB | Flagged valid high-amplitude spikes |
| Chauvenet | Outlier Removal | Hurt LB | Distribution-based rejection destroys d ≈ 0.02 signal |
| PCA (elbow = 8) | Dimensionality | LB 0.505 | Rotated weak signal onto noise-dominant directions |

### 6.2 · Hilbert Transform Variants

| Variant | LOSO AUC | LB | Why It Failed |
|---------|----------|----|---------------|
| Hilbert Envelope Covariance | 0.4998 | — | Removes phase info; **phase synchrony IS the signal** |
| Per-Timepoint Hilbert + Cluster + PL | 0.5138 | 0.508 | Instantaneous features lack temporal averaging |

### 6.3 · Domain Adaptation Methods

| Method | Result | Verdict |
|--------|--------|---------|
| CORAL | Hurt or neutral | ❌ |
| Subspace Alignment | Hurt or neutral | ❌ |
| Euclidean Alignment | LOSO 0.510 | ❌ |
| Rank normalization | Hurt or neutral | ❌ |
| Procrustes (136 features) | LOSO 0.4928 | ❌ Destroyed signal |

### 6.4 · Other Failed Approaches

| Method | Why It Failed |
|--------|---------------|
| K-Means clustering on test data | Data leakage — won't generalize to new test set |
| Pseudo-labeling on test data | Data leakage |
| Data augmentation (Mixup/TTA) | Bandpass filter kills injected noise |
| Multi-band stacking | More features = more overfitting |
| Ensembles (same or different) | Higher LOSO → worse LB (anti-correlation) |
| FFT / Fourier features | ~0.499 — spectral features failed as model input |

---

## 7 · The LOSO Anti-Correlation Problem

> ⚠️ **LOSO cross-validation anti-correlates with leaderboard score** — confirmed **7 times**.
>
> Higher LOSO often means lower LB.

| Attempt | LOSO Δ | LB Δ | Direction |
|---------|--------|------|-----------|
| Correlation features blend | +0.009 | **−0.021** | 📉 Anti-correlated |
| CSP + phase ensemble | +0.005 | **−0.021** | 📉 Anti-correlated |
| Pipeline mods ensemble | +0.009 | **−0.019** | 📉 Anti-correlated |
| LOF outlier removal | +0.006 | **−0.003** | 📉 Anti-correlated |
| **V8b feature removal** | **+0.012** | **+0.006** | ✅ **Only exception** |

> **Key insight**: Feature **removal** (not addition) is the only way to break the anti-correlation. Adding features or complexity always makes LB worse even when LOSO improves.

---

## 8 · Post-Processing Experiments

| Method | LB Score | Status |
|--------|----------|--------|
| V8b standalone | 0.557 | ✅ Valid |
| V8b + cluster blend | 0.562 | ⚠️ Validity concern (test data dependency) |
| V8b + cluster blend + pseudo-labeling | 0.561 | ❌ Data leakage |

---

## 9 · Validity Rules (Non-Negotiable)

These rules ensure the pipeline generalizes to completely unseen subjects (zero-shot):

### ✅ Must

- Each trial's prediction computed **independently**
- LOSO is the **only** valid validation
- Feature selection done on **training data only**
- Validate on **full data** (never sample — 6 trials/subject = overfitting trap)

### ❌ Must Not

- No clustering on test data (creates inter-trial dependencies)
- No pseudo-labeling on test data (data leakage)
- No hyperparameter tuning on test AUC
- No cross-trial feature pooling

> **Generalizability test**: Could the instructor run this on a new test set (different subjects) and get a similar score?

---

## 10 · Key Conclusions

After 340+ LOSO experiments and 63 LB submissions, the conclusions are:

1. **Only useful feature**: Theta-band (4–8 Hz) spatial covariance over `[70:130]` timepoints
2. **Only useful preprocessing**: Bandpass → z-score → covariance → feature selection (K=75)
3. **Only useful classifier**: Any linear classifier with regularization (LDA ≈ LogReg ≈ Ridge)
4. **All deep learning failed**: Signal d ≈ 0.02 is below the neural network threshold
5. **Feature removal > feature addition**: The only path to improvement is removing noise, not adding signal
6. **LOSO anti-correlates with LB**: Cannot blindly optimize LOSO
7. **Only remaining path**: Hyperalignment on V8b's 75 features (expected gain: +0.005–0.010, < 10% chance of reaching 0.57)
