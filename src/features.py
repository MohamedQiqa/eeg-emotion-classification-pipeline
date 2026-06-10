"""
features.py — Feature engineering for the EEG pipeline.

Responsibilities:
    - Extract covariance matrix features from the filtered EEG window
    - Z-score normalise features per subject
    - Compute combined stability score across LOSO folds
    - Select top-K features by combined score (V8b method)
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import config


# Upper-triangle indices for a 16×16 covariance matrix → 136 features
TRIU_IDX = np.triu_indices(config.N_CHANNELS)


# ============================================================
# COVARIANCE FEATURE EXTRACTION
# ============================================================

def extract_covariance_features(
    filt_z: np.ndarray,
    win_start: int = config.WIN_START,
    win_end:   int = config.WIN_END,
) -> np.ndarray:
    """
    Extract upper-triangle covariance features from the theta-filtered EEG
    within the defined time window.

    Parameters
    ----------
    filt_z    : np.ndarray  (trials, channels, timepoints) — filtered & Z-scored
    win_start : start index of the covariance window (default 70 → 350 ms)
    win_end   : end index  of the covariance window (default 130 → 650 ms)

    Returns
    -------
    feats : np.ndarray  (trials, 136)  — upper-triangle cov features
    """
    win_size = win_end - win_start
    X_win    = filt_z[:, :, win_start:win_end]          # (T, C, W)

    # Efficient batch covariance:  cov[t] = X_win[t].T @ X_win[t] / W
    # einsum: for each trial i → outer product over window axis k
    covs = np.einsum('ijk,ilk->ijl', X_win, X_win) / win_size  # (T, C, C)

    # Extract upper triangle → 136 features per trial
    feats = covs[:, TRIU_IDX[0], TRIU_IDX[1]].copy()   # (T, 136)

    return feats


def zscore_features_per_subject(
    feats: np.ndarray,
    bounds: list[tuple[int, int]],
) -> np.ndarray:
    """
    Z-score feature matrix independently per subject (feature-level normalisation).

    Parameters
    ----------
    feats  : np.ndarray  (total_trials, n_features)
    bounds : list of (start, end) trial index pairs

    Returns
    -------
    feats_z : np.ndarray  same shape, normalised per subject
    """
    feats_z = np.zeros_like(feats)
    for s, e in bounds:
        chunk      = feats[s:e]
        mu         = chunk.mean(axis=0)
        sd         = chunk.std(axis=0)
        sd[sd == 0] = 1
        feats_z[s:e] = (chunk - mu) / sd
    return feats_z


# ============================================================
# FEATURE SELECTION
# ============================================================

def compute_loso_coef_matrix(
    feats:    np.ndarray,
    y:        np.ndarray,
    bounds:   list[tuple[int, int]],
    n_trials: int,
) -> np.ndarray:
    """
    Run one LDA per LOSO fold on the training split and collect coefficients.

    For each fold the held-out subject is excluded; the LDA is fitted on the
    remaining subjects. This gives us how stable each feature's coefficient is
    across folds — the basis of the combined-score selection.

    Parameters
    ----------
    feats    : (total_trials, n_features)
    y        : (total_trials,)
    bounds   : subject boundary pairs
    n_trials : total number of trials

    Returns
    -------
    coef_matrix : (n_subjects, n_features)  — LDA coefs per fold
    """
    n_subj   = len(bounds)
    n_feats  = feats.shape[1]
    coef_matrix = np.zeros((n_subj, n_feats))

    for si, (s, e) in enumerate(bounds):
        train_idx = np.concatenate([np.arange(0, s), np.arange(e, n_trials)])
        clf = LinearDiscriminantAnalysis(
            solver=config.LDA_SOLVER,
            shrinkage=config.LDA_SHRINKAGE,
        )
        clf.fit(feats[train_idx], y[train_idx])
        coef_matrix[si] = clf.coef_[0]

    return coef_matrix


def compute_combined_score(coef_matrix: np.ndarray) -> np.ndarray:
    """
    Compute combined stability score for each feature.

    Formula:
        sign_agreement = |mean(sign(coef))|   ← consistency of direction
        mean_abs_coef  = mean(|coef|)          ← average magnitude
        coef_cv        = std(|coef|) / (mean(|coef|) + eps)  ← relative variability
        score          = sign_agreement * mean_abs_coef / (coef_cv + epsilon)

    High score → feature has stable sign AND large magnitude AND low variability
    across all LOSO folds. These are the features that generalise.

    Parameters
    ----------
    coef_matrix : (n_subjects, n_features)

    Returns
    -------
    combined_score : (n_features,)
    """
    sign_agreement = np.abs(np.sign(coef_matrix).mean(axis=0))
    mean_abs_coef  = np.mean(np.abs(coef_matrix), axis=0)
    coef_cv        = np.std(np.abs(coef_matrix), axis=0) / (mean_abs_coef + 1e-10)
    combined_score = (
        sign_agreement * mean_abs_coef / (coef_cv + config.SCORE_CV_EPS)
    )
    return combined_score


def select_top_k_features(
    feats:          np.ndarray,
    combined_score: np.ndarray,
    k:              int = config.BEST_K,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select the top-K features by combined stability score.

    Parameters
    ----------
    feats          : (trials, n_features)
    combined_score : (n_features,)
    k              : number of features to keep

    Returns
    -------
    feats_sel  : (trials, k)       — feature-selected matrix
    best_mask  : (k,) int array    — indices of selected features (for test reuse)
    """
    best_mask  = np.argsort(-combined_score)[:k]   # descending → top-k indices
    feats_sel  = feats[:, best_mask]
    return feats_sel, best_mask


# ============================================================
# CONVENIENCE: FULL FEATURE PIPELINE (training)
# ============================================================

def build_training_features(
    filt_z:   np.ndarray,
    y:        np.ndarray,
    bounds:   list[tuple[int, int]],
    k:        int = config.BEST_K,
    verbose:  bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    End-to-end feature pipeline for training data:
        1. Extract covariance features
        2. Z-score per subject
        3. Compute combined-score via LOSO coef matrix
        4. Select top-K features

    Parameters
    ----------
    filt_z  : (trials, channels, timepoints) — preprocessed EEG
    y       : (trials,)
    bounds  : subject boundary pairs
    k       : number of features to select

    Returns
    -------
    feats_sel      : (trials, k)    selected features
    best_mask      : (k,)           selected feature indices
    feats_all      : (trials, 136)  all Z-scored features (for inspection)
    combined_score : (136,)         per-feature stability score
    """
    n_trials = filt_z.shape[0]

    if verbose:
        print(f"Extracting covariance features  "
              f"(window [{config.WIN_START}:{config.WIN_END}])...")
    feats_raw = extract_covariance_features(filt_z)

    if verbose:
        print(f"Z-scoring features per subject  "
              f"({len(bounds)} subjects)...")
    feats_all = zscore_features_per_subject(feats_raw, bounds)

    if verbose:
        print(f"Computing LOSO coefficient matrix  "
              f"({len(bounds)} folds)...")
    coef_matrix = compute_loso_coef_matrix(feats_all, y, bounds, n_trials)

    combined_score = compute_combined_score(coef_matrix)

    if verbose:
        print(f"Selecting top-{k} features by combined score...")
    feats_sel, best_mask = select_top_k_features(feats_all, combined_score, k)

    if verbose:
        print(f"  All features : {feats_all.shape}")
        print(f"  Selected     : {feats_sel.shape}  (indices saved in best_mask)")
        top3 = np.argsort(-combined_score)[:3]
        print(f"  Top-3 feature indices : {top3}  "
              f"scores={combined_score[top3].round(4)}")

    return feats_sel, best_mask, feats_all, combined_score


def build_test_features(
    filt_z_test: np.ndarray,
    best_mask:   np.ndarray,
) -> np.ndarray:
    """
    Apply the same feature pipeline to a single test subject.
    Uses best_mask learned from training — no new selection.

    Parameters
    ----------
    filt_z_test : (trials, channels, timepoints) — preprocessed test EEG
    best_mask   : (k,) indices from training feature selection

    Returns
    -------
    feats_sel : (trials, k)
    """
    feats_raw = extract_covariance_features(filt_z_test)

    # Z-score within the test subject
    mu  = feats_raw.mean(axis=0)
    sd  = feats_raw.std(axis=0)
    sd[sd == 0] = 1
    feats_z = (feats_raw - mu) / sd

    return feats_z[:, best_mask]


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == '__main__':
    from data import load_training_data, preprocess_training

    print("=" * 55)
    print("features.py -- self-test")
    print("=" * 55)

    # --- Load & preprocess ---
    X_all, y_all, bounds, filenames = load_training_data()
    filt_z = preprocess_training(X_all, bounds)

    # --- Feature pipeline ---
    print()
    feats_sel, best_mask, feats_all, combined_score = build_training_features(
        filt_z, y_all, bounds
    )

    # --- Sanity checks ---
    assert feats_all.shape  == (10209, 136),          f"Unexpected feats_all shape: {feats_all.shape}"
    assert feats_sel.shape  == (10209, config.BEST_K), f"Unexpected feats_sel shape: {feats_sel.shape}"
    assert best_mask.shape  == (config.BEST_K,),       f"Unexpected best_mask shape: {best_mask.shape}"
    assert combined_score.shape == (136,),             f"Unexpected score shape: {combined_score.shape}"

    print(f"\nSanity checks passed:")
    print(f"  feats_all      : {feats_all.shape}          (all 136 cov features, Z-scored)")
    print(f"  combined_score : {combined_score.shape}              (one score per feature)")
    print(f"  feats_sel      : {feats_sel.shape}    (top-{config.BEST_K} selected)")
    print(f"  best_mask      : {best_mask.shape}              (feature indices)")
    print(f"  Score range    : [{combined_score.min():.4f}, {combined_score.max():.4f}]")
    print(f"  feats_sel mean : {feats_sel.mean():.6f}  (near 0 expected)")
    print(f"  feats_sel std  : {feats_sel.std():.6f}   (near 1 expected)")

    print("\nfeatures.py self-test PASSED")
