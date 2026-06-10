"""
model.py — Model training, LOSO validation, and inference for the EEG pipeline.

Responsibilities:
    - LOSO (Leave-One-Subject-Out) cross-validation
    - Final model training on all subjects
    - Per-subject test prediction (with full preprocessing inline)
    - Submission DataFrame builder (broadcast to all timepoints)
"""

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

import config
from data import preprocess_test_subject
from features import build_test_features


# ============================================================
# LOSO VALIDATION
# ============================================================

def run_loso_validation(
    feats_sel:  np.ndarray,
    y:          np.ndarray,
    bounds:     list[tuple[int, int]],
    filenames:  list[str] | None = None,
    verbose:    bool = True,
) -> tuple[np.ndarray, list[float], float]:
    """
    Leave-One-Subject-Out cross-validation using Shrinkage LDA.

    For each fold:
        - Train on all subjects except fold subject
        - Predict P(emotional) on fold subject
        - Compute fold-level AUC

    Parameters
    ----------
    feats_sel : (total_trials, k)   — selected feature matrix
    y         : (total_trials,)     — labels  0=neutral, 1=emotional
    bounds    : subject boundary pairs
    filenames : optional list of subject filenames (for logging)
    verbose   : print fold-by-fold AUC

    Returns
    -------
    preds        : (total_trials,)  — out-of-fold probability predictions
    per_subj_auc : list of per-fold AUC values
    global_auc   : overall LOSO AUC (all subjects pooled)
    """
    n_trials     = len(y)
    n_subj       = len(bounds)
    preds        = np.zeros(n_trials)
    per_subj_auc = []

    if verbose:
        print(f"Running LOSO validation  ({n_subj} folds)...")

    for si, (s, e) in enumerate(bounds):
        train_idx = np.concatenate([np.arange(0, s), np.arange(e, n_trials)])
        test_idx  = np.arange(s, e)

        clf = LinearDiscriminantAnalysis(
            solver=config.LDA_SOLVER,
            shrinkage=config.LDA_SHRINKAGE,
        )
        clf.fit(feats_sel[train_idx], y[train_idx])
        preds[test_idx] = clf.predict_proba(feats_sel[test_idx])[:, 1]

        fold_auc = roc_auc_score(y[test_idx], preds[test_idx])
        per_subj_auc.append(fold_auc)

        if verbose:
            label = (filenames[si][:10] if filenames else f"S{si:02d}")
            print(f"  [{si+1:02d}/{n_subj}]  {label:<12}  AUC={fold_auc:.4f}")

    global_auc = roc_auc_score(y, preds)

    if verbose:
        print(f"\n  LOSO AUC (pooled)      : {global_auc:.4f}")
        print(f"  Per-subject mean ± std : "
              f"{np.mean(per_subj_auc):.4f} +/- {np.std(per_subj_auc):.4f}")
        print(f"  Per-subject range      : "
              f"[{min(per_subj_auc):.4f}, {max(per_subj_auc):.4f}]")

    return preds, per_subj_auc, global_auc


# ============================================================
# FINAL MODEL TRAINING
# ============================================================

def train_final_model(
    feats_sel: np.ndarray,
    y:         np.ndarray,
    verbose:   bool = True,
) -> LinearDiscriminantAnalysis:
    """
    Train a Shrinkage LDA on the full training set (all 14 subjects).

    Parameters
    ----------
    feats_sel : (total_trials, k)
    y         : (total_trials,)

    Returns
    -------
    Fitted LinearDiscriminantAnalysis model.
    """
    clf = LinearDiscriminantAnalysis(
        solver=config.LDA_SOLVER,
        shrinkage=config.LDA_SHRINKAGE,
    )
    clf.fit(feats_sel, y)

    if verbose:
        print(f"Final model trained on {len(y)} trials  "
              f"| {feats_sel.shape[1]} features  "
              f"| solver={config.LDA_SOLVER}, shrinkage={config.LDA_SHRINKAGE}")

    return clf


# ============================================================
# TEST INFERENCE
# ============================================================

def predict_test_subject(
    model:     LinearDiscriminantAnalysis,
    test_raw:  np.ndarray,
    best_mask: np.ndarray,
    subject_id: str = "?",
    verbose:   bool = True,
) -> np.ndarray:
    """
    Full inference pipeline for one test subject:
        1. Bandpass filter + Z-score (signal level)
        2. Covariance features + Z-score (feature level)
        3. Select training features (best_mask)
        4. Predict P(emotional) per trial

    Parameters
    ----------
    model      : trained LDA model
    test_raw   : (trials, channels, timepoints) — raw test EEG
    best_mask  : (k,) feature indices from training selection
    subject_id : string for logging

    Returns
    -------
    trial_preds : (trials,)  — probability of emotional reactivation per trial
    """
    # Step 1: preprocess (bandpass + Z-score)
    filt_z = preprocess_test_subject(test_raw)

    # Step 2 & 3: extract + select features
    feats_sel = build_test_features(filt_z, best_mask)

    # Step 4: predict
    trial_preds = model.predict_proba(feats_sel)[:, 1]

    if verbose:
        nt = len(trial_preds)
        print(f"  Subject {subject_id}: {nt} trials  "
              f"| mean={trial_preds.mean():.4f}  "
              f"std={trial_preds.std():.4f}  "
              f"range=[{trial_preds.min():.3f}, {trial_preds.max():.3f}]")

    return trial_preds


# ============================================================
# SUBMISSION BUILDER
# ============================================================

def build_submission(
    model:      LinearDiscriminantAnalysis,
    test_list:  list[np.ndarray],
    test_ids:   list[str],
    best_mask:  np.ndarray,
    n_tp:       int = config.N_TIMEPOINTS,
    verbose:    bool = True,
) -> pd.DataFrame:
    """
    Generate the submission DataFrame by broadcasting trial predictions
    across all 200 timepoints.

    Row format:  {subject_id}_{trial_number}_{timepoint}, prediction

    Parameters
    ----------
    model     : trained final LDA
    test_list : list of raw EEG arrays per test subject
    test_ids  : list of subject ID strings
    best_mask : feature indices from training
    n_tp      : number of timepoints to broadcast (default 200)

    Returns
    -------
    pd.DataFrame with columns ['id', 'prediction']
    """
    if verbose:
        print(f"\nBuilding submission  "
              f"({len(test_list)} subjects, broadcasting to {n_tp} timepoints)...")

    rows = []
    for test_raw, sid in zip(test_list, test_ids):
        trial_preds = predict_test_subject(model, test_raw, best_mask, sid, verbose)

        # Broadcast: same prediction at every timepoint
        for tr_idx, prob in enumerate(trial_preds):
            for tp in range(n_tp):
                rows.append({'id': f"{sid}_{tr_idx}_{tp}", 'prediction': prob})

    df = pd.DataFrame(rows)

    if verbose:
        print(f"\n  Submission rows : {len(df):,}")
        print(f"  Prediction mean : {df['prediction'].mean():.4f}")
        print(f"  Prediction std  : {df['prediction'].std():.4f}")
        print(f"  Prediction range: [{df['prediction'].min():.4f}, "
              f"{df['prediction'].max():.4f}]")

    return df


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == '__main__':
    from data     import load_training_data, preprocess_training, load_test_data
    from features import build_training_features

    print("=" * 55)
    print("model.py -- self-test")
    print("=" * 55)

    # --- Data ---
    X_all, y_all, bounds, filenames = load_training_data()
    filt_z = preprocess_training(X_all, bounds)

    # --- Features ---
    print()
    feats_sel, best_mask, _, _ = build_training_features(filt_z, y_all, bounds)

    # --- LOSO ---
    print()
    preds, per_subj_auc, global_auc = run_loso_validation(
        feats_sel, y_all, bounds, filenames
    )

    assert abs(global_auc - 0.53) < 0.05, \
        f"LOSO AUC unexpectedly far from baseline: {global_auc:.4f}"

    # --- Final model ---
    print()
    final_model = train_final_model(feats_sel, y_all)

    # --- Test data & submission ---
    print()
    test_list, test_ids = load_test_data()
    submission_df = build_submission(final_model, test_list, test_ids, best_mask)

    expected_rows = sum(arr.shape[0] for arr in test_list) * config.N_TIMEPOINTS
    assert len(submission_df) == expected_rows, \
        f"Row count mismatch: {len(submission_df)} != {expected_rows}"

    print(f"\nSanity checks passed:")
    print(f"  Global LOSO AUC : {global_auc:.4f}")
    print(f"  Submission rows : {len(submission_df):,}  (expected {expected_rows:,})")
    print(f"  Columns         : {list(submission_df.columns)}")
    print(f"  Sample IDs      : {submission_df['id'].head(3).tolist()}")
    print("\nmodel.py self-test PASSED")
