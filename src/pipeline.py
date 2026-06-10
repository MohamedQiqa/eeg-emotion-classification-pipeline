"""
pipeline.py — End-to-end EEG pipeline orchestrator.

This file owns ZERO logic. It coordinates the other modules in order:
    config  → data → features → model → visualization

Run:
    ml_env\\Scripts\\python.exe pipeline_folder/pipeline.py
"""

import os
import time
import config
from data          import load_training_data, preprocess_training, load_test_data
from features      import build_training_features
from model         import run_loso_validation, train_final_model, build_submission
from visualization import save_all_plots


def run_pipeline() -> None:
    t0 = time.time()

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  EEG PIPELINE  —  Theta cov + combined-score K=75)")
    print("=" * 60)
    config.summary()

    # ── Step 1: Load training data ────────────────────────────────────────────
    print("\n[1/7] Loading training data...")
    X_all, y_all, bounds, filenames = load_training_data(config.TRAIN_PATH)

    # ── Step 2: Preprocess (bandpass + Z-score) ───────────────────────────────
    print("\n[2/7] Preprocessing (bandpass + Z-score)...")
    filt_z = preprocess_training(X_all, bounds)

    # ── Step 3: Feature engineering ───────────────────────────────────────────
    print("\n[3/7] Extracting & selecting features...")
    feats_sel, best_mask, feats_all, combined_score = build_training_features(
        filt_z, y_all, bounds, k=config.BEST_K
    )

    # ── Step 4: LOSO validation ───────────────────────────────────────────────
    print("\n[4/7] Running LOSO validation...")
    preds, per_subj_auc, global_auc = run_loso_validation(
        feats_sel, y_all, bounds, filenames
    )

    # ── Step 5: Train final model ─────────────────────────────────────────────
    print("\n[5/7] Training final model on all subjects...")
    final_model = train_final_model(feats_sel, y_all)

    # ── Step 6: Load test data + build submission ─────────────────────────────
    print("\n[6/7] Loading test data & generating submission...")
    test_list, test_ids = load_test_data(config.TEST_PATH)
    submission_df = build_submission(final_model, test_list, test_ids, best_mask)

    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"  Submission saved: {config.SUBMISSION_PATH}  "
          f"({len(submission_df):,} rows)")

    # ── Step 7: Visualisation ─────────────────────────────────────────────────
    print("\n[7/7] Generating plots...")
    save_all_plots(
        X_all, filt_z, y_all, bounds, filenames,
        preds, per_subj_auc, global_auc,
        combined_score, best_mask,
        submission_df,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  LOSO AUC        : {global_auc:.4f}")
    print(f"  Per-subj mean   : {sum(per_subj_auc)/len(per_subj_auc):.4f}  "
          f"(14 subjects)")
    print(f"  Submission rows : {len(submission_df):,}")
    print(f"  Submission path : {config.SUBMISSION_PATH}")
    print(f"  Reports dir     : {config.REPORTS_DIR}")
    print(f"  Total time      : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    run_pipeline()
