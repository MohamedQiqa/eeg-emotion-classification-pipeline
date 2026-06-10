"""
config.py — Central configuration for the EEG pipeline.
all constants, paths, and hyperparameters live here.
"""

import os

# ============================================================
# PATHS
# ============================================================
# Base directory: this file's directory (pipeline_folder)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, 'dataset', 'training')
TEST_PATH  = os.path.join(BASE_DIR, 'dataset', 'testing')

# Output paths (relative to pipeline_folder)
PIPELINE_DIR  = os.path.dirname(__file__)
REPORTS_DIR   = os.path.join(PIPELINE_DIR, 'Reports')
SUBMISSION_PATH = os.path.join(PIPELINE_DIR, 'submission.csv')

# ============================================================
# SIGNAL PARAMETERS
# ============================================================
FS            = 200          # Sampling frequency (Hz)
N_CHANNELS    = 16           # EEG channels
N_TIMEPOINTS  = 200          # Timepoints per trial (1 second @ 200 Hz)

# Theta bandpass filter
BANDPASS_LO   = 4            # Hz
BANDPASS_HI   = 8            # Hz
FILTER_ORDER  = 4            # Butterworth filter order

# ============================================================
# FEATURE EXTRACTION
# ============================================================
WIN_START     = 70           # Start timepoint of covariance window (350 ms)
WIN_END       = 130          # End timepoint of covariance window   (650 ms)
N_COV_FEATS   = 136          # Upper-triangle features from 16x16 cov matrix

# ============================================================
# FEATURE SELECTION (V8b combined-score)
# ============================================================
BEST_K        = 75           # Top-K features to select (tuned: 50/75/100/125/136)
SCORE_CV_EPS  = 0.1          # Epsilon in denominator of combined score

# ============================================================
# MODEL
# ============================================================
LDA_SOLVER    = 'lsqr'
LDA_SHRINKAGE = 'auto'       # Ledoit-Wolf automatic shrinkage

# ============================================================
# MISC
# ============================================================
RANDOM_SEED   = 42


def summary() -> None:
    """Print a human-readable configuration summary."""
    print("=" * 50)
    print("EEG PIPELINE — CONFIGURATION")
    print("=" * 50)
    print(f"  BASE_DIR      : {BASE_DIR}")
    print(f"  TRAIN_PATH    : {TRAIN_PATH}")
    print(f"  TEST_PATH     : {TEST_PATH}")
    print(f"  REPORTS_DIR   : {REPORTS_DIR}")
    print(f"  SUBMISSION    : {SUBMISSION_PATH}")
    print("-" * 50)
    print(f"  FS            : {FS} Hz")
    print(f"  N_CHANNELS    : {N_CHANNELS}")
    print(f"  N_TIMEPOINTS  : {N_TIMEPOINTS}")
    print(f"  Bandpass      : {BANDPASS_LO}-{BANDPASS_HI} Hz  (order={FILTER_ORDER})")
    print(f"  Window        : [{WIN_START}:{WIN_END}]  -> {WIN_END - WIN_START} samples")
    print(f"  Cov features  : {N_COV_FEATS}")
    print(f"  Selected K    : {BEST_K}")
    print(f"  LDA           : solver={LDA_SOLVER}, shrinkage={LDA_SHRINKAGE}")
    print("=" * 50)


if __name__ == '__main__':
    summary()
