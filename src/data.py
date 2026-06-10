"""
data.py — Data loading and preprocessing for the EEG pipeline.

Responsibilities:
    - Load HDF5 MATLAB v7.3 .mat files
    - Load and concatenate all training subjects
    - Load test subjects
    - Bandpass filter (Butterworth)
    - Subject-wise Z-score normalization (signal level & feature level)
"""

import os
import glob
import warnings
import numpy as np
import h5py
from scipy.signal import butter, filtfilt

import config

warnings.filterwarnings('ignore')




def load_hdf5_data(filepath: str) -> dict:
    """
    Load an HDF5-backed MATLAB v7.3 .mat file.

    Returns
    -------
    dict with keys:
        'trial'     : np.ndarray  shape (trials, channels, timepoints)
        'trialinfo' : np.ndarray  label/metadata matrix, or None
        'time'      : np.ndarray  timepoint vector (seconds, >= 0 only)
    """
    def _resolve_field(f, group, name):
        field = group[name]
        if isinstance(field, h5py.Dataset):
            rv = field[()]
            if isinstance(rv, h5py.Reference):
                return f[rv]
            if hasattr(rv, 'shape') and rv.shape == (1, 1):
                ref = rv.item()
                if isinstance(ref, h5py.Reference):
                    return f[ref]
                if isinstance(ref, bytes):
                    ref = ref.decode('utf-8')
                return f[ref]
            return field
        return field

    with h5py.File(filepath, 'r') as f:
        dr = f['data']
        trial_data  = np.array(_resolve_field(f, dr, 'trial')).T

        try:
            trial_info = np.array(_resolve_field(f, dr, 'trialinfo')).T
        except Exception:
            trial_info = None

        time_vec = np.array(_resolve_field(f, dr, 'time')).flatten()

        # Keep only t >= 0 (trim pre-stimulus baseline)
        mask = time_vec >= 0
        if np.any(~mask):
            time_vec   = time_vec[mask]
            trial_data = trial_data[:, :, mask]

    return {'trial': trial_data, 'trialinfo': trial_info, 'time': time_vec}


# ============================================================
# SIGNAL PREPROCESSING
# ============================================================

def bandpass_filter(
    data: np.ndarray,
    lo: float  = config.BANDPASS_LO,
    hi: float  = config.BANDPASS_HI,
    fs: int    = config.FS,
    order: int = config.FILTER_ORDER,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter applied along the last axis.
    Parameters
    ----------
    data : array of any shape; filter applied along axis=-1
    lo, hi : low / high cut frequencies (Hz)
    fs     : sampling frequency (Hz)
    order  : filter order

    Returns
    -------
    Filtered array, same shape as input.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)


def zscore_per_subject(
    data: np.ndarray,
    bounds: list[tuple[int, int]],
) -> np.ndarray:
    """
    Z-score normalise data independently per subject.

    Parameters
    ----------
    data   : np.ndarray  shape (total_trials, ...)
    bounds : list of (start, end) trial index tuples, one per subject

    Returns
    -------
    Z-scored array, same shape as input.
    """
    out = np.zeros_like(data)
    for s, e in bounds:
        chunk = data[s:e]
        mu  = chunk.mean(axis=0)
        sd  = chunk.std(axis=0)
        sd[sd == 0] = 1          # avoid divide-by-zero
        out[s:e] = (chunk - mu) / sd
    return out


# ============================================================
# TRAINING DATA LOADER
# ============================================================

def load_training_data(
    train_path: str = config.TRAIN_PATH,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[str]]:
    """
    Load all training subjects, concatenate neutral + emotional trials.

    Returns
    -------
    X_all     : np.ndarray  (total_trials, channels, timepoints)  raw EEG
    y_all     : np.ndarray  (total_trials,)  labels  0=neutral, 1=emotional
    bounds    : list of (start, end) index pairs — one per subject
    filenames : list of subject file basenames (for reporting)
    """
    neu_dir    = os.path.join(train_path, 'sleep_neu')
    filenames  = sorted(f for f in os.listdir(neu_dir) if f.endswith('.mat'))

    if not filenames:
        raise FileNotFoundError(f"No .mat files found in: {neu_dir}")

    raw_list, label_list, counts = [], [], []

    print(f"Loading {len(filenames)} training subjects from: {train_path}")

    for sf in filenames:
        neu = load_hdf5_data(os.path.join(train_path, 'sleep_neu', sf))
        emo = load_hdf5_data(os.path.join(train_path, 'sleep_emo', sf))

        combined = np.concatenate([neu['trial'], emo['trial']], axis=0)
        labels   = np.concatenate([
            np.zeros(neu['trial'].shape[0], dtype=int),   # neutral  = 0
            np.ones(emo['trial'].shape[0],  dtype=int),   # emotional = 1
        ])

        raw_list.append(combined)
        label_list.append(labels)
        counts.append(len(labels))

        print(f"  {sf}: {combined.shape[0]} trials "
              f"({neu['trial'].shape[0]} neu + {emo['trial'].shape[0]} emo)")

    X_all  = np.concatenate(raw_list,  axis=0)
    y_all  = np.concatenate(label_list, axis=0)

    cumsum = np.concatenate([[0], np.cumsum(counts)])
    bounds = [(int(cumsum[i]), int(cumsum[i + 1])) for i in range(len(counts))]

    print(f"\nTotal training: {len(y_all)} trials | "
          f"{len(counts)} subjects | shape {X_all.shape}")

    return X_all, y_all, bounds, filenames


# ============================================================
# TEST DATA LOADER
# ============================================================

def load_test_data(
    test_path: str = config.TEST_PATH,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Load all test subject .mat files.

    Returns
    -------
    test_list : list of np.ndarray  each (trials, channels, timepoints)
    test_ids  : list of str  subject ID strings (e.g. '1', '7', '12')
    """
    pattern   = os.path.join(test_path, 'test_subject_*.mat')
    flat_files = sorted(glob.glob(pattern))

    if not flat_files:
        raise FileNotFoundError(f"No test files matching: {pattern}")

    test_list, test_ids = [], []

    print(f"Loading {len(flat_files)} test subjects from: {test_path}")

    for fp in flat_files:
        bn  = os.path.basename(fp)
        sid = bn.split('_')[2].split('.')[0]      # e.g. 'test_subject_1.mat' -> '1'
        data = load_hdf5_data(fp)

        test_list.append(data['trial'])
        test_ids.append(sid)

        print(f"  Subject {sid}: {data['trial'].shape[0]} trials | shape {data['trial'].shape}")

    return test_list, test_ids


# ============================================================
# FULL PREPROCESSING PIPELINE (training)
# ============================================================

def preprocess_training(
    X_raw: np.ndarray,
    bounds: list[tuple[int, int]],
) -> np.ndarray:
    """
    Apply bandpass filter + subject-wise Z-score to raw training EEG.

    Parameters
    ----------
    X_raw  : (total_trials, channels, timepoints)
    bounds : subject boundary index pairs

    Returns
    -------
    filt_z : (total_trials, channels, timepoints)  filtered & normalised
    """
    n_trials, n_ch, _ = X_raw.shape
    print(f"Bandpass filtering {n_trials} trials x {n_ch} channels "
          f"({config.BANDPASS_LO}-{config.BANDPASS_HI} Hz)...")

    # Filter every trial-channel signal
    filt = np.zeros_like(X_raw)
    for i in range(n_trials):
        for j in range(n_ch):
            filt[i, j, :] = bandpass_filter(X_raw[i, j, :])

    # Subject-wise Z-score
    print("Z-scoring per subject (signal level)...")
    filt_z = zscore_per_subject(filt, bounds)

    return filt_z


def preprocess_test_subject(test_raw: np.ndarray) -> np.ndarray:
    """
    Apply bandpass filter + Z-score to a single test subject's EEG.
    Z-score is computed within the test subject (no training stats used,
    consistent with cross-subject pipeline).

    Parameters
    ----------
    test_raw : (trials, channels, timepoints)

    Returns
    -------
    filt_z   : (trials, channels, timepoints)
    """
    nt, n_ch, _ = test_raw.shape

    filt = np.zeros_like(test_raw)
    for i in range(nt):
        for j in range(n_ch):
            filt[i, j, :] = bandpass_filter(test_raw[i, j, :])

    # Z-score using the whole test-subject as the reference group
    mu = filt.mean(axis=0)
    sd = filt.std(axis=0)
    sd[sd == 0] = 1
    filt_z = (filt - mu) / sd

    return filt_z


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == '__main__':
    print("=" * 55)
    print("data.py — self-test")
    print("=" * 55)

    # --- Training ---
    X_all, y_all, bounds, filenames = load_training_data()

    print(f"\nX_all shape   : {X_all.shape}")
    print(f"y_all shape   : {y_all.shape}  (unique: {np.unique(y_all)})")
    print(f"Subjects      : {len(bounds)}")
    print(f"Bounds[0]     : {bounds[0]}")
    print(f"Bounds[-1]    : {bounds[-1]}")

    # --- Preprocess ---
    filt_z = preprocess_training(X_all, bounds)
    print(f"\nfilt_z shape  : {filt_z.shape}")
    print(f"filt_z mean   : {filt_z.mean():.6f}  (should be near 0)")
    print(f"filt_z std    : {filt_z.std():.6f}   (should be near 1)")

    # --- Test ---
    print()
    test_list, test_ids = load_test_data()
    for sid, arr in zip(test_ids, test_list):
        pz = preprocess_test_subject(arr)
        print(f"  Test subject {sid}: preprocessed shape {pz.shape}")

    print("\ndata.py self-test PASSED")
