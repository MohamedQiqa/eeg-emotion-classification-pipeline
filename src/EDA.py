import numpy as np, h5py, os, glob
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
print("All packages imported successfully")

FS = 200

TRAIN_PATH = './dataset/training'
TEST_PATH = './dataset/testing'


# ============================================================
# DATA LOADING
# ============================================================
def load_hdf5_data(filepath):
    def load_field(f, dr, fn):
        field = dr[fn]
        if isinstance(field, h5py.Dataset):
            rv = field[()]
            if isinstance(rv, h5py.Reference): return f[rv]
            elif hasattr(rv, 'shape') and rv.shape == (1,1):
                ref = rv.item()
                if isinstance(ref, h5py.Reference): return f[ref]
                else:
                    if isinstance(ref, bytes): ref = ref.decode('utf-8')
                    return f[ref]
            else: return field
        else: return field
    with h5py.File(filepath, 'r') as f:
        dr = f['data']
        td = np.array(load_field(f, dr, 'trial')).T
        try: ti = np.array(load_field(f, dr, 'trialinfo')).T
        except: ti = None
        tv = np.array(load_field(f, dr, 'time')).flatten()
        mask = tv >= 0
        if np.any(~mask): tv = tv[mask]; td = td[:,:,mask]
        return {'trial': td, 'trialinfo': ti, 'time': tv}
print("Data loading function defined")

def bandpass(data, lo, hi, fs=200, order=4):
    b, a = butter(order, [lo/(0.5*fs), hi/(0.5*fs)], btype='band')
    return filtfilt(b, a, data, axis=-1)

triu_idx = np.triu_indices(16)

# ============================================================
# LOAD TRAINING DATA
# ============================================================
neu_path = os.path.join(TRAIN_PATH, 'sleep_neu')
train_files = sorted([f for f in os.listdir(neu_path) if f.endswith('.mat')])
print(f"\nLoading {len(train_files)} training subjects...")

train_raw_list, train_labels_list, train_counts = [], [], []
for sf in train_files:
    neu = load_hdf5_data(os.path.join(TRAIN_PATH, 'sleep_neu', sf))
    emo = load_hdf5_data(os.path.join(TRAIN_PATH, 'sleep_emo', sf))
    combined = np.concatenate([neu['trial'], emo['trial']], axis=0)
    labels = np.concatenate([np.zeros(neu['trial'].shape[0], dtype=int),
                              np.ones(emo['trial'].shape[0], dtype=int)])
    train_raw_list.append(combined)
    train_labels_list.append(labels)
    train_counts.append(len(labels))
    print(f"  {sf}: {combined.shape[0]} trials")

X_all = np.concatenate(train_raw_list, axis=0)
y_all = np.concatenate(train_labels_list, axis=0)
bounds = []
cs = np.concatenate([[0], np.cumsum(train_counts)])
for i in range(len(train_counts)):
    bounds.append((int(cs[i]), int(cs[i+1])))
n_subj = len(train_counts)
n_tp = X_all.shape[2]
n_trials = len(y_all)
print(f"Total: {n_trials} trials, {n_subj} subjects, shape {X_all.shape}")

# ============================================================
# STEP 1: THETA BANDPASS FILTER + Z-SCORE
# ============================================================
# bandpass filter 4-8Hz
filt = np.zeros_like(X_all)
for i in range(n_trials):
    for j in range(16):
        filt[i,j,:] = bandpass(X_all[i,j,:], 4, 8)

# Z-score per subject
filt_z = np.zeros_like(filt)
for s, e in bounds:
    d = filt[s:e]; m = d.mean(0); sd = d.std(0); sd[sd==0] = 1
    filt_z[s:e] = (d - m) / sd
# ============================================================
# STEP 2: COVARIANCE FEATURES [70:130]
# ============================================================
ws, we = 70, 130
win_size = we - ws
X_win = filt_z[:, :, ws:we]
covs = np.einsum('ijk,ilk->ijl', X_win, X_win) / win_size
feats_all = covs[:, triu_idx[0], triu_idx[1]].copy()

# Z-score features per subject
for si in range(n_subj):
    s, e = bounds[si]
    d = feats_all[s:e]; mu = d.mean(0); sd = d.std(0); sd[sd==0] = 1
    feats_all[s:e] = (d - mu) / sd
print(f"  Features shape: {feats_all.shape}")

# ============================================================
# STEP 3: FEATURE SELECTION
# ============================================================
coef_matrix = np.zeros((n_subj, 136))
for si in range(n_subj):
    s, e = bounds[si]
    tr = np.concatenate([np.arange(0, s), np.arange(e, n_trials)])
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(feats_all[tr], y_all[tr])
    coef_matrix[si] = clf.coef_[0]

sign_agreement = np.abs(np.sign(coef_matrix).mean(0))
mean_abs_coef = np.mean(np.abs(coef_matrix), axis=0)
coef_cv = np.std(np.abs(coef_matrix), axis=0) / (mean_abs_coef + 1e-10)
combined_score = sign_agreement * mean_abs_coef / (coef_cv + 0.1)
sorted_by_combined = np.argsort(-combined_score)  # descending

# ============================================================
# STEP 4: LOSO VALIDATION
# ============================================================
best_k = 75  # I tried 50, 75, 100, 125, 136 and 75 is the best
best_mask = sorted_by_combined[:best_k]
feats_sel = feats_all[:, best_mask]
preds = np.zeros(n_trials)
per_subj_auc = []
for si in range(n_subj):
    s, e = bounds[si]
    tr = np.concatenate([np.arange(0, s), np.arange(e, n_trials)])
    te = np.arange(s, e)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(feats_sel[tr], y_all[tr])
    preds[te] = clf.predict_proba(feats_sel[te])[:, 1]
    auc_si = roc_auc_score(y_all[te], preds[te])
    per_subj_auc.append(auc_si)
    print(f"  {train_files[si][:6]}: AUC={auc_si:.4f}")

auc = roc_auc_score(y_all, preds)

print(f"\n  LOSO = {auc:.4f}")
print(f"  Per-subject mean: {np.mean(per_subj_auc):.4f} +/- {np.std(per_subj_auc):.4f}")

# ============================================================
# STEP 5: LOAD TEST DATA
# ============================================================
flat_files = sorted(glob.glob(os.path.join(TEST_PATH, 'test_subject_*.mat')))
test_list, test_ids = [], []
for fp in flat_files:
    bn = os.path.basename(fp)
    sid = bn.split('_')[2].split('.')[0]
    data = load_hdf5_data(fp)
    test_list.append(data['trial'])
    test_ids.append(sid)
    print(f"  Subject {sid}: {data['trial'].shape[0]} trials")

# ============================================================
# STEP 6: TRAIN FINAL MODEL + PREDICT TEST
# ============================================================
final_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
final_clf.fit(feats_sel, y_all)
print(f"  Trained on {n_trials} trials, {best_k} features")
lo, hi = 4, 8
submission_rows = []

for si_test in range(len(test_list)):
    sid = test_ids[si_test]
    test_raw = test_list[si_test]
    nt = test_raw.shape[0]

    # Bandpass filter
    test_filt = np.zeros_like(test_raw)
    for i in range(nt):
        for j in range(16):
            test_filt[i,j,:] = bandpass(test_raw[i,j,:], lo, hi)

    # Z-score within test subject
    m = test_filt.mean(0); sd = test_filt.std(0); sd[sd==0] = 1
    test_filt_z = (test_filt - m) / sd

    # Covariance
    X_win_test = test_filt_z[:, :, ws:we]
    covs_test = np.einsum('ijk,ilk->ijl', X_win_test, X_win_test) / win_size
    feats_test = covs_test[:, triu_idx[0], triu_idx[1]]

    # Z-score cov features within test subject
    mu_te = feats_test.mean(0); sd_te = feats_test.std(0); sd_te[sd_te==0] = 1
    feats_test = (feats_test - mu_te) / sd_te

    # Select features
    feats_test_sel = feats_test[:, best_mask]

    # Predict
    trial_preds = final_clf.predict_proba(feats_test_sel)[:, 1]

    print(f"  Subject {sid}: {nt} trials, mean={trial_preds.mean():.4f}, "
          f"std={trial_preds.std():.4f}, range=[{trial_preds.min():.3f}, {trial_preds.max():.3f}]")

    # Broadcast to all timepoints
    for tr in range(nt):
        for tp in range(n_tp):
            submission_rows.append({'id': f"{sid}_{tr}_{tp}", 'prediction': trial_preds[tr]})

# ============================================================
# STEP 7: SAVE SUBMISSION
# ============================================================
submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission.csv', index=False)

print(f"{'='*60}")
print(f"SUBMISSION SAVED: submission.csv")
print(f"  Rows: {len(submission_df)}")
print(f"  Mean: {submission_df['prediction'].mean():.4f}")
print(f"  Std:  {submission_df['prediction'].std():.4f}")
print(f"  Min:  {submission_df['prediction'].min():.4f}")
print(f"  Max:  {submission_df['prediction'].max():.4f}")
print(f"{'='*60}")
print(f"LOSO={auc:.4f}, LB=0.557 (ranked 2nd)")
print(f"{'='*60}")
