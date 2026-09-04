"""
visualization.py — All plots for the EEG pipeline.
Saved to reports/ directory. No display needed (headless).

Pipeline plots  (run automatically via save_all_plots):
  01  How many neutral vs emotional trials each subject has
  02  What the signal looks like before and after preprocessing
  03  Average brain connectivity pattern: emotional vs neutral
  04  How well the model did per subject (LOSO AUC)
  05  Which features the model found most stable and useful
  06  Confusion matrix: how often the model predicted correctly
  07  Model confidence distribution on unseen test subjects
  08  Which brain channel pairs carry the most signal

EDA plots  (run separately via save_eda_plots — adapted from _ARCHIVE/key_eda_findings.py):
  eda_01  Average brain response over time: emotional vs neutral
  eda_02  Raw signal amplitude range and quality check
  eda_03  Which frequency bands differ between classes (per channel)
  eda_04  Which band + channel combination differs most
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch

import config

# ── Colours ─────────────────────────────────────────────────────────────────
BG, PANEL  = '#0f1117', '#1a1d27'
BLUE, RED  = '#4f8ef7', '#e06c75'
GREEN, YELLOW, TEAL = '#98c379', '#e5c07b', '#56b6c2'
TEXT, GRID = '#abb2bf', '#2c313c'

CHANNEL_NAMES = ['C3','C4','O1','O2','CP3','F3','F4','CP4',
                 'C5','Cz','C6','CP5','P7','Pz','P8','CP6']

FREQ_BANDS = {
    'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
    'Sigma': (12, 16), 'Beta': (16, 30),
}

plt.rcParams.update({
    'figure.facecolor': BG,   'axes.facecolor': PANEL,
    'axes.edgecolor': GRID,   'axes.labelcolor': TEXT,
    'xtick.color': TEXT,      'ytick.color': TEXT,
    'text.color': TEXT,       'grid.color': GRID,
    'grid.linestyle': '--',   'grid.linewidth': 0.6,
    'font.size': 10,
})
LEGEND = dict(fontsize=9, facecolor=PANEL, edgecolor=GRID)

os.makedirs(config.REPORTS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(config.REPORTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Plot 01 ─────────────────────────────────────────────────────────────────
# How many neutral vs emotional trials each subject has.
# Useful to spot class imbalance (e.g. S_13 has 5x more trials than S_10).

def plot_label_distribution(y_all, bounds, filenames):
    labels     = [f[:8] for f in filenames]
    neu_counts = [np.sum(y_all[s:e] == 0) for s, e in bounds]
    emo_counts = [np.sum(y_all[s:e] == 1) for s, e in bounds]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x, neu_counts, color=BLUE,  label='Neutral',   zorder=3)
    ax.bar(x, emo_counts, color=GREEN, label='Emotional', bottom=neu_counts, zorder=3)

    for i, (n, e) in enumerate(zip(neu_counts, emo_counts)):
        ax.text(i, n / 2,     str(n), ha='center', va='center', fontsize=8, color='white')
        ax.text(i, n + e / 2, str(e), ha='center', va='center', fontsize=8, color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Trial count')
    ax.set_title('Label Distribution per Subject', fontsize=13, fontweight='bold', color='white', pad=12)
    ax.grid(axis='y', zorder=0)
    ax.legend(**LEGEND)
    fig.tight_layout()
    return _save(fig, '01_label_distribution.png')


# ── Plot 02 ─────────────────────────────────────────────────────────────────
# What the signal looks like before and after preprocessing.
# Top: raw EEG with slow drift and noise. Bottom: clean theta oscillations
# after bandpass filtering + z-score. Yellow band = covariance window used.

def plot_signal_preprocessing(X_raw, filt_z, trial_idx=0, ch_idx=0):
    time = np.linspace(0, 1, config.N_TIMEPOINTS)
    ws   = config.WIN_START / config.FS
    we   = config.WIN_END   / config.FS

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)

    axes[0].plot(time, X_raw[trial_idx, ch_idx], color=BLUE, lw=1.2)
    axes[0].set_ylabel('Amplitude (uV)')
    axes[0].set_title(f'Raw EEG  (trial {trial_idx}, {CHANNEL_NAMES[ch_idx]})',
                      color='white', fontsize=11)
    axes[0].grid(True)
    axes[0].axvspan(ws, we, alpha=0.15, color=YELLOW)

    axes[1].plot(time, filt_z[trial_idx, ch_idx], color=GREEN, lw=1.2)
    axes[1].set_ylabel('Z-score')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('After Bandpass (4-8 Hz) + Z-score', color='white', fontsize=11)
    axes[1].axhline(0, color=GRID, lw=0.8)
    axes[1].axvspan(ws, we, alpha=0.15, color=YELLOW, label='Cov window')
    axes[1].grid(True)
    axes[1].legend(**LEGEND)

    fig.suptitle('Effect of Preprocessing', fontsize=13, fontweight='bold', color='white', y=1.01)
    fig.tight_layout()
    return _save(fig, '02_signal_preprocessing.png')


# ── Plot 03 ─────────────────────────────────────────────────────────────────
# Average covariance (brain connectivity) matrix for emotional vs neutral.
# The two matrices look almost identical → the signal difference is very small.
# - CH1 and CH6 (Strong Correlation): Because that square is a deep red, it means the brain activity in the area recorded by Channel 1 is highly synchronized with Channel 6. When one goes up, the other goes up.
# - CH1 and CH15 (Weak Correlation): Because that square is pale/almost white, it means those two areas of the brain are essentially ignoring each other during this specific time window. Their signals are moving independently.
# they are visually identical to the human eye. The differences between the emotional and neutral states are so tiny

def plot_mean_covariance(filt_z, y_all):
    X_win    = filt_z[:, :, config.WIN_START:config.WIN_END]
    win_size = config.WIN_END - config.WIN_START

    def mean_cov(mask):
        X    = X_win[mask]
        covs = np.einsum('ijk,ilk->ijl', X, X) / win_size
        return covs.mean(axis=0)

    cov_neu = mean_cov(y_all == 0)
    cov_emo = mean_cov(y_all == 1)
    vmax    = max(np.abs(cov_neu).max(), np.abs(cov_emo).max())
    labels  = [f'CH{i+1:02d}' for i in range(config.N_CHANNELS)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cov, title in zip(axes, [cov_neu, cov_emo], ['Neutral', 'Emotional']):
        im = ax.imshow(cov, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(config.N_CHANNELS)); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(config.N_CHANNELS)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=12, color='white')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Mean Covariance Matrix — Emotional vs Neutral',
                 fontsize=13, fontweight='bold', color='white')
    fig.tight_layout()
    return _save(fig, '03_mean_covariance.png')


# ── Plot 04 ─────────────────────────────────────────────────────────────────
# LOSO AUC per subject — how well the model predicted each subject
# when that subject was held out of training.
# Green = good (>= 0.53), Blue = near chance, Red = below chance.

def plot_loso_per_subject(per_subj_auc, global_auc, filenames=None):
    from matplotlib.patches import Patch
    from matplotlib.lines   import Line2D

    n      = len(per_subj_auc)
    auc    = np.array(per_subj_auc)
    labels = [f[:8] for f in filenames] if filenames else [f"S{i:02d}" for i in range(n)]
    order  = np.argsort(auc)
    auc_s  = auc[order]
    lab_s  = [labels[i] for i in order]
    colors = [GREEN if v >= 0.53 else BLUE if v >= 0.50 else RED for v in auc_s]

    fig, ax = plt.subplots(figsize=(9, 0.55 * n + 1.2))
    ax.barh(range(n), auc_s, color=colors, height=0.65, zorder=3)

    for i, v in enumerate(auc_s):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9, color=TEXT)

    ax.axvline(0.50,       color=RED,    lw=1.2, ls='--', label='Chance (0.50)')
    ax.axvline(global_auc, color=YELLOW, lw=1.8, ls='-',  label=f'Pooled ({global_auc:.4f})')
    ax.set_yticks(range(n)); ax.set_yticklabels(lab_s, fontsize=9)
    ax.set_xlabel('AUC')
    ax.set_title('LOSO AUC per Subject', fontsize=13, fontweight='bold', color='white', pad=12)
    ax.set_xlim(0.44, max(auc_s) + 0.03)
    ax.grid(axis='x', zorder=0)
    ax.legend(handles=[
        Patch(color=GREEN, label='>= 0.53'),
        Patch(color=BLUE,  label='0.50 - 0.53'),
        Patch(color=RED,   label='< 0.50'),
        Line2D([], [], color=RED,    lw=1.2, ls='--', label='Chance (0.50)'),
        Line2D([], [], color=YELLOW, lw=1.8, ls='-',  label=f'Pooled ({global_auc:.4f})'),
    ], loc='lower right', **LEGEND)
    fig.tight_layout()
    return _save(fig, '04_loso_per_subject.png')


# ── Plot 05 ─────────────────────────────────────────────────────────────────
# Feature stability scores for all 136 covariance features, ranked highest to lowest.
# Teal bars = the top-K features the model actually uses.
# A sharp drop after K shows the selection cut-off is well-placed.

def plot_feature_scores(combined_score, best_mask):
    from matplotlib.patches import Patch
    from matplotlib.lines   import Line2D

    rank     = np.argsort(-combined_score)
    scores   = combined_score[rank]
    k        = len(best_mask)
    mask_set = set(best_mask)
    colors   = [TEAL if r in mask_set else PANEL for r in rank]
    edges    = [TEAL if r in mask_set else BLUE  for r in rank]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(range(len(scores)), scores, color=colors, edgecolor=edges, linewidth=0.5, width=1.0, zorder=3)
    ax.axvline(k - 0.5, color=RED, lw=1.8, ls='--')
    ax.set_xlabel('Feature rank (descending)')
    ax.set_ylabel('Combined stability score')
    ax.set_title('Feature Stability Scores', fontsize=13, fontweight='bold', color='white', pad=12)
    ax.set_xlim(-0.5, len(scores) - 0.5)
    ax.grid(axis='y', zorder=0)
    ax.legend(handles=[
        Patch(color=TEAL,  label=f'Selected (K={k})'),
        Patch(facecolor=PANEL, edgecolor=BLUE, label='Dropped'),
        Line2D([], [], color=RED, lw=1.8, ls='--', label=f'K={k} cut-off'),
    ], **LEGEND)
    fig.tight_layout()
    return _save(fig, '05_feature_stability_scores.png')


# ── Plot 06 ─────────────────────────────────────────────────────────────────
# Confusion matrix from LOSO out-of-fold predictions (threshold = 0.5).
# Shows whether the model is making systematic errors — e.g. predicting
# everything as neutral — vs making balanced mistakes.

def plot_confusion_matrix(y_true, loso_preds):
    y_pred = (loso_preds >= 0.5).astype(int)
    cm     = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=18, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else TEXT)

    ax.set_xticks([0, 1]); ax.set_xticklabels(['Neutral', 'Emotional'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Neutral', 'Emotional'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (LOSO)', fontsize=13, fontweight='bold', color='white', pad=12)
    fig.text(0.5, -0.02, f'Accuracy: {np.trace(cm)/cm.sum():.3f}', ha='center', fontsize=10, color=TEXT)
    fig.tight_layout()
    return _save(fig, '06_confusion_matrix.png')


# ── Plot 07 ─────────────────────────────────────────────────────────────────
# Distribution of model predictions on each test subject.
# A score near 0.5 = unsure (means the model is completely guessing (a coin toss)). 
# A spread toward 0 or 1 = more confident.
# Dashed lines show each subject's mean prediction.
# All three test subjects have almost the exact same distribution curve. The model isn't failing on just one weird subject; it's behaving consistently across all unseen data.

def plot_prediction_distribution(submission_df):
    df = submission_df.copy()
    df['trial']   = df['id'].str.rsplit('_', n=1).str[0]
    df['subject'] = df['id'].str.split('_').str[0]
    df = df.drop_duplicates('trial')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (sid, grp), color in zip(df.groupby('subject'), [BLUE, GREEN, YELLOW]):
        preds = grp['prediction'].values
        ax.hist(preds, bins=35, alpha=0.65, color=color, edgecolor='none',
                density=True, zorder=3, label=f'Subject {sid} (n={len(preds)})')
        ax.axvline(preds.mean(), color=color, lw=1.5, ls='--', alpha=0.9)

    ax.axvline(0.5, color=RED, lw=1.4, label='Chance (0.50)')
    ax.set_xlabel('P(emotional)'); ax.set_ylabel('Density')
    ax.set_title('Test Prediction Distribution', fontsize=13, fontweight='bold', color='white', pad=12)
    ax.grid(axis='y', zorder=0)
    ax.legend(**LEGEND)
    fig.tight_layout()
    return _save(fig, '07_test_prediction_distribution.png')


# ── Plot 08 ─────────────────────────────────────────────────────────────────
# The 136 feature stability scores mapped back onto the 16x16 channel grid.
# Bright (Yellow / Orange) = that channel pair is a reliable emotional signal. ( The model relies heavily on these spots.)
# Dark ( Purple / Black )= noise. The model keeps only the brightest K=75 squares. (means the relationship between those channels is random or varies too wildly from person to person to be useful. The model ignores these to avoid getting confused)

def plot_score_heatmap(combined_score):
    n_ch = config.N_CHANNELS
    triu_r, triu_c = np.triu_indices(n_ch)
    mat = np.full((n_ch, n_ch), np.nan)
    for idx, (r, c) in enumerate(zip(triu_r, triu_c)):
        mat[r, c] = mat[c, r] = combined_score[idx]

    labels = [f'CH{i+1:02d}' for i in range(n_ch)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im   = ax.imshow(mat, cmap='magma', aspect='auto', vmin=0, vmax=np.nanmax(mat))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Stability Score', color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)
    ax.set_xticks(range(n_ch)); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_ch)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title('Channel-Pair Stability Heatmap', fontsize=13, fontweight='bold', color='white', pad=12)
    fig.tight_layout()
    return _save(fig, '08_score_heatmap.png')


# ── Pipeline orchestrator ────────────────────────────────────────────────────

def save_all_plots(X_raw, filt_z, y_all, bounds, filenames,
                   loso_preds, per_subj_auc, global_auc,
                   combined_score, best_mask, submission_df):
    """Run all 8 pipeline plots and save to reports/."""
    print(f"\nSaving plots to: {config.REPORTS_DIR}")
    paths = [
        plot_label_distribution(y_all, bounds, filenames),
        plot_signal_preprocessing(X_raw, filt_z),
        plot_mean_covariance(filt_z, y_all),
        plot_loso_per_subject(per_subj_auc, global_auc, filenames),
        plot_feature_scores(combined_score, best_mask),
        plot_confusion_matrix(y_all, loso_preds),
        plot_prediction_distribution(submission_df),
        plot_score_heatmap(combined_score),
    ]
    print(f"  {len(paths)} plots saved.\n")
    return paths


# =============================================================================
# Run separately: save_eda_plots(X_raw, y_all)
# These work on raw unfiltered data. PSD + band-power are slow (~2-3 min).
# =============================================================================


# ── EDA Plot 01 ──────────────────────────────────────────────────────────────
# Average brain signal over time for all 16 channels.
# Blue = neutral, Red = emotional, shading = uncertainty (±SEM).
# Yellow band = the 300ms covariance window the model uses.
# If the two lines overlap almost perfectly, the time-domain difference is tiny.

def plot_grand_mean_erp(X_raw, y_all):
    from scipy.stats import sem as _sem
    time    = np.linspace(0, 1, config.N_TIMEPOINTS)
    X_neu   = X_raw[y_all == 0]
    X_emo   = X_raw[y_all == 1]
    mean_neu, mean_emo = X_neu.mean(axis=0), X_emo.mean(axis=0)
    sem_neu,  sem_emo  = _sem(X_neu, axis=0), _sem(X_emo, axis=0)
    ws, we  = config.WIN_START / config.FS, config.WIN_END / config.FS

    fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharey=False)
    for ch, ax in enumerate(axes.flatten()):
        mn, me = mean_neu[ch], mean_emo[ch]
        sn, se = sem_neu[ch],  sem_emo[ch]
        ax.plot(time, mn, color=BLUE, lw=1.5, label='Neutral')
        ax.plot(time, me, color=RED,  lw=1.5, label='Emotional')
        ax.fill_between(time, mn - sn, mn + sn, color=BLUE, alpha=0.2)
        ax.fill_between(time, me - se, me + se, color=RED,  alpha=0.2)
        ax.axhline(0, color=GRID, lw=0.5)
        ax.axvspan(ws, we, alpha=0.12, color=YELLOW)
        ax.set_title(CHANNEL_NAMES[ch], fontweight='bold', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True)
        if ch == 0:
            ax.legend(fontsize=7)

    fig.suptitle('Grand Mean ERP — Neutral vs Emotional  (shading=±SEM, yellow=cov window)',
                 fontsize=12, fontweight='bold', color='white')
    fig.tight_layout()
    return _save(fig, 'eda_01_grand_mean_erp.png')


# ── EDA Plot 02 ──────────────────────────────────────────────────────────────
# Raw signal amplitude overview: histogram, per-channel RMS, and box plots.
# If both classes overlap completely → preprocessing needed to separate them.
# RMS bars show which channels have the strongest signal.

def plot_amplitude_distributions(X_raw, y_all):
    X_neu_flat = X_raw[y_all == 0].flatten()
    X_emo_flat = X_raw[y_all == 1].flatten()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    bins = np.linspace(-200, 200, 100)
    axes[0].hist(X_neu_flat, bins=bins, color=BLUE, alpha=0.6, density=True, label='Neutral')
    axes[0].hist(X_emo_flat, bins=bins, color=RED,  alpha=0.6, density=True, label='Emotional')
    axes[0].set_xlabel('Amplitude (uV)'); axes[0].set_ylabel('Density')
    axes[0].set_title('Amplitude Distribution', fontweight='bold', color='white')
    axes[0].legend(fontsize=8); axes[0].grid(True)

    rms_neu = np.sqrt((X_raw[y_all == 0] ** 2).mean(axis=(0, 2)))
    rms_emo = np.sqrt((X_raw[y_all == 1] ** 2).mean(axis=(0, 2)))
    x_pos   = np.arange(config.N_CHANNELS)
    axes[1].bar(x_pos - 0.2, rms_neu, 0.4, color=BLUE, alpha=0.8, label='Neutral')
    axes[1].bar(x_pos + 0.2, rms_emo, 0.4, color=RED,  alpha=0.8, label='Emotional')
    axes[1].set_xticks(x_pos); axes[1].set_xticklabels(CHANNEL_NAMES, rotation=60, fontsize=7)
    axes[1].set_ylabel('RMS (uV)')
    axes[1].set_title('Per-Channel RMS', fontweight='bold', color='white')
    axes[1].legend(fontsize=8); axes[1].grid(True, axis='y')

    bp = axes[2].boxplot(
        [X_neu_flat[::100], X_emo_flat[::100]],
        labels=['Neutral', 'Emotional'],
        patch_artist=True, notch=True,
        medianprops=dict(color='white', lw=2)
    )
    bp['boxes'][0].set_facecolor(BLUE); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(RED);  bp['boxes'][1].set_alpha(0.7)
    axes[2].set_ylabel('Amplitude (uV)')
    axes[2].set_title('Box Plots per Class', fontweight='bold', color='white')
    axes[2].set_ylim([-300, 300]); axes[2].grid(True, axis='y')

    fig.suptitle('Raw EEG Amplitude Distributions (no preprocessing)',
                 fontsize=12, fontweight='bold', color='white')
    fig.tight_layout()
    return _save(fig, 'eda_02_amplitude_distributions.png')


# ── EDA Plot 03 ──────────────────────────────────────────────────────────────
# Power Spectral Density per channel (Welch method, log scale).
# Shows exactly which frequency bands (theta, alpha, etc.) differ between classes.
# Confirms WHY we filter to 4-8 Hz (theta) — that's where the gap is.
# ⚠ Slow: runs Welch on every trial.

def plot_psd_per_channel(X_raw, y_all):
    X_neu    = X_raw[y_all == 0]
    X_emo    = X_raw[y_all == 1]
    nperseg  = min(128, config.N_TIMEPOINTS)
    noverlap = nperseg // 2
    band_colors = ['purple', 'blue', 'green', 'orange', 'red']

    fig, axes = plt.subplots(4, 4, figsize=(18, 12))
    for ch, ax in enumerate(axes.flatten()):
        psd_neu = np.array([welch(t[ch], fs=config.FS, nperseg=nperseg, noverlap=noverlap)[1]
                            for t in X_neu])
        psd_emo = np.array([welch(t[ch], fs=config.FS, nperseg=nperseg, noverlap=noverlap)[1]
                            for t in X_emo])
        f = welch(X_neu[0, ch], fs=config.FS, nperseg=nperseg, noverlap=noverlap)[0]

        mask   = (f >= 1) & (f <= 40)
        f_plot = f[mask]
        mn     = psd_neu.mean(axis=0)[mask]
        me     = psd_emo.mean(axis=0)[mask]

        ax.semilogy(f_plot, mn, color=BLUE, lw=1.5, label='Neutral')
        ax.semilogy(f_plot, me, color=RED,  lw=1.5, label='Emotional')
        ax.fill_between(f_plot, mn * 0.95, mn * 1.05, color=BLUE, alpha=0.15)
        ax.fill_between(f_plot, me * 0.95, me * 1.05, color=RED,  alpha=0.15)

        for (_, (blo, bhi)), bc in zip(FREQ_BANDS.items(), band_colors):
            ax.axvspan(blo, bhi, alpha=0.04, color=bc)

        ax.set_title(CHANNEL_NAMES[ch], fontweight='bold', fontsize=9)
        ax.set_xlim([1, 40]); ax.tick_params(labelsize=7); ax.grid(True)
        if ch == 0:
            ax.legend(fontsize=7)

    fig.suptitle('Power Spectral Density — Neutral vs Emotional  (Welch, log scale)',
                 fontsize=12, fontweight='bold', color='white')
    fig.tight_layout()
    return _save(fig, 'eda_03_psd_per_channel.png')


# ── EDA Plot 04 ──────────────────────────────────────────────────────────────
# Band power summary: 3 rows (Neutral / Emotional / Difference) x 5 bands.
# The Difference row shows WHERE the emotional signal is strongest.
# Red bars = emotional > neutral. Blue = neutral > emotional.
# ⚠ Slow: runs Welch on every trial for every band (~2-3 min).

def plot_band_power_summary(X_raw, y_all):
    X_neu    = X_raw[y_all == 0]
    X_emo    = X_raw[y_all == 1]
    nperseg  = min(128, config.N_TIMEPOINTS)
    noverlap = nperseg // 2
    band_names = list(FREQ_BANDS.keys())
    n_bands    = len(FREQ_BANDS)

    def bandpower_matrix(X_class):
        bp = np.zeros((config.N_CHANNELS, n_bands))
        for ch in range(config.N_CHANNELS):
            for bi, (_, (blo, bhi)) in enumerate(FREQ_BANDS.items()):
                powers = [welch(t[ch], fs=config.FS, nperseg=nperseg, noverlap=noverlap)
                          for t in X_class]
                bp[ch, bi] = np.mean([p[(f >= blo) & (f <= bhi)].mean() for f, p in powers])
        return bp

    print('  Computing band-power matrices (this takes ~2-3 min)...')
    bp_neu = bandpower_matrix(X_neu)
    bp_emo = bandpower_matrix(X_emo)

    x      = np.arange(config.N_CHANNELS)
    fig, axes = plt.subplots(3, n_bands, figsize=(18, 10))

    for bi, bname in enumerate(band_names):
        blo, bhi = FREQ_BANDS[bname]

        axes[0, bi].bar(x, bp_neu[:, bi], color=BLUE, alpha=0.8)
        axes[0, bi].set_title(f'{bname}\n({blo}-{bhi} Hz)', fontsize=9, fontweight='bold', color='white')
        axes[0, bi].set_xticks(x); axes[0, bi].set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=6)
        axes[0, bi].grid(True, axis='y')
        if bi == 0: axes[0, bi].set_ylabel('Power (uV2/Hz)', fontsize=8)

        axes[1, bi].bar(x, bp_emo[:, bi], color=RED, alpha=0.8)
        axes[1, bi].set_xticks(x); axes[1, bi].set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=6)
        axes[1, bi].grid(True, axis='y')
        if bi == 0: axes[1, bi].set_ylabel('Power (uV2/Hz)', fontsize=8)

        diff   = bp_emo[:, bi] - bp_neu[:, bi]
        colors = [RED if d > 0 else BLUE for d in diff]
        axes[2, bi].bar(x, diff, color=colors, alpha=0.8)
        axes[2, bi].axhline(0, color=TEXT, lw=0.8)
        axes[2, bi].set_xticks(x); axes[2, bi].set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=6)
        axes[2, bi].grid(True, axis='y')
        if bi == 0: axes[2, bi].set_ylabel('Delta (Emo-Neu)', fontsize=8)

    for label, row_idx in [('Neutral', 0), ('Emotional', 1), ('Difference', 2)]:
        axes[row_idx, 0].annotate(label, xy=(-0.35, 0.5), xycoords='axes fraction',
                                  fontsize=10, fontweight='bold', color='white',
                                  rotation=90, va='center')

    fig.suptitle('Band Power Summary — Per Channel & Band',
                 fontsize=12, fontweight='bold', color='white')
    fig.tight_layout()
    return _save(fig, 'eda_04_band_power_summary.png')


# ── EDA orchestrator ─────────────────────────────────────────────────────────

def save_eda_plots(X_raw, y_all, skip_slow=False):
    """
    Run all EDA plots and save to reports/.
    skip_slow=True  → only eda_01 and eda_02  (fast, < 5s)
    skip_slow=False → all 4 including PSD and band power (~2-3 min extra)
    """
    print(f'\n[EDA] Saving EDA plots to: {config.REPORTS_DIR}')
    paths = [
        plot_grand_mean_erp(X_raw, y_all),
        plot_amplitude_distributions(X_raw, y_all),
    ]
    if not skip_slow:
        paths.append(plot_psd_per_channel(X_raw, y_all))
        paths.append(plot_band_power_summary(X_raw, y_all))
    else:
        print('  [EDA] PSD + band-power skipped. Pass skip_slow=False to include.')
    print(f'  [EDA] {len(paths)} EDA plots saved.\n')
    return paths


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from data     import load_training_data, preprocess_training, load_test_data
    from features import build_training_features
    from model    import run_loso_validation, train_final_model, build_submission
    
    print("=" * 55)
    print("visualization.py -- self-test")
    print("=" * 55)

    X_all, y_all, bounds, filenames = load_training_data()
    filt_z = preprocess_training(X_all, bounds)

    feats_sel, best_mask, _, combined_score = build_training_features(
        filt_z, y_all, bounds, verbose=False
    )
    loso_preds, per_subj_auc, global_auc = run_loso_validation(
        feats_sel, y_all, bounds, filenames, verbose=False
    )
    final_model = train_final_model(feats_sel, y_all, verbose=False)

    test_list, test_ids = load_test_data()
    submission_df = build_submission(
        final_model, test_list, test_ids, best_mask, verbose=False
    )

    print()
    paths = save_all_plots(
        X_all, filt_z, y_all, bounds, filenames,
        loso_preds, per_subj_auc, global_auc,
        combined_score, best_mask, submission_df,
    )
    eda_paths = save_eda_plots(X_all, y_all, skip_slow=False)

    assert len(paths) == 8
    for p in paths:
        assert os.path.exists(p), f"Missing: {p}"
    
    # EDA plots — X_all and y_all already loaded above
    eda_paths = save_eda_plots(X_all, y_all, skip_slow=False)
    for p in eda_paths:
        assert os.path.exists(p), f"Missing EDA plot: {p}"

    print("visualization.py self-test PASSED")
