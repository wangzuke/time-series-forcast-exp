"""R0710 visualization: MissTSM grouped-query graph-style improvements
(Approach D: observed-data correlation regrouping, grouped_q4_corrobs;
 Approach E: predefined-prior + adaptive-routing fusion, grouped_q4_fuse).

Compares six variants: full / grouped_q4 (sequential) / grouped_q4_corr (0709 A) /
grouped_q4_corrobs (0710 D) / grouped_q4_soft (0709 B) / grouped_q4_fuse (0710 E).

Chart text is kept in English since no CJK font is available in this
environment (consistent with scripts/r0709_visualize.py). Colors follow the
dataviz skill's validated categorical palette (fixed hue order, CVD-checked).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

OUT_DIR = 'docs/figures/r0710'
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ['Weather', 'Electricity', 'Traffic']
MISS_TYPES = ['random_point', 'continuous_segment']
RATES = [0.3, 0.5, 0.7]
SEEDS = [2024, 2025]

VARIANT_PREFIX = {
    'full': 'r4_P1__misstsm_full__{ds}__{mt}_{rate}__h96_s{seed}',
    'grouped_q4': 'r4_P2B__misstsm_gq4__{ds}__{mt}_{rate}__h96_s{seed}',
    'grouped_q4_corr': 'r0709_P1A__misstsm_gq4corr__{ds}__{mt}_{rate}__h96_s{seed}',
    'grouped_q4_corrobs': 'r0710_P1__misstsm_gq4corrobs__{ds}__{mt}_{rate}__h96_s{seed}',
    'grouped_q4_soft': 'r0709_P1B__misstsm_gq4soft__{ds}__{mt}_{rate}__h96_s{seed}',
    'grouped_q4_fuse': 'r0710_P2__misstsm_gq4fuse__{ds}__{mt}_{rate}__h96_s{seed}',
}
VARIANT_LABELS = {
    'full': 'full',
    'grouped_q4': 'grouped_q4(seq)',
    'grouped_q4_corr': 'gq4_corr(A,static)',
    'grouped_q4_corrobs': 'gq4_corrobs(D,observed)',
    'grouped_q4_soft': 'gq4_soft(B,adaptive)',
    'grouped_q4_fuse': 'gq4_fuse(E,fused)',
}
# dataviz skill validated categorical palette, fixed hue order (light mode):
# 1 blue 2 aqua 3 yellow 4 green 5 violet 6 red
VARIANT_COLORS = {
    'full': '#2a78d6',
    'grouped_q4': '#1baf7a',
    'grouped_q4_corr': '#eda100',
    'grouped_q4_corrobs': '#008300',
    'grouped_q4_soft': '#4a3aa7',
    'grouped_q4_fuse': '#e34948',
}
ORDER = ['full', 'grouped_q4', 'grouped_q4_corr', 'grouped_q4_corrobs', 'grouped_q4_soft', 'grouped_q4_fuse']


def load_all():
    data = {}
    for variant, tmpl in VARIANT_PREFIX.items():
        data[variant] = {}
        for ds in DATASETS:
            for mt in MISS_TYPES:
                for rate in RATES:
                    mses, maes, times, nparams = [], [], [], []
                    for seed in SEEDS:
                        tag = tmpl.format(ds=ds, mt=mt, rate=int(rate * 100), seed=seed)
                        path = os.path.join('results', f'{tag}.json')
                        if not os.path.exists(path):
                            continue
                        d = json.load(open(path))
                        mses.append(d['test']['mse'])
                        maes.append(d['test']['mae'])
                        epoch_times = [h['train']['time_sec'] for h in d['history']]
                        times.append(np.mean(epoch_times))
                        nparams.append(d['n_params'])
                    key = (ds, mt, rate)
                    data[variant][key] = {
                        'mse': np.mean(mses) if mses else np.nan,
                        'mae': np.mean(maes) if maes else np.nan,
                        'std': np.std(mses) if mses else np.nan,
                        'time': np.mean(times) if times else np.nan,
                        'nparams': np.mean(nparams) if nparams else np.nan,
                        'n': len(mses),
                    }
    return data


def add_bar_labels(ax, bars, fmt='{:.3f}', fontsize=6, rotation=45):
    for bar in bars:
        h = bar.get_height()
        if h > 0 and not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                    fmt.format(h), ha='center', va='bottom',
                    fontsize=fontsize, rotation=rotation)


def cond_labels():
    conditions = [(mt, r) for mt in MISS_TYPES for r in RATES]
    labels = [f'{"rp" if mt == "random_point" else "cs"}_{int(r*100)}' for mt, r in conditions]
    return conditions, labels


def fig1_six_variant_bars(data):
    conditions, labels = cond_labels()
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        x = np.arange(len(conditions))
        w = 0.13
        for i, variant in enumerate(ORDER):
            vals = [data[variant][(ds, mt, r)]['mse'] for mt, r in conditions]
            bars = ax.bar(x + (i - 2.5) * w, vals, w, label=VARIANT_LABELS[variant],
                          color=VARIANT_COLORS[variant])
            add_bar_labels(ax, bars, fontsize=5)
        ax.set_xlabel('Condition')
        ax.set_ylabel('Test MSE')
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=30)
        ax.grid(axis='y', alpha=0.3)
        if di == 0:
            ax.legend(fontsize=7, loc='upper left')
    fig.suptitle('Six-variant MSE comparison: full / grouped_q4(seq) / corr(A) / corrobs(D) / soft(B) / fuse(E)',
                fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig1_six_variant_bars.png'))
    plt.close(fig)
    print('  Saved fig1_six_variant_bars.png')


def fig2_phase1_heatmap(data):
    """Hypothesis D: does observed-data correlation (corrobs) fix static correlation (corr)'s degradation?"""
    conditions, labels = cond_labels()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pairs = [('grouped_q4_corrobs', 'grouped_q4_corr', 'corrobs(D) vs corr(A, static)'),
             ('grouped_q4_corrobs', 'grouped_q4', 'corrobs(D) vs grouped_q4(seq)')]
    for ai, (new_v, base_v, title) in enumerate(pairs):
        ax = axes[ai]
        mat = np.zeros((len(DATASETS), len(conditions)))
        for di, ds in enumerate(DATASETS):
            for ci, (mt, r) in enumerate(conditions):
                base = data[base_v][(ds, mt, r)]['mse']
                new = data[new_v][(ds, mt, r)]['mse']
                mat[di, ci] = (base - new) / base * 100 if base else np.nan
        vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
        im = ax.imshow(mat, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(labels, fontsize=8, rotation=30)
        ax.set_yticks(range(len(DATASETS)))
        ax.set_yticklabels(DATASETS, fontsize=10)
        for di in range(len(DATASETS)):
            for ci in range(len(conditions)):
                ax.text(ci, di, f'{mat[di, ci]:+.1f}', ha='center', va='center',
                        fontsize=8, color='black')
        ax.set_title(f'{title}, improvement %')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Phase 1 (Approach D): positive (green) = improvement, negative (red) = degradation', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig2_phase1_heatmap.png'))
    plt.close(fig)
    print('  Saved fig2_phase1_heatmap.png')


def fig3_phase2_heatmap(data):
    """Hypothesis E: does fuse beat the better of corr/soft (real synergy, not just picking a winner)?"""
    conditions, labels = cond_labels()
    fig, ax = plt.subplots(figsize=(8, 6))
    mat = np.zeros((len(DATASETS), len(conditions)))
    for di, ds in enumerate(DATASETS):
        for ci, (mt, r) in enumerate(conditions):
            corr = data['grouped_q4_corr'][(ds, mt, r)]['mse']
            soft = data['grouped_q4_soft'][(ds, mt, r)]['mse']
            fuse = data['grouped_q4_fuse'][(ds, mt, r)]['mse']
            best = min(corr, soft)
            mat[di, ci] = (best - fuse) / best * 100 if best else np.nan
    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30)
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(DATASETS, fontsize=10)
    for di in range(len(DATASETS)):
        for ci in range(len(conditions)):
            ax.text(ci, di, f'{mat[di, ci]:+.1f}', ha='center', va='center',
                    fontsize=8, color='black')
    ax.set_title('gq4_fuse(E) vs best(corr(A), soft(B)), improvement %')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Phase 2 (Approach E): positive = fusion beats the better single path (real synergy)', fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig3_phase2_heatmap.png'))
    plt.close(fig)
    print('  Saved fig3_phase2_heatmap.png')


def fig4_weather_continuous_zoom(data):
    """Weather continuous_segment: the specific condition where 0709 found degradation."""
    rates_labels = [f'{int(r*100)}%' for r in RATES]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(RATES))
    w = 0.13
    for i, variant in enumerate(ORDER):
        vals = [data[variant][('Weather', 'continuous_segment', r)]['mse'] for r in RATES]
        bars = ax.bar(x + (i - 2.5) * w, vals, w, label=VARIANT_LABELS[variant],
                      color=VARIANT_COLORS[variant])
        add_bar_labels(ax, bars, fontsize=7)
    ax.set_xlabel('Missing rate (continuous_segment)')
    ax.set_ylabel('Test MSE')
    ax.set_title('Weather continuous_segment: was the 0709 degradation fixed by D/E?')
    ax.set_xticks(x)
    ax.set_xticklabels(rates_labels)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig4_weather_continuous_zoom.png'))
    plt.close(fig)
    print('  Saved fig4_weather_continuous_zoom.png')


def fig5_training_time(data):
    conditions, labels = cond_labels()
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        x = np.arange(len(conditions))
        w = 0.2
        pairs = [('grouped_q4_corr', 'grouped_q4_corrobs'), ('grouped_q4_soft', 'grouped_q4_fuse')]
        vals_corr = [data['grouped_q4_corr'][(ds, mt, r)]['time'] for mt, r in conditions]
        vals_corrobs = [data['grouped_q4_corrobs'][(ds, mt, r)]['time'] for mt, r in conditions]
        vals_soft = [data['grouped_q4_soft'][(ds, mt, r)]['time'] for mt, r in conditions]
        vals_fuse = [data['grouped_q4_fuse'][(ds, mt, r)]['time'] for mt, r in conditions]
        ax.bar(x - 1.5*w, vals_corr, w, label='corr(A)', color=VARIANT_COLORS['grouped_q4_corr'])
        ax.bar(x - 0.5*w, vals_corrobs, w, label='corrobs(D)', color=VARIANT_COLORS['grouped_q4_corrobs'])
        ax.bar(x + 0.5*w, vals_soft, w, label='soft(B)', color=VARIANT_COLORS['grouped_q4_soft'])
        ax.bar(x + 1.5*w, vals_fuse, w, label='fuse(E)', color=VARIANT_COLORS['grouped_q4_fuse'])
        ax.set_xlabel('Condition')
        ax.set_ylabel('Mean per-epoch training time (s)')
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=30)
        ax.grid(axis='y', alpha=0.3)
        if di == 0:
            ax.legend(fontsize=8)
    fig.suptitle('Training time: corrobs(D) vs corr(A) should be ~equal; fuse(E) is the costliest (superset of soft(B))', fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig5_training_time.png'))
    plt.close(fig)
    print('  Saved fig5_training_time.png')


def fig6_stability_bars(data):
    conditions, labels = cond_labels()
    fig, axes = plt.subplots(1, 3, figsize=(24, 5.5))
    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        x = np.arange(len(conditions))
        w = 0.13
        for i, variant in enumerate(ORDER):
            vals = [data[variant][(ds, mt, r)]['std'] for mt, r in conditions]
            ax.bar(x + (i - 2.5) * w, vals, w, label=VARIANT_LABELS[variant],
                  color=VARIANT_COLORS[variant])
        ax.set_xlabel('Condition')
        ax.set_ylabel('Cross-seed MSE std')
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=30)
        ax.grid(axis='y', alpha=0.3)
        if di == 0:
            ax.legend(fontsize=7)
    fig.suptitle('MSE standard deviation across 2 seeds (lower = more stable)', fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig6_stability_bars.png'))
    plt.close(fig)
    print('  Saved fig6_stability_bars.png')


if __name__ == '__main__':
    print("Loading 0710 + 0709 + R4 baseline results...")
    data = load_all()

    print("\n[1/6] Six-variant MSE bars")
    fig1_six_variant_bars(data)

    print("\n[2/6] Phase 1 (Approach D) improvement heatmap")
    fig2_phase1_heatmap(data)

    print("\n[3/6] Phase 2 (Approach E) improvement heatmap")
    fig3_phase2_heatmap(data)

    print("\n[4/6] Weather continuous_segment zoom-in")
    fig4_weather_continuous_zoom(data)

    print("\n[5/6] Training time comparison")
    fig5_training_time(data)

    print("\n[6/6] Stability bars")
    fig6_stability_bars(data)

    print(f"\nDone! All figures saved to {OUT_DIR}/")
    print(f"Total files: {len(os.listdir(OUT_DIR))}")
