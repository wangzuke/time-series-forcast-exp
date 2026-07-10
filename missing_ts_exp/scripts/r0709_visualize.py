"""R0709 visualization: MissTSM automatic query-grouping
(Approach A: correlation-based grouping, grouped_q4_corr;
 Approach B: learnable soft routing, grouped_q4_soft).

Compares four variants: full / grouped_q4 (sequential slicing, R4 result) /
grouped_q4_corr (Approach A) / grouped_q4_soft (Approach B).

Chart text is kept in English since no CJK font is available in this
environment (consistent with scripts/r4_visualize.py).
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

OUT_DIR = 'figures_r0709'
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ['Weather', 'Electricity', 'Traffic']
MISS_TYPES = ['random_point', 'continuous_segment']
RATES = [0.3, 0.5, 0.7]
SEEDS = [2024, 2025]

VARIANT_PREFIX = {
    'full': 'r4_P1__misstsm_full__{ds}__{mt}_{rate}__h96_s{seed}',
    'grouped_q4': 'r4_P2B__misstsm_gq4__{ds}__{mt}_{rate}__h96_s{seed}',
    'grouped_q4_corr': 'r0709_P1A__misstsm_gq4corr__{ds}__{mt}_{rate}__h96_s{seed}',
    'grouped_q4_soft': 'r0709_P1B__misstsm_gq4soft__{ds}__{mt}_{rate}__h96_s{seed}',
}
VARIANT_LABELS = {
    'full': 'MissTSM(full)',
    'grouped_q4': 'grouped_q4(sequential)',
    'grouped_q4_corr': 'grouped_q4_corr(A)',
    'grouped_q4_soft': 'grouped_q4_soft(B)',
}
VARIANT_COLORS = {
    'full': '#4E79A7',
    'grouped_q4': '#E15759',
    'grouped_q4_corr': '#59A14F',
    'grouped_q4_soft': '#F28E2B',
}


def load_all():
    data = {}
    for variant, tmpl in VARIANT_PREFIX.items():
        data[variant] = {}
        for ds in DATASETS:
            for mt in MISS_TYPES:
                for rate in RATES:
                    mses, maes, times = [], [], []
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
                    key = (ds, mt, rate)
                    data[variant][key] = {
                        'mse': np.mean(mses) if mses else np.nan,
                        'mae': np.mean(maes) if maes else np.nan,
                        'std': np.std(mses) if mses else np.nan,
                        'time': np.mean(times) if times else np.nan,
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


def fig1_four_variant_bars(data):
    conditions, labels = cond_labels()
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        x = np.arange(len(conditions))
        w = 0.2
        for i, variant in enumerate(['full', 'grouped_q4', 'grouped_q4_corr', 'grouped_q4_soft']):
            vals = [data[variant][(ds, mt, r)]['mse'] for mt, r in conditions]
            bars = ax.bar(x + (i - 1.5) * w, vals, w, label=VARIANT_LABELS[variant],
                          color=VARIANT_COLORS[variant])
            add_bar_labels(ax, bars)
        ax.set_xlabel('Condition')
        ax.set_ylabel('Test MSE')
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=30)
        ax.grid(axis='y', alpha=0.3)
        if di == 0:
            ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('Four-variant MSE comparison: full / grouped_q4(sequential) / grouped_q4_corr(A) / grouped_q4_soft(B)',
                fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig1_four_variant_bars.png'))
    plt.close(fig)
    print('  Saved fig1_four_variant_bars.png')


def fig2_improvement_heatmap(data):
    """Improvement-rate heatmap: grouped_q4_corr / grouped_q4_soft vs grouped_q4 (sequential slicing)."""
    conditions, labels = cond_labels()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ai, variant in enumerate(['grouped_q4_corr', 'grouped_q4_soft']):
        ax = axes[ai]
        mat = np.zeros((len(DATASETS), len(conditions)))
        for di, ds in enumerate(DATASETS):
            for ci, (mt, r) in enumerate(conditions):
                base = data['grouped_q4'][(ds, mt, r)]['mse']
                new = data[variant][(ds, mt, r)]['mse']
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
        ax.set_title(f'{VARIANT_LABELS[variant]} vs grouped_q4(sequential), improvement %')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Positive (green) = improvement, negative (red) = degradation', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig2_improvement_heatmap.png'))
    plt.close(fig)
    print('  Saved fig2_improvement_heatmap.png')


def fig3_traffic_zoom(data):
    conditions, labels = cond_labels()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(conditions))
    w = 0.2
    for i, variant in enumerate(['full', 'grouped_q4', 'grouped_q4_corr', 'grouped_q4_soft']):
        vals = [data[variant][('Traffic', mt, r)]['mse'] for mt, r in conditions]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=VARIANT_LABELS[variant],
                      color=VARIANT_COLORS[variant])
        add_bar_labels(ax, bars)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Test MSE')
    ax.set_title('Traffic (862 channels): did grouped_q4_soft fix the sequential-slicing instability?')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig3_traffic_zoom.png'))
    plt.close(fig)
    print('  Saved fig3_traffic_zoom.png')


def fig4_stability_bars(data):
    """Cross-seed MSE std comparison: does grouped_q4_soft introduce extra instability?"""
    conditions, labels = cond_labels()
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        x = np.arange(len(conditions))
        w = 0.2
        for i, variant in enumerate(['full', 'grouped_q4', 'grouped_q4_corr', 'grouped_q4_soft']):
            vals = [data[variant][(ds, mt, r)]['std'] for mt, r in conditions]
            ax.bar(x + (i - 1.5) * w, vals, w, label=VARIANT_LABELS[variant],
                  color=VARIANT_COLORS[variant])
        ax.set_xlabel('Condition')
        ax.set_ylabel('Cross-seed MSE std')
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=30)
        ax.grid(axis='y', alpha=0.3)
        if di == 0:
            ax.legend(fontsize=8)
    fig.suptitle('MSE standard deviation across 2 seeds (lower = more stable)', fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig4_stability_bars.png'))
    plt.close(fig)
    print('  Saved fig4_stability_bars.png')


def fig5_training_time(data):
    conditions, labels = cond_labels()
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for di, ds in enumerate(DATASETS):
        ax = axes[di]
        x = np.arange(len(conditions))
        gq4_t = [data['grouped_q4'][(ds, mt, r)]['time'] for mt, r in conditions]
        soft_t = [data['grouped_q4_soft'][(ds, mt, r)]['time'] for mt, r in conditions]
        w = 0.35
        b1 = ax.bar(x - w/2, gq4_t, w, label='grouped_q4', color=VARIANT_COLORS['grouped_q4'])
        b2 = ax.bar(x + w/2, soft_t, w, label='grouped_q4_soft', color=VARIANT_COLORS['grouped_q4_soft'])
        add_bar_labels(ax, b1, fmt='{:.1f}')
        add_bar_labels(ax, b2, fmt='{:.1f}')
        ax.set_xlabel('Condition')
        ax.set_ylabel('Mean per-epoch training time (s)')
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=30)
        ax.grid(axis='y', alpha=0.3)
        if di == 0:
            ax.legend(fontsize=9)
    fig.suptitle('Training time: grouped_q4_soft routing overhead vs grouped_q4', fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig5_training_time.png'))
    plt.close(fig)
    print('  Saved fig5_training_time.png')


if __name__ == '__main__':
    print("Loading 0709 + R4 baseline results...")
    data = load_all()

    print("\n[1/5] Four-variant MSE bars")
    fig1_four_variant_bars(data)

    print("\n[2/5] Improvement heatmap")
    fig2_improvement_heatmap(data)

    print("\n[3/5] Traffic zoom-in")
    fig3_traffic_zoom(data)

    print("\n[4/5] Stability bars")
    fig4_stability_bars(data)

    print("\n[5/5] Training time comparison")
    fig5_training_time(data)

    print(f"\nDone! All figures saved to {OUT_DIR}/")
    print(f"Total files: {len(os.listdir(OUT_DIR))}")
