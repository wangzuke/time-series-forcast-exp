"""第三轮实验可视化脚本：生成 0704 实验报告所需的全部图表。"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

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

OUT_DIR = 'figures_r3'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load R3 data ──
r3 = {}
for f in sorted(glob.glob('results/r3_*.json')):
    tag = os.path.basename(f).replace('.json', '')
    d = json.load(open(f))
    cfg = d['config']
    r3[tag] = {
        'group': tag.split('__')[0],
        'method': cfg['method'], 'dataset': cfg['dataset'],
        'missing_type': cfg['missing_type'], 'missing_rate': cfg['missing_rate'],
        'pred_len': cfg['pred_len'], 'seed': cfg['seed'],
        'test_mse': d['test']['mse'], 'test_mae': d['test']['mae'],
        'impute_mse': d['test'].get('impute_mse', 0),
        'n_params': d['n_params'], 'peak_mem_mb': d.get('peak_mem_mb', 0),
        'predictor': cfg.get('predictor', ''),
        'impute': cfg.get('impute', 'none'),
        'misstsm_variant': cfg.get('misstsm_variant', 'full'),
        'mask_aware': cfg.get('mask_aware', 'none'),
    }

# ── Load R2 data (for A/B comparison + trend lines) ──
r2 = {}
for f in sorted(glob.glob('results/r2_*.json')):
    tag = os.path.basename(f).replace('.json', '')
    try:
        d = json.load(open(f))
        cfg = d['config']
        r2[tag] = {
            'group': tag.split('__')[0],
            'method': cfg['method'], 'dataset': cfg['dataset'],
            'missing_type': cfg['missing_type'], 'missing_rate': cfg['missing_rate'],
            'pred_len': cfg['pred_len'], 'seed': cfg['seed'],
            'test_mse': d['test']['mse'], 'test_mae': d['test']['mae'],
            'predictor': cfg.get('predictor', ''),
            'impute': cfg.get('impute', 'none'),
        }
    except:
        pass

# ── Load R1 data (two-stage baselines) ──
r1 = {}
for f in sorted(glob.glob('results/*.json')):
    tag = os.path.basename(f).replace('.json', '')
    if tag.startswith('r2_') or tag.startswith('r3_'):
        continue
    try:
        d = json.load(open(f))
        cfg = d['config']
        if d['test']['mse'] > 10:
            continue
        r1[tag] = {
            'method': cfg['method'], 'dataset': cfg['dataset'],
            'missing_type': cfg['missing_type'], 'missing_rate': cfg['missing_rate'],
            'pred_len': cfg['pred_len'], 'seed': cfg['seed'],
            'test_mse': d['test']['mse'], 'test_mae': d['test']['mae'],
            'predictor': cfg.get('predictor', ''),
            'impute': cfg.get('impute', 'none'),
        }
    except:
        pass

print(f"Loaded: R3={len(r3)}, R2={len(r2)}, R1={len(r1)}")


def avg_by(data, key_fn, val_fn):
    groups = defaultdict(list)
    for v in data.values():
        k = key_fn(v)
        if k is not None:
            groups[k].append(val_fn(v))
    return {k: np.mean(vs) for k, vs in groups.items()}


def add_bar_labels(ax, bars, fmt='{:.3f}', fontsize=7, rotation=0):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                    fmt.format(h), ha='center', va='bottom',
                    fontsize=fontsize, rotation=rotation)


# ===== FIGURE 1: A/B Migration Bias Fix Heatmap =====
def fig1_fix_heatmap():
    datasets = ['ETTh1', 'Weather', 'Electricity']
    methods = ['CoIFNet', 'CRIB']

    # R3 A2 (CoIFNet fixed)
    a2 = {k: v for k, v in r3.items() if v['group'] == 'r3_A2'}
    # R3 B1 (CRIB fixed)
    b1 = {k: v for k, v in r3.items() if v['group'] == 'r3_B1'}
    # R2 old versions
    r2_coifnet = {k: v for k, v in r2.items()
                  if v['group'] == 'r2_a' and v['method'] == 'coifnet'}
    r2_crib = {k: v for k, v in r2.items()
               if v['group'] == 'r2_a' and v['method'] == 'crib'}

    r3_coifnet_avg = avg_by(a2, lambda v: v['dataset'], lambda v: v['test_mse'])
    r3_crib_avg = avg_by(b1, lambda v: v['dataset'], lambda v: v['test_mse'])
    r2_coifnet_avg = avg_by(r2_coifnet, lambda v: v['dataset'], lambda v: v['test_mse'])
    r2_crib_avg = avg_by(r2_crib, lambda v: v['dataset'], lambda v: v['test_mse'])

    improvement = np.full((2, 3), np.nan)
    for j, ds in enumerate(datasets):
        old = r2_coifnet_avg.get(ds, 0)
        new = r3_coifnet_avg.get(ds, 0)
        if old > 0:
            improvement[0, j] = (old - new) / old * 100

        old = r2_crib_avg.get(ds, 0)
        new = r3_crib_avg.get(ds, 0)
        if old > 0:
            improvement[1, j] = (old - new) / old * 100

    fig, ax = plt.subplots(figsize=(9, 4))
    mask = np.isnan(improvement)
    display = np.where(mask, 0, improvement)
    im = ax.imshow(display, cmap='RdYlGn', aspect='auto', vmin=-40, vmax=10)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=12)

    for i in range(2):
        for j in range(3):
            if mask[i, j]:
                ax.text(j, i, 'OOM', ha='center', va='center',
                        fontsize=14, fontweight='bold', color='gray')
            else:
                val = improvement[i, j]
                color = 'white' if abs(val) > 20 else 'black'
                sign = '+' if val > 0 else ''
                ax.text(j, i, f'{sign}{val:.1f}%', ha='center', va='center',
                        color=color, fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label='MSE Change (%): + = improved, - = degraded')
    ax.set_title('Migration Bias Fix Effect (R3 fixed vs R2 old)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig1_fix_heatmap.png')
    plt.close()
    print("  fig1_fix_heatmap.png")


# ===== FIGURE 2: Global Method Ranking =====
def fig2_method_ranking():
    datasets = ['ETTh1', 'Weather', 'Electricity']

    # R1 two-stage baselines
    r1_interp_itrans = avg_by(
        {k: v for k, v in r1.items()
         if v['method'] == 'simple' and v['impute'] == 'linear'
         and v['predictor'] == 'iTransformer'
         and v['dataset'] in datasets
         and v['missing_type'] in ('random_point', 'continuous_segment')
         and v['missing_rate'] in (0.1, 0.3)},
        lambda v: 'all', lambda v: v['test_mse'])

    r1_interp_patch = avg_by(
        {k: v for k, v in r1.items()
         if v['method'] == 'simple' and v['impute'] == 'linear'
         and v['predictor'] == 'PatchTST'
         and v['dataset'] in datasets
         and v['missing_type'] in ('random_point', 'continuous_segment')
         and v['missing_rate'] in (0.1, 0.3)},
        lambda v: 'all', lambda v: v['test_mse'])

    # R2 old versions
    r2_coifnet = avg_by(
        {k: v for k, v in r2.items()
         if v['group'] == 'r2_a' and v['method'] == 'coifnet'
         and v['dataset'] in datasets},
        lambda v: 'all', lambda v: v['test_mse'])

    r2_crib = avg_by(
        {k: v for k, v in r2.items()
         if v['group'] == 'r2_a' and v['method'] == 'crib'
         and v['dataset'] in datasets},
        lambda v: 'all', lambda v: v['test_mse'])

    # R3 fixed
    r3_coifnet = avg_by(
        {k: v for k, v in r3.items()
         if v['group'] == 'r3_A2' and v['dataset'] in datasets},
        lambda v: 'all', lambda v: v['test_mse'])

    r3_crib = avg_by(
        {k: v for k, v in r3.items()
         if v['group'] == 'r3_B1' and v['dataset'] in datasets},
        lambda v: 'all', lambda v: v['test_mse'])

    # R2 MissTSM
    r2_misstsm = avg_by(
        {k: v for k, v in r2.items()
         if v['group'] == 'r2_a' and v['method'] == 'misstsm'
         and v['dataset'] in datasets},
        lambda v: 'all', lambda v: v['test_mse'])

    labels = ['Interp+\nPatchTST', 'Interp+\niTrans',
              'CoIFNet\n(R2 old)', 'CRIB\n(R2 old)',
              'CoIFNet', 'MissTSM', 'CRIB']
    values = [
        r1_interp_patch.get('all', 0),
        r1_interp_itrans.get('all', 0),
        r2_coifnet.get('all', 0),
        r2_crib.get('all', 0),
        r3_coifnet.get('all', 0),
        r2_misstsm.get('all', 0),
        r3_crib.get('all', 0),
    ]
    colors = ['#4a90d9', '#6ab0de',
              '#e8a87c', '#d4a373',
              '#e07a5f', '#c9ada7', '#b5651d']

    sorted_idx = np.argsort(values)
    labels = [labels[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Global Mean MSE (ETTh1 + Weather + Electricity)')
    ax.set_title('Method Ranking After Migration Bias Fix (lower = better)', fontsize=14)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylim(0, max(values) * 1.12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig2_method_ranking.png')
    plt.close()
    print("  fig2_method_ranking.png")


# ===== FIGURE 3: C Group MissTSM Architecture Comparison =====
def fig3_c_group():
    c1 = {k: v for k, v in r3.items() if v['group'] == 'r3_C1'}
    c2 = {k: v for k, v in r3.items() if v['group'] == 'r3_C2'}
    c3 = {k: v for k, v in r3.items() if v['group'] == 'r3_C3'}
    # R2 MissTSM baseline (pred_len=96 only for C group comparison)
    r2_mt = {k: v for k, v in r2.items()
             if v['group'] == 'r2_a' and v['method'] == 'misstsm'
             and v['pred_len'] == 96}

    datasets = ['Weather', 'Electricity', 'ETTh1', 'ExchangeRate']
    variant_data = [
        ('MissTSM baseline', r2_mt, '#4a90d9'),
        ('C1 Cond-Q', c1, '#2ecc71'),
        ('C2 Multi-Q', c2, '#e07a5f'),
        ('C3 Soft-Skip', c3, '#9b59b6'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for di, ds in enumerate(datasets):
        ax = axes[di]
        conditions = []
        for mt in ['random_point', 'continuous_segment']:
            for rate in [0.1, 0.3]:
                mt_short = 'RP' if mt == 'random_point' else 'CS'
                conditions.append((mt, rate, f'{mt_short}\n{int(rate*100)}%'))

        x = np.arange(len(conditions))
        w = 0.2
        n_variants = len(variant_data)

        for vi, (label, data, color) in enumerate(variant_data):
            vals = []
            for mt, rate, _ in conditions:
                subset = [v['test_mse'] for v in data.values()
                          if v['dataset'] == ds and v['missing_type'] == mt
                          and v['missing_rate'] == rate]
                vals.append(np.mean(subset) if subset else 0)

            offset = (vi - n_variants / 2 + 0.5) * w
            bars = ax.bar(x + offset, vals, w, label=label, color=color,
                         edgecolor='black', linewidth=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels([c[2] for c in conditions], fontsize=9)
        ax.set_ylabel('Test MSE')
        ax.set_title(f'{ds} (pred_len=96)')
        if di == 0:
            ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('C Group: MissTSM Architecture Variants Comparison', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig3_c_group.png')
    plt.close()
    print("  fig3_c_group.png")


# ===== FIGURE 4: D Group Mask-Aware Comparison =====
def fig4_d_group():
    d1 = {k: v for k, v in r3.items() if v['group'] == 'r3_D1'}
    d2 = {k: v for k, v in r3.items() if v['group'] == 'r3_D2'}

    datasets = ['ETTh1', 'Weather']
    colors = {'D1 concat': '#3498db', 'D2 add': '#e07a5f'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for di, ds in enumerate(datasets):
        ax = axes[di]
        conditions = []
        for mt in ['random_point', 'continuous_segment']:
            for rate in [0.1, 0.3]:
                for H in [96, 336]:
                    mt_short = 'RP' if mt == 'random_point' else 'CS'
                    conditions.append((mt, rate, H, f'{mt_short} {int(rate*100)}%\nh{H}'))

        x = np.arange(len(conditions))
        w = 0.35

        for vi, (label, data, color) in enumerate([
            ('D1 concat', d1, '#3498db'),
            ('D2 add', d2, '#e07a5f'),
        ]):
            vals = []
            for mt, rate, H, _ in conditions:
                subset = [v['test_mse'] for v in data.values()
                          if v['dataset'] == ds and v['missing_type'] == mt
                          and v['missing_rate'] == rate and v['pred_len'] == H]
                vals.append(np.mean(subset) if subset else 0)

            offset = (vi - 0.5) * w
            bars = ax.bar(x + offset, vals, w, label=label, color=color,
                         edgecolor='black', linewidth=0.4)
            add_bar_labels(ax, bars, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([c[3] for c in conditions], fontsize=8)
        ax.set_ylabel('Test MSE')
        ax.set_title(f'{ds}')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('D Group: Mask-Aware iTransformer (concat vs add)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig4_d_group.png')
    plt.close()
    print("  fig4_d_group.png")


# ===== FIGURE 5: E Group High Missing Rate =====
def fig5_e_group():
    e_data = {k: v for k, v in r3.items() if v['group'] == 'r3_E'}

    # Separate by method from E group tags
    e_interp = {k: v for k, v in e_data.items() if v['method'] == 'simple'}
    e_misstsm = {k: v for k, v in e_data.items() if v['method'] == 'misstsm'}
    e_coifnet = {k: v for k, v in e_data.items() if v['method'] == 'coifnet'}

    datasets = ['ETTh1', 'Weather', 'Electricity', 'ExchangeRate']
    method_groups = [
        ('Interp+iTrans', e_interp, '#4a90d9'),
        ('MissTSM', e_misstsm, '#e07a5f'),
        ('CoIFNet', e_coifnet, '#2ecc71'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for di, ds in enumerate(datasets):
        ax = axes[di]
        conditions = []
        for mt in ['random_point', 'continuous_segment']:
            for rate in [0.5, 0.7]:
                mt_short = 'RP' if mt == 'random_point' else 'CS'
                conditions.append((mt, rate, f'{mt_short}\n{int(rate*100)}%'))

        x = np.arange(len(conditions))
        w = 0.25
        n_methods = len(method_groups)

        for mi, (label, data, color) in enumerate(method_groups):
            vals = []
            for mt, rate, _ in conditions:
                subset = [v['test_mse'] for v in data.values()
                          if v['dataset'] == ds and v['missing_type'] == mt
                          and v['missing_rate'] == rate]
                vals.append(np.mean(subset) if subset else 0)

            offset = (mi - n_methods / 2 + 0.5) * w
            bars = ax.bar(x + offset, vals, w, label=label, color=color,
                         edgecolor='black', linewidth=0.4)
            add_bar_labels(ax, bars, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([c[2] for c in conditions], fontsize=9)
        ax.set_ylabel('Test MSE')
        ax.set_title(f'{ds} (pred_len=96)')
        if di == 0:
            ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('E Group: High Missing Rate (50% & 70%) Method Comparison', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig5_e_group.png')
    plt.close()
    print("  fig5_e_group.png")


# ===== FIGURE 6: Missing Rate Trend (0.1 → 0.7) =====
def fig6_missing_rate_trend():
    methods_map = {
        'Interp+iTrans': lambda v: v.get('method') == 'simple' and v.get('impute') == 'linear' and v.get('predictor') == 'iTransformer',
        'MissTSM': lambda v: v.get('method') == 'misstsm',
        'CoIFNet': lambda v: v.get('method') == 'coifnet',
    }
    colors = {'Interp+iTrans': '#4a90d9', 'MissTSM': '#e07a5f', 'CoIFNet': '#2ecc71'}

    ds_list = ['Weather', 'Electricity']
    mt_list = ['random_point', 'continuous_segment']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for di, ds in enumerate(ds_list):
        for mi, mt in enumerate(mt_list):
            ax = axes[di][mi]
            mt_label = 'Random Point' if mt == 'random_point' else 'Continuous Segment'

            for method_name, filter_fn in methods_map.items():
                rates_vals = {}
                # R1 data (0.1, 0.3) — two-stage
                if method_name == 'Interp+iTrans':
                    for v in r1.values():
                        if filter_fn(v) and v['dataset'] == ds and v['missing_type'] == mt and v['pred_len'] == 96:
                            rates_vals.setdefault(v['missing_rate'], []).append(v['test_mse'])
                # R2 data (0.1, 0.3) — aware methods
                for v in r2.values():
                    if v['group'] == 'r2_a' and filter_fn(v) and v['dataset'] == ds and v['missing_type'] == mt and v['pred_len'] == 96:
                        rates_vals.setdefault(v['missing_rate'], []).append(v['test_mse'])
                # R3 A2 (CoIFNet fixed, 0.1/0.3)
                if method_name == 'CoIFNet':
                    for v in r3.values():
                        if v['group'] == 'r3_A2' and v['dataset'] == ds and v['missing_type'] == mt and v['pred_len'] == 96:
                            rates_vals.setdefault(v['missing_rate'], []).append(v['test_mse'])
                # R3 E data (0.5, 0.7)
                for v in r3.values():
                    if v['group'] == 'r3_E' and filter_fn(v) and v['dataset'] == ds and v['missing_type'] == mt and v['pred_len'] == 96:
                        rates_vals.setdefault(v['missing_rate'], []).append(v['test_mse'])

                if rates_vals:
                    sorted_rates = sorted(rates_vals.keys())
                    means = [np.mean(rates_vals[r]) for r in sorted_rates]
                    ax.plot(sorted_rates, means, 'o-', label=method_name,
                            color=colors[method_name], linewidth=2, markersize=6)

            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('Test MSE')
            ax.set_title(f'{ds} - {mt_label} (pred_len=96)')
            ax.legend(fontsize=9)
            ax.set_xticks([0.1, 0.3, 0.5, 0.7])
            ax.grid(True, alpha=0.3)

    fig.suptitle('Missing Rate Trend: 10% → 70%', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig6_missing_rate_trend.png')
    plt.close()
    print("  fig6_missing_rate_trend.png")


# ===== FIGURE 7: Radar Chart — Low vs High Missing Rate =====
def fig7_radar():
    datasets = ['ETTh1', 'Weather', 'Electricity']

    # Low missing rate (0.1, 0.3) — use R1/R2 data
    methods_low = {
        'Interp+PatchTST': lambda v: v.get('method') == 'simple' and v.get('impute') == 'linear' and v.get('predictor') == 'PatchTST',
        'Interp+iTrans': lambda v: v.get('method') == 'simple' and v.get('impute') == 'linear' and v.get('predictor') == 'iTransformer',
        'CoIFNet (R2)': lambda v: v.get('method') == 'coifnet',
        'MissTSM': lambda v: v.get('method') == 'misstsm',
    }

    # High missing rate (0.5, 0.7) — use R3 E data
    methods_high = {
        'Interp+iTrans': lambda v: v.get('method') == 'simple',
        'CoIFNet': lambda v: v.get('method') == 'coifnet',
        'MissTSM': lambda v: v.get('method') == 'misstsm',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                     subplot_kw=dict(polar=True))
    colors_low = ['#4a90d9', '#6ab0de', '#2ecc71', '#e07a5f']
    colors_high = ['#4a90d9', '#2ecc71', '#e07a5f']

    angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
    angles += angles[:1]

    # Left: low missing rate
    r1r2_combined = {**r1, **{f'r2_{k}': v for k, v in r2.items() if v['group'] == 'r2_a'}}
    low_data = {k: v for k, v in r1r2_combined.items()
                if v['dataset'] in datasets
                and v.get('missing_type') in ('random_point', 'continuous_segment')
                and v.get('missing_rate') in (0.1, 0.3)}

    max_mse_low = 0.4
    for ci, (method_name, filter_fn) in enumerate(methods_low.items()):
        vals = []
        for ds in datasets:
            subset = [v['test_mse'] for v in low_data.values()
                      if filter_fn(v) and v['dataset'] == ds]
            vals.append(np.mean(subset) if subset else max_mse_low)
        vals_inv = [max_mse_low - v for v in vals]
        vals_inv += vals_inv[:1]
        ax1.plot(angles, vals_inv, 'o-', linewidth=2, label=method_name, color=colors_low[ci])
        ax1.fill(angles, vals_inv, alpha=0.08, color=colors_low[ci])

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(datasets)
    ax1.set_title('Low Missing Rate (0.1~0.3)\n(larger area = better)', pad=20, fontsize=12)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

    # Right: high missing rate
    e_data = {k: v for k, v in r3.items()
              if v['group'] == 'r3_E' and v['dataset'] in datasets}

    max_mse_high = 0.5
    for ci, (method_name, filter_fn) in enumerate(methods_high.items()):
        vals = []
        for ds in datasets:
            subset = [v['test_mse'] for v in e_data.values()
                      if filter_fn(v) and v['dataset'] == ds]
            vals.append(np.mean(subset) if subset else max_mse_high)
        vals_inv = [max_mse_high - v for v in vals]
        vals_inv += vals_inv[:1]
        ax2.plot(angles, vals_inv, 'o-', linewidth=2, label=method_name, color=colors_high[ci])
        ax2.fill(angles, vals_inv, alpha=0.08, color=colors_high[ci])

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(datasets)
    ax2.set_title('High Missing Rate (0.5~0.7)\n(larger area = better)', pad=20, fontsize=12)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

    fig.suptitle('Method Comparison: Low vs High Missing Rate', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig7_radar.png')
    plt.close()
    print("  fig7_radar.png")


# ===== FIGURE 8: Comprehensive Heatmap =====
def fig8_comprehensive_heatmap():
    datasets = ['ETTh1', 'Weather', 'Electricity']

    # Methods for low rate
    methods_low = [
        ('Interp+PatchTST', 'r1', lambda v: v.get('method') == 'simple' and v.get('impute') == 'linear' and v.get('predictor') == 'PatchTST'),
        ('Interp+iTrans', 'r1', lambda v: v.get('method') == 'simple' and v.get('impute') == 'linear' and v.get('predictor') == 'iTransformer'),
        ('CoIFNet', 'r3_A2', None),
        ('CRIB', 'r3_B1', None),
        ('MissTSM', 'r2', lambda v: v.get('method') == 'misstsm'),
    ]

    # Methods for high rate
    methods_high = [
        ('Interp+iTrans', 'r3_E', lambda v: v.get('method') == 'simple'),
        ('CoIFNet', 'r3_E', lambda v: v.get('method') == 'coifnet'),
        ('MissTSM', 'r3_E', lambda v: v.get('method') == 'misstsm'),
    ]

    # Build matrix: rows = all unique methods, cols = dataset x rate_regime
    all_methods = ['Interp+PatchTST', 'Interp+iTrans', 'CoIFNet', 'CRIB', 'MissTSM']
    col_labels = []
    for ds in datasets:
        col_labels.append(f'{ds}\nlow (0.1~0.3)')
        col_labels.append(f'{ds}\nhigh (0.5~0.7)')

    matrix = np.full((len(all_methods), len(col_labels)), np.nan)

    # Low rate data
    r1r2_data = {**r1, **{f'r2a_{k}': v for k, v in r2.items() if v['group'] == 'r2_a'}}
    r3_a2 = {k: v for k, v in r3.items() if v['group'] == 'r3_A2'}
    r3_b1 = {k: v for k, v in r3.items() if v['group'] == 'r3_B1'}

    for mi, method_name in enumerate(all_methods):
        for di, ds in enumerate(datasets):
            col_low = di * 2
            col_high = di * 2 + 1

            # Low rate
            if method_name == 'Interp+PatchTST':
                subset = [v['test_mse'] for v in r1.values()
                          if v['method'] == 'simple' and v['impute'] == 'linear'
                          and v['predictor'] == 'PatchTST' and v['dataset'] == ds
                          and v['missing_rate'] in (0.1, 0.3)]
                if subset:
                    matrix[mi, col_low] = np.mean(subset)
            elif method_name == 'Interp+iTrans':
                subset = [v['test_mse'] for v in r1.values()
                          if v['method'] == 'simple' and v['impute'] == 'linear'
                          and v['predictor'] == 'iTransformer' and v['dataset'] == ds
                          and v['missing_rate'] in (0.1, 0.3)]
                if subset:
                    matrix[mi, col_low] = np.mean(subset)
            elif method_name == 'CoIFNet':
                subset = [v['test_mse'] for v in r3_a2.values()
                          if v['dataset'] == ds]
                if subset:
                    matrix[mi, col_low] = np.mean(subset)
            elif method_name == 'CRIB':
                subset = [v['test_mse'] for v in r3_b1.values()
                          if v['dataset'] == ds]
                if subset:
                    matrix[mi, col_low] = np.mean(subset)
            elif method_name == 'MissTSM':
                subset = [v['test_mse'] for v in r2.values()
                          if v['group'] == 'r2_a' and v['method'] == 'misstsm'
                          and v['dataset'] == ds]
                if subset:
                    matrix[mi, col_low] = np.mean(subset)

            # High rate (E group)
            e_data = {k: v for k, v in r3.items()
                      if v['group'] == 'r3_E' and v['dataset'] == ds}
            if method_name == 'Interp+iTrans':
                subset = [v['test_mse'] for v in e_data.values() if v['method'] == 'simple']
                if subset:
                    matrix[mi, col_high] = np.mean(subset)
            elif method_name == 'CoIFNet':
                subset = [v['test_mse'] for v in e_data.values() if v['method'] == 'coifnet']
                if subset:
                    matrix[mi, col_high] = np.mean(subset)
            elif method_name == 'MissTSM':
                subset = [v['test_mse'] for v in e_data.values() if v['method'] == 'misstsm']
                if subset:
                    matrix[mi, col_high] = np.mean(subset)

    fig, ax = plt.subplots(figsize=(14, 5))
    display = np.where(np.isnan(matrix), 0, matrix)
    im = ax.imshow(display, cmap='YlOrRd', aspect='auto', vmin=0.15, vmax=0.55)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(all_methods)))
    ax.set_yticklabels(all_methods, fontsize=11)

    for i in range(len(all_methods)):
        for j in range(len(col_labels)):
            if np.isnan(matrix[i, j]):
                ax.text(j, i, '-', ha='center', va='center', fontsize=10, color='gray')
            else:
                val = matrix[i, j]
                color = 'white' if val > 0.35 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)

    # Vertical line separating datasets
    for sep in [1.5, 3.5]:
        ax.axvline(x=sep, color='white', linewidth=2)

    plt.colorbar(im, ax=ax, label='Mean MSE (lower = better)', shrink=0.8)
    ax.set_title('Comprehensive: Method × Dataset × Missing Rate Regime', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig8_comprehensive_heatmap.png')
    plt.close()
    print("  fig8_comprehensive_heatmap.png")


# ===== RUN ALL =====
if __name__ == '__main__':
    print("Generating R3 figures...")
    fig1_fix_heatmap()
    fig2_method_ranking()
    fig3_c_group()
    fig4_d_group()
    fig5_e_group()
    fig6_missing_rate_trend()
    fig7_radar()
    fig8_comprehensive_heatmap()
    print(f"Done! All figures saved to {OUT_DIR}/")
