"""第二轮实验可视化脚本：生成实验报告所需的全部图表。"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    'savefig.pad_inches': 0.1,
})

OUT_DIR = 'figures_r2'

# ── Load data ──
r2 = {}
for f in sorted(glob.glob('results/r2_[a-f]_*.json')):
    tag = os.path.basename(f).replace('.json','')
    d = json.load(open(f))
    cfg = d['config']
    r2[tag] = {
        'group': tag.split('__')[0],
        'method': cfg['method'], 'dataset': cfg['dataset'],
        'missing_type': cfg['missing_type'], 'missing_rate': cfg['missing_rate'],
        'pred_len': cfg['pred_len'], 'seed': cfg['seed'],
        'test_mse': d['test']['mse'], 'test_mae': d['test']['mae'],
        'impute_mse': d['test'].get('impute_mse', 0),
        'n_params': d['n_params'], 'peak_mem_mb': d['peak_mem_mb'],
        'epochs_run': len(d['history']),
        'train_time_per_epoch': d['history'][0]['train']['time_sec'] if d['history'] else 0,
        'predictor': cfg.get('predictor',''), 'impute': cfg.get('impute','none'),
    }

r1 = {}
for f in sorted(glob.glob('results/*.json')):
    tag = os.path.basename(f).replace('.json','')
    if tag.startswith('r2_'): continue
    try:
        d = json.load(open(f))
        cfg = d['config']
        if d['test']['mse'] > 10: continue
        r1[tag] = {
            'method': cfg['method'], 'dataset': cfg['dataset'],
            'missing_type': cfg['missing_type'], 'missing_rate': cfg['missing_rate'],
            'pred_len': cfg['pred_len'], 'seed': cfg['seed'],
            'test_mse': d['test']['mse'], 'test_mae': d['test']['mae'],
            'predictor': cfg.get('predictor',''), 'impute': cfg.get('impute','none'),
        }
    except: pass


def avg_by(data, key_fn, val_fn):
    """Group results and compute mean."""
    groups = defaultdict(list)
    for v in data.values():
        k = key_fn(v)
        if k is not None:
            groups[k].append(val_fn(v))
    return {k: np.mean(vs) for k, vs in groups.items()}


def avg_by_std(data, key_fn, val_fn):
    groups = defaultdict(list)
    for v in data.values():
        k = key_fn(v)
        if k is not None:
            groups[k].append(val_fn(v))
    return {k: (np.mean(vs), np.std(vs)) for k, vs in groups.items()}


# ── R1 baselines for comparison ──
def r1_method_label(v):
    if v['method'] == 'simple' and v['impute'] == 'linear' and v['predictor'] == 'iTransformer':
        return 'Interp+iTrans'
    if v['method'] == 'saits' and v['predictor'] == 'iTransformer':
        return 'SAITS+iTrans'
    return None

r1_baselines = avg_by(
    {k:v for k,v in r1.items()
     if v['missing_type'] in ('random_point','continuous_segment')
     and v['missing_rate'] in (0.1, 0.3)
     and v['dataset'] in ('Weather','Electricity','Traffic')},
    lambda v: (r1_method_label(v), v['dataset']),
    lambda v: v['test_mse']
)


# ===== FIGURE 1: Group A Main Comparison =====
def fig1_main_comparison():
    a_data = {k:v for k,v in r2.items() if v['group']=='r2_a'}
    methods_r2 = ['misstsm', 'crib', 'coifnet']
    method_labels = {
        'misstsm': 'MissTSM-full', 'crib': 'CRIB-full', 'coifnet': 'CoIFNet-full',
    }
    datasets = ['Weather', 'Electricity', 'Traffic']

    # Average across seeds, missing types, rates, pred_lens
    r2_avg = avg_by(a_data,
        lambda v: (v['method'], v['dataset']),
        lambda v: v['test_mse'])

    # R1 aware methods (simplified versions) for comparison
    r1_aware = avg_by(
        {k:v for k,v in r1.items()
         if v['method'] in methods_r2
         and v['dataset'] in datasets
         and v['missing_type'] in ('random_point','continuous_segment')
         and v['missing_rate'] in (0.1, 0.3)},
        lambda v: (v['method'], v['dataset']),
        lambda v: v['test_mse'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    x = np.arange(len(methods_r2))
    w = 0.25

    for i, ds in enumerate(datasets):
        ax = axes[i]
        r1_vals = [r1_aware.get((m, ds), 0) for m in methods_r2]
        r2_vals = [r2_avg.get((m, ds), 0) for m in methods_r2]

        # R1 baselines
        interp_val = r1_baselines.get(('Interp+iTrans', ds), 0)
        saits_val = r1_baselines.get(('SAITS+iTrans', ds), 0)

        bars1 = ax.bar(x - w, r1_vals, w, label='R1 Simplified', color='#ff9999', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x, r2_vals, w, label='R2 Full', color='#66b3ff', edgecolor='black', linewidth=0.5)

        ax.axhline(y=interp_val, color='green', linestyle='--', linewidth=1.5, label=f'Interp+iTrans (R1)')
        ax.axhline(y=saits_val, color='orange', linestyle=':', linewidth=1.5, label=f'SAITS+iTrans (R1)')

        ax.set_xticks(x)
        ax.set_xticklabels([method_labels[m] for m in methods_r2], rotation=15)
        ax.set_title(ds)
        ax.set_ylabel('Test MSE' if i == 0 else '')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

        # Add value labels
        for bar_group in [bars1, bars2]:
            for bar in bar_group:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f'{h:.3f}',
                            ha='center', va='bottom', fontsize=7)

    fig.suptitle('Group A: Full vs Simplified Aware Methods (avg over all configs)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig1_main_comparison.png')
    plt.close()
    print("  fig1_main_comparison.png")


# ===== FIGURE 2: Group B High Missing Rate =====
def fig2_high_missing_rate():
    b_data = {k:v for k,v in r2.items() if v['group']=='r2_b'}

    methods = ['simple', 'misstsm', 'crib', 'coifnet']
    method_labels = ['Interp+iTrans', 'MissTSM-full', 'CRIB-full', 'CoIFNet-full']
    rates = [0.5, 0.7]
    datasets = ['Weather', 'Traffic']

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for i, ds in enumerate(datasets):
        for j, mt in enumerate(['random_point', 'continuous_segment']):
            ax = axes[i][j]
            mt_label = 'Random Point' if mt == 'random_point' else 'Continuous Segment'

            x = np.arange(len(rates))
            w = 0.18

            for mi, m in enumerate(methods):
                vals = []
                for r in rates:
                    subset = [v['test_mse'] for v in b_data.values()
                              if v['method']==m and v['dataset']==ds
                              and v['missing_type']==mt and v['missing_rate']==r]
                    vals.append(np.mean(subset) if subset else 0)

                offset = (mi - len(methods)/2 + 0.5) * w
                bars = ax.bar(x + offset, vals, w, label=method_labels[mi], edgecolor='black', linewidth=0.5)
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.3f}',
                                ha='center', va='bottom', fontsize=7, rotation=45)

            ax.set_xticks(x)
            ax.set_xticklabels(['50%', '70%'])
            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('Test MSE')
            ax.set_title(f'{ds} - {mt_label}')
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.suptitle('Group B: High Missing Rate Exploration (50% & 70%)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig2_high_missing_rate.png')
    plt.close()
    print("  fig2_high_missing_rate.png")


# ===== FIGURE 3: R1 vs R2 Improvement =====
def fig3_r1_vs_r2():
    methods = ['misstsm', 'crib', 'coifnet']
    method_labels = ['MissTSM', 'CRIB', 'CoIFNet']
    datasets = ['Weather', 'Electricity', 'Traffic']

    r2_a = {k:v for k,v in r2.items() if v['group']=='r2_a'}
    r2_avg = avg_by(r2_a, lambda v: (v['method'], v['dataset']), lambda v: v['test_mse'])

    r1_aware = avg_by(
        {k:v for k,v in r1.items()
         if v['method'] in methods
         and v['dataset'] in datasets
         and v['missing_type'] in ('random_point','continuous_segment')
         and v['missing_rate'] in (0.1, 0.3)},
        lambda v: (v['method'], v['dataset']),
        lambda v: v['test_mse'])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    w = 0.12
    colors_r1 = ['#ffcccc', '#ccffcc', '#ccccff']
    colors_r2 = ['#ff4444', '#44aa44', '#4444ff']

    for mi, m in enumerate(methods):
        r1_vals = [r1_aware.get((m, ds), 0) for ds in datasets]
        r2_vals = [r2_avg.get((m, ds), 0) for ds in datasets]

        offset1 = (mi * 2 - len(methods) + 0.5) * w
        offset2 = offset1 + w

        ax.bar(x + offset1, r1_vals, w, label=f'{method_labels[mi]} (R1 simplified)',
               color=colors_r1[mi], edgecolor='black', linewidth=0.5)
        ax.bar(x + offset2, r2_vals, w, label=f'{method_labels[mi]} (R2 full)',
               color=colors_r2[mi], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('Test MSE')
    ax.set_title('Simplified (R1) vs Full (R2) Implementation Comparison')
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig3_r1_vs_r2.png')
    plt.close()
    print("  fig3_r1_vs_r2.png")


# ===== FIGURE 4: Missing Rate Trend (0.1 ~ 0.7) =====
def fig4_missing_rate_trend():
    # Combine R1 (0.1, 0.3) and R2 B (0.5, 0.7) for Weather
    methods_map = {
        'Interp+iTrans': lambda v: v['method']=='simple' and v['impute']=='linear' and v['predictor']=='iTransformer',
        'MissTSM': lambda v: v['method']=='misstsm',
        'CoIFNet': lambda v: v['method']=='coifnet',
        'CRIB': lambda v: v['method']=='crib',
    }
    ds = 'Weather'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for mi, mt in enumerate(['random_point', 'continuous_segment']):
        ax = axes[mi]
        mt_label = 'Random Point' if mt == 'random_point' else 'Continuous Segment'

        for method_name, filter_fn in methods_map.items():
            rates_vals = {}
            # R1 data (0.1, 0.3)
            for v in r1.values():
                if filter_fn(v) and v['dataset']==ds and v['missing_type']==mt and v['pred_len']==96:
                    r = v['missing_rate']
                    rates_vals.setdefault(r, []).append(v['test_mse'])
            # R2 data (0.1, 0.3 from group A; 0.5, 0.7 from group B)
            for v in r2.values():
                if filter_fn(v) and v['dataset']==ds and v['missing_type']==mt and v['pred_len']==96:
                    r = v['missing_rate']
                    rates_vals.setdefault(r, []).append(v['test_mse'])

            if rates_vals:
                sorted_rates = sorted(rates_vals.keys())
                means = [np.mean(rates_vals[r]) for r in sorted_rates]
                ax.plot(sorted_rates, means, 'o-', label=method_name, linewidth=2, markersize=6)

        ax.set_xlabel('Missing Rate')
        ax.set_ylabel('Test MSE')
        ax.set_title(f'{ds} - {mt_label} (pred_len=96)')
        ax.legend(fontsize=9)
        ax.set_xticks([0.1, 0.3, 0.5, 0.7])
        ax.grid(True, alpha=0.3)

    fig.suptitle('Missing Rate Trend: 10% → 70% (Weather)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig4_missing_rate_trend.png')
    plt.close()
    print("  fig4_missing_rate_trend.png")


# ===== FIGURE 5: Group C SAITS+PatchTST Fix =====
def fig5_saits_fix():
    c_data = {k:v for k,v in r2.items() if v['group']=='r2_c'}
    datasets = ['Weather', 'Electricity']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, ds in enumerate(datasets):
        ax = axes[i]
        configs = []
        mse_vals = []

        for mt in ['random_point', 'continuous_segment']:
            for r in [0.1, 0.3]:
                for H in [96, 336]:
                    subset = [v['test_mse'] for v in c_data.values()
                              if v['dataset']==ds and v['missing_type']==mt
                              and v['missing_rate']==r and v['pred_len']==H]
                    if subset:
                        mt_short = 'RP' if mt == 'random_point' else 'CS'
                        label = f'{mt_short}\n{int(r*100)}%\nh{H}'
                        configs.append(label)
                        mse_vals.append(np.mean(subset))

        bars = ax.bar(range(len(configs)), mse_vals, color='#66b3ff', edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, fontsize=8)
        ax.set_ylabel('Test MSE')
        ax.set_title(f'{ds} - SAITS+PatchTST (Fixed)')

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f'{h:.4f}',
                    ha='center', va='bottom', fontsize=7)

    fig.suptitle('Group C: SAITS+PatchTST Overflow Fix Results', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig5_saits_fix.png')
    plt.close()
    print("  fig5_saits_fix.png")


# ===== FIGURE 6: Group D Mask Ablation =====
def fig6_mask_ablation():
    d_data = {k:v for k,v in r2.items() if v['group']=='r2_d'}

    methods = ['simple', 'misstsm', 'crib']
    method_labels = ['A: Value only\n(Interp+iTrans)', 'B: Value+Mask\n(MissTSM-full)', 'C: Value+Mask+KL\n(CRIB-full)']
    datasets = ['Weather', 'Traffic']
    rates = [0.3, 0.5]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, ds in enumerate(datasets):
        ax = axes[i]
        x = np.arange(len(rates))
        w = 0.25

        for mi, m in enumerate(methods):
            vals = []
            for r in rates:
                subset = [v['test_mse'] for v in d_data.values()
                          if v['method']==m and v['dataset']==ds and v['missing_rate']==r]
                vals.append(np.mean(subset) if subset else 0)

            offset = (mi - 1) * w
            bars = ax.bar(x + offset, vals, w, label=method_labels[mi], edgecolor='black', linewidth=0.5)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f'{h:.3f}',
                            ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(['30%', '50%'])
        ax.set_xlabel('Missing Rate')
        ax.set_ylabel('Test MSE')
        ax.set_title(f'{ds} - continuous_segment, h=96')
        ax.legend(fontsize=8)

    fig.suptitle('Group D: Mask Ablation with Full-version Methods', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig6_mask_ablation.png')
    plt.close()
    print("  fig6_mask_ablation.png")


# ===== FIGURE 7: Group E Extended Scenarios =====
def fig7_extended():
    e_data = {k:v for k,v in r2.items() if v['group']=='r2_e'}

    methods = ['simple', 'misstsm', 'crib', 'coifnet']
    method_labels = ['Interp+iTrans', 'MissTSM-full', 'CRIB-full', 'CoIFNet-full']
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for i, ds in enumerate(['Electricity', 'Traffic']):
        for j, mt in enumerate(['variable_channel', 'mixed']):
            ax = axes[i][j]
            mt_label = 'Variable Channel' if mt == 'variable_channel' else 'Mixed'

            x = np.arange(2)  # h96, h336
            w = 0.18

            for mi, m in enumerate(methods):
                vals = []
                for H in [96, 336]:
                    subset = [v['test_mse'] for v in e_data.values()
                              if v['method']==m and v['dataset']==ds
                              and v['missing_type']==mt and v['pred_len']==H]
                    vals.append(np.mean(subset) if subset else 0)

                offset = (mi - len(methods)/2 + 0.5) * w
                bars = ax.bar(x + offset, vals, w, label=method_labels[mi],
                             color=colors[mi], edgecolor='black', linewidth=0.5)
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x()+bar.get_width()/2, h+0.003, f'{h:.3f}',
                                ha='center', va='bottom', fontsize=7, rotation=45)

            ax.set_xticks(x)
            ax.set_xticklabels(['h=96', 'h=336'])
            ax.set_ylabel('Test MSE')
            ax.set_title(f'{ds} - {mt_label} (rate=30%)')
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.suptitle('Group E: Extended Missing Scenarios (variable_channel & mixed)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig7_extended.png')
    plt.close()
    print("  fig7_extended.png")


# ===== FIGURE 8: Group F Error Propagation =====
def fig8_error_prop():
    f_data = {k:v for k,v in r2.items() if v['group']=='r2_f'}

    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ['misstsm', 'crib', 'coifnet']
    markers = ['o', 's', '^']
    colors = ['#e74c3c', '#3498db', '#9b59b6']

    for mi, m in enumerate(methods):
        imp_vals = []
        pred_vals = []
        labels = []
        for v in f_data.values():
            if v['method'] == m:
                imp_vals.append(v['impute_mse'])
                pred_vals.append(v['test_mse'])
                labels.append(f"{v['dataset']}\n{int(v['missing_rate']*100)}%")

        ax.scatter(imp_vals, pred_vals, marker=markers[mi], c=colors[mi],
                   s=80, label=m.upper(), edgecolors='black', linewidth=0.5, zorder=3)

    # Also add R1 two-stage baselines
    r1_err = {k:v for k,v in r1.items()
              if v['method'] in ('simple','saits')
              and v['predictor'] == 'iTransformer'
              and v['dataset'] in ('Weather','Electricity')
              and v['missing_type'] == 'random_point'
              and v['missing_rate'] in (0.1, 0.3)
              and v['pred_len'] == 96}

    ax.set_xlabel('Imputation MSE (at missing positions)')
    ax.set_ylabel('Forecast MSE')
    ax.set_title('Group F: Imputation Error vs Forecast Error\n(Full-version aware methods)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig8_error_prop.png')
    plt.close()
    print("  fig8_error_prop.png")


# ===== FIGURE 9: Overall Heatmap =====
def fig9_heatmap():
    # Average MSE for each (method, dataset) in Group A
    a_data = {k:v for k,v in r2.items() if v['group']=='r2_a'}
    methods = ['misstsm', 'crib', 'coifnet']
    datasets = ['Weather', 'Electricity', 'Traffic']

    # R2 full versions
    r2_avg = avg_by(a_data, lambda v: (v['method'], v['dataset']), lambda v: v['test_mse'])

    # R1 simplified
    r1_aware = avg_by(
        {k:v for k,v in r1.items()
         if v['method'] in methods
         and v['dataset'] in datasets
         and v['missing_type'] in ('random_point','continuous_segment')
         and v['missing_rate'] in (0.1, 0.3)},
        lambda v: (v['method'], v['dataset']),
        lambda v: v['test_mse'])

    # Build improvement matrix
    improvement = np.zeros((len(methods), len(datasets)))
    for i, m in enumerate(methods):
        for j, ds in enumerate(datasets):
            r1_val = r1_aware.get((m, ds), 0)
            r2_val = r2_avg.get((m, ds), 0)
            if r1_val > 0:
                improvement[i, j] = (r1_val - r2_val) / r1_val * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(improvement, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=40)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.upper() for m in methods])

    for i in range(len(methods)):
        for j in range(len(datasets)):
            val = improvement[i, j]
            color = 'white' if abs(val) > 20 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center', color=color, fontsize=12, fontweight='bold')

    plt.colorbar(im, ax=ax, label='MSE Improvement (%): R1 simplified → R2 full')
    ax.set_title('Full Implementation Improvement over Simplified Version')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig9_improvement_heatmap.png')
    plt.close()
    print("  fig9_improvement_heatmap.png")


# ===== FIGURE 10: Comprehensive Summary Radar =====
def fig10_summary_radar():
    # Compare 5 methods across dimensions
    all_methods = {
        'Interp+iTrans': lambda v: v['method']=='simple' and v['predictor']=='iTransformer',
        'MissTSM-full': lambda v: v['method']=='misstsm',
        'CRIB-full': lambda v: v['method']=='crib',
        'CoIFNet-full': lambda v: v['method']=='coifnet',
    }

    # Use group A + B data
    ab_data = {k:v for k,v in r2.items() if v['group'] in ('r2_a','r2_b')}
    # Add R1 Interp+iTrans
    r1_interp = {k:v for k,v in r1.items()
                 if v['method']=='simple' and v['impute']=='linear' and v['predictor']=='iTransformer'
                 and v['dataset'] in ('Weather','Electricity','Traffic')
                 and v['missing_type'] in ('random_point','continuous_segment')
                 and v['missing_rate'] in (0.1, 0.3)}

    combined = {**ab_data, **{f'r1_{k}':v for k,v in r1_interp.items()}}

    dimensions = ['Weather', 'Electricity', 'Traffic']

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

    for mi, (method_name, filter_fn) in enumerate(all_methods.items()):
        vals = []
        for ds in dimensions:
            subset = [v['test_mse'] for v in combined.values()
                      if filter_fn(v) and v['dataset']==ds]
            vals.append(np.mean(subset) if subset else 0)

        # Invert so lower MSE = bigger area
        max_mse = 0.8
        vals_inv = [max_mse - v for v in vals]
        vals_inv += vals_inv[:1]

        ax.plot(angles, vals_inv, 'o-', linewidth=2, label=method_name, color=colors[mi])
        ax.fill(angles, vals_inv, alpha=0.1, color=colors[mi])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions)
    ax.set_title('Method Comparison Across Datasets\n(larger area = better)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig10_radar.png')
    plt.close()
    print("  fig10_radar.png")


# ===== Generate all tables data =====
def generate_tables():
    """Generate markdown tables for the report."""
    out = []

    # Table 1: Group A detailed results
    a_data = {k:v for k,v in r2.items() if v['group']=='r2_a'}
    datasets = ['Weather', 'Electricity', 'Traffic']
    methods = ['misstsm', 'crib', 'coifnet']

    out.append("### Group A: Detailed Results\n")
    for ds in datasets:
        out.append(f"\n**{ds}**\n")
        out.append("| Missing Type | Rate | H | MissTSM-full | CRIB-full | CoIFNet-full |")
        out.append("|---|---:|---:|---:|---:|---:|")

        for mt in ['random_point', 'continuous_segment']:
            for r in [0.1, 0.3]:
                for H in [96, 336]:
                    row = [mt.replace('_',' '), f'{int(r*100)}%', str(H)]
                    for m in methods:
                        subset = [v for v in a_data.values()
                                  if v['method']==m and v['dataset']==ds
                                  and v['missing_type']==mt and v['missing_rate']==r
                                  and v['pred_len']==H]
                        if subset:
                            mse = np.mean([v['test_mse'] for v in subset])
                            mae = np.mean([v['test_mae'] for v in subset])
                            row.append(f'{mse:.4f} / {mae:.4f}')
                        else:
                            row.append('-')
                    out.append('| ' + ' | '.join(row) + ' |')

    # Table 2: Group B
    b_data = {k:v for k,v in r2.items() if v['group']=='r2_b'}
    out.append("\n### Group B: High Missing Rate\n")
    out.append("| Dataset | Missing Type | Rate | Interp+iTrans | MissTSM-full | CRIB-full | CoIFNet-full |")
    out.append("|---|---|---:|---:|---:|---:|---:|")

    b_methods = [
        ('simple', 'Interp+iTrans'),
        ('misstsm', 'MissTSM-full'),
        ('crib', 'CRIB-full'),
        ('coifnet', 'CoIFNet-full'),
    ]

    for ds in ['Weather', 'Traffic']:
        for mt in ['random_point', 'continuous_segment']:
            for r in [0.5, 0.7]:
                row = [ds, mt.replace('_',' '), f'{int(r*100)}%']
                for m, _ in b_methods:
                    subset = [v for v in b_data.values()
                              if v['method']==m and v['dataset']==ds
                              and v['missing_type']==mt and v['missing_rate']==r]
                    if subset:
                        mse = np.mean([v['test_mse'] for v in subset])
                        mae = np.mean([v['test_mae'] for v in subset])
                        row.append(f'{mse:.4f} / {mae:.4f}')
                    else:
                        row.append('-')
                out.append('| ' + ' | '.join(row) + ' |')

    with open('/tmp/r2_tables.md', 'w') as f:
        f.write('\n'.join(out))
    print("  tables saved to /tmp/r2_tables.md")


# ===== RUN ALL =====
print("Generating R2 figures...")
fig1_main_comparison()
fig2_high_missing_rate()
fig3_r1_vs_r2()
fig4_missing_rate_trend()
fig5_saits_fix()
fig6_mask_ablation()
fig7_extended()
fig8_error_prop()
fig9_heatmap()
fig10_summary_radar()
generate_tables()
print("Done!")
