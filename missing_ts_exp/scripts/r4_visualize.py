"""第四轮实验可视化脚本：生成 0706 实验报告所需的全部图表。"""
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

OUT_DIR = 'figures_r4'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load R4 data ──
r4 = {}
for f in sorted(glob.glob('results/r4_*.json')):
    tag = os.path.basename(f).replace('.json', '')
    d = json.load(open(f))
    cfg = d['config']
    r4[tag] = {
        'group': tag.split('__')[0],
        'tag': tag,
        'method': cfg['method'], 'dataset': cfg['dataset'],
        'predictor': cfg.get('predictor', ''),
        'missing_type': cfg['missing_type'], 'missing_rate': cfg['missing_rate'],
        'pred_len': cfg['pred_len'], 'seed': cfg['seed'],
        'impute': cfg.get('impute', 'none'),
        'test_mse': d['test']['mse'], 'test_mae': d['test']['mae'],
        'impute_mse': d['test'].get('impute_mse', 0),
        'n_params': d['n_params'],
        'mask_aware': cfg.get('mask_aware', 'none'),
        'misstsm_variant': cfg.get('misstsm_variant', 'full'),
        'coifnet_hidden': cfg.get('coifnet_hidden', 256),
        'coifnet_embed_type': cfg.get('coifnet_embed_type', 'shared'),
        'coifnet_input_form': cfg.get('coifnet_input_form', 'x_cat_mask'),
    }

# ── Load R3 E group (for ExchangeRate comparison) ──
r3e = {}
for f in sorted(glob.glob('results/r3_E__*.json')):
    tag = os.path.basename(f).replace('.json', '')
    d = json.load(open(f))
    cfg = d['config']
    r3e[tag] = {
        'group': tag.split('__')[0],
        'method': cfg['method'], 'dataset': cfg['dataset'],
        'missing_type': cfg['missing_type'], 'missing_rate': cfg['missing_rate'],
        'pred_len': cfg['pred_len'], 'seed': cfg['seed'],
        'test_mse': d['test']['mse'], 'test_mae': d['test']['mae'],
        'predictor': cfg.get('predictor', ''),
        'impute': cfg.get('impute', 'none'),
        'mask_aware': cfg.get('mask_aware', 'none'),
    }

# ── Load R3 D2 group (mask_aware add for ETTh1/Weather at 10%/30%) ──
r3d2 = {}
for f in sorted(glob.glob('results/r3_D2__*.json')):
    tag = os.path.basename(f).replace('.json', '')
    d = json.load(open(f))
    cfg = d['config']
    r3d2[tag] = {
        'group': 'r3_D2',
        'tag': tag,
        'method': cfg['method'], 'dataset': cfg['dataset'],
        'predictor': cfg.get('predictor', ''),
        'missing_type': cfg['missing_type'], 'missing_rate': cfg['missing_rate'],
        'pred_len': cfg['pred_len'], 'seed': cfg['seed'],
        'test_mse': d['test']['mse'], 'test_mae': d['test']['mae'],
        'mask_aware': cfg.get('mask_aware', 'none'),
    }

print(f"Loaded: R4={len(r4)}, R3-E={len(r3e)}, R3-D2={len(r3d2)}")


def avg_by(data, key_fn, val_fn):
    groups = defaultdict(list)
    for v in (data.values() if isinstance(data, dict) else data):
        k = key_fn(v)
        if k is not None:
            groups[k].append(val_fn(v))
    return {k: np.mean(vs) for k, vs in groups.items()}


def avg_by_std(data, key_fn, val_fn):
    groups = defaultdict(list)
    for v in (data.values() if isinstance(data, dict) else data):
        k = key_fn(v)
        if k is not None:
            groups[k].append(val_fn(v))
    return {k: (np.mean(vs), np.std(vs)) for k, vs in groups.items()}


def add_bar_labels(ax, bars, fmt='{:.3f}', fontsize=7, rotation=0):
    for bar in bars:
        h = bar.get_height()
        if h > 0 and not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                    fmt.format(h), ha='center', va='bottom',
                    fontsize=fontsize, rotation=rotation)


def _method_label(v):
    """Build a readable method label from record fields."""
    tag = v.get('tag', '')
    if 'P0A' in tag or 'P0B' in tag:
        parts = tag.split('__')
        for p in parts:
            if p.startswith('linear_') or p.startswith('mask_aware'):
                return p
    if 'P1' in tag:
        parts = tag.split('__')
        if len(parts) >= 2:
            return parts[1]
    return v.get('method', '')


# ===== FIGURE 1: P0-A ExchangeRate Baselines =====
def fig1_exchange_baselines():
    """ExchangeRate: two-stage baselines vs missing-aware methods.

    Data sources:
    - P0-A (R4): Interp+DLinear/PatchTST/iTransformer at all rates
    - P1 (R4):   CoIFNet/MissTSM/Interp+iTrans at all rates (complete)
    - R3 E:      CoIFNet/MissTSM/Interp+iTrans at 50%/70% only (subset, superseded by P1)
    We use P1 for missing-aware methods so all 4 rates are covered.
    """
    p0a = {k: v for k, v in r4.items() if v['group'] == 'r4_P0A' and v['pred_len'] == 96}
    p1_ex = {k: v for k, v in r4.items()
             if v['group'] == 'r4_P1' and v['dataset'] == 'ExchangeRate' and v['pred_len'] == 96}

    methods_map = {
        'Interp+DLinear': lambda v: 'r4_P0A' in v.get('tag', '') and v['predictor'] == 'DLinear',
        'Interp+PatchTST': lambda v: 'r4_P0A' in v.get('tag', '') and v['predictor'] == 'PatchTST',
        'Interp+iTransformer': lambda v: 'r4_P0A' in v.get('tag', '') and v['predictor'] == 'iTransformer',
        'CoIFNet(R2var)': lambda v: 'coifnet_R2var' in v.get('tag', ''),
        'MissTSM': lambda v: 'misstsm_full' in v.get('tag', ''),
        'CoIFNet(orig)': lambda v: 'coifnet_faithful' in v.get('tag', ''),
    }

    all_data = list(p0a.values()) + list(p1_ex.values())

    for mt_label, mt_val in [('Random Point', 'random_point'), ('Continuous Segment', 'continuous_segment')]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        rates = [0.1, 0.3, 0.5, 0.7]
        all_methods = list(methods_map.keys())
        colors = ['#4E79A7', '#59A14F', '#F28E2B', '#E15759', '#B07AA1', '#76B7B2']
        n = len(all_methods)
        w = 0.12
        x = np.arange(len(rates))

        for i, mname in enumerate(all_methods):
            filt_fn = methods_map[mname]
            vals = []
            for r in rates:
                sub = [v['test_mse'] for v in all_data
                       if filt_fn(v) and v['missing_type'] == mt_val
                       and abs(v['missing_rate'] - r) < 0.01]
                vals.append(np.mean(sub) if sub else np.nan)
            bars = ax.bar(x + (i - n/2 + 0.5) * w, vals, w, label=mname, color=colors[i % len(colors)])
            add_bar_labels(ax, bars, fontsize=6, rotation=45)

        ax.set_xlabel('Missing Rate')
        ax.set_ylabel('Test MSE')
        ax.set_title(f'ExchangeRate — {mt_label} (pred_len=96)')
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in rates])
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        fname = f'fig1_exchange_{mt_val}.png'
        fig.savefig(os.path.join(OUT_DIR, fname))
        plt.close(fig)
        print(f'  Saved {fname}')


# ===== FIGURE 2: P0-B Mask-Aware add Improvement Heatmap =====
def fig2_mask_aware_heatmap():
    """Heatmap of improvement from mask-aware add vs baseline.

    Data sources for mask-aware add results:
    - R4 P0B: Electricity/ExchangeRate all rates, ETTh1/Weather 50%/70%
    - R4 P2C: ETTh1/Weather 10%/30% (PatchTST only)
    - R3 D2:  ETTh1/Weather 10%/30% (iTransformer only)
    Baselines (interp_iTrans / interp_PatchTST) are all in R4 P1.
    """
    p0b_add = list(r4[k] for k in r4 if r4[k]['group'] == 'r4_P0B')
    p2c_add = list(r4[k] for k in r4 if r4[k]['group'] == 'r4_P2C')
    r3d2_add = list(r3d2.values())
    all_add = p0b_add + p2c_add + r3d2_add

    p1_base = {k: v for k, v in r4.items() if v['group'] == 'r4_P1'}

    datasets = ['ETTh1', 'Weather', 'Electricity', 'ExchangeRate']
    rates = [0.1, 0.3, 0.5, 0.7]

    for predictor, pred_short in [('iTransformer', 'iTrans'), ('PatchTST', 'PatchTST')]:
        for mt_label, mt_val in [('Random Point', 'random_point'), ('Continuous Segment', 'continuous_segment')]:
            if predictor == 'iTransformer':
                base_tag_part = 'interp_iTrans'
            else:
                base_tag_part = 'interp_PatchTST'

            matrix = np.full((len(datasets), len(rates)), np.nan)
            for di, ds in enumerate(datasets):
                for ri, r in enumerate(rates):
                    add_vals = [v['test_mse'] for v in all_add
                                if v.get('predictor', '') == predictor and v['dataset'] == ds
                                and v['missing_type'] == mt_val and abs(v['missing_rate'] - r) < 0.01
                                and v['pred_len'] == 96
                                and v.get('mask_aware', 'none') == 'add']
                    base_vals = [v['test_mse'] for v in p1_base.values()
                                 if base_tag_part in v['tag'] and v['dataset'] == ds
                                 and v['missing_type'] == mt_val and abs(v['missing_rate'] - r) < 0.01]
                    if add_vals and base_vals:
                        base_m = np.mean(base_vals)
                        add_m = np.mean(add_vals)
                        matrix[di, ri] = (add_m - base_m) / base_m * 100

            fig, ax = plt.subplots(figsize=(7, 4))
            vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 10)
            im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(rates)))
            ax.set_xticklabels([str(r) for r in rates])
            ax.set_yticks(range(len(datasets)))
            ax.set_yticklabels(datasets)
            ax.set_xlabel('Missing Rate')
            ax.set_title(f'Mask-Aware {predictor}(add) MSE Change % — {mt_label}\n(negative=improvement, positive=degradation)')
            for di in range(len(datasets)):
                for ri in range(len(rates)):
                    val = matrix[di, ri]
                    if not np.isnan(val):
                        color = 'white' if abs(val) > vmax * 0.5 else 'black'
                        ax.text(ri, di, f'{val:+.1f}%', ha='center', va='center',
                                fontsize=9, fontweight='bold', color=color)
            plt.colorbar(im, ax=ax, label='MSE Change %', shrink=0.8)
            fname = f'fig2_mask_aware_{pred_short}_{mt_val}.png'
            fig.savefig(os.path.join(OUT_DIR, fname))
            plt.close(fig)
            print(f'  Saved {fname}')


# ===== FIGURE 3: P1 Winner Heatmap =====
def fig3_winner_heatmap():
    """Condition matrix: which method wins at each (dataset, rate, missing_type)."""
    p1 = {k: v for k, v in r4.items() if v['group'] == 'r4_P1'}

    method_tags = ['interp_iTrans', 'interp_PatchTST', 'mask_add_iTrans',
                   'misstsm_full', 'coifnet_faithful', 'coifnet_R2var']
    method_labels = ['Interp+iTrans', 'Interp+PatchTST', 'MaskAdd+iTrans',
                     'MissTSM', 'CoIFNet(orig)', 'CoIFNet(R2var)']
    method_colors = ['#4E79A7', '#59A14F', '#F28E2B', '#E15759', '#B07AA1', '#76B7B2']

    datasets = ['ETTh1', 'ExchangeRate', 'Weather', 'Electricity', 'Traffic']
    rates = [0.1, 0.3, 0.5, 0.7]

    for mt_label, mt_val in [('Random Point', 'random_point'), ('Continuous Segment', 'continuous_segment')]:
        matrix = np.full((len(datasets), len(rates)), np.nan)
        annot = [['' for _ in rates] for _ in datasets]

        for di, ds in enumerate(datasets):
            for ri, r in enumerate(rates):
                best_mse = float('inf')
                best_idx = -1
                for mi, mtag in enumerate(method_tags):
                    vals = [v['test_mse'] for v in p1.values()
                            if mtag in v['tag'] and v['dataset'] == ds
                            and v['missing_type'] == mt_val and abs(v['missing_rate'] - r) < 0.01]
                    if vals:
                        m = np.mean(vals)
                        if m < best_mse:
                            best_mse = m
                            best_idx = mi
                if best_idx >= 0:
                    matrix[di, ri] = best_idx
                    annot[di][ri] = method_labels[best_idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(method_colors)
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-0.5, vmax=len(method_tags)-0.5)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([str(r) for r in rates])
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets)
        ax.set_xlabel('Missing Rate')
        ax.set_title(f'Best Method — {mt_label} (pred_len=96)')
        for di in range(len(datasets)):
            for ri in range(len(rates)):
                if annot[di][ri]:
                    ax.text(ri, di, annot[di][ri], ha='center', va='center',
                            fontsize=7, fontweight='bold', color='white',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.3))
        cbar = plt.colorbar(im, ax=ax, ticks=range(len(method_tags)), shrink=0.8)
        cbar.ax.set_yticklabels(method_labels, fontsize=8)
        fname = f'fig3_winner_{mt_val}.png'
        fig.savefig(os.path.join(OUT_DIR, fname))
        plt.close(fig)
        print(f'  Saved {fname}')


# ===== FIGURE 4: P1 Missing Rate Trend Lines =====
def fig4_rate_trends():
    """MSE vs missing rate for each dataset, with all methods."""
    p1 = {k: v for k, v in r4.items() if v['group'] == 'r4_P1'}

    method_tags = ['interp_iTrans', 'interp_PatchTST', 'mask_add_iTrans',
                   'misstsm_full', 'coifnet_faithful', 'coifnet_R2var']
    method_labels = ['Interp+iTrans', 'Interp+PatchTST', 'MaskAdd+iTrans',
                     'MissTSM', 'CoIFNet(orig)', 'CoIFNet(R2var)']
    colors = ['#4E79A7', '#59A14F', '#F28E2B', '#E15759', '#B07AA1', '#76B7B2']
    markers = ['o', 's', '^', 'D', 'v', 'P']

    datasets = ['ETTh1', 'Weather', 'Electricity', 'Traffic']
    rates = [0.1, 0.3, 0.5, 0.7]

    for mt_label, mt_val in [('Random Point', 'random_point'), ('Continuous Segment', 'continuous_segment')]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for di, ds in enumerate(datasets):
            ax = axes[di]
            for mi, (mtag, mlabel) in enumerate(zip(method_tags, method_labels)):
                vals = []
                for r in rates:
                    sub = [v['test_mse'] for v in p1.values()
                           if mtag in v['tag'] and v['dataset'] == ds
                           and v['missing_type'] == mt_val and abs(v['missing_rate'] - r) < 0.01]
                    vals.append(np.mean(sub) if sub else np.nan)
                valid = [not np.isnan(v) for v in vals]
                if any(valid):
                    ax.plot(rates, vals, color=colors[mi], marker=markers[mi],
                            label=mlabel, linewidth=1.5, markersize=5)

            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('Test MSE')
            ax.set_title(ds)
            ax.grid(alpha=0.3)
            if di == 0:
                ax.legend(fontsize=7, loc='upper left')

        fig.suptitle(f'MSE vs Missing Rate — {mt_label}', fontsize=14, y=1.01)
        fig.tight_layout()
        fname = f'fig4_trend_{mt_val}.png'
        fig.savefig(os.path.join(OUT_DIR, fname))
        plt.close(fig)
        print(f'  Saved {fname}')


# ===== FIGURE 5: P1 Two-Stage vs Missing-Aware Delta =====
def fig5_delta():
    """MSE difference: missing-aware minus best two-stage."""
    p1 = {k: v for k, v in r4.items() if v['group'] == 'r4_P1'}

    two_stage_tags = ['interp_iTrans', 'interp_PatchTST']
    aware_tags = ['misstsm_full', 'coifnet_faithful', 'coifnet_R2var']
    aware_labels = ['MissTSM', 'CoIFNet(orig)', 'CoIFNet(R2var)']
    colors = ['#E15759', '#B07AA1', '#76B7B2']

    datasets = ['ETTh1', 'Weather', 'Electricity', 'Traffic']
    rates = [0.1, 0.3, 0.5, 0.7]

    for mt_label, mt_val in [('Random Point', 'random_point'), ('Continuous Segment', 'continuous_segment')]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for di, ds in enumerate(datasets):
            ax = axes[di]
            for ai, (atag, alabel) in enumerate(zip(aware_tags, aware_labels)):
                deltas = []
                for r in rates:
                    best_ts = float('inf')
                    for tstag in two_stage_tags:
                        sub = [v['test_mse'] for v in p1.values()
                               if tstag in v['tag'] and v['dataset'] == ds
                               and v['missing_type'] == mt_val and abs(v['missing_rate'] - r) < 0.01]
                        if sub:
                            best_ts = min(best_ts, np.mean(sub))

                    sub_aw = [v['test_mse'] for v in p1.values()
                              if atag in v['tag'] and v['dataset'] == ds
                              and v['missing_type'] == mt_val and abs(v['missing_rate'] - r) < 0.01]
                    if sub_aw and best_ts < float('inf'):
                        deltas.append(np.mean(sub_aw) - best_ts)
                    else:
                        deltas.append(np.nan)

                ax.plot(rates, deltas, color=colors[ai], marker='o', label=alabel, linewidth=1.5)

            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Missing Rate')
            ax.set_ylabel('MSE (aware - best_two_stage)')
            ax.set_title(ds)
            ax.grid(alpha=0.3)
            if di == 0:
                ax.legend(fontsize=8)

        fig.suptitle(f'Missing-Aware vs Best Two-Stage MSE Delta — {mt_label}\n(negative = aware better)',
                     fontsize=13, y=1.02)
        fig.tight_layout()
        fname = f'fig5_delta_{mt_val}.png'
        fig.savefig(os.path.join(OUT_DIR, fname))
        plt.close(fig)
        print(f'  Saved {fname}')


# ===== FIGURE 6: P2-A CoIFNet Ablation =====
def fig6_coifnet_ablation():
    """CoIFNet 5-variant ablation on Weather and Electricity."""
    p2a = {k: v for k, v in r4.items() if v['group'] == 'r4_P2A'}

    variant_map = {
        'A0': (128, 'independent', 'xmask_cat_mask'),
        'A1': (256, 'independent', 'xmask_cat_mask'),
        'A2': (128, 'independent', 'x_cat_mask'),
        'A3': (256, 'shared', 'x_cat_mask'),
        'A4': (256, 'independent', 'x_cat_mask'),
    }
    variant_names = list(variant_map.keys())
    variant_labels = [
        'A0: h128+indep+xm*m',
        'A1: h256+indep+xm*m',
        'A2: h128+indep+x_m',
        'A3: h256+shared+x_m',
        'A4: h256+indep+x_m',
    ]
    colors = ['#4E79A7', '#59A14F', '#F28E2B', '#E15759', '#B07AA1']

    datasets = ['Weather', 'Electricity']
    conditions = [('random_point', 0.3), ('random_point', 0.5),
                  ('continuous_segment', 0.3), ('continuous_segment', 0.5)]
    cond_labels = ['rp_30', 'rp_50', 'cs_30', 'cs_50']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for di, ds in enumerate(datasets):
        ax = axes[di]
        n = len(variant_names)
        w = 0.14
        x = np.arange(len(conditions))

        for vi, vname in enumerate(variant_names):
            h, et, inf = variant_map[vname]
            vals = []
            for mt, mr in conditions:
                sub = [v['test_mse'] for v in p2a.values()
                       if v['dataset'] == ds and v['coifnet_hidden'] == h
                       and v['coifnet_embed_type'] == et and v['coifnet_input_form'] == inf
                       and v['missing_type'] == mt and abs(v['missing_rate'] - mr) < 0.01]
                vals.append(np.mean(sub) if sub else np.nan)
            bars = ax.bar(x + (vi - n/2 + 0.5) * w, vals, w,
                         label=variant_labels[vi], color=colors[vi])
            add_bar_labels(ax, bars, fontsize=6, rotation=45)

        ax.set_xlabel('Condition')
        ax.set_ylabel('Test MSE')
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('P2-A: CoIFNet Variant Ablation', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig6_coifnet_ablation.png'))
    plt.close(fig)
    print('  Saved fig6_coifnet_ablation.png')


# ===== FIGURE 7: P2-B MissTSM Grouped Query =====
def fig7_grouped_query():
    """MissTSM grouped_q4 vs full baseline."""
    p2b = {k: v for k, v in r4.items() if v['group'] == 'r4_P2B'}
    p1_misstsm = {k: v for k, v in r4.items()
                  if v['group'] == 'r4_P1' and 'misstsm_full' in v['tag']}

    datasets = ['Weather', 'Electricity', 'Traffic']
    rates = [0.3, 0.5, 0.7]
    miss_types = ['random_point', 'continuous_segment']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for di, ds in enumerate(datasets):
        ax = axes[di]
        conditions = [(mt, r) for mt in miss_types for r in rates]
        cond_labels = [f'{"rp" if mt=="random_point" else "cs"}_{int(r*100)}'
                       for mt, r in conditions]

        full_vals = []
        gq4_vals = []
        for mt, r in conditions:
            sub_full = [v['test_mse'] for v in p1_misstsm.values()
                        if v['dataset'] == ds and v['missing_type'] == mt
                        and abs(v['missing_rate'] - r) < 0.01]
            sub_gq4 = [v['test_mse'] for v in p2b.values()
                       if v['dataset'] == ds and v['missing_type'] == mt
                       and abs(v['missing_rate'] - r) < 0.01]
            full_vals.append(np.mean(sub_full) if sub_full else np.nan)
            gq4_vals.append(np.mean(sub_gq4) if sub_gq4 else np.nan)

        x = np.arange(len(conditions))
        w = 0.35
        b1 = ax.bar(x - w/2, full_vals, w, label='MissTSM(full)', color='#4E79A7')
        b2 = ax.bar(x + w/2, gq4_vals, w, label='MissTSM(grouped_q4)', color='#E15759')
        add_bar_labels(ax, b1, fontsize=6, rotation=45)
        add_bar_labels(ax, b2, fontsize=6, rotation=45)

        ax.set_xlabel('Condition')
        ax.set_ylabel('Test MSE')
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=8, rotation=30)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('P2-B: MissTSM Grouped Query (G=4) vs Full', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig7_grouped_query.png'))
    plt.close(fig)
    print('  Saved fig7_grouped_query.png')


# ===== FIGURE 8: P2-C PatchTST Mask Add =====
def fig8_patchtst_add():
    """PatchTST(add) vs PatchTST baseline on ETTh1 and Weather."""
    p2c = {k: v for k, v in r4.items() if v['group'] == 'r4_P2C'}
    p0b_ptst = {k: v for k, v in r4.items()
                if v['group'] == 'r4_P0B' and v['predictor'] == 'PatchTST'}
    p1_ptst = {k: v for k, v in r4.items()
               if v['group'] == 'r4_P1' and 'interp_PatchTST' in v['tag']}

    all_add = list(p2c.values()) + list(p0b_ptst.values())

    datasets = ['ETTh1', 'Weather', 'Electricity', 'ExchangeRate']
    rates = [0.1, 0.3, 0.5, 0.7]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for di, ds in enumerate(datasets):
        ax = axes[di]
        for mt_label, mt_val, offset in [('rp', 'random_point', -0.2), ('cs', 'continuous_segment', 0.2)]:
            base_vals = []
            add_vals = []
            valid_rates = []
            for r in rates:
                sub_base = [v['test_mse'] for v in p1_ptst.values()
                            if v['dataset'] == ds and v['missing_type'] == mt_val
                            and abs(v['missing_rate'] - r) < 0.01]
                sub_add = [v['test_mse'] for v in all_add
                           if v['dataset'] == ds and v['missing_type'] == mt_val
                           and abs(v['missing_rate'] - r) < 0.01]
                if sub_base and sub_add:
                    base_vals.append(np.mean(sub_base))
                    add_vals.append(np.mean(sub_add))
                    valid_rates.append(r)

            if valid_rates:
                x = np.arange(len(valid_rates))
                w = 0.15
                b1 = ax.bar(x + offset - w/2, base_vals, w,
                           label=f'PatchTST ({mt_label})', alpha=0.7,
                           color='#4E79A7' if mt_label == 'rp' else '#59A14F')
                b2 = ax.bar(x + offset + w/2, add_vals, w,
                           label=f'PatchTST+add ({mt_label})', alpha=0.7,
                           color='#F28E2B' if mt_label == 'rp' else '#E15759')
                add_bar_labels(ax, b1, fontsize=5, rotation=45)
                add_bar_labels(ax, b2, fontsize=5, rotation=45)
                ax.set_xticks(x)
                ax.set_xticklabels([str(r) for r in valid_rates])

        ax.set_xlabel('Missing Rate')
        ax.set_ylabel('Test MSE')
        ax.set_title(ds)
        ax.grid(axis='y', alpha=0.3)
        if di == 0:
            ax.legend(fontsize=7)

    fig.suptitle('P2-C: PatchTST(add) vs PatchTST Baseline', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig8_patchtst_add.png'))
    plt.close(fig)
    print('  Saved fig8_patchtst_add.png')


# ===== FIGURE 9: Comprehensive Heatmap =====
def fig9_comprehensive_heatmap():
    """Method × (dataset × rate regime) average MSE."""
    p1 = {k: v for k, v in r4.items() if v['group'] == 'r4_P1'}

    method_tags = ['interp_iTrans', 'interp_PatchTST', 'mask_add_iTrans',
                   'misstsm_full', 'coifnet_faithful', 'coifnet_R2var']
    method_labels = ['Interp+iTrans', 'Interp+PatchTST', 'MaskAdd+iTrans',
                     'MissTSM', 'CoIFNet(orig)', 'CoIFNet(R2var)']
    datasets = ['ETTh1', 'ExchangeRate', 'Weather', 'Electricity', 'Traffic']
    regimes = [('Low (0.1-0.3)', [0.1, 0.3]), ('High (0.5-0.7)', [0.5, 0.7])]

    col_labels = [f'{ds}\n{regime}' for ds in datasets for regime, _ in regimes]
    matrix = np.full((len(method_tags), len(col_labels)), np.nan)

    for mi, mtag in enumerate(method_tags):
        ci = 0
        for ds in datasets:
            for regime_name, regime_rates in regimes:
                vals = [v['test_mse'] for v in p1.values()
                        if mtag in v['tag'] and v['dataset'] == ds
                        and v['missing_rate'] in regime_rates]
                if vals:
                    matrix[mi, ci] = np.mean(vals)
                ci += 1

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=30, ha='right')
    ax.set_yticks(range(len(method_labels)))
    ax.set_yticklabels(method_labels)
    ax.set_title('Comprehensive MSE Heatmap: Method × (Dataset × Rate Regime)')

    for mi in range(len(method_tags)):
        for ci in range(len(col_labels)):
            val = matrix[mi, ci]
            if not np.isnan(val):
                color = 'white' if val > np.nanmedian(matrix) else 'black'
                ax.text(ci, mi, f'{val:.3f}', ha='center', va='center',
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label='Test MSE', shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig9_comprehensive_heatmap.png'))
    plt.close(fig)
    print('  Saved fig9_comprehensive_heatmap.png')


# ===== FIGURE 10: Radar Chart =====
def fig10_radar():
    """Radar chart: method ranking across dataset dimensions."""
    p1 = {k: v for k, v in r4.items() if v['group'] == 'r4_P1'}

    method_tags = ['interp_iTrans', 'interp_PatchTST', 'mask_add_iTrans',
                   'misstsm_full', 'coifnet_R2var']
    method_labels = ['Interp+iTrans', 'Interp+PatchTST', 'MaskAdd+iTrans',
                     'MissTSM', 'CoIFNet(R2var)']
    colors = ['#4E79A7', '#59A14F', '#F28E2B', '#E15759', '#76B7B2']

    categories = [
        ('ETTh1\n(low-dim)', 'ETTh1'),
        ('ExchangeRate\n(low-dim)', 'ExchangeRate'),
        ('Weather\n(mid-dim)', 'Weather'),
        ('Electricity\n(high-dim)', 'Electricity'),
        ('Traffic\n(ultra-high)', 'Traffic'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))
    rate_regimes = [('Low Missing (0.1-0.3)', [0.1, 0.3]),
                    ('High Missing (0.5-0.7)', [0.5, 0.7])]

    for ax_idx, (regime_label, regime_rates) in enumerate(rate_regimes):
        ax = axes[ax_idx]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        for mi, (mtag, mlabel) in enumerate(zip(method_tags, method_labels)):
            values = []
            for cat_label, ds in categories:
                vals = [v['test_mse'] for v in p1.values()
                        if mtag in v['tag'] and v['dataset'] == ds
                        and v['missing_rate'] in regime_rates]
                values.append(np.mean(vals) if vals else np.nan)

            if all(np.isnan(v) for v in values):
                continue

            max_val = max(v for v in values if not np.isnan(v))
            min_val = min(v for v in values if not np.isnan(v))
            if max_val == min_val:
                norm_values = [0.5] * len(values)
            else:
                norm_values = [1.0 - (v - min_val) / (max_val - min_val + 1e-8)
                               if not np.isnan(v) else 0 for v in values]

            norm_values += norm_values[:1]
            ax.plot(angles, norm_values, color=colors[mi], linewidth=1.5, label=mlabel)
            ax.fill(angles, norm_values, color=colors[mi], alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c[0] for c in categories], fontsize=8)
        ax.set_title(regime_label, fontsize=12, pad=20)
        ax.set_ylim(0, 1)
        ax.set_yticklabels([])

    axes[0].legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=8)
    fig.suptitle('Method Ranking Radar (higher = better)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig10_radar.png'))
    plt.close(fig)
    print('  Saved fig10_radar.png')


# ===== Run all =====
if __name__ == '__main__':
    print("Generating R4 visualizations...")

    print("\n[1/10] ExchangeRate baselines")
    fig1_exchange_baselines()

    print("\n[2/10] Mask-Aware improvement heatmap")
    fig2_mask_aware_heatmap()

    print("\n[3/10] P1 winner heatmap")
    fig3_winner_heatmap()

    print("\n[4/10] P1 rate trend lines")
    fig4_rate_trends()

    print("\n[5/10] P1 two-stage vs aware delta")
    fig5_delta()

    print("\n[6/10] P2-A CoIFNet ablation")
    fig6_coifnet_ablation()

    print("\n[7/10] P2-B grouped query")
    fig7_grouped_query()

    print("\n[8/10] P2-C PatchTST add")
    fig8_patchtst_add()

    print("\n[9/10] Comprehensive heatmap")
    fig9_comprehensive_heatmap()

    print("\n[10/10] Radar chart")
    fig10_radar()

    print(f"\nDone! All figures saved to {OUT_DIR}/")
    print(f"Total files: {len(os.listdir(OUT_DIR))}")
