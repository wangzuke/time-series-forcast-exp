"""Read-only data audit; saves summaries, never imputes or trains models."""
from pathlib import Path
import io
import json
import hashlib
import zipfile
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def csvs_in_zip(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in archive.namelist():
            if name.lower().endswith('.zip'):
                yield from csvs_in_zip(archive.read(name))
            elif name.lower().endswith('.csv'):
                yield name, archive.read(name)

def summary(frame):
    missing = frame.isna()
    rates = missing.mean() * 100
    max_gaps = {}
    for col in frame:
        s = missing[col]
        lengths = s.groupby((s != s.shift()).cumsum()).sum()
        max_gaps[str(col)] = int(lengths.max())
    return {
        'shape_time_by_station': list(frame.shape),
        'start': str(frame.index.min()), 'end': str(frame.index.max()),
        'unique_timestamps': int(frame.index.nunique()),
        'nonhour_steps': int((frame.index.to_series().diff().dropna() != pd.Timedelta(hours=1)).sum()),
        'missing_percent': float(missing.to_numpy().mean() * 100),
        'station_missing_percent': {str(k): float(v) for k, v in rates.items()},
        'longest_missing_run_hours': max_gaps,
        'monthly_missing_percent': {str(k): float(v) for k, v in missing.mean(axis=1).groupby(frame.index.to_period('M')).mean().mul(100).items()},
    }

def main():
    out = {'audit_date': '2026-10-16', 'scope': 'Raw missingness only; causes are not identified by these files.'}
    local = ROOT / 'dataset/AQI/AQI36_PriSTI/pm25_ground.txt'
    aq = pd.read_csv(local, index_col=0, parse_dates=True)
    out['AQI36'] = dict(summary(aq), sha256=digest(local), source=str(local))
    train = aq.iloc[:int(len(aq) * .7)]
    availability = train.notna().rolling(24, min_periods=24).mean()
    target = train.shift(-6)
    high = target.ge(train.quantile(.9), axis=1) & target.notna()
    low = availability.le(.5)
    out['AQI36_training_only_scenario_counts'] = {
        'definition': 'First 70% rows only; previous 24h availability <=50%; station training q90; target t+6h. Overlapping station-hours are NOT independent pollution events.',
        'train_rows': len(train),
        'valid_future_labels': int(target.notna().to_numpy().sum()),
        'high_q90_labels': int(high.to_numpy().sum()),
        'low_availability_valid_future_labels': int((low & target.notna()).to_numpy().sum()),
        'low_availability_high_future_labels': int((low & high).to_numpy().sum()),
    }
    path = HERE / '数据审计输入/beijing12.zip'
    if not path.exists():
        path = Path('/tmp/scenario_beijing12_20261016.zip')
    if path.exists() and zipfile.is_zipfile(path):
        columns = []
        for name, blob in csvs_in_zip(path.read_bytes()):
            if 'PRSA_Data_' not in name:
                continue
            d = pd.read_csv(io.BytesIO(blob))
            dt = pd.to_datetime(d[['year', 'month', 'day', 'hour']])
            columns.append(pd.Series(d['PM2.5'].to_numpy(), index=dt, name=d['station'].iloc[0]))
        bj = pd.concat(columns, axis=1).sort_index()
        out['Beijing12'] = dict(summary(bj), sha256=digest(path), source='https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data')
    path = HERE / '数据审计输入/parking_birmingham.zip'
    if not path.exists():
        path = Path('/tmp/scenario_parking_20261016.zip')
    if path.exists() and zipfile.is_zipfile(path):
        name, blob = next(csvs_in_zip(path.read_bytes()))
        d = pd.read_csv(io.BytesIO(blob))
        t = pd.to_datetime(d.LastUpdated)
        out['ParkingBirmingham'] = {
            'sha256': digest(path), 'source': 'https://archive.ics.uci.edu/dataset/482/parking+birmingham',
            'rows': len(d), 'distinct_IDs_not_verified_independent_sites': int(d.SystemCodeNumber.nunique()),
            'start': str(t.min()), 'end': str(t.max()),
            'recorded_hours': sorted(t.dt.hour.unique().tolist()),
            'nulls': d.isna().sum().to_dict(),
            'rows_per_ID': d.groupby('SystemCodeNumber').size().to_dict(),
            'negative_occupancy_rows': int((d.Occupancy < 0).sum()),
            'over_capacity_rows': int((d.Occupancy > d.Capacity).sum()),
            'columns': list(d.columns),
            'caution': 'No empty CSV cells does not imply a complete time grid; no coordinates in this CSV; overnight non-recording is not automatically an outage.'
        }
    (HERE / '数据可用性审计.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        datasets = [k for k in ['AQI36', 'Beijing12'] if k in out]
        fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 3.5), squeeze=False, sharey=True)
        for ax, key in zip(axes[0], datasets):
            vals = sorted(out[key]['station_missing_percent'].values())
            ax.bar(range(1, len(vals)+1), vals, color='#2a788e')
            ax.set(title=f'{key}: raw missingness by station', xlabel='Stations, sorted by missingness', ylabel='Missing observations (%)')
            ax.grid(axis='y', alpha=.2)
        fig.tight_layout()
        fig.savefig(HERE / '站点原始缺失率.png', dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps({k: v.get('shape_time_by_station', v.get('rows')) for k, v in out.items() if isinstance(v, dict)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
