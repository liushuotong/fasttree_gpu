#!/usr/bin/env python3
"""Summarize strict within-batch CPU/GPU pairs; never pair across directories."""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(dirs):
    rows = []
    for directory in dirs:
        path = Path(directory).resolve() / 'timings.tsv'
        with path.open() as f:
            for row in csv.DictReader(f, delimiter='\t'):
                row['seconds'] = float(row['seconds'])
                if not math.isfinite(row['seconds']) or row['seconds'] <= 0:
                    raise ValueError(f'{path}: invalid elapsed time')
                if row.get('exact_vs_cpu') != 'True':
                    raise ValueError(f'{path}: failed or unverified output: {row}')
                row['dir'] = str(path.parent)
                rows.append(row)
    if not rows:
        raise ValueError('no measurements')
    return rows


def analyze(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = (row['dir'], row['case'], row['repeat'])
        if row['mode'] in groups[key]:
            raise ValueError(f'duplicate measurement: {key} {row["mode"]}')
        groups[key][row['mode']] = row['seconds']
    pairs = defaultdict(list)
    for (directory, case, repeat), modes in groups.items():
        if 'cpu' not in modes or not any(m in modes for m in ('auto', 'cuda')):
            raise ValueError(f'incomplete pair: {directory} {case} {repeat}')
        if set(modes) - {'cpu', 'auto', 'cuda'}:
            raise ValueError(f'unknown backend: {modes.keys()}')
        for mode in ('auto', 'cuda'):
            if mode in modes:
                pairs[(case, mode)].append((modes['cpu'], modes[mode]))
    result = {}
    for key, values in sorted(pairs.items()):
        cpu, gpu = [v[0] for v in values], [v[1] for v in values]
        diffs = [c-g for c, g in values]
        cm, gm = statistics.median(cpu), statistics.median(gpu)
        result[key] = dict(n=len(values), cpu_median=cm, gpu_median=gm,
                           saved_pct=100*(1-gm/cm), speedup=cm/gm,
                           cpu_spread_pct=100*(max(cpu)-min(cpu))/cm,
                           gpu_spread_pct=100*(max(gpu)-min(gpu))/gm,
                           diff_median=statistics.median(diffs), diff_min=min(diffs),
                           diff_max=max(diffs), wins=sum(d > 0 for d in diffs))
    return result


def main():
    if len(sys.argv) < 2:
        print('usage: summarize_stability.py RUN_DIR [RUN_DIR ...]', file=sys.stderr)
        return 2
    try:
        rows = load(sys.argv[1:])
        summary = analyze(rows)
    except (OSError, ValueError, KeyError) as e:
        print(f'Invalid measurements: {e}', file=sys.stderr)
        return 1
    print('case\tmode\tpairs\tcpu_median_s\tgpu_median_s\tsaved_pct\tspeedup\tcpu_spread_pct\tgpu_spread_pct\tdiff_median_s\tdiff_min_s\tdiff_max_s\tgpu_wins')
    for (case, mode), s in summary.items():
        print(f'{case}\t{mode}\t{s["n"]}\t{s["cpu_median"]:.3f}\t{s["gpu_median"]:.3f}\t{s["saved_pct"]:.3f}\t{s["speedup"]:.4f}\t{s["cpu_spread_pct"]:.3f}\t{s["gpu_spread_pct"]:.3f}\t{s["diff_median"]:.3f}\t{s["diff_min"]:.3f}\t{s["diff_max"]:.3f}\t{s["wins"]}')
    print(f'# runs={len(rows)} all outputs checked; spread=(max-min)/median*100')
    return 0


if __name__ == '__main__':
    sys.exit(main())
