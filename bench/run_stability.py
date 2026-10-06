#!/usr/bin/env python3
"""Paired CPU/GPU stability test on the large dataset ladder.

For every rung of the ladder it runs the same binary with different backends in
alternating order (to cancel machine drift), checks that the Newick output is
byte-identical to the CPU reference, and reports medians, spread and speedup.

  run_stability.py --build BUILD_DIR --out RUN_DIR [--repeats N]
                   [--cases FILE] [--modes cpu,auto,cuda] [--cpu 8]
                   [--timeout 7200] [--extra-opts "-gtr"] [--tag NAME]

The case file has one case per line:  name|arguments|alignment
(explicit case files prevent silently substituting a different dataset).

Timing runs do not use -gpu-stats (it costs time); a separate diagnostic run per
case/mode captures the statistics block when --stats is given.
"""
import argparse
import csv
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path




def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--modes', default='cpu,auto,cuda')
    ap.add_argument('--cpu', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=7200)
    ap.add_argument('--extra-opts', default='')
    ap.add_argument('--tag', default='')
    ap.add_argument('--stats', action='store_true',
                    help='also run one -gpu-stats diagnostic per case/mode')
    args = ap.parse_args()

    root = Path.cwd()
    build = (root / args.build).resolve()
    binary = build / 'fasttree_gpu'
    if not binary.exists():
        sys.exit('missing %s' % binary)
    out = (root / args.out)
    out.mkdir(parents=True, exist_ok=False)

    modes = [m for m in args.modes.split(',') if m]
    if args.repeats < 1 or not modes or modes[0] != 'cpu' or len(set(modes)) != len(modes) or set(modes)-{'cpu','auto','cuda'} or len(modes)<2:
        sys.exit('use positive repeats and unique modes starting with cpu, followed by auto/cuda')
    extra = shlex.split(args.extra_opts) if args.extra_opts else []

    if args.cases:
        cases = []
        for line in Path(args.cases).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, argstr, aln = line.split('|')
            cases.append((name, shlex.split(argstr), aln))

    if not cases or len({c[0] for c in cases}) != len(cases):
        sys.exit('cases must be nonempty with unique names')
    for name, _, _ in cases:
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            sys.exit('case names may contain only letters, digits, underscore and hyphen')

    affinity = sorted(os.sched_getaffinity(0))
    if args.cpu not in affinity:
        sys.exit('CPU %d is not in the allowed set %s' % (args.cpu, affinity))

    meta = {
        'binary': str(binary),
        'binary_sha256': sha256(binary),
        'cpu_pinned': args.cpu,
        'allowed_cpus': len(affinity),
        'modes': modes,
        'repeats': args.repeats,
        'extra_opts': extra,
        'tag': args.tag,
        'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'cases': [{'name': n, 'args': a, 'alignment': d,
                   'sha256': sha256(root / d) if (root / d).exists() else None}
                  for n, a, d in cases],
    }
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2))

    rows = []
    summary = {}
    with (out / 'timings.tsv').open('w') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['case', 'repeat', 'mode', 'seconds', 'maxrss_kb',
                    'exact_vs_cpu', 'tree_sha256'])
        for name, case_args, aln in cases:
            if not (root / aln).exists():
                sys.exit('missing alignment: %s' % aln)
            gold = None
            values = {m: [] for m in modes}
            for rep in range(args.repeats):
                order = modes if rep % 2 == 0 else list(reversed(modes))
                for mode in order:
                    prefix = out / ('%s.%d.%s' % (name, rep, mode))
                    cmd = ['taskset', '-c', str(args.cpu), str(binary),
                           '-seed', '12345', '-backend', mode, *extra,
                           *case_args, aln]
                    Path(str(prefix) + '.command.json').write_text(json.dumps(cmd))
                    tree = Path(str(prefix) + '.tree')
                    log = Path(str(prefix) + '.log')
                    t0 = time.perf_counter()
                    with tree.open('wb') as tf, log.open('wb') as lf:
                        rc = subprocess.run(cmd, cwd=root, stdout=tf, stderr=lf,
                                            timeout=args.timeout).returncode
                    wall = time.perf_counter() - t0
                    if rc != 0:
                        print('FAIL %s %s rep%d rc=%d' % (name, mode, rep, rc), flush=True)
                        w.writerow([name, rep, mode, 'FAILED_%d' % rc, '-', '-', '-'])
                        fh.flush()
                        sys.exit(1)
                    content = tree.read_bytes()
                    if mode == 'cpu':
                        if gold is None:
                            gold = content
                    exact = (content == gold) if gold is not None else None
                    if exact is False:
                        print('!! OUTPUT DIFFERS: %s %s rep%d' % (name, mode, rep), flush=True)
                    w.writerow([name, rep, mode, '%.3f' % wall, '-', exact,
                                hashlib.sha256(content).hexdigest()])
                    fh.flush()
                    if exact is not True:
                        sys.exit('output differs or CPU baseline is missing')
                    values[mode].append(wall)
                    print('%-10s rep%d %-5s %8.2f s  exact=%s' %
                          (name, rep, mode, wall, exact), flush=True)
            entry = {}
            for mode in modes:
                v = values[mode]
                if not v:
                    continue
                entry[mode] = {
                    'n': len(v),
                    'median_s': statistics.median(v),
                    'min_s': min(v),
                    'max_s': max(v),
                    'spread_pct': 100.0 * (max(v) - min(v)) / statistics.median(v),
                    'samples_s': v,
                }
            if 'cpu' in entry:
                base = entry['cpu']['median_s']
                for mode in modes:
                    if mode == 'cpu' or mode not in entry:
                        continue
                    entry[mode]['speedup_vs_cpu'] = base / entry[mode]['median_s']
                    entry[mode]['time_saved_pct'] = 100.0 * (base - entry[mode]['median_s']) / base
            summary[name] = entry
            (out / 'summary.json').write_text(json.dumps(summary, indent=2))

    # diagnostic statistics run (not part of the timings)
    if args.stats:
        for name, case_args, aln in cases:
            if not (root / aln).exists():
                continue
            for mode in modes:
                if mode == 'cpu':
                    continue
                log = out / ('%s.diag.%s.stats.log' % (name, mode))
                cmd = ['taskset', '-c', str(args.cpu), str(binary), '-seed', '12345',
                       '-backend', mode, '-gpu-stats', *extra, *case_args, aln]
                with log.open('wb') as lf, open(os.devnull, 'wb') as nf:
                    subprocess.run(cmd, cwd=root, stdout=nf, stderr=lf,
                                   timeout=args.timeout, check=True)
                text = log.read_text(errors='replace')
                keep = [l for l in text.splitlines() if l.startswith('# GPU')]
                (out / ('%s.diag.%s.stats.txt' % (name, mode))).write_text('\n'.join(keep) + '\n')
                print('diag %s %s: %d stats lines' % (name, mode, len(keep)), flush=True)

    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('\n=== summary ===')
    for name, entry in summary.items():
        for mode, e in entry.items():
            extra_s = ''
            if 'time_saved_pct' in e:
                extra_s = '  saved=%+.2f%%  speedup=%.3fx' % (e['time_saved_pct'], e['speedup_vs_cpu'])
            print('%-10s %-5s median=%8.2f s  spread=%5.1f%%  n=%d%s' %
                  (name, mode, e['median_s'], e['spread_pct'], e['n'], extra_s))


if __name__ == '__main__':
    main()
