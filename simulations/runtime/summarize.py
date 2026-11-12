#!/usr/bin/env python3
"""Collect the per-run lines of Section 4.3 into two CSV files (standard library only).

usage: python3 summarize.py <dir>
  reads  <dir>/raw/*.csv   (one line per network and method, written by run_network.sh)
  writes <dir>/runs.csv     all runs, with a header
         <dir>/summary.csv  one row per (setting id, method)

Reported as in the paper: average error, median running time and median peak memory
over the networks of a setting. mem_over_none is the peak memory minus that of the
"no method" process (Input_only) on the same network.
"""
import csv, glob, math, os, statistics, sys

COLS = ['id', 'n', 'L', 'rho', 'seed', 'method', 'seconds', 't1', 't2', 't3', 'err', 'err_active',
        'err_nonactive', 'active_count', 'groups_active', 'min_active_width', 'peak_rss_kb',
        'process_wall', 'host', 'cpu']
ORDER = ['Input_only', 'Algorithm1', 'All_layer', 'UASE'] + [f'Top{p}' for p in range(10, 101, 10)]


def quantile(v, p):
    v = sorted(v); k = (len(v) - 1) * p; i = int(k)
    return v[i] if i + 1 >= len(v) else v[i] + (v[i + 1] - v[i]) * (k - i)


def num(x):
    try:
        return float(x)
    except ValueError:
        return math.nan


def mean_of(rows, key):
    v = [num(r[key]) for r in rows]
    v = [x for x in v if not math.isnan(x)]
    return statistics.mean(v) if v else ''


def summarize(rows, path):
    gib = lambda r: int(r['peak_rss_kb']) / 1024**2
    none = {r['seed']: gib(r) for r in rows if r['method'] == 'Input_only'}
    groups = {}
    for r in rows:
        groups.setdefault((int(r['id']), r['method']), []).append(r)
    rank = lambda k: (k[0], ORDER.index(k[1]) if k[1] in ORDER else 99)
    with open(path, 'w', newline='') as h:
        w = csv.writer(h)
        w.writerow(['id', 'n', 'L', 'rho', 'method', 'networks', 'err_mean', 'err_active_mean',
                    'err_nonactive_mean', 'active_mean', 'time_median', 'time_q25', 'time_q75',
                    'mem_gib_median', 'mem_gib_q25', 'mem_gib_q75', 'mem_over_none_median', 'hosts'])
        for key in sorted(groups, key=rank):
            g = groups[key]; t = [float(r['seconds']) for r in g]; m = [gib(r) for r in g]
            over = [gib(r) - none[r['seed']] for r in g if r['seed'] in none]
            w.writerow([key[0], g[0]['n'], g[0]['L'], g[0]['rho'], key[1], len(g),
                        mean_of(g, 'err'), mean_of(g, 'err_active'), mean_of(g, 'err_nonactive'),
                        mean_of(g, 'active_count'), quantile(t, .5), quantile(t, .25), quantile(t, .75),
                        quantile(m, .5), quantile(m, .25), quantile(m, .75),
                        quantile(over, .5) if over else '', ';'.join(sorted({r['host'] for r in g}))])
    return len(groups)


def main(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, 'raw', '*.csv'))):
        with open(f) as h:
            rows += [dict(zip(COLS, r)) for r in csv.reader(h)]
    with open(os.path.join(d, 'runs.csv'), 'w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    k = summarize(rows, os.path.join(d, 'summary.csv'))
    print(f'{len(rows)} runs, {k} (setting, method) groups -> {d}')


if __name__ == '__main__':
    main(sys.argv[1])
