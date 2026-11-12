"""Table 3 and every number in the text of Section 5, from the stored results.

    python3 make_tables.py [RESULTS_DIR]

RESULTS_DIR defaults to ../results (it must contain divisions/S008 ... divisions/S064).
Also writes RESULTS_DIR/summary.csv: one row per division, M and method.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lib'))
from methods import ari   # noqa: E402

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'results'
SMALL = (8, 16, 32, 64)
METHODS = {'algorithm1': 'Algorithm 1', 'all_layer': 'All-layer', 'uase': 'UASE'}
GENRES = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Romance', 'Sci-Fi', 'Thriller']


def load(s):
    files = sorted((RESULTS / 'divisions' / f'S{s:03d}').glob('r*.json'))
    return [json.loads(f.read_text()) for f in files]


def stats(labels, genre, typ):
    """ARI with genres; share of small layers in the group of most large layers of their genre;
    whether the large layers of every genre are in one group; the groups of genres (by large layers)."""
    labels, genre, typ = np.asarray(labels), np.asarray(genre), np.asarray(typ)
    ok = total = 0
    same = True
    groups = collections.defaultdict(list)
    for g in GENRES:
        large = labels[(genre == g) & (typ == 'large')]
        small = labels[(genre == g) & (typ == 'small')]
        major = np.bincount(large).argmax()
        same &= len(set(large)) == 1
        ok += (small == major).sum()
        total += len(small)
        groups[int(major)].append(g)
    partition = ' | '.join(sorted(', '.join(v) for v in groups.values()))
    return ari(labels, genre), ok / total, same, partition


def summarize():
    rows = []
    for s in SMALL:
        for d in load(s):
            for m in ('2', '3'):
                for meth in METHODS:
                    a, share, same, part = stats(d['labels'][m][meth], d['genre'], d['type'])
                    row = dict(S=s, rep=d['rep'], M=int(m), method=meth, ari=a, small_in_genre_group=share,
                               large_same_group=same, genre_groups=part,
                               n_layers=len(d['genre']), n_active=sum(d['active']),
                               n_active_small=sum(f and t == 'small' for f, t in zip(d['active'], d['type'])))
                    if 'removal' in d:
                        k = d['removal']['kept']
                        ra, rs, _, _ = stats(d['removal']['labels'][m][meth],
                                             [d['genre'][i] for i in k], [d['type'][i] for i in k])
                        row.update(ari_removal=ra, small_in_genre_group_removal=rs,
                                   n_active_removal=sum(d['removal']['active']))
                    rows.append(row)
    return pd.DataFrame(rows)


def best_count(t, col):
    w = t.pivot_table(index=['S', 'M', 'rep'], columns='method', values=col)
    return ((w.algorithm1 > w.all_layer) & (w.algorithm1 > w.uase)).groupby(level=[0, 1]).sum()


def main():
    t = summarize()
    t.to_csv(RESULTS / 'summary.csv', index=False)
    n_div = t.groupby('S').rep.nunique().to_dict()
    print('divisions per S:', n_div)

    # ---------------------------------------------------------------- Section 5.1
    d32 = load(32)
    print('\n== Section 5.1: active layers (32 small layers per genre) ==')
    rho = np.array([d['rho_hat'] for d in d32]); thr = np.array([d['threshold'] for d in d32])
    n = 5000
    large = [c for d in d32 for c, ty in zip(d['ncols'], d['type']) if ty == 'large']
    small = [c for d in d32 for c, ty in zip(d['ncols'], d['type']) if ty == 'small']
    print(f'rho_hat = {rho.min():.4f}..{rho.max():.4f}; threshold n_l >= {thr.min():.1f}..{thr.max():.1f} movies; '
          f'beta_star = {np.log(thr.max()) / np.log(n):.3f}')
    print(f'large layers: at least {min(large)} movies (beta_l >= {np.log(min(large)) / np.log(n):.3f}); '
          f'small layers: at most {max(small)} movies (beta_l <= {np.log(max(small)) / np.log(n):.3f})')
    exact = all(d['active'] == [ty == 'large' for ty in d['type']] for d in d32)
    print('active layers = exactly the 32 large layers in every division:', exact)

    # ---------------------------------------------------------------- Section 5.2, division 1
    print('\n== Section 5.2: division 1 (Figures 5-6) ==')
    d = d32[0]
    typ = np.array(d['type'])
    for m in ('2', '3'):
        a, share, same, part = stats(d['labels'][m]['algorithm1'], d['genre'], d['type'])
        print(f'M={m} Algorithm 1 groups: {part}; large layers of each genre in one group: {same}; '
              f'small layers in their genre\'s group: {round(share * (typ == "small").sum())} of {(typ == "small").sum()}')
    lab = np.array(d['labels']['3']['all_layer'])
    big = np.bincount(lab[typ == 'large'], minlength=3)
    print(f'All-layer M=3: {big.max()} of 32 large layers in one group; '
          f'small layers in that group: {(lab[typ == "small"] == big.argmax()).sum()}')

    # ---------------------------------------------------------------- Section 5.2, 50 divisions
    print('\n== Section 5.2: 50 divisions, 32 small layers per genre ==')
    a1 = t[(t.S == 32) & (t.method == 'algorithm1')]
    for m in (2, 3):
        print(f'M={m} genre groups of Algorithm 1:')
        for part, c in a1[a1.M == m].genre_groups.value_counts().items():
            print(f'   {c:3d} divisions: {part}')
    x = t[t.S == 32].groupby(['M', 'method'])[['small_in_genre_group', 'ari', 'ari_removal']].mean()
    print('averages (share of small layers in their genre\'s group, ARI, ARI after removal):')
    print(x.round(4).to_string())
    for col in ('small_in_genre_group', 'ari', 'ari_removal'):
        print(f'Algorithm 1 best in {col}:', best_count(t[t.S == 32], col).droplevel(0).to_dict())
    print('active layers after removal:', sorted(int(x) for x in t[t.S == 32].n_active_removal.unique()))

    # ---------------------------------------------------------------- Table 3
    print('\n== Table 3: average ARI with genres ==')
    tab = t.groupby(['S', 'M', 'method']).ari.mean().unstack(['M', 'method'])
    act = t.groupby('S')[['n_layers', 'n_active']].first()
    out = pd.DataFrame({'layers (all / active)': [f'{a} / {b}' for a, b in act.values]}, index=act.index)
    for m in (2, 3):
        for meth, name in METHODS.items():
            out[f'M={m} {name}'] = tab[(m, meth)].map('{:.2f}'.format)
    print(out.to_string())
    print('\nAlgorithm 1 has the highest ARI in this many divisions:')
    print(best_count(t, 'ari').unstack('S').to_string())
    print('\nactive small layers per genre (S = 8 and 16):')
    for s in (8, 16):
        ds = load(s)
        per = {g: np.mean([sum(f and ty == 'small' and ge == g for f, ty, ge in zip(x['active'], x['type'], x['genre']))
                           for x in ds]) for g in GENRES}
        print(f'  S={s}:', {g: round(float(v), 2) for g, v in per.items() if v > 0})


if __name__ == '__main__':
    main()
