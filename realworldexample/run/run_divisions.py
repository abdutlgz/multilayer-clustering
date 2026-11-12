"""Fit the three methods on random divisions of the MovieLens movies (Section 5).

    python3 run_divisions.py S [--reps 50] [--workers 5] [--out DIR]

S is the number of small layers per genre (the paper uses 8, 16, 32 and 64). For every division
rep = 1..reps, all three methods are fitted with M = 2 and M = 3 groups. For S = 32, each division is
also refitted after removing at random 2 of the 4 large layers of every genre.

One JSON file per division is written to DIR/S{S:03d}/r{rep:03d}.json (default DIR: ../output/divisions).
Files that already exist are skipped, so an interrupted run can be restarted.
For S = 32, division 1 also saves the layer similarity matrix used by Figure 5.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lib'))
from data import MovieLens, GENRES, LARGE_PER_GENRE   # noqa: E402
from methods import fit                                # noqa: E402

DATA = None
ARGS = None


def removal_kept(n_small, rep):
    """Indices of the layers kept after removing 2 of the 4 large layers of every genre at random."""
    rng = np.random.default_rng([20261109, 77, rep])
    block = LARGE_PER_GENRE + n_small
    kept = []
    for k in range(len(GENRES)):
        large = list(range(k * block, k * block + LARGE_PER_GENRE))
        gone = set(rng.choice(large, 2, replace=False).tolist())
        kept += [i for i in large if i not in gone] + list(range(k * block + LARGE_PER_GENRE, (k + 1) * block))
    return kept


def job(rep):
    s = ARGS.S
    out = Path(ARGS.out) / f'S{s:03d}'
    f = out / f'r{rep:03d}.json'
    if f.exists():
        return f.name
    layers = DATA.divide(s, rep)
    mats = [r['a'] for r in layers]
    res = {'S': s, 'rep': rep,
           'genre': [r['genre'] for r in layers], 'type': [r['type'] for r in layers],
           'ncols': [int(a.shape[1]) for a in mats], 'nnz': [int(a.nnz) for a in mats],
           'labels': {}}
    for m in (2, 3):
        labels, info = fit(mats, m, 8411 + rep)
        res['labels'][str(m)] = labels
    res['active'] = info['active']
    res['rho_hat'] = info['rho_hat']
    res['threshold'] = info['threshold']
    if s == 32 and rep == 1:
        np.savez_compressed(out / 'similarity_r001.npz', G=info['similarity'])
    if s == 32:
        kept = removal_kept(s, rep)
        res['removal'] = {'kept': kept, 'labels': {}}
        for m in (2, 3):
            labels, info = fit([mats[i] for i in kept], m, 8411 + rep)
            res['removal']['labels'][str(m)] = labels
        res['removal']['active'] = info['active']
    f.write_text(json.dumps(res))
    return f.name


def main():
    global DATA, ARGS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('S', type=int, help='number of small layers per genre')
    p.add_argument('--reps', type=int, default=50, help='number of random divisions (default 50)')
    p.add_argument('--first', type=int, default=1, help='first division (default 1)')
    p.add_argument('--workers', type=int, default=5, help='parallel processes (default 5)')
    p.add_argument('--out', default=str(HERE.parent / 'output' / 'divisions'))
    ARGS = p.parse_args()
    (Path(ARGS.out) / f'S{ARGS.S:03d}').mkdir(parents=True, exist_ok=True)
    t = time.time()
    DATA = MovieLens()
    print(f'loaded MovieLens in {time.time() - t:.0f}s', flush=True)
    reps = range(ARGS.first, ARGS.first + ARGS.reps)
    if ARGS.workers == 1:
        done = map(job, reps)
    else:
        import multiprocessing as mp
        pool = mp.get_context('fork').Pool(ARGS.workers)   # workers share the loaded data
        done = pool.imap_unordered(job, reps)
    for k, name in enumerate(done, 1):
        print(f'S={ARGS.S}: {k}/{len(reps)} {name} {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    main()
