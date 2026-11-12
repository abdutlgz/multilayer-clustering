"""Rerun one division and compare it with the stored results.

    python3 check_division.py [S] [REP]       (default: S = 32, REP = 1; about 30 s plus 10 s to load the data)

Prints MATCH if the active layers and the group labels of all three methods (M = 2 and 3, and for S = 32
also after removing half of the large layers) are identical to results/divisions/S{S}/r{REP}.json.
"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'run'))
import run_divisions as R   # noqa: E402


def main():
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    rep = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    stored = json.loads((HERE.parent / 'results' / 'divisions' / f'S{s:03d}' / f'r{rep:03d}.json').read_text())
    with tempfile.TemporaryDirectory() as tmp:
        sys.argv = ['run_divisions.py', str(s), '--first', str(rep), '--reps', '1', '--workers', '1', '--out', tmp]
        R.main()
        new = json.loads((Path(tmp) / f'S{s:03d}' / f'r{rep:03d}.json').read_text())
    ok = True
    for key in ('genre', 'type', 'ncols', 'nnz', 'active'):
        same = new[key] == stored[key]
        ok &= same
        print(f'{key:8s}', 'same' if same else 'DIFFERENT')
    parts = [('', new['labels'], stored['labels'])]
    if 'removal' in stored:
        parts.append(('after removal ', new['removal']['labels'], stored['removal']['labels']))
    for name, a, b in parts:
        for m in ('2', '3'):
            for meth in a[m]:
                same = a[m][meth] == b[m][meth]
                ok &= same
                print(f'{name}M={m} {meth:10s}', 'same' if same else 'DIFFERENT')
    print('MATCH' if ok else 'MISMATCH')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
