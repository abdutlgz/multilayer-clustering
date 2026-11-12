"""Figures 5-7 of Section 5, from the stored results.

    python3 make_figures.py [RESULTS_DIR] [FIGURE_DIR]

RESULTS_DIR defaults to ../results and FIGURE_DIR to RESULTS_DIR/figures.
Writes (PDF and PNG; the names are those used in the paper's LaTeX source):
  movielens_genre_split_Nov9    Figure 5: groups of Algorithm 1 (M = 2) on the spectral embedding
  movielens_layer_groups        Figure 6: groups of every layer, three methods, M = 2 and 3
  movielens_small_layers_Nov10  Figure 7: ARI against the number of small layers per genre
Figures 5-6 use division 1 with 32 small layers per genre.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lib'))
from methods import ari                              # noqa: E402

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'results'
FIGS = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / 'figures'
FIGS.mkdir(parents=True, exist_ok=True)
GENRES = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Romance', 'Sci-Fi', 'Thriller']
BLUE, ORANGE, GREEN = '#0072BD', '#D95319', '#77AC30'
D1 = json.loads((RESULTS / 'divisions' / 'S032' / 'r001.json').read_text())


def save(fig, name):
    fig.savefig(FIGS / f'{name}.pdf')
    fig.savefig(FIGS / f'{name}.png', dpi=200)
    plt.close(fig)
    print('wrote', FIGS / f'{name}.pdf')


def figure5():
    """Large layers: two leading eigenvectors of the similarity matrix of the 32 large (active) layers,
    G(i, j) = ||U_i^T U_j||_F^2 (the embedding of Step 1c). Small layers: placed by their similarity to
    the large layers, G[small, large] w / lambda. Every point is then scaled to unit length."""
    g = np.load(RESULTS / 'divisions' / 'S032' / 'similarity_r001.npz')['G']
    act = np.array([t == 'large' for t in D1['type']])
    v, w = np.linalg.eigh(g[np.ix_(act, act)])
    o = np.argsort(v)[::-1][:2]
    v, w = v[o], w[:, o]
    if w[:, 0].mean() < 0:
        w[:, 0] *= -1
    x = np.zeros((len(act), 2))
    x[act] = w
    x[~act] = g[np.ix_(~act, act)] @ w / v
    x = x / np.linalg.norm(x, axis=1)[:, None]
    lab = np.array(D1['labels']['2']['algorithm1'])
    if lab[0] != 0:                                   # group 1 = the group of Action
        lab = 1 - lab
    pos = {'Sci-Fi': (.47, .60), 'Action': (.58, .49), 'Adventure': (.71, .40), 'Thriller': (.82, .17),
           'Crime': (.85, -.12), 'Comedy': (.76, -.40), 'Drama': (.62, -.46), 'Romance': (.50, -.56)}
    col = [BLUE, ORANGE]
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for m in (0, 1):
        j = (lab == m) & ~act
        i = (lab == m) & act
        ax.scatter(x[j, 0], x[j, 1], s=16, facecolor='white', edgecolor=col[m], lw=.7, zorder=2)
        ax.scatter(x[i, 0], x[i, 1], s=30, color=col[m], edgecolor='white', lw=.4, zorder=3)
    genre = np.array(D1['genre'])
    for gname in GENRES:
        a = np.median(x[(genre == gname) & act], 0)
        t = pos[gname]
        ax.plot([t[0], a[0]], [t[1], a[1]], color='#888', lw=.6, zorder=1)
        ax.text(*t, gname, ha='center', va='center', fontsize=8.5,
                bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none'), zorder=4)
    ax.set_xlabel('Embedding coordinate 1')
    ax.set_ylabel('Embedding coordinate 2')
    ax.grid(color='#e6e6e6', lw=.6)
    ax.set_axisbelow(True)
    h = [Line2D([], [], marker='o', ls='', color=col[0], ms=6, label='Group 1'),
         Line2D([], [], marker='o', ls='', color=col[1], ms=6, label='Group 2'),
         Line2D([], [], marker='o', ls='', color='#555', ms=6, label='large (active)'),
         Line2D([], [], marker='o', ls='', mfc='white', mec='#555', ms=5.5, label='small (non-active)')]
    ax.legend(handles=h, loc='center left', frameon=False, fontsize=8, handletextpad=.3, bbox_to_anchor=(0.0, 0.5))
    fig.tight_layout()
    save(fig, 'movielens_genre_split_Nov9')


def figure6():
    """Every layer as a circle; rows = genres; filled = large, open = small; color = group.
    Groups of the other methods are relabeled to match Algorithm 1 as closely as possible."""
    order = ['Action', 'Adventure', 'Sci-Fi', 'Thriller', 'Crime', 'Drama', 'Comedy', 'Romance']
    col = [BLUE, ORANGE, GREEN]
    methods = [('algorithm1', 'Algorithm 1'), ('all_layer', 'All-layer spectral clustering'),
               ('uase', 'UASE-based method')]
    genre = np.array(D1['genre'])
    typ = np.array(D1['type'])
    first = {g: np.flatnonzero(genre == g)[0] for g in GENRES}      # first large layer of each genre
    plt.rcParams.update({'font.size': 8.5, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.spines.bottom': False})
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.6), sharey=True)
    for c, m in enumerate((2, 3)):
        ref = np.array(D1['labels'][str(m)]['algorithm1'])
        # group 1 = Action's group, then Thriller's (M = 3), then Comedy's
        anchors = [ref[first[g]] for g in (['Action', 'Comedy'] if m == 2 else ['Action', 'Thriller', 'Comedy'])]
        ref = np.array([{g: i for i, g in enumerate(anchors)}[x] for x in ref])
        for r, (meth, name) in enumerate(methods):
            p = np.array(D1['labels'][str(m)][meth])
            best = max(itertools.permutations(range(m)), key=lambda perm: sum(perm[a] == b for a, b in zip(p, ref)))
            p = np.array([best[x] for x in p])
            ax = axes[r, c]
            for y, gname in enumerate(order):
                big = np.sort(p[(genre == gname) & (typ == 'large')])
                sm = np.sort(p[(genre == gname) & (typ == 'small')])
                for x, l in enumerate(big):
                    ax.scatter(x, -y, s=22, color=col[l], edgecolor=col[l], lw=.8, zorder=3)
                for x, l in enumerate(sm):
                    ax.scatter(5.2 + x, -y, s=22, facecolor='white', edgecolor=col[l], lw=1.1, zorder=3)
            ax.set_title(f'{name}, $M={m}$', fontsize=9, loc='left')
            ax.set_xticks([])
            ax.set_xlim(-.8, 37.8)
            ax.set_yticks(-np.arange(8))
            ax.set_yticklabels(order)
            ax.tick_params(axis='y', length=0)
            ax.set_ylim(-7.6, .6)
    h = [Line2D([], [], marker='o', ls='', color=col[i], ms=6, label=f'Group {i + 1}') for i in range(3)]
    h += [Line2D([], [], marker='o', ls='', color='#444', ms=6, label='large (active) layer'),
          Line2D([], [], marker='o', ls='', mfc='white', mec='#444', ms=6, mew=1.1, label='small (non-active) layer')]
    fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, -.005), fontsize=8,
               handletextpad=.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, .04, 1, 1), h_pad=.6, w_pad=1.2)
    save(fig, 'movielens_layer_groups')


def figure7():
    """Average ARI between the groups of all layers and their genres (+- one standard deviation)."""
    rows = []
    for s in (8, 16, 32, 64):
        for f in sorted((RESULTS / 'divisions' / f'S{s:03d}').glob('r*.json')):
            d = json.loads(f.read_text())
            for m in ('2', '3'):
                for meth, lab in d['labels'][m].items():
                    rows.append(dict(S=s, M=int(m), method=meth, ari=ari(lab, d['genre'])))
    t = pd.DataFrame(rows).groupby(['M', 'S', 'method']).ari.agg(['mean', 'std']).reset_index()
    style = {'algorithm1': ('Algorithm 1', BLUE, 'o', '-'),
             'all_layer': ('All-layer spectral clustering', ORANGE, 's', '--'),
             'uase': ('UASE-based method', GREEN, '^', '-.')}
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
                         'axes.spines.left': True, 'axes.spines.bottom': True})
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.7), sharex=True)
    for j, m in enumerate([2, 3]):
        for meth, (lab, col, mk, ls) in style.items():
            s = t[(t.M == m) & (t.method == meth)].sort_values('S')
            ax[j].plot(s.S, s['mean'], ls, color=col, marker=mk, ms=4.5, lw=1.6, label=lab)
            ax[j].fill_between(s.S, s['mean'] - s['std'], s['mean'] + s['std'], color=col, alpha=.15, lw=0)
        ax[j].set_xscale('log', base=2)
        ax[j].set_xticks([8, 16, 32, 64])
        ax[j].set_xticklabels(['8', '16', '32', '64'])
        ax[j].minorticks_off()
        ax[j].set_title(f'$M={m}$')
        ax[j].set_xlabel('Number of small layers per genre')
        ax[j].set_ylim(bottom=0)
        ax[j].grid(axis='y', color='#e5e5e3', lw=.6)
    ax[0].set_ylabel('ARI with genres')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, 'movielens_small_layers_Nov10')


if __name__ == '__main__':
    figure5()
    figure6()
    figure7()
