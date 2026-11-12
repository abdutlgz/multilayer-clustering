#!/usr/bin/env python3
"""Figures of Section 4 (simulation study) from the summary CSV files.

usage: python3 make_figures.py [results_dir] [out_dir]
  results_dir  default: ../results (the stored results of the paper)
  out_dir      default: <results_dir>/figures

  sim_threshold_curves.pdf  Figure 1: error vs fixed threshold, rho = 0.05
  sim_error_by_n_L_rho.pdf  Figure 2: total error vs n, L = 150, three methods
  sim_runtime_memory.pdf    Figure 3: error, running time and memory, settings 1-8
  sim_active_sweep.pdf      Figure 4: widest 10%, ..., 100% of the layers active
Each figure is also saved as PNG.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "results")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(res, "figures")
os.makedirs(out, exist_ok=True)

curves = pd.read_csv(f"{res}/accuracy/curves.csv")
acc = pd.read_csv(f"{res}/accuracy/settings_summary.csv")
runtime = pd.read_csv(f"{res}/runtime/summary.csv")
sweep = pd.read_csv(f"{res}/active_sweep/runs.csv")

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#888888", "axes.linewidth": 0.6,
                     "xtick.color": "#444444", "ytick.color": "#444444", "pdf.fonttype": 42})


def style(ax):
    ax.grid(True, color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    fig.savefig(f"{out}/{name}.pdf", metadata={"CreationDate": None})  # no date, so rebuilt PDFs are byte-identical
    fig.savefig(f"{out}/{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", f"{out}/{name}.pdf")


# Figure 1: mean error vs fixed threshold beta, rho = 0.05; the band is +-2 standard errors.
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
for ax, (n, L) in zip(axes, [(200, 100), (200, 250), (300, 100), (300, 250)]):
    d = curves[(curves.n == n) & (curves.L == L) & (curves.rho == 0.05)].sort_values("beta")
    d = d[d.n_valid > 0]
    ax.fill_between(d.beta, 100 * (d.mean_err - 2 * d.se_err), 100 * (d.mean_err + 2 * d.se_err),
                    color="#2a78d6", alpha=0.18, lw=0)
    ax.plot(d.beta, 100 * d.mean_err, color="#2a78d6", marker="o", ms=3, lw=1.6,
            label="Error for a fixed threshold")
    best = d.loc[d.mean_err.idxmin()]
    ax.axvline(best.beta, color="#444444", ls="--", lw=1.0, label="Best threshold (uses true labels)")
    q = acc[(acc.n == n) & (acc.L == L) & (acc.rho == 0.05)].iloc[0]
    ax.plot(q.rule_beta, 100 * q.rule_err, ls="none", marker="D", ms=6, color="#eb6834",
            mec="white", mew=0.8, label=r"Our threshold $n_l\geq\log n/\hat\rho$")
    ax.set_title(f"$n={n}$, $L={L}$", fontsize=9)
    ax.set_xlim(0.27, 1.13)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9, 1.1])
    ax.set_xticklabels(["0.3", "0.5", "0.7", "0.9", "1.1"])
    ax.set_xlabel(r"$\beta_\star$")
    style(ax)
axes[0].set_ylabel("Mean error (%)")
axes[0].set_ylim(0, 60)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.1, 1, 1))
save(fig, "sim_threshold_curves")

# Figure 2: total error vs n, L = 150, one panel per rho; the band is +-2 standard errors.
series = [("rule_err", "rule_se", "Algorithm 1", "#2a78d6", "o"),
          ("all_layer_err", "all_layer_se", "All-layer clustering", "#1baf7a", "^"),
          ("uase_err", "uase_se", "UASE-based", "#eda100", "D")]
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.55), sharey=True)
for ax, rho in zip(axes, [0.05, 0.10, 0.15, 0.20]):
    d = acc[(acc.L == 150) & np.isclose(acc.rho, rho)].sort_values("n")
    for col, se, label, color, marker in series:
        ax.fill_between(d.n, 100 * (d[col] - 2 * d[se]), 100 * (d[col] + 2 * d[se]),
                        color=color, alpha=0.18, lw=0)
        ax.plot(d.n, 100 * d[col], color=color, marker=marker, lw=1.6, ms=4,
                mec="white", mew=0.6, label=label)
    ax.set_title(rf"$\rho_{{\mathrm{{sim}}}}={rho:.2f}$", fontsize=9)
    ax.set_xticks([100, 200, 300, 400])
    ax.set_xlabel(r"$n$")
    style(ax)
axes[0].set_ylabel("Mean total error (%)")
axes[0].set_ylim(0, 62)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.09, 1, 1))
save(fig, "sim_error_by_n_L_rho")

# Figure 3: average error, median time and median memory; one parameter changes per column.
d = runtime
series = [("Algorithm1", "Algorithm 1", "#2a78d6", "o", "-"),
          ("All_layer", "All-layer clustering", "#1baf7a", "^", "-"),
          ("UASE", "UASE-based", "#eda100", "D", "-"),
          ("Input_only", "No method", "#666666", None, ":")]
cols = [("n", (d.L == 3000) & np.isclose(d.rho, 0.01), r"$n$ ($L=3000$, $\rho_{\mathrm{sim}}=0.01$)",
         [2000, 3000, 4000], False),
        ("L", (d.n == 3000) & np.isclose(d.rho, 0.01), r"$L$ ($n=3000$, $\rho_{\mathrm{sim}}=0.01$)",
         [1000, 2000, 3000, 4000], False),
        ("rho", (d.n == 3000) & (d.L == 3000), r"$\rho_{\mathrm{sim}}$ ($n=3000$, $L=3000$)",
         [0.005, 0.01, 0.02], True)]
fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.6), sharex="col")
for j, (x, mask, xlabel, ticks, logx) in enumerate(cols):
    s = d[mask]
    for meth, label, color, marker, ls in series:
        e = s[s.method == meth].sort_values(x)
        kw = dict(color=color, marker=marker, ms=4, lw=1.6 if marker else 1.4, ls=ls,
                  mec="white", mew=0.6, label=label)
        if meth != "Input_only":
            axes[0, j].plot(e[x], 100 * e.err_mean, **kw)
            axes[1, j].plot(e[x], e.time_median, **kw)
        axes[2, j].plot(e[x], e.mem_gib_median, **kw)
    for i in range(3):
        ax = axes[i, j]
        style(ax)
        if logx:
            ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
    axes[0, j].set_ylim(-2, 62)
    axes[1, j].set_yscale("log")
    axes[1, j].set_yticks([10, 20, 50, 100, 200])
    axes[1, j].set_yticklabels(["10", "20", "50", "100", "200"])
    axes[1, j].minorticks_off()
    axes[1, j].set_ylim(7, 220)
    axes[2, j].set_ylim(1.5, 9.0)
    axes[2, j].set_xlabel(xlabel)
axes[0, 0].set_ylabel("Error (%)")
axes[1, 0].set_ylabel("Time (s)")
axes[2, 0].set_ylabel("Memory (GiB)")
h, l = axes[2, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.035, 1, 1))
save(fig, "sim_runtime_memory")

# Figure 4: active-layer sweep, setting n = 3000, L = 1000, rho = 0.05.
r = sweep.copy()
r["mem"] = r.peak_rss_kb / 1024**2
r["over"] = r.mem - r.seed.map(r[r.method == "Input_only"].set_index("seed").mem)
r["pct"] = 100 * r.active_count / r.L
s = r[r.method != "Input_only"].groupby("method").agg(
    pct=("pct", "mean"), err=("err", "mean"), t=("seconds", "median"), m=("over", "median"))
top, alg = s.loc[[f"Top{p}" for p in range(10, 101, 10)]], s.loc["Algorithm1"]
blue, red = "#2a78d6", "#d03b3b"
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
for ax, (y, k, lab) in zip(axes, [("err", 100, "Error (%)"), ("t", 1, "Time (s)"),
                                  ("m", 1, "Extra memory (GiB)")]):
    ax.plot(top.pct, k * top[y], color=blue, marker="o", ms=4, lw=1.6, mec="white", mew=0.6,
            label="Widest layers active")
    ax.axvline(alg.pct, color=red, lw=0.8, ls=":", zorder=1)
    ax.plot([alg.pct], [k * alg[y]], ls="none", marker="D", ms=6, color=red, mec="white", mew=0.8,
            label=r"Threshold $n_l\geq\log n/\hat\rho$", zorder=3)
    ax.set_xticks(range(20, 101, 20))
    ax.set_xlim(5, 105)
    ax.set_xlabel("Active layers (%)")
    ax.set_ylabel(lab)
    style(ax)
axes[0].set_ylim(-1, 28)
axes[1].set_ylim(0, 30)
axes[2].set_ylim(0, 0.22)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "sim_active_sweep")
