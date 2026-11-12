#!/usr/bin/env python3
"""Numbers of the tables of Section 4, printed from the summary CSV files.

usage: python3 make_tables.py [results_dir]      (default: ../results)
  Table 1  active layers, clustering and classification error of Algorithm 1
  Table 2  error, running time and memory for three settings
  Sweep    the numbers quoted in the text about Figure 4
"""
import os
import sys
import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "results")
acc = pd.read_csv(f"{res}/accuracy/settings_summary.csv")
runtime = pd.read_csv(f"{res}/runtime/summary.csv")
sweep = pd.read_csv(f"{res}/active_sweep/summary.csv")

print("Table 1 (n, L, rho changed one at a time from n=250, L=150, rho=0.10; 100 networks each)")
cols = [(n, 150, 0.10) for n in (100, 200, 300, 400)] + \
       [(250, L, 0.10) for L in (100, 150, 200, 250)] + \
       [(250, 150, r) for r in (0.05, 0.10, 0.15, 0.20)]
rows = {"Active layers": [], "Clustering error": [], "Classification error": []}
for n, L, rho in cols:
    q = acc[(acc.n == n) & (acc.L == L) & np.isclose(acc.rho, rho)].iloc[0]
    rows["Active layers"].append(f"{q.rule_count:.1f}")
    rows["Clustering error"].append(f"{100 * q.rule_err_active:.1f}")
    rows["Classification error"].append(f"{100 * q.rule_err_nonactive:.1f}")
for name, v in rows.items():
    print(f"{name}\n & " + " &\n & ".join(" & ".join(v[i:i + 4]) for i in range(0, 12, 4)) + r"\\")

print("\nTable 2 (n = 3000; average error, median time, median peak memory over 10 networks)")
names = {"Algorithm1": r"Algorithm~\ref{alg:layer}", "All_layer": "All-layer clustering",
         "UASE": "UASE-based clustering"}
for sid, title in [(6, r"$\rho_{\mathrm{sim}}=0.05$, $L=1000$"), (12, r"$\rho_{\mathrm{sim}}=0.01$, $L=1000$"),
                   (13, r"$\rho_{\mathrm{sim}}=0.01$, $L=3000$")]:
    print(r"\multicolumn{4}{l}{" + title + r"}\\")
    for m, label in names.items():
        q = runtime[(runtime.id == sid) & (runtime.method == m)].iloc[0]
        print(f"\\quad {label:<28} & {100 * q.err_mean:.1f} & {q.time_median:.2f} & {q.mem_gib_median:.2f}\\\\")

print("\nSettings 1-8 of Figure 3 (paper setting: id in runtime/settings.csv)")
for k, sid in enumerate([13, 14, 15, 12, 16, 17, 18, 19], 1):
    a = runtime[(runtime.id == sid) & (runtime.method == "Algorithm1")].iloc[0]
    b = runtime[(runtime.id == sid) & (runtime.method == "All_layer")].iloc[0]
    print(f"  setting {k} (id {sid}): n={a.n} L={a.L} rho={a.rho}: Algorithm 1 error {100 * a.err_mean:.1f}%, "
          f"{a.time_median:.2f} s, {b.time_median / a.time_median:.2f}x faster than all-layer")

print("\nActive-layer sweep (n=3000, L=1000, rho=0.05; Figure 4)")
print(f"  {'method':<11}{'active %':>9}{'error %':>9}{'time s':>9}{'extra GiB':>11}")
for _, q in sweep[sweep.method != "Input_only"].iterrows():
    print(f"  {q.method:<11}{q.active_mean / q.L * 100:>9.1f}{100 * q.err_mean:>9.2f}"
          f"{q.time_median:>9.2f}{q.mem_over_none_median:>11.3f}")
