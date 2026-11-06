#!/usr/bin/env python3
"""Loss against number of evaluations for a calibration run.

Reads <run>/convergence.csv (one row per CMA-ES generation) and, for runs
made before that file existed, the progress lines CMA-ES prints every 20
generations in <run>.log. Top: the objective (best so far, and the best and
median of each generation). Bottom: the CMA-ES step size, which shrinks as the
search settles.

    python plot_convergence.py results/j14_chandra_dims --compare results/j12_dims
"""
import argparse, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7, "axes.linewidth": .6, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})


def from_log(path):
    """(evaluations, generation best, step size) from CMA-ES display lines."""
    rows = []
    for line in open(path):
        m = re.match(r"\s*(\d+)\s+(\d+)\s+([-+0-9.eE]+)\s+[-+0-9.eE]+\s+([-+0-9.eE]+)\s", line)
        if m:
            rows.append((int(m.group(2)), float(m.group(3)), float(m.group(4))))
    d = pd.DataFrame(rows, columns=["evaluations", "gen_best", "sigma"])
    d["best_so_far"] = d.gen_best.cummin()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = pd.read_csv(f"{a.run}/convergence.csv")
    fig, ax = plt.subplots(2, 1, figsize=(7.1, 4.6), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1]})
    A = ax[0]
    A.plot(d.evaluations, d.gen_median, color="#BBBBBB", lw=.5, label="median of each generation")
    A.plot(d.evaluations, d.gen_best, color="#7FA7C9", lw=.5, label="best of each generation")
    A.plot(d.evaluations, d.best_so_far, color="#1F4E79", lw=1.4,
           label=f"best so far, {os.path.basename(a.run)}")
    for r, col in zip(a.compare, ("#B8441F", "#1F7A6B", "#8A8F98")):
        log = r.rstrip("/") + ".log"
        if os.path.exists(f"{r}/convergence.csv"):
            c = pd.read_csv(f"{r}/convergence.csv")
        elif os.path.exists(log):
            c = from_log(log)
        else:
            continue
        A.plot(c.evaluations, c.best_so_far, color=col, lw=1.1, ls="--",
               label=f"best so far, {os.path.basename(r)}")
        ax[1].plot(c.evaluations, c.sigma, color=col, lw=.9, ls="--")
    lo = d.best_so_far.min()
    A.set_ylim(lo - 0.05 * lo, np.percentile(d.gen_median, 95) * 1.05)
    A.set_ylabel("training objective $F$ (mean over training vehicles)")
    A.legend(fontsize=6, frameon=False)
    A.set_title(f"final best $F$ = {lo:.4f} after {int(d.evaluations.iloc[-1]):,} evaluations, "
                f"{d.minutes.iloc[-1]:.0f} min", fontsize=7)
    ax[1].plot(d.evaluations, d.sigma, color="#1F4E79", lw=1)
    ax[1].set_yscale("log"); ax[1].set_ylabel("CMA-ES step size")
    ax[1].set_xlabel("objective evaluations (one evaluation = one simulation of the whole record)")
    for x in ax: x.tick_params(length=2, pad=1.5)
    fig.tight_layout()
    out = a.out or f"{a.run}/fig_convergence.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)
    # improvement over the last stretch of the run, a plain convergence check
    for frac in (0.25, 0.1):
        k = d.evaluations >= d.evaluations.iloc[-1] * (1 - frac)
        drop = d.best_so_far[~k].iloc[-1] - lo if (~k).any() else float("nan")
        print(f"improvement over the last {int(100*frac)}% of evaluations: {drop:.4f} "
              f"({100*drop/lo:.2f}% of final F)")


if __name__ == "__main__":
    main()
