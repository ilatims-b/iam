#!/usr/bin/env python3
"""Run report: convergence, class-wise and pooled errors, sensitivity profiles.

Builds paper_joint/runs_report/runs_report.pdf, with every figure in
paper_joint/runs_report/figs/. Runs still going are shown with their
convergence so far and an "ongoing" row; re-run this script when they finish
to fill them in. Errors of finished runs are recomputed by re-simulating the
whole record with the saved class parameters (so the frame counts needed for
pooled RMSE are available) and cached in figs/scores_<run>.json.

    python make_runs_report.py
"""
import json, os, shutil, subprocess
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iam_calib.config import PipelineConfig
from iam_calib.calibrate import default_params
from iam_calib.fastcore import PARAM_ORDER
from iam_calib import calibrate_joint as cj
from iam_calib.plot_convergence import from_log

OUT = "paper_joint/runs_report"; FIG = f"{OUT}/figs"
# run, short label, description
RUNS = [
    ("j9_signal", "J9", "all IDM and IAM parameters fitted, stop line, old wide boxes, "
                        "default vehicle sizes, 20\\,000 evaluations"),
    ("j11_allfree", "J11-free", "all fitted, stop line, realistic boxes, default sizes, 2\\,h"),
    ("j11_idmfixed", "J11-IDMfix", "IDM fixed at Chandrashekar's values for every class "
                                   "($v_0$\\,=\\,18.76, $T$\\,=\\,0.89, $s_0$\\,=\\,1.05, $a$\\,=\\,2.0, "
                                   "$b$\\,=\\,2.63, $\\delta$\\,=\\,4.42), IAM fitted, realistic boxes, "
                                   "default sizes, 2\\,h"),
    ("j12_dims", "J12-sizes", "all fitted, stop line, realistic boxes, new vehicle sizes, 2\\,h"),
    ("j13_duggal_idm", "J13-Duggal", "IDM fixed per class at Duggal's pairwise means "
                                     "(Auto, Bike, Car) and extrapolated values (LCV, Truck, "
                                     "MultiAxle), IAM fitted, realistic boxes, new sizes, 2\\,h"),
]
COL = ["#B8441F", "#1F4E79", "#1F7A6B", "#8A5FB0", "#C9A227"]
CLASSES = ["Auto", "Bike", "Car", "LCV", "Truck", "MultiAxle", "Trailer"]
plt.rcParams.update({"font.size": 7, "axes.linewidth": .6, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})


def conv(run):
    p = f"results/{run}/convergence.csv"
    if os.path.exists(p):
        return pd.read_csv(p)
    if os.path.exists(f"results/{run}.log"):
        return from_log(f"results/{run}.log")
    return None


def finished(run):
    return os.path.exists(f"results/{run}/class_params.json")


def scores(run):
    """Per class and pooled over all, for train and test: vehicles, mean F,
    pooled RMSE of longitudinal position, lateral position and speed."""
    cache = f"{FIG}/scores_{run}.json"
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(
            f"results/{run}/class_params.json"):
        return json.load(open(cache))
    rc = json.load(open(f"results/{run}/run_config.json"))
    cj.apply_bounds({k: tuple(v) for k, v in rc["overrides"].items()})
    cfg = PipelineConfig(); cfg.road.y_right, cfg.road.y_left, cfg.road.enabled = 0.0, 8.9, True
    cfg.vehicle_dims.update({k: tuple(v) for k, v in rc.get("dims", {}).items()})
    df = pd.read_csv("D3F1_lcl_smtF.csv")
    tr, te, _ = cj.whole_record(df, cfg, 1 - rc.get("train_frac", 0.7), rc.get("split_seed", 1),
                                rc.get("exit_signal", False))
    cp = json.load(open(f"results/{run}/class_params.json")); classes = sorted(cp)
    P = np.array([[{**default_params(), **cp[c]}[k] for k in PARAM_ORDER] for c in classes])
    grp, _ = cj.group_thin([tr], cfg.vehicle_dims, 10)
    cidx = {c: classes.index(grp[c]) for c in grp}
    out = {}
    for split, s in (("train", tr), ("test", te)):
        rows = cj.scene_errors(s, P, cidx, cfg.road, with_n=True)
        d = pd.DataFrame(rows, columns=["F", "x", "y", "v", "cls", "n"])
        d["F"] = d.F.where(np.isfinite(d.F), 60.0).clip(upper=60.0)
        res = {}
        for c, g in list(d.groupby("cls")) + [("all", d)]:
            pool = lambda k: float(np.sqrt((g.n * g[k] ** 2).sum() / g.n[np.isfinite(g[k])].sum()))
            res[c] = dict(n=int(len(g)), F=float(g.F.mean()), x=pool("x"), y=pool("y"), v=pool("v"))
        out[split] = res
    json.dump(out, open(cache, "w"), indent=1)
    return out


def fig_conv(run, i):
    d = conv(run)
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 3.6), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1]})
    if "gen_median" in d:
        ax[0].plot(d.evaluations, d.gen_median, color="#BBBBBB", lw=.5, label="median of each generation")
    ax[0].plot(d.evaluations, d.gen_best, color="#9DB9D5", lw=.5, label="best of each generation")
    ax[0].plot(d.evaluations, d.best_so_far, color=COL[i], lw=1.4, label="best so far")
    lo = d.best_so_far.min()
    hi = np.percentile(d.gen_median if "gen_median" in d else d.gen_best, 95)
    ax[0].set_ylim(lo * .95, max(hi * 1.05, lo * 1.2))
    ax[0].set_ylabel("training objective $F$"); ax[0].legend(fontsize=6, frameon=False)
    ax[1].plot(d.evaluations, d.sigma, color=COL[i], lw=1)
    ax[1].set_yscale("log"); ax[1].set_ylabel("CMA-ES step size")
    ax[1].set_xlabel("objective evaluations (one evaluation = one simulation of the whole record)")
    fig.tight_layout(); fig.savefig(f"{FIG}/conv_{run}.pdf", bbox_inches="tight"); plt.close(fig)
    last = d.evaluations.iloc[-1]
    k = d.evaluations >= last * 0.75
    gain = d.best_so_far[~k].iloc[-1] - lo if (~k).any() else float("nan")
    mins = d.minutes.iloc[-1] if "minutes" in d else float("nan")
    return dict(best=lo, evals=int(last), minutes=mins, sigma=float(d.sigma.iloc[-1]),
                gain25=100 * gain / lo)


def fig_conv_all(runs):
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, (run, lab, _) in enumerate(RUNS):
        d = conv(run)
        if d is None: continue
        ax.plot(d.evaluations, d.best_so_far, color=COL[i], lw=1.2,
                ls="-" if finished(run) else "--", label=lab + ("" if finished(run) else " (ongoing)"))
    ax.set_ylabel("best training objective $F$ so far"); ax.set_xlabel("objective evaluations")
    ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(f"{FIG}/conv_all.pdf", bbox_inches="tight"); plt.close(fig)


def fig_scores(sc):
    """Held-out pooled RMSE per class, one bar per finished run."""
    done = [(i, r, l) for i, (r, l, _) in enumerate(RUNS) if r in sc]
    cls = [c for c in CLASSES if any(c in sc[r]["test"] for _, r, _ in done)] + ["all"]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
    w = 0.8 / max(1, len(done))
    for A, (k, t) in zip(ax, (("x", "longitudinal position (m)"), ("y", "lateral position (m)"),
                              ("v", "speed (m/s)"))):
        for j, (i, r, l) in enumerate(done):
            A.bar(np.arange(len(cls)) + (j - (len(done) - 1) / 2) * w,
                  [sc[r]["test"].get(c, {}).get(k, np.nan) for c in cls], w,
                  color=COL[i], label=l)
        A.set_xticks(range(len(cls))); A.set_xticklabels(cls, rotation=45, ha="right", fontsize=6)
        A.set_title("held-out pooled RMSE, " + t, fontsize=7)
    ax[0].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/scores_test.pdf", bbox_inches="tight"); plt.close(fig)


def tex_scores(sc, run, lab):
    s = sc[run]
    cls = [c for c in CLASSES if c in s["train"]] + ["all"]
    rows = []
    for c in cls:
        a, b = s["train"][c], s["test"].get(c)
        cells = [f"{a['n']}", f"{a['F']:.2f}", f"{a['x']:.1f}", f"{a['y']:.2f}", f"{a['v']:.2f}"]
        cells += ([f"{b['n']}", f"{b['F']:.2f}", f"{b['x']:.1f}", f"{b['y']:.2f}", f"{b['v']:.2f}"]
                  if b else ["--"] * 5)
        name = "\\textbf{all (pooled)}" if c == "all" else c
        rows.append(("\\midrule " if c == "all" else "") + name + " & " + " & ".join(cells) + r" \\")
    return (r"""\begin{table}[H]\centering\footnotesize
\caption{%s: errors per class and over all vehicles. $F$ is the mean per-vehicle objective;
RMSEs are pooled over data points.}
\begin{tabular}{l rrrrr rrrrr}\toprule
 & \multicolumn{5}{c}{training vehicles} & \multicolumn{5}{c}{held-out vehicles}\\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}
class & $n$ & $F$ & long.\ (m) & lat.\ (m) & speed (m/s) & $n$ & $F$ & long.\ (m) & lat.\ (m) & speed (m/s)\\\midrule
""" % lab) + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n"


def main():
    os.makedirs(FIG, exist_ok=True)
    cv, sc = {}, {}
    for i, (run, lab, _) in enumerate(RUNS):
        if conv(run) is not None:
            cv[run] = fig_conv(run, i)
        if finished(run):
            print("scoring", run); sc[run] = scores(run)
    fig_conv_all(RUNS); fig_scores(sc)
    oat = "results/j11_allfree"
    for f in ("profiles", "rmse_long", "rmse_lat", "rmse_speed", "spread"):
        shutil.copy(f"{oat}/fig_oat_{f}.pdf", f"{FIG}/oat_{f}_j11_allfree.pdf")
    summ = pd.read_csv(f"{oat}/oat_summary.csv")

    L = [r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm,top=1.8cm]{geometry}
\usepackage{graphicx,booktabs,float,amsmath,caption}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}
\graphicspath{{figs/}}
\captionsetup{font=small}
\title{IAM calibration runs: convergence, errors and sensitivity}\date{\today}
\begin{document}\maketitle
\tableofcontents
\section{Runs}
All runs simulate the whole 781\,s record in one go, with the exit stop line except where noted,
and fit on 70\,\% of each class's vehicles (training) while the other 30\,\% are simulated but not
scored (held-out). $F$ is the mean per-vehicle objective, capped at 60.
Pooled RMSE: $\sqrt{\sum_i n_i\,\mathrm{rmse}_i^2/\sum_i n_i}$ over vehicles $i$ with $n_i$ frames.
\begin{table}[H]\centering\footnotesize
\begin{tabular}{l p{8.2cm} r r r r}\toprule
run & set-up & evaluations & best $F$ & last step size & gain, last 25\,\%\\\midrule"""]
    for run, lab, desc in RUNS:
        c = cv.get(run)
        st = "" if finished(run) else " \\emph{(ongoing)}"
        L.append(f"{lab}{st} & {desc} & " + (f"{c['evals']:,} & {c['best']:.3f} & {c['sigma']:.3f} & "
                                            f"{c['gain25']:.1f}\\,\\%" if c else "-- & -- & -- & --") + r" \\")
    L.append(r"""\bottomrule\end{tabular}
\caption{Runs in this report. ``Gain, last 25\,\%'': how much the best $F$ fell over the last quarter of
evaluations, as a share of the final $F$; near zero means converged.
The first fixed-IDM run, whose IDM exponent $\delta$ was not applied because of a stale compiled kernel,
is left out.}\end{table}

\section{Convergence}
\begin{figure}[H]\centering\includegraphics[width=.9\textwidth]{conv_all.pdf}
\caption{Best training objective so far, all runs. Dashed: still running.}\end{figure}""")
    for i, (run, lab, desc) in enumerate(RUNS):
        if run not in cv: continue
        c = cv[run]
        L.append(r"\subsection{%s%s}" % (lab, "" if finished(run) else " (ongoing)"))
        L.append(r"\begin{figure}[H]\centering\includegraphics[width=.82\textwidth]{conv_%s.pdf}" % run)
        L.append(r"\caption{%s: %s. Best $F$ = %.3f after %s evaluations.}\end{figure}"
                 % (lab, desc, c["best"], f"{c['evals']:,}"))

    L.append(r"\section{Errors by class and overall}")
    if sc:
        L.append(r"""\begin{table}[H]\centering\footnotesize
\caption{Pooled over all vehicles, finished runs.}
\begin{tabular}{l rrrr rrrr}\toprule
 & \multicolumn{4}{c}{training} & \multicolumn{4}{c}{held-out}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
run & $F$ & long.\ (m) & lat.\ (m) & speed (m/s) & $F$ & long.\ (m) & lat.\ (m) & speed (m/s)\\\midrule""")
        for run, lab, _ in RUNS:
            if run not in sc: continue
            a, b = sc[run]["train"]["all"], sc[run]["test"]["all"]
            L.append(f"{lab} & {a['F']:.2f} & {a['x']:.1f} & {a['y']:.2f} & {a['v']:.2f} & "
                     f"{b['F']:.2f} & {b['x']:.1f} & {b['y']:.2f} & {b['v']:.2f}" + r" \\")
        L.append(r"\bottomrule\end{tabular}\end{table}")
        L.append(r"""\begin{figure}[H]\centering\includegraphics[width=\textwidth]{scores_test.pdf}
\caption{Held-out pooled RMSE per class, finished runs. Trailer shares MultiAxle's parameters.}\end{figure}""")
        for run, lab, _ in RUNS:
            if run in sc: L.append(tex_scores(sc, run, lab))
    pend = [lab for run, lab, _ in RUNS if not finished(run)]
    if pend:
        L.append("Still running, errors to be added: " + ", ".join(pend) + ".")

    rng = summ.pivot(index="param", columns="cls", values="range")
    flat = summ.pivot(index="param", columns="cls", values="flat")
    order = [p for p in PARAM_ORDER if p in rng.index]
    cols = [c for c in CLASSES if c in rng.columns]
    L.append(r"""\section{Sensitivity: one-at-a-time profiles (J11-free)}
Around the J11-free fit, each parameter of each class vector is swept across its search box in steps
of its granularity, everything else held at the fit, and the whole record re-simulated
(3\,151 simulations). Neighbouring grid points differ by about 0.26 in $F$ (2\,\%) from simulation
noise alone, so a range below about 1.5 is not distinguishable from noise.
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{oat_profiles_j11_allfree.pdf}
\caption{y-axis: training objective $F$ with that one parameter set to the x value, minus $F$ at the
fit (12.487). Each point is one simulation of the whole record (deterministic, so one value per point,
not a distribution); dots: fitted values.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{oat_spread_j11_allfree.pdf}
\caption{Distribution of training $F$ over all the values tried for each parameter, one box per class
(box: quartiles; whiskers: 1.5 IQR; points: outliers). A tall box means moving that parameter across
its box changes the fit a lot; a box hugging the dashed line means the data hardly constrain it.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{oat_rmse_long_j11_allfree.pdf}
\caption{Same sweeps, y-axis: pooled RMSE of longitudinal position on the training vehicles
(absolute; grey line: value at the fit).}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{oat_rmse_lat_j11_allfree.pdf}
\caption{Same sweeps, y-axis: pooled RMSE of lateral position on the training vehicles.}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{oat_rmse_speed_j11_allfree.pdf}
\caption{Same sweeps, y-axis: pooled RMSE of speed on the training vehicles.}\end{figure}
\begin{table}[H]\centering\footnotesize
\caption{Range of $F$ across the box (how much each parameter can matter) and, in brackets, the
share of the box within 1\,\% of the best $F$ on the profile (1 = unconstrained).}
\begin{tabular}{l """ + "r" * len(cols) + r"""}\toprule
parameter & """ + " & ".join(cols) + r"""\\\midrule""")
    for p in order:
        L.append(p.replace("_", r"\_") + " & " + " & ".join(
            f"{rng.loc[p, c]:.2f} ({flat.loc[p, c]:.2f})" for c in cols) + r" \\")
    L.append(r"\bottomrule\end{tabular}\end{table}")
    L.append(r"\end{document}")
    open(f"{OUT}/runs_report.tex", "w").write("\n".join(L))
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "runs_report.tex"], cwd=OUT,
                       stdout=subprocess.DEVNULL)
    for ext in ("aux", "log", "out", "toc"):
        if os.path.exists(f"{OUT}/runs_report.{ext}"): os.remove(f"{OUT}/runs_report.{ext}")
    print("wrote", f"{OUT}/runs_report.pdf")


if __name__ == "__main__":
    main()
