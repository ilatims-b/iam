#!/usr/bin/env python3
"""Figures and tables for paper_joint/calib_report/calib_report.tex.

The text of the report is written by hand in calib_report.tex; this script
only regenerates what it \\input's or \\includegraphics's, all into
paper_joint/calib_report/figs/:

  tab_bounds.tex          search box, scale and granularity of every parameter
  tab_runs.tex            F and pooled RMSE of the three compared runs
  tab_cls_<key>.tex       the same per class, one table per run
  tab_par_<key>.tex       fitted (and fixed) class parameters, one per run
  fig_st_x.pdf            space-time x(t), measured vs calibrated  (all-free run)
  fig_st_y.pdf            lateral position y(t), measured vs calibrated
  fig_xy.pdf              plan view, best and worst held-out vehicles per class
  fig_classspeed.pdf      speed distribution per class
  fig_oat_spread.pdf      sensitivity: spread of F over each parameter's box
  fig_conv_<key>.pdf      convergence of each run

    python make_calib_report.py
"""
import json, os, shutil
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from iam_calib.config import SPEC_BY_NAME, REALISTIC_BOUNDS, GRANULARITY, PARAM_SPECS
import iam_calib.make_figs as mf
from iam_calib import calibrate_joint as cj
from iam_calib.config import PipelineConfig
from iam_calib.calibrate import default_params
from iam_calib.fastcore import PARAM_ORDER

OUT = "paper_joint/calib_report"; FIG = f"{OUT}/figs"
FREE = "j12_dims"
CHANDRA = "j14_chandra_dims"
RUNS = [("chandra", CHANDRA, "Fixed IDM, single set \\cite{chandra}"),
        ("duggal", "j13_duggal_idm", "Fixed IDM, per class \\cite{duggal}"),
        ("free", FREE, "All parameters free")]
CLASSES = ["Auto", "Bike", "Car", "LCV", "Truck", "MultiAxle", "Trailer"]
CONV_COL = ["#1F4E79", "#1F7A6B", "#8A5FB0"]


def conv(run):
    return pd.read_csv(f"results/{run}/convergence.csv")


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


def fig_conv(run, key, col):
    d = conv(run)
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 3.6), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1]})
    ax[0].plot(d.evaluations, d.gen_median, color="#BBBBBB", lw=.5, label="median of each generation")
    ax[0].plot(d.evaluations, d.gen_best, color="#9DB9D5", lw=.5, label="best of each generation")
    ax[0].plot(d.evaluations, d.best_so_far, color=col, lw=1.4, label="best so far")
    lo = d.best_so_far.min()
    hi = np.percentile(d.gen_median, 95)
    ax[0].set_ylim(lo * .95, max(hi * 1.05, lo * 1.2))
    ax[0].set_ylabel("training objective $F$"); ax[0].legend(fontsize=6, frameon=False)
    ax[1].plot(d.evaluations, d.sigma, color=col, lw=1)
    ax[1].set_yscale("log"); ax[1].set_ylabel("CMA-ES step size")
    ax[1].set_xlabel("objective evaluations (one evaluation = one simulation of the whole record)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_conv_{key}.pdf", bbox_inches="tight"); plt.close(fig)

IDM = ["v0", "T", "s0", "a", "b", "delta"]
IAM = ["s0y", "s0yLat", "sensLat", "tauLatOVM", "pushLong", "pushLat", "accBiasLat",
       "accLongBRef", "accLatBRef", "s0yB", "s0yLatB"]
SYM = {"v0": r"$v_0$", "T": r"$T$", "s0": r"$s_0$", "a": r"$a$", "b": r"$b$", "delta": r"$\delta$",
       "s0y": r"$s_{0y}$", "s0yLat": r"$s_{0y,\mathrm{lat}}$", "sensLat": r"$\sigma$",
       "tauLatOVM": r"$\tau$", "pushLong": r"$\lambda_\parallel$", "pushLat": r"$\lambda_\perp$",
       "accBiasLat": r"$\beta$", "accLongBRef": r"$f_B$", "accLatBRef": r"$g_B$",
       "s0yB": r"$s_{0y,B}$", "s0yLatB": r"$s_{0y,\mathrm{lat}B}$"}
UNIT = {"v0": "m/s", "T": "s", "s0": "m", "a": "m/s$^2$", "b": "m/s$^2$", "delta": "--",
        "s0y": "m", "s0yLat": "m", "sensLat": "s", "tauLatOVM": "s", "pushLong": "--",
        "pushLat": "--", "accBiasLat": "m/s$^2$", "accLongBRef": "m/s$^2$",
        "accLatBRef": "m/s$^2$", "s0yB": "m", "s0yLatB": "m"}


def g(x, nd=2):
    return f"{x:.{nd}f}"


# --------------------------------------------------------------------------
# tables
def tab_bounds():
    rows = []
    for blk, names in (("IDM (car-following core)", IDM), ("IAM (lateral, anisotropy, bias, road edge)", IAM)):
        rows.append(r"\multicolumn{6}{@{}l}{\emph{%s}}\\" % blk)
        for n in names:
            lo, hi, log = REALISTIC_BOUNDS[n]; sp = SPEC_BY_NAME[n]
            gr = GRANULARITY[n]
            gr = gr.replace("%", r"\,\%") if gr.endswith("%") else gr + " " + UNIT[n].replace("--", "")
            start = "per class$^{a}$" if n == "v0" else f"{min(max(sp.default, lo), hi):g}"
            rows.append(f"{SYM[n]} & {UNIT[n]} & [{lo:g}, {hi:g}] & {'log' if log else 'linear'} & "
                        f"{start} & {gr}" + r"\\")
    body = "\n".join(rows)
    open(f"{FIG}/tab_bounds.tex", "w").write(
        r"\begin{tabular}{@{}llllll@{}}\toprule" "\n"
        r"parameter & unit & search box & scale & start & granularity\\\midrule" "\n"
        + body + "\n" r"\bottomrule\end{tabular}" "\n")


def constant_velocity(run):
    """Baseline: each vehicle keeps its measured entry speed, longitudinal and
    lateral, for its whole time on the section. Scored like the simulation (same
    vehicles, frames on which the vehicle is present, at least 50 frames): mean
    capped F and pooled RMSE, for training and held-out vehicles."""
    rc = json.load(open(f"results/{run}/run_config.json"))
    cfg = PipelineConfig(); cfg.road.y_right, cfg.road.y_left, cfg.road.enabled = 0.0, 8.9, True
    cfg.vehicle_dims.update({k: tuple(v) for k, v in rc.get("dims", {}).items()})
    tr, te, _ = cj.whole_record(pd.read_csv("D3F1_lcl_smtF.csv"), cfg, 1 - rc.get("train_frac", 0.7),
                                rc.get("split_seed", 1), rc.get("exit_signal", False))
    out = {}
    for split, sc_ in (("train", tr), ("test", te)):
        F, n, ex, ey, ev = [], [], [], [], []
        for i in range(sc_["X"].shape[0]):
            m = sc_["present"][i]
            if m.sum() < 50 or not sc_["score"][i]: continue
            f = np.where(m)[0]; k0 = f[0]; tt = (f - k0) * sc_["dt"]
            x = sc_["X"][i, k0] + sc_["VX"][i, k0] * tt
            y = sc_["Y"][i, k0] + sc_["VY"][i, k0] * tt
            v = np.full(len(f), sc_["VX"][i, k0])
            rx = np.sqrt(np.mean((x - sc_["X"][i, f]) ** 2)); ry = np.sqrt(np.mean((y - sc_["Y"][i, f]) ** 2))
            rv = np.sqrt(np.mean((v - sc_["VX"][i, f]) ** 2))
            rms = max(np.sqrt(np.mean(sc_["VX"][i, f] ** 2)), 0.5)
            F.append(min(rx / 2.0 + ry / 0.5 + rv / rms, 60.0)); n.append(len(f))
            ex.append(rx); ey.append(ry); ev.append(rv)
        n = np.array(n, float)
        pool = lambda e: float(np.sqrt(np.sum(n * np.array(e) ** 2) / n.sum()))
        out[split] = dict(n=len(n), F=float(np.mean(F)), x=pool(ex), y=pool(ey), v=pool(ev))
    return out


def tab_runs(sc):
    rows = []
    for key, run, lab in RUNS:
        a, b = sc[key]["train"]["all"], sc[key]["test"]["all"]
        rows.append(f"{lab} & {g(a['F'])} & {g(a['x'],1)} & {g(a['y'])} & {g(a['v'])} & "
                    f"{g(b['F'])} & {g(b['x'],1)} & {g(b['y'])} & {g(b['v'])}" + r"\\")
    cv = constant_velocity(FREE); a, b = cv["train"], cv["test"]
    rows.append(r"\midrule" "\n" f"Constant velocity & {g(a['F'])} & {g(a['x'],1)} & {g(a['y'])} & "
                f"{g(a['v'])} & {g(b['F'])} & {g(b['x'],1)} & {g(b['y'])} & {g(b['v'])}" + r"\\")
    json.dump(cv, open(f"{FIG}/baseline_cv.json", "w"), indent=1)
    open(f"{FIG}/tab_runs.tex", "w").write(
        r"\begin{tabular}{@{}l rrrr rrrr@{}}\toprule" "\n"
        r" & \multicolumn{4}{c}{training vehicles (386)} & \multicolumn{4}{c}{held-out vehicles (165)}\\" "\n"
        r"\cmidrule(lr){2-5}\cmidrule(l){6-9}" "\n"
        r"run & $F$ & long.\ (m) & lat.\ (m) & speed (m/s) & $F$ & long.\ (m) & lat.\ (m) & speed (m/s)\\\midrule" "\n"
        + "\n".join(rows) + "\n" r"\bottomrule\end{tabular}" "\n")


def tab_cls(sc, key):
    s = sc[key]; rows = []
    for c in [c for c in CLASSES if c in s["train"]] + ["all"]:
        a, b = s["train"][c], s["test"].get(c)
        cells = [str(a["n"]), g(a["F"]), g(a["x"], 1), g(a["y"]), g(a["v"])]
        cells += [str(b["n"]), g(b["F"]), g(b["x"], 1), g(b["y"]), g(b["v"])] if b else ["--"] * 5
        rows.append((r"\midrule all & " if c == "all" else f"{c} & ") + " & ".join(cells) + r"\\")
    open(f"{FIG}/tab_cls_{key}.tex", "w").write(
        r"\begin{tabular}{@{}l rrrrr rrrrr@{}}\toprule" "\n"
        r" & \multicolumn{5}{c}{training} & \multicolumn{5}{c}{held out}\\" "\n"
        r"\cmidrule(lr){2-6}\cmidrule(l){7-11}" "\n"
        r"class & $n$ & $F$ & long. & lat. & speed & $n$ & $F$ & long. & lat. & speed\\\midrule" "\n"
        + "\n".join(rows) + "\n" r"\bottomrule\end{tabular}" "\n")


def tab_par(key, run):
    cp = json.load(open(f"results/{run}/class_params.json"))
    rc = json.load(open(f"results/{run}/run_config.json"))
    free = set(rc.get("free", []))
    cls = [c for c in CLASSES if c in cp]
    rows = []
    for blk, names in (("IDM", IDM), ("IAM", IAM)):
        rows.append(r"\multicolumn{%d}{@{}l}{\emph{%s}}\\" % (len(cls) + 1, blk))
        for n in names:
            cells = []
            for c in cls:
                v = cp[c].get(n, SPEC_BY_NAME[n].default)
                t = f"{v:.2f}" if abs(v) >= 0.095 or v == 0 else f"{v:.3f}"
                if n not in free:
                    t = r"\textit{" + t + "}"
                cells.append(t)
            rows.append(f"{SYM[n]} & " + " & ".join(cells) + r"\\")
    open(f"{FIG}/tab_par_{key}.tex", "w").write(
        r"\begin{tabular}{@{}l" + "r" * len(cls) + r"@{}}\toprule" "\n"
        r"parameter & " + " & ".join(cls) + r"\\\midrule" "\n"
        + "\n".join(rows) + "\n" r"\bottomrule\end{tabular}" "\n")


DUGGAL = {  # pairwise IDM means and SDs by follower type (Duggal, constrained pairwise table)
    "Auto": dict(v0=(20.27, 7.65), T=(0.31, 0.28), s0=(0.64, 0.68), a=(2.57, 1.34), b=(0.93, 1.08), delta=(4.50, 1.35)),
    "Bike": dict(v0=(21.28, 6.83), T=(0.25, 0.24), s0=(1.04, 1.51), a=(1.86, 1.27), b=(1.28, 1.46), delta=(4.43, 1.23)),
    "Car": dict(v0=(21.43, 6.77), T=(0.29, 0.32), s0=(1.00, 1.36), a=(2.09, 1.30), b=(1.35, 1.60), delta=(4.33, 1.31))}


def tab_duggal_cmp():
    """Free-fit IDM values against the pairwise means: for each class a row of
    fitted values, a row of pairwise means and a row of the difference in SDs."""
    cp = json.load(open(f"results/{FREE}/class_params.json"))
    rows = []
    for c, d in DUGGAL.items():
        fit = [cp[c].get(n, 4.0) for n in IDM]
        rows.append(f"{c} & free fit & " + " & ".join(f"{v:.2f}" for v in fit) + r"\\")
        rows.append(r" & pairwise mean & " + " & ".join(f"{d[n][0]:.2f}" for n in IDM) + r"\\")
        rows.append(r" & difference (SD) & " + " & ".join(f"{(v - d[n][0]) / d[n][1]:+.1f}"
                                                         for v, n in zip(fit, IDM)) + r"\\")
        if c != list(DUGGAL)[-1]: rows.append(r"\midrule")
    open(f"{FIG}/tab_duggal_cmp.tex", "w").write(
        r"\begin{tabular}{@{}ll" + "r" * len(IDM) + r"@{}}\toprule" "\n"
        r"class & & " + " & ".join(SYM[n] for n in IDM) + r"\\\midrule" "\n"
        + "\n".join(rows) + "\n" r"\bottomrule\end{tabular}" "\n")


# --------------------------------------------------------------------------
# figures of the all-free run
def lateral_time(rec, dt, stops, out, step=10):
    """y(t) per class, measured left and simulated right, coloured by speed."""
    from matplotlib.collections import LineCollection
    r = rec[rec.frame % step == 0]
    order = mf.ORDER
    fig, ax = plt.subplots(len(order), 2, figsize=(7.1, 1.18 * len(order)), sharex=True, sharey=True)
    norm = plt.Normalize(0, 8); lc = None
    for k, c in enumerate(order):
        gg = r[r.cls == c]; gm = gg[gg.pres]; gs = gg[gg.on]
        for col, (t, y, v, vid) in enumerate((
                (gm.frame * dt, gm.y_m, gm.v_m, gm.vid.astype(str).values),
                (gs.frame * dt, gs.y_s, gs.v_s, gs.vid.astype(str).values))):
            a = ax[k, col]
            for p, q in stops:
                a.axvspan(p, q, color="#C0392B", alpha=.10, lw=0, zorder=0)
            sg, cv = mf._segments(np.asarray(t), np.asarray(y), np.asarray(v), vid)
            if sg:
                lc = LineCollection(np.concatenate(sg), cmap="viridis", norm=norm,
                                    linewidths=.45, rasterized=True)
                lc.set_array(np.concatenate(cv)); a.add_collection(lc)
            a.set_xlim(0, 781); a.set_ylim(*mf.EDGES)
            a.set_yticks([0, 4, 8]); a.tick_params(length=2, pad=1.5, labelsize=6)
            if k == 0: a.set_title("measured" if col == 0 else "calibrated (all parameters free)", fontsize=7.5)
        ax[k, 0].set_ylabel(f"{c}\n(n={rec[rec.cls == c].vid.nunique()})", fontsize=6.5,
                            rotation=0, ha="right", va="center", labelpad=16)
    for a in ax[-1]: a.set_xlabel("time (s)")
    fig.text(.5, -.005, "red bands: periods when the exit stop line is on", ha="center",
             fontsize=6.5, color="#C0392B")
    cb = fig.colorbar(lc, ax=ax, pad=.012, fraction=.016)
    cb.set_label("longitudinal speed (m/s)", fontsize=6.5); cb.ax.tick_params(labelsize=5.5, length=2)
    fig.supylabel("lateral position from the median kerb (m)", fontsize=7, x=.005)
    fig.savefig(f"{out}/fig_st_y.pdf", bbox_inches="tight", dpi=200); plt.close(fig)


def all_spacetime(rec, dt, x0, stops, out, step=10):
    """One space-time diagram of all vehicles together, measured above and
    calibrated below, coloured by longitudinal speed: x(t) in fig_all_x.pdf and
    y(t) in fig_all_y.pdf."""
    from matplotlib.collections import LineCollection
    r = rec[rec.frame % step == 0]
    gm, gs = r[r.pres], r[r.on]
    norm = plt.Normalize(0, 8)
    for key, ylab, ylim, name in (("x", "position along the section (m)", (0, mf.SEC), "fig_all_x.pdf"),
                                  ("y", "lateral position from the median kerb (m)", mf.EDGES, "fig_all_y.pdf")):
        fig, ax = plt.subplots(2, 1, figsize=(7.1, 6.2), sharex=True, sharey=True)
        lc = None
        for a, g, (vk, yk), title in ((ax[0], gm, ("v_m", f"{key}_m"), "measured"),
                                      (ax[1], gs, ("v_s", f"{key}_s"), "calibrated (all parameters free)")):
            for p, q in stops:
                a.axvspan(p, q, color="#C0392B", alpha=.10, lw=0, zorder=0)
            y = g[yk] - 0.5 * g.L - x0 if key == "x" else g[yk]
            sg, cv = mf._segments(np.asarray(g.frame * dt), np.asarray(y), np.asarray(g[vk]),
                                  g.vid.astype(str).values)
            lc = LineCollection(np.concatenate(sg), cmap="viridis", norm=norm, linewidths=.3,
                                rasterized=True)
            lc.set_array(np.concatenate(cv)); a.add_collection(lc)
            a.set_xlim(0, 781); a.set_ylim(*ylim); a.set_title(title, fontsize=8)
            a.set_ylabel(ylab, fontsize=7); a.tick_params(length=2, pad=1.5, labelsize=6.5)
        ax[1].set_xlabel("time (s)")
        cb = fig.colorbar(lc, ax=ax, pad=.012, fraction=.02)
        cb.set_label("longitudinal speed (m/s)", fontsize=7); cb.ax.tick_params(labelsize=6, length=2)
        fig.text(.45, .005, "red bands: periods when the exit stop line is on", ha="center",
                 fontsize=6.5, color="#C0392B")
        fig.savefig(f"{out}/{name}", bbox_inches="tight", dpi=200); plt.close(fig)


def vehicle_scores(rec):
    """Per held-out vehicle: F as in the objective, over the frames its measured
    twin is present, and how far it travels."""
    t = rec[(rec.split == "test") & rec.pres]
    out = []
    for vid, h in t.groupby("vid"):
        if len(h) < 250: continue
        ex = np.sqrt(np.mean((h.x_s - h.x_m) ** 2)); ey = np.sqrt(np.mean((h.y_s - h.y_m) ** 2))
        ev = np.sqrt(np.mean((h.v_s - h.v_m) ** 2)); rv = max(np.sqrt(np.mean(h.v_m ** 2)), 0.5)
        out.append(dict(vid=vid, cls=h.cls.iloc[0], F=ex / 2 + ey / 0.5 + ev / rv,
                        ex=ex, ey=ey, dist=h.x_m.max() - h.x_m.min()))
    return pd.DataFrame(out)


def plan_good_bad(rec, x0, out, k=2):
    """x-y per class: the k best held-out vehicles (lowest F) left, the k worst right."""
    vs = vehicle_scores(rec)
    vs = vs[vs.dist >= 60]
    cl = [c for c in mf.ORDER if (vs.cls == c).any()]
    fig, ax = plt.subplots(len(cl), 2, figsize=(7.1, 1.35 * len(cl)), sharex=True, sharey=True)
    for x_, t_ in ((.29, "best held-out vehicles (lowest $F$)"), (.76, "worst held-out vehicles (highest $F$)")):
        fig.text(x_, 1.0, t_, ha="center", va="bottom", fontsize=8)
    shades = [("#1F4E79", "#B8441F"), ("#5DA5DA", "#E8975B")]
    for r, c in enumerate(cl):
        v = vs[vs.cls == c].sort_values("F")
        n = min(k, len(v) // 2) if len(v) > 1 else 1
        for col, pick in ((0, v.head(n)), (1, v.tail(n).iloc[::-1] if len(v) > 1 else v.iloc[0:0])):
            a = ax[r, col]
            a.axhspan(*mf.EDGES, color="#8A8F98", alpha=.10, lw=0)
            for yy, lw, ls in ((mf.EDGES[0], .7, "-"), (mf.EDGES[1], .7, "-"), (0.2, .5, ":")):
                a.axhline(yy, color="#555", lw=lw, ls=ls)
            labs = []
            for j, row in enumerate(pick.itertuples()):
                h = rec[(rec.vid == row.vid) & rec.pres].sort_values("frame")
                gone = ~h.on.to_numpy()          # simulated twin already past the exit line
                xs = np.where(gone, np.nan, h.x_s - 0.5 * h.L - x0)
                ys = np.where(gone, np.nan, h.y_s)
                cm, cs = shades[j % 2]
                a.plot(h.x_m - 0.5 * h.L - x0, h.y_m, color=cm, lw=1.1)
                a.plot(xs, ys, color=cs, lw=1.1, ls="--")
                a.plot((h.x_m - 0.5 * h.L - x0).iloc[0], h.y_m.iloc[0], "o", color=cm, ms=2.5, mew=0)
                labs.append(("dark" if j == 0 else "light")
                            + f" $F$={row.F:.1f} ({row.ex:.0f} m, {row.ey:.1f} m)")
            a.set_title("; ".join(labs), fontsize=5.8, loc="left", pad=2, color="#333")
            a.set_ylim(-.5, mf.EDGES[1] + .5); a.set_xlim(0, mf.SEC)
            a.set_yticks([0, 4, 8]); a.tick_params(length=2, pad=1.5, labelsize=6)
        ax[r, 0].set_ylabel(c, fontsize=7, rotation=0, ha="right", va="center", labelpad=14)
    for a in ax[-1]: a.set_xlabel("position along the section (m)")
    fig.legend(handles=[Line2D([], [], color="#1F4E79", lw=1.1, label="measured"),
                        Line2D([], [], color="#B8441F", lw=1.1, ls="--", label="calibrated"),
                        Line2D([], [], color="#555", lw=.5, ls=":", label="painted edge line")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.03))
    fig.supylabel("lateral position (m)", fontsize=7, x=.01)
    fig.tight_layout(rect=[0, .03, 1, 1])
    fig.savefig(f"{out}/fig_xy.pdf", bbox_inches="tight", dpi=200); plt.close(fig)
    return vs


def per_vehicle_rmse(rec):
    """Each scored vehicle's own RMSE of x, y and v over the frames its twin is
    present (as in scene_errors: at least 50 frames)."""
    t = rec[rec.pres].copy()
    t["ex"] = (t.x_s - t.x_m) ** 2; t["ey"] = (t.y_s - t.y_m) ** 2; t["ev"] = (t.v_s - t.v_m) ** 2
    g = t.groupby(["split", "cls", "vid"])
    d = g[["ex", "ey", "ev"]].mean().pow(.5).join(g.size().rename("n")).reset_index()
    return d[d.n >= 50]


def fig_rmse_spread(rec, out):
    """Per-vehicle RMSE by class: box of the distribution, with mean and one
    standard deviation; training and held-out side by side."""
    d = per_vehicle_rmse(rec)
    cl = [c for c in mf.ORDER if (d.cls == c).any()]
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.5))
    rows = []
    for A, (k, t) in zip(ax, (("ex", "longitudinal position (m)"), ("ey", "lateral position (m)"),
                              ("ev", "speed (m/s)"))):
        for j, (sp, col) in enumerate((("train", "#7FA7C9"), ("test", "#E3A36F"))):
            data = [d[(d.cls == c) & (d.split == sp)][k].to_numpy() for c in cl]
            pos = np.arange(len(cl)) + (j - .5) * .36
            bp = A.boxplot(data, positions=pos, widths=.3, patch_artist=True, showfliers=False,
                           medianprops=dict(color="k", lw=.6), whiskerprops=dict(lw=.5),
                           capprops=dict(lw=.5), boxprops=dict(lw=.4))
            for b in bp["boxes"]: b.set_facecolor(col)
            m = [x.mean() if len(x) else np.nan for x in data]
            s = [x.std(ddof=1) if len(x) > 1 else np.nan for x in data]
            A.errorbar(pos, m, yerr=s, fmt="o", ms=2.2, color="#333", lw=.6, capsize=1.5)
            for c, x in zip(cl, data):
                rows.append(dict(metric=k, split=sp, cls=c, n=len(x), mean=x.mean() if len(x) else np.nan,
                                 sd=x.std(ddof=1) if len(x) > 1 else np.nan))
        A.set_xticks(range(len(cl))); A.set_xticklabels(cl, rotation=45, ha="right", fontsize=6)
        A.set_title("per-vehicle RMSE, " + t, fontsize=7); A.tick_params(length=2, pad=1.5, labelsize=6)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#7FA7C9", label="training vehicles"),
                        plt.Rectangle((0, 0), 1, 1, color="#E3A36F", label="held-out vehicles"),
                        Line2D([], [], color="#333", marker="o", ms=2.5, lw=.6, label="mean $\\pm$ 1 SD")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.06), fontsize=6.5)
    fig.tight_layout(rect=[0, .04, 1, 1])
    fig.savefig(f"{out}/fig_rmse_spread.pdf", bbox_inches="tight"); plt.close(fig)
    # table: mean (SD) of per-vehicle RMSE, held-out vehicles
    s = pd.DataFrame(rows)
    lines = []
    for c in cl:
        cells = []
        for sp in ("train", "test"):
            for k in ("ex", "ey", "ev"):
                r = s[(s.cls == c) & (s.split == sp) & (s.metric == k)].iloc[0]
                nd = 1 if k == "ex" else 2
                cells.append(f"{r['mean']:.{nd}f} ({r['sd']:.{nd}f})" if r.n > 1 else
                             (f"{r['mean']:.{nd}f}" if r.n else "--"))
        lines.append(f"{c} & " + " & ".join(cells) + r"\\")
    open(f"{out}/tab_rmse_sd.tex", "w").write(
        r"\begin{tabular}{@{}l lll lll@{}}\toprule" "\n"
        r" & \multicolumn{3}{c}{training} & \multicolumn{3}{c}{held out}\\" "\n"
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}" "\n"
        r"class & long.\ (m) & lat.\ (m) & speed (m/s) & long.\ (m) & lat.\ (m) & speed (m/s)\\\midrule" "\n"
        + "\n".join(lines) + "\n" r"\bottomrule\end{tabular}" "\n")


def fig_rmse_time(rec, dt, stops, out, bin_s=5.0, min_veh=10):
    """Pooled RMSE against time since each vehicle entered (does the error grow
    the longer a vehicle is simulated?) and against clock time (when in the
    record are the errors made?), all scored vehicles, per class and overall."""
    t = rec[rec.pres].copy()
    t["age"] = (t.frame - t.groupby("vid").frame.transform("min")) * dt
    t["clock"] = t.frame * dt
    for k, (a, b) in dict(ex=("x_s", "x_m"), ey=("y_s", "y_m"), ev=("v_s", "v_m")).items():
        t[k] = (t[a] - t[b]) ** 2
    cl = [c for c in mf.ORDER if c != "Trailer"]
    cc = {"Auto": "#B9A3D0", "Bike": "#7FA7C9", "Car": "#9BC59D", "LCV": "#E3C16F",
          "MultiAxle": "#808080", "Truck": "#C9A27F"}
    fig, ax = plt.subplots(2, 3, figsize=(7.1, 4.3))
    for r, key in enumerate(("age", "clock")):
        t["bin"] = (t[key] // bin_s) * bin_s + bin_s / 2
        for A, (k, lab) in zip(ax[r], (("ex", "longitudinal (m)"), ("ey", "lateral (m)"), ("ev", "speed (m/s)"))):
            if key == "clock":
                for p, q in stops: A.axvspan(p, q, color="#C0392B", alpha=.10, lw=0, zorder=0)
            for c in cl:
                h = t[t.cls == c].groupby("bin").agg(e=(k, "mean"), nv=("vid", "nunique"))
                h = h[h.nv >= (min_veh if key == "age" else 1)]
                if key == "clock":                 # a class absent from the section is a gap
                    h = h.reindex(np.arange(bin_s / 2, 781, bin_s))
                A.plot(h.index, np.sqrt(h.e), color=cc[c], lw=.7, label=c)
            h = t.groupby("bin").agg(e=(k, "mean"), nv=("vid", "nunique"))
            h = h[h.nv >= min_veh]
            A.plot(h.index, np.sqrt(h.e), color="k", lw=1.3, label="all")
            A.set_title("pooled RMSE, " + lab, fontsize=7); A.tick_params(length=2, pad=1.5, labelsize=6)
            A.set_xlabel("time since the vehicle entered (s)" if key == "age" else "time in the record (s)",
                         fontsize=6.5)
            if key == "clock": A.set_xlim(0, 781)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, fontsize=6.5, bbox_to_anchor=(.5, -.02))
    fig.tight_layout(rect=[0, .035, 1, 1])
    fig.savefig(f"{out}/fig_rmse_time.pdf", bbox_inches="tight"); plt.close(fig)


def results_figs():
    df, cfg, P, cidx, train, test, cov = mf.load(f"results/{FREE}", 30.0, True)
    dt = train[0]["dt"]; x0 = df.X_Smt.min()
    rec = mf.simulate_record(train, test, P, cidx, cfg.road)
    from iam_calib.jointsim import prep_frames, exit_stops
    _, _, runs = exit_stops(prep_frames(df, cfg)[0])
    stops = [(p * dt, q * dt) for p, q in runs]
    mf.SUFFIX = ""
    mf.spacetime_pair(rec, test, dt, x0, FIG, stops=stops)
    os.replace(f"{FIG}/fig_stpair.pdf", f"{FIG}/fig_st_x.pdf")
    lateral_time(rec, dt, stops, FIG)
    all_spacetime(rec, dt, x0, stops, FIG)
    vs = plan_good_bad(rec, x0, FIG)
    mf.class_speed(rec, x0, FIG)
    fig_rmse_spread(rec, FIG)
    fig_rmse_time(rec, dt, stops, FIG)
    # queued share, for the text
    q = dict(measured=float((rec[rec.pres].v_m < .5).mean()), calibrated=float((rec[rec.on].v_s < .5).mean()))
    return vs, q


# --------------------------------------------------------------------------
def main():
    os.makedirs(FIG, exist_ok=True)
    tab_bounds(); tab_duggal_cmp()
    # measured-only figures: the exit-signal evidence and the literature flowchart
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "iam_calib.signal_fig", "--out", FIG], check=True,
                   stdout=subprocess.DEVNULL)
    shutil.copy("literature_review_approaches.png", f"{FIG}/fig_litreview.png")
    sc ={key: scores(run) for key, run, _ in RUNS}
    tab_runs(sc)
    for key, run, _ in RUNS:
        tab_cls(sc, key); tab_par(key, run)
        fig_conv(run, key, CONV_COL[["chandra", "duggal", "free"].index(key)])
    spread = f"results/{FREE}/fig_oat_spread.pdf"
    if os.path.exists(spread):
        shutil.copy(spread, f"{FIG}/fig_oat_spread.pdf")
        for m in ("F", "long", "lat", "speed"):
            shutil.copy(f"results/{FREE}/fig_oat_rmse_{m}.pdf", f"{FIG}/fig_oat_prof_{m}.pdf")
        for m in ("key", "weak", "T_rmse"):
            shutil.copy(f"results/{FREE}/fig_oat_{m}.pdf", f"{FIG}/fig_oat_{m}.pdf")
        shutil.copy(f"results/{FREE}/oat_summary.csv", f"{FIG}/oat_summary.csv")
    else:
        print("no sensitivity profiles yet for", FREE)
    vs, q = results_figs()
    json.dump(dict(chandra_run=CHANDRA, queued=q,
                   heldout_vehicle_F=vs.groupby("cls").F.describe().round(2).to_dict()),
              open(f"{FIG}/numbers.json", "w"), indent=1)
    print("Chandrashekar run used:", CHANDRA, "| queued", q)


if __name__ == "__main__":
    main()
