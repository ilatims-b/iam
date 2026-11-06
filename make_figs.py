#!/usr/bin/env python3
"""Figures for the joint-simulation report.

Every figure is drawn from one fitted run. The whole record is simulated block
by block (each block is one scene of the run's scene length, every vehicle
simulated at once), and measured and simulated traffic are then compared
vehicle by vehicle, class by class and over the whole section. Held-out
blocks are shaded wherever time is on an axis, and the single-vehicle figures
use held-out vehicles only.

    python make_figs.py --run results/j6_30_a --scene-s 30
    python make_figs.py --run results/j12_dims --whole-record

Under --whole-record the record is one scene with no reset anywhere, the
held-out set is a set of vehicles rather than of blocks, and nothing is shaded.
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from iam_calib.config import PipelineConfig
from iam_calib.fastcore import PARAM_ORDER
from iam_calib.calibrate import default_params
from iam_calib.calibrate_joint import run_scene, build_pool, group_thin, whole_record

EDGES = (0.0, 8.9)
ORDER = ["Bike", "Car", "LCV", "Truck", "Auto", "MultiAxle", "Trailer"]
plt.rcParams.update({"font.size": 7, "axes.linewidth": .6, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})
MEAS, SIM, TEST = "#1F4E79", "#B8441F", "#E7D9A8"
SEC = 204.0                                        # section length (m)


# --------------------------------------------------------------------------
def load(run, scene_s, whole=False):
    cfg = PipelineConfig()
    cfg.road.y_right, cfg.road.y_left, cfg.road.enabled = EDGES[0], EDGES[1], True
    df = pd.read_csv("D3F1_lcl_smtF.csv")
    rc = f"{run}/run_config.json"
    if os.path.exists(rc):
        cfg.vehicle_dims.update({k: tuple(v) for k, v in json.load(open(rc)).get("dims", {}).items()})
    if whole:
        sig = os.path.exists(rc) and json.load(open(rc)).get("exit_signal", False)
        tr, te, cov = whole_record(df, cfg, 0.3, 1, sig); train, test = [tr], [te]
    else:
        train, test, cov = build_pool(df, cfg, scene_s, 1, None, 1)
    cp = json.load(open(f"{run}/class_params.json"))
    grp, _ = group_thin(train + test, cfg.vehicle_dims, 10)
    classes = sorted(cp); cidx = {c: classes.index(grp[c]) for c in grp}
    P = np.array([[{**default_params(), **cp[c]}[k] for k in PARAM_ORDER] for c in classes])
    return df, cfg, P, cidx, train, test, cov


def scored(s, i):
    """Whether vehicle i belongs to this scene's split. A block scene holds
    only its own split; a whole-record scene holds every vehicle and marks its
    split's vehicles in `score`."""
    return s.get("score") is None or bool(s["score"][i])


def simulate_record(train, test, P, cidx, road):
    """Long-format measured and simulated state for every block, in time order.
    `on` marks a simulated vehicle that is on the section (not yet retired at the
    exit line); `pres` marks frames where its measured twin is present."""
    rows = []
    for split, scenes in (("train", train), ("test", test)):
        for s in scenes:
            sx, sy, svx = run_scene(s, P, cidx, road)
            xe = s["x_exit_c"]
            for i in range(s["X"].shape[0]):
                f = np.where(np.isfinite(sx[i]))[0]
                if f.size == 0 or not scored(s, i): continue
                rows.append(pd.DataFrame(dict(
                    split=split, vid=s["vids"][i], cls=s["cls"][i],
                    frame=s["f0"] + f, L=s["L"][i],
                    x_s=sx[i, f], y_s=sy[i, f], v_s=svx[i, f],
                    on=sx[i, f] - 0.5 * s["L"][i] < xe,
                    pres=s["present"][i, f],
                    x_m=s["X"][i, f], y_m=s["Y"][i, f], v_m=s["VX"][i, f])))
    return pd.concat(rows, ignore_index=True)


def shade_test(ax, test, dt):
    for s in test:
        if "score" in s: return                     # held out by vehicle, not time
        a = s["f0"] * dt; ax.axvspan(a, a + s["len_s"], color=TEST, alpha=.55, lw=0, zorder=0)


SUFFIX = ""                                         # set by --suffix


def save(fig, out, name):
    name = name.replace(".pdf", f"{SUFFIX}.pdf")
    fig.savefig(f"{out}/{name}", bbox_inches="tight", dpi=200); plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------
# single vehicles, held out
def pick(test, P, cidx, road):
    """Longest-present held-out vehicle of each class, with its simulated twin."""
    best = {}
    for s in test:
        sx, sy, svx = run_scene(s, P, cidx, road)
        for i in range(s["X"].shape[0]):
            if not scored(s, i): continue
            m = s["present"][i] & np.isfinite(sx[i]); c = s["cls"][i]
            if m.sum() > best.get(c, (0,))[0]:
                f = np.where(m)[0]; t = (f - f[0]) * s["dt"]
                # once the simulated vehicle has left the section it is scored at
                # the exit line, but drawing that hold would show a stopped
                # vehicle still "moving"; end the simulated curve at the exit
                gone = sx[i][m] - 0.5 * s["L"][i] >= s["x_exit_c"]
                sim = [np.where(gone, np.nan, v) for v in (sx[i][m], sy[i][m], svx[i][m])]
                best[c] = (m.sum(), t, s["X"][i][m], s["Y"][i][m], s["VX"][i][m], *sim)
    return best


def trio(best, out):
    cl = [c for c in ORDER if c in best]
    rows = [("x", "distance\ntravelled (m)"), ("y", "lateral\nposition (m)"), ("v", "speed (m/s)")]
    fig, ax = plt.subplots(3, len(cl), figsize=(7.1, 4.0), sharex=True)
    for r, (key, ylab) in enumerate(rows):
        for k, c in enumerate(cl):
            a = ax[r, k]; _, t, X, Y, V, SX, SY, SV = best[c]
            mm = {"x": (X - X[0], SX - SX[0]), "y": (Y, SY), "v": (V, SV)}[key]
            if key == "y":
                a.axhspan(*EDGES, color="#8A8F98", alpha=.13, lw=0)
                a.set_ylim(EDGES[0] - .5, EDGES[1] + .5)
            a.plot(t, mm[0], color=MEAS, lw=1.0); a.plot(t, mm[1], color=SIM, lw=1.0, ls="--")
            a.tick_params(length=2, pad=1.5, labelsize=6)
            if r == 0: a.set_title(c, fontsize=7)
            if r == 2: a.set_xlabel("time (s)", fontsize=6.5)
        ax[r, 0].set_ylabel(ylab, fontsize=6.5)
    fig.legend(handles=[Line2D([], [], color=MEAS, lw=1.1, label="measured"),
                        Line2D([], [], color=SIM, lw=1.1, ls="--", label="simulated")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.03))
    fig.tight_layout(rect=[0, .03, 1, 1]); save(fig, out, "fig_trio.pdf")


def planview(test, P, cidx, road, xmin, out, per_class=2):
    """The carriageway seen from above, 204 m by 8.9 m: the two held-out vehicles
    of each class that travel furthest, measured against simulated."""
    cand = {c: [] for c in ORDER}
    for s in test:
        sx, sy, _ = run_scene(s, P, cidx, road)
        for i in range(s["X"].shape[0]):
            m = s["present"][i] & np.isfinite(sx[i])
            if m.sum() < 200 or not scored(s, i): continue
            h = 0.5 * s["L"][i]                       # front bumper -> centroid
            X = s["X"][i][m] - h
            cand[s["cls"][i]].append((X.max() - X.min(), X, s["Y"][i][m], sx[i][m] - h, sy[i][m]))
    cl = [c for c in ORDER if cand.get(c)]
    fig, ax = plt.subplots(len(cl), 1, figsize=(7.1, 1.30 * len(cl)), sharex=True)
    for a, c in zip(np.atleast_1d(ax), cl):
        a.axhspan(*EDGES, color="#8A8F98", alpha=.12, lw=0)
        for yy, lw, ls in ((EDGES[0], .7, "-"), (EDGES[1], .7, "-"), (0.2, .5, ":")):
            a.axhline(yy, color="#555", lw=lw, ls=ls)
        for k, (_, X, Y, SX, SY) in enumerate(sorted(cand[c], key=lambda r: -r[0])[:per_class]):
            al = 1 - .35 * k
            a.plot(X - xmin, Y, color=MEAS, lw=1.1, alpha=al)
            a.plot(SX - xmin, SY, color=SIM, lw=1.1, ls="--", alpha=al)
            a.plot(X[0] - xmin, Y[0], "o", color=MEAS, ms=2.6, mew=0, alpha=al)
        a.set_ylim(-0.5, EDGES[1] + 0.5); a.set_xlim(0, SEC)
        a.set_yticks([0, 4, 8]); a.tick_params(length=2, pad=1.5, labelsize=6)
        a.set_ylabel(c, fontsize=7, rotation=0, ha="right", va="center", labelpad=14)
    np.atleast_1d(ax)[-1].set_xlabel("position along the 204 m section (m)")
    fig.legend(handles=[Line2D([], [], color=MEAS, lw=1.1, label="measured"),
                        Line2D([], [], color=SIM, lw=1.1, ls="--", label="simulated"),
                        Line2D([], [], color="#555", lw=.5, ls=":", label="painted edge line")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.035))
    fig.supylabel("lateral position across carriageway (m)", fontsize=7, x=.015)
    fig.tight_layout(rect=[0, .035, 1, 1]); save(fig, out, "fig_planview.pdf")


# --------------------------------------------------------------------------
# whole record, measured against simulated
def _segments(t, x, v, vid):
    segs, cols = [], []
    for k in np.unique(vid):
        idx = np.where(vid == k)[0]
        if idx.size < 3: continue
        idx = idx[np.argsort(t[idx], kind="stable")]  # the record is split-ordered, not time-ordered
        pts = np.column_stack([t[idx], x[idx]]).reshape(-1, 1, 2)
        segs.append(np.concatenate([pts[:-1], pts[1:]], axis=1)); cols.append(v[idx][:-1])
    return segs, cols


def spacetime_pair(rec, test, dt, x0, out, step=10, stops=()):
    """Space-time per class over the whole record: measured left, simulated
    right, coloured by speed. Simulated trajectories break at block boundaries,
    where each scene starts again from measurement; under --whole-record there
    are none."""
    r = rec[rec.frame % step == 0]
    fig, ax = plt.subplots(len(ORDER), 2, figsize=(7.1, 1.18 * len(ORDER)), sharex=True, sharey=True)
    norm = plt.Normalize(0, 8); lc = None
    for k, c in enumerate(ORDER):
        g = r[r.cls == c]
        gm = g[g.pres]; gs = g[g.on]
        key_s = gs.vid.astype(str) + "_" + (gs.frame // int(round(test[0]["len_s"] / dt))).astype(str)
        for col, (t, x, v, vid) in enumerate((
                (gm.frame * dt, gm.x_m - 0.5 * gm.L - x0, gm.v_m, gm.vid.astype(str).values),
                (gs.frame * dt, gs.x_s - 0.5 * gs.L - x0, gs.v_s, key_s.values))):
            a = ax[k, col]; shade_test(a, test, dt)
            for p, q in stops:                  # measured stop periods at the exit
                a.axvspan(p, q, color="#C0392B", alpha=.10, lw=0, zorder=0)
            sg, cv = _segments(np.asarray(t), np.asarray(x), np.asarray(v), vid)
            if sg:
                lc = LineCollection(np.concatenate(sg), cmap="viridis", norm=norm,
                                    linewidths=.45, rasterized=True)
                lc.set_array(np.concatenate(cv)); a.add_collection(lc)
            a.set_xlim(0, 781); a.set_ylim(0, SEC)
            a.set_yticks([0, 100, 200]); a.tick_params(length=2, pad=1.5, labelsize=6)
            if k == 0: a.set_title("measured" if col == 0 else "simulated (joint)", fontsize=7.5)
        ax[k, 0].set_ylabel(f"{c}\n(n={rec[rec.cls == c].vid.nunique()})", fontsize=6.5,
                            rotation=0, ha="right", va="center", labelpad=16)
    for a in ax[-1]: a.set_xlabel("time (s)")
    if stops:
        fig.text(.5, -.005, "red bands: periods when the measured traffic at the exit is stopped",
                 ha="center", fontsize=6.5, color="#C0392B")
    cb = fig.colorbar(lc, ax=ax, pad=.012, fraction=.016)
    cb.set_label("longitudinal speed (m/s)", fontsize=6.5); cb.ax.tick_params(labelsize=5.5, length=2)
    fig.supylabel("position along section (m)", fontsize=7, x=.005)
    save(fig, out, "fig_stpair.pdf")


def _fd(occ, spd, ax, color, label, edges):
    dens = occ / SEC; k = np.digitize(dens, edges) - 1
    q = pd.DataFrame({"k": k, "v": spd}).groupby("k")["v"]
    idx = range(len(edges) - 1); n = q.size().reindex(idx).fillna(0)
    ok = (n >= 30).values; mid = 0.5 * (edges[:-1] + edges[1:])
    ax.fill_between(mid[ok], q.quantile(.25).reindex(idx)[ok], q.quantile(.75).reindex(idx)[ok],
                    color=color, alpha=.18, lw=0)
    ax.plot(mid[ok], q.median().reindex(idx)[ok], color=color, lw=1.3, label=label)


def coverage_pair(rec, test, dt, out):
    """Occupancy, mean speed and the speed-density relation, measured against
    simulated. A frame's simulated occupancy counts every simulated vehicle
    still on the section, including ones whose measured twin has already left."""
    m = rec[rec.pres].groupby("frame").agg(n=("v_m", "size"), v=("v_m", "mean"))
    s = rec[rec.on].groupby("frame").agg(n=("v_s", "size"), v=("v_s", "mean"))
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 1.95))
    for a, col, ylab, title in ((ax[0], "n", "vehicles on section", "occupancy"),
                                (ax[1], "v", "mean speed (m/s)", "mean speed")):
        shade_test(a, test, dt)
        a.plot(m.index[::4] * dt, m[col].values[::4], color=MEAS, lw=.6, label="measured")
        a.plot(s.index[::4] * dt, s[col].values[::4], color=SIM, lw=.6, label="simulated")
        a.set_xlabel("time (s)"); a.set_ylabel(ylab); a.set_title(title, fontsize=7)
    edges = np.linspace(0.05, 0.40, 19)
    _fd(m.n.values, m.v.values, ax[2], MEAS, "measured", edges)
    _fd(s.n.values, s.v.values, ax[2], SIM, "simulated", edges)
    ax[2].set_xlabel("density (veh/m)"); ax[2].set_ylabel("mean speed (m/s)")
    ax[2].set_title("fundamental diagram (median, IQR)", fontsize=7)
    ax[0].legend(fontsize=5.5, frameon=False, loc="upper left")
    for a in ax: a.tick_params(length=2, pad=1.5)
    fig.tight_layout(); save(fig, out, "fig_coverage.pdf")


def coverage_measured(df, out):
    """Measured only: what the record contains, before any fitting."""
    g = df.groupby("GTime_sec")          # the Time column is the same clock with
    occ = g.size(); spd = g["Long_Speed"].mean()   # float noise; never group on it
    t = occ.index.values - occ.index.values.min()
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 1.9))
    ax[0].plot(t[::4], occ.values[::4], color=MEAS, lw=.6)
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("vehicles on section"); ax[0].set_title("occupancy", fontsize=7)
    ax[1].plot(t[::4], spd.values[::4], color=MEAS, lw=.6)
    ax[1].axhline(df.Long_Speed.median(), color=SIM, lw=.8, ls="--")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("mean speed (m/s)"); ax[1].set_title("mean speed", fontsize=7)
    dens = occ.values / SEC
    hb = ax[2].hexbin(dens, spd.values, gridsize=38, cmap="Blues", mincnt=1, linewidths=0, rasterized=True)
    cb = fig.colorbar(hb, ax=ax[2], pad=.02, fraction=.045)
    cb.set_label("frames", fontsize=6); cb.ax.tick_params(labelsize=5, length=2)
    _fd(occ.values, spd.values, ax[2], SIM, None, np.linspace(dens.min(), dens.max(), 19))
    ax[2].set_xlabel("density (veh/m)"); ax[2].set_ylabel("mean speed (m/s)")
    ax[2].set_title("fundamental diagram", fontsize=7)
    for a in ax: a.tick_params(length=2, pad=1.5)
    fig.tight_layout(); save(fig, out, "fig_coverage_meas.pdf")


def class_speed(rec, x0, out):
    """Speed distribution per class and mean speed along the section, measured
    against simulated."""
    fig, ax = plt.subplots(2, 4, figsize=(7.1, 3.1))
    for a, c in zip(ax.flat, ORDER):
        g = rec[rec.cls == c]
        a.hist(g[g.pres].v_m, bins=40, range=(0, 14), density=True, color=MEAS, alpha=.35, lw=0)
        a.hist(g[g.on].v_s, bins=40, range=(0, 14), density=True, histtype="step", color=SIM, lw=1.0)
        a.set_title(c, fontsize=7); a.tick_params(length=2, pad=1.5, labelsize=6)
        a.set_xlabel("speed (m/s)", fontsize=6)
    a = ax.flat[-1]; b = np.linspace(0, SEC, 35)
    for sel, xcol, vcol, color, lab in ((rec.pres, "x_m", "v_m", MEAS, "measured"),
                                        (rec.on, "x_s", "v_s", SIM, "simulated")):
        g = rec[sel]; k = np.digitize(g[xcol] - 0.5 * g.L - x0, b) - 1
        mv = g[vcol].groupby(k).mean(); mv = mv[(mv.index >= 0) & (mv.index < len(b) - 1)]
        a.plot(0.5 * (b[mv.index] + b[mv.index + 1]), mv.values, color=color, lw=1.1, label=lab)
    a.set_title("mean speed along section", fontsize=7); a.set_xlabel("position (m)", fontsize=6)
    a.tick_params(length=2, pad=1.5, labelsize=6); a.legend(fontsize=5.5, frameon=False)
    fig.tight_layout(); save(fig, out, "fig_classspeed.pdf")


def class_speed_time(rec, test, dt, out, bin_s=5.0):
    """Mean speed of each class over the record, measured against simulated:
    whether the model reproduces when each class slows and recovers."""
    fb = int(round(bin_s / dt))
    fig, ax = plt.subplots(len(ORDER), 1, figsize=(7.1, 1.0 * len(ORDER)), sharex=True)
    for a, c in zip(ax, ORDER):
        g = rec[rec.cls == c]; shade_test(a, test, dt)
        for sel, vcol, color in ((g.pres, "v_m", MEAS), (g.on, "v_s", SIM)):
            h = g[sel]; mv = h[vcol].groupby(h.frame // fb).mean()
            mv = mv.reindex(range(int(rec.frame.max()) // fb + 1))   # gaps stay gaps
            a.plot(mv.index * bin_s, mv.values, color=color, lw=.8)
        a.set_ylabel(c, fontsize=6.5, rotation=0, ha="right", va="center", labelpad=12)
        a.tick_params(length=2, pad=1.5, labelsize=6); a.set_ylim(bottom=0)
    ax[-1].set_xlabel("time (s)"); ax[-1].set_xlim(0, 781)
    fig.supylabel("mean speed (m/s)", fontsize=7, x=.01)
    h = [Line2D([], [], color=MEAS, lw=1, label="measured"),
         Line2D([], [], color=SIM, lw=1, label="simulated")]
    if "score" not in test[0]:
        h.append(plt.Rectangle((0, 0), 1, 1, color=TEST, alpha=.55, label="held-out blocks"))
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, bbox_to_anchor=(.5, -.03))
    fig.tight_layout(rect=[0, .03, 1, 1]); save(fig, out, "fig_classspeed_time.pdf")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scene-s", type=float, default=30.0)
    ap.add_argument("--whole-record", action="store_true")
    ap.add_argument("--out", default="paper_joint")
    ap.add_argument("--suffix", default="", help="appended to every figure file name")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); SUFFIX = a.suffix
    df, cfg, P, cidx, train, test, cov = load(a.run, a.scene_s, a.whole_record)
    dt = train[0]["dt"]; x0 = df.X_Smt.min()
    best = pick(test, P, cidx, cfg.road)
    trio(best, a.out)
    planview(test, P, cidx, cfg.road, x0, a.out)
    rec = simulate_record(train, test, P, cidx, cfg.road)
    stops = ()
    if a.whole_record:
        from iam_calib.jointsim import prep_frames, exit_stops
        _, _, runs = exit_stops(prep_frames(df, cfg)[0])
        stops = [(p * dt, q * dt) for p, q in runs]
    spacetime_pair(rec, test, dt, x0, a.out, stops=stops)
    coverage_pair(rec, test, dt, a.out)
    class_speed(rec, x0, a.out)
    class_speed_time(rec, test, dt, a.out)
