#!/usr/bin/env python3
"""Class-level calibration against a fully simultaneous simulation.

Every vehicle is integrated at once and reacts only to other SIMULATED
vehicles. Nothing is replayed from the dataset inside a scene except each
vehicle's state at the frame it enters. There is no mid-scene reset, so a
scene runs free for its whole length.

Because trajectories are coupled, parameters cannot be fitted per vehicle;
one vector per vehicle class is fitted jointly (7 x 16 = 112 free values).
Scenes, not vehicles, are split into train and test, except under
--whole-record, where the record is one scene and vehicles are split.

    python calibrate_joint.py D3F1_lcl_smtF.csv --scene-s 30 --out results/joint30
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd

from iam_calib.config import (PipelineConfig, MODEL_CONSTANTS as C,
                              SPEC_BY_NAME, FREE_PARAMS, dims_for_class,
                              REALISTIC_BOUNDS)
from iam_calib.calibrate import to_norm, from_norm, default_params
from iam_calib.fastcore import PARAM_ORDER
from iam_calib.jointsim import build_scenes, prep_frames, scene_slice, exit_stops, _run_scene

NORM_HI = 10.0


def freeflow_v0(df, pcfg, pct=85, decile=0.10):
    """Per-class free-flow speed: the `pct`-th percentile of speed in the least
    dense decile of frames. This is the traffic-engineering definition of a
    free-flow speed, and it is a far better starting point for v0 than one
    global default, since a desired speed is exactly what the parameter means."""
    c = pcfg.data
    occ = df.groupby(c.col_time).size()
    free = df[df[c.col_time].isin(occ[occ <= occ.quantile(decile)].index)]
    return {k: float(np.percentile(g[c.col_vx], pct))
            for k, g in free.groupby(c.col_class)}


def group_thin(scenes, dims, min_veh=10):
    """No fleet-median shrinkage. A class with too few vehicles to identify its
    own parameters instead SHARES the vector of the dimensionally nearest class
    that does have enough, which keeps every fitted value a real fit."""
    cnt = {}
    for s in scenes:
        for v, c in zip(s["vids"], s["cls"]):
            cnt.setdefault(c, set()).add(v)
    n = {c: len(v) for c, v in cnt.items()}
    big = [c for c in n if n[c] >= min_veh]
    grp = {}
    for c in n:
        if c in big:
            grp[c] = c
        else:
            Lc, Wc = dims_for_class(c, dims)
            grp[c] = min(big, key=lambda b: abs(dims_for_class(b, dims)[0]-Lc)
                                          + abs(dims_for_class(b, dims)[1]-Wc))
    return grp, n


LATERAL_BOUNDS = {"sensLat": (0.002, 6.0, True), "tauLatOVM": (0.15, 30.0, True)}


def apply_bounds(overrides):
    """Set search bounds (lo, hi, log) on the module-level parameter specs.

    Pool workers are spawned, so they re-import config and see only the
    defaults. Bounds changed in the main process do NOT reach them, which once
    let the optimiser search one box while the saved parameters were decoded
    from another. Every worker therefore calls this in its initialiser, and
    main() checks that the saved parameters reproduce the optimiser's
    objective."""
    for n, (lo, hi, log) in overrides.items():
        sp = SPEC_BY_NAME[n]
        object.__setattr__(sp, "lo", lo); object.__setattr__(sp, "hi", hi)
        object.__setattr__(sp, "log", log)


CLASSES = ["Bike", "Car", "LCV", "Truck", "Auto", "MultiAxle", "Trailer"]


def block_table(d, spb, nblk):
    """Per-block traffic state: mean occupancy, mean speed, fraction of vehicle
    frames below 0.5 m/s (the queued share), and vehicles present per class."""
    g = d[d.frame < nblk * spb].assign(blk=lambda x: x.frame // spb)
    rows = []
    for b, gb in g.groupby("blk"):
        ct = gb.groupby("cls").vid.nunique()
        rows.append(dict(blk=b, occ=gb.groupby("frame").size().mean(),
                         speed=gb.vx.mean(), queued=(gb.vx < 0.5).mean(),
                         **{c: int(ct.get(c, 0)) for c in CLASSES}))
    return pd.DataFrame(rows).set_index("blk")


def stratified_test_blocks(B, n_test, seed=1, n_try=200_000, max_run=2):
    """Choose test blocks that mirror the training blocks.

    A split by contiguous time chunk, or by random scene, can leave training
    without a traffic phase or a vehicle type the test set then asks about. The
    record is far from stationary: the queued share of a block runs from 0.00 to
    0.71 and some classes are absent from whole stretches. So draw many random
    assignments and keep the one whose test blocks best match the rest on mean
    occupancy, mean speed, queued share, the split of blocks across queued-share
    terciles, and each class's share of appearances. Runs of adjacent test
    blocks are capped at `max_run`, which also caps test scenes at that length."""
    n = len(B); rng = np.random.default_rng(seed)
    M = np.zeros((n_try, n), bool)
    M[np.arange(n_try)[:, None], np.argsort(rng.random((n_try, n)), axis=1)[:, :n_test]] = True
    run = np.zeros(n_try, bool)
    for i in range(n - max_run):
        run |= M[:, i:i + max_run + 1].all(axis=1)
    M = M[~run].astype(float); tr = 1.0 - M; nt = n_test; nr = n - n_test
    def diff(col):
        v = B[col].to_numpy(float)
        return np.abs(M @ v / nt - tr @ v / nr) / v.std()
    target = nt / n
    cls_dev = np.mean([np.abs(M @ B[c].to_numpy(float) / B[c].sum() - target) / target
                       for c in CLASSES], axis=0)
    terc = np.digitize(B.queued, np.quantile(B.queued, [1 / 3, 2 / 3]))
    ph_dev = sum(np.abs(M @ (terc == t).astype(float) - (terc == t).sum() * target)
                 for t in range(3))
    score = diff("occ") + diff("speed") + diff("queued") + cls_dev + 0.5 * ph_dev
    return sorted(B.index[M[int(np.argmin(score))] > 0].tolist())


def coverage_report(B, test):
    te = B.loc[test]; tr = B.drop(index=test)
    return {"test_blocks": [int(x) for x in test],
            "occupancy": [float(tr.occ.mean()), float(te.occ.mean())],
            "speed": [float(tr.speed.mean()), float(te.speed.mean())],
            "queued": [float(tr.queued.mean()), float(te.queued.mean())],
            "class_share_in_test": {c: float(te[c].sum() / B[c].sum()) for c in CLASSES}}


def build_pool(df, pcfg, min_s=30.0, max_blocks=4, n_test=None, seed=1):
    """Scenes of every length from `min_s` up to `max_blocks * min_s`.

    The record is cut into blocks of `min_s` seconds and each block is assigned
    to train or test (see stratified_test_blocks). Every contiguous run of
    same-split blocks then yields a scene of each length from one block up to
    the run length, capped at `max_blocks`. Splitting by block, not by scene,
    matters: scenes of different lengths overlap in time, and a random split
    over them would put the same seconds of traffic in both sets. With
    `max_blocks=1` this reduces to fixed-length scenes."""
    d, frames, dt = prep_frames(df, pcfg)
    spb = int(round(min_s / dt)); nblk = (int(frames.max()) + 1) // spb
    B = block_table(d, spb, nblk)
    n_test = n_test or int(round(0.3 * nblk))
    test_blocks = stratified_test_blocks(B, n_test, seed)
    is_test = np.isin(np.arange(nblk), test_blocks)
    train, test = [], []
    b = 0
    while b < nblk:
        e = b
        while e + 1 < nblk and is_test[e + 1] == is_test[b]:
            e += 1
        n = e - b + 1
        for k in range(1, min(n, max_blocks) + 1):
            for st in range(n - k + 1):
                sc = scene_slice(d, frames, (b + st) * spb, k * spb, dt)
                if sc is not None:
                    (test if is_test[b] else train).append(sc)
        b = e + 1
    return train, test, coverage_report(B, test_blocks)


# classes the exit stop line holds: in the data, Bikes that reach the exit
# during a stop mostly filter through and leave (54 of 69); every other class
# stops (Car 18 of 18, LCV 3 of 3, Truck 2 of 2, MultiAxle 2 of 2, Auto 2 of 2)
SIGNAL_EXEMPT = ("Bike",)


def whole_record(df, pcfg, frac=0.3, seed=1, signal=False):
    """The whole record as ONE scene, so nothing is ever reset, with the
    vehicles split into train and test instead of time.

    Within each class, vehicles are ordered by entry time and cut into groups
    of ten consecutive entrants; `frac` of each group, drawn at random, is held
    out. Every class and every stretch of the record therefore has the same
    share of test vehicles. All vehicles are simulated together either way:
    a test vehicle's measured trajectory after entry is never used in fitting,
    but its simulated self is still a neighbour of the training vehicles."""
    d, frames, dt = prep_frames(df, pcfg)
    s = scene_slice(d, frames, 0, int(frames.max()) + 1, dt)
    rng = np.random.default_rng(seed)
    first = np.argmax(s["present"], axis=1)
    test = np.zeros(len(first), bool)
    for c in sorted(set(s["cls"])):
        idx = np.array([i for i, k in enumerate(s["cls"]) if k == c])
        idx = idx[np.argsort(first[idx], kind="stable")]
        for g in range(0, len(idx), 10):
            grp = idx[g:g + 10]
            test[rng.choice(grp, int(round(frac * len(grp))), replace=False)] = True
    pres = s["present"]; occ = pres.sum(0)
    def stats(m):
        q = pres[m]; v = s["VX"][m][q]
        return dict(vehicles=int(m.sum()), frames=int(q.sum()), speed=float(v.mean()),
                    queued=float((v < 0.5).mean()),
                    occupancy=float((q * occ).sum() / q.sum()))
    cov = {"mode": "whole-record, split by vehicle", "frac": frac, "seed": seed,
           "train": stats(~test), "test": stats(test),
           "class_share_in_test": {c: float(np.mean([test[i] for i, k in enumerate(s["cls"]) if k == c]))
                                   for c in sorted(set(s["cls"]))},
           "test_vids": [v for v, t in zip(s["vids"], test) if t]}
    if signal:
        stop, x_stop, runs = exit_stops(d)
        s = dict(s, stop=stop[:s["T"]], x_stop=x_stop,
                 obey=np.array([c not in SIGNAL_EXEMPT for c in s["cls"]]))
        cov["exit_stops_s"] = [[a * dt, b * dt] for a, b in runs]
        cov["x_stop_from_start_m"] = x_stop - float((d.x - 0.5 * d.L).min())
    return dict(s, score=~test), dict(s, score=test), cov


NEIGHBOUR_CAP = 96     # above the most vehicles that ever share a +-80 m window


def run_scene(s, P, cidx, road, K=NEIGHBOUR_CAP, return_trunc=False):
    cid = np.array([cidx[c] for c in s["cls"]], np.int64)
    sx, sy, svx, nt = _run_scene(
        s["X"], s["Y"], s["VX"], s["VY"], s["AX"], s["present"],
        s["L"], s["W"], cid, P, s["T"], s["dt"], K, s["x_exit_c"], 80.0, 8.0,
        C.bmax, C.longParReductFactor, C.bIntCrit, C.smaxfollow,
        C.accLatIntMaxFactor, C.useSeveralLeaders, C.accLatBMax, C.v0max,
        C.cool, C.use_narrow, C.use_clamp, C.dvdumax, C.dotdvdumax,
        True, road.y_left, road.y_right,
        s.get("stop", np.zeros(s["T"], np.bool_)),
        s.get("obey", np.zeros(len(s["cls"]), np.bool_)), float(s.get("x_stop", 0.0)))
    return (sx, sy, svx, nt) if return_trunc else (sx, sy, svx)


def scene_errors(s, P, cidx, road, x_sc=2.0, y_sc=0.5, min_steps=50, with_n=False):
    """Per-vehicle (F, rmse_x, rmse_y, rmse_v, class) for one scene, plus the
    number of frames scored when `with_n`, so that errors can be pooled over
    data points rather than averaged over vehicles."""
    sx, sy, svx = run_scene(s, P, cidx, road)
    out = []
    score = s.get("score")
    for i in range(s["X"].shape[0]):
        if score is not None and not score[i]:
            continue
        m = s["present"][i] & np.isfinite(sx[i])
        if m.sum() < min_steps:
            continue
        ex = np.sqrt(np.mean((sx[i][m] - s["X"][i][m]) ** 2))
        ey = np.sqrt(np.mean((sy[i][m] - s["Y"][i][m]) ** 2))
        ev = np.sqrt(np.mean((svx[i][m] - s["VX"][i][m]) ** 2))
        vs = max(np.sqrt(np.mean(s["VX"][i][m] ** 2)), 0.5)
        if not np.isfinite(ex + ey + ev):
            ex = ey = ev = np.nan
        row = (ex / x_sc + ey / y_sc + ev / vs, ex, ey, ev, s["cls"][i])
        out.append(row + (int(m.sum()),) if with_n else row)
    return out


def objective(z, scenes, classes, cidx, names, bases, road, cap):
    """Mean capped F over every trace in `scenes`. `bases[k]` holds class k's
    values for any parameter not in `names`, which is how a parameter is held
    fixed at a class-specific value rather than fitted."""
    P = np.empty((len(classes), len(PARAM_ORDER)))
    for k, c in enumerate(classes):
        p = from_norm(z[k * len(names):(k + 1) * len(names)], names, bases[k])
        P[k] = [p[n] for n in PARAM_ORDER]
    tot, n = 0.0, 0
    for s in scenes:
        for f, *_ in scene_errors(s, P, cidx, road):
            tot += cap if not np.isfinite(f) else min(f, cap); n += 1
    return tot / max(n, 1)


_G = None


def _init_pool(g, overrides):
    global _G
    _G = g
    apply_bounds(overrides)


def _pool_eval(z):
    return objective(z, *_G)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--out", default="results/joint")
    ap.add_argument("--scene-s", type=float, default=30.0)
    ap.add_argument("--road-edges", type=float, nargs=2, default=[0.0, 8.9])
    ap.add_argument("--maxfevals", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1, help="optimiser seed")
    ap.add_argument("--split-seed", type=int, default=1,
                    help="seed of the train/test block split, kept separate so "
                         "that a second optimiser seed sees the same test set")
    ap.add_argument("--bound", nargs=3, action="append", default=[],
                    metavar=("NAME", "LO", "HI"),
                    help="override one parameter's search box (repeatable); the "
                         "parameter keeps its usual linear or log scaling")
    ap.add_argument("--fix-v0-freeflow", action="store_true",
                    help="hold v0 at each class's free-flow speed instead of fitting it")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--cap", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--init", default=None, help="class_params.json to warm-start from")
    ap.add_argument("--sigma0", type=float, default=2.0)
    ap.add_argument("--a-bounds", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"), help="search box for the maximum "
                    "acceleration a (searched in log space)")
    ap.add_argument("--multi", action="store_true",
                    help="scenes of every length from --scene-s up to "
                         "--max-blocks x --scene-s, split by time block")
    ap.add_argument("--max-blocks", type=int, default=4,
                    help="longest scene, in blocks of --scene-s, under --multi")
    ap.add_argument("--test-blocks", type=int, default=None,
                    help="number of test blocks (default: 30%% of the record)")
    ap.add_argument("--whole-record", action="store_true",
                    help="simulate the whole record as one scene and split the "
                         "vehicles, not time, into train and test")
    ap.add_argument("--exit-signal", action="store_true",
                    help="under --whole-record: a stop line at the exit, active "
                         "only while the measured exit traffic is stopped")
    ap.add_argument("--bounds", choices=["default", "realistic"], default="default",
                    help="realistic: the dataset-grounded boxes of config.REALISTIC_BOUNDS "
                         "for every parameter (--a-bounds and --bound still override)")
    ap.add_argument("--fix", nargs="+", default=[], metavar="NAME=VALUE",
                    help="hold parameters at one value for every class, e.g. "
                         "values from an external IDM calibration; delta (the "
                         "IDM exponent, otherwise 4) can be set this way")
    ap.add_argument("--fix-class", default=None, metavar="FILE.json",
                    help="hold parameters at class-specific values, a JSON file "
                         "{class: {name: value}}; every fitted class must be given "
                         "and with the same parameter names")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop the search after this many minutes of fitting")
    ap.add_argument("--dims", nargs="+", default=[], metavar="CLASS=LxW",
                    help="vehicle length x width in m per class, e.g. Auto=2.7x1.4; "
                         "classes not named keep config.DEFAULT_VEHICLE_DIMS")
    ap.add_argument("--min-veh", type=int, default=10,
                    help="classes below this share a similar class's vector")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    overrides = dict(REALISTIC_BOUNDS) if a.bounds == "realistic" else dict(LATERAL_BOUNDS)
    if a.a_bounds:
        overrides["a"] = (a.a_bounds[0], a.a_bounds[1], True)
    for name, lo, hi in a.bound:
        overrides[name] = (float(lo), float(hi), SPEC_BY_NAME[name].log)
    apply_bounds(overrides)
    cfg = PipelineConfig()
    cfg.road.y_right, cfg.road.y_left, cfg.road.enabled = min(a.road_edges), max(a.road_edges), True
    dims = {}
    for item in a.dims:
        c, lw = item.split("="); L, W = lw.lower().split("x")
        dims[c.strip().lower()] = (float(L), float(W))
    cfg.vehicle_dims.update(dims)
    df = pd.read_csv(a.csv)
    if a.whole_record:
        tr, te, cov = whole_record(df, cfg, 1.0 - a.train_frac, a.split_seed, a.exit_signal)
        if a.exit_signal:
            print("  exit stop line at {:.1f} m during {}".format(
                cov["x_stop_from_start_m"], [(round(p), round(q)) for p, q in cov["exit_stops_s"]]))
        train, test, scenes = [tr], [te], [tr]
        print(f"whole record as one {tr['len_s']:.0f}s scene; "
              f"{cov['train']['vehicles']} train / {cov['test']['vehicles']} test vehicles")
        print("  train vs test   occupancy {:.1f} / {:.1f}   speed {:.2f} / {:.2f}   "
              "queued {:.2f} / {:.2f}".format(*[cov[k][q] for q in ("occupancy", "speed", "queued")
                                               for k in ("train", "test")]))
    else:
        train, test, cov = build_pool(df, cfg, a.scene_s,
                                      a.max_blocks if a.multi else 1,
                                      a.test_blocks, a.split_seed)
        scenes = train + test
        lens = sorted({int(round(x["len_s"])) for x in scenes})
        print(f"scene lengths {lens}s; {len(train)} train / {len(test)} test scenes; "
              f"test blocks {cov['test_blocks']}")
        print("  train vs test   occupancy {:.1f} / {:.1f}   speed {:.2f} / {:.2f}   "
              "queued {:.2f} / {:.2f}".format(*cov["occupancy"], *cov["speed"], *cov["queued"]))
    print("  share of each class in test: " + "  ".join(
        f"{c} {100*v:.0f}%" for c, v in cov["class_share_in_test"].items()))
    os.makedirs(a.out, exist_ok=True)
    json.dump(cov, open(f"{a.out}/split.json", "w"), indent=1)
    grp, nveh = group_thin(scenes, cfg.vehicle_dims, a.min_veh)
    classes = sorted(set(grp.values()))
    cidx = {c: classes.index(grp[c]) for c in grp}
    for c in sorted(grp):
        if grp[c] != c:
            print(f"  {c} ({nveh[c]} vehicles) shares the {grp[c]} parameter vector")
    names = list(FREE_PARAMS) + ["accLongBRef", "accLatBRef", "s0yB", "s0yLatB"]
    base = default_params()
    ff = freeflow_v0(df, cfg)
    lo, hi = SPEC_BY_NAME["v0"].lo, SPEC_BY_NAME["v0"].hi
    v0_ff = {c: float(np.clip(ff.get(c, base["v0"]), lo, hi)) for c in classes}
    if a.fix_v0_freeflow:
        names.remove("v0")
        bases = [{**base, "v0": v0_ff[c]} for c in classes]
        print("  v0 FIXED per class at the free-flow 85th percentile: "
              + ", ".join(f"{c} {v0_ff[c]:.1f}" for c in classes))
    else:
        bases = [dict(base) for _ in classes]
    fixed = {k: float(v) for k, v in (kv.split("=") for kv in a.fix)}
    for k, v in fixed.items():
        if k not in SPEC_BY_NAME:
            raise SystemExit(f"--fix: unknown parameter {k}")
        if k in names: names.remove(k)
        for bb in bases: bb[k] = v
    if fixed:
        print("  FIXED for every class: " + ", ".join(f"{k}={v:g}" for k, v in fixed.items()))
    fixed_class = json.load(open(a.fix_class)) if a.fix_class else {}
    if fixed_class:
        keys = {frozenset(fixed_class.get(c, {})) for c in classes}
        if len(keys) != 1 or not next(iter(keys)):
            raise SystemExit(f"--fix-class: give the same parameters for each of {classes}")
        for k in next(iter(keys)):
            if k not in SPEC_BY_NAME:
                raise SystemExit(f"--fix-class: unknown parameter {k}")
            if k in names: names.remove(k)
        for c, bb in zip(classes, bases):
            bb.update({k: float(v) for k, v in fixed_class[c].items()})
            print(f"  FIXED for {c}: " + ", ".join(f"{k}={v:g}" for k, v in fixed_class[c].items()))
    for name in sorted(overrides):
        sp = SPEC_BY_NAME[name]
        print(f"  search box {name}: [{sp.lo:g}, {sp.hi:g}]{' log' if sp.log else ''}")
    json.dump({"scene_s": a.scene_s, "multi": a.multi, "whole_record": a.whole_record,
               "exit_signal": a.exit_signal,
               "train_frac": a.train_frac,
               "max_blocks": a.max_blocks if a.multi else 1,
               "split_seed": a.split_seed, "seed": a.seed, "maxfevals": a.maxfevals,
               "overrides": {k: list(v) for k, v in overrides.items()},
               "fixed_v0": v0_ff if a.fix_v0_freeflow else None, "fixed": fixed, "fixed_class": fixed_class,
               "max_minutes": a.max_minutes, "free": names,
               "dims": {k: list(v) for k, v in dims.items()}},
              open(f"{a.out}/run_config.json", "w"), indent=1)
    print(f"{len(classes)} classes, "
          f"{len(classes)*len(names)} free parameters, a in "
          f"[{SPEC_BY_NAME['a'].lo:g}, {SPEC_BY_NAME['a'].hi:g}]")

    global _G
    _G = (train, classes, cidx, names, bases, cfg.road, a.cap)
    import cma
    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(a.workers, initializer=_init_pool,
                               initargs=(_G, overrides)) if a.workers > 1 else None
    if a.init:
        src = json.load(open(a.init))
        z0 = np.concatenate([to_norm({**bases[k], **src[c]}, names)
                             for k, c in enumerate(classes)])
        print(f"warm start from {a.init}, sigma0={a.sigma0}")
    else:
        # v0 starts from each class's free-flow speed (when it is fitted at all)
        z0 = np.concatenate([to_norm({**bases[k], "v0": v0_ff[c]}, names)
                             for k, c in enumerate(classes)])
        if "v0" in names:
            print("  v0 seeded per class from the free-flow 85th percentile: "
                  + ", ".join(f"{c} {v0_ff[c]:.1f}" for c in classes))
    es = cma.CMAEvolutionStrategy(z0, a.sigma0, {
        "bounds": [0, NORM_HI], "maxfevals": a.maxfevals, "seed": a.seed,
        "verb_disp": 20, "verb_log": 0, "tolfun": 1e-4})
    t0 = time.time(); best = (np.inf, None)
    conv = open(f"{a.out}/convergence.csv", "w")
    conv.write("generation,evaluations,minutes,gen_best,gen_median,best_so_far,sigma\n")
    stop_reason = None
    while not es.stop():
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            stop_reason = f"time limit of {a.max_minutes:g} min"; break
        X = es.ask()
        if pool is None:
            fit = [objective(z, train, classes, cidx, names, bases, cfg.road, a.cap) for z in X]
        else:
            fit = list(pool.map(_pool_eval, X))
        es.tell(X, fit)
        if min(fit) < best[0]:
            best = (min(fit), np.array(X[int(np.argmin(fit))]))
        conv.write(f"{es.countiter},{es.countevals},{(time.time()-t0)/60:.3f},"
                   f"{min(fit):.6f},{np.median(fit):.6f},{best[0]:.6f},{es.sigma:.6g}\n")
        conv.flush()
        es.disp()
    conv.close()
    print("stopped:", stop_reason or es.stop())
    if pool is not None: pool.shutdown()
    zb = best[1]; print(f"\nfitted in {(time.time()-t0)/60:.1f} min, train F = {best[0]:.4f}")

    os.makedirs(a.out, exist_ok=True); np.save(f"{a.out}/z_best.npy", zb)
    chk = objective(zb, train, classes, cidx, names, bases, cfg.road, a.cap)
    print(f"self-check: optimiser best {best[0]:.6f}, re-evaluated in main {chk:.6f}")
    if abs(chk - best[0]) > 1e-6:
        print("ERROR: the saved parameters do not reproduce the optimiser's "
              "objective; workers and main disagree on the search box.",
              file=sys.stderr)
        return 2

    P = np.empty((len(classes), len(PARAM_ORDER))); pj = {}
    for k, c in enumerate(classes):
        p = from_norm(zb[k*len(names):(k+1)*len(names)], names, bases[k])
        P[k] = [p[n] for n in PARAM_ORDER]; pj[c] = p
    json.dump(pj, open(f"{a.out}/class_params.json", "w"), indent=2)

    rows = []
    for tag, ss in (("train", train), ("test", test)):
        for s in ss:
            for f, ex, ey, ev, cl in scene_errors(s, P, cidx, cfg.road):
                rows.append((tag, int(round(s["len_s"])), cl, f, ex, ey, ev))
    v = pd.DataFrame(rows, columns=["split", "len_s", "vclass", "F",
                                    "rmse_x_m", "rmse_y_m", "rmse_vx_ms"])
    v.to_csv(f"{a.out}/validation.csv", index=False)
    for tag in ("train", "test"):
        sub = v[v.split == tag]
        groups = [("all", sub)]
        if a.multi:
            groups += [(f"{L}s", sub[sub.len_s == L]) for L in sorted(sub.len_s.unique())]
        for label, g in groups:
            print(f"  {tag:5s} {label:>4s} n={len(g):5d}  F {g.F.median():7.3f}  "
                  f"long {g.rmse_x_m.median():8.3f}  lat {g.rmse_y_m.median():6.3f}  "
                  f"speed {g.rmse_vx_ms.median():6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
