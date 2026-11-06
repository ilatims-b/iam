#!/usr/bin/env python3
"""One-at-a-time sensitivity profiles around a fitted parameter set.

For every class vector and every model parameter, sweep that one value across
its search box (config.REALISTIC_BOUNDS) in steps of config.GRANULARITY, with
everything else held at the fit, and re-simulate the whole record each time.
Recorded per point, on the training vehicles: the objective F (mean capped F,
the quantity the fit minimises) and the pooled RMSE of longitudinal position,
lateral position and speed.

A flat profile means the data cannot pin the parameter down; a sharp minimum
means it is identified; a minimum on the edge of the box means the box binds.

    python oat_profiles.py --run results/j12_dims --workers 8
"""
import argparse, json, os, time
import numpy as np, pandas as pd

from iam_calib.config import PipelineConfig, REALISTIC_BOUNDS, GRANULARITY
from iam_calib.calibrate import default_params
from iam_calib.fastcore import PARAM_ORDER
from iam_calib import calibrate_joint as cj

MAX_POINTS = 60
_G = {}


def grid(name):
    lo, hi, log = REALISTIC_BOUNDS[name]
    g = GRANULARITY[name]
    if g.endswith("%"):
        r = 1 + float(g[:-1]) / 100
        v = lo * r ** np.arange(int(np.floor(np.log(hi / lo) / np.log(r))) + 1)
    else:
        st = float(g); v = lo + st * np.arange(int(round((hi - lo) / st)) + 1)
    v = v[v <= hi + 1e-9]
    if len(v) > MAX_POINTS:
        v = v[np.linspace(0, len(v) - 1, MAX_POINTS).round().astype(int)]
    return v


def _init(run):
    cfg = PipelineConfig(); cfg.road.y_right, cfg.road.y_left, cfg.road.enabled = 0.0, 8.9, True
    rc = json.load(open(f"{run}/run_config.json"))
    cj.apply_bounds({k: tuple(v) for k, v in rc["overrides"].items()})
    cfg.vehicle_dims.update({k: tuple(v) for k, v in rc.get("dims", {}).items()})
    df = pd.read_csv("D3F1_lcl_smtF.csv")
    tr, _, _ = cj.whole_record(df, cfg, 1 - rc.get("train_frac", 0.7), rc.get("split_seed", 1),
                               rc.get("exit_signal", False))
    cp = json.load(open(f"{run}/class_params.json")); classes = sorted(cp)
    P = np.array([[{**default_params(), **cp[c]}[k] for k in PARAM_ORDER] for c in classes])
    grp, _ = cj.group_thin([tr], cfg.vehicle_dims, 10)
    _G.update(tr=tr, P=P, classes=classes, road=cfg.road,
              cidx={c: classes.index(grp[c]) for c in grp})


def _eval(job):
    k, name, val = job
    P = _G["P"].copy()
    if k is not None:
        P[k, PARAM_ORDER.index(name)] = val
    rows = cj.scene_errors(_G["tr"], P, _G["cidx"], _G["road"], with_n=True)
    F = np.array([60.0 if not np.isfinite(r[0]) else min(r[0], 60.0) for r in rows])
    n = np.array([r[5] for r in rows], float)
    pooled = lambda j: float(np.sqrt(np.nansum(n * np.array([r[j] for r in rows]) ** 2)
                                     / n[np.isfinite([r[j] for r in rows])].sum()))
    return dict(cls=None if k is None else _G["classes"][k], param=name, value=val,
                F=float(F.mean()), long=pooled(1), lat=pooled(2), speed=pooled(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--params", nargs="*", default=None)
    a = ap.parse_args()
    _init(a.run)
    names = a.params or [n for n in PARAM_ORDER if n in REALISTIC_BOUNDS]
    jobs = [(None, "base", np.nan)]
    for k in range(len(_G["classes"])):
        for n in names:
            jobs += [(k, n, float(v)) for v in grid(n)]
    print(f"{len(jobs)} simulations on {a.workers} workers")
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.run,)) as ex:
        res = list(ex.map(_eval, jobs, chunksize=4))
    d = pd.DataFrame(res); d.to_csv(f"{a.run}/oat_profiles.csv", index=False)
    print(f"done in {(time.time()-t0)/60:.1f} min; wrote {a.run}/oat_profiles.csv")


if __name__ == "__main__":
    main()
