#!/usr/bin/env python3
"""
Run the full IAM calibration pipeline.

    python run_calibration.py D3F1_lcl_smtF.csv --out results/

Start with a smoke test before committing to the full run:

    python run_calibration.py D3F1_lcl_smtF.csv --max-vehicles 8 \
        --maxfevals 400 --workers 4 --out results/smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from iam_calib import (CMAConfig, DataConfig, ObjectiveConfig, PipelineConfig,
                       RoadConfig, FREE_PARAMS)
from iam_calib import (build_windows, calibrate_all, load_frame, summarise)
from iam_calib.diagnostics import (bound_hit_report, identifiability,
                                   pool_by_class, results_frame,
                                   shrink_thin_classes, train_test_split,
                                   validate)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--window", type=float, default=30.0, help="window length [s]")
    ap.add_argument("--min-window", type=float, default=15.0)
    ap.add_argument("--reset-s", type=float, default=3.0,
                    help="re-init ego state from data every N s; 0 = free-running")
    ap.add_argument("--x-scale", type=float, default=2.0,
                    help="longitudinal error normalisation [m]")
    ap.add_argument("--dt", type=float, default=0.04)
    ap.add_argument("--max-vehicles", type=int, default=0, help="0 = all")
    ap.add_argument("--maxfevals", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=0, help="0 = all cores")
    ap.add_argument("--staged", action="store_true",
                    help="longitudinal stage, then lateral, then joint")
    ap.add_argument("--no-polish", action="store_true")
    ap.add_argument("--restarts", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--y-scale", type=float, default=0.5,
                    help="lateral error normalisation [m]; the key GoF knob")
    ap.add_argument("--road-width", type=float, default=0.0,
                    help="enable boundary forces with this width [m], "
                         "centred on the median of Y_Smt")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--skip-diagnostics", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    cfg = PipelineConfig(
        data=DataConfig(dt=args.dt, window_s=args.window,
                        min_window_s=args.min_window),
        objective=ObjectiveConfig(y_scale=args.y_scale,
                                  x_scale=args.x_scale,
                                  reset_s=args.reset_s),
        cma=CMAConfig(maxfevals=args.maxfevals, seed=args.seed,
                      polish=not args.no_polish, n_restarts=args.restarts),
        free_params=FREE_PARAMS,
        staged=args.staged,
        n_workers=args.workers,
        max_vehicles=args.max_vehicles,
    )

    # -- load ---------------------------------------------------------------
    print(f"loading {args.csv} ...", flush=True)
    df = load_frame(args.csv, cfg.data)
    print(f"  {len(df):,} rows, "
          f"{df[cfg.data.col_id].nunique():,} vehicles, "
          f"{df[cfg.data.col_time].max() - df[cfg.data.col_time].min():.0f} s span")

    smry = summarise(df, cfg.data)
    smry.to_csv(os.path.join(args.out, "class_summary.csv"))
    print("\nper-class summary:")
    print(smry.to_string())

    if args.road_width > 0:
        y_mid = float(df[cfg.data.col_y].median())
        cfg.road = RoadConfig(enabled=True,
                              y_left=y_mid + args.road_width / 2,
                              y_right=y_mid - args.road_width / 2)
        cfg.free_params = tuple(list(FREE_PARAMS) +
                                ["accLongBRef", "accLatBRef", "s0yB", "s0yLatB"])
        print(f"\nboundary forces ON: y in "
              f"[{cfg.road.y_right:.2f}, {cfg.road.y_left:.2f}]")
    else:
        print("\nboundary forces OFF (pass --road-width to enable)")

    # -- windows ------------------------------------------------------------
    print("\nbuilding windows ...", flush=True)
    windows = build_windows(df, cfg)
    if not windows:
        print("ERROR: no usable windows. Lower --min-window.", file=sys.stderr)
        return 1

    train, test = train_test_split(windows, frac=args.train_frac, seed=args.seed)
    print(f"  {len(train)} train / {len(test)} test vehicles")

    # -- calibrate ----------------------------------------------------------
    n_free = len(cfg.free_params)
    print(f"\ncalibrating {n_free} free parameters on {len(train)} vehicles "
          f"({cfg.cma.maxfevals} evals each) ...", flush=True)
    results = calibrate_all(train, cfg)

    res_df = results_frame(results)
    res_df.to_csv(os.path.join(args.out, "per_vehicle_params.csv"), index=False)
    n_ok = int(res_df["converged"].sum())
    print(f"\n  {n_ok}/{len(res_df)} converged; "
          f"median fitness {res_df.loc[res_df['converged'], 'fitness'].median():.4f}")

    # -- pool ---------------------------------------------------------------
    pooled = pool_by_class(res_df, cfg.free_params)
    pooled_s = shrink_thin_classes(pooled, cfg.free_params)
    pooled_s.to_csv(os.path.join(args.out, "class_params.csv"), index=False)
    print("\nclass-level parameters (after shrinkage):")
    cols = ["vclass", "n_vehicles", "median_fitness"] + list(cfg.free_params)
    print(pooled_s[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    params_by_class = {
        r["vclass"]: {n: float(r[n]) for n in cfg.free_params}
        for _, r in pooled_s.iterrows()
    }
    with open(os.path.join(args.out, "class_params.json"), "w") as fh:
        json.dump(params_by_class, fh, indent=2)

    # -- diagnostics --------------------------------------------------------
    if not args.skip_diagnostics:
        print("\nbound hits (high rate => bound wrong, or param unidentified):")
        bh = bound_hit_report(res_df, cfg.free_params)
        bh.to_csv(os.path.join(args.out, "bound_hits.csv"), index=False)
        print(bh.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

        print("\nidentifiability screen:")
        ident = identifiability(train, results, cfg, n_vehicles=20)
        if not ident.empty:
            ident.to_csv(os.path.join(args.out, "identifiability.csv"), index=False)
            print(ident.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # -- validate -----------------------------------------------------------
    if test:
        print(f"\nvalidating class parameters on {len(test)} held-out vehicles ...")
        val = validate(test, params_by_class, cfg)
        val.to_csv(os.path.join(args.out, "validation.csv"), index=False)
        ok = val[val["total"] < cfg.objective.penalty]
        print(f"  {len(ok)}/{len(val)} simulated without divergence")
        if len(ok):
            print(f"  median total GoF      {ok['total'].median():.4f}")
            print(f"  median RMSE long [m]  {ok['rmse_x_m'].median():.3f}")
            print(f"  median RMSE lat  [m]  {ok['rmse_y_m'].median():.3f}")
            print(f"  median RMSE speed[m/s]{ok['rmse_vx_ms'].median():.3f}")

    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min -> {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
