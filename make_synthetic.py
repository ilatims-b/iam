#!/usr/bin/env python3
"""
Generate synthetic lane-free trajectory data with KNOWN parameters, in the same
CSV schema as D3F1_lcl_smtF.csv.

This is the pipeline's ground truth test: run the calibration on the output and
check that it recovers the parameters used here. If recovery fails on synthetic
data it will certainly fail on real data, and you will not be able to tell.

    python make_synthetic.py --out synthetic.csv --vehicles 40 --duration 60
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from iam_calib.config import RoadConfig, dims_for_class, DEFAULT_VEHICLE_DIMS
from iam_calib.model import accelerations

TRUE_PARAMS = {
    "v0": 11.0, "T": 1.1, "s0": 1.4, "a": 1.3, "b": 1.8,
    "s0y": 0.22, "s0yLat": 0.35, "sensLat": 1.6, "tauLatOVM": 0.9,
    "pushLong": 0.12, "pushLat": 0.15,
    "accLongBRef": 0.2, "accLatBRef": 4.0, "s0yB": 0.1, "s0yLatB": 0.1,
}

CLASS_MIX = [("car", 0.35), ("auto", 0.25), ("motorcycle", 0.30), ("bus", 0.10)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic.csv")
    ap.add_argument("--vehicles", type=int, default=40)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.04)
    ap.add_argument("--road-width", type=float, default=7.0)
    ap.add_argument("--length", type=float, default=400.0)
    ap.add_argument("--noise", type=float, default=0.15,
                    help="acceleration noise std [m/s^2]")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    N = args.vehicles
    T = int(args.duration / args.dt)
    road = RoadConfig(enabled=True, y_left=args.road_width / 2,
                      y_right=-args.road_width / 2)

    classes, probs = zip(*CLASS_MIX)
    cls = rng.choice(classes, size=N, p=np.array(probs) / sum(probs))
    L = np.array([dims_for_class(c, DEFAULT_VEHICLE_DIMS)[0] for c in cls])
    W = np.array([dims_for_class(c, DEFAULT_VEHICLE_DIMS)[1] for c in cls])

    # per-vehicle heterogeneity, so pooling has something to average over
    p_veh = []
    for i in range(N):
        p = dict(TRUE_PARAMS)
        p["v0"] *= float(rng.normal(1.0, 0.12))
        p["T"] *= float(rng.normal(1.0, 0.15))
        p["a"] *= float(rng.normal(1.0, 0.12))
        if cls[i] == "bus":
            p["v0"] *= 0.85
            p["a"] *= 0.7
        if cls[i] == "motorcycle":
            p["v0"] *= 1.15
        p_veh.append(p)

    x = np.sort(rng.uniform(0, args.length, N))[::-1].copy()
    y = rng.uniform(-args.road_width / 2 + 0.6, args.road_width / 2 - 0.6, N)
    vx = np.array([p["v0"] for p in p_veh]) * rng.uniform(0.6, 0.95, N)
    vy = np.zeros(N)

    rows = []
    valid_all = np.ones(N - 1, dtype=bool)

    for k in range(T):
        t = k * args.dt
        ax = np.zeros(N); ay = np.zeros(N)
        for i in range(N):
            j = np.arange(N) != i
            ax[i], ay[i] = accelerations(
                x[i], y[i], vx[i], vy[i],
                x[j], y[j], vx[j], vy[j], L[j], W[j], valid_all,
                L[i], W[i], p_veh[i], road,
            )
        ax += rng.normal(0, args.noise, N)
        ay += rng.normal(0, args.noise * 0.5, N)

        for i in range(N):
            rows.append((f"V_{i:03d}", cls[i], t, 1000.0 + t,
                         x[i] - 0.5 * L[i], y[i],      # X, Y (centroid)
                         x[i] - 0.5 * L[i], y[i],      # X_Smt, Y_Smt
                         vx[i], vy[i], np.hypot(vx[i], vy[i]),
                         ax[i], ay[i], np.hypot(ax[i], ay[i])))

        vx_n = np.maximum(0.0, vx + ax * args.dt)
        vy_n = vy + ay * args.dt
        x = x + vx * args.dt + 0.5 * ax * args.dt ** 2
        y = y + vy * args.dt + 0.5 * ay * args.dt ** 2
        y = np.clip(y, -args.road_width / 2 - 0.3, args.road_width / 2 + 0.3)
        vx, vy = vx_n, vy_n

    df = pd.DataFrame(rows, columns=[
        "VehicleID", "Class", "Time", "GTime_sec", "X", "Y", "X_Smt", "Y_Smt",
        "Long_Speed", "Lat_Speed", "Speed", "Long_Acc", "Lat_Acc",
        "Acceleration"])
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows, {N} vehicles, {args.duration:g}s -> {args.out}")
    print("true parameters:")
    for k, v in TRUE_PARAMS.items():
        print(f"  {k:<14s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
