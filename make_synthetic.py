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

from iam_calib.config import (MODEL_CONSTANTS as C, RoadConfig,
                              dims_for_class, DEFAULT_VEHICLE_DIMS)
from iam_calib.model import accelerations

TRUE_PARAMS = {
    "v0": 11.0, "T": 1.1, "s0": 1.4, "a": 1.3, "b": 1.8,
    "s0y": 0.22, "s0yLat": 0.35, "sensLat": 1.6, "tauLatOVM": 0.9,
    "pushLong": 0.12, "pushLat": 0.15,
    "accLongBRef": 0.2, "accLatBRef": 4.0, "s0yB": 0.1, "s0yLatB": 0.1,
    "accBiasLat": 0.0,
}

# Matches D3F1_lcl_smtF.csv's real class proportions (Bike 46%, Car 23%,
# LCV 12%, Truck 9%, Auto 6%, MultiAxle 3%, Trailer 1%), so the ground-truth
# test spans the same fleet the real calibration has to fit -- previously
# this covered only 4 of the 7 real classes (no LCV/Truck/MultiAxle/Trailer),
# which meant the classes with the worst real-data lateral RMSE (Truck,
# MultiAxle -- see the doc's §7/§8) never got a synthetic ground-truth check.
CLASS_MIX = [("motorcycle", 0.46), ("car", 0.23), ("lcv", 0.12),
             ("truck", 0.09), ("auto", 0.06), ("multiaxle", 0.03),
             ("trailer", 0.01)]

# NOTE on --length: vehicles are placed simultaneously along [0, length] at
# t=0 (this generator has no vehicles entering/leaving over time), so density
# = vehicles/length is set ENTIRELY by these two flags together -- raising
# --vehicles without raising --length proportionally silently packs vehicles
# bumper-to-bumper. D3F1_lcl_smtF.csv's own measured concurrent density is
# ~0.21 vehicles/m (830,240 rows / 781s of window / 204m corridor extent);
# match that, don't just scale --vehicles alone.


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
        if cls[i] == "motorcycle":
            p["v0"] *= 1.15
        # Heavy-class multipliers below are GUESSED, exactly like the
        # motorcycle line above -- lower desired speed/acceleration/braking
        # for bigger, heavier vehicles is directionally defensible, but the
        # specific factors are not measured or sourced from literature. If
        # you have real per-class speed/accel data, replace these.
        if cls[i] == "lcv":
            p["v0"] *= 0.90
            p["a"] *= 0.85
        if cls[i] == "truck":
            # sim_bicycles_IAM.js accBiasRightTruck=+0.8 (right-positive);
            # our frame is left-positive. This one IS sourced from the JS.
            p["accBiasLat"] = -0.8
            p["v0"] *= 0.75
            p["a"] *= 0.60
            p["b"] *= 0.90
        if cls[i] == "multiaxle":
            p["v0"] *= 0.65
            p["a"] *= 0.50
            p["b"] *= 0.85
        if cls[i] == "trailer":
            p["v0"] *= 0.60
            p["a"] *= 0.45
            p["b"] *= 0.80
        p_veh.append(p)

    x = np.sort(rng.uniform(0, args.length, N))[::-1].copy()
    y = rng.uniform(-args.road_width / 2 + 0.6, args.road_width / 2 - 0.6, N)
    vx = np.array([p["v0"] for p in p_veh]) * rng.uniform(0.6, 0.95, N)
    vy = np.zeros(N)

    rows = []
    valid_all = np.ones(N - 1, dtype=bool)
    ax_prev = np.zeros(N)                       # previous-step accelerations
    dvdu = vy / np.maximum(vx, 1e-4)            # stored heading angle (tan)
    dt = args.dt

    for k in range(T):
        t = k * dt
        ax = np.zeros(N); ay = np.zeros(N)
        for i in range(N):
            j = np.arange(N) != i
            ax[i], ay[i] = accelerations(
                x[i], y[i], vx[i], vy[i],
                x[j], y[j], vx[j], vy[j], ax_prev[j], L[j], W[j], valid_all,
                L[i], W[i], p_veh[i], road, ax_prev[i],
            )
        ax += rng.normal(0, args.noise, N)
        ay += rng.normal(0, args.noise * 0.5, N)
        ax_prev = ax.copy()

        for i in range(N):
            rows.append((f"V_{i:03d}", cls[i], t, 1000.0 + t,
                         x[i] - 0.5 * L[i], y[i],      # X, Y (centroid)
                         x[i] - 0.5 * L[i], y[i],      # X_Smt, Y_Smt
                         vx[i], vy[i], np.hypot(vx[i], vy[i]),
                         ax[i], ay[i], np.hypot(ax[i], ay[i])))

        # same integration as objective.simulate / fastcore.simulate_core
        vx_n = vx + ax * dt
        vy_n = vy + ay * dt
        dvdu_new = vy_n / np.maximum(vx_n, 1e-4)
        move = np.abs(dvdu_new - dvdu) > dt * C.dotdvdumax
        dvdu = np.where(move, dvdu + np.sign(dvdu_new - dvdu) * C.dotdvdumax * dt,
                        dvdu)
        dvdu = np.clip(dvdu, -C.dvdumax, C.dvdumax)
        vy_max = np.abs(dvdu * vx_n)
        vy_n = np.clip(vy_n, -vy_max, vy_max)
        dy_step = 0.5 * (vy + vy_n) * dt
        stopped = vx_n < -1e-6
        dx_step = np.where(stopped,
                           -0.5 * vx * vx / np.where(ax == 0, 1.0, ax),
                           0.5 * (vx + vx_n) * dt)
        vx_n = np.maximum(0.0, vx_n)
        x = x + dx_step
        y = np.clip(y + dy_step, -args.road_width / 2 - 0.3,
                    args.road_width / 2 + 0.3)
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
