"""
Data loading and preparation.

Turns a flat trajectory CSV into a list of `VehicleWindow` objects, each
carrying the ego trajectory plus a padded (T, K) array of candidate neighbour
states replayed from the data.

The candidate set is fixed once, from the MEASURED ego position, using a
generous radius -- the simulated ego drifts, and must not lose a neighbour
mid-window just because it fell behind. Actual interaction selection happens
inside model.accelerations() against the SIMULATED position.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import DataConfig, PipelineConfig, dims_for_class


@dataclass
class VehicleWindow:
    """One ego vehicle over one contiguous time window."""
    vid: object
    vclass: str
    L: float
    W: float
    dt: float
    t: np.ndarray          # (T,) seconds, relative to window start
    x: np.ndarray          # (T,) observed longitudinal position [m]
    y: np.ndarray          # (T,) observed lateral position [m]
    vx: np.ndarray         # (T,) observed longitudinal speed [m/s]
    vy: np.ndarray         # (T,) observed lateral speed [m/s]
    ax: np.ndarray         # (T,) observed longitudinal acceleration [m/s^2]
    nb_x: np.ndarray       # (T, K)
    nb_y: np.ndarray
    nb_vx: np.ndarray
    nb_vy: np.ndarray
    nb_ax: np.ndarray      # neighbours' measured long. acceleration (a_l of ACC)
    nb_L: np.ndarray
    nb_W: np.ndarray
    valid: np.ndarray      # (T, K) bool

    @property
    def n_steps(self) -> int:
        return len(self.t)

    @property
    def mean_neighbours(self) -> float:
        return float(self.valid.sum(axis=1).mean())


# ---------------------------------------------------------------------------
def load_frame(path: str, cfg: DataConfig,
               usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Read the CSV and validate that the expected columns exist."""
    needed = [cfg.col_id, cfg.col_class, cfg.col_time, cfg.col_x, cfg.col_y,
              cfg.col_vx, cfg.col_vy, cfg.col_ax]
    df = pd.read_csv(path, usecols=usecols)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing expected columns {missing}. Found: {list(df.columns)}. "
            f"Adjust DataConfig column names to match your file."
        )
    df = df.dropna(subset=needed)
    return df


def summarise(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Per-class summary. Run this before calibrating anything."""
    g = df.groupby(cfg.col_class)
    out = pd.DataFrame({
        "n_rows": g.size(),
        "n_vehicles": g[cfg.col_id].nunique(),
        "mean_duration_s": g.size() / g[cfg.col_id].nunique() * cfg.dt,
        "mean_speed_ms": g[cfg.col_vx].mean(),
        "y_min": g[cfg.col_y].min(),
        "y_max": g[cfg.col_y].max(),
    })
    return out.sort_values("n_vehicles", ascending=False)


# ---------------------------------------------------------------------------
def build_windows(df: pd.DataFrame, pcfg: PipelineConfig,
                  verbose: bool = True) -> List[VehicleWindow]:
    """
    Build one calibration window per vehicle.

    Vehicles longer than `window_s` are truncated to their first window; a
    vehicle contributes at most one window so that per-vehicle parameter
    estimates stay independent.
    """
    cfg = pcfg.data
    dims = pcfg.vehicle_dims

    d = df[[cfg.col_id, cfg.col_class, cfg.col_time, cfg.col_x, cfg.col_y,
            cfg.col_vx, cfg.col_vy, cfg.col_ax]].copy()
    d.columns = ["vid", "cls", "t", "x", "y", "vx", "vy", "ax"]

    # discrete frame index -- makes the neighbour join exact
    t0 = d["t"].min()
    d["frame"] = np.rint((d["t"] - t0) / cfg.dt).astype(np.int64)

    L_map = {c: dims_for_class(c, dims)[0] for c in d["cls"].unique()}
    W_map = {c: dims_for_class(c, dims)[1] for c in d["cls"].unique()}
    d["L"] = d["cls"].map(L_map).astype(float)
    d["W"] = d["cls"].map(W_map).astype(float)

    if getattr(cfg, "x_is_front", False) is False:
        # dataset gives the centroid; the model wants the front bumper
        d["x"] = d["x"] + 0.5 * d["L"]

    d = d.sort_values(["frame", "vid"], kind="mergesort").reset_index(drop=True)

    # frame -> [start, end) row slice
    frames = d["frame"].to_numpy()
    uniq_frames, frame_start = np.unique(frames, return_index=True)
    frame_end = np.append(frame_start[1:], len(d))
    frame_lookup = {int(f): (int(s), int(e))
                    for f, s, e in zip(uniq_frames, frame_start, frame_end)}

    arr = {c: d[c].to_numpy() for c in ("x", "y", "vx", "vy", "ax", "L", "W")}
    vid_arr = d["vid"].to_numpy()

    min_steps = int(round(cfg.min_window_s / cfg.dt))
    max_steps = int(round(cfg.window_s / cfg.dt))
    K = cfg.max_neighbours

    windows: List[VehicleWindow] = []
    n_trunc = 0

    for vid, grp in d.groupby("vid", sort=False):
        grp = grp.sort_values("frame")
        gframes = grp["frame"].to_numpy()

        # keep the longest contiguous run (guards against ID reuse / gaps)
        breaks = np.where(np.diff(gframes) != 1)[0]
        seg_start = np.concatenate([[0], breaks + 1])
        seg_end = np.concatenate([breaks + 1, [len(gframes)]])
        seg_len = seg_end - seg_start
        k = int(np.argmax(seg_len))
        s, e = int(seg_start[k]), int(seg_end[k])

        if (e - s) < min_steps:
            continue
        if (e - s) > max_steps:
            e = s + max_steps
            n_trunc += 1

        sub = grp.iloc[s:e]
        T = len(sub)
        ego_frames = sub["frame"].to_numpy()
        ex = sub["x"].to_numpy(float)
        ey = sub["y"].to_numpy(float)
        evx = sub["vx"].to_numpy(float)
        evy = sub["vy"].to_numpy(float)
        eax = sub["ax"].to_numpy(float)
        cls = str(sub["cls"].iloc[0])
        Lego = float(sub["L"].iloc[0])
        Wego = float(sub["W"].iloc[0])

        nb = {n: np.zeros((T, K), dtype=np.float64)
              for n in ("x", "y", "vx", "vy", "ax", "L", "W")}
        valid = np.zeros((T, K), dtype=bool)
        overflow = 0

        for i in range(T):
            span = frame_lookup.get(int(ego_frames[i]))
            if span is None:
                continue
            a, b = span
            sl = slice(a, b)
            others = vid_arr[sl] != vid
            if not others.any():
                continue
            ox, oy = arr["x"][sl], arr["y"][sl]
            near = others & (np.abs(ox - ex[i]) <= cfg.radius_x) \
                          & (np.abs(oy - ey[i]) <= cfg.radius_y)
            idx = np.flatnonzero(near)
            if idx.size == 0:
                continue
            if idx.size > K:
                # keep the K longitudinally closest
                overflow += 1
                order = np.argsort(np.abs(ox[idx] - ex[i]))[:K]
                idx = idx[order]
            n = idx.size
            gidx = a + idx
            for name in ("x", "y", "vx", "vy", "ax", "L", "W"):
                nb[name][i, :n] = arr[name][gidx]
            valid[i, :n] = True

        if overflow > 0.05 * T and verbose:
            warnings.warn(
                f"vehicle {vid}: neighbour cap K={K} hit on {overflow}/{T} "
                f"steps; consider raising DataConfig.max_neighbours")

        windows.append(VehicleWindow(
            vid=vid, vclass=cls, L=Lego, W=Wego, dt=cfg.dt,
            t=np.arange(T) * cfg.dt,
            x=ex, y=ey, vx=evx, vy=evy, ax=eax,
            nb_x=nb["x"], nb_y=nb["y"], nb_vx=nb["vx"], nb_vy=nb["vy"],
            nb_ax=nb["ax"], nb_L=nb["L"], nb_W=nb["W"], valid=valid,
        ))

        if pcfg.max_vehicles and len(windows) >= pcfg.max_vehicles:
            break

    if verbose:
        dens = np.mean([w.mean_neighbours for w in windows]) if windows else 0
        print(f"built {len(windows)} windows "
              f"({n_trunc} truncated to {cfg.window_s:g}s), "
              f"mean {dens:.1f} candidate neighbours/step")
    return windows
