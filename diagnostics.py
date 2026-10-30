"""
Post-calibration diagnostics.

Per-vehicle calibration of an 11-parameter model will produce a good in-sample
fit almost regardless of whether the model is right. These checks are what tell
you whether the numbers mean anything.

Run, in order:
    bound_hit_report()   -- which parameters are pinned to their bounds
    identifiability()    -- which parameters the objective is flat in
    pool_by_class()      -- class-level estimates with spread
    validate()           -- held-out fit
"""

from __future__ import annotations

from typing import Tuple, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .calibrate import CalibResult, default_params
from .config import DEFAULT_VEHICLE_DIMS, dims_for_class, PipelineConfig, SPEC_BY_NAME
from .data import VehicleWindow
from .objective import gof, gof_components


def results_frame(results: Sequence[CalibResult]) -> pd.DataFrame:
    return pd.DataFrame([r.flat() for r in results])


# ---------------------------------------------------------------------------
def bound_hit_report(df: pd.DataFrame, free_params: Sequence[str],
                     tol: float = 0.02) -> pd.DataFrame:
    """
    Fraction of vehicles whose estimate sits within `tol` (relative to range)
    of a bound.

    A high rate means one of three things, and you must decide which:
      - the bound is too tight and should be widened;
      - the parameter is unidentified and CMA-ES is drifting to a bound;
      - the model is misspecified for that class.
    Widening the bound is the least likely correct answer.
    """
    ok = df[df["converged"]] if "converged" in df else df
    rows = []
    for n in free_params:
        s = SPEC_BY_NAME[n]
        v = ok[n].to_numpy(float)
        if s.log:
            lo, hi, vv = np.log(s.lo), np.log(s.hi), np.log(np.maximum(v, 1e-12))
        else:
            lo, hi, vv = s.lo, s.hi, v
        u = (vv - lo) / (hi - lo)
        rows.append({
            "param": n,
            "at_lower": float(np.mean(u < tol)),
            "at_upper": float(np.mean(u > 1 - tol)),
            "median": float(np.median(v)),
            "iqr": float(np.subtract(*np.percentile(v, [75, 25]))),
            "cv": float(np.std(v) / max(abs(np.mean(v)), 1e-9)),
        })
    return pd.DataFrame(rows).sort_values("cv", ascending=False)


def identifiability(windows: Sequence[VehicleWindow],
                    results: Sequence[CalibResult], pcfg: PipelineConfig,
                    n_vehicles: int = 20, rel_step: float = 0.25,
                    seed: int = 0) -> pd.DataFrame:
    """
    One-at-a-time local sensitivity at each vehicle's own optimum.

    For each parameter, perturb it by +/- rel_step of its search range and
    record the relative rise in the objective. Parameters with a near-zero rise
    are not constrained by your data: fix them at Table 1 values and say so,
    rather than reporting a number the data never determined.

    This is a local screen, not a global Sobol analysis. It is cheap and
    catches the usual offenders (pushLat, s0y in sparse traffic).
    """
    rng = np.random.default_rng(seed)
    by_vid = {r.vid: r for r in results if r.converged}
    pool = [w for w in windows if w.vid in by_vid]
    if not pool:
        return pd.DataFrame()
    pick = rng.choice(len(pool), size=min(n_vehicles, len(pool)), replace=False)

    rows: List[Dict[str, float]] = []
    for i in pick:
        w = pool[int(i)]
        r = by_vid[w.vid]
        f0 = gof(w, r.params, pcfg.road, pcfg.objective)
        if not np.isfinite(f0) or f0 <= 0:
            continue
        for n in pcfg.free_params:
            s = SPEC_BY_NAME[n]
            span = (np.log(s.hi) - np.log(s.lo)) if s.log else (s.hi - s.lo)
            deltas = []
            for sgn in (-1.0, 1.0):
                p = dict(r.params)
                if s.log:
                    p[n] = float(np.clip(np.exp(np.log(p[n]) + sgn * rel_step * span),
                                         s.lo, s.hi))
                else:
                    p[n] = float(np.clip(p[n] + sgn * rel_step * span, s.lo, s.hi))
                fp = gof(w, p, pcfg.road, pcfg.objective)
                if fp >= pcfg.objective.penalty:
                    continue          # divergence, not sensitivity
                deltas.append(fp - f0)
            if deltas:
                rows.append({"vid": w.vid, "param": n,
                             "rel_rise": float(max(deltas) / f0)})

    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    out = d.groupby("param")["rel_rise"].agg(["median", "mean", "max"])
    out = out.sort_values("median", ascending=False)
    out["verdict"] = np.where(out["median"] < 0.01, "UNIDENTIFIED",
                       np.where(out["median"] < 0.05, "weak", "identified"))
    return out.reset_index()


# ---------------------------------------------------------------------------
def pool_by_class(df: pd.DataFrame, free_params: Sequence[str],
                  min_vehicles: int = 15,
                  trim_quantile: float = 0.9) -> pd.DataFrame:
    """
    Class-level point estimates: median over the best-fitting vehicles.

    Trimming to the best `trim_quantile` of fits by fitness matters more than
    it looks. Poorly-fitting vehicles are usually ones where the replayed-
    neighbour setup broke down (leader left the camera footprint, ID switch),
    and their parameter estimates are noise that will drag a mean around. The
    median plus trimming is the cheap robust option; the principled one is the
    hierarchical model described in the README.
    """
    ok = df[df["converged"]].copy()
    rows = []
    for cls, g in ok.groupby("vclass"):
        cut = g["fitness"].quantile(trim_quantile)
        gg = g[g["fitness"] <= cut]
        row: Dict[str, object] = {
            "vclass": cls,
            "n_vehicles": len(g),
            "n_used": len(gg),
            "median_fitness": float(gg["fitness"].median()),
            "thin": len(g) < min_vehicles,
        }
        for n in free_params:
            row[n] = float(gg[n].median())
            row[f"{n}_p25"] = float(gg[n].quantile(0.25))
            row[f"{n}_p75"] = float(gg[n].quantile(0.75))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n_vehicles", ascending=False)
    return out


def shrink_thin_classes(pooled: pd.DataFrame, free_params: Sequence[str],
                        min_vehicles: int = 10,
                        dims: Optional[Dict[str, Tuple[float, float]]] = None
                        ) -> pd.DataFrame:
    """
    Give a thin class the parameters of a similar vehicle type.

    An earlier version shrank thin classes toward the fleet median, which
    produces a number that is a measurement of neither the class nor anything
    else. A class with fewer than `min_vehicles` vehicles now simply adopts the
    fitted vector of the dimensionally nearest class that clears the threshold,
    so every reported value remains an actual fit to actual vehicles of a
    comparable size.
    """
    dims = dims or DEFAULT_VEHICLE_DIMS
    out = pooled.copy()
    big = pooled[pooled["n_vehicles"] >= min_vehicles]
    if big.empty:                      # nothing to borrow from
        for n in free_params:
            out[f"{n}_shared"] = False
        return out
    for i, row in pooled.iterrows():
        if row["n_vehicles"] >= min_vehicles:
            for n in free_params:
                out.at[i, f"{n}_shared"] = False
            continue
        Lc, Wc = dims_for_class(row["vclass"], dims)
        donor = min(
            big.itertuples(),
            key=lambda r: abs(dims_for_class(r.vclass, dims)[0] - Lc)
                        + abs(dims_for_class(r.vclass, dims)[1] - Wc))
        for n in free_params:
            out.at[i, n] = getattr(donor, n)
            out.at[i, f"{n}_shared"] = True
        out.at[i, "shared_with"] = donor.vclass
    return out


# ---------------------------------------------------------------------------
def validate(windows: Sequence[VehicleWindow], params_by_class: Dict[str, Dict[str, float]],
             pcfg: PipelineConfig,
             fallback: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Apply class-level parameters to held-out vehicles and report fit.

    This is the number to publish. In-sample per-vehicle fitness is not
    evidence about the model -- an 11-parameter model fitted to one 40 s
    trajectory will fit that trajectory.
    """
    fallback = fallback or default_params()
    rows = []
    for w in windows:
        p = params_by_class.get(w.vclass, fallback)
        c = gof_components(w, p, pcfg.road, pcfg.objective)
        rows.append({"vid": w.vid, "vclass": w.vclass, **c})
    return pd.DataFrame(rows)


def train_test_split(windows: Sequence[VehicleWindow], frac: float = 0.7,
                     seed: int = 0):
    """Split by VEHICLE, never by timestep -- adjacent frames are not independent."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(windows))
    n = int(frac * len(windows))
    return ([windows[int(i)] for i in idx[:n]],
            [windows[int(i)] for i in idx[n:]])
