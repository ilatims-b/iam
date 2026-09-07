"""
CMA-ES calibration.

Per-vehicle calibration is embarrassingly parallel, so vehicles are farmed out
to a process pool. Within a vehicle:

    1. warm start (v0 from observed free-flow speed, rest from defaults)
    2. CMA-ES in a normalised [0, 10] box
    3. optional Nelder-Mead polish

Parameters are searched in a normalised space rather than physical units
because CMA-ES assumes a roughly isotropic initial distribution, and physical
ranges here span three orders of magnitude (s0y ~ 0.1 m, v0 ~ 15 m/s). Scale-
like parameters (widths, times) are additionally searched in log space.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (CMAConfig, ObjectiveConfig, PipelineConfig, RoadConfig,
                     SPEC_BY_NAME, STAGE_LATERAL, STAGE_LONGITUDINAL)
from .data import VehicleWindow
from .objective import free_flow_speed_estimate, gof, gof_components

_NORM_HI = 10.0


# ---------------------------------------------------------------------------
# Parameter <-> normalised vector
# ---------------------------------------------------------------------------
def to_norm(p: Dict[str, float], names: Sequence[str]) -> np.ndarray:
    out = np.empty(len(names))
    for i, n in enumerate(names):
        s = SPEC_BY_NAME[n]
        if s.log:
            lo, hi, v = np.log(s.lo), np.log(s.hi), np.log(max(p[n], s.lo))
        else:
            lo, hi, v = s.lo, s.hi, p[n]
        out[i] = _NORM_HI * (v - lo) / (hi - lo)
    return np.clip(out, 0.0, _NORM_HI)


def from_norm(z: np.ndarray, names: Sequence[str],
              base: Dict[str, float]) -> Dict[str, float]:
    p = dict(base)
    for i, n in enumerate(names):
        s = SPEC_BY_NAME[n]
        u = float(np.clip(z[i], 0.0, _NORM_HI)) / _NORM_HI
        if s.log:
            p[n] = float(np.exp(np.log(s.lo) + u * (np.log(s.hi) - np.log(s.lo))))
        else:
            p[n] = float(s.lo + u * (s.hi - s.lo))
    return p


def default_params() -> Dict[str, float]:
    return {n: s.default for n, s in SPEC_BY_NAME.items()}


# ---------------------------------------------------------------------------
@dataclass
class CalibResult:
    vid: object
    vclass: str
    params: Dict[str, float]
    fitness: float
    n_evals: int
    n_steps: int
    duration_s: float
    converged: bool
    components: Dict[str, float]

    def flat(self) -> Dict[str, object]:
        row = {"vid": self.vid, "vclass": self.vclass, "fitness": self.fitness,
               "n_evals": self.n_evals, "n_steps": self.n_steps,
               "duration_s": self.duration_s, "converged": self.converged}
        row.update(self.params)
        row.update({f"gof_{k}": v for k, v in self.components.items()})
        return row


# ---------------------------------------------------------------------------
def make_eval(win: VehicleWindow, names: Sequence[str], base: Dict[str, float],
              road: RoadConfig, ocfg: ObjectiveConfig):
    """Normalised vector -> objective value, through the JIT core."""
    from .objective import make_fast_gof
    from .fastcore import pack
    fast = make_fast_gof(win, road, ocfg)

    def ev(z: np.ndarray) -> float:
        return fast(pack(from_norm(np.asarray(z, dtype=float), names, base)))

    return ev


def _run_cma(win: VehicleWindow, names: Sequence[str], base: Dict[str, float],
             road: RoadConfig, ocfg: ObjectiveConfig, ccfg: CMAConfig,
             x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, int]:
    import cma

    if x0 is None:
        x0 = to_norm(base, names)

    counter = {"n": 0}
    ev = make_eval(win, names, base, road, ocfg)

    def f(z):
        counter["n"] += 1
        return ev(z)

    opts = {
        "bounds": [0.0, _NORM_HI],
        "maxfevals": ccfg.maxfevals,
        "tolfun": ccfg.tolfun,
        "seed": ccfg.seed,
        "verbose": -9,
        "verb_log": 0,
        "verb_disp": 0,
    }
    if ccfg.popsize:
        opts["popsize"] = ccfg.popsize

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        es = cma.CMAEvolutionStrategy(list(x0), ccfg.sigma0, opts)
        es.optimize(f)
        z_best = np.asarray(es.result.xbest, dtype=float)
        f_best = float(es.result.fbest)

    return z_best, f_best, counter["n"]


def _polish(win: VehicleWindow, names: Sequence[str], base: Dict[str, float],
            z0: np.ndarray, road: RoadConfig, ocfg: ObjectiveConfig,
            ccfg: CMAConfig) -> Tuple[np.ndarray, float, int]:
    from scipy.optimize import minimize

    counter = {"n": 0}
    ev = make_eval(win, names, base, road, ocfg)

    def f(z):
        counter["n"] += 1
        return ev(z)

    res = minimize(f, z0, method="Nelder-Mead",
                   options={"maxiter": ccfg.polish_maxiter,
                            "xatol": 1e-3, "fatol": 1e-5, "disp": False})
    z = np.clip(np.asarray(res.x, dtype=float), 0.0, _NORM_HI)
    return z, float(f(z)), counter["n"]


def calibrate_vehicle(win: VehicleWindow, pcfg: PipelineConfig
                      ) -> CalibResult:
    """Calibrate one vehicle. Safe to call in a worker process."""
    t_start = time.time()
    road, ocfg, ccfg = pcfg.road, pcfg.objective, pcfg.cma

    base = default_params()
    # warm start v0 from the data -- much better than a global default
    base["v0"] = float(np.clip(free_flow_speed_estimate(win),
                               SPEC_BY_NAME["v0"].lo, SPEC_BY_NAME["v0"].hi))

    n_evals = 0
    try:
        if pcfg.staged:
            stage1 = [n for n in STAGE_LONGITUDINAL if n in pcfg.free_params]
            stage2 = [n for n in STAGE_LATERAL if n in pcfg.free_params]
            z1, _f1, e1 = _run_cma(win, stage1, base, road, ocfg, ccfg)
            base = from_norm(z1, stage1, base)
            n_evals += e1
            z2, _f2, e2 = _run_cma(win, stage2, base, road, ocfg, ccfg)
            base = from_norm(z2, stage2, base)
            n_evals += e2
            names = list(pcfg.free_params)   # joint refinement
            z, fit, e3 = _run_cma(win, names, base, road, ocfg, ccfg,
                                  x0=to_norm(base, names))
            n_evals += e3
        else:
            names = list(pcfg.free_params)
            z, fit, e = _run_cma(win, names, base, road, ocfg, ccfg)
            n_evals += e

        for r in range(ccfg.n_restarts):
            rng = np.random.default_rng(ccfg.seed + 100 * (r + 1))
            x0 = np.clip(z + rng.normal(0, 2.0, size=len(names)), 0, _NORM_HI)
            z_r, f_r, e_r = _run_cma(win, names, base, road, ocfg, ccfg, x0=x0)
            n_evals += e_r
            if f_r < fit:
                z, fit = z_r, f_r

        if ccfg.polish:
            z_p, f_p, e_p = _polish(win, names, base, z, road, ocfg, ccfg)
            n_evals += e_p
            if f_p < fit:
                z, fit = z_p, f_p

        params = from_norm(z, names, base)
        comps = gof_components(win, params, road, ocfg)
        converged = fit < ocfg.penalty

    except Exception as exc:                       # keep the pool alive
        warnings.warn(f"vehicle {win.vid} failed: {exc!r}")
        params, fit, comps, converged = default_params(), float("inf"), {}, False

    return CalibResult(
        vid=win.vid, vclass=win.vclass, params=params, fitness=fit,
        n_evals=n_evals, n_steps=win.n_steps,
        duration_s=time.time() - t_start, converged=converged,
        components=comps,
    )


# ---------------------------------------------------------------------------
_WORKER_CFG: Optional[PipelineConfig] = None


def _init_worker(cfg: PipelineConfig) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg
    # each worker is single-threaded; parallelism is across vehicles
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _work(win: VehicleWindow) -> CalibResult:
    return calibrate_vehicle(win, _WORKER_CFG)


def calibrate_all(windows: List[VehicleWindow], pcfg: PipelineConfig,
                  verbose: bool = True) -> List[CalibResult]:
    """Calibrate every window, in parallel across processes."""
    n_workers = pcfg.n_workers or (os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(windows)))

    if n_workers == 1:
        results = []
        for i, w in enumerate(windows):
            results.append(calibrate_vehicle(w, pcfg))
            if verbose and (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(windows)}", flush=True)
        return results

    from concurrent.futures import ProcessPoolExecutor, as_completed

    results: List[CalibResult] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker,
                             initargs=(pcfg,)) as ex:
        futures = {ex.submit(_work, w): w.vid for w in windows}
        for i, fut in enumerate(as_completed(futures)):
            results.append(fut.result())
            if verbose and (i + 1) % 25 == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (len(windows) - i - 1) / max(rate, 1e-9)
                print(f"  {i + 1}/{len(windows)}  "
                      f"{rate:.1f} veh/s  ETA {eta / 60:.1f} min", flush=True)
    return results
