"""
Simulation and goodness-of-fit.

Global (trajectory-integrated) calibration: the ego vehicle's ODE is integrated
forward from its measured initial state while all neighbours are replayed from
data. This is the criterion the model is actually for. One-step-ahead fitting
is available via `local_objective` as a cheap warm start.

Integration uses the ballistic update, as in the reference implementation --
RK4 buys nothing here because the right-hand side is not smooth (the min/max in
Eq. 7, the kink in min(1, exp(.)) in Eq. 5, and discrete switching of which
neighbour is selected).
"""

from typing import Dict, Tuple

import numpy as np

from .config import ObjectiveConfig, RoadConfig
from .data import VehicleWindow
from .model import accelerations, idm_acc_free


def simulate(win: VehicleWindow, p: Dict[str, float], road: RoadConfig,
             ocfg: ObjectiveConfig) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray, int]:
    """
    Integrate the ego trajectory. Returns (x, y, vx, vy, n_valid_steps).

    Terminates early if the ego diverges past the configured error bounds;
    `n_valid_steps` reports how far it got so the objective can penalise.
    """
    T, dt = win.n_steps, win.dt
    x = np.empty(T); y = np.empty(T); vx = np.empty(T); vy = np.empty(T)
    x[0], y[0], vx[0], vy[0] = win.x[0], win.y[0], win.vx[0], win.vy[0]
    reset_every = int(round(ocfg.reset_s / dt)) if ocfg.reset_s > 0 else 0

    for i in range(T - 1):
        if reset_every and i > 0 and i % reset_every == 0:
            x[i], y[i] = win.x[i], win.y[i]
            vx[i], vy[i] = win.vx[i], win.vy[i]
        ax, ay = accelerations(
            x[i], y[i], vx[i], vy[i],
            win.nb_x[i], win.nb_y[i], win.nb_vx[i], win.nb_vy[i],
            win.nb_L[i], win.nb_W[i], win.valid[i],
            win.L, win.W, p, road,
        )
        # ballistic update: second order in position, first order in velocity
        vx_new = vx[i] + ax * dt
        vy_new = vy[i] + ay * dt
        x[i + 1] = x[i] + vx[i] * dt + 0.5 * ax * dt * dt
        y[i + 1] = y[i] + vy[i] * dt + 0.5 * ay * dt * dt
        # no reversing: vehicles are directed flow
        vx[i + 1] = max(0.0, vx_new)
        vy[i + 1] = vy_new

        if (abs(x[i + 1] - win.x[i + 1]) > ocfg.max_x_error or
                abs(y[i + 1] - win.y[i + 1]) > ocfg.max_y_error or
                not np.isfinite(x[i + 1] + y[i + 1])):
            return x, y, vx, vy, i + 2

    return x, y, vx, vy, T


def gof(win: VehicleWindow, p: Dict[str, float], road: RoadConfig,
        ocfg: ObjectiveConfig) -> float:
    """
    Dimensionless mixed goodness-of-fit. Lower is better.

        F = w_x * RMSE(x) / RMS(distance travelled)
          + w_y * RMSE(y) / y_scale
          + w_v * RMSE(vx) / RMS(vx_obs)

    Longitudinal terms are relative (Treiber & Kesting report relative measures
    give more stable, more transferable CF parameters). The lateral term is
    absolute because y passes through zero, so a relative measure would be
    unbounded. y_scale sets the exchange rate between lateral metres and
    longitudinal relative error -- it is the single most consequential knob in
    the whole objective, so change it deliberately and report it.
    """
    x, y, vx, _vy, n = simulate(win, p, road, ocfg)

    if n < win.n_steps:
        # partial credit + penalty proportional to how early it diverged
        return ocfg.penalty * (1.0 + (win.n_steps - n) / win.n_steps)

    v_scale = max(np.sqrt(np.mean(win.vx ** 2)), 0.5)
    x_scale = ocfg.x_scale

    e_x = np.sqrt(np.mean((x - win.x) ** 2)) / x_scale
    e_y = np.sqrt(np.mean((y - win.y) ** 2)) / ocfg.y_scale
    e_v = np.sqrt(np.mean((vx - win.vx) ** 2)) / v_scale

    f = ocfg.w_x * e_x + ocfg.w_y * e_y + ocfg.w_v * e_v
    return float(f) if np.isfinite(f) else ocfg.penalty


def gof_components(win: VehicleWindow, p: Dict[str, float], road: RoadConfig,
                   ocfg: ObjectiveConfig) -> Dict[str, float]:
    """Same as gof() but returns the individual terms, for diagnostics."""
    x, y, vx, _vy, n = simulate(win, p, road, ocfg)
    if n < win.n_steps:
        return {"total": ocfg.penalty, "e_x": np.nan, "e_y": np.nan,
                "e_v": np.nan, "diverged_at_s": n * win.dt}
    v_scale = max(np.sqrt(np.mean(win.vx ** 2)), 0.5)
    e_x = np.sqrt(np.mean((x - win.x) ** 2)) / ocfg.x_scale
    e_y = np.sqrt(np.mean((y - win.y) ** 2)) / ocfg.y_scale
    e_v = np.sqrt(np.mean((vx - win.vx) ** 2)) / v_scale
    return {
        "total": ocfg.w_x * e_x + ocfg.w_y * e_y + ocfg.w_v * e_v,
        "e_x": e_x, "e_y": e_y, "e_v": e_v,
        "rmse_x_m": float(np.sqrt(np.mean((x - win.x) ** 2))),
        "rmse_y_m": float(np.sqrt(np.mean((y - win.y) ** 2))),
        "rmse_vx_ms": float(np.sqrt(np.mean((vx - win.vx) ** 2))),
        "diverged_at_s": np.nan,
    }


def local_objective(win: VehicleWindow, p: Dict[str, float], road: RoadConfig
                    ) -> float:
    """
    One-step-ahead acceleration error, using measured ego states throughout.

    Cheap, smooth, no error accumulation -- useful as a warm start for the
    global fit. Do not use it as the final criterion: it rewards models that
    are wrong in ways only visible dynamically, and it inherits the noise of
    differentiated position data.
    """
    T = win.n_steps
    ax_obs = np.gradient(win.vx, win.dt)
    ay_obs = np.gradient(win.vy, win.dt)
    ax_sim = np.empty(T); ay_sim = np.empty(T)
    for i in range(T):
        ax_sim[i], ay_sim[i] = accelerations(
            win.x[i], win.y[i], win.vx[i], win.vy[i],
            win.nb_x[i], win.nb_y[i], win.nb_vx[i], win.nb_vy[i],
            win.nb_L[i], win.nb_W[i], win.valid[i],
            win.L, win.W, p, road,
        )
    return float(np.sqrt(np.mean((ax_sim - ax_obs) ** 2)) +
                 np.sqrt(np.mean((ay_sim - ay_obs) ** 2)))


def make_fast_gof(win: VehicleWindow, road: RoadConfig, ocfg: ObjectiveConfig):
    """
    Build a closure evaluating the objective from a packed parameter VECTOR
    via the JIT core. This is what the optimizer calls; `gof` above stays as
    the readable reference implementation.

    Arrays are hoisted out of the closure so each evaluation allocates nothing
    at Python level.
    """
    from .fastcore import gof_core
    from .config import MODEL_CONSTANTS as C

    ox = np.ascontiguousarray(win.x, dtype=np.float64)
    oy = np.ascontiguousarray(win.y, dtype=np.float64)
    ovx = np.ascontiguousarray(win.vx, dtype=np.float64)
    ovy = np.ascontiguousarray(win.vy, dtype=np.float64)
    nx = np.ascontiguousarray(win.nb_x, dtype=np.float64)
    ny = np.ascontiguousarray(win.nb_y, dtype=np.float64)
    nvx = np.ascontiguousarray(win.nb_vx, dtype=np.float64)
    nvy = np.ascontiguousarray(win.nb_vy, dtype=np.float64)
    nL = np.ascontiguousarray(win.nb_L, dtype=np.float64)
    nW = np.ascontiguousarray(win.nb_W, dtype=np.float64)
    vd = np.ascontiguousarray(win.valid)

    road_on = 1 if road.enabled else 0
    reset_every = int(round(ocfg.reset_s / win.dt)) if ocfg.reset_s > 0 else 0

    def f(pvec: np.ndarray) -> float:
        return float(gof_core(
            ox, oy, ovx, ovy, nx, ny, nvx, nvy, nL, nW, vd,
            win.L, win.W, np.ascontiguousarray(pvec, dtype=np.float64), win.dt,
            C.bmax, C.longParReductFactor, C.bIntCrit, C.smaxfollow,
            C.accLatIntMaxFactor, C.useSeveralLeaders, C.accLatBMax, C.v0max,
            road_on, road.y_left, road.y_right,
            ocfg.max_x_error, ocfg.max_y_error, reset_every, ocfg.penalty,
            ocfg.w_x, ocfg.w_y, ocfg.w_v, ocfg.x_scale, ocfg.y_scale))

    return f


def free_flow_speed_estimate(win: VehicleWindow, quantile: float = 0.95
                             ) -> float:
    """
    Data-driven starting value for v0: a high quantile of observed speed,
    restricted to steps with no interacting neighbour ahead. Warm-starting v0
    from the data rather than a global default markedly speeds up CMA-ES.
    """
    ahead = win.valid & (win.nb_x - win.x[:, None] > 0) \
                      & (np.abs(win.nb_y - win.y[:, None]) < 2.0) \
                      & (win.nb_x - win.x[:, None] < 30.0)
    free = ~ahead.any(axis=1)
    v = win.vx[free] if free.sum() > 20 else win.vx
    return float(np.quantile(v, quantile))
