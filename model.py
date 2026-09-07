"""
Vectorised NumPy port of the Intelligent-Agent Model.

Ported from github.com/MTGermany/mixedTraffic (GPL-3.0), js/models.js and
js/road.js, cross-checked against Treiber & Chaudhari, arXiv:2310.16816.

Where code and paper disagree, the CODE wins. Documented divergences:

1. s_x is clamped at zero: sx = max(0, dx - L_j). The paper says negative gaps
   must yield maximum deceleration; the code achieves that instead via the
   IDM's max(s, 0.1*s0) denominator plus the -bmax floor.
2. alpha(s_y) = min(exp(-s_y/s0y), 1) with SIGNED s_y, not |s_y| as printed in
   Eq. (5). With overlap (s_y < 0) the exponential exceeds 1 and is clamped, so
   alpha = 1. Using |s_y| as literally written would wrongly attenuate
   overlapping vehicles.
3. Parallel driving (dx < L_j and s_y > 0) multiplies the longitudinal force by
   longParReductFactor = 0.2 rather than setting it to zero as the paper says.
4. Two attenuation widths (s0y longitudinal, s0yLat lateral), not one.
5. Two anisotropy factors (pushLong, pushLat), not one lambda.
6. Lateral interaction is clamped to +/- 4*b.
7. Neighbours are selected by an interaction-strength threshold (bIntCrit),
   not by a geometric rule.

All functions take arrays over the neighbour axis K and return scalars for the
ego vehicle. `valid` is a boolean mask marking real neighbours in the padded
K-axis.
"""

from typing import Dict, Tuple

import numpy as np

from .config import MODEL_CONSTANTS as C
from .config import RoadConfig


# ---------------------------------------------------------------------------
# IDM
# ---------------------------------------------------------------------------
def idm_acc_free(v: float, v0: float, a: float) -> float:
    """IDM free-flow acceleration. Linear branch above v0 keeps it bounded."""
    v0eff = max(0.01, v0)
    if v < v0eff:
        return a * (1.0 - (v / v0eff) ** 4)
    return a * (1.0 - v / v0eff)


def idm_acc_int(s: np.ndarray, v: float, vl: np.ndarray,
                p: Dict[str, float]) -> np.ndarray:
    """
    IDM interaction (braking) term, vectorised over neighbours.
    Always <= 0. Returns array shaped like `s`.
    """
    a, b, s0, T = p["a"], p["b"], p["s0"], p["T"]
    sstar = s0 + np.maximum(0.0, v * T + 0.5 * v * (v - vl) / np.sqrt(a * b))
    acc = -a * (sstar / np.maximum(s, 0.1 * s0)) ** 2
    return np.maximum(-C.bmax, acc)


# ---------------------------------------------------------------------------
# IAM interaction terms
# ---------------------------------------------------------------------------
def _long_int(dx: np.ndarray, dy: np.ndarray, vx: float, vxl: np.ndarray,
              Ll: np.ndarray, Wavg: np.ndarray,
              p: Dict[str, float]) -> np.ndarray:
    """MTM.calcAccLongInt -- longitudinal force from each neighbour."""
    sx = np.maximum(0.0, dx - Ll)
    sy = np.abs(dy) - Wavg
    acc_cf = idm_acc_int(sx, vx, vxl, p)
    alpha = np.minimum(np.exp(-sy / p["s0y"]), 1.0)
    parallel = (dx < Ll) & (sy > 0.0)
    alpha = np.where(parallel, alpha * C.longParReductFactor, alpha)
    return alpha * acc_cf


def _long_select(dx: np.ndarray, dy: np.ndarray, vx: float, vxl: np.ndarray,
                 Ll: np.ndarray, Wavg: np.ndarray,
                 p: Dict[str, float]) -> np.ndarray:
    """
    MTM.calcAccLongLeaderSelect -- simplified force used only to decide which
    neighbours count. Uses max(s0y, s0yLat) and ignores the parallel reduction.
    """
    sx = np.maximum(0.0, dx - Ll)
    sy = np.abs(dy) - Wavg
    acc_cf = idm_acc_int(sx, vx, vxl, p)
    s0y_big = max(p["s0y"], p["s0yLat"])
    alpha = np.minimum(np.exp(-sy / s0y_big), 1.0)
    return alpha * acc_cf


def _lat_int(dx: np.ndarray, dy: np.ndarray, vx: float, vxl: np.ndarray,
             vy: float, vyl: np.ndarray, Ll: np.ndarray, Wavg: np.ndarray,
             p: Dict[str, float], is_pushing: bool) -> np.ndarray:
    """
    MTM.calcAccLatInt -- lateral force from each neighbour.

    dy > 0 means the neighbour is to one side; alpha carries the sign so the
    ego is pushed away. The `is_pushing` branch zeroes the force when the
    lateral gap to a follower is already opening.

    The boundary-proximity override in the JS (alpha forced to +/-1 when the
    leader hugs an edge) is applied in accelerations() where road geometry is
    available.
    """
    sx = np.maximum(0.0, dx - Ll)
    acc_cf = idm_acc_int(sx, vx, vxl, p)

    sign_dy = np.where(dy < 0.0, -1.0, 1.0)
    overlap = np.abs(dy) < Wavg
    alpha = -sign_dy * np.where(
        overlap,
        np.abs(dy) / np.maximum(Wavg, 1e-9),
        np.exp(-(np.abs(dy) - Wavg) / p["s0yLat"]),
    )

    v0_lat = -p["sensLat"] * alpha * acc_cf
    acc_lat = v0_lat / p["tauLatOVM"]

    if is_pushing:
        # zero the push when (y_l - y)*(vy_l - vy) > 0: gap already opening
        acc_lat = np.where(dy * (vyl - vy) > 0.0, 0.0, acc_lat)

    lim = C.accLatIntMaxFactor * p["b"]
    return np.clip(acc_lat, -lim, lim)


# ---------------------------------------------------------------------------
# Boundary forces
# ---------------------------------------------------------------------------
def _boundary_acc(y: float, vx: float, W_ego: float, p: Dict[str, float],
                  road: RoadConfig) -> Tuple[float, float]:
    """
    Simplified MTM.calcAccB: non-anticipative (j = 0 term only).

    The JS discretises an anticipation horizon over nj = 8 steps; its own
    comments note the complications were largely deactivated because results
    were no worse without them. For calibration on straight-road data the j = 0
    term is the operative part and is far cheaper.
    """
    if not road.enabled:
        return 0.0, 0.0

    sy_left = (road.y_left - y) - 0.5 * W_ego
    sy_right = (y - road.y_right) - 0.5 * W_ego

    def alpha_long(sy: float) -> float:
        return np.exp(-sy / p["s0yB"]) if sy > 0 else 1.0

    def alpha_lat(sy: float) -> float:
        return np.exp(-sy / p["s0yLatB"]) if sy > 0 else 1.0 - sy / p["s0yLatB"]

    aL_left, aL_right = alpha_long(sy_left), alpha_long(sy_right)
    aT_left, aT_right = alpha_lat(sy_left), alpha_lat(sy_right)

    acc_long_b = p["accLongBRef"] * (-aL_left - aL_right) * (vx / C.v0max)
    # positive lateral accel pushes toward +y (left); right edge pushes left
    acc_lat_b = p["accLatBRef"] * (aT_right - aT_left)
    acc_lat_b *= (0.2 + 0.8 * vx / C.v0max)
    acc_lat_b = float(np.clip(acc_lat_b, -C.accLatBMax, C.accLatBMax))
    return float(acc_long_b), acc_lat_b


# ---------------------------------------------------------------------------
# Full acceleration
# ---------------------------------------------------------------------------
def accelerations(x: float, y: float, vx: float, vy: float,
                  nb_x: np.ndarray, nb_y: np.ndarray,
                  nb_vx: np.ndarray, nb_vy: np.ndarray,
                  nb_L: np.ndarray, nb_W: np.ndarray,
                  valid: np.ndarray,
                  L_ego: float, W_ego: float,
                  p: Dict[str, float],
                  road: RoadConfig) -> Tuple[float, float]:
    """
    Total (accLong, accLat) for the ego vehicle at one instant.

    Replicates road.calcAccelerationsOfVehicle:
        accLong = accLongFree + [min over leaders] + [max over followers]
        accLat  = -vy/tau     + sum over leaders - pushLat * sum over followers
                  + boundary

    x, y are the ego's SIMULATED state; nb_* are the neighbours' MEASURED
    states, replayed from data.

    Positions are front-bumper longitudinal and lateral centre, matching the
    JS convention (u = front).
    """
    acc_long_free = idm_acc_free(vx, p["v0"], p["a"])
    acc_lat_free = -vy / p["tauLatOVM"]

    if not np.any(valid):
        bl, bt = _boundary_acc(y, vx, W_ego, p, road)
        return acc_long_free + bl, acc_lat_free + bt

    dx = nb_x - x
    dy = nb_y - y
    Wavg = 0.5 * (W_ego + nb_W)

    # -- classify neighbours -------------------------------------------------
    ahead = valid & (dx >= 0.0)
    behind = valid & (dx < 0.0)

    acc_sel_lead = _long_select(dx, dy, vx, nb_vx, nb_L, Wavg, p)
    is_leader = ahead & (acc_sel_lead < -C.bIntCrit)

    # follower push: force THEY would feel from the ego, so roles swap
    dx_b = -dx
    acc_sel_fol = _long_select(dx_b, -dy, nb_vx, np.full_like(nb_vx, vx),
                               np.full_like(nb_L, L_ego), Wavg, p)
    gap_behind = (x - L_ego) - nb_x
    is_follower = behind & (acc_sel_fol < -C.bIntCrit) & (gap_behind < C.smaxfollow)

    # -- longitudinal --------------------------------------------------------
    acc_long_int = 0.0
    if np.any(is_leader):
        f_lead = _long_int(dx, dy, vx, nb_vx, nb_L, Wavg, p)
        if C.useSeveralLeaders:
            acc_long_int = float(np.sum(np.where(is_leader, f_lead, 0.0)))
        else:
            acc_long_int = float(min(0.0, np.min(np.where(is_leader, f_lead, 0.0))))

    if np.any(is_follower):
        # f_push_ij = -pushLong * f_int_ji  (Eq. 6); take max over followers
        f_ji = _long_int(dx_b, -dy, nb_vx, np.full_like(nb_vx, vx),
                         np.full_like(nb_L, L_ego), Wavg, p)
        push = -p["pushLong"] * f_ji
        acc_long_int += float(max(0.0, np.max(np.where(is_follower, push, 0.0))))

    # -- lateral (no shielding: sum over all, per Sect. 2.2.2) ---------------
    acc_lat_int = 0.0
    if np.any(is_leader):
        g_lead = _lat_int(dx, dy, vx, nb_vx, vy, nb_vy, nb_L, Wavg, p,
                          is_pushing=False)
        acc_lat_int += float(np.sum(np.where(is_leader, g_lead, 0.0)))

    if np.any(is_follower):
        # The FOLLOWER is the subject here, so its own speeds go in the
        # subject slots and the ego's in the leader slots. Getting this
        # backwards silently corrupts only the lateral push term, which is
        # exactly the term that is hardest to notice by eye.
        g_fol = _lat_int(dx_b, -dy, nb_vx, np.full_like(nb_vx, vx),
                         nb_vy, np.full_like(nb_vy, vy),
                         np.full_like(nb_L, L_ego), Wavg, p, is_pushing=True)
        acc_lat_int -= p["pushLat"] * float(np.sum(np.where(is_follower, g_fol, 0.0)))

    bl, bt = _boundary_acc(y, vx, W_ego, p, road)

    return (acc_long_free + acc_long_int + bl,
            acc_lat_free + acc_lat_int + bt)
