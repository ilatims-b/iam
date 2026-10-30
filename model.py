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
8. The underlying CF model is ACC (IDM blended with a Constant-Acceleration
   Heuristic that uses the leader's acceleration a_l), NOT plain IDM -- see
   acc_int(). The paper says IDM; models.js uses ACC.
9. Lateral interaction has a "leader too close to a road boundary" override
   (alpha forced to +/-1), models.js calcAccLatInt. Not in the paper.
10. A constant lateral bias acceleration per vehicle type (accBiasLat),
   road.js step (6). Not in the paper.

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
def idm_acc_free(v: float, v0: float, a: float, delta: float = 4.0) -> float:
    """IDM free-flow acceleration. Linear branch above v0 keeps it bounded."""
    v0eff = max(0.01, v0)
    if v < v0eff:
        return a * (1.0 - (v / v0eff) ** delta)
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


def _tanh(x):
    """models.js myTanh (saturates at +/-50; numpy tanh agrees to <1e-15)."""
    return np.tanh(np.clip(x, -50.0, 50.0))


def acc_free_arr(v, p):
    """IDM free acceleration, vectorised over v."""
    v0e = max(0.01, p["v0"])
    v = np.asarray(v, dtype=float)
    return np.where(v < v0e, p["a"] * (1.0 - (v / v0e) ** p.get("delta", 4.0)),
                    p["a"] * (1.0 - v / v0e))


def acc_int(s, v, vl, al, p) -> np.ndarray:
    """
    ACC interaction term, models.js ACC.calcAccInt. Always <= 0 after the
    bmax floor. Vectorised over s, v, vl, al (any may be an array).

        IDM  : a_free + a_int_IDM(s, v, vl)               (no leader accel.)
        CAH  : constant-acceleration heuristic using a_l
        mix  : IDM if IDM > CAH else CAH + b*tanh((IDM-CAH)/b)
        ACC  : 0.99*mix + 0.01*IDM       (the 1% IDM keeps string stability)
        out  : max(-bmax, ACC - a_free)  (caller adds a_free back)

    Why CAH: a vehicle cutting in makes the gap collapse, so IDM brakes hard,
    but the cutter has a_l ~ 0 and the gap will not shrink further; CAH sees
    that and the blend softens IDM's reaction.
    """
    a, b, s0, T = p["a"], p["b"], p["s0"], p["T"]
    s = np.asarray(s, dtype=float)
    vl = np.asarray(vl, dtype=float)
    al = np.asarray(al, dtype=float)
    v = np.asarray(v, dtype=float)

    sstar = s0 + np.maximum(0.0, v * T + 0.5 * v * (v - vl) / np.sqrt(a * b))
    acc_free = acc_free_arr(v, p)
    acc_idm = acc_free - a * (sstar / np.maximum(s, 0.1 * s0)) ** 2

    cond = vl * (v - vl) < -2.0 * s * al
    den = vl * vl - 2.0 * s * al
    safe_den = np.where(cond & (np.abs(den) > 1e-12), den, 1.0)
    cah_a = v * v * al / safe_den
    cah_b = al - (v - vl) ** 2 / (2.0 * np.maximum(s, 0.1)) * (v > vl)
    acc_cah = np.minimum(np.where(cond & (np.abs(den) > 1e-12), cah_a, cah_b),
                         a)

    acc_mix = np.where(acc_idm > acc_cah, acc_idm,
                       acc_cah + b * _tanh((acc_idm - acc_cah) / b))
    acc_acc = C.cool * acc_mix + (1.0 - C.cool) * acc_idm
    return np.maximum(-C.bmax, acc_acc - acc_free)


# ---------------------------------------------------------------------------
# IAM interaction terms
# ---------------------------------------------------------------------------
def _long_int(dx: np.ndarray, dy: np.ndarray, vx, vxl: np.ndarray,
              axl: np.ndarray, Ll: np.ndarray, Wavg: np.ndarray,
              p: Dict[str, float]) -> np.ndarray:
    """MTM.calcAccLongInt -- longitudinal force from each neighbour."""
    sx = np.maximum(0.0, dx - Ll)
    sy = np.abs(dy) - Wavg
    acc_cf = acc_int(sx, vx, vxl, axl, p)
    alpha = np.minimum(np.exp(-sy / p["s0y"]), 1.0)
    parallel = (dx < Ll) & (sy > 0.0)
    alpha = np.where(parallel, alpha * C.longParReductFactor, alpha)
    return alpha * acc_cf


def _long_select(dx: np.ndarray, dy: np.ndarray, vx, vxl: np.ndarray,
                 axl: np.ndarray, Ll: np.ndarray, Wavg: np.ndarray,
                 p: Dict[str, float]) -> np.ndarray:
    """
    MTM.calcAccLongLeaderSelect -- simplified force used only to decide which
    neighbours count. Uses max(s0y, s0yLat) and ignores the parallel reduction.
    """
    sx = np.maximum(0.0, dx - Ll)
    sy = np.abs(dy) - Wavg
    acc_cf = acc_int(sx, vx, vxl, axl, p)
    s0y_big = max(p["s0y"], p["s0yLat"])
    alpha = np.minimum(np.exp(-sy / s0y_big), 1.0)
    return alpha * acc_cf


def _lat_int(dx: np.ndarray, dy: np.ndarray, vx, vxl: np.ndarray,
             vy, vyl: np.ndarray, axl: np.ndarray, Ll: np.ndarray,
             Wavg: np.ndarray, p: Dict[str, float], is_pushing: bool,
             y_sub, y_lead, W_sub, W_lead, road: RoadConfig) -> np.ndarray:
    """
    MTM.calcAccLatInt -- lateral force from each neighbour.

    dy > 0 means the neighbour is to one side; alpha carries the sign so the
    ego is pushed away. The `is_pushing` branch zeroes the force when the
    lateral gap to a follower is already opening.

    "Subject" = the vehicle being pushed (ego for leaders, the follower for
    pushes); "lead" = the other one, with dy = y_lead - y_sub.
    """
    sx = np.maximum(0.0, dx - Ll)
    acc_cf = acc_int(sx, vx, vxl, axl, p)

    sign_dy = np.where(dy < 0.0, -1.0, 1.0)
    overlap = np.abs(dy) < Wavg
    alpha = -sign_dy * np.where(
        overlap,
        np.abs(dy) / np.maximum(Wavg, 1e-9),
        np.exp(-(np.abs(dy) - Wavg) / p["s0yLat"]),
    )

    # models.js calcAccLatInt: if the vehicles overlap laterally and the leader
    # sits too close to a road boundary for the subject to pass on that side,
    # force the lateral desire to the other side (stops vehicles being
    # trapped behind slow vehicles at the edge). Skipped if BOTH sides are
    # too narrow, and when no road is configured. Our frame: +y = left.
    if road.enabled and C.use_narrow:
        gap_right = (y_lead - road.y_right) - 0.5 * W_lead
        gap_left = (road.y_left - y_lead) - 0.5 * W_lead
        narrow_r = gap_right < W_sub + p["s0yLatB"]
        narrow_l = gap_left < W_sub + p["s0yLatB"]
        act = overlap & ~(narrow_r & narrow_l)
        alpha = np.where(act & narrow_r & (y_sub < y_lead), 1.0, alpha)
        alpha = np.where(act & narrow_l & (y_sub > y_lead), -1.0, alpha)

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
                  nb_vx: np.ndarray, nb_vy: np.ndarray, nb_ax: np.ndarray,
                  nb_L: np.ndarray, nb_W: np.ndarray,
                  valid: np.ndarray,
                  L_ego: float, W_ego: float,
                  p: Dict[str, float],
                  road: RoadConfig,
                  a_ego: float = 0.0) -> Tuple[float, float]:
    """
    Total (accLong, accLat) for the ego vehicle at one instant.

    Replicates road.calcAccelerationsOfVehicle:
        accLong = accLongFree + [min over leaders] + [max over followers]
        accLat  = -vy/tau     + sum over leaders - pushLat * sum over followers
                  + boundary + bias

    x, y are the ego's SIMULATED state; nb_* are the neighbours' MEASURED
    states, replayed from data. nb_ax[k] is neighbour k's acceleration (the
    a_l of ACC when k leads the ego). a_ego is the ego's own PREVIOUS-STEP
    SIMULATED acceleration, used as a_l when the ego leads a follower (the JS
    reads leadveh.accLong, which at that point is last step's value).

    Positions are front-bumper longitudinal and lateral centre, matching the
    JS convention (u = front).
    """
    acc_long_free = idm_acc_free(vx, p["v0"], p["a"], p.get("delta", 4.0))
    acc_lat_free = -vy / p["tauLatOVM"]
    bias = p["accBiasLat"]

    if not np.any(valid):
        bl, bt = _boundary_acc(y, vx, W_ego, p, road)
        return acc_long_free + bl, acc_lat_free + bt + bias

    dx = nb_x - x
    dy = nb_y - y
    Wavg = 0.5 * (W_ego + nb_W)
    ego_vx = np.full_like(nb_vx, vx)
    ego_vy = np.full_like(nb_vy, vy)
    ego_ax = np.full_like(nb_vx, a_ego)
    ego_L = np.full_like(nb_L, L_ego)

    # -- classify neighbours -------------------------------------------------
    ahead = valid & (dx >= 0.0)
    behind = valid & (dx < 0.0)

    acc_sel_lead = _long_select(dx, dy, vx, nb_vx, nb_ax, nb_L, Wavg, p)
    is_leader = ahead & (acc_sel_lead < -C.bIntCrit)

    # follower push: force THEY would feel from the ego, so roles swap and
    # the ego's (simulated) acceleration is the follower's a_l
    dx_b = -dx
    acc_sel_fol = _long_select(dx_b, -dy, nb_vx, ego_vx, ego_ax, ego_L,
                               Wavg, p)
    gap_behind = (x - L_ego) - nb_x
    is_follower = behind & (acc_sel_fol < -C.bIntCrit) & (gap_behind < C.smaxfollow)

    # -- longitudinal --------------------------------------------------------
    acc_long_int = 0.0
    if np.any(is_leader):
        f_lead = _long_int(dx, dy, vx, nb_vx, nb_ax, nb_L, Wavg, p)
        if C.useSeveralLeaders:
            acc_long_int = float(np.sum(np.where(is_leader, f_lead, 0.0)))
        else:
            acc_long_int = float(min(0.0, np.min(np.where(is_leader, f_lead, 0.0))))

    if np.any(is_follower):
        # f_push_ij = -pushLong * f_int_ji  (Eq. 6); take max over followers
        f_ji = _long_int(dx_b, -dy, nb_vx, ego_vx, ego_ax, ego_L, Wavg, p)
        push = -p["pushLong"] * f_ji
        acc_long_int += float(max(0.0, np.max(np.where(is_follower, push, 0.0))))

    # -- lateral (no shielding: sum over all, per Sect. 2.2.2) ---------------
    acc_lat_int = 0.0
    if np.any(is_leader):
        g_lead = _lat_int(dx, dy, vx, nb_vx, vy, nb_vy, nb_ax, nb_L, Wavg, p,
                          False, y, nb_y, W_ego, nb_W, road)
        acc_lat_int += float(np.sum(np.where(is_leader, g_lead, 0.0)))

    if np.any(is_follower):
        # The FOLLOWER is the subject here, so its own speeds go in the
        # subject slots and the ego's in the leader slots. Getting this
        # backwards silently corrupts only the lateral push term, which is
        # exactly the term that is hardest to notice by eye.
        g_fol = _lat_int(dx_b, -dy, nb_vx, ego_vx, nb_vy, ego_vy, ego_ax,
                         ego_L, Wavg, p, True, nb_y, y, nb_W, W_ego, road)
        acc_lat_int -= p["pushLat"] * float(np.sum(np.where(is_follower, g_fol, 0.0)))

    bl, bt = _boundary_acc(y, vx, W_ego, p, road)

    return (acc_long_free + acc_long_int + bl,
            acc_lat_free + acc_lat_int + bt + bias)
