"""
Numba-JIT simulation core.

Numerically identical to model.accelerations + objective.simulate, but with the
whole time loop inside one nopython function. The pure-Python versions remain
the readable reference and are checked against this one in tests/test_agreement.py
-- if you change one, change both and re-run that test.

Speedup is roughly 100x, which is the difference between a 15-hour run and a
10-minute one on ~800 vehicles.

Parameter vector layout (PARAM_ORDER, must match config.PARAM_SPECS order):
    0 v0   1 T   2 s0   3 a   4 b
    5 s0y  6 s0yLat  7 sensLat  8 tauLatOVM  9 pushLong  10 pushLat
   11 accLongBRef  12 accLatBRef  13 s0yB  14 s0yLatB  15 accBiasLat
   16 delta (IDM acceleration exponent)
"""

from typing import Dict, Sequence

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                    # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):                                 # type: ignore
        def deco(f):
            return f
        return deco if not a else a[0]


PARAM_ORDER = (
    "v0", "T", "s0", "a", "b",
    "s0y", "s0yLat", "sensLat", "tauLatOVM", "pushLong", "pushLat",
    "accLongBRef", "accLatBRef", "s0yB", "s0yLatB",
    "accBiasLat", "delta",
)
PARAM_INDEX = {n: i for i, n in enumerate(PARAM_ORDER)}


def pack(p: Dict[str, float]) -> np.ndarray:
    return np.array([float(p[n]) for n in PARAM_ORDER], dtype=np.float64)


def unpack(v: np.ndarray) -> Dict[str, float]:
    return {n: float(v[i]) for i, n in enumerate(PARAM_ORDER)}


# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def _idm_acc_int(s, v, vl, a, b, s0, T, bmax):
    sstar = s0 + max(0.0, v * T + 0.5 * v * (v - vl) / np.sqrt(a * b))
    den = s if s > 0.1 * s0 else 0.1 * s0
    acc = -a * (sstar / den) * (sstar / den)
    return acc if acc > -bmax else -bmax


@njit(cache=True, fastmath=False)
def _idm_acc_free(v, v0, a, delta=4.0):
    v0e = v0 if v0 > 0.01 else 0.01
    if v < v0e:
        r = v / v0e
        rd = r * r * r * r if delta == 4.0 else r ** delta
        return a * (1.0 - rd)
    return a * (1.0 - v / v0e)


@njit(cache=True, fastmath=False)
def _acc_int(s, v, vl, al, a, b, s0, T, v0, bmax, cool, delta=4.0):
    """models.js ACC.calcAccInt; identical to model.acc_int."""
    sstar = s0 + max(0.0, v * T + 0.5 * v * (v - vl) / np.sqrt(a * b))
    acc_free = _idm_acc_free(v, v0, a, delta)
    den_s = s if s > 0.1 * s0 else 0.1 * s0
    acc_idm = acc_free - a * (sstar / den_s) * (sstar / den_s)

    den = vl * vl - 2.0 * s * al
    if vl * (v - vl) < -2.0 * s * al and abs(den) > 1e-12:
        acc_cah = v * v * al / den
    else:
        sm = s if s > 0.1 else 0.1
        acc_cah = al - (v - vl) * (v - vl) / (2.0 * sm) * (1.0 if v > vl else 0.0)
    if acc_cah > a:
        acc_cah = a

    if acc_idm > acc_cah:
        acc_mix = acc_idm
    else:
        z = (acc_idm - acc_cah) / b
        z = 50.0 if z > 50.0 else (-50.0 if z < -50.0 else z)
        acc_mix = acc_cah + b * np.tanh(z)
    acc_acc = cool * acc_mix + (1.0 - cool) * acc_idm
    out = acc_acc - acc_free
    return out if out > -bmax else -bmax


@njit(cache=True, fastmath=False)
def _lat_alpha(dy, Wavg, s0yLat, y_sub, y_lead, W_sub, W_lead, s0yLatB,
               road_on, y_left, y_right):
    """Normalised lateral desire incl. models.js too-narrow-boundary override.
    dy = y_lead - y_sub; our frame: +y = left."""
    sgn = -1.0 if dy < 0.0 else 1.0
    overlap = abs(dy) < Wavg
    if overlap:
        alpha = -sgn * (abs(dy) / Wavg)
    else:
        alpha = -sgn * np.exp(-(abs(dy) - Wavg) / s0yLat)
    if overlap and road_on != 0:
        gap_right = (y_lead - y_right) - 0.5 * W_lead
        gap_left = (y_left - y_lead) - 0.5 * W_lead
        narrow_r = gap_right < W_sub + s0yLatB
        narrow_l = gap_left < W_sub + s0yLatB
        if not (narrow_r and narrow_l):
            if narrow_r and y_sub < y_lead:
                alpha = 1.0
            if narrow_l and y_sub > y_lead:
                alpha = -1.0
    return alpha


@njit(cache=True, fastmath=False)
def _boundary(y, vx, W_ego, p, road_on, y_left, y_right, accLatBMax, v0max):
    if road_on == 0:
        return 0.0, 0.0
    syL = (y_left - y) - 0.5 * W_ego
    syR = (y - y_right) - 0.5 * W_ego
    s0yB = p[13]
    s0yLatB = p[14]

    aLL = np.exp(-syL / s0yB) if syL > 0.0 else 1.0
    aLR = np.exp(-syR / s0yB) if syR > 0.0 else 1.0
    aTL = np.exp(-syL / s0yLatB) if syL > 0.0 else 1.0 - syL / s0yLatB
    aTR = np.exp(-syR / s0yLatB) if syR > 0.0 else 1.0 - syR / s0yLatB

    accLongB = p[11] * (-aLL - aLR) * (vx / v0max)
    accLatB = p[12] * (aTR - aTL) * (0.2 + 0.8 * vx / v0max)
    if accLatB > accLatBMax:
        accLatB = accLatBMax
    elif accLatB < -accLatBMax:
        accLatB = -accLatBMax
    return accLongB, accLatB


@njit(cache=True, fastmath=False)
def _accel(x, y, vx, vy, nb_x, nb_y, nb_vx, nb_vy, nb_ax, nb_L, nb_W, valid,
           a_ego, L_ego, W_ego, p,
           bmax, longPar, bIntCrit, smaxfollow, accLatIntMaxFactor,
           useSeveralLeaders, accLatBMax, v0max, cool, narrow_on,
           road_on, y_left, y_right):
    v0 = p[0]; T = p[1]; s0 = p[2]; a = p[3]; b = p[4]
    s0y = p[5]; s0yLat = p[6]; sensLat = p[7]; tau = p[8]
    pushLong = p[9]; pushLat = p[10]; s0yLatB = p[14]; bias = p[15]
    delta = p[16]

    acc_long = _idm_acc_free(vx, v0, a, delta)
    acc_lat = -vy / tau

    s0y_big = s0y if s0y > s0yLat else s0yLat
    lat_lim = accLatIntMaxFactor * b

    best_lead = 0.0     # min over leaders (Eq. 7 first term)
    sum_lead = 0.0      # used when useSeveralLeaders
    best_push = 0.0     # max over followers (Eq. 7 second term)
    lat_lead = 0.0
    lat_fol = 0.0

    K = nb_x.shape[0]
    for k in range(K):
        if not valid[k]:
            continue
        dx = nb_x[k] - x
        dy = nb_y[k] - y
        Wavg = 0.5 * (W_ego + nb_W[k])
        Ll = nb_L[k]
        vxl = nb_vx[k]
        vyl = nb_vy[k]
        axl = nb_ax[k]

        if dx >= 0.0:
            # ---- candidate leader ----
            sx = dx - Ll
            if sx < 0.0:
                sx = 0.0
            sy = abs(dy) - Wavg
            cf = _acc_int(sx, vx, vxl, axl, a, b, s0, T, v0, bmax, cool, delta)

            al_sel = np.exp(-sy / s0y_big)
            if al_sel > 1.0:
                al_sel = 1.0
            if al_sel * cf >= -bIntCrit:
                continue                       # not an interacting leader

            # longitudinal force (calcAccLongInt)
            al = np.exp(-sy / s0y)
            if al > 1.0:
                al = 1.0
            if dx < Ll and sy > 0.0:
                al = al * longPar
            f = al * cf
            if f < best_lead:
                best_lead = f
            sum_lead += f

            # lateral force (calcAccLatInt, isPushing = False)
            alpha = _lat_alpha(dy, Wavg, s0yLat, y, nb_y[k], W_ego, nb_W[k],
                               s0yLatB, narrow_on, y_left, y_right)
            g = (-sensLat * alpha * cf) / tau
            if g > lat_lim:
                g = lat_lim
            elif g < -lat_lim:
                g = -lat_lim
            lat_lead += g

        else:
            # ---- candidate follower: roles swap ----
            dxb = -dx
            gap_behind = (x - L_ego) - nb_x[k]
            if gap_behind >= smaxfollow:
                continue
            sxb = dxb - L_ego
            if sxb < 0.0:
                sxb = 0.0
            syb = abs(dy) - Wavg
            cfb = _acc_int(sxb, vxl, vx, a_ego, a, b, s0, T, v0, bmax, cool, delta)

            al_sel = np.exp(-syb / s0y_big)
            if al_sel > 1.0:
                al_sel = 1.0
            if al_sel * cfb >= -bIntCrit:
                continue

            alb = np.exp(-syb / s0y)
            if alb > 1.0:
                alb = 1.0
            if dxb < L_ego and syb > 0.0:
                alb = alb * longPar
            push = -pushLong * (alb * cfb)
            if push > best_push:
                best_push = push

            # lateral push (isPushing = True): zero if lateral gap opening
            dyb = -dy
            alpha = _lat_alpha(dyb, Wavg, s0yLat, nb_y[k], y, nb_W[k], W_ego,
                               s0yLatB, narrow_on, y_left, y_right)
            g = (-sensLat * alpha * cfb) / tau
            if dyb * (vy - vyl) > 0.0:
                g = 0.0
            if g > lat_lim:
                g = lat_lim
            elif g < -lat_lim:
                g = -lat_lim
            lat_fol += g

    if useSeveralLeaders:
        acc_long += sum_lead + best_push
    else:
        acc_long += best_lead + best_push
    acc_lat += lat_lead - pushLat * lat_fol

    bl, bt = _boundary(y, vx, W_ego, p, road_on, y_left, y_right,
                       accLatBMax, v0max)
    return acc_long + bl, acc_lat + bt + bias


@njit(cache=True, fastmath=False)
def simulate_core(obs_x, obs_y, obs_vx, obs_vy, obs_ax,
                  nb_x, nb_y, nb_vx, nb_vy, nb_ax, nb_L, nb_W, valid,
                  L_ego, W_ego, p, dt,
                  bmax, longPar, bIntCrit, smaxfollow, accLatIntMaxFactor,
                  useSeveralLeaders, accLatBMax, v0max, dvdumax, dotdvdumax,
                  cool, narrow_on, clamp_on,
                  road_on, y_left, y_right,
                  max_x_err, max_y_err, reset_every):
    """
    JS-style integration of the ego trajectory (see objective.simulate).
    Returns (x, y, vx, vy, n).

    `reset_every` > 0 re-initialises the ego state from the measurement every
    that many steps, splitting the window into independent sub-simulations.

    This matters more than it looks. Lane-free lateral position is only weakly
    determined: with the TRUE parameters and realistic acceleration noise, a
    free-running 15 s simulation still accumulates ~1.4 m of lateral error,
    because lateral motion random-walks. Left unbounded, that noise dominates
    the objective and the optimiser spends its budget chasing it. Resetting
    keeps the criterion dynamic (multi-step, not one-step-ahead) while
    capping accumulation.
    """
    T = obs_x.shape[0]
    x = np.empty(T); y = np.empty(T); vx = np.empty(T); vy = np.empty(T)
    x[0] = obs_x[0]; y[0] = obs_y[0]; vx[0] = obs_vx[0]; vy[0] = obs_vy[0]
    a_prev = obs_ax[0]
    dvdu = vy[0] / max(vx[0], 1e-4)

    for i in range(T - 1):
        if reset_every > 0 and i > 0 and i % reset_every == 0:
            x[i] = obs_x[i]; y[i] = obs_y[i]
            vx[i] = obs_vx[i]; vy[i] = obs_vy[i]
            a_prev = obs_ax[i]
            dvdu = vy[i] / max(vx[i], 1e-4)
        ax, ay = _accel(x[i], y[i], vx[i], vy[i],
                        nb_x[i], nb_y[i], nb_vx[i], nb_vy[i], nb_ax[i],
                        nb_L[i], nb_W[i], valid[i], a_prev,
                        L_ego, W_ego, p,
                        bmax, longPar, bIntCrit, smaxfollow,
                        accLatIntMaxFactor, useSeveralLeaders,
                        accLatBMax, v0max, cool, narrow_on * road_on,
                        road_on, y_left, y_right)
        a_prev = ax

        vxn = vx[i] + ax * dt
        vyn = vy[i] + ay * dt

        if clamp_on != 0:
            dvdu_new = vyn / max(vxn, 1e-4)
            if abs(dvdu_new - dvdu) > dt * dotdvdumax:
                dvdu += (1.0 if dvdu_new - dvdu > 0.0 else -1.0) * dotdvdumax * dt
            if dvdu > dvdumax:
                dvdu = dvdumax
            elif dvdu < -dvdumax:
                dvdu = -dvdumax
            vy_max = abs(dvdu * vxn)
            if vyn > vy_max:
                vyn = vy_max
            elif vyn < -vy_max:
                vyn = -vy_max

        dy_step = 0.5 * (vy[i] + vyn) * dt
        if vxn < -1e-6:
            dx_step = -0.5 * vx[i] * vx[i] / ax
            vxn = 0.0
        else:
            dx_step = 0.5 * (vx[i] + vxn) * dt
            if vxn < 0.0:
                vxn = 0.0
        xn = x[i] + dx_step
        yn = y[i] + dy_step

        x[i + 1] = xn; y[i + 1] = yn; vx[i + 1] = vxn; vy[i + 1] = vyn

        if (abs(xn - obs_x[i + 1]) > max_x_err or
                abs(yn - obs_y[i + 1]) > max_y_err or
                not np.isfinite(xn + yn)):
            return x, y, vx, vy, i + 2

    return x, y, vx, vy, T


@njit(cache=True, fastmath=False)
def gof_core(obs_x, obs_y, obs_vx, obs_vy, obs_ax,
             nb_x, nb_y, nb_vx, nb_vy, nb_ax, nb_L, nb_W, valid,
             L_ego, W_ego, p, dt,
             bmax, longPar, bIntCrit, smaxfollow, accLatIntMaxFactor,
             useSeveralLeaders, accLatBMax, v0max, dvdumax, dotdvdumax,
             cool, narrow_on, clamp_on,
             road_on, y_left, y_right,
             max_x_err, max_y_err, reset_every, penalty,
             w_x, w_y, w_v, x_scale, y_scale):
    """Full objective in one JIT call: no Python-level allocation per eval."""
    x, y, vx, vy, n = simulate_core(
        obs_x, obs_y, obs_vx, obs_vy, obs_ax,
        nb_x, nb_y, nb_vx, nb_vy, nb_ax, nb_L, nb_W, valid,
        L_ego, W_ego, p, dt,
        bmax, longPar, bIntCrit, smaxfollow, accLatIntMaxFactor,
        useSeveralLeaders, accLatBMax, v0max, dvdumax, dotdvdumax,
        cool, narrow_on, clamp_on,
        road_on, y_left, y_right,
        max_x_err, max_y_err, reset_every)

    T = obs_x.shape[0]
    if n < T:
        return penalty * (1.0 + (T - n) / T)

    sx = 0.0; sy = 0.0; sv = 0.0; svo = 0.0
    for i in range(T):
        ex = x[i] - obs_x[i]; sx += ex * ex
        ey = y[i] - obs_y[i]; sy += ey * ey
        ev = vx[i] - obs_vx[i]; sv += ev * ev
        svo += obs_vx[i] * obs_vx[i]

    v_scale = np.sqrt(svo / T)
    if v_scale < 0.5:
        v_scale = 0.5

    # Longitudinal position is normalised by a FIXED metre scale, not by
    # distance travelled. Dividing by distance travelled (~250 m over 45 s)
    # makes a 10 m spacing error look like 0.04 and effectively removes the
    # longitudinal signal from the objective.
    f = (w_x * np.sqrt(sx / T) / x_scale
         + w_y * np.sqrt(sy / T) / y_scale
         + w_v * np.sqrt(sv / T) / v_scale)
    if not np.isfinite(f):
        return penalty
    return f
