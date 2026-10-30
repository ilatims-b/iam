"""
Test suite. Run with:  python -m pytest tests/ -v

The agreement test is the important one. model.py is the readable reference and
fastcore.py is what actually runs; if you edit one without the other, this test
is what catches it. It already caught a swapped subject/leader lateral speed in
the follower-push branch that was invisible in the trajectories.
"""

import numpy as np
import pytest

from iam_calib import (DataConfig, ObjectiveConfig, PipelineConfig, RoadConfig,
                       build_windows, load_frame)
from iam_calib.calibrate import default_params, from_norm, to_norm
from iam_calib.config import MODEL_CONSTANTS as C
from iam_calib.config import FREE_PARAMS, SPEC_BY_NAME
from iam_calib.fastcore import _accel, pack, unpack
from iam_calib.model import acc_int, accelerations, idm_acc_free, idm_acc_int
from iam_calib.objective import gof, make_fast_gof, simulate

CSV = "synthetic.csv"


@pytest.fixture(scope="module")
def windows():
    cfg = PipelineConfig(data=DataConfig(window_s=25, min_window_s=15),
                         max_vehicles=6)
    return build_windows(load_frame(CSV, cfg.data), cfg, verbose=False), cfg


def random_params(rng):
    p = default_params()
    for n in FREE_PARAMS:
        s = SPEC_BY_NAME[n]
        p[n] = float(rng.uniform(s.lo, s.hi))
    return p


# --------------------------------------------------------------------------
class TestIDM:
    def test_free_acc_zero_at_v0(self):
        assert idm_acc_free(12.0, 12.0, 1.0) == pytest.approx(0.0)

    def test_free_acc_negative_above_v0(self):
        assert idm_acc_free(20.0, 12.0, 1.0) < 0

    def test_interaction_always_braking(self):
        p = default_params()
        s = np.array([0.1, 1.0, 5.0, 50.0, 500.0])
        acc = idm_acc_int(s, 10.0, 8.0, p)
        assert np.all(acc <= 0)

    def test_interaction_floored_at_bmax(self):
        p = default_params()
        acc = idm_acc_int(np.array([1e-6]), 30.0, 0.0, p)
        assert acc[0] >= -C.bmax - 1e-9

    def test_interaction_vanishes_at_large_gap(self):
        p = default_params()
        assert abs(idm_acc_int(np.array([1e5]), 10.0, 10.0, p)[0]) < 1e-4


class TestModelInvariants:
    def test_no_neighbours_gives_free_acceleration(self):
        p = default_params()
        z = np.zeros(4)
        al, at = accelerations(0.0, 0.0, 5.0, 0.0, z, z, z, z, z, z, z,
                              np.zeros(4, bool), 4.0, 1.7, p,
                              RoadConfig(enabled=False))
        assert al == pytest.approx(idm_acc_free(5.0, p["v0"], p["a"]))
        assert at == pytest.approx(0.0)

    def test_lateral_free_term_damps_lateral_speed(self):
        p = default_params()
        z = np.zeros(4)
        _, at = accelerations(0.0, 0.0, 5.0, 1.0, z, z, z, z, z, z, z,
                              np.zeros(4, bool), 4.0, 1.7, p,
                              RoadConfig(enabled=False))
        assert at < 0                       # opposes the lateral speed

    def test_close_leader_causes_braking(self):
        p = default_params()
        nb = lambda v: np.array([v, 0.0, 0.0, 0.0])
        al, _ = accelerations(
            0.0, 0.0, 10.0, 0.0,
            nb(6.0), nb(0.0), nb(2.0), nb(0.0), nb(0.0), nb(4.0), nb(1.7),
            np.array([True, False, False, False]), 4.0, 1.7, p,
            RoadConfig(enabled=False))
        assert al < -0.5

    def test_lateral_repulsion_pushes_away(self):
        """A leader offset to +y with lateral overlap must push the ego to -y."""
        p = default_params()
        nb = lambda v: np.array([v, 0.0, 0.0, 0.0])
        _, at = accelerations(
            0.0, 0.0, 10.0, 0.0,
            nb(7.0), nb(0.5), nb(3.0), nb(0.0), nb(0.0), nb(4.0), nb(1.7),
            np.array([True, False, False, False]), 4.0, 1.7, p,
            RoadConfig(enabled=False))
        assert at < 0

    def test_boundary_pushes_inward(self):
        p = default_params()
        road = RoadConfig(enabled=True, y_left=3.5, y_right=-3.5)
        z = np.zeros(2)
        _, at = accelerations(0.0, 3.3, 10.0, 0.0, z, z, z, z, z, z, z,
                              np.zeros(2, bool), 4.0, 1.7, p, road)
        assert at < 0                       # near left edge -> pushed right


class TestACC:
    def test_acc_matches_idm_when_leader_brakes_like_idm_expects(self):
        """ACC is within the 1% IDM blend of CAH-softened IDM, never harder."""
        p = default_params()
        s = np.array([2.0, 5.0, 20.0])
        idm = idm_acc_int(s, 10.0, 8.0, p)
        acc = acc_int(s, 10.0, 8.0, 0.0, p)
        assert np.all(acc >= idm - 1e-9)           # never brakes harder than IDM

    def test_cut_in_is_softer_than_idm(self):
        """Tiny gap to a non-braking leader: IDM panics, ACC does not."""
        p = default_params()
        idm = idm_acc_int(np.array([1.0]), 8.0, 7.0, p)[0]
        acc = acc_int(np.array([1.0]), 8.0, 7.0, 0.0, p)[0]
        assert acc > idm + 1.0

    def test_leader_braking_hard_gives_harder_braking(self):
        p = default_params()
        calm = acc_int(np.array([6.0]), 10.0, 8.0, 0.0, p)[0]
        hard = acc_int(np.array([6.0]), 10.0, 8.0, -4.0, p)[0]
        assert hard < calm

    def test_floored_at_bmax(self):
        p = default_params()
        acc = acc_int(np.array([1e-6]), 30.0, 0.0, -9.0, p)
        assert acc[0] >= -C.bmax - 1e-9


class TestLateralBiasAndOverride:
    def test_bias_adds_constant_lateral_acceleration(self):
        p = default_params()
        z = np.zeros(2)
        args = (0.0, 0.0, 5.0, 0.0, z, z, z, z, z, z, z, np.zeros(2, bool),
                4.0, 1.7)
        _, a0 = accelerations(*args, p, RoadConfig(enabled=False))
        p2 = dict(p, accBiasLat=-0.8)
        _, a1 = accelerations(*args, p2, RoadConfig(enabled=False))
        assert a1 - a0 == pytest.approx(-0.8)

    def test_too_narrow_override_pushes_to_open_side(self):
        """Slow leader hugging the right edge, ego laterally overlapping and
        slightly to its right: the right side is too narrow, so the ego must
        be pushed LEFT (+y) -- opposite to ordinary repulsion."""
        p = default_params()
        road = RoadConfig(enabled=True, y_left=4.35, y_right=-4.35)
        nb = lambda v: np.array([v, 0.0])
        valid = np.array([True, False])
        # leader at y=-3.6 (0.75 m from right edge); ego at y=-3.8 (right of it)
        _, at = accelerations(0.0, -3.8, 10.0, 0.0,
                              nb(6.0), nb(-3.6), nb(1.0), nb(0.0), nb(0.0),
                              nb(4.0), nb(1.7), valid, 4.0, 1.7, p, road)
        _, at_off = accelerations(0.0, -3.8, 10.0, 0.0,
                                  nb(6.0), nb(-3.6), nb(1.0), nb(0.0),
                                  nb(0.0), nb(4.0), nb(1.7), valid, 4.0, 1.7,
                                  p, RoadConfig(enabled=False))
        assert at_off < 0 < at                    # sign flips with override


class TestJITAgreement:
    """fastcore must reproduce model.py to machine precision."""

    def test_accelerations_match(self, windows):
        wins, cfg = windows
        rng = np.random.default_rng(3)
        worst = 0.0
        for w in wins:
            for _ in range(5):
                p = random_params(rng)
                pv = pack(p)
                for i in range(0, w.n_steps, 11):
                    a1 = accelerations(
                        w.x[i], w.y[i], w.vx[i], w.vy[i],
                        w.nb_x[i], w.nb_y[i], w.nb_vx[i], w.nb_vy[i],
                        w.nb_ax[i], w.nb_L[i], w.nb_W[i], w.valid[i],
                        w.L, w.W, p, cfg.road, w.ax[i])
                    a2 = _accel(
                        w.x[i], w.y[i], w.vx[i], w.vy[i],
                        w.nb_x[i], w.nb_y[i], w.nb_vx[i], w.nb_vy[i],
                        w.nb_ax[i], w.nb_L[i], w.nb_W[i], w.valid[i],
                        w.ax[i], w.L, w.W, pv, C.bmax, C.longParReductFactor,
                        C.bIntCrit, C.smaxfollow, C.accLatIntMaxFactor,
                        C.useSeveralLeaders, C.accLatBMax, C.v0max,
                        C.cool, 0, 0, 0.0, 0.0)
                    worst = max(worst, abs(a1[0] - a2[0]), abs(a1[1] - a2[1]))
        assert worst < 1e-10, f"JIT diverged from reference by {worst:.2e}"

    def test_objective_matches(self, windows):
        wins, cfg = windows
        rng = np.random.default_rng(11)
        for w in wins[:3]:
            f = make_fast_gof(w, cfg.road, cfg.objective)
            for _ in range(4):
                p = random_params(rng)
                assert f(pack(p)) == pytest.approx(
                    gof(w, p, cfg.road, cfg.objective), rel=1e-9, abs=1e-9)

    def test_agreement_with_boundaries_on(self, windows):
        wins, cfg = windows
        cfg2 = PipelineConfig(data=cfg.data, objective=cfg.objective,
                              road=RoadConfig(True, 4.0, -4.0))
        rng = np.random.default_rng(5)
        w = wins[0]
        f = make_fast_gof(w, cfg2.road, cfg2.objective)
        for _ in range(4):
            p = random_params(rng)
            assert f(pack(p)) == pytest.approx(
                gof(w, p, cfg2.road, cfg2.objective), rel=1e-9, abs=1e-9)


class TestParamTransform:
    def test_roundtrip(self):
        p = default_params()
        names = list(FREE_PARAMS)
        back = from_norm(to_norm(p, names), names, p)
        for n in names:
            assert back[n] == pytest.approx(p[n], rel=1e-9)

    def test_pack_unpack_roundtrip(self):
        p = default_params()
        for k, v in unpack(pack(p)).items():
            assert v == pytest.approx(p[k])

    def test_norm_respects_bounds(self):
        names = list(FREE_PARAMS)
        for z in (np.full(len(names), -5.0), np.full(len(names), 99.0)):
            p = from_norm(z, names, default_params())
            for n in names:
                s = SPEC_BY_NAME[n]
                assert s.lo - 1e-9 <= p[n] <= s.hi + 1e-9


class TestData:
    def test_windows_are_contiguous_and_shaped(self, windows):
        wins, cfg = windows
        for w in wins:
            assert w.n_steps >= int(cfg.data.min_window_s / cfg.data.dt)
            assert w.nb_x.shape == (w.n_steps, cfg.data.max_neighbours)
            assert w.valid.shape == w.nb_x.shape
            assert np.all(np.diff(w.t) > 0)

    def test_ego_excluded_from_own_neighbours(self, windows):
        wins, _ = windows
        for w in wins:
            d = np.hypot(w.nb_x - w.x[:, None], w.nb_y - w.y[:, None])
            assert np.all(d[w.valid] > 1e-9)


class TestSimulation:
    def test_starts_at_observed_state(self, windows):
        wins, cfg = windows
        w = wins[0]
        x, y, vx, vy, _ = simulate(w, default_params(), cfg.road, cfg.objective)
        assert x[0] == pytest.approx(w.x[0])
        assert y[0] == pytest.approx(w.y[0])

    def test_speed_never_negative(self, windows):
        wins, cfg = windows
        rng = np.random.default_rng(2)
        for w in wins[:3]:
            _, _, vx, _, n = simulate(w, random_params(rng), cfg.road,
                                      cfg.objective)
            assert np.all(vx[:n] >= -1e-12)

    def test_objective_finite_over_whole_search_box(self, windows):
        """No parameter set inside the bounds may produce NaN or inf."""
        wins, cfg = windows
        rng = np.random.default_rng(9)
        f = make_fast_gof(wins[0], cfg.road, cfg.objective)
        for _ in range(200):
            v = f(pack(random_params(rng)))
            assert np.isfinite(v) and v >= 0


@pytest.mark.slow
class TestRecovery:
    """End-to-end: can CMA-ES recover parameters it is known to have?"""

    def test_recovers_synthetic_parameters(self):
        from iam_calib import CMAConfig, calibrate_all
        from iam_calib.diagnostics import results_frame
        import make_synthetic

        cfg = PipelineConfig(
            data=DataConfig(window_s=30, min_window_s=20),
            objective=ObjectiveConfig(reset_s=3.0, x_scale=2.0, y_scale=0.5),
            cma=CMAConfig(maxfevals=3000), n_workers=0, max_vehicles=20)
        wins = build_windows(load_frame(CSV, cfg.data), cfg, verbose=False)
        df = results_frame(calibrate_all(wins, cfg, verbose=False))
        ok = df[df.converged]
        assert len(ok) >= 0.8 * len(df)

        truth = make_synthetic.TRUE_PARAMS
        # v0 and the lateral relaxation time are the well-identified ones;
        # s0 and pushLat are NOT expected to recover in free-flow data.
        for name, tol in (("v0", 0.25), ("tauLatOVM", 0.5), ("b", 0.5)):
            est = ok[name].median()
            assert abs(est - truth[name]) / truth[name] < tol, \
                f"{name}: got {est:.3f}, true {truth[name]:.3f}"


# ---------------------------------------------------------------------------
# Search bounds must reach the pool workers. They are spawned, so a bound set
# only in the main process is invisible to them, and the optimiser then
# searches a different box from the one used to decode the saved parameters.
def _worker_bounds(name):
    from iam_calib.config import SPEC_BY_NAME
    sp = SPEC_BY_NAME[name]
    return sp.lo, sp.hi, sp.log


def test_search_bounds_reach_spawned_workers():
    from concurrent.futures import ProcessPoolExecutor
    from iam_calib.calibrate_joint import apply_bounds, _init_pool
    ov = {"a": (0.02, 4.0, True), "sensLat": (0.002, 6.0, True)}
    with ProcessPoolExecutor(1, initializer=_init_pool,
                             initargs=(None, ov)) as pool:
        assert pool.submit(_worker_bounds, "a").result() == (0.02, 4.0, True)
        assert pool.submit(_worker_bounds, "sensLat").result() == (0.002, 6.0, True)


# ---------------------------------------------------------------------------
# Joint (simultaneous) simulation. Two defects got past an earlier version of
# these tests: a 24-neighbour cap that filled slots behind the ego first and so
# dropped leaders, and vehicles vanishing when their measured twin left.
def _joint_scene(states, T, present=None):
    """states: list of (x, y, vx, L, W). Returns a scene dict for _run_scene."""
    n = len(states)
    X = np.full((n, T), np.nan); Y = X.copy(); VX = X.copy(); VY = X.copy(); AX = X.copy()
    for k, (x, y, v, L_, W_) in enumerate(states):
        X[k, :] = x; Y[k, :] = y; VX[k, :] = v; VY[k, :] = 0.0; AX[k, :] = 0.0
    pres = np.ones((n, T), bool) if present is None else present
    return dict(X=X, Y=Y, VX=VX, VY=VY, AX=AX, present=pres,
                L=np.array([s_[3] for s_ in states]),
                W=np.array([s_[4] for s_ in states]),
                cls=["car"] * n, T=T, dt=0.04, x_exit_c=1e9)


def _joint_run(scene, p, K=96):
    from iam_calib.calibrate_joint import run_scene
    P = np.array([[p[k] for k in PARAM_ORDER_]])
    road = RoadConfig(enabled=True, y_left=8.9, y_right=0.0)
    return run_scene(scene, P, {"car": 0}, road, K=K, return_trunc=True)


from iam_calib.fastcore import PARAM_ORDER as PARAM_ORDER_


def _reference_step(states, i, p, keep):
    """x after one 0.04 s step for vehicle i, from the readable reference model
    and the neighbours listed in `keep`."""
    road = RoadConfig(enabled=True, y_left=8.9, y_right=0.0)
    xs = np.array([states[j][0] for j in keep]); ys = np.array([states[j][1] for j in keep])
    vs = np.array([states[j][2] for j in keep]); zero = np.zeros(len(keep))
    ax, _ = accelerations(states[i][0], states[i][1], states[i][2], 0.0,
                          xs, ys, vs, zero, zero,
                          np.array([states[j][3] for j in keep]),
                          np.array([states[j][4] for j in keep]),
                          np.ones(len(keep), bool), states[i][3], states[i][4],
                          p, road, 0.0)
    vn = max(0.0, states[i][2] + ax * 0.04)
    return states[i][0] + 0.5 * (states[i][2] + vn) * 0.04


def test_joint_step_matches_reference_model():
    p = default_params()
    states = [(100.0, 4.0, 8.0, 4.0, 1.7), (112.0, 4.2, 5.0, 4.0, 1.7),
              (92.0, 3.9, 9.0, 4.0, 1.7), (130.0, 5.0, 3.0, 4.0, 1.7)]
    sx, _, _, nt = _joint_run(_joint_scene(states, 2), p)
    assert nt == 0
    for i in range(len(states)):
        keep = [j for j in range(len(states)) if j != i]
        assert sx[i, 1] == pytest.approx(_reference_step(states, i, p, keep), abs=1e-9)


def test_joint_cap_keeps_nearest_neighbours_from_both_sides():
    """With room for two neighbours, the ego must see the two NEAREST by
    longitudinal distance, here one leader and one follower, not the two
    nearest vehicles behind it."""
    p = default_params()
    states = [(100.0, 4.0, 8.0, 4.0, 1.7),      # ego
              (104.0, 4.0, 2.0, 4.0, 1.7),      # leader 4 m ahead
              (95.0, 4.2, 8.0, 4.0, 1.7),       # followers behind, nearer than...
              (94.0, 3.8, 8.0, 4.0, 1.7),
              (93.0, 4.1, 8.0, 4.0, 1.7),
              (160.0, 4.0, 8.0, 4.0, 1.7)]      # ...a far vehicle ahead
    sx, _, _, nt = _joint_run(_joint_scene(states, 2), p, K=2)
    assert nt > 0                                     # the cap really was exceeded
    assert sx[0, 1] == pytest.approx(_reference_step(states, 0, p, [1, 2]), abs=1e-9)


def test_joint_vehicle_outlives_its_measured_twin():
    """A leader whose measured twin stops being present must stay on the road
    and keep influencing the follower. The follower is closing on a stopped
    leader, so keeping it must slow the follower versus the leader never
    existing."""
    p = default_params(); T = 200
    follower = (50.0, 4.0, 10.0, 4.0, 1.7); leader = (80.0, 4.0, 0.0, 4.0, 1.7)
    pres = np.ones((2, T), bool); pres[1, 20:] = False          # twin leaves at t=20
    with_leader = _joint_run(_joint_scene([follower, leader], T, pres), p)[0]
    alone = _joint_run(_joint_scene([follower], T), p)[0]
    assert with_leader[0, -1] < alone[0, -1] - 1.0


def test_joint_exit_holds_position_and_is_scored():
    """A vehicle crossing the exit line is retired, and frames where its twin is
    still present carry the held exit position rather than NaN."""
    from iam_calib.calibrate_joint import run_scene
    p = default_params(); T = 400
    sc = _joint_scene([(100.0, 4.0, 12.0, 4.0, 1.7)], T); sc["x_exit_c"] = 110.0
    P = np.array([[p[k] for k in PARAM_ORDER_]])
    sx, _, svx, _ = run_scene(sc, P, {"car": 0}, RoadConfig(enabled=True, y_left=8.9, y_right=0.0),
                              return_trunc=True)
    assert np.isfinite(sx).all()
    held = sx[0, -1]
    assert held - 0.5 * 4.0 >= 110.0 and held < 125.0
    assert sx[0, -1] == sx[0, -50]                    # frozen once it has left
