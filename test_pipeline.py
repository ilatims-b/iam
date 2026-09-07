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
from iam_calib.model import accelerations, idm_acc_free, idm_acc_int
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
        al, at = accelerations(0.0, 0.0, 5.0, 0.0, z, z, z, z, z, z,
                              np.zeros(4, bool), 4.0, 1.7, p,
                              RoadConfig(enabled=False))
        assert al == pytest.approx(idm_acc_free(5.0, p["v0"], p["a"]))
        assert at == pytest.approx(0.0)

    def test_lateral_free_term_damps_lateral_speed(self):
        p = default_params()
        z = np.zeros(4)
        _, at = accelerations(0.0, 0.0, 5.0, 1.0, z, z, z, z, z, z,
                              np.zeros(4, bool), 4.0, 1.7, p,
                              RoadConfig(enabled=False))
        assert at < 0                       # opposes the lateral speed

    def test_close_leader_causes_braking(self):
        p = default_params()
        nb = lambda v: np.array([v, 0.0, 0.0, 0.0])
        al, _ = accelerations(
            0.0, 0.0, 10.0, 0.0,
            nb(6.0), nb(0.0), nb(2.0), nb(0.0), nb(4.0), nb(1.7),
            np.array([True, False, False, False]), 4.0, 1.7, p,
            RoadConfig(enabled=False))
        assert al < -0.5

    def test_lateral_repulsion_pushes_away(self):
        """A leader offset to +y with lateral overlap must push the ego to -y."""
        p = default_params()
        nb = lambda v: np.array([v, 0.0, 0.0, 0.0])
        _, at = accelerations(
            0.0, 0.0, 10.0, 0.0,
            nb(7.0), nb(0.5), nb(3.0), nb(0.0), nb(4.0), nb(1.7),
            np.array([True, False, False, False]), 4.0, 1.7, p,
            RoadConfig(enabled=False))
        assert at < 0

    def test_boundary_pushes_inward(self):
        p = default_params()
        road = RoadConfig(enabled=True, y_left=3.5, y_right=-3.5)
        z = np.zeros(2)
        _, at = accelerations(0.0, 3.3, 10.0, 0.0, z, z, z, z, z, z,
                              np.zeros(2, bool), 4.0, 1.7, p, road)
        assert at < 0                       # near left edge -> pushed right


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
                        w.nb_L[i], w.nb_W[i], w.valid[i],
                        w.L, w.W, p, cfg.road)
                    a2 = _accel(
                        w.x[i], w.y[i], w.vx[i], w.vy[i],
                        w.nb_x[i], w.nb_y[i], w.nb_vx[i], w.nb_vy[i],
                        w.nb_L[i], w.nb_W[i], w.valid[i],
                        w.L, w.W, pv, C.bmax, C.longParReductFactor,
                        C.bIntCrit, C.smaxfollow, C.accLatIntMaxFactor,
                        C.useSeveralLeaders, C.accLatBMax, C.v0max,
                        0, 0.0, 0.0)
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
