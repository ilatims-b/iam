"""
Configuration: parameter specifications, bounds, vehicle geometry, road geometry.

Parameter naming follows the reference JS implementation
(github.com/MTGermany/mixedTraffic, js/models.js), NOT the paper's notation,
because the code is the operational definition. Mapping to the paper:

    code            paper (Treiber & Chaudhari 2023)
    ------------------------------------------------
    s0y             s_0y   (longitudinal attenuation width)
    s0yLat          s_0y   (paper uses ONE width; code splits it in two)
    sensLat         sigma  (lateral sensitivity)
    tauLatOVM       tau / tau_y  (lateral relaxation time)
    pushLong        lambda (SFM anisotropy, longitudinal)
    pushLat         lambda (paper uses ONE lambda; code splits it in two)
    accLongBRef     f_B    (long. deceleration at boundary)
    accLatBRef      g_B    (lat. acceleration at boundary)
    s0yB, s0yLatB   s_B0   (boundary attenuation width; again split in two)

Defaults below are the JS defaults from sim_bicycles_IAM.js, which differ
from the paper's Table 1 (e.g. s0y=0.10 in code vs 0.3 in Table 1).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    """One calibratable parameter."""
    name: str
    default: float
    lo: float
    hi: float
    log: bool = False   # search in log space (for scale-like params)
    group: str = "idm"  # idm | iam | boundary


# ----------------------------------------------------------------------------
# Parameter table
# ----------------------------------------------------------------------------
# Bounds are physical plausibility ranges. Widen with care: a parameter that
# converges onto its bound across many vehicles is a sign the bound is wrong
# OR that the parameter is unidentified. check_bound_hits() in diagnostics.py
# reports this.

PARAM_SPECS: List[ParamSpec] = [
    # --- underlying IDM ---
    ParamSpec("v0",         12.0,  1.0,  35.0, False, "idm"),   # desired speed [m/s]
    ParamSpec("T",           1.2,  0.2,   4.0, False, "idm"),   # time gap [s]
    ParamSpec("s0",          1.5,  0.1,   6.0, False, "idm"),   # min gap [m]
    ParamSpec("a",           1.0,  0.2,   4.0, False, "idm"),   # max accel [m/s^2]
    ParamSpec("b",           1.5,  0.5,   5.0, False, "idm"),   # comf decel [m/s^2]
    # --- IAM-specific ---
    ParamSpec("s0y",         0.10, 0.02,  2.0, True,  "iam"),   # long. atten. width [m]
    ParamSpec("s0yLat",      0.15, 0.02,  2.0, True,  "iam"),   # lat. atten. width [m]
    ParamSpec("sensLat",     1.0,  0.05,  6.0, True,  "iam"),   # lateral sensitivity [s]
    ParamSpec("tauLatOVM",   1.0,  0.15,  6.0, True,  "iam"),   # lat. relaxation [s]
    ParamSpec("pushLong",    0.1,  0.0,   0.6, False, "iam"),   # lambda_long
    ParamSpec("pushLat",     0.1,  0.0,   0.6, False, "iam"),   # lambda_lat
    # --- boundary (fixed by default; see FREE_PARAMS) ---
    ParamSpec("accLongBRef", 0.2,  0.0,   2.0, False, "boundary"),
    ParamSpec("accLatBRef",  4.0,  0.5,  10.0, False, "boundary"),
    ParamSpec("s0yB",        0.1,  0.02,  1.0, True,  "boundary"),
    ParamSpec("s0yLatB",     0.1,  0.02,  1.0, True,  "boundary"),
]

SPEC_BY_NAME: Dict[str, ParamSpec] = {p.name: p for p in PARAM_SPECS}

# Which parameters CMA-ES actually searches. Everything else is held at
# `default` (or at a value you pass in `fixed`).
#
# Boundary parameters are excluded by default: they are only identifiable if
# (a) you know the road edges in the same frame as Y_Smt, and (b) vehicles
# actually approach them. Add them back once road geometry is configured.
FREE_PARAMS: Tuple[str, ...] = (
    "v0", "T", "s0", "a", "b",
    "s0y", "s0yLat", "sensLat", "tauLatOVM",
    "pushLong", "pushLat",
)

# Staged calibration: longitudinal first, then lateral with longitudinal frozen.
STAGE_LONGITUDINAL: Tuple[str, ...] = ("v0", "T", "s0", "a", "b", "s0y", "pushLong")
STAGE_LATERAL: Tuple[str, ...] = ("s0yLat", "sensLat", "tauLatOVM", "pushLat")


# ----------------------------------------------------------------------------
# Constants that the JS code treats as global / non-calibrated
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConstants:
    bmax: float = 9.0                 # max IDM deceleration [m/s^2] (paper: 9)
    longParReductFactor: float = 0.2  # long. force reduction when driving parallel
    bIntCrit: float = 0.02            # threshold [m/s^2] for counting a neighbour
    smaxfollow: float = 30.0          # max gap [m] for a follower to push
    accLatIntMaxFactor: float = 4.0   # accLatIntMax = factor * b
    accLatBMax: float = 20.0          # clamp on boundary lateral accel
    useSeveralLeaders: bool = False   # False => Eq.(7) min/max form (JS default)
    v0max: float = 20.0               # reference speed used in boundary scaling


MODEL_CONSTANTS = ModelConstants()


# ----------------------------------------------------------------------------
# Vehicle geometry -- NOT calibrated (see README: calibrating L/W lets it
# absorb the physics of s0y and sensLat)
# ----------------------------------------------------------------------------
# Keys must match the `Class` column of your CSV, case-insensitively.
# EDIT THESE to match your site's measured dimensions.
DEFAULT_VEHICLE_DIMS: Dict[str, Tuple[float, float]] = {
    # class            (length_m, width_m)
    "car":             (4.0, 1.7),
    "auto":            (3.2, 1.4),   # auto-rickshaw
    "motorcycle":      (1.8, 0.7),
    "moto":            (1.8, 0.7),
    "2w":              (1.8, 0.7),
    "bike":            (1.8, 0.6),
    "bicycle":         (1.8, 0.6),
    "cycle":           (1.8, 0.6),
    "bus":            (10.5, 2.5),
    "lcv":             (5.5, 2.0),
    "tractor":         (4.5, 2.0),
    "pedestrian":      (0.5, 0.5),
    # Site-measured dimensions for D3F1_lcl_smtF.csv (avg. observed by PhD
    # student on-site, 2026-09-07). Supersede the earlier literature guesses.
    "truck":           (8.2, 2.3),
    "multiaxle":       (12.5, 2.5),
    "trailer":         (16.3, 2.5),
}
FALLBACK_DIMS: Tuple[float, float] = (4.0, 1.7)


@dataclass
class RoadConfig:
    """
    Lateral road geometry, in the SAME coordinate frame as Y_Smt.

    If `enabled` is False, boundary forces are switched off entirely. That is
    the right default until you have confirmed where the road edges actually
    are -- a wrong road width injects a spurious lateral force everywhere and
    will silently corrupt every lateral parameter.

    y_left  : lateral coordinate of the left edge  (larger value)
    y_right : lateral coordinate of the right edge (smaller value)
    """
    enabled: bool = False
    y_left: float = 0.0
    y_right: float = 0.0

    @property
    def width(self) -> float:
        return self.y_left - self.y_right

    @classmethod
    def from_data_extent(cls, y: np.ndarray, quantile: float = 0.995,
                         margin: float = 0.5) -> "RoadConfig":
        """
        Crude fallback: infer edges from the lateral extent of observed
        trajectories. Use ONLY as a starting guess -- vehicles do not drive at
        the physical edge, so this underestimates the width.
        """
        lo = float(np.quantile(y, 1.0 - quantile))
        hi = float(np.quantile(y, quantile))
        return cls(enabled=True, y_left=hi + margin, y_right=lo - margin)


# ----------------------------------------------------------------------------
# Data / simulation configuration
# ----------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Column names and windowing controls."""
    col_id: str = "VehicleID"
    col_class: str = "Class"
    col_time: str = "GTime_sec"      # global clock -- required for neighbour joins
    col_x: str = "X_Smt"             # smoothed longitudinal position
    col_y: str = "Y_Smt"             # smoothed lateral position
    col_vx: str = "Long_Speed"
    col_vy: str = "Lat_Speed"
    col_ax: str = "Long_Acc"
    col_ay: str = "Lat_Acc"

    dt: float = 0.04                 # sampling interval [s]
    window_s: float = 30.0           # calibration window length [s]
    min_window_s: float = 15.0       # discard trajectories shorter than this
    # Candidate-neighbour radius. Generous, because the simulated ego drifts
    # away from its measured position and must not lose neighbours mid-run.
    radius_x: float = 80.0
    radius_y: float = 8.0
    max_neighbours: int = 24         # padding width K
    y_sign_positive_is_left: bool = True
    # The model treats the longitudinal coordinate as the FRONT bumper (JS
    # convention: u = front). Most trajectory datasets give the centroid.
    # False => data.py adds L/2. Set True if your X is already the front.
    # Confirmed with PhD student (2026-09-07): X_Smt/Y_Smt are the vehicle
    # centroid for D3F1_lcl_smtF.csv, so False is correct here.
    x_is_front: bool = False


@dataclass
class ObjectiveConfig:
    """Goodness-of-fit weights. See objective.py for the formula."""
    w_x: float = 1.0     # longitudinal position error
    w_y: float = 1.0     # lateral position error
    w_v: float = 1.0     # relative longitudinal speed error
    x_scale: float = 2.0 # [m] normalisation for longitudinal error
    y_scale: float = 0.5 # [m] normalisation for lateral error
    # Re-initialise the ego state from data every `reset_s` seconds. 0 = never
    # (free-running over the whole window). See fastcore.simulate_core for why
    # this is not optional in lane-free traffic.
    reset_s: float = 3.0
    # Stop integrating a window early if the ego diverges absurdly; returns a
    # large penalty. Guards against parameter sets that blow up.
    max_x_error: float = 150.0
    max_y_error: float = 12.0
    penalty: float = 1e3


@dataclass
class CMAConfig:
    sigma0: float = 2.0          # in normalised [0, 10] search space
    maxfevals: int = 3000
    popsize: int = 0             # 0 => CMA default (4 + 3*ln(n))
    tolfun: float = 1e-4
    seed: int = 1
    polish: bool = True          # Nelder-Mead refinement after CMA-ES
    polish_maxiter: int = 300
    n_restarts: int = 0          # extra CMA runs from perturbed starts


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    cma: CMAConfig = field(default_factory=CMAConfig)
    road: RoadConfig = field(default_factory=RoadConfig)
    vehicle_dims: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_VEHICLE_DIMS))
    free_params: Tuple[str, ...] = FREE_PARAMS
    staged: bool = False         # True => longitudinal stage, then lateral
    n_workers: int = 0           # 0 => os.cpu_count()
    max_vehicles: int = 0        # 0 => all; set small for smoke tests


def dims_for_class(cls_name: str, table: Dict[str, Tuple[float, float]]
                   ) -> Tuple[float, float]:
    if cls_name is None:
        return FALLBACK_DIMS
    key = str(cls_name).strip().lower()
    if key in table:
        return table[key]
    for k, v in table.items():          # substring fallback
        if k in key or key in k:
            return v
    return FALLBACK_DIMS
