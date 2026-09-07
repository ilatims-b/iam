"""
iam_calib -- CMA-ES calibration of the Intelligent-Agent Model (Treiber &
Chaudhari, arXiv:2310.16816) against lane-free trajectory data.

Model code ported from github.com/MTGermany/mixedTraffic (GPL-3.0).
"""

from .config import (CMAConfig, DataConfig, ObjectiveConfig, PipelineConfig,
                     RoadConfig, FREE_PARAMS, PARAM_SPECS, SPEC_BY_NAME)
from .data import VehicleWindow, build_windows, load_frame, summarise
from .model import accelerations
from .objective import gof, gof_components, simulate
from .calibrate import (CalibResult, calibrate_all, calibrate_vehicle,
                        default_params)
from .diagnostics import (bound_hit_report, identifiability, pool_by_class,
                          results_frame, shrink_thin_classes,
                          train_test_split, validate)

__version__ = "0.1.0"

__all__ = [
    "CMAConfig", "DataConfig", "ObjectiveConfig", "PipelineConfig",
    "RoadConfig", "FREE_PARAMS", "PARAM_SPECS", "SPEC_BY_NAME",
    "VehicleWindow", "build_windows", "load_frame", "summarise",
    "accelerations", "gof", "gof_components", "simulate",
    "CalibResult", "calibrate_all", "calibrate_vehicle", "default_params",
    "bound_hit_report", "identifiability", "pool_by_class", "results_frame",
    "shrink_thin_classes", "train_test_split", "validate",
]
