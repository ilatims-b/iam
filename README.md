# iam_calib: calibrating the intelligent-agent model to lane-free mixed traffic

Code and report for calibrating the intelligent-agent model (IAM) of Treiber
and Chaudhari ([arXiv:2310.16816](https://arxiv.org/abs/2310.16816)) to a 781 s
record of lane-free mixed traffic: 554 vehicles in seven classes on a 204 m
section. All vehicles are simulated together from their measured entry states,
with one parameter vector per class, an exit stop line read from the measured
stops, and a train/test split by vehicle inside the same simulation.

The model code is a NumPy/Numba port of Treiber's reference implementation,
[MTGermany/mixedTraffic](https://github.com/MTGermany/mixedTraffic) (GPL-3.0),
not a transcription of the paper's equations. The differences are listed in
[Model port](#model-port).

**Report:** `paper_joint/calib_report/calib_report.tex` (UGRP Technical Report,
CE4901).

**Further experiments and details:** all other runs, the earlier write-ups and
the fitted parameters of every run are in a separate archive:
**[further experiments and details (Google Drive)](https://drive.google.com/drive/folders/1se1xUjHdJvRikvYFdVKnxmwq0XoU9EeE)**. It
contains a report of every experiment run before the final set-up (per-vehicle
calibration, scene-based simultaneous simulation, the first whole-record fits),
the bugs found along the way and the results they invalidated.

---

## Setup

The code is imported as the package `iam_calib`, so clone into a folder of that
name; the commands below run from inside it with `PYTHONPATH=..`.

```bash
git clone https://github.com/ilatims-b/iam iam_calib
cd iam_calib
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Use `.venv/bin/python` (or activate the environment) wherever the commands
below say `python`. Developed with Python 3.12.

## Data

The trajectory record (`D3F1_lcl_smtF.csv`) and the vehicle dimensions were
shared for this project and are not included. Place the file in the repository
root to rerun anything below.

## Local files and disk space

A working copy takes about 800 MB, almost all of it in files that git ignores:

| File or folder | Size | Needed for | Recreate with |
|---|---|---|---|
| `.venv/` | ~460 MB | everything | the setup commands above |
| `D3F1_lcl_smtF.csv` | ~170 MB | calibration, figures, tables | not recreatable; keep a copy |
| `synthetic.csv` | ~95 MB | tests only | `python make_synthetic.py ...` (see [Tests](#tests)) |
| `__pycache__/` | ~2 MB | compiled Numba kernels | rebuilt automatically on the next run |
| `paper_joint/calib_report/*.pdf` | ~5 MB each | reading the report | `pdflatex` (see below) |

`.venv/`, `synthetic.csv` and `__pycache__/` can be deleted at any time to free
space. `git gc` compresses the local git history.

## What is included

- `results/j12_dims`: all parameters free.
- `results/j13_duggal_idm`: IDM parameters fixed per class.
- `results/j14_chandra_dims`: IDM parameters fixed at a single set.

For each: the fitted class parameters, run configuration, split and convergence
history, and for the free fit the one-at-a-time sensitivity profiles. The
per-class IDM values of the fixed runs are in `results/idm_duggal_by_class.json`.
The report's figures and tables in `paper_joint/calib_report/figs/` are
generated from these and the data.

## Rebuild the report's figures and tables

A few minutes; needs the data.

```bash
# from this directory; PYTHONPATH=.. makes the package importable as iam_calib
PYTHONPATH=.. python plot_profiles.py --run results/j12_dims
PYTHONPATH=.. python make_calib_report.py
PYTHONPATH=.. python overlap_compare.py          # Appendix: dimensions and overlaps
cd paper_joint/calib_report && pdflatex calib_report.tex && pdflatex calib_report.tex
```

## Rerun the calibrations

Two hours of fitting each, from the parent directory:

```bash
DIMS="Auto=2.7x1.4 Bike=1.8x0.65 Car=4x1.7 LCV=4.5x1.75 Truck=8.5x2.4 MultiAxle=12x2.5 Trailer=15.5x2.5"
COMMON="iam_calib/D3F1_lcl_smtF.csv --whole-record --exit-signal --bounds realistic \
        --dims $DIMS --maxfevals 1000000 --max-minutes 120 --workers 4"
python -m iam_calib.calibrate_joint $COMMON --out iam_calib/results/j12_dims
python -m iam_calib.calibrate_joint $COMMON --fix-class iam_calib/results/idm_duggal_by_class.json \
    --out iam_calib/results/j13_duggal_idm
python -m iam_calib.calibrate_joint $COMMON \
    --fix v0=18.7615 T=0.8899 s0=1.05 a=2.0 b=2.6327 delta=4.42 --out iam_calib/results/j14_chandra_dims
# sensitivity profiles of the free fit (3151 simulations, about 15 min on 6 cores)
cd iam_calib && PYTHONPATH=.. python oat_profiles.py --run results/j12_dims --workers 6
```

Numba caches compiled kernels in `__pycache__`. After changing `fastcore.py`,
delete the cached `jointsim._run_scene*` files, since a cached caller is not
recompiled when a function it calls changes.

## Code

| File | Role |
|---|---|
| `config.py` | parameters, search boxes (`REALISTIC_BOUNDS`), granularity, vehicle dimensions, model constants |
| `model.py` | readable reference implementation of the IAM accelerations |
| `fastcore.py` | Numba version of the same accelerations, used in all simulations |
| `jointsim.py` | simultaneous simulation of all vehicles; exit stop-line detection |
| `calibrate_joint.py` | whole-record calibration with CMA-ES (command line above) |
| `oat_profiles.py`, `plot_profiles.py` | one-at-a-time sensitivity profiles and their figures |
| `make_calib_report.py`, `make_figs.py` | figures and tables of the report |
| `plot_convergence.py`, `signal_fig.py`, `overlap_figs.py`, `overlap_compare.py` | convergence, exit-signal evidence and vehicle-overlap figures |
| `calibrate.py`, `data.py`, `objective.py`, `diagnostics.py` | parameter transforms and helpers shared with the earlier per-vehicle pipeline |
| `make_synthetic.py`, `test_pipeline.py` | synthetic data for the tests, and the tests |

## Tests

```bash
python make_synthetic.py --out synthetic.csv --vehicles 300 --duration 60 \
    --length 1430 --road-width 8.7
PYTHONPATH=.. python -m pytest test_pipeline.py -q -m "not slow"   # 32 tests
PYTHONPATH=.. python -m pytest test_pipeline.py -q -m slow         # recovery test (minutes)
```

The agreement tests check that `fastcore.py` matches `model.py`, and the
joint-simulation tests check the neighbour selection and the exit handling.

---

## Model port

`model.py` follows the reference code (MTGermany/mixedTraffic, files
`models.js`, `road.js`, `vehicle.js`, lane-free scenario `sim_bicycles_IAM.js`)
where it and the paper disagree:

1. **The car-following model is ACC, not IDM.** `ACC.calcAccInt` blends IDM
   with a constant-acceleration heuristic (CAH) that uses the leader's
   acceleration `a_l`: `mix = IDM` if `IDM > CAH` else
   `CAH + b·tanh((IDM−CAH)/b)`; final `0.99·mix + 0.01·IDM`. See
   `model.acc_int`.
2. `s_x` is clamped at zero: `sx = max(0, dx - L_j)`.
3. `α(s_y) = min(exp(−s_y/s0y), 1)` uses **signed** `s_y`, not `|s_y|`.
4. Parallel driving (`dx < L_j` and `s_y > 0`) multiplies the longitudinal
   force by `longParReductFactor = 0.2` (`sim_bicycles`; `sim_lanes` uses 0.1).
5. **Two** attenuation widths (`s0y`, `s0yLat`) and **two** anisotropy factors
   (`pushLong`, `pushLat`), not one each. The boundary has two widths too.
6. Lateral interaction is clamped to ±4·b.
7. Neighbour preselection: `|force| > bIntCrit = 0.1`; followers only if their
   gap is below `smaxfollow = 0.5 m`.
8. Lateral push by followers is zeroed when the lateral gap is already opening.
9. **Leader too close to a road edge:** if vehicles overlap laterally and the
   leader leaves less than `W_subject + s0yLatB` on one side, the lateral
   desire `α` is forced to ±1 towards the open side. Not in the paper.
10. Boundary: the lateral force grows linearly past the edge, is scaled by
    `0.2 + 0.8·v/v0max`, and has two attenuation widths.
11. **Constant lateral bias per vehicle type** (`accBiasLat`, road.js). The JS
    counts right as positive; this code counts left as positive. Here it is a
    fitted class parameter.
12. **Integration** (`road.updateSpeedPositions`): speeds update first; lateral
    speed is limited by the heading angle `dvdu = vy/vx`, itself rate-limited
    (`dotdvdumax = 0.3`) and capped (`dvdumax = 0.3`); positions use the
    trapezoid of old and new speed.
13. **Exponent `delta`** of the IDM is a parameter (default 4), so that the
    fixed-IDM runs can use calibrated values.

**Not ported** (not applicable to lane-free data, or would make the objective
stochastic): the floor field (lanes), `bikesNoLeftSpace`, acceleration noise,
obstacles and traffic lights. The exit stop line is added separately in
`jointsim.py`, only in the measured stop periods.

Defaults also differ from the paper's Table 1: the JS uses `s0y = 0.10 m` where
Table 1 says 0.3 m. The JS values are used here. Eq. (3) uses τ_y and Eq. (9) a
bare τ; the JS has a single `tauLatOVM` serving both, so this port uses one
parameter.

## Licence

Model code is derived from MTGermany/mixedTraffic and is therefore **GPL-3.0**.
