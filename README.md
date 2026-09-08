# iam_calib — CMA-ES calibration of the Intelligent-Agent Model

Calibrates all IAM parameters (Treiber & Chaudhari, [arXiv:2310.16816](https://arxiv.org/abs/2310.16816))
against lane-free trajectory data, per vehicle, then pools by vehicle class.

The model code is a NumPy/Numba port of Treiber's reference JavaScript
implementation, [MTGermany/mixedTraffic](https://github.com/MTGermany/mixedTraffic) (GPL-3.0),
**not** a transcription of the paper's equations. They differ — see
[Paper vs. code](#paper-vs-code).

---

## Quick start

```bash
pip install -r requirements.txt

# smoke test first — always
python run_calibration.py D3F1_lcl_smtF.csv --out results/smoke \
    --max-vehicles 8 --maxfevals 400 --workers 4

# full run
python run_calibration.py D3F1_lcl_smtF.csv --out results/full --workers 8
```

Expected runtime for ~800 vehicles at 3000 evaluations each: **40–60 min on 8 cores.**

### Outputs

| File | Contents |
|---|---|
| `class_summary.csv` | rows, vehicles, durations, speeds, lateral extent per class |
| `per_vehicle_params.csv` | one row per vehicle: 11 parameters, fitness, GoF components |
| `class_params.csv` / `.json` | pooled class-level parameters with IQR, after shrinkage |
| `bound_hits.csv` | fraction of vehicles pinned to each parameter bound |
| `identifiability.csv` | local sensitivity screen with an `UNIDENTIFIED`/`weak`/`identified` verdict |
| `validation.csv` | held-out fit using class parameters — **this is the number to report** |

---

## Before you run it

Three things must be set correctly or the results are meaningless.

**1. Vehicle dimensions.** `config.DEFAULT_VEHICLE_DIMS` maps your `Class`
values to (length, width). Edit it to match your site's measured dimensions.
These are deliberately **not** calibrated: letting L and W float lets them
absorb the physics of `s0y` and `sensLat`, and you get a good fit with
meaningless parameters.

**2. The X convention.** The model treats the longitudinal coordinate as the
**front bumper** (the JS uses `u` = front). Most trajectory datasets give the
centroid, so `DataConfig.x_is_front=False` adds L/2. If your `X_Smt` is already
the front bumper, set it to `True`.

**3. Road boundaries.** Off by default. Boundary forces need the road edges in
the same frame as `Y_Smt`; a wrong width injects a spurious lateral force
everywhere and silently corrupts every lateral parameter. Enable with
`--road-width W` (centred on the median of `Y_Smt`) only once you have
confirmed the geometry, which also adds `accLongBRef`, `accLatBRef`, `s0yB`,
`s0yLatB` to the free set.

---

## How it works

For each vehicle: take the longest contiguous run, truncate to 30 s, and
integrate the ego's IAM dynamics forward while **replaying all neighbours from
the data**. CMA-ES searches the 11 free parameters in a normalised `[0, 10]`
box (log-scaled for widths and times), followed by a Nelder–Mead polish.
Vehicles are farmed out to a process pool.

### The objective

```
F = w_x · RMSE(x)/x_scale  +  w_y · RMSE(y)/y_scale  +  w_v · RMSE(vx)/RMS(vx_obs)
```

Defaults: `x_scale = 2.0 m`, `y_scale = 0.5 m`, all weights 1.0.

**The scale ratio is the single most consequential choice in the whole
pipeline.** It sets the exchange rate between a metre of longitudinal error and
a metre of lateral error. Change it deliberately and report it.

Two design decisions here were forced by measurement, not taste:

- **Longitudinal error is normalised by a fixed metre scale, not by distance
  travelled.** Dividing by distance travelled (~250 m over 30 s) makes a 10 m
  spacing error look like 0.04 and effectively deletes the longitudinal signal.
  Measured at the true parameters, that choice left the objective 95% lateral.

- **The ego state is re-initialised from data every 3 s** (`--reset-s`).
  Lane-free lateral position is only weakly determined: with the *true*
  parameters and realistic acceleration noise, a free-running 15 s simulation
  still accumulates ~1.4 m of lateral error, because lateral motion
  random-walks. Free-running, the objective is dominated by noise the model
  cannot reproduce. Resetting keeps the criterion dynamic — genuinely
  multi-step, not one-step-ahead — while capping accumulation.

  | reset interval | e_x | e_y | e_v | lateral RMSE |
  |---|---|---|---|---|
  | none | 2.264 | 5.668 | 0.126 | 2.83 m |
  | 5 s | 0.127 | 1.588 | 0.051 | 0.79 m |
  | **3 s** | **0.041** | **0.886** | **0.034** | **0.44 m** |

  (all at the true parameters, so every number here is irreducible noise)

Integration uses the **ballistic update**, as in the reference implementation.
RK4 buys nothing: the right-hand side isn't smooth, because of the min/max in
Eq. (7), the kink in `min(1, exp(·))` in Eq. (5), and discrete switching of
which neighbour is selected.

### Neighbours

Candidates are fixed once per window from the *measured* ego position using a
generous radius (±80 m, ±8 m), because the simulated ego drifts and must not
lose a neighbour mid-run. Interaction selection then happens against the
*simulated* position at each step, using the interaction-strength threshold
`bIntCrit` from `road.js` — not a geometric rule.

---

## Validating a run

Read the outputs in this order.

1. **`bound_hits.csv`** — parameters pinned to a bound. High rate means the
   bound is too tight, the parameter is unidentified, or the model is
   misspecified for that class. Widening the bound is the *least* likely
   correct answer.
2. **`identifiability.csv`** — anything marked `UNIDENTIFIED` should be fixed at
   its Table 1 value and reported as fixed, not quoted as an estimate.
3. **`validation.csv`** — class parameters on held-out vehicles. In-sample
   per-vehicle fitness is not evidence about the model: an 11-parameter model
   fitted to one 30 s trajectory will fit that trajectory.

The split is **by vehicle, never by timestep** — adjacent frames at 0.04 s are
not independent observations.

### Ground truth test

`make_synthetic.py` generates data from the model with known parameters
(`TRUE_PARAMS`), in your exact CSV schema — the pipeline's ground truth test.
"True" here means only "the vector that generated this file," not "true of
real vehicles": it's a plausible point picked deliberately away from
`config.py`'s own defaults, not measured or drawn from the paper's Table 1.
`CLASS_MIX` spans all 7 real-fleet classes (bike/car/lcv/truck/auto/
multiaxle/trailer); the heavy-class speed/accel multipliers are marked
GUESSED in the code, same caveat as `TRUE_PARAMS` itself.

**Density matters more than it looks.** Vehicles are placed simultaneously
along `[0, --length]` at t=0 — this generator has no vehicles entering or
leaving over time — so `--vehicles / --length` sets the traffic density
directly. Raising `--vehicles` without raising `--length` to match packs
vehicles bumper-to-bumper: at 554 vehicles on the 400 m default, that's 1.4
vehicles/metre, gridlock from the first timestep, and it produces a lateral
"floor" *worse* than real calibrated D3F1 error — backwards for what should
be a best-case number. Match your real corridor's own measured concurrent
density instead (rows / window seconds / corridor extent) — for
`D3F1_lcl_smtF.csv` that's ~0.21 vehicles/m, hence `--length 1430` for 300
vehicles below.

On a 300-vehicle, 7-class, density-matched set (`--length 1430
--road-width 8.7`, matching D3F1's confirmed carriageway):

| | v0 | T | s0 | a | b | s0y | s0yLat | sensLat | tauLatOVM | pushLong | pushLat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| true (car/auto) | 11.0 | 1.10 | 1.40 | 1.30 | 1.80 | 0.220 | 0.350 | 1.60 | 0.90 | 0.120 | 0.150 |
| recovered, median | 11.83 | 1.11 | 1.37 | 1.10 | 1.77 | 0.206 | 0.353 | 1.59 | 0.89 | 0.126 | 0.154 |
| error | 13% | 1% | 2% | 6% | 2% | 6% | 1% | 1% | 2% | 5% | 3% |

Every core parameter now recovers within ~13% — at 45 vehicles, `s0` (77%)
and `pushLat` (57%) looked like structural failures; at 300 vehicles and the
right density they largely disappear. The lesson wasn't "these two are
unidentifiable," it was "45 vehicles wasn't enough data to identify them."
The boundary parameters tell the opposite story and stay genuinely hard: at
this same scale, `accLongBRef` and `s0yB` still come back **62–113% off**
(`accLatBRef`/`s0yLatB` ~20%) — consistent with real D3F1, where the
identifiability screen independently flags the same two as unidentified (see
`iam_calib`'s "Calibrating the IAM" doc, §8). Two parameters that only
jointly matter when a vehicle is close enough to a road edge stay hard to
pin down regardless of fleet size, because that situation itself stays rare.

Held-out validation, same 300-vehicle set, 90 held-out vehicles: **0.30 m
longitudinal / 0.12 m lateral / 0.24 m/s speed**, essentially matching the
0.30/0.08/0.23 floor from simulating the literal `TRUE_PARAMS` directly with
no fitting at all — at this scale, CMA-ES recovers nearly everything
reachable. Applying the real, D3F1-calibrated class parameters to D3F1's own
155 held-out vehicles instead gives 0.73/0.14/0.55 — 2.3–2.4× this floor
longitudinally and in speed, ~1.2× (near parity) laterally. That gap is the
honest answer to "is this a good fit": real driver behaviour and real sensor
noise the synthetic model doesn't reproduce, not a broken pipeline.

Run it yourself:

```bash
python make_synthetic.py --out synthetic.csv --vehicles 300 --duration 60 \
    --length 1430 --road-width 8.7
python -m pytest tests/ -v          # 21 tests
python -m pytest tests/ -v -m slow  # recovery test (minutes)
```

---

## Paper vs. code

`model.py` follows the JS implementation where the two disagree. Documented
divergences, all of which affect calibration:

1. `s_x` is clamped at zero: `sx = max(0, dx - L_j)`. Negative gaps produce
   maximum deceleration via the IDM's `max(s, 0.1·s0)` denominator and the
   `-bmax` floor, not via a negative gap.
2. `α(s_y) = min(exp(−s_y/s0y), 1)` uses **signed** `s_y`, not `|s_y|` as
   printed in Eq. (5). Taking Eq. (5) literally would wrongly attenuate
   overlapping vehicles; the code clamps to 1 instead.
3. Parallel driving (`dx < L_j` and `s_y > 0`) multiplies the longitudinal
   force by `longParReductFactor = 0.2`, rather than zeroing it as the paper says.
4. **Two** attenuation widths (`s0y` longitudinal, `s0yLat` lateral), not one.
5. **Two** anisotropy factors (`pushLong`, `pushLat`), not one λ.
6. Lateral interaction is clamped to ±4·b.
7. Boundary forces in the JS discretise an anticipation horizon over 8 steps;
   this port implements the j = 0 term only. The JS's own comments note the
   complications were largely deactivated with no loss.

Defaults also differ from the paper's Table 1 — the JS uses `s0y = 0.10 m`
where Table 1 says 0.3 m. The JS values are used here.

Φ₀ and the floor field (Eq. 11) are **absent**: lane-free flow has no lanes.

### Ambiguity resolved

Eq. (3) uses τ_y and Eq. (9) uses a bare τ. The JS has a single `tauLatOVM`
serving both, so this port uses one parameter. If you believe they're distinct,
add a second spec in `config.py` and to `FREE_PARAMS`.

---

## Extending

**Staged calibration** (`--staged`): longitudinal parameters first, then lateral
with longitudinal frozen, then joint refinement. The `α(s_y)` factor makes the
two directions nearly decoupled under overlap, which is what makes this work.
Try the joint fit first — it may just work.

**Hierarchical Bayesian.** The right treatment of driver heterogeneity, and
worth it if calibration is the contribution rather than a means to an end. Draw
per-vehicle parameters from per-class hyperpriors and run NUTS (NumPyro/Stan)
on a differentiable reimplementation — replace the min/max in Eq. (7) with
softmin/softmax and smooth `α`. The posterior tells you directly which
parameters the data constrains, instead of leaving you to infer it from bound
hits. `shrink_thin_classes()` is the cheap stand-in: it shrinks thin classes
toward the population median, which beats reporting a bus parameter fitted on
four buses.

**Second calibration level.** Trajectory-level fitting cannot identify the
parameters governing *emergent* structure — `pushLat` and `s0yLat` control the
lateral distributions in the paper's Fig. 5. Fit those by simulating the full
scenario and matching aggregate targets (lateral position histograms, speed–
density relation, staggering statistics) via method of simulated moments or
Bayesian optimisation, after fixing the per-vehicle parameters.

---

## Known limitations

- **Truncated neighbours.** A selected leader leaving the camera footprint
  mid-window causes a discontinuous jump in the interaction force. Windows are
  capped at 30 s to limit this, but it is not detected explicitly. If fit
  quality correlates with proximity to the footprint edge, this is why.
- **One window per vehicle**, so per-vehicle estimates stay independent. Long
  trajectories are truncated rather than split.
- **Local, not global, sensitivity.** `identifiability()` is a one-at-a-time
  screen at each vehicle's own optimum. It catches the usual offenders but is
  not a substitute for Sobol indices.
- **The neighbour radius is a modelling choice**, not an implementation
  detail. Results are sensitive to it. Report it, and check sensitivity.

## Licence

Model code is derived from MTGermany/mixedTraffic and is therefore **GPL-3.0**.
