# Adaptive Suspension: Algorithms and Evidence

How the software works, why each part is designed the way it is, and what the
simulation shows. Every number below comes from simulated drives
(`docs/simulation_results.txt`, `docs/model_report.md`; figures in `docs/figures/`). They show the design
works as intended on a physically consistent model; they are **not** measurements
of the real cart, and every threshold marked CALIBRATE in `config.h` has to be
re-derived from recordings once the hardware exists.

---

## 1. Pipeline

```
MPU6050 (100 Hz) ──► 1 s rolling buffer ──► 23 features ──► int8 neural ensemble ──► class probabilities
      │                                       (every 30 ms)    (+ uncertainty)          + confidence, disagreement, novelty
      │                                                                                       │
      ├──► impact reflex (one-sample jerk) ──────────────────────────────┐                    ▼
      │                                                                 │     confidence → damping curve ──► actuator shaper ──► servo
      └──► Extended Kalman Filter ──► attitude, vertical velocity ──────┼──────────► (skyhook term)            (100 Hz)
                ▲ trust in accelerometer set by class probabilities     │
                                                                        │
HC-SR04 (25 Hz) ──► obstacle monitor (hard safety) ──► range tracker ──► interval Type-2 fuzzy speed ──► motors
                                                                        │       ▲ roughness = 1 − P(smooth)
                         state machine: IDLE / RUNNING / APPROACH / ESTOP / FAULT / MAINT
                         predictive maintenance: reference-bump damping test → EWMA + CUSUM drift
```

All control code (`adaptive_suspension/src/core/`) is hardware-independent C++11.
It talks to the hardware only through `hal.h`, so the same code runs in the
simulator now and on the ESP32 later.

---

## 2. Road-surface classification model

### Features (`ml_features.cpp`)
Features 0–15 are computed over the last 1 s (100 samples) of d = |a| − mean|a|.
Features 16–20 use only the last 0.25 s, so a fresh impact is not diluted by the
second of road before it. Using the acceleration magnitude makes all of them
independent of how the sensor is mounted.

| # | Feature | Captures |
|---|---|---|
| 0–3 | log RMS, peak-to-peak, largest rise, largest drop | vibration level; a pothole drops before it rises |
| 4–5 | log max / RMS jerk (m/s³) | sharp impacts |
| 6 | swing count across ±0.3 m/s² | sustained vibration vs. a single event |
| 7–8 | crest factor, log kurtosis | impulsive vs. steady |
| 9–13 | **FFT band energies** 1–2, 3–5, 6–10, 11–20, 21–49 Hz | frequency signature (Hann-windowed DFT) |
| 14–15 | log gyro std, largest gyro rise | rotation from uneven ground |
| 16–20 | last 0.25 s: log RMS, peak-to-peak, rise, drop, max jerk | a fresh impact, seen immediately |
| 21 | autocorrelation peak (lags 30–400 ms) | how *regular* the vibration is: a rumble strip or washboard repeats, gravel does not |
| 22 | spectral flatness | broadband noise (gravel) vs. a few strong tones (repeating ridges) |

A sixth extra input, the drive speed (PWM), was tested and dropped. It gave no gain
on a different cart (below), and a cart pushed by hand would report speed 0.

### Training data: built for robustness
The biggest risk is the jump from simulation to the real cart. The training set
therefore uses **domain randomisation** (`sim dataset --domain robust --augment`).
Every drive gets a random cart and road:
- weight ×0.7–1.4, springs ×0.8–1.25, dampers ×0.8–1.2, shock wear 50–100%;
- sensor noise, bias and digital filter setting (10/21/44 Hz), mounting tilt up to 8°, drivetrain vibration;
- speeds 0.12–0.65 m/s, gravel stone sizes, gravel roughness 0.8–3.5 mm, bumps and potholes 0.8–3.2 cm.

On top of that, about half the windows get an **augmented copy**: the vibration and
rotation are scaled ×0.7–1.35 and extra sensor noise is added. Augmented copies are
used for training only, never for validation or testing.

The training drives also contain **"other" road** (`--other`): cobblestones and
washboard ripples, labelled as their own category. The model is not taught to name
them. It is taught to be *unsure* on them (outlier exposure, below), so that road it
has never seen is more likely to be flagged. The rumble strip used to test this is
never in the training data.

Robustness is measured on a **"different cart"** that no model trains on
(`--domain shifted`): 35–50% heavier, softer springs, a noisier sensor on another
filter setting, mounted 10° crooked, faster, bigger stones and taller bumps. Several
of these are outside the training ranges.

### Model (`ml_model.cpp`, trained by `tools/train_torch.py` or `host/train`)
Two independently written trainers produce the same artefacts:
- `tools/train_torch.py` (PyTorch) is the readable reference and the source of the deployed model.
- `host/train` (dependency-free C++) is the alternative when no Python packages are available.

Both reach the same accuracy on the same data (99.0% for both, macro-F1 0.954 for
both on the robust test runs; unseen-surface AUROC 0.871 vs 0.863), and the same parity test checks both exports against
the on-device int8 code. Two separate
implementations agreeing is an independent check of the quantization pipeline.

- 5 multilayer perceptrons, 23 → 16 → 16 → 4, ReLU.
- Each is trained on a bootstrap resample with class-weighted cross-entropy (Adam, 30 epochs).
- **Outlier exposure:** on "other" road windows the target is the uniform distribution (equal probability on all four surfaces), weight 0.5 (`--oe`). This teaches the members to spread their answer, and to disagree, on unfamiliar road.
- **Quantization-aware training (#6):** after float training, the int8 input range of each layer is calibrated (99.99th percentile of |activation|). Then 12 epochs fine-tune with weights and activations rounded to int8 in the forward pass, and gradients pass straight through (straight-through estimator) except where values were clipped.
- **Deployed arithmetic:** int8 weights, int8 activations, int32 accumulation, one float rescale per neuron. Weights total 3,440 bytes; the novelty model adds 8.5 kB of floats. The classifier runs every 30 ms.
- A parity test (`test_ml_model.cpp`) checks that the on-device int8 code reproduces the trainer's own fake-quantized forward pass (PyTorch autograd for the deployed model) on real test windows.

### Results: what made it better (ablation)
All five models are 5-member int8 QAT ensembles. Both test sets are separate
datasets that none of them trained on.

| Model | Normal carts: acc. / F1 | Different cart: acc. / F1 | Different cart: breakers caught | Different cart: flagged unsure |
|---|---|---|---|---|
| A. Previous: nominal data, 16 features | 99.4% / 0.973 | 94.9% / 0.840 | 59% | 20.2% |
| B. + randomised data and augmentation | 99.4% / 0.977 | 97.0% / 0.908 | 78% | 8.5% |
| C. + last-0.25 s features | 99.7% / 0.985 | 97.7% / 0.918 | 76% | 8.7% |
| D. + drive speed input (dropped) | 99.7% / 0.985 | 97.6% / 0.911 | 72% | 9.3% |
| **E. C + repetition features + outlier exposure (deployed)** | **99.7% / 0.983** | **97.9% / 0.928** | **82%** | 7.4% |
| Rule-based, for reference | 84.1% / 0.754 | 38.9% / 0.455 | 26% | – |

- **Randomised training is the big win for robustness:** different-cart F1 0.840 → 0.908, and caught speed breakers 59% → 78%.
- **The short-window features** add accuracy on both test sets, and above all speed (see "Detection speed" below).
- **The speed input** adds nothing, so it was dropped.
- **Repetition features and outlier exposure** were added for unseen-surface detection (section 3); they also help a little on the different cart.
- **The fixed rules collapse on the different cart** (38.9%): hand-set thresholds don't transfer, which is the strongest argument for the learned model.

On its own (harder, randomised) test runs the deployed model scores 99.0%
(macro-F1 0.954). In closed loop on unseen nominal tracks it is correct on 99.9% of
windows; on the different cart, 97.1%.

**Missed breakers are not missed impacts.** Most of the different cart's "missed"
breakers (82% recall) are called *pothole*, and both get full stiffness. Counting a
breaker or pothole recognised as either one, **impact recall is 99.7%** on both the
normal and the different carts (rules: 93.0% / 85.0%).

| Same model, offline, randomised test runs | Accuracy | Macro-F1 |
|---|---|---|
| Rule-based | 77.0% | 0.635 |
| 1 MLP float32 | 99.0% | 0.953 |
| 1 MLP int8, post-training quantization | 98.9% | 0.952 |
| 1 MLP int8, QAT | 99.0% | 0.953 |
| **Ensemble int8 QAT (deployed)** | **99.0%** | **0.954** |

### Detection speed
Time from the front wheel reaching a breaker or pothole to the classifier putting
more than 50% on "impact" (closed loop, 73 events):

| Version | Average | 90th percentile | Servo physically stiff (≥140°) |
|---|---|---|---|
| Previous (1 s features, every 100 ms) | 89 ms | 129 ms | 195 ms |
| New features, still every 100 ms | 67 ms | 107 ms | 173 ms |
| **New features, every 30 ms (deployed)** | **27 ms** | **37 ms** | **131 ms** |

To avoid extra servo wear from the faster updates, only *stiffening* reacts at the
30 ms rate; softening still updates at 10 Hz. Servo wear is covered in section 4.

**Honest note on QAT:** at this model size, plain post-training quantization already
loses nothing, so QAT gives no measurable gain here. It is kept because it costs
nothing at run time, and it matters if the model grows or real data is noisier.

---

## 3. Uncertainty and safety fallback (#8)

Two complementary estimates, computed in `mlPredict`:

1. **Model uncertainty: deep ensemble** (approximate Bayesian model averaging).
   - p = mean of softmax(z_m / T) over the 5 members.
   - T is fitted on the validation runs (temperature scaling).
   - Mutual information MI = H(p) − mean_m H(p_m) measures how much the members disagree.
2. **Data novelty:** a Gaussian per surface, **each with its own covariance**, fitted on the training features.
   - novelty = min over surfaces of the squared Mahalanobis distance.
   - It is high when the input looks like none of the trained surfaces.
   - A separate covariance per surface fits each surface's own shape much more tightly than one shared covariance (gravel varies in completely different directions from a pothole). Cost: 4 × 23 × 23 floats = 8.5 kB of flash instead of 2 kB.

A window is *uncertain* when confidence < 0.824, MI > 0.0533 or novelty > 88.2. These
thresholds come from validation data (the 2.5th / 97.5th / 99th percentiles of
correct windows). Uncertainty is not only a flag: it feeds the damping curve
continuously (section 4) and the fuzzy speed controller (section 6).

| Result (offline test set) | Value |
|---|---|
| Coverage (windows accepted) | 95.5% |
| Accuracy on accepted / flagged windows | 99.8% / 82.0% |
| Transition windows flagged | 49.3% |
| Different cart: windows flagged | 7.4% (first model: 20.2%) |
| Unseen surface (rumble strip) flagged | 7.6% |
| Unseen-surface detection AUROC, deployed rule / single model | 0.87 / 0.63 |

### Detecting road it has never seen
Robust training had made this worse: after seeing a wide range of carts and roads,
a rumble strip no longer looked unusual (AUROC 0.78 → 0.60). Three changes, each
measured on the same unseen rumble strip in three test sets:

| Version | Own test runs | Normal carts | Different cart |
|---|---|---|---|
| Robust model, 21 features, shared covariance | 0.60 | 0.66 | 0.75 |
| + repetition features (autocorrelation, spectral flatness) | 0.78 | 0.82 | 0.82 |
| + outlier exposure on cobbles / washboard | 0.80 | 0.84 | 0.83 |
| **+ per-surface covariance (deployed)** | **0.87** | **0.90** | **0.88** |

(AUROC: 0.5 = no better than chance, 1.0 = perfect.) None of the three cost any
classification accuracy (section 2, row E).

**Still limited:** the flag threshold is set so only 1% of normal windows trip the
novelty check, and at that setting only 6–8% of rumble-strip windows are flagged
offline (1.3% on the different cart, 0% in the closed-loop test). The model now
*ranks* unfamiliar road as more unusual, but most of it still looks enough like gravel
to pass. In closed loop the rumble strip gets soft, gravel-like damping (about 50°),
which is the comfortable choice for that kind of vibration, so the practical risk is
low. Lowering the threshold would flag more of it, but also more normal road, and every
flag moves the damping toward the middle setting. The accepted/flagged split
(99.8% vs 82.0% accuracy) shows the uncertainty signals still separate right from
wrong answers well.

---

## 4. Confidence → graded damping curve (core contribution)

`response_curve.cpp`. The classifier output becomes a **continuous** stiffness
s ∈ [0, 1] (servo 45°…150°), not an on/off switch or a fixed angle per class.

```
s_exp  = Σ_c p(c) · S[c]                                   expected stiffness under the posterior
c_eff  = confidence · 2^(−MI / MI_thr) · min(1, nov_thr / novelty)   trust, lowered by disagreement and novelty
g      = smoothstep((c_eff − 0.55) / (0.90 − 0.55))          how much the classification is trusted
s_base = S_safe + g · (s_exp − S_safe)                       blend toward a safe middle setting (S_safe = 0.43, 90°)
s_risk = 0.9 · (p(breaker) + p(pothole))                     protective floor: stiffen even when unsure
s_sky  = 0.15 · (1 − p(gravel)) · min(|v_z| / 0.05 m/s, 1)   extra damping when the EKF sees the body bounce, except on gravel
s      = clamp(max(s_base, s_risk) + s_sky, 0, 1)
```

Figure: `docs/figures/response_curve.png`.

**Per-class targets S[c] were measured, not guessed** (`sim tune`; figure
`docs/figures/damping_characterisation.png`). The servo was
locked at six settings on each surface, and comfort and protection were measured:

| Surface | soft 45° | … | stiff 150° | Best |
|---|---|---|---|---|
| smooth (cost) | 0.085 | 0.049 from 108° up | 0.049 | **0.6** (start of the plateau) |
| gravel | **1.17** | … | 6.45 | **0.0**: soft isolates stone vibration |
| breaker | 8.43 | … | **0.59** | **1.0**: stiff prevents bottoming out |
| pothole | 17.8 | … | **2.85** | **1.0** |

This contradicts the original SRA mapping, which put gravel at a medium 110°.

**Why the body-motion term is switched off on gravel.** On gravel the body vibrates
all the time, so the skyhook term kept adding stiffness, which is exactly what makes
gravel worse (table above), and it kept the servo moving. Weighting it by
(1 − p(gravel)) cut servo movement by 34% and gravel vibration by 13% on the normal
carts, with no change in protection. Switching it off only halfway was worse than
either extreme.

**Actuator shaping, tuned to the servo's response time** (`ActuatorShaper`):
- stiffening uses the MG90S's full loaded slew rate (300 °/s), so protection is never held back;
- softening is limited to 60 °/s, so a brief misclassification can't pump the suspension;
- a 3° deadband suppresses jitter and wear (2.5% less servo movement than 2°, no measurable effect on the ride);
- the **impact reflex** jumps straight to full stiffness on one very sharp sample (jerk > 3000 m/s³, set above gravel's 99th percentile), without waiting for the classifier;
- stiffening follows the classifier every 30 ms; softening only every 100 ms, so the servo does not chase small changes.

### Closed-loop evidence (8 unseen tracks, same cars and speed for every strategy)
Figure: `docs/figures/ride_comparison.png`.

| Strategy | Gravel RMS | Event peak | Bottom-outs / event | Servo °/m | Missed potholes |
|---|---|---|---|---|---|
| Passive soft (45°) | **1.42** | 47.7 | 8.68 | 1 | – |
| Passive stiff (150°) | 7.32 | **10.8** | **1.05** | 1 | – |
| SRA: rules + fixed angles | 5.19 | 33.3 | 2.90 | 80 | 0 |
| Neural + fixed angles | 4.98 | 34.1 | 3.10 | 64 | **2** |
| **Neural + graded curve** | 1.83 | 12.6 | 1.32 | **43** | **0** |

- **Against the SRA design:** 65% less gravel vibration, 62% lower peak jolt at breakers and potholes, 54% fewer bottom-outs, 47% less servo movement.
- **Against either passive setting:** no single fixed setting wins on both comfort and protection. The adaptive system gets close to the best of each: near-stiff protection with much better gravel comfort than stiff.
- **Neural classifier on fixed angles missed 2 potholes** (16 with the first model). It falls back to neutral whenever it is unsure. The curve's protective floor is what removes those misses, which is direct evidence for the graded mapping.

**On the different cart** (closed loop, same comparison):

| Strategy | Gravel RMS | Event peak | Bottom-outs / event | Classifier correct |
|---|---|---|---|---|
| Passive stiff (150°) | 6.30 | **28.9** | **4.48** | – |
| SRA: rules + fixed angles | 4.60 | 36.2 | 5.26 | 87.7% |
| **Neural + graded curve** | **3.31** | 32.9 | 5.79 | **97.1%** |

The classifier transfers well (97.1% correct), and comfort stays clearly better,
and the servo moves no more than in the SRA design (43 °/m each).
Bottom-outs in total are slightly *worse* than the SRA design (5.79 vs 5.26 per
event), but splitting them by where they happen shows that protection is not the
problem:

| Bottom-outs per event, different cart | At breakers/potholes | Elsewhere (on gravel) |
|---|---|---|
| Passive stiff (150°) | 4.12 | 0.36 |
| SRA: rules + fixed angles | 4.68 | 0.58 |
| **Neural + graded curve** | **4.26** | 1.53 |

At breakers and potholes the deployed design is better than the SRA design. The
extra ones are on gravel. This cart is heavier, with softer springs and partly worn
shocks, and on gravel the fully soft setting occasionally uses up its travel and touches
the bump stop. These touches are light: the body acceleration when they happen (median
4.2 m/s²) is below what ordinary gravel vibration reaches (7.8 m/s² at the 99th
percentile), and gravel comfort is still 28% better than the SRA design. On the
nominal cart there are none.

*Tried and not adopted:*
- A gravel stiffness margin. 0.1 made gravel 21% rougher on the normal cart and did
  not reduce bottom-outs. 0.2 cut them by only 3%, at 45% more vibration.
- An adaptive version that stiffens when it feels bump-stop hits. The IMU cannot tell
  these light touches from normal gravel vibration, so there is nothing to adapt on.

The `sim tune` targets were measured with healthy shocks; repeat it on the real cart,
including with worn shocks, before changing them.

**Tried and removed: automatic on-cart tuning.** A `tune` command was built that
drove the cart over its own road, tried several smooth-road settings and kept the
one with the fewest bottom-outs (estimated from the IMU). It did not improve the
different cart, and on the normal cart the results were mixed, so it was removed. Two
reasons were found:
- The smooth-road setting is also the cart's "readiness" when a bump arrives
  unannounced, so tuning it for comfort alone trades away protection.
- The IMU bottom-out estimate was confused by ordinary hard hits at full stiffness.

Retuning the targets is better done on the bench with the real cart, following
`sim tune`.

**Limitation:** the system reacts; it cannot see a pothole before the wheel hits
it. The front wheel's impact is absorbed at whatever setting the suspension was
in. Average time to physically reach ≥140° after the front wheel hits is about
131 ms: about 27 ms to detect, the rest servo travel. The rear wheel is the one that
benefits.

---

## 5. Extended Kalman Filter (#2)

`ekf.cpp`. The state is roll, pitch, gyro biases x/y, and vertical velocity.

- **Predict (100 Hz):** bias-corrected gyro rates go through the nonlinear Euler kinematics; gravity-removed vertical acceleration is integrated with a 0.5 /s leak.
- **Update:**
  1. The accelerometer is treated as a gravity measurement, h(x) = g · [−sin θ, sin φ cos θ, cos φ cos θ], with R inflated when |a| ≠ 1 g.
  2. A weak "mean vertical velocity is zero" pseudo-measurement keeps the velocity from drifting.
- Joseph-form covariance update.
- Analytic Jacobians, checked against numerical derivatives in the tests.

**Classifier link:** the gravity-update noise is R × Σ p(c) · factor[c], with factors smooth 1, gravel 2, breaker 4, pothole 8. The filter trusts the accelerometer less when the classifier says the reading is an impact.

**Results:**
- Attitude error is about 0.5° against 3.9–4.5° for accelerometer tilt alone.
- The vertical velocity used by the curve is accurate to 0.015 m/s.

**Honest finding:**
- **Strong classifier-driven multipliers (30–100×) made things worse.** Gravel vibration averages out, so ignoring the accelerometer throws away good data. The mild multipliers above are harmless, but the simulation shows **no measurable gain** from the classifier link.
- **Attitude accuracy is limited by accelerometer bias** (0.08 m/s² ≈ 0.47°), not by vibration. Calibrating the MPU6050 offsets on the cart matters more than any filter tuning.

---

## 6. Interval Type-2 fuzzy speed control (#5)

`fuzzy_speed.cpp`.
- **Inputs:** obstacle distance, closing speed (from an alpha-beta filter on echoes) and roughness = 1 − P(smooth). Because uncertainty spreads probability away from "smooth", **an unsure classifier also slows the cart**.
- **Rules:** 12. Every membership function has a footprint of uncertainty (an upper and a lower curve).
- **Type reduction:** exact Karnik–Mendel center-of-sets, found by trying every switch point. Output u = (y_l + y_r) / 2.
- **Fallback:** Type-1 uses only the middle curves.

The fuzzy controller only shapes speed *inside* the safety envelope. The hard
< 10 cm emergency brake, the sensor-fault stop and the no-echo slow-down always win.
While approaching an obstacle, speed may only decrease unless the obstacle moves
more than 10 cm away. This stops noisy echoes from causing re-acceleration.

Figure: `docs/figures/obstacle_approach.png`.

| Wall approach, 20 runs, noisy sonar | SRA steps | Type-1 | Type-2 |
|---|---|---|---|
| 0.45 m/s: worst gap / hard emergency stops | 5.8 cm / 20 | 15.0 cm / 0 | 11.8 cm / 3 |
| 0.70 m/s | 2.6 cm / 20 | 16.2 cm / 0 | 12.6 cm / 1 |
| 0.90 m/s | **hit the wall** / 20 | 14.3 cm / 0 | 15.6 cm / 0 |

- **Both fuzzy controllers stop gently and never hit the wall**; the SRA steps always end in an emergency stop and collide at 0.9 m/s.
- **Honest finding:** in this simulation, **Type-2 is no better than Type-1**. It stops slightly closer and its speed command is a little less steady. It is the default because the project specification asks for it. `speed t1` switches modes, and the two should be compared on the real cart.

---

## 7. State machine

`supervisor.h`. States are IDLE, RUNNING, **APPROACH** (obstacle within 60 cm, released at 70 cm), ESTOP, FAULT and MAINT.

- The cart never starts on its own.
- Faults latch until the operator restarts the cart.
- After an emergency stop the cart resumes when the path is clear, except after a maintenance test, which returns to IDLE.
- The earlier project's fall/theft/crash states can be added as further states, with a branch each in `update()`.

Simulated fail-safe checks, all passing:

| Check | Result |
|---|---|
| IMU unplugged | stops in 39 ms |
| Ultrasonic sensor unplugged | stops in 109 ms |
| Object closer than 2 cm | emergency stop |
| 10% dropped echoes | no false fault |
| IMU replugged | stays faulted until `start` |

---

## 8. Predictive maintenance

`maintenance.cpp`, run with `maint test`.

**The test:**
1. The servo is locked soft and the cart drives slowly over a reference bump (1.5 cm).
2. 1.5 s of gravity-free vertical acceleration (from the EKF) is recorded.
3. The damping ratio ζ is estimated from the free decay: peaks of alternating sign after the forced part, a least-squares fit of ln|peak| against time (decay rate σ), and the peak spacing (damped frequency ω_d). Then ζ = σ / √(σ² + ω_d²).

**Wear detection:**
- A baseline is learned from the first 5 tests.
- Drift is tracked with EWMA (λ = 0.3) and a one-sided CUSUM on (baseline − ζ)/σ.
- **WATCH:** CUSUM > 3 or a 15% drop. **SERVICE:** CUSUM > 6 or a 30% drop.
- The record is stored in flash (IStorage).

**Result** (figure `docs/figures/maintenance_drift.png`): with wear of 2% per day starting on day 10, WATCH is raised at 74%
damping and SERVICE at 60%. Healthy shocks produced 0 false alarms in 40 days, even
though single tests scatter widely (ζ 0.077–0.144 when healthy). That scatter is
why the drift detectors are needed.

---

## 9. What still needs the hardware

- Running the ESP32 firmware on the cart for the first time. The drivers (MPU-6050, HC-SR04, servo, L298N, NVS storage, watchdog), the pin map, the Arduino sketch and the WiFi dashboard are written for the agreed wiring but untested on hardware (`docs/HARDWARE_BRINGUP.md`).
- Real recordings, then `replay --label`, then `train`, `calibrate.py` and `sim tune` repeated against bench measurements.
- Measuring top speed, braking and servo slew, then re-checking obstacle stopping distance.
- Accelerometer offset calibration.
- The hardware acceptance tests in BRD Section 10.
