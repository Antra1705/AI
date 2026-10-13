# Adaptive Suspension: Algorithms and Evidence

How the software works, why each part is designed the way it is, and what the
simulation shows. Every number below comes from simulated drives
(`docs/simulation_results.txt`, `docs/model_report.md`). They show the design
works as intended on a physically consistent model; they are **not** measurements
of the real cart, and every threshold marked CALIBRATE in `config.h` has to be
re-derived from recordings once the hardware exists.

---

## 1. Pipeline

```
MPU6050 (100 Hz) ──► 1 s rolling buffer ──► 16 features ──► int8 neural ensemble ──► class probabilities
      │                                         (10 Hz)        (+ uncertainty)          + confidence, disagreement, novelty
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
They are computed over the last 1 s (100 samples) of d = |a| − mean|a|. Using the
acceleration magnitude makes them independent of how the sensor is mounted.

| # | Feature | Captures |
|---|---|---|
| 0–3 | log RMS, peak-to-peak, largest rise, largest drop | vibration level; a pothole drops before it rises |
| 4–5 | log max / RMS jerk (m/s³) | sharp impacts |
| 6 | swing count across ±0.3 m/s² | sustained vibration vs. a single event |
| 7–8 | crest factor, log kurtosis | impulsive vs. steady |
| 9–13 | **FFT band energies** 1–2, 3–5, 6–10, 11–20, 21–49 Hz | frequency signature (Hann-windowed DFT) |
| 14–15 | log gyro std, largest gyro rise | rotation from uneven ground |

### Model (`ml_model.cpp`, trained by `host/train`)
- 5 multilayer perceptrons, 16 → 16 → 16 → 4, ReLU.
- Each is trained on a bootstrap resample with class-weighted cross-entropy (Adam, 30 epochs).
- **Quantization-aware training (#6):** after float training, the int8 input range of each layer is calibrated (99.99th percentile of |activation|). Then 12 epochs fine-tune with weights and activations rounded to int8 in the forward pass, and gradients pass straight through (straight-through estimator) except where values were clipped.
- **Deployed arithmetic:** int8 weights, int8 activations, int32 accumulation, one float rescale per neuron. Weights total 2,880 bytes.
- A parity test (`test_ml_model.cpp`) checks that the on-device int8 code reproduces the trainer's independently written fake-quantized forward pass on real test windows.

### Results (test runs never seen in training)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Rule-based, thresholds tuned on the same data | 83.9% | 0.725 |
| 1 MLP float32 | 99.1% | 0.957 |
| 1 MLP int8, post-training quantization | 99.1% | 0.957 |
| 1 MLP int8, QAT | 99.1% | 0.956 |
| **Ensemble int8 QAT (deployed)** | **99.3%** | **0.968** |

In closed loop on unseen tracks, the classifier is correct on 98.6% of windows.

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
2. **Data novelty:** class-conditional Gaussian with shared covariance, fitted on the training features.
   - novelty = min over classes of the squared Mahalanobis distance.
   - It is high when the input looks like none of the trained surfaces.

A window is *uncertain* when confidence < 0.983, MI > 0.0086 or novelty > 62.9. These
thresholds come from validation data (the 2.5th / 97.5th / 99th percentiles of
correct windows). Uncertainty is not only a flag: it feeds the damping curve
continuously (section 4) and the fuzzy speed controller (section 6).

| Result (offline test set) | Value |
|---|---|
| Coverage (windows accepted) | 95.0% |
| Accuracy on accepted / flagged windows | 99.9% / 87.9% |
| Transition windows flagged | 50.8% |
| Unseen surface (rumble strip) flagged | 12.2% (single model + confidence gate: 4.2%) |
| Unseen-surface detection AUROC, deployed rule / single model | 0.78 / 0.66 |

**Limitation:** detection of a genuinely new surface is weak. In closed loop the
rumble strip is mostly treated as gravel, so the suspension goes soft, which is a
benign response for that kind of vibration. The ensemble and novelty model are
clearly better than the simplified single-model gate (AUROC 0.78 vs 0.66), but this
is not a guarantee of detecting every new surface.

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
s_sky  = 0.15 · min(|v_z| / 0.05 m/s, 1)                     extra damping when the EKF sees the body bounce
s      = clamp(max(s_base, s_risk) + s_sky, 0, 1)
```

**Per-class targets S[c] were measured, not guessed** (`sim tune`). The servo was
locked at six settings on each surface, and comfort and protection were measured:

| Surface | soft 45° | … | stiff 150° | Best |
|---|---|---|---|---|
| smooth (cost) | 0.085 | 0.049 from 108° up | 0.049 | **0.6** (start of the plateau) |
| gravel | **1.17** | … | 6.45 | **0.0**: soft isolates stone vibration |
| breaker | 8.43 | … | **0.59** | **1.0**: stiff prevents bottoming out |
| pothole | 17.8 | … | **2.85** | **1.0** |

This contradicts the original SRA mapping, which put gravel at a medium 110°.

**Actuator shaping, tuned to the servo's response time** (`ActuatorShaper`):
- stiffening uses the MG90S's full loaded slew rate (300 °/s), so protection is never held back;
- softening is limited to 60 °/s, so a brief misclassification can't pump the suspension;
- a 2° deadband suppresses jitter and wear;
- the **impact reflex** jumps straight to full stiffness on one very sharp sample (jerk > 3000 m/s³, set above gravel's 99th percentile), without waiting for the 10 Hz classifier.

### Closed-loop evidence (8 unseen tracks, same cars and speed for every strategy)
| Strategy | Gravel RMS | Event peak | Bottom-outs / event | Servo °/m | Missed potholes |
|---|---|---|---|---|---|
| Passive soft (45°) | **1.42** | 47.7 | 8.68 | 1 | – |
| Passive stiff (150°) | 7.32 | **10.8** | **1.05** | 1 | – |
| SRA: rules + fixed angles | 5.18 | 31.8 | 2.79 | 80 | 0 |
| Neural + fixed angles | 4.96 | 32.6 | 3.27 | 68 | **13** |
| **Neural + graded curve** | 2.18 | 13.5 | 1.38 | 63 | **0** |

- **Against the SRA design:** 58% less gravel vibration, 57% lower peak jolt at breakers and potholes, 51% fewer bottom-outs, 21% less servo movement.
- **Against either passive setting:** no single fixed setting wins on both comfort and protection. The adaptive system gets close to the best of each: near-stiff protection with much better gravel comfort than stiff.
- **Neural classifier on fixed angles missed 13 potholes.** It falls back to neutral whenever it is unsure. The curve's protective floor is what removes those misses, which is direct evidence for the graded mapping.

**Limitation:** the system reacts; it cannot see a pothole before the wheel hits
it. The front wheel's impact is absorbed at whatever setting the suspension was
in. Average time to physically reach ≥140° after the front wheel hits is about
200 ms, which is mostly servo travel. The rear wheel is the one that benefits.

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

| Wall approach, 20 runs, noisy sonar | SRA steps | Type-1 | Type-2 |
|---|---|---|---|
| 0.45 m/s: worst gap / hard emergency stops | 5.8 cm / 20 | 15.0 cm / 0 | 11.8 cm / 3 |
| 0.70 m/s | 2.6 cm / 20 | 16.2 cm / 0 | 12.6 cm / 1 |
| 0.90 m/s | **hit the wall** / 20 | 14.2 cm / 0 | 14.2 cm / 0 |

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

**Result:** with wear of 2% per day starting on day 10, WATCH is raised at 74%
damping and SERVICE at 60%. Healthy shocks produced 0 false alarms in 40 days, even
though single tests scatter widely (ζ 0.077–0.144 when healthy). That scatter is
why the drift detectors are needed.

---

## 9. What still needs the hardware

- ESP32 drivers (MPU6050, HC-SR04, servo, L298N, NVS storage, watchdog), the pin map and the Arduino sketch.
- WiFi serving of the dashboard: the same page, with `/events` and `/cmd` on the ESP32.
- Real recordings, then `replay --label`, then `train`, `calibrate.py` and `sim tune` repeated against bench measurements.
- Measuring top speed, braking and servo slew, then re-checking obstacle stopping distance.
- Accelerometer offset calibration.
- The hardware acceptance tests in BRD Section 10.
