# Business Requirements Document (BRD)
## Adaptive Suspension Test Cart — Software Subsystem

| | |
|---|---|
| **Project** | ESP32-based Adaptive Suspension Test Cart |
| **Document** | Business Requirements Document — Software |
| **Version** | 0.2 (Draft for review) |
| **Date** | 13 October 2026 |
| **Prepared by** | _[Your name]_ — Software |
| **Reviewers** | _[Project guide / supervisor]_, _[Hardware teammate]_ |
| **Source** | *Software Requirements & Architecture* v1.0 (Oct 2026); project description (abstract and functionalities, Oct 2026) |
| **Status** | Draft. Software built and simulation-tested; hardware work pending (Section 9.1) |

**Changes in v0.2.** The project description adds a trained classifier, uncertainty
estimation, a graded confidence-to-damping curve, an Extended Kalman Filter,
interval Type-2 fuzzy speed control, predictive maintenance and a WiFi dashboard.
These were out of scope in v0.1; they are now in scope as BR-14 to BR-21. BR-4,
BR-5 and BR-11 are revised. The v0.1 rule-based classifier and fixed servo angles
remain as the documented fallback and comparison baseline.

---

## 1. Purpose

This document defines **what** the software for the adaptive suspension test cart must achieve and **how we will know it succeeded**. It does not cover implementation detail; that lives in the Software Requirements & Architecture (SRA) document.

It also:
- fixes the boundary between the software and hardware work,
- splits delivery into phases so a working, safe cart exists early,
- records problems found in the SRA that must be settled before coding starts (Section 12).

---

## 2. Background and Problem Statement

A vehicle's suspension is usually set once: soft for comfort or stiff for protection. Neither suits every road. A soft setting bottoms out on potholes, and a stiff setting is harsh on smooth roads. Adjusting it by hand is too slow to react to a pothole that is a fraction of a second away.

**The project builds a small-scale test cart that shows automatic suspension adaptation:** it senses the road through vibration, decides what kind of surface it is on, and changes suspension stiffness with a servo, all within a fraction of a second. The cart also has to avoid colliding with obstacles in front of it.

---

## 3. Business Objectives

| ID | Objective | Measure of success |
|---|---|---|
| **OBJ-1** | Show that suspension stiffness can adapt automatically to the road surface | Cart changes servo position correctly on at least 90% of test passes over each surface type (Section 10) |
| **OBJ-2** | React fast enough to matter on a sudden impact | Time from impact to servo command under 100 ms |
| **OBJ-3** | Operate safely without a human steering it | Zero collisions in obstacle tests; cart stops before contact every time |
| **OBJ-4** | Be reliable enough to run a full demo | At least 10 minutes of continuous running with no crash or reset |
| **OBJ-5** | Make the system's decisions visible | Live serial readout of sensor values, surface class, confidence, servo angle and motor speed |

---

## 4. Scope

### 4.1 In scope (software)
1. Firmware for the ESP32, written in Arduino C++ (single sketch, about 600 lines).
2. Initialising and checking all sensors and actuators at start-up.
3. Sampling the IMU (MPU6050) at a fixed 100 Hz into a 1-second rolling buffer.
4. Computing vibration features (including FFT frequency bands) and classifying the road surface (smooth, gravel, speed breaker, pothole) with a trained 8-bit neural-network ensemble, a confidence score and an uncertainty estimate. A rule-based classifier is kept as the fallback.
5. Mapping classifier confidence to a graded suspension stiffness (servo angle) through the response curve; fixed angles per class are kept as a fallback.
6. Measuring the distance to obstacles (HC-SR04) and slowing or stopping the drive motors.
7. Driving the servo and both DC motors through the L298N.
8. Safety behaviour: emergency stop, fail-safe on sensor faults, watchdog reset.
9. Debug telemetry over serial at 115200 baud.
10. Calibration support: logging raw features so thresholds can be tuned from real data.
11. Health counters (BR-11).
12. State estimation with an Extended Kalman Filter (BR-18).
13. Interval Type-2 fuzzy speed control near obstacles, with an obstacle-proximity state (BR-19).
14. Predictive maintenance by tracking drift in the suspension's damping (BR-20).
15. Telemetry for a web dashboard, and the dashboard itself (BR-21).
16. Host tools: physical simulator, training pipeline, replay, calibration, test suite.

### 4.2 Out of scope
- Mechanical design of the suspension linkage, chassis and wheels.
- Electrical design: wiring, power supply, level shifting, battery selection. *(Owned by the hardware teammate. The software depends on it, see Section 7.)*
- Steering or navigation. The cart drives in a straight line only.
- On-device training. The model is trained on a computer and flashed as a constant table.
- The fall/theft/crash states of the earlier project. They are merged into the state machine when the hardware teammate provides them.
- Use on a real vehicle.

---

## 5. Stakeholders

| Stakeholder | Role | Interest |
|---|---|---|
| Software owner (you) | Builds and tests the firmware | Clear, achievable requirements |
| Hardware teammate | Builds the cart, wiring and power | Agreed pin map, power budget and interface |
| Project guide / supervisor | Evaluates the project | Working demo, sound engineering, documentation |
| Demo audience / examiners | Watch the cart run | Visible and convincing adaptation; safe operation |

---

## 6. Business Requirements

Priority uses MoSCoW: **M** = Must, **S** = Should, **C** = Could.
"SRA ref" points to the matching item in the source document.

### BR-1 Reliable start-up — **M**
The system must check every sensor and actuator at power-on and **refuse to drive the motors** if a critical device is missing.
- **Acceptance:** With the MPU6050 unplugged, the serial monitor shows a clear error, the motors stay off and the servo holds at neutral (90°). With everything connected, it prints "System ready".
- SRA ref: Module 1, Failure Mode 1

### BR-2 Controlled start and stop — **M**
The cart must **not** start moving on power-up. An operator action (button press or serial command) must start it, and there must be a way to stop it.
- **Acceptance:** After power-on the motors stay at 0 until the start command; the stop command brings them to 0 within one control cycle.
- SRA ref: *Not covered in SRA, new requirement (see Section 12, issue I-13)*

### BR-3 Continuous road sensing — **M**
The system must sample acceleration and rotation at a steady **100 Hz (every 10 ms ±1 ms)** and keep the most recent 1 second of data.
- **Acceptance:** A logged run of 60 seconds contains 6000 ±60 samples with timestamps spaced 10 ms apart.
- SRA ref: REQ-1, Module 2

### BR-4 Road surface recognition — **M** (Phase 2)
The system must classify the surface as **Smooth, Gravel, Speed breaker or Pothole**, with calibrated class probabilities and a confidence value from 0 to 1, 10 times per second. When it is unsure it must say so (BR-16), not guess.
- **Acceptance:** Classification accuracy ≥ 90% per surface type on the test track (Section 10).
- SRA ref: REQ-2, REQ-3, Modules 3–4

### BR-5 Adaptive suspension — **M** (Phase 2)
The system must set suspension stiffness from the classifier output through the graded response curve (BR-17).
The original fixed angles below are kept as the fallback mode (`curve discrete`) and as the comparison baseline:

| Surface | Servo angle (fallback mode) | Meaning |
|---|---|---|
| Smooth | 45° | Soft, for comfort |
| Gravel | 110° | Medium |
| Bump | 150° | Stiff |
| Pothole | 150° | Maximum stiffness |
| Unknown / low confidence (< 0.60) | 90° | Neutral; no change |

- **Acceptance:** In graded mode, each surface settles near its measured target stiffness (BR-17), and the servo does not jitter on ambiguous input. In fallback mode, the servo reaches the angle in the table, and below 0.60 confidence it holds at 90°.
- SRA ref: REQ-4, Module 5

### BR-6 Fast reaction — **M**
From the moment an impact appears in the IMU data, the servo command must be issued within **100 ms**.
- **Acceptance:** Measured with timestamps in the serial log, ≥ 95% of pothole events show impact-to-command ≤ 100 ms.
- *Note:* Physical servo travel takes extra time, see issue I-9. There are two paths. A one-sample reflex covers very hard hits (command within 10 ms). Other potholes go through the 10 Hz classifier: command within about 100 ms, and the servo physically stiff after about 200 ms in simulation.
- SRA ref: Functional Requirements (Classification Latency)

### BR-7 Obstacle avoidance — **M**
The cart must slow down when an obstacle is close and stop before hitting it.

| Distance in front | Action |
|---|---|
| > 20 cm | Normal speed (PWM 200) |
| 10–20 cm | Slow speed (PWM 127) |
| < 10 cm | **Emergency stop** (PWM 0, both motors) |

- **Acceptance:** In 20 out of 20 approach runs toward a wall at normal speed, the cart stops without touching it.
- SRA ref: REQ-6, Module 6

### BR-8 Fail-safe behaviour — **M**
If any safety-relevant reading is missing or invalid, the system must choose the **safe** action (stop or slow), not the normal one.
- Distance sensor timeout or out-of-range reading → treat as "obstacle possible": slow or stop, never full speed.
- IMU read failure → servo neutral, motors stopped.
- **Acceptance:** Unplugging the ultrasonic sensor or the IMU while driving stops the cart within 200 ms.
- SRA ref: Module 8, Section 9 *(changes the SRA's behaviour, see issue I-6)*

### BR-9 Self-recovery — **M**
If the firmware hangs, a watchdog must reset the ESP32 automatically and it must come back up in the safe state (motors off, servo neutral).
- **Acceptance:** A deliberately inserted infinite loop triggers a reset within 3 s; after the reset the motors are off.
- SRA ref: Failure Mode 5

### BR-10 Live visibility — **S**
The system should print a readable status line to the serial monitor every 500 ms: features, surface class, confidence, distance, servo angle, motor speed and emergency-stop state.
- **Acceptance:** The status line appears at 2 Hz and does not disturb the 100 Hz sampling (BR-3 still passes with logging on).
- SRA ref: printDebugInfo()

### BR-11 Health logging — **S**
The system should keep health counters (sensor read errors, watchdog resets, emergency stops, impacts, timing) and print them on request. Predictive maintenance proper is BR-20.
- **Acceptance:** A serial command prints the counters, and they increment correctly when faults are injected.
- SRA ref: Executive Summary item 6 *(no design exists yet, see issue I-12)*

### BR-12 Tunable thresholds — **S**
All classification thresholds, servo angles, speeds and distance limits should be defined as named constants in one place, so they can be changed after calibration without editing logic.
- **Acceptance:** A reviewer can change any threshold by editing one constant.
- SRA ref: Implementation Guide Step 5

### BR-13 Calibration data capture — **S** (Phase 2)
The system should have a mode that streams raw features as CSV over serial, so that real data from each surface can be collected and used to set thresholds.
- **Acceptance:** Data captured on each surface can be opened in a spreadsheet and plotted.
- SRA ref: Implementation Guide Step 5

### Requirements added in v0.2 (project description)

### BR-14 Trained road-surface model — **M**
The classifier must be a model trained on windowed IMU data with engineered features (jerk, peak-to-peak, FFT band energies and others), not hand-set thresholds alone.
- **Acceptance:** On held-out drives, accuracy ≥ 90% per surface, and higher than the rule-based baseline on the same data. *(Simulation: 99.3% vs 83.9%.)*

### BR-15 8-bit deployment (quantization-aware training) — **M**
The model must run in 8-bit integer arithmetic on the ESP32, trained with quantization-aware training.
- **Acceptance:** On-device int8 outputs match the trainer's reference within tolerance (parity test), and int8 accuracy is within 1 point of float. *(Simulation: equal; two independent trainers, PyTorch and C++, agree.)*

### BR-16 Uncertainty and safety fallback — **M**
The system must estimate its own uncertainty (ensemble disagreement and input novelty) and fall back to a safe behaviour when unsure: a safe middle stiffness, plus lower speed.
- **Acceptance:** Accepted windows are more accurate than flagged ones, and uncertainty measurably shifts the damping and speed commands. *(Simulation: 99.9% vs 88.3%. Detection of a never-seen surface is weak, AUROC 0.78; see docs/ALGORITHMS.md.)*

### BR-17 Graded confidence-to-damping curve — **M** (core contribution)
The damping command must be a continuous function of the class probabilities and confidence, shaped to the actuator's response time, not a binary or per-class switch.
- **Acceptance:** In closed loop, fewer bottom-outs and lower impact peaks than the SRA fixed-angle design, with no missed potholes. *(Simulation: −51% bottom-outs, −57% impact peak, 0 misses vs 16 for the same classifier on fixed angles.)*

### BR-18 Extended Kalman Filter — **S**
The system should estimate attitude, gyro bias and vertical body velocity with an EKF, using classifier confidence to set the accelerometer noise.
- **Acceptance:** Attitude error well below accelerometer-only tilt; vertical velocity available to the damping curve. *(Simulation: about 0.5° vs about 4°. The classifier link shows no measurable gain; accelerometer bias dominates.)*

### BR-19 Obstacle-proximity state with fuzzy speed control — **M**
The state machine must have an obstacle-proximity state. In it, drive speed is set smoothly by an interval Type-2 fuzzy controller from distance, closing speed and road roughness, inside the hard safety rules of BR-7 and BR-8.
- **Acceptance:** No collisions, and a gentle stop before the 10 cm emergency brake in normal approaches. *(Simulation: stops 12–16 cm from the wall; Type-1 performs as well as Type-2.)*

### BR-20 Predictive maintenance — **S**
The system should run a repeatable reference-bump test, estimate the suspension's damping, learn a baseline, and flag long-term drift (worn shocks), with the record kept in flash.
- **Acceptance:** Flags simulated wear before damping falls below 60%, with no false alarms on healthy shocks. *(Simulation: WATCH at 74%, SERVICE at 60%, 0 false alarms in 40 days.)*

### BR-21 Live dashboard — **S**
The system should stream state, classification, uncertainty, actuator and maintenance data for a web dashboard. It is served over WiFi by the ESP32 once the hardware exists, and by a relay on a computer until then.
- **Acceptance:** The dashboard shows live data and can send operator commands.

---

## 7. Assumptions, Constraints and Dependencies

### 7.1 Assumptions
- A1. The hardware matches the SRA bill of materials: ESP32 dev board, MPU6050 (GY-521), HC-SR04, MG90S servo, 2× TT motors, L298N, battery pack.
- A2. The cart's top speed is low enough to stop within 10 cm (to be measured, see Risk R-3).
- A3. A test track with the four surface types can be built (for example, a flat board, a gravel tray, a speed-bump strip and a pothole cut-out).
- A4. The Arduino IDE with the Adafruit MPU6050, Adafruit Unified Sensor and ESP32Servo libraries is the toolchain.

### 7.2 Constraints
- C1. Single-microcontroller, bare-metal Arduino loop. No RTOS tasks are required, but the loop must never block for long (no `delay()` in the main path).
- C2. No ML library on the ESP32. The model is trained on a computer (PyTorch, `tools/train_torch.py`; or the dependency-free C++ trainer) and compiled in as int8 tables (about 3 kB).
- C3. ESP32 limits: about 320 KB RAM, 4 MB flash (typical dev board).
- C4. Battery-powered, so runtime depends on the hardware power budget.

### 7.3 Dependencies on hardware (interface contract)
The hardware teammate must confirm the following before coding starts:

| Signal | SRA pin | Status |
|---|---|---|
| I2C SDA / SCL (MPU6050) | GPIO 21 / 22 | Given |
| Servo PWM | GPIO 33 | Given |
| Motor A ENA (PWM) | GPIO 18 | Given |
| Motor B ENB (PWM) | GPIO 19 | Given |
| Motor A IN1 / IN2 | GPIO 27 / 14 (from Data Flow section only) | **Confirm** |
| Motor B IN3 / IN4 | — | **Missing** |
| HC-SR04 TRIG / ECHO | — | **Missing** |
| Start / stop button (BR-2) | — | **New, needs a pin** |

Also needed from hardware:
- The HC-SR04 ECHO pin outputs 5 V and the ESP32 is 3.3 V, so a voltage divider or level shifter is required.
- The servo needs its own 5 V supply. Powering it from the ESP32's 3.3 V pin causes brown-out resets.
- The measured top speed of the cart at PWM 200.

---

## 8. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Timing | IMU sampling at 100 Hz; control decisions at 10 Hz; serial status at 2 Hz |
| NFR-2 | Latency | Impact-to-servo-command ≤ 100 ms (BR-6) |
| NFR-3 | CPU | Main loop work ≤ 50% of available time, leaving headroom |
| NFR-4 | Memory | RAM use ≤ 50 KB. No dynamic allocation in the loop. *(Flash limit to be revised, see I-11)* |
| NFR-5 | Stability | ≥ 10 minutes continuous running with no crash or reset |
| NFR-6 | Safety | Obstacle stop success 100% in testing (BR-7) |
| NFR-7 | Maintainability | Code split into the functions named in the SRA; named constants (BR-12); comments on every threshold |
| NFR-8 | Determinism | The same inputs always give the same outputs |

---

## 9. Delivery Phases

| Phase | Goal | Requirements delivered | Exit criteria |
|---|---|---|---|
| **Phase 1: Safe platform** | The cart drives straight, stops for obstacles, and every sensor is read correctly | BR-1, BR-2, BR-3, BR-7, BR-8, BR-9, BR-10 | Obstacle test passes 20/20; 10-minute soak test passes; IMU at 100 Hz verified |
| **Phase 2: Adaptive suspension** | The cart recognises surfaces and adjusts the servo | BR-4, BR-5, BR-6, BR-12, BR-13 | Calibration data collected; accuracy ≥ 90% per surface; latency ≤ 100 ms |
| **Phase 3: Polish** | Health logging and demo readiness | BR-11, optional WiFi telemetry | Full demo run with no faults |

Safety is delivered first on purpose. A cart that adapts its suspension but can hit a wall is not demo-able.

Target dates: _[to be filled in against the project timeline]_

### 9.1 Status (13 October 2026)
All hardware-independent software for BR-1 to BR-21 is written and tested against a
physical simulation of the cart: 128 unit and closed-loop tests, plus the
evaluations in `docs/simulation_results.txt`. Still to do once the hardware
exists:
- ESP32 drivers, pin map and Arduino sketch;
- WiFi serving of the dashboard;
- real recordings, then retraining and recalibration (`replay --label`, `train`, `calibrate.py`, repeating the `sim tune` procedure on the bench);
- measured speed, braking and servo slew;
- accelerometer offset calibration;
- the acceptance tests in Section 10.

---

## 10. Acceptance Testing Approach

| Test | Method | Pass criterion |
|---|---|---|
| T10 Model on real data | Retrain on recordings from ≥ 20 drives; evaluate on held-out drives | ≥ 90% per surface (BR-14) |
| T11 Graded vs fixed | Same course driven in `curve graded` and `curve discrete` | Fewer bottom-outs and lower peaks in graded mode (BR-17) |
| T12 Gentle obstacle stop | 20 approaches in `speed it2` and `speed t1` | Stops without the hard brake and without contact (BR-19) |
| T13 Maintenance | `maint test` repeated; then with softened or loosened damper | Baseline learned; drift flagged (BR-20) |
| T14 Dashboard over WiFi | Phone or laptop on the cart's WiFi | Live view and commands work (BR-21) |
| T1 Start-up check | Power on with each sensor disconnected in turn | Correct error, motors off |
| T2 Sampling rate | Log 60 s of timestamps | 6000 ±60 samples, 10 ms spacing |
| T3 Surface accuracy | 20 passes over each surface; compare logged class with true surface | ≥ 90% correct per surface |
| T4 Reaction time | Pothole passes; compare impact timestamp with servo-command timestamp | ≥ 95% within 100 ms |
| T5 Obstacle stop | 20 approaches to a wall at normal speed | 20/20 stop with no contact |
| T6 Fail-safe | Unplug ultrasonic sensor, then IMU, while driving | Stop within 200 ms |
| T7 Watchdog | Firmware build with a forced hang | Reset within 3 s; safe state afterwards |
| T8 Soak test | Run for 10 minutes on mixed surfaces | No crash or reset |
| T9 No jitter | Run on an ambiguous surface | Servo holds at 90°; no oscillation |

---

## 11. Risks

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-1 | SRA thresholds do not match real sensor data, so classification fails | **High** | High | Phase 2 starts with calibration data capture (BR-13); thresholds set from measured data |
| R-2 | The test-cart scale is too small for clear vibration differences between surfaces | Medium | High | Build exaggerated test surfaces; test early with raw data |
| R-3 | The cart cannot stop within 10 cm at full speed | Medium | **Critical** | Measure stopping distance in Phase 1; lower the speed or raise the stop distance if needed |
| R-4 | Servo current causes brown-out resets | Medium | High | Separate servo supply (hardware); brown-out logged in health counters |
| R-5 | `pulseIn()` blocks for up to 30 ms and disturbs 100 Hz sampling | Medium | Medium | Use a short timeout or interrupt-based echo timing |
| R-6 | Servo PWM and motor PWM conflict over ESP32 timer channels | Low | Medium | Assign LEDC channels explicitly; test servo and motors together in Phase 1 |
| R-7 | 4×AA battery cannot run motors, servo and ESP32 for 2–3 hours | Medium | Medium | Hardware decision; software just reports the issue |

---

## 12. Issues Found in the Source SRA (resolve before coding)

These are inconsistencies or defects in the SRA that change what gets built. Each needs a decision.

| ID | Issue | Why it matters | Proposed resolution |
|---|---|---|---|
| **I-1** | **Jerk units do not match thresholds.** Jerk is computed as Δa ÷ dt with dt = 0.01 s, which multiplies values by 100. The thresholds (0.5, 1.0, 2.5) assume a per-sample difference. Any change above 0.005 m/s² between samples breaks the "Smooth" rule. | Nearly every window would classify as **Pothole**. | Either drop the ÷ dt, or rescale the thresholds. Set the final values from calibration data. |
| **I-2** | **The average acceleration magnitude hides impacts.** Averaging over 100 samples means a 2-sample spike to 14 m/s² raises the mean by only about 0.08. The Bump (> 12) and Gravel (> 10.5) rules will almost never fire. | Bump and Gravel are effectively undetectable. | Use peak-to-peak or RMS of (magnitude − 9.81) instead of the mean. Peak-to-peak is already computed but not used by the classifier. |
| **I-3** | **Bump rule is defined two ways.** Module 4 uses 12 < mag < 14; Algorithm 2 uses mag > 12. | Different behaviour depending on which section is followed. | Pick one, after calibration. |
| **I-4** | **Low confidence: stop or keep driving?** Module 8 says low confidence → motor speed 0. Algorithm 3 keeps driving at 200. | With Module 8's rule the cart stops on every Unknown reading and during the first second after start. | Low confidence should affect only the servo (neutral). Motor speed depends only on distance. |
| **I-5** | **Unthrottled IMU reads.** The loop reads the IMU on every pass with no 10 ms timer, so the rate is set by how fast the loop runs, not 100 Hz. The buffer would then not hold 1 second. | Breaks the feature maths and BR-3. | Read only when 10 ms have passed since the last sample (`micros()`-based scheduling). |
| **I-6** | **Distance fails unsafe.** A timeout, or a reading < 2 cm, returns −1, which is treated as "no obstacle" → full speed. An object pressed against the sensor or an unplugged sensor makes the cart speed up. | Safety defect. | Treat invalid readings as unsafe (BR-8). Require a few consecutive invalid readings before stopping, to filter out noise. |
| **I-7** | **Circular-buffer wrap breaks jerk.** The jerk loop compares `buffer[i]` with `buffer[i−1]` in array order. At the wrap point it compares the newest sample with the oldest one. | A false jerk spike → false Pothole. | Iterate in time order starting from `buffer_index`. |
| **I-8** | **The timestamp field `ts` is used but never stored.** The jerk formula reads `imu_buffer[i].ts`, but Module 2 does not save it. | Does not compile as written / divides by zero. | Add the timestamp to the sample struct, or use a fixed dt. |
| **I-9** | **Servo speed vs 100 ms target.** MG90S moves about 60° per 0.12 s. Going from 45° to 150° (105°) takes about 0.21 s. The SRA timeline claims 50 ms. | BR-6 cannot be met if it means "servo reached position". | BR-6 measures *command issued* ≤ 100 ms; full travel is reported separately. |
| **I-10** | **Emergency-stop timing.** Distance is checked every 100 ms, and `pulseIn` can take up to 30 ms more. The SRA itself notes 10 cm = 0.1 s at 1 m/s. | The cart may cover the whole 10 cm margin before reacting. | Check distance more often (e.g. every 30–50 ms) and confirm stopping distance in test T5. |
| **I-11** | **Flash limit of < 80 KB is not realistic.** A minimal Arduino-ESP32 sketch is already well over 200 KB. | The requirement would fail regardless of the code. | Change to "fits the ESP32 app partition (≤ 1.2 MB) with margin". |
| **I-12** | **"Predictive maintenance" logging has no design.** It is listed in the executive summary, but no module implements it. | Scope gap. | Reduced to simple health counters (BR-11, Could-have). |
| **I-13** | **No operator start/stop.** The motors start as soon as the board powers up. | Safety and demo-handling problem. | Add BR-2 (button or serial command). |
| **I-14** | **Watchdog is assumed to be "built in".** The Arduino loop task watchdog is not enabled by default. | BR-9 would silently not work. | Enable the loop watchdog explicitly (`enableLoopWDT()` or `esp_task_wdt`). |
| **I-15** | **Confidence values are fixed per class, not computed.** The examples in "How Confidence Works" (0.70 for Gravel) do not match the code (always 0.80). | Confidence carries no real information, so the 0.60 gate only ever filters Unknown. | Accept fixed values for Phase 2. Optionally scale confidence by distance from the threshold later. |
| **I-16** | **Small text inconsistencies.** The summary table says motor speeds are "127–255" (the actual values are 0/127/200). The gyro variance uses `abs(gz)` but is described as std-dev of gz. The L298N is described as "bridging 6V to 3.3V logic" (it is an H-bridge motor driver). | Documentation accuracy. | Correct in the next SRA revision. |

---

## 13. Open Questions

1. What are the remaining pin assignments (Section 7.3)? *(Hardware teammate)*
2. What is the cart's real top speed and stopping distance at PWM 200? *(Hardware teammate, Phase 1 test)*
3. Is a physical start/stop button acceptable, or a serial command only? *(Supervisor)*
4. How will the four test surfaces be built, and how many passes count as a valid accuracy test? *(Supervisor)*
5. Is WiFi telemetry wanted for the demo, or is serial enough? *(Supervisor)*
6. What is the final demo / submission date? *(Supervisor)*

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **IMU** | Inertial Measurement Unit: the MPU6050, which measures acceleration and rotation |
| **Jerk** | Rate of change of acceleration; high during sudden impacts |
| **Peak-to-peak** | Largest minus smallest value in the 1-second window |
| **Confidence** | How sure the classifier is about the surface (0–1) |
| **Confidence gate** | The rule that the servo moves only if confidence > 0.60 |
| **PWM** | Pulse-width modulation: how the ESP32 sets motor speed (0–255) and servo angle |
| **Watchdog** | A hardware timer that resets the chip if the software stops responding |
| **Fail-safe** | Defaulting to the safe action (stop) when data is missing |
| **MoSCoW** | Must / Should / Could priority scheme |

---

## 15. Traceability: BRD to SRA

| BRD | SRA reference |
|---|---|
| BR-1 | Module 1; Failure Mode 1 |
| BR-2 | *New* |
| BR-3 | REQ-1; Module 2 |
| BR-4 | REQ-2, REQ-3; Modules 3–4; Algorithms 1–2 |
| BR-5 | REQ-4; Module 5; Algorithm 3 |
| BR-6 | Functional Req: Classification Latency |
| BR-7 | REQ-6; Module 6 |
| BR-8 | Module 8; Failure Modes 1–2 |
| BR-9 | Failure Mode 5 |
| BR-10 | printDebugInfo() |
| BR-11 | Executive Summary item 6 |
| BR-12, BR-13 | Implementation Guide Step 5 |

---

**Sign-off**

| Name | Role | Signature | Date |
|---|---|---|---|
| | Software | | |
| | Hardware | | |
| | Project guide | | |
