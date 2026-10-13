# Adaptive Suspension: Software

Control software for the ESP32 adaptive-suspension test cart. It classifies the
road from IMU vibration with an 8-bit neural-network ensemble, maps the
classifier's confidence to a **graded** damping command, estimates body motion
with an Extended Kalman Filter, slows near obstacles with interval Type-2 fuzzy
control, tracks shock wear for predictive maintenance, and streams everything to
a web dashboard.

- Requirements: [BRD_Adaptive_Suspension_Software.md](BRD_Adaptive_Suspension_Software.md)
- How each algorithm works, and the evidence: [docs/ALGORITHMS.md](docs/ALGORITHMS.md)
- Results: [docs/simulation_results.txt](docs/simulation_results.txt) and [docs/model_report.md](docs/model_report.md)

**Status:** all hardware-independent software is complete and tested against a
physical simulation of the cart (128 unit and closed-loop tests). The ESP32
drivers, pin map, Arduino sketch and WiFi serving are added once the hardware
exists. All results so far come from simulation.

## Layout

```
adaptive_suspension/src/          code that runs on the ESP32 (C++11, no dynamic memory)
  config.h                        every threshold, gain and timing value (CALIBRATE = re-derive from real data)
  core/
    app.*                         main loop: scheduling, state machine wiring, commands, telemetry
    hal.h                         hardware interfaces (IMU, ultrasonic, servo, motors, serial, watchdog, storage)
    features.* ml_features.*      vibration features incl. FFT band energies
    ml_model.*  model_data.cpp    int8 neural ensemble + uncertainty (model_data.cpp is generated)
    classifier.*                  rule-based fallback classifier
    response_curve.*              confidence -> graded stiffness, actuator shaping, surface debouncing
    ekf.*                         Extended Kalman Filter (attitude, gyro bias, vertical velocity)
    fuzzy_speed.*                 interval Type-2 / Type-1 fuzzy speed control, range tracker
    obstacle_monitor.h            hard safety rules (emergency stop, sensor faults)
    supervisor.h                  IDLE / RUNNING / APPROACH / ESTOP / FAULT / MAINT
    maintenance.*                 damping estimate from a reference bump, EWMA + CUSUM drift
    servo_policy.h                fixed-angle (SRA) damping, kept as the comparison baseline
    telemetry.* command.*         serial formats (status, CSV, JSON) and operator commands
host/                             runs on a computer; no hardware, no extra libraries
  tests/                          unit + closed-loop tests
  fakes/fake_hal.h                simulated hardware interfaces
  sim/                            physical cart model, closed-loop harness, evaluations, live mode
  train/                          model training (QAT, ensemble, calibration, uncertainty, export)
  replay/                         run recordings through the cart's code; turn them into training data
  dashboard/                      web dashboard (index.html) + relay server (server.py)
tools/calibrate.py                thresholds for the rule-based fallback
docs/                             algorithms, model report, simulation results
```

## Quick start (any Mac/Linux with a C++ compiler, make and Python 3)

```sh
cd host
make test                      # 128 tests: units, physics, model parity, closed loop
make all
./build/bin/sim all            # every evaluation (about 15 s)
python3 dashboard/server.py --sim
#   then open http://localhost:8080 and press Start
```

The dashboard shows state, road surface (with the simulator's ground truth),
confidence and uncertainty, the live confidence-to-stiffness curve, speed,
obstacle distance, Kalman attitude against truth, maintenance status and an event
log. Simulator buttons place obstacles, wear the shocks, or unplug sensors.

## Simulator commands

| Command | What it shows |
|---|---|
| `sim tune` | damping characterisation per surface: where the curve's targets come from |
| `sim road [runs]` | passive soft/stiff vs SRA vs neural+fixed vs neural+graded: comfort, protection, wear |
| `sim obstacle` | SRA steps vs Type-1 vs Type-2 fuzzy at three speeds with noisy sonar |
| `sim faults` | unplugged sensors, open space, sudden close object, dropped echoes |
| `sim ekf [runs]` | EKF vs fixed-noise EKF vs accelerometer only, with and without bias |
| `sim maint` | 40 days of shock wear: when WATCH and SERVICE are raised, false alarms |
| `sim uncertain [runs]` | behaviour on a surface never seen in training |
| `sim dataset [runs] [seed]` | labelled training windows |
| `sim record <surface> <s>` | a raw log as the cart's `log raw` would print it |
| `sim live` | real-time cart for the dashboard |

## Training the model

```sh
cd host
./build/bin/sim dataset 80 1 > data/dataset.csv          # or real recordings, see below
./build/bin/train data/dataset.csv                       # writes core/model_data.cpp, tests/model_vectors.h, docs/model_report.md
python3 ../tools/calibrate.py --dataset data/dataset.csv  # thresholds for the rule-based fallback
make test                                                # includes the int8 parity test against the trainer
```

Training takes about 10 s. The data is split by drive, never by window.

### With the real cart
1. Type `log raw` in the serial monitor, drive on one surface, and save the output.
   Start saving once the cart is moving. For speed breakers and potholes, drive
   over them repeatedly.
2. Turn each recording into training rows, one run number per recording:
   ```sh
   ./build/bin/replay smooth1.csv  --label smooth  --run 0 --header > data/real.csv
   ./build/bin/replay gravel1.csv  --label gravel  --run 1 >> data/real.csv
   ./build/bin/replay breaker1.csv --label breaker --run 2 >> data/real.csv
   ./build/bin/replay pothole1.csv --label pothole --run 3 >> data/real.csv
   ```
   Recordings from at least 20 drives give a usable train/validation/test split.
3. `./build/bin/train data/real.csv`, then rebuild and flash.
4. Re-measure the damping targets (`STIFF_*`) and EKF noise on the real cart,
   following the procedure in `sim tune`.

## Serial / dashboard commands

| Command | Effect |
|---|---|
| `start`, `stop` | drive / stop; the cart never starts by itself |
| `status`, `health` | status line; fault counters and timing |
| `log raw` / `features` / `json` / `off` | data streams for recording, calibration, dashboard |
| `servo <0-180>` / `servo auto` | manual servo / automatic |
| `motor <pwm> [ms]` | short motor test while IDLE |
| `maint`, `maint test`, `maint reset` | maintenance status / reference-bump test / new baseline |
| `mode ml` / `mode rules` | neural ensemble / rule-based fallback |
| `curve graded` / `curve discrete` | graded confidence curve / SRA fixed angles |
| `speed it2` / `t1` / `step` | Type-2 fuzzy / Type-1 fuzzy / SRA steps near obstacles |

## Key results (simulation)

| Area | Result |
|---|---|
| Classification | 99.3% test accuracy (int8 ensemble) vs 83.9% for tuned rules; 98.6% in closed loop |
| Damping vs SRA design | −58% gravel vibration, −57% impact peak, −51% bottom-outs, −21% servo travel |
| Obstacles | fuzzy control stops gently 12–16 cm from the wall; SRA steps hit it at 0.9 m/s |
| Fail-safe | IMU loss → stop in 39 ms, sonar loss → stop in 109 ms |
| Maintenance | WATCH at 74% damping, SERVICE at 60%, no false alarms in 40 healthy days |

Honest limits are listed in [docs/ALGORITHMS.md](docs/ALGORITHMS.md):
- quantization-aware training adds nothing measurable at this model size;
- the EKF's classifier link gives no measurable gain in simulation;
- Type-2 fuzzy is no better than Type-1 here;
- detection of a never-seen surface is weak;
- the system cannot pre-empt the front wheel's impact.

## Still to do with the hardware

ESP32 drivers and pin map, Arduino sketch, WiFi dashboard serving, real
recordings and retraining, measured speed, braking and servo slew, accelerometer
calibration, and the hardware acceptance tests in BRD Section 10. The fall,
theft and crash states from the earlier project can be added to `supervisor.h`.
