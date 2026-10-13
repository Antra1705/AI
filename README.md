# Adaptive Suspension: Software

Control software for the ESP32 adaptive-suspension test cart. It classifies the
road from IMU vibration with an 8-bit neural-network ensemble, maps the
classifier's confidence to a **graded** damping command, estimates body motion
with an Extended Kalman Filter, slows near obstacles with interval Type-2 fuzzy
control, tracks shock wear for predictive maintenance, and streams everything to
a web dashboard.

- Requirements: [BRD_Adaptive_Suspension_Software.md](BRD_Adaptive_Suspension_Software.md)
- How each algorithm works, and the evidence: [docs/ALGORITHMS.md](docs/ALGORITHMS.md)
- Results: [docs/simulation_results.txt](docs/simulation_results.txt), [docs/model_report.md](docs/model_report.md), figures in [docs/figures/](docs/figures/)

**Status:** all hardware-independent software is complete and tested against a
physical simulation of the cart (128 unit and closed-loop tests). The ESP32
drivers, pin map, Arduino sketch and WiFi serving are added once the hardware
exists. All results so far come from simulation.

## How to run everything (step by step)

Run every command from the project folder (the one containing this README).
No cart or ESP32 is needed: a simulated cart stands in for the hardware.

### Step 0: Prerequisites (once)

| You need | Check with | Install if missing |
|---|---|---|
| A C++ compiler and `make` | `c++ --version` and `make --version` | macOS: `xcode-select --install` · Ubuntu/Debian: `sudo apt install build-essential` |
| Python 3.9 or newer | `python3 --version` | python.org, or your package manager |
| A web browser | – | for the dashboard |

### Step 1: Python packages (once; only needed for training and figures)

Building, testing, the simulator and the dashboard work without this step.

```sh
python3 -m venv .venv
.venv/bin/pip install -r tools/requirements.txt     # numpy, PyTorch, scikit-learn, matplotlib (~1 GB)
```

### Step 2: Build and test

```sh
make -C host all       # builds the tests, simulator, replay and C++ trainer into host/build/bin/
make -C host test      # expected last line: "128 tests, 0 failed"
```

### Step 3: Watch it run on the live dashboard

```sh
python3 host/dashboard/server.py --sim
```

Open **http://localhost:8080**, then try:

| Click | What you should see |
|---|---|
| **Start** | The cart drives. The surface tile follows the road (the true surface is shown under it), and the stiffness and speed charts change with the road. |
| **Obstacle ahead** | The state changes to "Obstacle ahead", and speed ramps down to a gentle stop about 12–16 cm before the obstacle. **Clear** lets it continue. Obstacles also appear on their own every 20–30 s. |
| **Neural / Rules**, **Graded curve / Fixed angles**, **Type-2 / Type-1 / Steps** | Switches between the advanced design and its simpler fallback while driving, so you can compare them live. |
| **Unplug IMU** or **Unplug sonar** | The cart stops within about 0.1 s and shows a fault. Replug, then press **Start** again. |
| **Stop**, then **Maintenance test** 5 times | Each test lays a 1.5 cm reference bump just ahead (on the real cart you drive over your own reference bump). The cart drives slowly over it and measures the damping. After 5 tests the baseline is learned and the tile shows *Healthy*. |
| Move the **Shocks** slider to 50%, then **Maintenance test** a few more times | Worn shocks: the tile changes to *Watch*, then *Service*. |
| **Show table** | All current values as a table. |

Stop the server with **Ctrl+C**. To run the simulation faster than real time, add
`--speed 2`. If port 8080 is taken, add `--port 8090`.

### Step 4: Run the evaluations (the numbers in the docs)

```sh
host/build/bin/sim all        # ride, obstacles, faults, Kalman filter, maintenance, uncertainty (~15 s)
host/build/bin/sim tune       # how each surface's damping target was chosen
```

Each prints a table to the terminal. Individual scenarios are listed under
[Simulator commands](#simulator-commands).

### Step 5: Retrain the model (optional)

```sh
host/build/bin/sim dataset 80 1 > host/data/dataset.csv        # 80 simulated drives, ~20 s
.venv/bin/python tools/train_torch.py host/data/dataset.csv    # PyTorch, ~90 s; writes the model and docs/model_report.md
make -C host all test                                          # rebuild with the new model; the parity test checks it
```

No Python packages? `host/build/bin/train host/data/dataset.csv` does the same in C++.

### Step 6: Regenerate results and figures (optional, after changes)

```sh
host/build/bin/sim all > docs/simulation_results.txt
host/build/bin/sim tune >> docs/simulation_results.txt
.venv/bin/python tools/make_figures.py                         # writes docs/figures/*.png
```

### Step 7: Talk to the cart from a terminal (optional)

```sh
host/build/bin/sim live
```

Type commands and press Enter. Start with `log off`, which turns off the JSON stream
and shows readable status lines, then try `start`, `status`, `health`, `maint` or
`stop`. All commands are listed under
[Serial / dashboard commands](#serial--dashboard-commands). Quit with Ctrl+C.

### Later: with the real cart

Once the ESP32 firmware exists, connect the cart by USB and use the same
dashboard on the real serial port:

```sh
ls /dev/cu.usb*                                                # macOS: find the port (Linux: /dev/ttyUSB*)
python3 host/dashboard/server.py --serial /dev/cu.usbserial-0001
```

To retrain on real data, see [With the real cart](#with-the-real-cart).

### Troubleshooting

| Problem | Fix |
|---|---|
| `Simulator not built. Run: make -C host all` | Run Step 2. |
| `.venv/bin/python: No such file or directory` | Run Step 1. |
| `Address already in use` | Add `--port 8090` and open http://localhost:8090. |
| Dashboard says "waiting for data" | Check that the terminal running `server.py` shows no error, then reload the page. |
| Odd build errors after editing files | `make -C host clean all test` rebuilds from scratch. |

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
  train/                          dependency-free C++ trainer (same output as tools/train_torch.py)
  replay/                         run recordings through the cart's code; turn them into training data
  dashboard/                      web dashboard (index.html) + relay server (server.py)
tools/train_torch.py              model training in PyTorch (the deployed model; readable reference)
tools/make_figures.py             report figures from the simulation results (matplotlib)
tools/calibrate.py                thresholds for the rule-based fallback
docs/                             algorithms, model report, simulation results
```

## Tech stack

| Where | What |
|---|---|
| On the cart (ESP32) | C++11: Arduino core, no dynamic memory, no ML library. The model is int8 tables compiled in. |
| Simulator, tests, replay | C++11, `clang++`/`make`; own test framework |
| Model training | **PyTorch** (`tools/train_torch.py`), with a dependency-free C++ trainer that gives the same results (`host/train`) |
| Figures | matplotlib (`tools/make_figures.py`) |
| Dashboard | Python standard library server (Server-Sent Events) + one self-contained HTML/JS page |

The Python packages (numpy, PyTorch, scikit-learn, matplotlib) live in the project's
`.venv`; recreate it with `python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt`.
Everything else needs only a C++ compiler and Python 3.

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
host/build/bin/sim dataset 80 1 > host/data/dataset.csv          # or real recordings, see below
.venv/bin/python tools/train_torch.py host/data/dataset.csv      # PyTorch, ~90 s
#   (no Python packages? host/build/bin/train host/data/dataset.csv gives the same result in ~10 s)
python3 tools/calibrate.py --dataset host/data/dataset.csv       # thresholds for the rule-based fallback
make -C host test                                                # includes the int8 parity test against the trainer
host/build/bin/sim all > docs/simulation_results.txt && host/build/bin/sim tune >> docs/simulation_results.txt
.venv/bin/python tools/make_figures.py                           # refresh docs/figures/
```

Training writes `core/model_data.cpp`, `tests/model_vectors.h` and
`docs/model_report.md`. The data is split by drive, never by window.

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
3. `.venv/bin/python ../tools/train_torch.py data/real.csv`, then rebuild and flash.
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
| Classification | 99.3% test accuracy (int8 ensemble) vs 83.9% for tuned rules; 98.7% in closed loop |
| Damping vs SRA design | −59% gravel vibration, −57% impact peak, −51% bottom-outs, −22% servo travel |
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
