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
physical simulation of the cart (133 unit and closed-loop tests). The ESP32
firmware (drivers for the agreed wiring, Arduino sketch, WiFi dashboard) is
written and compiles with Arduino-ESP32 3.3.12, but has **not yet run on the real cart**: flash it and test it step by step with
[docs/HARDWARE_BRINGUP.md](docs/HARDWARE_BRINGUP.md). All results so far come
from simulation.

## How to run everything (step by step)

There are two parts. **Part A** runs on any computer with a simulated cart, no
hardware needed. **Part B** puts the same software on the real ESP32 cart.
Run every command from the project folder (the one containing this README).

| I want to... | Go to |
|---|---|
| see the system working right now, without the cart | A1–A3 |
| reproduce the numbers in the docs, or retrain the model | A4–A6 |
| flash and test the real cart | B1–B4 |
| teach the model the real cart's roads | B5 |

Get the code (once):

```sh
git clone https://github.com/Antra1705/AI.git
cd AI
```

## Part A: on your computer (simulated cart)

### A0: Prerequisites (once)

| You need | Check with | Install if missing |
|---|---|---|
| A C++ compiler and `make` | `c++ --version` and `make --version` | macOS: `xcode-select --install` · Ubuntu/Debian: `sudo apt install build-essential` |
| Python 3.9 or newer | `python3 --version` | python.org, or your package manager |
| A web browser | – | for the dashboard |

Python packages are only needed for training and figures (A5, A6):

```sh
python3 -m venv .venv
.venv/bin/pip install -r tools/requirements.txt     # numpy, PyTorch, scikit-learn, matplotlib (~1 GB)
```

### A1: Build and test

```sh
make -C host all       # builds the tests, simulator, replay and C++ trainer into host/build/bin/
make -C host test      # expected last line: "133 tests, 0 failed"
```

### A2: Watch it run on the live dashboard

```sh
python3 host/dashboard/server.py --sim
```

Open **http://localhost:8080**, then try:

| Click | What you should see |
|---|---|
| **Start** | The cart drives. The instrument cluster shows the detected road surface (ground truth beside it), the stiffness dial and the speed; below it, a 3D model of the cart drives over the road it has detected (drag to orbit, or pick Chase / Side / Top / Front; **Side** at the top right switches to the 2D side view). Its body bounces on springs damped by the real servo setting, leans with the Kalman filter's roll and pitch, and the springs light up as the suspension stiffens. The **Last 30 s** strip chart shows road, stiffness, motor, obstacle distance and pitch on one time axis: hover (or tap) it to read every value at that moment; **Pause** freezes it. |
| **Place obstacle** | The state changes to "Obstacle ahead", and speed ramps down to a gentle stop about 12–16 cm before the obstacle (the shaded zones in the distance lane are slow and e-stop). **Remove obstacle** lets it continue. Obstacles also appear on their own every 20–30 s. |
| **Neural / Rules**, **Graded / Fixed angles**, **Type-2 fuzzy / Type-1 / Steps** | Switches between the advanced design and its simpler fallback while driving, so you can compare them live. |
| **Unplug IMU** or **Unplug sonar** | The cart stops within about 0.1 s and shows a fault. **Reconnect**, then press **Start** again. |
| **Stop**, then **Maintenance test** 5 times | Each test lays a 1.5 cm reference bump just ahead (on the real cart you drive over your own reference bump). The cart drives slowly over it and measures the damping. After 5 tests the baseline is learned and *Shocks* shows *Healthy*. |
| Move **Shock condition** to 50%, then **Maintenance test** a few more times | Worn shocks: *Shocks* changes to *Watch*, then *Service*. |
| **All values** (bottom) | Every number the cart reports, as a table. |

Stop the server with **Ctrl+C**. To run the simulation faster than real time, add
`--speed 2`. If port 8080 is taken, add `--port 8090`.

### A3: Talk to the simulated cart from a terminal (optional)

```sh
host/build/bin/sim live
```

Type commands and press Enter. These are exactly the commands the real cart
accepts over USB. Start with `log off`, which turns off the JSON stream and shows
readable status lines, then try `start`, `status`, `health`, `maint` or `stop`.
All commands are listed under [Serial / dashboard commands](#serial--dashboard-commands).
Quit with Ctrl+C.

### A4: Run the evaluations (the numbers in the docs)

```sh
host/build/bin/sim all                    # ride, obstacles, faults, Kalman filter, maintenance, uncertainty (~15 s)
host/build/bin/sim road --domain shifted  # the same ride comparison on a heavier, different cart
host/build/bin/sim tune                   # how each surface's damping target was chosen
```

Each prints a table to the terminal. Individual scenarios are listed under
[Simulator commands](#simulator-commands).

### A5: Retrain the model (optional)

```sh
host/build/bin/sim dataset 80 2 --domain robust --augment --other > host/data/dataset.csv   # randomised training drives, ~30 s
host/build/bin/sim dataset 30 4 > host/data/nominal_test.csv                                 # unseen test drives
host/build/bin/sim dataset 30 3 --domain shifted > host/data/shifted.csv                     # a "different cart", test only
.venv/bin/python tools/train_torch.py host/data/dataset.csv \
    --eval nominal_test=host/data/nominal_test.csv --eval shifted=host/data/shifted.csv   # ~4 min
make -C host all test                                          # rebuild with the new model; the parity test checks it
```

No Python packages? `host/build/bin/train host/data/dataset.csv` does the same in C++.
The new model is compiled into the firmware too: flash the cart again (B2) to use it.

### A6: Regenerate results and figures (optional, after changes)

```sh
host/build/bin/sim all > docs/simulation_results.txt
host/build/bin/sim road --domain shifted >> docs/simulation_results.txt   # section 1b: a different cart
host/build/bin/sim tune >> docs/simulation_results.txt
.venv/bin/python tools/make_figures.py                         # writes docs/figures/*.png
```

## Part B: on the real cart (ESP32)

The firmware is `adaptive_suspension/adaptive_suspension.ino`: the same control
code as the simulator, plus drivers for the wiring in *ESP32 Robot Car - Complete
Wiring and Hardware Guide*. Pins are in
[`adaptive_suspension/src/esp32/board_config.h`](adaptive_suspension/src/esp32/board_config.h).
It compiles with Arduino-ESP32 3.3.12 (77% of program space, 17% of RAM) but
has **not yet run on the cart**. Do B3 carefully the first time.

### B1: Install the ESP32 tools (once)

Either the **Arduino IDE** (easiest):
1. Install Arduino IDE 2.
2. *Boards Manager*: install **esp32 by Espressif Systems**, version **3.3.12**.

or the **command line** (macOS; on Linux/Windows install `arduino-cli` from arduino.cc):

```sh
brew install arduino-cli
arduino-cli config init
arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32@3.3.12          # ~2 GB download
```

### B2: Compile and flash

Plug the ESP32 into the computer by USB.

- **Arduino IDE:** open `adaptive_suspension/adaptive_suspension.ino`, choose
  *Tools → Board → ESP32 Dev Module* and the USB port, click **Upload**.
- **Command line:**
  ```sh
  ls /dev/cu.usb*                                                            # macOS: find the port (Linux: /dev/ttyUSB*)
  arduino-cli compile --fqbn esp32:esp32:esp32 adaptive_suspension           # check it builds
  arduino-cli upload  --fqbn esp32:esp32:esp32 -p /dev/cu.usbserial-0001 adaptive_suspension
  arduino-cli monitor -p /dev/cu.usbserial-0001 -c baudrate=115200           # serial monitor; Ctrl+C to quit
  ```

If the upload hangs at "Connecting...", hold the board's **BOOT** button until
it starts.

### B3: First power-on and hardware checks

Follow **[docs/HARDWARE_BRINGUP.md](docs/HARDWARE_BRINGUP.md)** step by step,
**with the wheels off the ground**. It covers:
- the wiring checks before power-on: ENA/ENB jumpers off, ECHO divider, servo power;
- testing the IMU, sonar, servo and motors one at a time, with the expected output;
- the safety checks: emergency stop, and unplugging a sensor while driving;
- the first drive.

The serial monitor (115200 baud, line ending *Newline*) accepts the same
commands as A3: `log off`, `status`, `health`, `servo 90`, `motor 120 500`,
`start`, `stop`, ...

### B4: Dashboard from the cart

The cart opens its own WiFi network. On a phone or laptop:
1. Join **SuspensionCart** (password `suspension1`).
2. Open **http://192.168.4.1**.

You get the same dashboard as in A2, live from the real sensors. The simulator-only
buttons are hidden. To change the network name or password, edit `board_config.h`
and flash again.

Over USB instead of WiFi (close the serial monitor first):

```sh
python3 host/dashboard/server.py --serial /dev/cu.usbserial-0001    # then open http://localhost:8080
```

### B5: Record real roads and retrain

The model has only seen simulated roads. Recordings from the real cart make it
fit this cart:

```sh
python3 tools/record_cart.py smooth1.csv       # laptop on the cart's WiFi; drive on ONE surface; Ctrl+C to stop
python3 tools/record_cart.py gravel1.csv       # add --serial /dev/cu.usbserial-0001 to record over USB
python3 tools/record_cart.py breaker1.csv      # drive over the bump many times
python3 tools/record_cart.py pothole1.csv
```

Then label them, retrain and flash again. The details are in
[With the real cart](#with-the-real-cart).

### Troubleshooting

| Problem | Fix |
|---|---|
| `Simulator not built. Run: make -C host all` | Run A1. |
| `.venv/bin/python: No such file or directory` | Run the Python step in A0. |
| `Address already in use` | Add `--port 8090` and open http://localhost:8090. |
| Dashboard says "waiting for data" | Check that the terminal running `server.py` shows no error, then reload the page. |
| Odd build errors after editing files | `make -C host clean all test` rebuilds from scratch. |
| `esp32_dashboard_copy_is_up_to_date` test fails | You edited `host/dashboard/index.html`: run `python3 tools/embed_dashboard.py`. |
| Upload hangs at "Connecting..." | Hold the ESP32's BOOT button while uploading. |
| Cart restarts, serial says `Reset reason: BROWN-OUT` | The 5 V supply dips when the servo moves: give the servo its own 5 V converter or a 470–1000 µF capacitor (B3). |
| `FAULT IMU stopped responding` / `IMU not responding` | Check SDA = GPIO 21, SCL = GPIO 22, MPU-6050 VCC on 3V3, AD0 to GND. |
| A wheel turns backwards | Swap that motor's two wires on the L298N, or set `LEFT_REVERSED` / `RIGHT_REVERSED` in `board_config.h`. |
| The servo turns the wrong way (150 = soft) | Set `SERVO_REVERSED = true` in `board_config.h`. |
| No "SuspensionCart" WiFi | Check the serial monitor for the `WiFi:` line after a reset; `WIFI_ENABLED` must be true in `board_config.h`. |

## Layout

```
adaptive_suspension/
  adaptive_suspension.ino         Arduino sketch: wires the ESP32 drivers to the control loop
adaptive_suspension/src/          code that runs on the ESP32 (C++11, no dynamic memory)
  esp32/board_config.h            pins and wiring options (the only file to edit if the wiring changes)
  esp32/esp32_hal.*               ESP32 drivers: MPU-6050, HC-SR04, servo, L298N, serial, watchdog, flash
  esp32/web_dashboard.cpp         the cart's own WiFi network + dashboard (runs on the second CPU core)
  esp32/dashboard_html.h          generated copy of host/dashboard/index.html (tools/embed_dashboard.py)
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
tools/record_cart.py              record raw IMU data from the real cart (over its WiFi or USB)
tools/embed_dashboard.py          copy the dashboard page into the firmware after editing it
tools/embed_fonts.py              embed the dashboard's fonts (Barlow, SIL OFL; host/dashboard/fonts/) into the page
platformio.ini                    PlatformIO alternative to the Arduino IDE
docs/                             algorithms, model report, simulation results, hardware bring-up guide
```

## Tech stack

| Where | What |
|---|---|
| On the cart (ESP32) | C++11 on the Arduino-ESP32 core 3.3.12: no dynamic memory, no ML library (the model is int8 tables compiled in). WiFi dashboard in its own FreeRTOS task on the second CPU core. |
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
| `sim road [runs] [--domain shifted] [--classify-ms N]` | passive soft/stiff vs SRA vs neural+fixed vs neural+graded: comfort, protection, wear, detection speed; optionally on a different cart |
| `sim obstacle` | SRA steps vs Type-1 vs Type-2 fuzzy at three speeds with noisy sonar |
| `sim faults` | unplugged sensors, open space, sudden close object, dropped echoes |
| `sim ekf [runs]` | EKF vs fixed-noise EKF vs accelerometer only, with and without bias |
| `sim maint` | 40 days of shock wear: when WATCH and SERVICE are raised, false alarms |
| `sim uncertain [runs]` | behaviour on a surface never seen in training |
| `sim dataset [runs] [seed] [--domain nominal\|robust\|shifted] [--augment] [--other]` | labelled windows: nominal, randomised training data, or a different cart for testing; `--other` adds cobbles/washboard for outlier exposure |
| `sim record <surface> <s>` | a raw log as the cart's `log raw` would print it |
| `sim live` | real-time cart for the dashboard |

## Training the model

```sh
host/build/bin/sim dataset 80 2 --domain robust --augment --other > host/data/dataset.csv   # or real recordings, see below
.venv/bin/python tools/train_torch.py host/data/dataset.csv \
    --eval nominal_test=host/data/nominal_test.csv --eval shifted=host/data/shifted.csv   # ~4 min
#   (no Python packages? host/build/bin/train host/data/dataset.csv gives nearly the same result in ~15 s)
python3 tools/calibrate.py --dataset host/data/dataset.csv       # thresholds for the rule-based fallback
make -C host test                                                # includes the int8 parity test against the trainer
host/build/bin/sim all > docs/simulation_results.txt                # then add road --domain shifted and tune (A6)
.venv/bin/python tools/make_figures.py                           # refresh docs/figures/
```

Training writes `core/model_data.cpp`, `tests/model_vectors.h` and
`docs/model_report.md`. The data is split by drive, never by window; augmented
copies are used for training only. Rows labelled "other" (cobbles, washboard) are
used only to teach the model to be unsure (`--oe 0.5`, both trainers); the rumble
strip is kept out of training to test that. `--features base` or `--features short
--no-export` trains the earlier 16- or 21-feature variants for comparison.
`tools/eval_model.py <test.csv> ...` scores the deployed model on any dataset.

### With the real cart
1. Record one surface per file, from a laptop on the cart's WiFi (or add
   `--serial <port>` for USB). For speed breakers and potholes, drive over them
   repeatedly:
   ```sh
   python3 tools/record_cart.py smooth1.csv     # Ctrl+C to stop
   ```
2. Turn each recording into training rows, one run number per recording:
   ```sh
   host/build/bin/replay smooth1.csv  --label smooth  --run 0 --header > host/data/real.csv
   host/build/bin/replay gravel1.csv  --label gravel  --run 1 >> host/data/real.csv
   host/build/bin/replay breaker1.csv --label breaker --run 2 >> host/data/real.csv
   host/build/bin/replay pothole1.csv --label pothole --run 3 >> host/data/real.csv
   ```
   Recordings from at least 20 drives give a usable train/validation/test split.
3. `.venv/bin/python tools/train_torch.py host/data/real.csv`, then
   `make -C host test` and flash again (B2).
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
| Classification | 99.7% on unseen normal carts, **97.9% on a different cart** (rules: 38.9%); 99.9% correct in closed loop |
| Impacts | breakers/potholes recognised as an impact 99.7% of the time, on both carts (rules: 93% / 85%) |
| Detection speed | impact recognised in 27 ms on average (was 89 ms); servo stiff in 131 ms (was 195 ms) |
| Damping vs SRA design | −65% gravel vibration, −62% impact peak, −54% bottom-outs, −47% servo travel |
| Unseen road | detection AUROC 0.87–0.90 (was 0.60–0.75) |
| Obstacles | fuzzy control stops gently 12–16 cm from the wall; SRA steps hit it at 0.9 m/s |
| Fail-safe | IMU loss → stop in 39 ms, sonar loss → stop in 109 ms |
| Maintenance | WATCH at 74% damping, SERVICE at 60%, no false alarms in 40 healthy days |

Honest limits are listed in [docs/ALGORITHMS.md](docs/ALGORITHMS.md):
- quantization-aware training adds nothing measurable at this model size;
- the EKF's classifier link gives no measurable gain in simulation;
- Type-2 fuzzy is no better than Type-1 here;
- a never-seen surface is now ranked as unusual far better (AUROC 0.60 → 0.87), but at the chosen false-alarm rate only 6–8% of its windows are flagged; it gets soft, gravel-like damping instead;
- on a heavier, softer-sprung cart with worn shocks, gravel causes light bump-stop touches (total bottom-outs slightly above the SRA design there, although at breakers and potholes it is better: 4.26 vs 4.68 per event);
- the system cannot pre-empt the front wheel's impact.

## Still to do with the hardware

Running the firmware on the cart for the first time
([docs/HARDWARE_BRINGUP.md](docs/HARDWARE_BRINGUP.md)), real recordings and
retraining, measured speed, braking and servo slew, accelerometer calibration,
and the hardware acceptance tests in BRD Section 10. The fall,
theft and crash states from the earlier project can be added to `supervisor.h`.
