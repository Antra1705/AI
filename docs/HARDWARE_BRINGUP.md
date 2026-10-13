# Hardware bring-up: flashing and first tests

For whoever has the cart. Follow the steps in order: each one checks one part
before the next relies on it. Takes about an hour the first time.

The firmware matches the wiring in *ESP32 Robot Car - Complete Wiring and Hardware
Guide* (38-pin ESP32 NodeMCU, L298N, MPU-6050, HC-SR04, MG90S). All pins are in
[`adaptive_suspension/src/esp32/board_config.h`](../adaptive_suspension/src/esp32/board_config.h);
if the wiring changes, that is the only file to edit.

> **Status:** the drivers are written against the datasheets and the wiring guide,
> and the firmware compiles with the official ESP32 tools (Arduino-ESP32 3.3.12);
> the control logic behind them is tested in simulation. They have **not yet run
> on the real cart**. Expect to adjust a sign or a pin. If a step fails,
> note what the serial monitor printed.

## Before powering on: four wiring checks

1. **Remove the ENA and ENB jumpers** on the L298N. The software controls speed by
   PWM on GPIO 18/19; with the jumpers on, the motors only run at full speed.
2. **ECHO divider:** HC-SR04 ECHO → 1 kΩ → GPIO 26 → 2 kΩ → GND. Without it the
   5 V echo can damage the ESP32.
3. **Servo power.** This software moves the servo often and fast (that is how it
   adapts the damping). The L298N's built-in 5 V regulator is small (about 0.5 A)
   and also powers the ESP32 and the sonar; an MG90S can draw 0.5–0.7 A when it
   pushes hard. If the ESP32 restarts while the servo moves, the serial monitor
   will say `Reset reason: BROWN-OUT`. Fix: give the servo its own 5 V buck
   converter (2 A or more; grounds connected), or at least solder a
   470–1000 µF capacitor across the servo's 5 V and GND.
4. **Mount the MPU-6050 firmly on the cart body** (not on a wheel or axle), flat,
   chip side up, X arrow pointing forward. Other orientations are fine but must be
   set in `board_config.h` (`FWD_AXIS`, `UP_SIGN`, ...).

Optional: 10 kΩ from ENA and from ENB to GND stops a possible motor twitch in the
half-second before the ESP32 has started.

## 1. Install and flash

1. Install **Arduino IDE 2**. Under *Boards Manager*, install **esp32 by
   Espressif Systems**, version **3.3.12** or another 3.x. The firmware compiles
   cleanly with 3.3.12 (77% of program space, 17% of RAM); the older 2.0.x
   versions have not been compile-checked.
2. Clone the repository and open `adaptive_suspension/adaptive_suspension.ino`.
3. *Tools → Board:* **ESP32 Dev Module**. *Tools → Port:* the cart's USB port.
4. Click **Upload**. If it hangs at "Connecting...", hold the **BOOT** button on
   the board until the upload starts.

(PlatformIO users: `pio run -t upload` from the repository root, using the
included `platformio.ini`. Note that PlatformIO's standard ESP32 platform still
ships the older 2.0.x Arduino core, which has not been compile-checked; the Arduino IDE
route above is the tested one.)

## 2. First boot (wheels off the ground from here until step 9)

Put the cart on a box so the wheels spin freely. Open *Tools → Serial Monitor*,
**115200 baud**, line ending **Newline**. Press the board's **EN** (reset) button.
Expected:

```
WiFi: join "SuspensionCart" (password "suspension1"), then open http://192.168.4.1
=== Adaptive Suspension v0.2.0 ===
Reset reason: power-on
Classifier: 5 x MLP 23-16-16-4 int8 QAT (PyTorch), test acc 99.0%
OK  MPU6050 initialized
System ready. State: IDLE. Type 'start' to drive, 'help' for commands.
```

The motors must **not** move. Type `log off` so status lines stay readable, then
`help` to see every command.

| If you see | Check |
|---|---|
| `IMU not responding` / `FAULT IMU` | SDA = GPIO 21, SCL = GPIO 22, VCC on **3V3**, AD0 to GND |
| `Reset reason: BROWN-OUT` | servo power, see above |
| Nothing at all | baud rate 115200; correct port; press EN |

## 3. IMU

With the cart still and level, type `log raw`. Lines look like
`R,<time>,ax,ay,az,gx,gy,gz,pwm`. Expected:

- **az ≈ +9.8** (gravity, pointing up), ax and ay near 0;
- gx, gy, gz near 0 (below about 0.05);
- about 100 lines per second.

Then lift the **front** of the cart: **ax should become positive** (a few m/s²).
Lift the **left** side: **ay should become positive**. If az is about −9.8, or ax /
ay move the wrong way, fix the axes/signs in `board_config.h` and flash again.
Type `log off`.

## 4. Sonar

Type `status` a few times while holding a hand in front of the sensor:
`dist=` should read about 30 cm, then 15 cm, then 5 cm, and `dist=clear` with
nothing in front. `health` → `dist no-response` should stay at 0; if it keeps
counting up, check TRIG = GPIO 25, ECHO divider to GPIO 26, VCC on 5 V.

## 5. Servo (damping)

- `servo 45`: the **softest** damper setting.
- `servo 150`: the **stiffest**.
- `servo 90`: middle.

If 150 is the soft end, set `SERVO_REVERSED = true` in `board_config.h`. If the
servo buzzes or strains at 45 or 150, the linkage is hitting its end stop:
adjust the horn position, or `SERVO_MIN_US` / `SERVO_MAX_US`. Finish with
`servo auto`.

## 6. Motors

`motor 120 500` runs both wheels for half a second at low speed. Both must turn
**forward**. If one runs backwards, swap its two wires on the L298N
(OUT1/OUT2 or OUT3/OUT4), or set `LEFT_REVERSED` / `RIGHT_REVERSED`. If a wheel
does not turn at 120, try `motor 160 500`; very low PWM may not overcome the
gearbox. Write down the lowest value that turns reliably (it matters later).

## 7. Button and WiFi dashboard

- Press the button on GPIO 4: the cart starts (wheels spin). Press again: stop.
- On a phone or laptop, join the WiFi network **SuspensionCart**
  (password `suspension1`) and open **http://192.168.4.1**. The live dashboard
  appears (the same page as the simulator's). The **Start/Stop** buttons work
  from there too.

## 8. Safety checks (still on the box)

1. `start`, then hold a hand **10 cm** in front of the sonar: the motors must stop
   at once (`ESTOP`), and restart when you take it away.
2. `start`, then unplug the **IMU** wire: the motors must stop within about 0.1 s
   with `FAULT`. Plug it back in, type `start` again.
3. `start`, then unplug the **sonar**: the cart stops with a sonar fault.
4. `health`: `slowest loop` should be well under 10000 us (10 ms).

## 9. First drive

On a smooth floor with space in front: `start`. The cart drives, the status line
shows `road=SMOOTH`, and it slows and stops before a wall. Then try a speed
breaker (a book under a mat works) and watch `road=` and `servo=` change.

## 10. Record data for training (the most useful thing after bring-up)

The model was trained on simulated data; real recordings make it fit this cart.
On a laptop joined to the cart's WiFi:

```sh
python3 tools/record_cart.py smooth1.csv       # drive on ONE surface, Ctrl+C to stop
python3 tools/record_cart.py gravel1.csv
python3 tools/record_cart.py breaker1.csv      # drive over the bump many times
python3 tools/record_cart.py pothole1.csv
```

(Or over USB: add `--serial /dev/cu.usbserial-XXXX`; on Windows use WSL or the
WiFi option.) Aim for at least 20 recordings across the four surfaces, a few
minutes in total per surface, different speeds. Commit them to `host/data/real/` or send them over;
the retraining steps are in the README ("With the real cart").

## 11. Predictive maintenance baseline

Lay a fixed reference bump (e.g. a 1.5 cm wooden strip). Type `maint test` and
let the cart drive slowly over it; repeat 5 times. After that `maint` shows the
learned baseline, and later tests flag worn shocks.

## What to report back

- the output of `health` after a drive;
- any step that did not behave as described, with the serial output;
- the lowest motor PWM that moves the cart;
- whether any sign/axis/pin in `board_config.h` had to change.
