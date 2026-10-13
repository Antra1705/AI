// adaptive_suspension.ino - ESP32 firmware for the adaptive-suspension cart.
//
// All control logic is in src/core (the same code the simulator and tests run);
// src/esp32 holds the hardware drivers, and src/esp32/board_config.h the pins.
// Setup and first-power-on checklist: docs/HARDWARE_BRINGUP.md
//
// Board: "ESP32 Dev Module" (Arduino-ESP32 core 2.x or 3.x). Serial monitor: 115200 baud.

#include "src/config.h"
#include "src/core/app.h"
#include "src/esp32/esp32_hal.h"

#ifdef SET_LOOP_TASK_STACK_SIZE
SET_LOOP_TASK_STACK_SIZE(16 * 1024);  // the control loop's formatting buffers need more than the default 8 KB
#endif

static Esp32Hardware hw;
static App app(hw.hal);

void setup() {
  hw.begin();   // motors off first, then serial, servo, sonar, storage, WiFi
  app.begin();  // brakes, centres the servo, starts the IMU and the watchdog; never drives by itself
}

void loop() {
  const uint32_t now = millis();
  hw.motors.service(now);
  if (hw.button.pressed(now)) {
    const RunState st = app.state();
    hw.console.inject(st == RunState::IDLE || st == RunState::FAULT ? "start" : "stop");
  }
  app.tick();
}
