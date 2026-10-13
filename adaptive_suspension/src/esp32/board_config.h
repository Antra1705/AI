// board_config.h - pins and wiring options for the real cart (ESP32 only).
//
// Matches "ESP32 Robot Car - Complete Wiring and Hardware Guide":
// 38-pin ESP32 NodeMCU, L298N, MPU-6050, HC-SR04, MG90S.
// If the wiring changes, this is the only file that needs editing.
#pragma once
#include <stdint.h>

namespace board {

// ---------------- MPU-6050 (I2C, AD0 to GND -> address 0x68) ----------------
constexpr int     PIN_I2C_SDA  = 21;
constexpr int     PIN_I2C_SCL  = 22;
constexpr uint8_t MPU_ADDR     = 0x68;
constexpr uint32_t I2C_HZ      = 400000;
// The sensor's built-in low-pass filter. 4 = 21 Hz accel / 20 Hz gyro, the setting
// the training data assumes (the simulator's default 21 Hz filter).
constexpr uint8_t MPU_DLPF_CFG = 4;
// Mounting: which sensor axis points along the cart's forward / left / up direction,
// and its sign. Default: sensor flat, chip side up, X towards the front.
// Example for a sensor mounted upside down: UP_AXIS = 2, UP_SIGN = -1, LEFT_SIGN = -1.
constexpr int FWD_AXIS = 0, FWD_SIGN = 1;
constexpr int LEFT_AXIS = 1, LEFT_SIGN = 1;
constexpr int UP_AXIS = 2, UP_SIGN = 1;

// ---------------- HC-SR04 (ECHO through a 1k / 2k divider) ----------------
constexpr int PIN_SONAR_TRIG = 25;
constexpr int PIN_SONAR_ECHO = 26;

// ---------------- MG90S servo (damper adjustment) ----------------
constexpr int PIN_SERVO = 33;
// Pulse width for 0 and 180 degrees. MG90S is typically 500-2400 us; if the
// horn does not reach the end stops (or buzzes against them), adjust these.
constexpr int SERVO_MIN_US = 500;
constexpr int SERVO_MAX_US = 2400;
// The software assumes a HIGHER angle = STIFFER damping (45 deg soft ... 150 deg stiff).
// If the linkage works the other way round, set this to true (angle -> 180 - angle).
constexpr bool SERVO_REVERSED = false;

// ---------------- L298N motor driver (ENA / ENB jumpers removed) ----------------
constexpr int PIN_LEFT_EN  = 18;  // ENA, PWM speed
constexpr int PIN_LEFT_IN1 = 27;  // IN1, HIGH = forward
constexpr int PIN_LEFT_IN2 = 14;  // IN2
constexpr int PIN_RIGHT_EN  = 19; // ENB, PWM speed
constexpr int PIN_RIGHT_IN3 = 23; // IN3, HIGH = forward
constexpr int PIN_RIGHT_IN4 = 13; // IN4
// If a motor runs backwards on "motor 120 500", swap its two wires on the L298N
// (OUT1/OUT2 or OUT3/OUT4), or set its flag here.
constexpr bool LEFT_REVERSED  = false;
constexpr bool RIGHT_REVERSED = false;
constexpr uint32_t MOTOR_PWM_HZ = 1000;
// brake(): short the motors for this long (fast stop), then let them coast so
// the cart can still be pushed by hand while stopped.
constexpr uint32_t BRAKE_ACTIVE_MS = 500;

// ---------------- Start / stop push button (to GND, internal pull-up) ----------------
constexpr int PIN_BUTTON = 4;  // set to -1 if no button is fitted
constexpr uint32_t BUTTON_DEBOUNCE_MS = 40;

// ---------------- WiFi dashboard ----------------
// The cart opens its own WiFi network; connect a phone or laptop to it and open
// http://192.168.4.1 . Set WIFI_ENABLED = false to run on USB serial only.
constexpr bool WIFI_ENABLED = true;
constexpr const char* WIFI_AP_SSID = "SuspensionCart";
constexpr const char* WIFI_AP_PASSWORD = "suspension1";  // at least 8 characters
constexpr int WEB_PORT = 80;
constexpr int WEB_MAX_STREAMS = 2;  // browsers watching at the same time

constexpr uint32_t SERIAL_BAUD = 115200;

}  // namespace board
