// esp32_hal.cpp - ESP32 drivers. See esp32_hal.h; pins in board_config.h.
#ifdef ARDUINO_ARCH_ESP32

#include "esp32_hal.h"

#include <Wire.h>
#include <esp_system.h>
#include <esp_task_wdt.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

#include "board_config.h"

#if __has_include(<esp_arduino_version.h>)
#include <esp_arduino_version.h>
#endif
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
#define CORE_V3 1
#else
#define CORE_V3 0
#endif

// ---------------------------------------------------------------- PWM (LEDC)
// Arduino-ESP32 3.x addresses PWM by pin, 2.x by channel. Channel numbers are
// only used on 2.x: servo on channel 0 (timer 0, 50 Hz), motors on 2 and 3
// (timer 1, MOTOR_PWM_HZ) so the two frequencies never share a timer.
namespace {
const int CH_SERVO = 0, CH_LEFT = 2, CH_RIGHT = 3;
const int SERVO_BITS = 16, MOTOR_BITS = 8;

void pwmAttach(int pin, int channel, uint32_t freq, int bits) {
#if CORE_V3
  (void)channel;
  ledcAttach(pin, freq, bits);
#else
  ledcSetup(channel, freq, bits);
  ledcAttachPin(pin, channel);
#endif
}

void pwmWrite(int pin, int channel, uint32_t duty) {
#if CORE_V3
  (void)channel;
  ledcWrite(pin, duty);
#else
  (void)pin;
  ledcWrite(channel, duty);
#endif
}

QueueHandle_t g_cmd_queue = nullptr;  // command lines from WiFi / button -> control loop
const int CMD_LEN = 80;
}  // namespace

// ---------------------------------------------------------------- MPU-6050

namespace {
const uint8_t REG_SMPLRT_DIV = 0x19, REG_CONFIG = 0x1A, REG_GYRO_CONFIG = 0x1B, REG_ACCEL_CONFIG = 0x1C;
const uint8_t REG_ACCEL_XOUT_H = 0x3B, REG_PWR_MGMT_1 = 0x6B, REG_WHO_AM_I = 0x75;
const float ACCEL_SCALE = 9.80665f / 4096.0f;                  // +-8 g
const float GYRO_SCALE = (3.14159265f / 180.0f) / 65.5f;       // +-500 deg/s
}  // namespace

bool EspImu::writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(board::MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission(true) == 0;
}

bool EspImu::readRegs(uint8_t reg, uint8_t* buf, int n) {
  Wire.beginTransmission(board::MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint16_t)board::MPU_ADDR, (size_t)n, true) != (size_t)n) return false;
  for (int i = 0; i < n; ++i) buf[i] = (uint8_t)Wire.read();
  return true;
}

bool EspImu::begin() {
  if (!wire_started_) {
    Wire.begin(board::PIN_I2C_SDA, board::PIN_I2C_SCL, board::I2C_HZ);
    Wire.setTimeOut(10);  // ms: an unplugged sensor must not stall the loop
    wire_started_ = true;
  }
  uint8_t who = 0;
  if (!readRegs(REG_WHO_AM_I, &who, 1)) return false;
  // Genuine parts answer 0x68; common clones answer 0x70, 0x72 or 0x98 and work the same.
  if (who != 0x68 && who != 0x70 && who != 0x72 && who != 0x98) return false;
  return writeReg(REG_PWR_MGMT_1, 0x01)           // wake up, gyro X clock
         && writeReg(REG_SMPLRT_DIV, 9)           // 1 kHz / (1 + 9) = 100 Hz
         && writeReg(REG_CONFIG, board::MPU_DLPF_CFG)
         && writeReg(REG_GYRO_CONFIG, 0x08)       // +-500 deg/s
         && writeReg(REG_ACCEL_CONFIG, 0x10);     // +-8 g
}

bool EspImu::read(ImuSample& out) {
  uint8_t b[14];
  if (!readRegs(REG_ACCEL_XOUT_H, b, 14)) return false;
  bool all_ff = true, all_00 = true;  // a floating bus reads all ones; a dead sensor all zeros
  for (int i = 0; i < 14; ++i) {
    all_ff = all_ff && b[i] == 0xFF;
    all_00 = all_00 && b[i] == 0x00;
  }
  if (all_ff || all_00) return false;
  float a[3], g[3];
  for (int i = 0; i < 3; ++i) {
    a[i] = (float)(int16_t)((b[2 * i] << 8) | b[2 * i + 1]) * ACCEL_SCALE;
    g[i] = (float)(int16_t)((b[8 + 2 * i] << 8) | b[9 + 2 * i]) * GYRO_SCALE;
  }
  out.ax = board::FWD_SIGN * a[board::FWD_AXIS];
  out.ay = board::LEFT_SIGN * a[board::LEFT_AXIS];
  out.az = board::UP_SIGN * a[board::UP_AXIS];
  out.gx = board::FWD_SIGN * g[board::FWD_AXIS];
  out.gy = board::LEFT_SIGN * g[board::LEFT_AXIS];
  out.gz = board::UP_SIGN * g[board::UP_AXIS];
  return true;
}

// ---------------------------------------------------------------- HC-SR04

namespace {
enum SonarState : int { SONAR_IDLE, SONAR_WAIT_RISE, SONAR_WAIT_FALL, SONAR_DONE, SONAR_BUSY };
volatile int s_state = SONAR_IDLE;
volatile uint32_t s_trig_us = 0, s_rise_us = 0, s_width_us = 0;
const uint32_t SONAR_RISE_TIMEOUT_US = 10000;  // echo normally starts ~0.5 ms after the trigger
const uint32_t SONAR_MAX_ECHO_US = 25000;      // ~430 cm: anything longer means "nothing in range"

void IRAM_ATTR sonarIsr() {
  const uint32_t now = (uint32_t)micros();
  if (digitalRead(board::PIN_SONAR_ECHO)) {
    if (s_state == SONAR_WAIT_RISE) {
      s_rise_us = now;
      s_state = SONAR_WAIT_FALL;
    }
  } else if (s_state == SONAR_WAIT_FALL) {
    s_width_us = now - s_rise_us;
    s_state = SONAR_DONE;
  }
}
}  // namespace

void EspSonar::begin() {
  pinMode(board::PIN_SONAR_TRIG, OUTPUT);
  digitalWrite(board::PIN_SONAR_TRIG, LOW);
  pinMode(board::PIN_SONAR_ECHO, INPUT);  // the divider's 2k resistor pulls it low if unplugged
  attachInterrupt(digitalPinToInterrupt(board::PIN_SONAR_ECHO), sonarIsr, CHANGE);
}

void EspSonar::trigger() {
  if (digitalRead(board::PIN_SONAR_ECHO)) {
    // Still timing out a previous "no echo" pulse (some modules hold it high for
    // up to ~200 ms). The sensor is alive and sees nothing in range.
    s_state = SONAR_BUSY;
    return;
  }
  s_state = SONAR_WAIT_RISE;
  s_trig_us = (uint32_t)micros();
  digitalWrite(board::PIN_SONAR_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(board::PIN_SONAR_TRIG, LOW);
}

bool EspSonar::poll(DistanceReading& out) {
  const uint32_t now = (uint32_t)micros();
  const int st = s_state;
  if (st == SONAR_BUSY) {
    s_state = SONAR_IDLE;
    out = {DistanceStatus::OUT_OF_RANGE, 0.0f};
    return true;
  }
  if (st == SONAR_DONE) {
    s_state = SONAR_IDLE;
    const float cm = (float)s_width_us / 58.0f;
    if (cm < 2.0f) out = {DistanceStatus::TOO_CLOSE, cm};
    else if (cm > 400.0f) out = {DistanceStatus::OUT_OF_RANGE, 0.0f};
    else out = {DistanceStatus::OK, cm};
    return true;
  }
  if (st == SONAR_WAIT_RISE && now - s_trig_us > SONAR_RISE_TIMEOUT_US) {
    s_state = SONAR_IDLE;
    out = {DistanceStatus::NO_RESPONSE, 0.0f};  // unplugged or dead sensor
    return true;
  }
  if (st == SONAR_WAIT_FALL && now - s_rise_us > SONAR_MAX_ECHO_US) {
    s_state = SONAR_IDLE;
    out = {DistanceStatus::OUT_OF_RANGE, 0.0f};
    return true;
  }
  return false;
}

// ---------------------------------------------------------------- servo

void EspServo::begin() { pwmAttach(board::PIN_SERVO, CH_SERVO, 50, SERVO_BITS); }

void EspServo::writeAngle(int degrees) {
  if (degrees < 0) degrees = 0;
  if (degrees > 180) degrees = 180;
  if (board::SERVO_REVERSED) degrees = 180 - degrees;
  const uint32_t us = (uint32_t)(board::SERVO_MIN_US + (board::SERVO_MAX_US - board::SERVO_MIN_US) * degrees / 180);
  const uint32_t duty = us * ((1u << SERVO_BITS) - 1) / 20000u;  // 20 ms period
  pwmWrite(board::PIN_SERVO, CH_SERVO, duty);
}

// ---------------------------------------------------------------- motors

void EspMotors::begin() {
  // Enable pins low first: the L298N must not run while the rest boots.
  const int pins[] = {board::PIN_LEFT_EN,  board::PIN_RIGHT_EN,  board::PIN_LEFT_IN1,
                      board::PIN_LEFT_IN2, board::PIN_RIGHT_IN3, board::PIN_RIGHT_IN4};
  for (int pin : pins) {
    pinMode(pin, OUTPUT);
    digitalWrite(pin, LOW);
  }
  pwmAttach(board::PIN_LEFT_EN, CH_LEFT, board::MOTOR_PWM_HZ, MOTOR_BITS);
  pwmAttach(board::PIN_RIGHT_EN, CH_RIGHT, board::MOTOR_PWM_HZ, MOTOR_BITS);
  pwmWrite(board::PIN_LEFT_EN, CH_LEFT, 0);
  pwmWrite(board::PIN_RIGHT_EN, CH_RIGHT, 0);
  mode_ = Mode::COAST;
}

void EspMotors::setSide(int en_pin, int in_a, int in_b, int channel, bool reversed, bool forward, int duty) {
  const bool dir = forward != reversed;
  digitalWrite(in_a, dir ? HIGH : LOW);
  digitalWrite(in_b, dir ? LOW : HIGH);
  pwmWrite(en_pin, channel, (uint32_t)duty);
}

void EspMotors::drive(int pwm, bool forward) {
  if (pwm <= 0) {
    brake();
    return;
  }
  if (pwm > 255) pwm = 255;
  setSide(board::PIN_LEFT_EN, board::PIN_LEFT_IN1, board::PIN_LEFT_IN2, CH_LEFT, board::LEFT_REVERSED, forward, pwm);
  setSide(board::PIN_RIGHT_EN, board::PIN_RIGHT_IN3, board::PIN_RIGHT_IN4, CH_RIGHT, board::RIGHT_REVERSED, forward,
          pwm);
  mode_ = Mode::DRIVE;
}

void EspMotors::brake() {
  if (mode_ != Mode::DRIVE) return;  // already braking or stopped
  // Both inputs high with the bridge enabled shorts each motor: a fast stop.
  digitalWrite(board::PIN_LEFT_IN1, HIGH);
  digitalWrite(board::PIN_LEFT_IN2, HIGH);
  digitalWrite(board::PIN_RIGHT_IN3, HIGH);
  digitalWrite(board::PIN_RIGHT_IN4, HIGH);
  pwmWrite(board::PIN_LEFT_EN, CH_LEFT, 255);
  pwmWrite(board::PIN_RIGHT_EN, CH_RIGHT, 255);
  mode_ = Mode::BRAKE;
  brake_start_ms_ = (uint32_t)::millis();
}

void EspMotors::service(uint32_t now_ms) {
  if (mode_ == Mode::BRAKE && now_ms - brake_start_ms_ >= board::BRAKE_ACTIVE_MS) {
    pwmWrite(board::PIN_LEFT_EN, CH_LEFT, 0);
    pwmWrite(board::PIN_RIGHT_EN, CH_RIGHT, 0);
    const int pins[] = {board::PIN_LEFT_IN1, board::PIN_LEFT_IN2, board::PIN_RIGHT_IN3, board::PIN_RIGHT_IN4};
    for (int pin : pins) digitalWrite(pin, LOW);
    mode_ = Mode::COAST;
  }
}

// ---------------------------------------------------------------- console

namespace {
WebDashboard g_web;
}

void EspConsole::begin() {
  Serial.setTxBufferSize(8192);  // printing must never wait for the UART
  Serial.begin(board::SERIAL_BAUD);
  // Newer cores report the free space of that buffer; older ones only the 128-byte
  // hardware FIFO, and then every long line would be dropped: fall back to plain writes.
  serial_space_known_ = Serial.availableForWrite() > 256;
  if (!g_cmd_queue) g_cmd_queue = xQueueCreate(8, CMD_LEN);
}

bool EspConsole::inject(const char* line) {
  if (!g_cmd_queue) return false;
  char item[CMD_LEN];
  strncpy(item, line, CMD_LEN - 1);
  item[CMD_LEN - 1] = '\0';
  return xQueueSend(g_cmd_queue, item, 0) == pdTRUE;
}

bool EspConsole::readLine(char* buf, int size) {
  char item[CMD_LEN];
  if (g_cmd_queue && xQueueReceive(g_cmd_queue, item, 0) == pdTRUE) {
    strncpy(buf, item, size - 1);
    buf[size - 1] = '\0';
    return true;
  }
  while (Serial.available() > 0) {
    const int c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serial_len_ == 0) continue;
      const int n = serial_len_ < size - 1 ? serial_len_ : size - 1;
      memcpy(buf, serial_line_, n);
      buf[n] = '\0';
      serial_len_ = 0;
      return true;
    }
    if (serial_len_ < (int)sizeof serial_line_ - 1) serial_line_[serial_len_++] = (char)c;
  }
  return false;
}

void EspConsole::print(const char* text) {
  const size_t len = strlen(text);
  if (!serial_space_known_ || (size_t)Serial.availableForWrite() >= len) Serial.write((const uint8_t*)text, len);
  else dropped_ += len;  // USB serial is behind: drop rather than stall the control loop

  if (!g_web.streaming()) {
    web_len_ = 0;
    return;
  }
  for (size_t i = 0; i < len; ++i) {
    const char c = text[i];
    if (c == '\n') {
      if (web_len_ > 0) g_web.publish(web_line_, web_len_);
      web_len_ = 0;
    } else if (c != '\r' && web_len_ < (int)sizeof web_line_) {
      web_line_[web_len_++] = c;
    }
  }
}

// ---------------------------------------------------------------- platform

void EspPlatform::watchdogBegin(uint32_t timeout_ms) {
#if CORE_V3
  esp_task_wdt_config_t c = {};
  c.timeout_ms = timeout_ms;
  c.idle_core_mask = 0;
  c.trigger_panic = true;
  if (esp_task_wdt_init(&c) != ESP_OK) esp_task_wdt_reconfigure(&c);  // already running: just change it
#else
  esp_task_wdt_init((timeout_ms + 999) / 1000, true);
#endif
  esp_task_wdt_add(nullptr);  // watch the task running loop()
}

void EspPlatform::watchdogFeed() { esp_task_wdt_reset(); }

const char* EspPlatform::resetReason() {
  switch (esp_reset_reason()) {
    case ESP_RST_POWERON: return "power-on";
    case ESP_RST_EXT: return "reset button";
    case ESP_RST_SW: return "software restart";
    case ESP_RST_PANIC: return "crash (panic)";
    case ESP_RST_INT_WDT:
    case ESP_RST_TASK_WDT:
    case ESP_RST_WDT: return "WATCHDOG (the loop stalled)";
    case ESP_RST_BROWNOUT: return "BROWN-OUT (supply voltage dipped: check the 5 V supply / servo power)";
    case ESP_RST_DEEPSLEEP: return "wake from deep sleep";
    default: return "unknown";
  }
}

// ---------------------------------------------------------------- storage

void EspStorage::begin() { ok_ = prefs_.begin("susp", false); }

bool EspStorage::load(const char* key, void* data, unsigned len) {
  if (!ok_ || prefs_.getBytesLength(key) != len) return false;
  return prefs_.getBytes(key, data, len) == len;
}

bool EspStorage::save(const char* key, const void* data, unsigned len) {
  return ok_ && prefs_.putBytes(key, data, len) == len;
}

// ---------------------------------------------------------------- button

void EspButton::begin() {
  if (board::PIN_BUTTON >= 0) pinMode(board::PIN_BUTTON, INPUT_PULLUP);
}

bool EspButton::pressed(uint32_t now_ms) {
  if (board::PIN_BUTTON < 0) return false;
  const bool raw = digitalRead(board::PIN_BUTTON) == HIGH;
  if (raw != last_raw_) {
    last_raw_ = raw;
    changed_ms_ = now_ms;
  }
  if (raw != stable_ && now_ms - changed_ms_ >= board::BUTTON_DEBOUNCE_MS) {
    stable_ = raw;
    return !stable_;  // went low = pressed
  }
  return false;
}

// ---------------------------------------------------------------- everything

void Esp32Hardware::begin() {
  motors.begin();  // first: motors off
  console.begin();
  servo.begin();
  sonar.begin();
  storage.begin();
  button.begin();
  if (board::WIFI_ENABLED) g_web.begin(console);
}

#endif  // ARDUINO_ARCH_ESP32
