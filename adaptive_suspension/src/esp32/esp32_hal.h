// esp32_hal.h - the real-hardware side of hal.h (ESP32 + Arduino core 2.x or 3.x).
//
// The control logic in core/ is identical on the cart and in the simulator; only
// these drivers differ. Pins and wiring options: board_config.h.
//
// Timing rule: nothing here may block the control loop. Serial output is dropped
// rather than waited for, the sonar is interrupt-driven, and WiFi runs in its own
// task on the other CPU core.
#pragma once
#ifdef ARDUINO_ARCH_ESP32

#include <Arduino.h>
#include <Preferences.h>

#include "../core/hal.h"

class EspClock : public IClock {
 public:
  uint32_t micros() override { return (uint32_t)::micros(); }
  uint32_t millis() override { return (uint32_t)::millis(); }
};

// MPU-6050 over I2C, 100 Hz sample rate, +-8 g, +-500 deg/s.
class EspImu : public IImu {
 public:
  bool begin() override;
  bool read(ImuSample& out) override;

 private:
  bool writeReg(uint8_t reg, uint8_t value);
  bool readRegs(uint8_t reg, uint8_t* buf, int n);
  bool wire_started_ = false;
};

// HC-SR04: trigger() sends the pulse, an interrupt times the echo, poll() reports.
class EspSonar : public IDistanceSensor {
 public:
  void begin();
  void trigger() override;
  bool poll(DistanceReading& out) override;
};

class EspServo : public IServo {
 public:
  void begin();
  void writeAngle(int degrees) override;
};

// L298N, both motors together. brake() shorts the motors briefly, then coasts.
class EspMotors : public IMotors {
 public:
  void begin();  // call first in setup(): motors off before anything else
  void drive(int pwm, bool forward) override;
  void brake() override;
  void service(uint32_t now_ms);  // from loop(): ends the active brake

 private:
  enum class Mode { COAST, DRIVE, BRAKE };
  void setSide(int en_pin, int in_a, int in_b, int channel, bool reversed, bool forward, int duty);
  Mode mode_ = Mode::COAST;
  uint32_t brake_start_ms_ = 0;
};

// Commands come from USB serial, the WiFi dashboard and the push button.
// Output goes to USB serial and to any browser watching the dashboard.
class EspConsole : public IConsole {
 public:
  void begin();
  bool readLine(char* buf, int size) override;
  void print(const char* text) override;
  bool inject(const char* line);  // queue a command (thread-safe)
  uint32_t droppedBytes() const { return dropped_; }

 private:
  char serial_line_[96];
  int serial_len_ = 0;
  char web_line_[1200];
  int web_len_ = 0;
  uint32_t dropped_ = 0;
  bool serial_space_known_ = false;
};

class EspPlatform : public IPlatform {
 public:
  void watchdogBegin(uint32_t timeout_ms) override;
  void watchdogFeed() override;
  const char* resetReason() override;
};

// NVS flash (Preferences), survives power-off and reflashing the sketch.
class EspStorage : public IStorage {
 public:
  void begin();
  bool load(const char* key, void* data, unsigned len) override;
  bool save(const char* key, const void* data, unsigned len) override;

 private:
  Preferences prefs_;
  bool ok_ = false;
};

// WiFi access point + dashboard (the same page as host/dashboard): GET / serves
// it, GET /events streams the cart's output, POST /cmd sends a command.
class WebDashboard {
 public:
  void begin(EspConsole& console);
  bool streaming() const;
  void publish(const char* line, int len);  // from the control loop; never blocks
};

// Debounced start/stop button.
class EspButton {
 public:
  void begin();
  bool pressed(uint32_t now_ms);  // true once per press

 private:
  bool stable_ = true;   // pulled up = not pressed
  bool last_raw_ = true;
  uint32_t changed_ms_ = 0;
};

struct Esp32Hardware {
  EspClock clock;
  EspImu imu;
  EspSonar sonar;
  EspServo servo;
  EspMotors motors;
  EspConsole console;
  EspPlatform platform;
  EspStorage storage;
  EspButton button;
  Hal hal{clock, imu, sonar, servo, motors, console, platform, storage};

  void begin();  // everything except App: motors safe, serial, sensors, WiFi
};

#endif  // ARDUINO_ARCH_ESP32
