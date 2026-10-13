// hal.h - the boundary between the control logic and the hardware.
//
// The core logic only talks to these interfaces. On the Mac they are
// implemented by fakes (host/fakes) so everything can be tested without the
// cart. The real ESP32 implementations are added once the hardware is here.
#pragma once
#include <stdint.h>

#include "types.h"

class IClock {
 public:
  virtual ~IClock() {}
  virtual uint32_t micros() = 0;  // wraps every ~71 minutes
  virtual uint32_t millis() = 0;
};

class IImu {
 public:
  virtual ~IImu() {}
  // Configure the sensor. Returns false if it is not responding.
  virtual bool begin() = 0;
  // Fill acceleration and gyro fields (not t_us). Returns false on a failed read.
  virtual bool read(ImuSample& out) = 0;
};

// Non-blocking distance sensor: trigger() starts a measurement, poll() returns
// true once that measurement is finished (result or timeout).
class IDistanceSensor {
 public:
  virtual ~IDistanceSensor() {}
  virtual void trigger() = 0;
  virtual bool poll(DistanceReading& out) = 0;
};

class IServo {
 public:
  virtual ~IServo() {}
  virtual void writeAngle(int degrees) = 0;  // 0-180
};

class IMotors {
 public:
  virtual ~IMotors() {}
  virtual void drive(int pwm, bool forward) = 0;  // both motors, pwm 1-255
  virtual void brake() = 0;                       // both motors stopped
};

class IConsole {
 public:
  virtual ~IConsole() {}
  // Copies one complete input line (without newline) into buf. Returns false if none is ready.
  virtual bool readLine(char* buf, int size) = 0;
  virtual void print(const char* text) = 0;
};

class IPlatform {
 public:
  virtual ~IPlatform() {}
  virtual void watchdogBegin(uint32_t timeout_ms) = 0;
  virtual void watchdogFeed() = 0;
  virtual const char* resetReason() = 0;
};

// Small persistent key/value store (NVS flash on the ESP32).
class IStorage {
 public:
  virtual ~IStorage() {}
  virtual bool load(const char* key, void* data, unsigned len) = 0;  // false if missing or wrong size
  virtual bool save(const char* key, const void* data, unsigned len) = 0;
};

struct Hal {
  IClock& clock;
  IImu& imu;
  IDistanceSensor& distance;
  IServo& servo;
  IMotors& motors;
  IConsole& console;
  IPlatform& platform;
  IStorage& storage;
};
