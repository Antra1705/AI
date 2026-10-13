// fake_hal.h - simulated hardware so the control logic runs on a computer.
#pragma once
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "config.h"
#include "core/hal.h"

// Time is kept in 64 bits; micros()/millis() wrap exactly like the ESP32's.
class FakeClock : public IClock {
 public:
  uint64_t t_us = 0;
  uint32_t micros() override { return (uint32_t)t_us; }
  uint32_t millis() override { return (uint32_t)(t_us / 1000); }
  void advanceUs(uint64_t d) { t_us += d; }
};

class FakeImu : public IImu {
 public:
  explicit FakeImu(FakeClock& c) : clock(c) {}
  FakeClock& clock;
  bool present = true;  // false = unplugged
  int begin_calls = 0;
  int reads = 0;
  // Fills acceleration/gyro for time t_us. Default: level and still.
  std::function<void(uint64_t t_us, ImuSample&)> source;

  bool begin() override {
    ++begin_calls;
    return present;
  }
  bool read(ImuSample& s) override {
    if (!present) return false;
    ++reads;
    s.ax = 0.0f; s.ay = 0.0f; s.az = 9.81f;
    s.gx = 0.0f; s.gy = 0.0f; s.gz = 0.0f;
    if (source) source(clock.t_us, s);
    return true;
  }
};

// Mimics a non-blocking HC-SR04 driver: the result arrives after the echo's
// round-trip time; a missing sensor reports NO_RESPONSE after a 30 ms timeout.
class FakeDistance : public IDistanceSensor {
 public:
  explicit FakeDistance(FakeClock& c) : clock(c) {}
  FakeClock& clock;
  std::function<DistanceReading(uint64_t t_us)> source;  // default: open space
  int triggers = 0;

  void trigger() override {
    ++triggers;
    pending_ = source ? source(clock.t_us) : DistanceReading{DistanceStatus::OUT_OF_RANGE, 0.0f};
    uint64_t delay_us = 30000;  // timeout for NO_RESPONSE / OUT_OF_RANGE
    if (pending_.status == DistanceStatus::OK) delay_us = 500 + (uint64_t)(pending_.cm * 58.0f);
    if (pending_.status == DistanceStatus::TOO_CLOSE) delay_us = 500;
    ready_at_ = clock.t_us + delay_us;
    in_flight_ = true;
  }
  bool poll(DistanceReading& out) override {
    if (!in_flight_ || clock.t_us < ready_at_) return false;
    in_flight_ = false;
    out = pending_;
    return true;
  }

 private:
  DistanceReading pending_{DistanceStatus::OUT_OF_RANGE, 0.0f};
  uint64_t ready_at_ = 0;
  bool in_flight_ = false;
};

class FakeServo : public IServo {
 public:
  explicit FakeServo(FakeClock& c) : clock(c) {}
  FakeClock& clock;
  int angle = -1;
  std::vector<std::pair<uint64_t, int>> history;  // (time us, angle)
  void writeAngle(int degrees) override {
    angle = degrees;
    history.push_back(std::make_pair(clock.t_us, degrees));
  }
};

class FakeMotors : public IMotors {
 public:
  explicit FakeMotors(FakeClock& c) : clock(c) {}
  FakeClock& clock;
  int pwm = 0;
  bool forward = true;
  bool braked = true;
  std::vector<std::pair<uint64_t, int>> history;  // (time us, pwm), 0 = brake
  void drive(int p, bool fwd) override {
    pwm = p;
    forward = fwd;
    braked = false;
    history.push_back(std::make_pair(clock.t_us, p));
  }
  void brake() override {
    pwm = 0;
    braked = true;
    history.push_back(std::make_pair(clock.t_us, 0));
  }
};

class FakeConsole : public IConsole {
 public:
  std::deque<std::string> input;
  std::string output;
  bool echo = false;  // also print to stdout

  bool readLine(char* buf, int size) override {
    if (input.empty()) return false;
    std::strncpy(buf, input.front().c_str(), (size_t)size - 1);
    buf[size - 1] = '\0';
    input.pop_front();
    return true;
  }
  void print(const char* text) override {
    output += text;
    if (echo) std::fputs(text, stdout);
  }
  bool contains(const std::string& s) const { return output.find(s) != std::string::npos; }
  int countLinesStartingWith(const std::string& prefix) const {
    int n = 0;
    size_t pos = 0;
    while (pos < output.size()) {
      size_t end = output.find('\n', pos);
      if (end == std::string::npos) end = output.size();
      if (output.compare(pos, prefix.size(), prefix) == 0) ++n;
      pos = end + 1;
    }
    return n;
  }
};

class FakePlatform : public IPlatform {
 public:
  uint32_t watchdog_timeout_ms = 0;
  uint64_t feeds = 0;
  void watchdogBegin(uint32_t timeout_ms) override { watchdog_timeout_ms = timeout_ms; }
  void watchdogFeed() override { ++feeds; }
  const char* resetReason() override { return "SIMULATED"; }
};

class FakeStorage : public IStorage {
 public:
  std::map<std::string, std::vector<uint8_t>> blobs;
  bool load(const char* key, void* data, unsigned len) override {
    auto it = blobs.find(key);
    if (it == blobs.end() || it->second.size() != len) return false;
    std::memcpy(data, it->second.data(), len);
    return true;
  }
  bool save(const char* key, const void* data, unsigned len) override {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    blobs[key] = std::vector<uint8_t>(p, p + len);
    return true;
  }
};

// All fakes wired together, plus a helper to run the main loop.
struct FakeRig {
  FakeClock clock;
  FakeImu imu{clock};
  FakeDistance distance{clock};
  FakeServo servo{clock};
  FakeMotors motors{clock};
  FakeConsole console;
  FakePlatform platform;
  FakeStorage storage;
  Hal hal{clock, imu, distance, servo, motors, console, platform, storage};

  // Call tick() every step_us until duration_us of simulated time has passed.
  template <typename AppT>
  void run(AppT& app, uint64_t duration_us, uint32_t step_us = 1000) {
    uint64_t end = clock.t_us + duration_us;
    while (clock.t_us < end) {
      app.tick();
      clock.advanceUs(step_us);
    }
  }
};
