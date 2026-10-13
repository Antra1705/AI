// obstacle_monitor.h - turns distance readings into a speed limit (BR-7, BR-8).
//
// Fail-safe rules (fixes I-6):
//  - no reading yet          -> speed 0
//  - any NO_RESPONSE         -> slow speed; DIST_FAIL_LIMIT in a row -> sensor fault, speed 0
//  - TOO_CLOSE (< 2 cm)      -> emergency stop (the SRA treated this as "no obstacle")
//  - OUT_OF_RANGE (> 400 cm) -> path clear
// The emergency stop releases only after ESTOP_RELEASE_READINGS readings beyond
// ESTOP_RELEASE_CM, so a reading hovering around 10 cm cannot make it chatter.
#pragma once
#include "../config.h"
#include "types.h"

class ObstacleMonitor {
 public:
  void reset() {
    have_reading_ = false;
    estop_ = false;
    clear_count_ = 0;
    no_response_count_ = 0;
    cm_ = 0.0f;
  }

  void update(const DistanceReading& r) {
    if (r.status == DistanceStatus::NO_RESPONSE) {
      ++no_response_count_;
      return;
    }
    no_response_count_ = 0;
    have_reading_ = true;
    if (r.status == DistanceStatus::TOO_CLOSE) {
      cm_ = 0.0f;
    } else if (r.status == DistanceStatus::OUT_OF_RANGE) {
      cm_ = cfg::DIST_MAX_CM + 1.0f;
    } else {
      cm_ = r.cm;
    }

    if (cm_ < cfg::ESTOP_CM) {
      estop_ = true;
      clear_count_ = 0;
    } else if (estop_) {
      if (cm_ >= cfg::ESTOP_RELEASE_CM) {
        if (++clear_count_ >= cfg::ESTOP_RELEASE_READINGS) {
          estop_ = false;
          clear_count_ = 0;
        }
      } else {
        clear_count_ = 0;
      }
    }
  }

  bool emergency() const { return estop_; }
  bool sensorFault() const { return no_response_count_ >= cfg::DIST_FAIL_LIMIT; }
  bool hasReading() const { return have_reading_; }
  float distanceCm() const { return cm_; }  // last valid distance (DIST_MAX_CM + 1 = clear)

  // Highest PWM the safety rules allow, without the 20 cm slow band (the fuzzy
  // controller shapes speed near obstacles itself).
  int safetyCap() const {
    if (!have_reading_ || sensorFault() || estop_) return 0;
    if (no_response_count_ > 0) return cfg::SPEED_SLOW;
    return cfg::SPEED_NORMAL;
  }

  // SRA step behaviour: normal, slow under SLOW_CM, stop under ESTOP_CM.
  int speedLimit() const {
    if (!have_reading_ || sensorFault() || estop_) return 0;
    if (no_response_count_ > 0 || cm_ < cfg::SLOW_CM) return cfg::SPEED_SLOW;
    return cfg::SPEED_NORMAL;
  }

 private:
  bool have_reading_ = false;
  bool estop_ = false;
  int clear_count_ = 0;
  int no_response_count_ = 0;
  float cm_ = 0.0f;
};
