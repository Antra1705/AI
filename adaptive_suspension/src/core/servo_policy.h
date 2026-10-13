// servo_policy.h - decides when the servo actually moves.
//
// "Protect fast, relax slowly":
//  - a stiffer setting is applied immediately;
//  - a softer setting must be requested SOFTEN_CONFIRM_CYCLES times in a row,
//    so one ambiguous window does not make the servo jitter (BR-5);
//  - impact() jumps to maximum stiffness at once and holds it for IMPACT_HOLD_MS.
#pragma once
#include <stdint.h>

#include "../config.h"

class ServoPolicy {
 public:
  ServoPolicy() { reset(cfg::SERVO_NEUTRAL); }

  void reset(int angle) {
    angle_ = angle;
    pending_ = angle;
    pending_count_ = 0;
    holding_ = false;
  }

  void impact(uint32_t now_ms) {
    angle_ = cfg::SERVO_IMPACT;
    hold_until_ms_ = now_ms + cfg::IMPACT_HOLD_MS;
    holding_ = true;
    pending_count_ = 0;
  }

  // Call once per control cycle with the wanted angle. Returns the angle to command.
  int update(int target, uint32_t now_ms) {
    if (holding_) {
      if ((int32_t)(now_ms - hold_until_ms_) < 0) return angle_;
      holding_ = false;
    }
    if (target >= angle_) {  // stiffer or same: apply now
      angle_ = target;
      pending_count_ = 0;
      return angle_;
    }
    if (target == pending_) {
      ++pending_count_;
    } else {
      pending_ = target;
      pending_count_ = 1;
    }
    if (pending_count_ >= cfg::SOFTEN_CONFIRM_CYCLES) {
      angle_ = target;
      pending_count_ = 0;
    }
    return angle_;
  }

  int angle() const { return angle_; }
  bool holding() const { return holding_; }

 private:
  int angle_ = cfg::SERVO_NEUTRAL;
  int pending_ = cfg::SERVO_NEUTRAL;
  int pending_count_ = 0;
  bool holding_ = false;
  uint32_t hold_until_ms_ = 0;
};
