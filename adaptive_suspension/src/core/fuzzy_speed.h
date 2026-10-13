// fuzzy_speed.h - obstacle/surface fusion -> smooth drive speed (#5).
//
// Inputs
//   distance  cm to the obstacle ahead (DIST_MAX_CM + 1 when the path is clear)
//   closing   cm/s, how fast that distance shrinks (from RangeTracker)
//   rough     0..1, 1 - P(smooth) from the classifier; an unsure classifier
//             spreads probability away from "smooth", so uncertainty also slows the cart
// Output
//   u         0..1 speed factor (0 = stop, 1 = normal speed)
//
// Interval Type-2: every membership function has a footprint of uncertainty
// (an upper and a lower curve, blur = how unsure we are about its edges,
// e.g. ultrasonic noise). Each of the 12 rules fires with an interval
// [lower, upper]; center-of-sets type reduction (exact Karnik-Mendel switch
// points, found by trying every split since there are only 12 rules) gives
// [y_l, y_r], and u = (y_l + y_r) / 2.
// Type-1 (the simplified fallback) uses only the middle curves.
//
// The fuzzy controller only shapes speed inside the safety envelope: the hard
// emergency stop (< ESTOP_CM), sensor-fault stop and no-echo slow-down in
// ObstacleMonitor always take priority.
#pragma once
#include "types.h"

enum class SpeedMode : uint8_t { STEP, FUZZY_T1, FUZZY_IT2 };

struct FuzzyResult {
  float u;         // speed factor 0..1
  float y_l, y_r;  // type-reduced interval (equal for Type-1)
  float firing;    // total upper firing strength (0 = no rule fired)
};

// blur scales every footprint of uncertainty (0 turns IT2 into Type-1).
FuzzyResult fuzzySpeed(float distance_cm, float closing_cms, float rough, SpeedMode mode, float blur = 1.0f);

const char* speedModeName(SpeedMode m);

// Alpha-beta filter on ultrasonic readings: smoothed distance and closing speed.
class RangeTracker {
 public:
  void reset() {
    valid_ = false;
    d_ = 0.0f;
    v_ = 0.0f;
  }
  // cm < 0 means "path clear" (out of range). Call only for answered readings.
  void update(float cm, float dt_s);
  bool valid() const { return valid_; }
  float distance() const { return d_; }
  float closingSpeed() const { return v_; }  // cm/s, positive = getting closer

 private:
  bool valid_ = false;
  float d_ = 0.0f;
  float v_ = 0.0f;
};
