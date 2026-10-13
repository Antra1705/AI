// classifier.h - rule-based road classifier: the simple fallback to the neural network
// (CLASSIFIER_MODE in config.h) and the baseline it is compared against.
#pragma once
#include "types.h"

// Thresholds are a struct (not just constants) so the replay tool can try
// different values against recorded data without recompiling.
struct ClassifierThresholds {
  float pothole_jerk;  // m/s^3
  float breaker_p2p;   // m/s^2
  float gravel_rms;    // m/s^2
  int gravel_osc;      // swings per window
  float smooth_jerk;   // m/s^3
  float smooth_p2p;    // m/s^2
};

ClassifierThresholds defaultThresholds();

// Rules, checked in this order (first match wins):
//   GRAVEL   rms > gravel_rms AND oscillations >= osc   (sustained vibration; checked
//                                                        first because gravel also jerks)
//   POTHOLE  jerk > pothole_jerk                        (one sharp impact)
//   BREAKER  peak-to-peak > breaker_p2p                 (one large, slower swing)
//   SMOOTH   jerk < smooth_jerk AND p2p < smooth_p2p
//   UNKNOWN  anything else
// Confidence grows with how far past the threshold the features are, so a
// reading sitting right on a threshold falls under the confidence gate.
Classification classify(const Features& f, const ClassifierThresholds& t);

// Servo angle for a classification. At or below CONFIDENCE_GATE -> neutral.
int servoAngleFor(const Classification& c);
