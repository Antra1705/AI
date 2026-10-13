// response_curve.h - classifier confidence -> graded damping command.
//
// This is the project's core contribution: the classifier output is mapped to
// a continuous stiffness s in [0, 1] (servo 45..150 deg) instead of a binary
// on/off or a fixed per-class angle.
//
//   1. Expected stiffness under the posterior   s_exp  = sum_c p(c) * S[c]
//        S = {smooth 0.0, gravel 0.55, breaker 0.9, pothole 1.0}
//   2. Effective confidence                     c_eff  = conf * 2^(-MI / MI_thr) * min(1, nov_thr / novelty)
//        (members disagreeing or an unfamiliar input both lower trust)
//   3. Confidence gate (smoothstep)             g      = smoothstep(c_eff; C_LOW, C_HIGH)
//   4. Blend toward a safe middle setting       s_base = S_SAFE + g * (s_exp - S_SAFE)
//   5. Protective floor                         s_risk = RISK_GAIN * (p(breaker) + p(pothole))
//        (impact classes stiffen even when the classifier is unsure: protect first)
//   6. Body-motion term from the EKF ("skyhook") s_sky = SKY_GAIN * min(|vz| / V_REF, 1)
//   s = clamp(max(s_base, s_risk) + s_sky, 0, 1)
//
// ActuatorShaper then turns s into servo commands the MG90S can actually
// follow: stiffening at the servo's full slew rate, softening slowly, a small
// deadband against jitter, and an impact reflex that jumps straight to full
// stiffness for a hard hit.
#pragma once
#include <stdint.h>

#include "../config.h"

struct CurveInput {
  bool valid;         // false while the classifier is warming up / faulted
  float p[4];         // class probabilities (smooth, gravel, breaker, pothole)
  float confidence;   // max p
  float mutual_info;  // ensemble disagreement (0 for the rule-based classifier)
  float novelty;      // Mahalanobis novelty (0 if unavailable)
  float vz;           // EKF vertical velocity, m/s
};

struct CurveThresholds {
  float mi;       // MI at which trust halves
  float novelty;  // novelty beyond which trust decays (0 = ignore novelty)
};

struct CurveOutput {
  float s;         // final stiffness 0..1
  float s_exp;     // expected stiffness under the posterior
  float c_eff;     // effective confidence
  float gate;      // 0..1 how much the classification is trusted
  float s_risk;    // protective floor
  float s_sky;     // body-motion term
  float angle;     // servo angle for s
};

CurveOutput responseCurve(const CurveInput& in, const CurveThresholds& th);

float stiffnessToAngle(float s);
float smoothstep01(float x);

class ActuatorShaper {
 public:
  ActuatorShaper() { reset((float)cfg::SERVO_NEUTRAL); }
  void reset(float angle);
  void setTarget(float angle) { target_ = angle; }
  void impact(uint32_t now_ms);  // hard hit: full stiffness now, hold for IMPACT_HOLD_MS

  // Advance by dt seconds; returns the angle to command (whole degrees).
  int update(float dt, uint32_t now_ms);

  float target() const { return target_; }
  float position() const { return pos_; }
  bool holding() const { return holding_; }

 private:
  float target_ = 90.0f;
  float pos_ = 90.0f;
  int out_ = 90;
  bool holding_ = false;
  uint32_t hold_until_ms_ = 0;
};

// Debounced road state for display and maintenance: a new surface (or
// "uncertain") must be seen SURFACE_CONFIRM times in a row before it is reported.
class SurfaceTracker {
 public:
  int update(int top, bool uncertain) {
    int cand = uncertain ? 4 : top;  // 4 = SURFACE_UNKNOWN
    if (cand == state_) {
      count_ = 0;
    } else if (cand == pending_) {
      if (++count_ >= cfg::SURFACE_CONFIRM) {
        state_ = cand;
        count_ = 0;
      }
    } else {
      pending_ = cand;
      count_ = 1;
      if (count_ >= cfg::SURFACE_CONFIRM) state_ = cand;
    }
    return state_;
  }
  int state() const { return state_; }
  void reset() {
    state_ = 4;
    pending_ = 4;
    count_ = 0;
  }

 private:
  int state_ = 4;
  int pending_ = 4;
  int count_ = 0;
};
