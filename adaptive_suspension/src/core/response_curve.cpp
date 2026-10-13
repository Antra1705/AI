#include "response_curve.h"

#include <math.h>

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

float smoothstep01(float x) {
  x = clampf(x, 0.0f, 1.0f);
  return x * x * (3.0f - 2.0f * x);
}

float stiffnessToAngle(float s) {
  return (float)cfg::SERVO_SOFT_ANGLE + clampf(s, 0.0f, 1.0f) * (float)(cfg::SERVO_STIFF_ANGLE - cfg::SERVO_SOFT_ANGLE);
}

CurveOutput responseCurve(const CurveInput& in, const CurveThresholds& th) {
  static const float S[4] = {cfg::STIFF_SMOOTH, cfg::STIFF_GRAVEL, cfg::STIFF_BREAKER, cfg::STIFF_POTHOLE};
  CurveOutput o;
  o.s_exp = 0.0f;
  for (int c = 0; c < 4; ++c) o.s_exp += in.p[c] * S[c];

  if (!in.valid) {
    o.c_eff = 0.0f;
  } else {
    float trust = in.confidence;
    if (th.mi > 0.0f) trust *= powf(2.0f, -in.mutual_info / th.mi);
    if (th.novelty > 0.0f && in.novelty > th.novelty) trust *= th.novelty / in.novelty;
    o.c_eff = trust;
  }
  o.gate = smoothstep01((o.c_eff - cfg::CURVE_C_LOW) / (cfg::CURVE_C_HIGH - cfg::CURVE_C_LOW));
  const float s_base = cfg::STIFF_SAFE + o.gate * (o.s_exp - cfg::STIFF_SAFE);
  o.s_risk = in.valid ? cfg::CURVE_RISK_GAIN * (in.p[2] + in.p[3]) : 0.0f;
  // Body-motion damping, except on gravel: there the body vibrates all the time and
  // extra stiffness only transmits more of it (measured: worse comfort, more servo wear).
  const float not_gravel = in.valid ? 1.0f - cfg::CURVE_SKY_GRAVEL_GATE * in.p[1] : 1.0f;
  o.s_sky = cfg::CURVE_SKY_GAIN * not_gravel * clampf(fabsf(in.vz) / cfg::CURVE_SKY_VREF, 0.0f, 1.0f);
  o.s = clampf((s_base > o.s_risk ? s_base : o.s_risk) + o.s_sky, 0.0f, 1.0f);
  o.angle = stiffnessToAngle(o.s);
  return o;
}

// ---------------------------------------------------------------- ActuatorShaper

void ActuatorShaper::reset(float angle) {
  target_ = angle;
  pos_ = angle;
  out_ = (int)floorf(angle + 0.5f);
  holding_ = false;
}

void ActuatorShaper::impact(uint32_t now_ms) {
  holding_ = true;
  hold_until_ms_ = now_ms + cfg::IMPACT_HOLD_MS;
  pos_ = (float)cfg::SERVO_IMPACT;  // jump: the servo moves at its own top speed
}

int ActuatorShaper::update(float dt, uint32_t now_ms) {
  float goal = target_;
  if (holding_) {
    if ((int32_t)(now_ms - hold_until_ms_) < 0) {
      if (goal < (float)cfg::SERVO_IMPACT) goal = (float)cfg::SERVO_IMPACT;
    } else {
      holding_ = false;
    }
  }
  // Stiffen as fast as the servo can move; soften slowly ("protect fast, relax slowly").
  float up = cfg::SLEW_UP_DEG_S * dt, down = cfg::SLEW_DOWN_DEG_S * dt;
  float diff = goal - pos_;
  if (diff > up) diff = up;
  if (diff < -down) diff = -down;
  pos_ += diff;

  int rounded = (int)floorf(pos_ + 0.5f);
  int goal_rounded = (int)floorf(goal + 0.5f);
  int moved = rounded - out_;
  if (moved < 0) moved = -moved;
  if (moved >= cfg::SERVO_DEADBAND_DEG || rounded == goal_rounded) out_ = rounded;
  return out_;
}
