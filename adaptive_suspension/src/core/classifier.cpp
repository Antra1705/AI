#include "classifier.h"

#include "../config.h"

ClassifierThresholds defaultThresholds() {
  ClassifierThresholds t;
  t.pothole_jerk = cfg::POTHOLE_JERK;
  t.breaker_p2p = cfg::BREAKER_P2P;
  t.gravel_rms = cfg::GRAVEL_RMS;
  t.gravel_osc = cfg::GRAVEL_OSC;
  t.smooth_jerk = cfg::SMOOTH_JERK;
  t.smooth_p2p = cfg::SMOOTH_P2P;
  return t;
}

static float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

// 0 right at the threshold, 1 when the value is double (or zero) of it.
static float marginAbove(float v, float thr) { return thr > 0.0f ? clamp01((v - thr) / thr) : 1.0f; }
static float marginBelow(float v, float thr) { return thr > 0.0f ? clamp01((thr - v) / thr) : 0.0f; }
static float minf(float a, float b) { return a < b ? a : b; }

static Classification make(Surface s, float margin) {
  Classification c;
  c.surface = s;
  c.confidence = cfg::CONF_MIN + (cfg::CONF_MAX - cfg::CONF_MIN) * margin;
  return c;
}

Classification classify(const Features& f, const ClassifierThresholds& t) {
  if (f.accel_rms_dev > t.gravel_rms && f.oscillations >= t.gravel_osc) {
    float m_osc = t.gravel_osc > 0 ? clamp01((float)(f.oscillations - t.gravel_osc) / (float)t.gravel_osc) : 1.0f;
    return make(SURFACE_GRAVEL, minf(marginAbove(f.accel_rms_dev, t.gravel_rms), m_osc));
  }
  if (f.jerk_max > t.pothole_jerk) {
    return make(SURFACE_POTHOLE, marginAbove(f.jerk_max, t.pothole_jerk));
  }
  if (f.peak_to_peak > t.breaker_p2p) {
    return make(SURFACE_BREAKER, marginAbove(f.peak_to_peak, t.breaker_p2p));
  }
  if (f.jerk_max < t.smooth_jerk && f.peak_to_peak < t.smooth_p2p) {
    return make(SURFACE_SMOOTH, minf(marginBelow(f.jerk_max, t.smooth_jerk), marginBelow(f.peak_to_peak, t.smooth_p2p)));
  }
  Classification c;
  c.surface = SURFACE_UNKNOWN;
  c.confidence = cfg::CONF_UNKNOWN;
  return c;
}

int servoAngleFor(const Classification& c) {
  if (c.confidence <= cfg::CONFIDENCE_GATE) return cfg::SERVO_NEUTRAL;
  switch (c.surface) {
    case SURFACE_SMOOTH: return cfg::SERVO_SMOOTH;
    case SURFACE_GRAVEL: return cfg::SERVO_GRAVEL;
    case SURFACE_BREAKER: return cfg::SERVO_BREAKER;
    case SURFACE_POTHOLE: return cfg::SERVO_POTHOLE;
    default: return cfg::SERVO_NEUTRAL;
  }
}
