#include "config.h"
#include "core/classifier.h"
#include "test.h"

static Features feat(float jerk, float p2p, float rms, int osc) {
  Features f = {9.81f, jerk, p2p, rms, 0.0f, osc};
  return f;
}

static const ClassifierThresholds T = defaultThresholds();

TEST(classify_smooth) {
  Classification c = classify(feat(10, 0.2f, 0.05f, 0), T);
  CHECK_EQ(c.surface, SURFACE_SMOOTH);
  CHECK(c.confidence > cfg::CONFIDENCE_GATE);
}

TEST(classify_pothole_on_sharp_jerk) {
  Classification c = classify(feat(T.pothole_jerk * 2, 10.0f, T.gravel_rms * 0.5f, 2), T);
  CHECK_EQ(c.surface, SURFACE_POTHOLE);
  CHECK_NEAR(c.confidence, cfg::CONF_MAX, 1e-6);
}

TEST(classify_gravel_checked_before_pothole) {
  // Gravel also produces big jerks; sustained vibration identifies it first.
  Classification c = classify(feat(T.pothole_jerk * 2, 10.0f, T.gravel_rms * 2, T.gravel_osc * 2), T);
  CHECK_EQ(c.surface, SURFACE_GRAVEL);
}

TEST(classify_gravel_needs_rms_and_oscillations) {
  CHECK_EQ(classify(feat(150, 2.0f, T.gravel_rms * 2, T.gravel_osc * 2), T).surface, SURFACE_GRAVEL);
  // Big single swing, few oscillations -> bump, not gravel.
  CHECK_EQ(classify(feat(T.pothole_jerk * 0.5f, T.breaker_p2p * 2, T.gravel_rms * 2, 2), T).surface, SURFACE_BREAKER);
  // Many small oscillations, low rms -> not gravel.
  CHECK(classify(feat(T.pothole_jerk * 0.5f, 0.8f, T.gravel_rms * 0.5f, 20), T).surface != SURFACE_GRAVEL);
}

TEST(classify_bump) {
  Classification c = classify(feat(T.pothole_jerk * 0.5f, T.breaker_p2p * 1.5f, T.gravel_rms * 0.5f, 2), T);
  CHECK_EQ(c.surface, SURFACE_BREAKER);
}

TEST(classify_unknown_between_classes) {
  // Too much jerk for smooth, too little of anything else.
  Classification c = classify(
      feat(0.5f * (T.smooth_jerk + T.pothole_jerk), 0.9f * T.breaker_p2p, T.gravel_rms * 0.5f, 2), T);
  CHECK_EQ(c.surface, SURFACE_UNKNOWN);
  CHECK_NEAR(c.confidence, cfg::CONF_UNKNOWN, 1e-6);
}

TEST(classify_thresholds_are_strict) {
  // Exactly on the pothole threshold is NOT a pothole (SRA uses '>').
  CHECK(classify(feat(T.pothole_jerk, 0.5f, 0.1f, 0), T).surface != SURFACE_POTHOLE);
  CHECK_EQ(classify(feat(T.pothole_jerk + 1, 0.5f, 0.1f, 0), T).surface, SURFACE_POTHOLE);
}

TEST(confidence_grows_with_margin) {
  float near_thr = classify(feat(T.pothole_jerk * 1.01f, 1, 0.1f, 0), T).confidence;
  float far_thr = classify(feat(T.pothole_jerk * 1.8f, 1, 0.1f, 0), T).confidence;
  CHECK(near_thr < far_thr);
  CHECK(near_thr <= cfg::CONFIDENCE_GATE);  // barely past a threshold -> not trusted
  CHECK(far_thr > cfg::CONFIDENCE_GATE);
}

TEST(servo_angle_mapping_and_gate) {
  Classification c = {SURFACE_SMOOTH, 0.9f};
  CHECK_EQ(servoAngleFor(c), cfg::SERVO_SMOOTH);
  c.surface = SURFACE_GRAVEL;
  CHECK_EQ(servoAngleFor(c), cfg::SERVO_GRAVEL);
  c.surface = SURFACE_BREAKER;
  CHECK_EQ(servoAngleFor(c), cfg::SERVO_BREAKER);
  c.surface = SURFACE_POTHOLE;
  CHECK_EQ(servoAngleFor(c), cfg::SERVO_POTHOLE);
  c.surface = SURFACE_UNKNOWN;
  CHECK_EQ(servoAngleFor(c), cfg::SERVO_NEUTRAL);

  c.surface = SURFACE_POTHOLE;
  c.confidence = cfg::CONFIDENCE_GATE;  // gate is "> 0.60"
  CHECK_EQ(servoAngleFor(c), cfg::SERVO_NEUTRAL);
  c.confidence = 0.3f;
  CHECK_EQ(servoAngleFor(c), cfg::SERVO_NEUTRAL);
}

TEST(custom_thresholds_are_used) {
  ClassifierThresholds t = T;
  t.pothole_jerk = T.pothole_jerk * 2.0f;
  Features f = feat(T.pothole_jerk * 1.5f, 0.5f, 0.1f, 0);
  CHECK(classify(f, T).surface == SURFACE_POTHOLE);
  CHECK(classify(f, t).surface != SURFACE_POTHOLE);
}
