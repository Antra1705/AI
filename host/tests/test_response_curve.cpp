#include "config.h"
#include "core/response_curve.h"
#include "test.h"

static CurveInput input(int cls, float conf) {
  CurveInput in = {true, {0, 0, 0, 0}, conf, 0.0f, 0.0f, 0.0f};
  for (int c = 0; c < 4; ++c) in.p[c] = (c == cls) ? conf : (1.0f - conf) / 3.0f;
  return in;
}
static const CurveThresholds TH = {0.02f, 50.0f};

TEST(curve_confident_classes_hit_their_targets) {
  CHECK_NEAR(responseCurve(input(0, 0.99f), TH).s, cfg::STIFF_SMOOTH, 0.05);
  CHECK_NEAR(responseCurve(input(1, 0.99f), TH).s, cfg::STIFF_GRAVEL, 0.05);
  CHECK_NEAR(responseCurve(input(3, 0.99f), TH).s, cfg::STIFF_POTHOLE, 0.01);
  CHECK_NEAR(responseCurve(input(0, 0.99f), TH).angle, stiffnessToAngle(cfg::STIFF_SMOOTH), 6);
}

TEST(curve_unsure_goes_to_safe_middle) {
  CurveInput in = input(0, 0.3f);
  CHECK_NEAR(responseCurve(in, TH).s, cfg::STIFF_SAFE, 0.05);
  in.valid = false;
  CHECK_NEAR(responseCurve(in, TH).s, cfg::STIFF_SAFE, 1e-6);
}

// Graded, not binary: as confidence in gravel (target below the safe middle)
// grows, stiffness moves smoothly from the safe setting down to the gravel target.
TEST(curve_is_monotonic_and_continuous_in_confidence) {
  CHECK(cfg::STIFF_GRAVEL < cfg::STIFF_SAFE);
  float prev = 2.0f;
  for (int i = 30; i <= 100; ++i) {
    float s = responseCurve(input(1, i / 100.0f), TH).s;
    CHECK(s <= prev + 1e-6f);                  // more sure of gravel -> never stiffer
    if (prev < 1.5f) CHECK(prev - s < 0.06f);  // no jumps
    prev = s;
  }
  CHECK_NEAR(prev, cfg::STIFF_GRAVEL, 0.05);
}

TEST(curve_member_disagreement_lowers_trust) {
  CurveInput a = input(1, 0.95f), b = a;
  b.mutual_info = 2 * TH.mi;
  CHECK(responseCurve(b, TH).c_eff < 0.5f * responseCurve(a, TH).c_eff);
  CHECK(responseCurve(b, TH).s > responseCurve(a, TH).s);  // less trust in "soft gravel" -> back toward safe
}

TEST(curve_novel_input_lowers_trust) {
  CurveInput a = input(0, 0.95f), b = a;
  b.novelty = 4 * TH.novelty;
  CHECK_NEAR(responseCurve(b, TH).c_eff, 0.25 * responseCurve(a, TH).c_eff, 1e-5);
}

TEST(curve_risk_floor_protects_even_when_unsure) {
  CurveInput in = {true, {0.3f, 0.1f, 0.1f, 0.5f}, 0.5f, 0.0f, 0.0f, 0.0f};  // probably pothole, not trusted
  CurveOutput o = responseCurve(in, TH);
  CHECK(o.gate < 0.1f);
  CHECK_NEAR(o.s, cfg::CURVE_RISK_GAIN * 0.6f, 1e-5);
  CHECK(o.s > cfg::STIFF_SAFE);
}

TEST(curve_body_bounce_adds_damping) {
  CurveInput in = input(0, 0.99f);
  float calm = responseCurve(in, TH).s;
  in.vz = 0.2f;
  CHECK_NEAR(responseCurve(in, TH).s - calm, cfg::CURVE_SKY_GAIN, 1e-5);
}

TEST(shaper_stiffens_fast_softens_slow) {
  ActuatorShaper a;
  a.reset(45);
  a.setTarget(150);
  int out = 0;
  for (int i = 0; i < 10; ++i) out = a.update(0.01f, 0);  // 0.1 s
  CHECK_NEAR(out, 45 + cfg::SLEW_UP_DEG_S * 0.1, 1.5);
  for (int i = 0; i < 100; ++i) out = a.update(0.01f, 0);
  CHECK_EQ(out, 150);
  a.setTarget(45);
  for (int i = 0; i < 10; ++i) out = a.update(0.01f, 0);
  CHECK_NEAR(out, 150 - cfg::SLEW_DOWN_DEG_S * 0.1, 2.5);
}

TEST(shaper_deadband_ignores_tiny_changes) {
  ActuatorShaper a;
  a.reset(90);
  a.setTarget(91.2f);  // 1 degree: below the deadband until reached exactly
  int out = a.update(0.001f, 0);
  CHECK_EQ(out, 90);
  for (int i = 0; i < 100; ++i) out = a.update(0.01f, 0);
  CHECK_EQ(out, 91);  // settled on the target
}

TEST(shaper_impact_reflex_holds) {
  ActuatorShaper a;
  a.reset(45);
  a.setTarget(45);
  a.impact(1000);
  CHECK_EQ(a.update(0.01f, 1000), cfg::SERVO_IMPACT);
  CHECK_EQ(a.update(0.01f, 1000 + cfg::IMPACT_HOLD_MS - 1), cfg::SERVO_IMPACT);
  int out = 0;
  for (int i = 0; i < 10; ++i) out = a.update(0.01f, 1000 + cfg::IMPACT_HOLD_MS + 10 * i);
  CHECK(out < cfg::SERVO_IMPACT);  // relaxing after the hold
}

TEST(surface_tracker_debounces) {
  SurfaceTracker t;
  CHECK_EQ(t.update(0, false), 4);  // first sighting: not yet
  CHECK_EQ(t.update(0, false), 0);  // confirmed
  CHECK_EQ(t.update(1, false), 0);
  CHECK_EQ(t.update(0, false), 0);  // single gravel blip ignored
  CHECK_EQ(t.update(1, true), 0);
  CHECK_EQ(t.update(1, true), 4);   // uncertain twice -> unknown
}
