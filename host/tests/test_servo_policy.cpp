#include "config.h"
#include "core/servo_policy.h"
#include "test.h"

TEST(servo_stiffens_immediately) {
  ServoPolicy p;
  p.reset(cfg::SERVO_SMOOTH);
  CHECK_EQ(p.update(cfg::SERVO_GRAVEL, 0), cfg::SERVO_GRAVEL);
  CHECK_EQ(p.update(cfg::SERVO_POTHOLE, 100), cfg::SERVO_POTHOLE);
}

TEST(servo_softens_only_after_confirmation) {
  ServoPolicy p;
  p.reset(cfg::SERVO_GRAVEL);
  uint32_t t = 0;
  for (int i = 0; i < cfg::SOFTEN_CONFIRM_CYCLES - 1; ++i) {
    CHECK_EQ(p.update(cfg::SERVO_SMOOTH, t += 100), cfg::SERVO_GRAVEL);
  }
  CHECK_EQ(p.update(cfg::SERVO_SMOOTH, t += 100), cfg::SERVO_SMOOTH);
}

TEST(servo_flip_flop_does_not_jitter) {
  ServoPolicy p;
  p.reset(cfg::SERVO_GRAVEL);
  uint32_t t = 0;
  // Alternating smooth / neutral requests: neither is confirmed, servo stays put.
  for (int i = 0; i < 20; ++i) {
    int target = (i % 2) ? cfg::SERVO_SMOOTH : cfg::SERVO_NEUTRAL;
    CHECK_EQ(p.update(target, t += 100), cfg::SERVO_GRAVEL);
  }
}

TEST(servo_impact_holds_then_relaxes) {
  ServoPolicy p;
  p.reset(cfg::SERVO_SMOOTH);
  p.impact(1000);
  CHECK_EQ(p.angle(), cfg::SERVO_IMPACT);
  // During the hold, a smooth request is ignored entirely.
  CHECK_EQ(p.update(cfg::SERVO_SMOOTH, 1000 + cfg::IMPACT_HOLD_MS - 1), cfg::SERVO_IMPACT);
  CHECK(p.holding());
  // After the hold the normal softening rule applies.
  uint32_t t = 1000 + cfg::IMPACT_HOLD_MS;
  int a = 0;
  for (int i = 0; i < cfg::SOFTEN_CONFIRM_CYCLES; ++i) a = p.update(cfg::SERVO_SMOOTH, t + (uint32_t)i * 100);
  CHECK_EQ(a, cfg::SERVO_SMOOTH);
}

TEST(servo_impact_hold_survives_millis_wrap) {
  ServoPolicy p;
  p.reset(cfg::SERVO_SMOOTH);
  uint32_t start = 0xFFFFFFFFu - 100;
  p.impact(start);
  CHECK_EQ(p.update(cfg::SERVO_SMOOTH, start + 500), cfg::SERVO_IMPACT);  // wrapped, still holding
  CHECK(p.holding());
}
