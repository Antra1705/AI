#include <cmath>

#include "config.h"
#include "core/fuzzy_speed.h"
#include "test.h"

static const float CLEAR = cfg::DIST_MAX_CM + 1.0f;

TEST(fuzzy_clear_smooth_road_is_full_speed) {
  CHECK_NEAR(fuzzySpeed(CLEAR, 0, 0.0f, SpeedMode::FUZZY_IT2).u, 1.0, 0.02);
  CHECK_NEAR(fuzzySpeed(CLEAR, 0, 0.0f, SpeedMode::FUZZY_T1).u, 1.0, 1e-5);
}

TEST(fuzzy_rough_road_slows_down) {
  CHECK(fuzzySpeed(CLEAR, 0, 1.0f, SpeedMode::FUZZY_IT2).u < 0.8f);
}

TEST(fuzzy_near_and_fast_means_stop) {
  CHECK(fuzzySpeed(8.0f, 40.0f, 0.0f, SpeedMode::FUZZY_IT2).u < 0.05f);
  CHECK(fuzzySpeed(8.0f, 40.0f, 0.0f, SpeedMode::FUZZY_T1).u < 0.05f);
}

TEST(fuzzy_speed_never_rises_as_obstacle_gets_closer) {
  for (int m = 1; m <= 2; ++m) {
    SpeedMode mode = (SpeedMode)m;
    float prev = 2.0f;
    for (float d = 150; d >= 2; d -= 0.5f) {
      float u = fuzzySpeed(d, 20.0f, 0.2f, mode).u;
      CHECK(u <= prev + 1e-4f);
      prev = u;
    }
  }
}

TEST(fuzzy_output_bounds_and_interval) {
  for (float d = 0; d < 200; d += 7)
    for (float v = -20; v < 80; v += 9)
      for (float r = 0; r <= 1.0f; r += 0.25f) {
        FuzzyResult f = fuzzySpeed(d, v, r, SpeedMode::FUZZY_IT2);
        CHECK(f.u >= 0.0f && f.u <= 1.0f);
        CHECK(f.y_l <= f.y_r + 1e-6f);
      }
}

TEST(fuzzy_it2_with_zero_blur_equals_type1) {
  for (float d = 5; d < 120; d += 11) {
    float a = fuzzySpeed(d, 25.0f, 0.4f, SpeedMode::FUZZY_IT2, 0.0f).u;
    float b = fuzzySpeed(d, 25.0f, 0.4f, SpeedMode::FUZZY_T1).u;
    CHECK_NEAR(a, b, 1e-5);
  }
}

// The footprint of uncertainty should make IT2 less sensitive to noisy distance
// readings than Type-1: smaller output swings for the same input noise.
TEST(fuzzy_it2_is_less_sensitive_to_distance_noise) {
  double sum_t1 = 0, sum_it2 = 0;
  for (float d = 20; d < 100; d += 0.5f) {
    sum_t1 += std::fabs(fuzzySpeed(d + 1.5f, 25, 0.2f, SpeedMode::FUZZY_T1).u - fuzzySpeed(d - 1.5f, 25, 0.2f, SpeedMode::FUZZY_T1).u);
    sum_it2 += std::fabs(fuzzySpeed(d + 1.5f, 25, 0.2f, SpeedMode::FUZZY_IT2).u - fuzzySpeed(d - 1.5f, 25, 0.2f, SpeedMode::FUZZY_IT2).u);
  }
  CHECK(sum_it2 <= sum_t1 * 1.05);
}

TEST(range_tracker_estimates_closing_speed) {
  RangeTracker t;
  for (int i = 0; i < 50; ++i) t.update(200.0f - 30.0f * i * 0.04f, 0.04f);  // approaching at 30 cm/s
  CHECK_NEAR(t.closingSpeed(), 30.0, 1.0);
  CHECK_NEAR(t.distance(), 200.0 - 30.0 * 49 * 0.04, 1.0);
  t.update(-1.0f, 0.04f);  // path clear
  CHECK(!t.valid());
  CHECK_NEAR(t.closingSpeed(), 0.0, 1e-6);
}
