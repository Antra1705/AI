#include "config.h"
#include "core/obstacle_monitor.h"
#include "test.h"

static DistanceReading ok(float cm) { return DistanceReading{DistanceStatus::OK, cm}; }
static DistanceReading st(DistanceStatus s) { return DistanceReading{s, 0.0f}; }

TEST(obstacle_no_reading_means_no_driving) {
  ObstacleMonitor m;
  CHECK_EQ(m.speedLimit(), 0);
  m.update(ok(100));
  CHECK_EQ(m.speedLimit(), cfg::SPEED_NORMAL);
}

TEST(obstacle_speed_bands_and_boundaries) {
  ObstacleMonitor m;
  m.update(ok(20.0f));  // SRA: slow when < 20
  CHECK_EQ(m.speedLimit(), cfg::SPEED_NORMAL);
  m.update(ok(19.9f));
  CHECK_EQ(m.speedLimit(), cfg::SPEED_SLOW);
  m.update(ok(10.0f));  // SRA: stop when < 10
  CHECK(!m.emergency());
  CHECK_EQ(m.speedLimit(), cfg::SPEED_SLOW);
  m.update(ok(9.9f));
  CHECK(m.emergency());
  CHECK_EQ(m.speedLimit(), 0);
}

TEST(obstacle_estop_release_needs_hysteresis) {
  ObstacleMonitor m;
  m.update(ok(5));
  CHECK(m.emergency());
  m.update(ok(12));  // above 10 but below the release distance
  m.update(ok(12));
  m.update(ok(12));
  CHECK(m.emergency());
  for (int i = 0; i < cfg::ESTOP_RELEASE_READINGS - 1; ++i) m.update(ok(30));
  CHECK(m.emergency());
  m.update(ok(30));
  CHECK(!m.emergency());
}

TEST(obstacle_release_count_resets_on_close_reading) {
  ObstacleMonitor m;
  m.update(ok(5));
  for (int i = 0; i < cfg::ESTOP_RELEASE_READINGS - 1; ++i) m.update(ok(30));
  m.update(ok(12));  // came back close: start counting again
  for (int i = 0; i < cfg::ESTOP_RELEASE_READINGS - 1; ++i) m.update(ok(30));
  CHECK(m.emergency());
}

// Regression for SRA issue I-6: invalid readings must not mean "full speed".
TEST(obstacle_too_close_is_emergency) {
  ObstacleMonitor m;
  m.update(ok(100));
  m.update(st(DistanceStatus::TOO_CLOSE));
  CHECK(m.emergency());
  CHECK_EQ(m.speedLimit(), 0);
}

TEST(obstacle_no_response_slows_then_faults) {
  ObstacleMonitor m;
  m.update(ok(100));
  m.update(st(DistanceStatus::NO_RESPONSE));
  CHECK_EQ(m.speedLimit(), cfg::SPEED_SLOW);
  CHECK(!m.sensorFault());
  for (int i = 1; i < cfg::DIST_FAIL_LIMIT; ++i) m.update(st(DistanceStatus::NO_RESPONSE));
  CHECK(m.sensorFault());
  CHECK_EQ(m.speedLimit(), 0);
  m.update(ok(100));  // sensor answers again
  CHECK(!m.sensorFault());
  CHECK_EQ(m.speedLimit(), cfg::SPEED_NORMAL);
}

TEST(obstacle_out_of_range_is_clear_path) {
  ObstacleMonitor m;
  m.update(st(DistanceStatus::OUT_OF_RANGE));
  CHECK_EQ(m.speedLimit(), cfg::SPEED_NORMAL);
  CHECK(!m.sensorFault());
}
