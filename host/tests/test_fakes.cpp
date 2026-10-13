#include "fake_hal.h"
#include "test.h"

TEST(fake_clock_wraps_like_esp32) {
  FakeClock c;
  c.t_us = 0xFFFFFFFFull - 5;
  uint32_t before = c.micros();
  c.advanceUs(10);
  CHECK_EQ((uint32_t)(c.micros() - before), 10u);
}

TEST(fake_distance_reports_after_echo_time) {
  FakeClock c;
  FakeDistance d(c);
  d.source = [](uint64_t) { return DistanceReading{DistanceStatus::OK, 100.0f}; };
  DistanceReading r{DistanceStatus::NO_RESPONSE, 0};
  d.trigger();
  CHECK(!d.poll(r));
  c.advanceUs(500 + 5800);
  CHECK(d.poll(r));
  CHECK(r.status == DistanceStatus::OK);
  CHECK_NEAR(r.cm, 100.0, 1e-6);
  CHECK(!d.poll(r));  // only once per trigger
}
