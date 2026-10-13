#include "core/imu_buffer.h"
#include "test.h"

static ImuSample sampleAt(uint32_t t) {
  ImuSample s = {t, 0, 0, 9.81f, 0, 0, 0};
  return s;
}

TEST(buffer_starts_empty) {
  ImuBuffer b;
  CHECK_EQ(b.size(), 0);
  CHECK(!b.full());
}

TEST(buffer_keeps_time_order_before_full) {
  ImuBuffer b;
  for (uint32_t i = 0; i < 10; ++i) b.push(sampleAt(i));
  CHECK_EQ(b.size(), 10);
  CHECK_EQ(b.at(0).t_us, 0u);
  CHECK_EQ(b.newest().t_us, 9u);
}

TEST(buffer_keeps_time_order_after_wrap) {
  ImuBuffer b;
  const int n = ImuBuffer::CAPACITY;
  for (int i = 0; i < n + 37; ++i) b.push(sampleAt((uint32_t)i));
  CHECK(b.full());
  CHECK_EQ(b.size(), n);
  CHECK_EQ(b.at(0).t_us, 37u);  // oldest surviving sample
  CHECK_EQ(b.newest().t_us, (uint32_t)(n + 36));
  for (int i = 1; i < n; ++i) CHECK_EQ(b.at(i).t_us - b.at(i - 1).t_us, 1u);
}

TEST(buffer_clear) {
  ImuBuffer b;
  for (int i = 0; i < 150; ++i) b.push(sampleAt((uint32_t)i));
  b.clear();
  CHECK_EQ(b.size(), 0);
  b.push(sampleAt(500));
  CHECK_EQ(b.at(0).t_us, 500u);
}
