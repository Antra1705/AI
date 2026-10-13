#include "config.h"
#include "core/features.h"
#include "test.h"

static const float G = 9.81f;

static ImuSample flat(uint32_t t_us, float az = G) {
  ImuSample s = {t_us, 0, 0, az, 0, 0, 0};
  return s;
}

// Fill a buffer with 100 samples 10 ms apart; az(i) supplied by the caller.
template <typename F>
static void fill(ImuBuffer& b, F az_of_index, uint32_t t0 = 0) {
  for (int i = 0; i < ImuBuffer::CAPACITY; ++i) b.push(flat(t0 + (uint32_t)i * 10000u, az_of_index(i)));
}

TEST(features_flat_signal_is_quiet) {
  ImuBuffer b;
  fill(b, [](int) { return G; });
  Features f = extractFeatures(b);
  CHECK_NEAR(f.accel_mean, G, 1e-4);
  CHECK_NEAR(f.jerk_max, 0, 1e-4);
  CHECK_NEAR(f.peak_to_peak, 0, 1e-4);
  CHECK_NEAR(f.accel_rms_dev, 0, 1e-4);
  CHECK_EQ(f.oscillations, 0);
}

TEST(features_single_spike) {
  ImuBuffer b;
  fill(b, [](int i) { return i == 50 ? G + 4.0f : G; });
  Features f = extractFeatures(b);
  CHECK_NEAR(f.peak_to_peak, 4.0, 1e-3);
  CHECK_NEAR(f.jerk_max, 4.0 / 0.01, 0.5);  // 4 m/s^2 in 10 ms = 400 m/s^3
  CHECK_NEAR(f.accel_mean, G + 0.04, 1e-3);
  CHECK_EQ(f.oscillations, 0);  // went above the band but never below it: not a swing
}

TEST(features_jerk_uses_real_timestamps) {
  ImuSample a = flat(0, G);
  ImuSample b = flat(20000, G + 1.0f);  // 1 m/s^2 over 20 ms
  CHECK_NEAR(instantJerk(a, b), 50.0, 0.01);
  CHECK_NEAR(instantJerk(b, b), 0.0, 1e-9);  // dt = 0 is ignored
  CHECK_NEAR(instantJerk(b, a), 0.0, 1e-9);  // going backwards is ignored
}

TEST(features_jerk_survives_micros_wrap) {
  ImuSample a = flat(0xFFFFFFFFu - 4999u, G);
  ImuSample b = flat(5000u, G + 1.0f);  // 10 ms later, across the wrap
  CHECK_NEAR(instantJerk(a, b), 100.0, 0.01);
}

// Regression for SRA issue I-7: a slow ramp that has wrapped the buffer must
// not produce a jerk spike between the newest and oldest sample.
TEST(features_no_false_jerk_at_buffer_wrap) {
  ImuBuffer b;
  for (int i = 0; i < 250; ++i) b.push(flat((uint32_t)i * 10000u, G + 0.001f * (float)i));
  Features f = extractFeatures(b);
  CHECK_NEAR(f.jerk_max, 0.1, 0.01);  // 0.001 m/s^2 per 10 ms everywhere
  CHECK_NEAR(f.peak_to_peak, 0.099, 1e-3);
}

TEST(features_oscillations_count_swings) {
  ImuBuffer b;
  // Square wave +/-1 m/s^2, period 100 ms: 10 periods = about 19-20 swings.
  fill(b, [](int i) { return (i / 5) % 2 ? G + 1.0f : G - 1.0f; });
  Features f = extractFeatures(b);
  CHECK(f.oscillations >= 18 && f.oscillations <= 20);
  CHECK_NEAR(f.accel_rms_dev, 1.0, 1e-3);
}

TEST(features_small_noise_below_band_does_not_oscillate) {
  ImuBuffer b;
  fill(b, [](int i) { return G + ((i % 2) ? 0.1f : -0.1f); });
  CHECK(0.1f < cfg::OSC_BAND);
  CHECK_EQ(extractFeatures(b).oscillations, 0);
}

TEST(features_gyro_std) {
  ImuBuffer b;
  for (int i = 0; i < ImuBuffer::CAPACITY; ++i) {
    ImuSample s = flat((uint32_t)i * 10000u);
    s.gy = (i % 2) ? 0.3f : 0.1f;  // |g| alternates 0.1 / 0.3 -> std 0.1
    b.push(s);
  }
  CHECK_NEAR(extractFeatures(b).gyro_std, 0.1, 1e-4);
}

TEST(features_handles_tiny_buffers) {
  ImuBuffer b;
  Features f = extractFeatures(b);
  CHECK_EQ(f.accel_mean, 0);
  b.push(flat(0));
  f = extractFeatures(b);
  CHECK_NEAR(f.accel_mean, G, 1e-4);
  CHECK_EQ(f.peak_to_peak, 0);
}
