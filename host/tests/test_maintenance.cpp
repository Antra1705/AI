#include <cmath>
#include <random>

#include "config.h"
#include "core/maintenance.h"
#include "test.h"

// Free decay of a damped oscillator after a forced part.
static void decay(float zeta, float f_hz, float* a, int n, float noise, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> g(0, noise);
  const float wn = 2 * 3.14159265f * f_hz, wd = wn * std::sqrt(1 - zeta * zeta);
  for (int i = 0; i < n; ++i) {
    float t = i * 0.01f;
    a[i] = (t < 0.08f ? 8.0f * std::sin(3.14159265f * t / 0.08f)  // the hit itself
                      : 5.0f * std::exp(-zeta * wn * (t - 0.08f)) * std::cos(wd * (t - 0.08f))) +
           g(rng);
  }
}

TEST(damping_estimate_recovers_zeta) {
  float a[150];
  for (float zeta : {0.06f, 0.10f, 0.15f, 0.20f}) {
    decay(zeta, 4.0f, a, 150, 0.03f, 1);
    DampingEstimate e = DampingAnalyzer::analyzeRecord(a, 150, 0.01f);
    CHECK(e.ok);
    CHECK_NEAR(e.zeta, zeta, 0.02);
    CHECK_NEAR(e.freq_hz, 4.0 * std::sqrt(1 - zeta * zeta), 0.3);
  }
}

TEST(damping_estimate_rejects_weak_or_flat_records) {
  float a[150] = {0};
  CHECK(!DampingAnalyzer::analyzeRecord(a, 150, 0.01f).ok);
  for (int i = 0; i < 150; ++i) a[i] = 0.3f * std::sin(i * 0.2f);
  CHECK(!DampingAnalyzer::analyzeRecord(a, 150, 0.01f).ok);  // peak below MAINT_MIN_PEAK
}

TEST(analyzer_collects_fixed_length) {
  DampingAnalyzer d;
  d.start();
  for (int i = 0; i < DampingAnalyzer::MAX_SAMPLES - 1; ++i) CHECK(!d.add(0.0f));
  CHECK(d.add(0.0f));
  CHECK(d.full());
}

TEST(monitor_learns_then_stays_healthy) {
  MaintenanceMonitor m;
  for (int i = 0; i < cfg::MAINT_BASELINE_TESTS - 1; ++i) CHECK(m.addTest(0.12f + 0.002f * (i % 2)) == MaintStatus::LEARNING);
  CHECK(m.addTest(0.12f) == MaintStatus::HEALTHY);
  std::mt19937 rng(3);
  std::normal_distribution<float> g(0.12f, 0.004f);
  for (int i = 0; i < 40; ++i) CHECK(m.addTest(g(rng)) == MaintStatus::HEALTHY);  // no false alarms
}

TEST(monitor_flags_gradual_wear) {
  MaintenanceMonitor m;
  for (int i = 0; i < cfg::MAINT_BASELINE_TESTS; ++i) m.addTest(0.12f);
  int first_watch = -1, first_service = -1;
  for (int day = 0; day < 40; ++day) {
    float zeta = 0.12f * (1.0f - 0.015f * day);  // loses 1.5 % per test
    MaintStatus s = m.addTest(zeta);
    if (s == MaintStatus::WATCH && first_watch < 0) first_watch = day;
    if (s == MaintStatus::SERVICE && first_service < 0) first_service = day;
  }
  CHECK(first_watch > 0);
  CHECK(first_service > first_watch);
  CHECK(first_service < 30);
}

TEST(monitor_state_round_trips) {
  MaintenanceMonitor a, b;
  for (int i = 0; i < 8; ++i) a.addTest(0.1f + 0.001f * i);
  CHECK(b.load(a.state()));
  CHECK(b.status() == a.status());
  CHECK_NEAR(b.state().base_mean, a.state().base_mean, 1e-9);
  MaintState junk = a.state();
  junk.magic = 0;
  CHECK(!b.load(junk));
}
