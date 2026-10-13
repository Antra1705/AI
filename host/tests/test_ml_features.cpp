#include <cmath>
#include <string>

#include "core/ml_features.h"
#include "test.h"

static const float G = 9.81f;
static const double PI = 3.14159265358979323846;

template <typename F>
static void fillZ(ImuBuffer& b, F az) {
  for (int i = 0; i < ImuBuffer::CAPACITY; ++i) {
    ImuSample s = {(uint32_t)i * 10000u, 0, 0, az(i), 0, 0, 0};
    b.push(s);
  }
}

static int strongestBand(const float f[ML_FEATURE_COUNT]) {
  int best = 9;
  for (int i = 10; i <= 13; ++i)
    if (f[i] > f[best]) best = i;
  return best;
}

TEST(ml_features_need_full_buffer) {
  ImuBuffer b;
  float f[ML_FEATURE_COUNT];
  CHECK(!extractMlFeatures(b, f));
  fillZ(b, [](int) { return G; });
  CHECK(extractMlFeatures(b, f));
}

TEST(ml_features_fft_puts_tones_in_the_right_band) {
  const double tones[] = {1.5, 4.0, 8.0, 15.0, 30.0};
  for (int t = 0; t < 5; ++t) {
    ImuBuffer b;
    double hz = tones[t];
    fillZ(b, [&](int i) { return G + 1.0f * (float)std::sin(2 * PI * hz * i * 0.01); });
    float f[ML_FEATURE_COUNT];
    extractMlFeatures(b, f);
    CHECK_EQ(strongestBand(f), 9 + t);
  }
}

TEST(ml_features_flat_signal_is_quiet) {
  ImuBuffer b;
  fillZ(b, [](int) { return G; });
  float f[ML_FEATURE_COUNT];
  extractMlFeatures(b, f);
  CHECK(f[0] < std::log(0.01));   // tiny RMS
  CHECK_EQ(f[6], 0);              // no swings
  for (int i = 9; i <= 13; ++i) CHECK(f[i] < std::log(1e-4));
}

TEST(ml_features_drop_vs_rise_asymmetry) {
  ImuBuffer dip, hump;
  fillZ(dip, [](int i) { return i == 50 ? G - 5.0f : G; });
  fillZ(hump, [](int i) { return i == 50 ? G + 5.0f : G; });
  float fd[ML_FEATURE_COUNT], fh[ML_FEATURE_COUNT];
  extractMlFeatures(dip, fd);
  extractMlFeatures(hump, fh);
  CHECK(fd[3] > fd[2]);  // dip: drop bigger than rise
  CHECK(fh[2] > fh[3]);
  CHECK(fd[7] > 5.0f);   // single spike -> high crest factor
}

TEST(ml_features_orientation_independent) {
  // Same vibration with the sensor tilted: |a| is unchanged, so features match.
  ImuBuffer level, tilted;
  const float c = std::cos(0.3f), s = std::sin(0.3f);
  for (int i = 0; i < ImuBuffer::CAPACITY; ++i) {
    float mag = G + 0.5f * (float)std::sin(2 * PI * 7.0 * i * 0.01);
    ImuSample a = {(uint32_t)i * 10000u, 0, 0, mag, 0, 0, 0};
    ImuSample b2 = {(uint32_t)i * 10000u, mag * s, 0, mag * c, 0, 0, 0};
    level.push(a);
    tilted.push(b2);
  }
  float fl[ML_FEATURE_COUNT], ft[ML_FEATURE_COUNT];
  extractMlFeatures(level, fl);
  extractMlFeatures(tilted, ft);
  for (int i = 0; i < 14; ++i) CHECK_NEAR(fl[i], ft[i], 1e-3);
  for (int i = 16; i < 21; ++i) CHECK_NEAR(fl[i], ft[i], 1e-3);
}

// A fresh impact in the last 0.25 s stands out in the short-window features even
// when the 1 s features are dominated by earlier road.
TEST(ml_features_short_window_sees_fresh_impact) {
  ImuBuffer old_hit, new_hit;
  fillZ(old_hit, [](int i) { return i == 20 ? G + 6.0f : G; });  // 0.8 s ago
  fillZ(new_hit, [](int i) { return i == 95 ? G + 6.0f : G; });  // 50 ms ago
  float fo[ML_FEATURE_COUNT], fn[ML_FEATURE_COUNT];
  extractMlFeatures(old_hit, fo);
  extractMlFeatures(new_hit, fn);
  CHECK_NEAR(fo[1], fn[1], 1e-4);  // 1 s peak-to-peak: identical
  CHECK(fn[17] > fo[17] + 3.0f);   // short-window peak-to-peak: only the fresh hit
  CHECK(fn[20] > fo[20] + 3.0f);   // short-window jerk
}


// A periodic signal (corrugation) has a strong autocorrelation peak and a peaky
// spectrum; random vibration has neither.
TEST(ml_features_repetition_separates_periodic_from_random) {
  ImuBuffer periodic, random;
  unsigned seed = 3;
  fillZ(periodic, [](int i) { return G + 1.0f * (float)std::sin(2 * PI * 8.0 * i * 0.01); });
  fillZ(random, [&](int) {
    seed = seed * 1103515245u + 12345u;
    return G + ((float)((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f);
  });
  float fp[ML_FEATURE_COUNT], fr[ML_FEATURE_COUNT];
  extractMlFeatures(periodic, fp);
  extractMlFeatures(random, fr);
  CHECK(fp[21] > 0.8f);
  CHECK(fr[21] < 0.4f);
  CHECK(fp[22] < 0.2f);
  CHECK(fr[22] > fp[22] + 0.2f);
}
