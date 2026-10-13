#include "ml_features.h"

#include <math.h>

#include "../config.h"
#include "features.h"

namespace {

const int N = ImuBuffer::CAPACITY;
const float PI_F = 3.14159265f;
const float EPS = 1e-3f;

// Frequency bands as DFT bin ranges (inclusive). Bin k = k Hz for N = 100 at 100 Hz.
const int BAND_LO[5] = {1, 3, 6, 11, 21};
const int BAND_HI[5] = {2, 5, 10, 20, 49};

float cos_table[N];
float sin_table[N];
float hann[N];
bool tables_ready = false;

void initTables() {
  for (int i = 0; i < N; ++i) {
    float a = 2.0f * PI_F * (float)i / (float)N;
    cos_table[i] = cosf(a);
    sin_table[i] = sinf(a);
    hann[i] = 0.5f - 0.5f * cosf(a);
  }
  tables_ready = true;
}

float gyroMag(const ImuSample& s) { return sqrtf(s.gx * s.gx + s.gy * s.gy + s.gz * s.gz); }

}  // namespace

bool extractMlFeatures(const ImuBuffer& buf, float out[ML_FEATURE_COUNT]) {
  if (!buf.full()) return false;
  if (!tables_ready) initTables();

  float d[N];
  float mean = 0.0f, gmean = 0.0f;
  for (int i = 0; i < N; ++i) {
    d[i] = accelMagnitude(buf.at(i));
    mean += d[i];
    gmean += gyroMag(buf.at(i));
  }
  mean /= (float)N;
  gmean /= (float)N;

  float sum2 = 0.0f, sum4 = 0.0f, dmax = -1e9f, dmin = 1e9f, gvar = 0.0f, gmax = 0.0f;
  float jerk_max = 0.0f, jerk_sq = 0.0f;
  int swings = 0, side = 0;
  for (int i = 0; i < N; ++i) {
    d[i] -= mean;
    float d2 = d[i] * d[i];
    sum2 += d2;
    sum4 += d2 * d2;
    if (d[i] > dmax) dmax = d[i];
    if (d[i] < dmin) dmin = d[i];
    if (d[i] > cfg::OSC_BAND) {
      if (side == -1) ++swings;
      side = 1;
    } else if (d[i] < -cfg::OSC_BAND) {
      if (side == 1) ++swings;
      side = -1;
    }
    float gd = gyroMag(buf.at(i)) - gmean;
    gvar += gd * gd;
    if (gd > gmax) gmax = gd;
    if (i > 0) {
      float j = instantJerk(buf.at(i - 1), buf.at(i));
      jerk_sq += j * j;
      if (j > jerk_max) jerk_max = j;
    }
  }
  const float rms = sqrtf(sum2 / (float)N);
  const float peak = dmax > -dmin ? dmax : -dmin;

  out[0] = logf(rms + EPS);
  out[1] = logf(dmax - dmin + EPS);
  out[2] = logf(dmax + EPS);
  out[3] = logf(-dmin + EPS);
  out[4] = logf(jerk_max + 1.0f);
  out[5] = logf(sqrtf(jerk_sq / (float)(N - 1)) + 1.0f);
  out[6] = (float)swings;
  out[7] = peak / (rms + EPS);
  float var2 = (sum2 / (float)N) * (sum2 / (float)N);
  out[8] = logf((sum4 / (float)N) / (var2 + 1e-9f) + EPS);

  // Band energies from a direct DFT of the Hann-windowed signal (about 10k
  // multiply-adds at 10 Hz: small even for the ESP32).
  float w[N];
  for (int i = 0; i < N; ++i) w[i] = d[i] * hann[i];
  for (int b = 0; b < 5; ++b) {
    float energy = 0.0f;
    for (int k = BAND_LO[b]; k <= BAND_HI[b]; ++k) {
      float re = 0.0f, im = 0.0f;
      int idx = 0;
      for (int n = 0; n < N; ++n) {
        re += w[n] * cos_table[idx];
        im -= w[n] * sin_table[idx];
        idx += k;
        if (idx >= N) idx -= N;
      }
      energy += (re * re + im * im) / (float)N;
    }
    out[9 + b] = logf(energy + 1e-6f);
  }

  out[14] = logf(sqrtf(gvar / (float)N) + 1e-4f);
  out[15] = logf(gmax + 1e-4f);
  return true;
}

const char* mlFeatureName(int i) {
  static const char* names[ML_FEATURE_COUNT] = {
      "log_rms",      "log_p2p",     "log_rise",     "log_drop",    "log_jerk_max", "log_jerk_rms",
      "swings",       "crest",       "log_kurtosis", "band_1_2hz",  "band_3_5hz",   "band_6_10hz",
      "band_11_20hz", "band_21_49hz", "log_gyro_std", "log_gyro_rise"};
  return (i >= 0 && i < ML_FEATURE_COUNT) ? names[i] : "?";
}
