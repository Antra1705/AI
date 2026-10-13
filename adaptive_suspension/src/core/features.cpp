#include "features.h"

#include <math.h>

float accelMagnitude(const ImuSample& s) {
  return sqrtf(s.ax * s.ax + s.ay * s.ay + s.az * s.az);
}

float instantJerk(const ImuSample& prev, const ImuSample& cur) {
  uint32_t dt_us = cur.t_us - prev.t_us;  // unsigned subtraction survives micros() wrap
  if (dt_us == 0 || dt_us > 0x7FFFFFFFu) return 0.0f;
  float dx = cur.ax - prev.ax;
  float dy = cur.ay - prev.ay;
  float dz = cur.az - prev.az;
  float dt_s = (float)dt_us * 1e-6f;
  return sqrtf(dx * dx + dy * dy + dz * dz) / dt_s;
}

static float gyroMagnitude(const ImuSample& s) {
  return sqrtf(s.gx * s.gx + s.gy * s.gy + s.gz * s.gz);
}

Features extractFeatures(const ImuBuffer& buf) {
  Features f = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0};
  const int n = buf.size();
  if (n == 0) return f;

  // Pass 1: magnitudes, mean, min/max, max jerk.
  float mag[ImuBuffer::CAPACITY];
  float sum = 0.0f, gyro_sum = 0.0f;
  float min_a = 1e9f, max_a = -1e9f;
  for (int i = 0; i < n; ++i) {
    const ImuSample& s = buf.at(i);
    mag[i] = accelMagnitude(s);
    sum += mag[i];
    gyro_sum += gyroMagnitude(s);
    if (mag[i] < min_a) min_a = mag[i];
    if (mag[i] > max_a) max_a = mag[i];
    if (i > 0) {
      float j = instantJerk(buf.at(i - 1), s);
      if (j > f.jerk_max) f.jerk_max = j;
    }
  }
  f.accel_mean = sum / (float)n;
  if (n < 2) return f;
  f.peak_to_peak = max_a - min_a;

  // Pass 2: spread around the mean, and swing count.
  const float gyro_mean = gyro_sum / (float)n;
  float var_sum = 0.0f, gyro_var_sum = 0.0f;
  int side = 0;  // -1 below the band, +1 above, 0 not yet outside
  for (int i = 0; i < n; ++i) {
    float dev = mag[i] - f.accel_mean;
    var_sum += dev * dev;
    float gdev = gyroMagnitude(buf.at(i)) - gyro_mean;
    gyro_var_sum += gdev * gdev;

    if (dev > cfg::OSC_BAND) {
      if (side == -1) ++f.oscillations;
      side = 1;
    } else if (dev < -cfg::OSC_BAND) {
      if (side == 1) ++f.oscillations;
      side = -1;
    }
  }
  f.accel_rms_dev = sqrtf(var_sum / (float)n);
  f.gyro_std = sqrtf(gyro_var_sum / (float)n);
  return f;
}
