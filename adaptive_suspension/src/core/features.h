// features.h - turns 1 second of raw IMU samples into numbers that describe the road.
#pragma once
#include "imu_buffer.h"
#include "types.h"

// |a| for one sample, m/s^2.
float accelMagnitude(const ImuSample& s);

// Jerk between two consecutive samples, m/s^3: |a(cur) - a(prev)| / dt using
// the real timestamps. Returns 0 if the timestamps are not increasing.
float instantJerk(const ImuSample& prev, const ImuSample& cur);

// Computes all features over the buffer. Needs at least 2 samples; with fewer
// it returns all zeros (accel_mean excepted when 1 sample is present).
Features extractFeatures(const ImuBuffer& buf);
