// ml_features.h - feature vector for the neural-network classifier.
//
// Computed on the cart from the last 1 s of IMU samples (100 samples at 100 Hz)
// and, in the host tools, by exactly this same code, so training and on-device
// inference always see identical numbers.
//
// The signal analysed is d = |a| - mean(|a|): orientation-independent, gravity removed.
//   0 log RMS of d             sustained vibration
//   1 log peak-to-peak
//   2 log largest rise above the mean
//   3 log largest drop below the mean       (a pothole drops first)
//   4 log max jerk             m/s^3
//   5 log RMS jerk
//   6 swing count (crossings of +/- OSC_BAND)
//   7 crest factor  max|d| / RMS             impulsive vs steady
//   8 log kurtosis                           spikiness
//   9-13 log energy in frequency bands 1-2, 3-5, 6-10, 11-20, 21-49 Hz
//        (Hann-windowed DFT of d; bin k = k Hz at 100 samples / 1 s)
//  14 log std of |gyro|
//  15 log largest |gyro| rise above its mean
#pragma once
#include "imu_buffer.h"

enum { ML_FEATURE_COUNT = 16 };

// Fills out[ML_FEATURE_COUNT]. Needs a full buffer; returns false otherwise.
bool extractMlFeatures(const ImuBuffer& buf, float out[ML_FEATURE_COUNT]);

const char* mlFeatureName(int i);
