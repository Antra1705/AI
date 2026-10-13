// scenario.h - random tracks, ground-truth window labels and dataset generation.
#pragma once
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "vehicle.h"

// Labels used in datasets and evaluation.
enum {
  LABEL_AMBIGUOUS = -1,  // window straddles a change, or the cart is (nearly) stopped
  LABEL_OOD = 4          // rumble strip: never used for training, only to test uncertainty
};

struct TrackOptions {
  double length_m = 50.0;
  bool rumble = true;  // include rumble strips (out-of-distribution surface)
};

// Random mix of smooth road, gravel, speed breakers, potholes (and rumble strips).
Track randomTrack(unsigned seed, const TrackOptions& opt);

// What a perfect classifier should say about the 1 s window ending at time t.
class WindowLabeler {
 public:
  explicit WindowLabeler(const Track& track) : track_(track) {}

  // Call at every IMU sample (10 ms) with the vehicle state.
  void record(double t, double front_x, double rear_x, double speed);

  // Surface index 0-3, LABEL_OOD, or LABEL_AMBIGUOUS.
  int label(double t) const;

 private:
  struct Entry {
    double t;
    TrackKind kind;
  };
  struct Rec {
    double t, xf, xr, v;
  };
  const Track& track_;
  std::vector<Rec> hist_;
  std::vector<Entry> entries_;
};

struct DatasetOptions {
  int runs = 60;
  unsigned seed = 1;
  double track_m = 50.0;
};

// Writes one CSV row per 100 ms window: run, time, label, speed, servo, the
// 16 ML features and the rule-based classifier's answer. Returns rows written.
long writeDataset(const DatasetOptions& opt, FILE* out);

// Randomised vehicle (mass, springs, wear, top speed) for one run.
VehicleParams randomVehicle(std::mt19937& rng);
