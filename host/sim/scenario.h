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
  LABEL_OOD = 4,         // rumble strip: never used for training, only to test uncertainty
  LABEL_OTHER = 5        // cobbles / washboard: "unfamiliar road" examples for training
};

// Which population of carts and roads to draw from.
//   NOMINAL  the original simulator settings (one cart design, modest variation)
//   ROBUST   wide domain randomisation for training: cart weight, springs,
//            dampers, sensor noise/bias/filter, mounting, speeds, stone sizes,
//            bump sizes, drivetrain vibration
//   SHIFTED  "a different cart" for testing only: heavier, softer springs, a
//            noisier sensor with another filter setting, mounted crooked, faster,
//            bigger stones and taller bumps. Partly outside the ROBUST ranges.
enum class Domain { NOMINAL, ROBUST, SHIFTED };
const char* domainName(Domain d);
bool parseDomain(const char* s, Domain* out);

struct TrackOptions {
  double length_m = 50.0;
  bool rumble = true;  // include rumble strips (out-of-distribution surface)
  bool other = false;  // include cobbles / washboard ("other" surfaces for training)
  Domain domain = Domain::NOMINAL;
};

// Random mix of smooth road, gravel, speed breakers, potholes (and rumble strips).
Track randomTrack(unsigned seed, const TrackOptions& opt);

// What a perfect classifier should say about the 1 s window ending at time t.
// An event (breaker/pothole) counts once a wheel has been on it for min_delay_s.
class WindowLabeler {
 public:
  explicit WindowLabeler(const Track& track, double min_delay_s = 0.03) : track_(track), min_delay_(min_delay_s) {}

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
  double min_delay_;
  std::vector<Rec> hist_;
  std::vector<Entry> entries_;
};

struct DatasetOptions {
  int runs = 60;
  unsigned seed = 1;
  double track_m = 50.0;
  Domain domain = Domain::NOMINAL;
  bool augment = false;  // also write an augmented copy of about half the windows (aug = 1)
  bool other = false;    // include cobbles / washboard, labelled LABEL_OTHER
};

// Writes one CSV row per 100 ms window: run, time, label, speed, servo, motor
// PWM, augmented flag, the ML features and the rule-based classifier's answer.
// Returns rows written.
long writeDataset(const DatasetOptions& opt, FILE* out);

// Randomised vehicle for one run of the given domain.
VehicleParams randomVehicle(std::mt19937& rng, Domain domain = Domain::NOMINAL);

// Cruising-speed range (m/s) the dataset driver picks from in this domain.
void speedRange(Domain domain, double* lo, double* hi);
