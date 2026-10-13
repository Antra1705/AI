// maintenance.h - predictive maintenance by tracking drift in the damping.
//
// Reference test ('maint test'): drive slowly over a known reference bump with
// the servo locked at the softest setting. After the hit, the body bounces
// and the bounce dies away; how fast it dies away is the damping ratio zeta.
// Worn shocks damp less, so zeta drifts down over weeks of use.
//
// DampingAnalyzer   records ~1.5 s of gravity-free vertical acceleration (EKF)
//                   after the hit and estimates zeta from the free decay:
//                     peaks of alternating sign after the last big excursion,
//                     ln|peak| vs time -> decay rate sigma (least squares),
//                     peak spacing -> damped frequency w_d,
//                     zeta = sigma / sqrt(sigma^2 + w_d^2).
// MaintenanceMonitor  learns a baseline from the first MAINT_BASELINE_TESTS
//                   tests, then watches for a sustained drop with
//                     EWMA of zeta (smooths single noisy tests) and
//                     one-sided CUSUM on (baseline - zeta) / sigma (catches slow drift).
//                   Status: LEARNING -> HEALTHY -> WATCH -> SERVICE.
// State is saved through IStorage (flash on the ESP32) so it survives power-off.
#pragma once
#include <stdint.h>

#include "../config.h"

enum class MaintStatus : uint8_t { LEARNING, HEALTHY, WATCH, SERVICE };
const char* maintStatusName(MaintStatus s);

struct DampingEstimate {
  bool ok;
  float zeta;      // damping ratio
  float freq_hz;   // damped bounce frequency
  float peak;      // largest |vertical accel| in the record, m/s^2
  int peaks_used;
};

class DampingAnalyzer {
 public:
  enum { MAX_SAMPLES = 150 };  // 1.5 s at 100 Hz
  void start() { n_ = 0; }
  bool add(float a_vert);      // returns true when the record is full
  bool full() const { return n_ >= MAX_SAMPLES; }
  int count() const { return n_; }
  DampingEstimate analyze(float dt_s) const;
  // For tests: analyze an external record.
  static DampingEstimate analyzeRecord(const float* a, int n, float dt_s);

 private:
  float a_[MAX_SAMPLES];
  int n_ = 0;
};

struct MaintState {
  uint32_t magic;
  uint16_t version;
  uint16_t tests;        // tests recorded (baseline + monitoring)
  float base_sum, base_sum2;
  float base_mean, base_std;
  float ewma, cusum;
  float last_zeta;
  uint8_t status;
  uint8_t hist_len;
  uint8_t pad[2];
  float history[cfg::MAINT_HISTORY];  // most recent zetas, oldest first
};

class MaintenanceMonitor {
 public:
  MaintenanceMonitor() { reset(); }
  void reset();
  MaintStatus addTest(float zeta);
  MaintStatus status() const { return (MaintStatus)st_.status; }
  const MaintState& state() const { return st_; }
  bool load(const MaintState& s);  // false if the blob is not valid
  float dropFraction() const;      // (baseline - ewma) / baseline

 private:
  MaintState st_;
};
