#include "maintenance.h"

#include <math.h>
#include <string.h>

static const uint32_t MAINT_MAGIC = 0x4D4E5431u;  // "MNT1"

const char* maintStatusName(MaintStatus s) {
  switch (s) {
    case MaintStatus::LEARNING: return "LEARNING";
    case MaintStatus::HEALTHY: return "HEALTHY";
    case MaintStatus::WATCH: return "WATCH";
    case MaintStatus::SERVICE: return "SERVICE";
  }
  return "?";
}

// ---------------------------------------------------------------- DampingAnalyzer

bool DampingAnalyzer::add(float a) {
  if (n_ < MAX_SAMPLES) a_[n_++] = a;
  return full();
}

DampingEstimate DampingAnalyzer::analyze(float dt_s) const { return analyzeRecord(a_, n_, dt_s); }

DampingEstimate DampingAnalyzer::analyzeRecord(const float* a, int n, float dt_s) {
  DampingEstimate e = {false, 0.0f, 0.0f, 0.0f, 0};
  if (n < 20) return e;
  float peak = 0.0f;
  for (int i = 0; i < n; ++i)
    if (fabsf(a[i]) > peak) peak = fabsf(a[i]);
  e.peak = peak;
  if (peak < cfg::MAINT_MIN_PEAK) return e;

  // Free decay starts after the last sample that is still part of the forced response.
  int start = 0;
  for (int i = 0; i < n; ++i)
    if (fabsf(a[i]) >= cfg::MAINT_FORCED_FRACTION * peak) start = i;

  // Alternating-sign extrema above the noise floor.
  float pk_t[32], pk_v[32];
  int np = 0, last_sign = 0;
  for (int i = start + 1; i < n - 1 && np < 32; ++i) {
    float v = a[i];
    bool is_max = v > a[i - 1] && v >= a[i + 1] && v > cfg::MAINT_NOISE_FLOOR;
    bool is_min = v < a[i - 1] && v <= a[i + 1] && v < -cfg::MAINT_NOISE_FLOOR;
    int sign = is_max ? 1 : (is_min ? -1 : 0);
    if (sign == 0) continue;
    if (sign == last_sign) {  // same side twice: keep the larger one
      if (fabsf(v) > fabsf(pk_v[np - 1])) {
        pk_v[np - 1] = v;
        pk_t[np - 1] = i * dt_s;
      }
      continue;
    }
    pk_t[np] = i * dt_s;
    pk_v[np] = v;
    ++np;
    last_sign = sign;
  }
  if (np < cfg::MAINT_MIN_PEAKS) return e;

  // Least squares: ln|peak| = c - sigma * t
  float st = 0, sy = 0, stt = 0, sty = 0;
  for (int k = 0; k < np; ++k) {
    float y = logf(fabsf(pk_v[k]));
    st += pk_t[k];
    sy += y;
    stt += pk_t[k] * pk_t[k];
    sty += pk_t[k] * y;
  }
  float den = np * stt - st * st;
  if (den <= 0.0f) return e;
  float sigma = -(np * sty - st * sy) / den;
  float half_period = (pk_t[np - 1] - pk_t[0]) / (float)(np - 1);
  if (sigma <= 0.0f || half_period <= 0.0f) return e;
  float wd = 3.14159265f / half_period;
  e.zeta = sigma / sqrtf(sigma * sigma + wd * wd);
  e.freq_hz = wd / (2.0f * 3.14159265f);
  e.peaks_used = np;
  e.ok = true;
  return e;
}

// ---------------------------------------------------------------- MaintenanceMonitor

void MaintenanceMonitor::reset() {
  memset(&st_, 0, sizeof st_);
  st_.magic = MAINT_MAGIC;
  st_.version = 1;
  st_.status = (uint8_t)MaintStatus::LEARNING;
}

bool MaintenanceMonitor::load(const MaintState& s) {
  if (s.magic != MAINT_MAGIC || s.version != 1 || s.hist_len > cfg::MAINT_HISTORY) return false;
  st_ = s;
  return true;
}

float MaintenanceMonitor::dropFraction() const {
  if (st_.base_mean <= 0.0f) return 0.0f;
  return (st_.base_mean - st_.ewma) / st_.base_mean;
}

MaintStatus MaintenanceMonitor::addTest(float zeta) {
  st_.last_zeta = zeta;
  if (st_.hist_len < cfg::MAINT_HISTORY) {
    st_.history[st_.hist_len++] = zeta;
  } else {
    memmove(st_.history, st_.history + 1, sizeof(float) * (cfg::MAINT_HISTORY - 1));
    st_.history[cfg::MAINT_HISTORY - 1] = zeta;
  }
  ++st_.tests;

  if (st_.tests <= cfg::MAINT_BASELINE_TESTS) {
    st_.base_sum += zeta;
    st_.base_sum2 += zeta * zeta;
    if (st_.tests == cfg::MAINT_BASELINE_TESTS) {
      float n = (float)cfg::MAINT_BASELINE_TESTS;
      st_.base_mean = st_.base_sum / n;
      float var = st_.base_sum2 / n - st_.base_mean * st_.base_mean;
      float sd = sqrtf(var > 0.0f ? var : 0.0f);
      float floor_sd = cfg::MAINT_MIN_REL_STD * st_.base_mean;  // tests too alike: don't over-react
      st_.base_std = sd > floor_sd ? sd : floor_sd;
      st_.ewma = st_.base_mean;
      st_.cusum = 0.0f;
      st_.status = (uint8_t)MaintStatus::HEALTHY;
    }
    return status();
  }

  st_.ewma = cfg::MAINT_EWMA_LAMBDA * zeta + (1.0f - cfg::MAINT_EWMA_LAMBDA) * st_.ewma;
  float z = (st_.base_mean - zeta) / st_.base_std;  // positive when damping dropped
  st_.cusum = st_.cusum + z - cfg::MAINT_CUSUM_K;
  if (st_.cusum < 0.0f) st_.cusum = 0.0f;

  float drop = dropFraction();
  MaintStatus s = MaintStatus::HEALTHY;
  if (st_.cusum > cfg::MAINT_CUSUM_SERVICE || drop >= cfg::MAINT_DROP_SERVICE) {
    s = MaintStatus::SERVICE;
  } else if (st_.cusum > cfg::MAINT_CUSUM_WATCH || drop >= cfg::MAINT_DROP_WATCH) {
    s = MaintStatus::WATCH;
  }
  st_.status = (uint8_t)s;
  return s;
}
