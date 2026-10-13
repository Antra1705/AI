#include "telemetry.h"

#include <stdio.h>
#include <stdlib.h>

#include "../config.h"

static const char* surfaceOrUnsure(int s) { return s >= 0 && s < 4 ? surfaceName((Surface)s) : "UNSURE"; }

static void formatDistance(char* buf, size_t n, const Snapshot& s) {
  if (!s.have_distance) {
    snprintf(buf, n, "---");
    return;
  }
  switch (s.dist.status) {
    case DistanceStatus::OK: snprintf(buf, n, "%.0fcm", (double)s.dist.cm); break;
    case DistanceStatus::TOO_CLOSE: snprintf(buf, n, "<2cm"); break;
    case DistanceStatus::OUT_OF_RANGE: snprintf(buf, n, "clear"); break;
    case DistanceStatus::NO_RESPONSE: snprintf(buf, n, "no-echo"); break;
  }
}

int formatStatus(char* buf, size_t n, const Snapshot& s) {
  char dist[16];
  formatDistance(dist, sizeof dist, s);
  char flags[96];
  int k = snprintf(flags, sizeof flags, "%s%s", s.imu_fault ? " IMU-FAULT" : "", s.servo_manual ? " MANUAL-SERVO" : "");
  if (s.state == RunState::FAULT && s.fault_reason[0] && k >= 0 && (size_t)k < sizeof flags) {
    snprintf(flags + k, sizeof flags - (size_t)k, " (%s)", s.fault_reason);
  }
  const double t = (double)s.now_ms / 1000.0;
  if (!s.ready) {
    return snprintf(buf, n, "[%8.2fs] %-8s road=(warming up) servo=%3d motor=%3d dist=%s%s\n", t,
                    runStateName(s.state), s.servo, s.motor_pwm, dist, flags);
  }
  return snprintf(buf, n,
                  "[%8.2fs] %-8s road=%-7s p=%.2f%s s=%.2f servo=%3d | motor=%3d u=%.2f dist=%s | "
                  "pitch=%.1f roll=%.1f vz=%.3f%s\n",
                  t, runStateName(s.state), surfaceOrUnsure(s.surface), (double)s.confidence,
                  s.uncertain ? " (unsure)" : "", (double)s.stiffness, s.servo, s.motor_pwm, (double)s.fuzzy_u, dist,
                  (double)s.pitch * 57.2958, (double)s.roll * 57.2958, (double)s.vz, flags);
}

int formatJson(char* buf, size_t n, const Snapshot& s) {
  const char* dstat = !s.have_distance ? "none" : distanceStatusName(s.dist.status);
  return snprintf(
      buf, n,
      "J,{\"t\":%.2f,\"state\":\"%s\",\"fault\":\"%s\",\"imu_fault\":%d,\"ready\":%d,"
      "\"classifier\":\"%s\",\"p\":[%.3f,%.3f,%.3f,%.3f],\"top\":\"%s\",\"surface\":\"%s\","
      "\"conf\":%.3f,\"mi\":%.4f,\"novelty\":%.1f,\"uncertain\":%d,"
      "\"jerk\":%.0f,\"p2p\":%.2f,\"rms\":%.3f,"
      "\"curve\":\"%s\",\"s\":%.3f,\"gate\":%.3f,\"c_eff\":%.3f,\"s_exp\":%.3f,\"s_risk\":%.3f,"
      "\"s_sky\":%.3f,\"c_low\":%.2f,\"c_high\":%.2f,\"s_safe\":%.2f,\"servo\":%d,\"servo_manual\":%d,"
      "\"speed_mode\":\"%s\",\"motor\":%d,\"u\":%.3f,\"dist_status\":\"%s\",\"dist\":%.1f,"
      "\"range\":%.1f,\"closing\":%.1f,"
      "\"roll\":%.2f,\"pitch\":%.2f,\"vz\":%.4f,\"a_vert\":%.3f,"
      "\"maint\":\"%s\",\"maint_zeta\":%.4f,\"maint_tests\":%d,"
      "\"estops\":%lu,\"faults\":%lu,\"impacts\":%lu}\n",
      (double)s.now_ms / 1000.0, runStateName(s.state), s.fault_reason, s.imu_fault ? 1 : 0, s.ready ? 1 : 0,
      s.classifier, (double)s.p[0], (double)s.p[1], (double)s.p[2], (double)s.p[3], surfaceOrUnsure(s.top),
      surfaceOrUnsure(s.surface), (double)s.confidence, (double)s.mutual_info, (double)s.novelty, s.uncertain ? 1 : 0,
      (double)s.f.jerk_max, (double)s.f.peak_to_peak, (double)s.f.accel_rms_dev, s.curve_mode, (double)s.stiffness,
      (double)s.gate, (double)s.c_eff, (double)s.s_exp, (double)s.s_risk, (double)s.s_sky, (double)cfg::CURVE_C_LOW,
      (double)cfg::CURVE_C_HIGH, (double)cfg::STIFF_SAFE, s.servo, s.servo_manual ? 1 : 0, s.speed_mode, s.motor_pwm,
      (double)s.fuzzy_u, dstat, (double)s.dist.cm, (double)s.range_cm, (double)s.closing_cms,
      (double)s.roll * 57.2958, (double)s.pitch * 57.2958, (double)s.vz, (double)s.a_vert, maintStatusName(s.maint),
      (double)s.maint_zeta, s.maint_tests, (unsigned long)s.estops, (unsigned long)s.faults,
      (unsigned long)s.impacts);
}

int formatHealth(char* buf, size_t n, const Health& h, uint32_t uptime_ms, const char* reset_reason) {
  return snprintf(buf, n,
                  "--- health ---\n"
                  "uptime            %lu.%03lu s\n"
                  "reset reason      %s\n"
                  "imu read errors   %lu\n"
                  "imu faults        %lu\n"
                  "imu overruns      %lu\n"
                  "dist no-response  %lu\n"
                  "faults            %lu\n"
                  "emergency stops   %lu\n"
                  "impacts           %lu\n"
                  "impact latency    last %lu us, max %lu us\n"
                  "slowest loop      %lu us\n",
                  (unsigned long)(uptime_ms / 1000), (unsigned long)(uptime_ms % 1000), reset_reason,
                  (unsigned long)h.imu_read_errors, (unsigned long)h.imu_faults,
                  (unsigned long)h.imu_overruns, (unsigned long)h.dist_no_response,
                  (unsigned long)h.faults, (unsigned long)h.estops, (unsigned long)h.impacts,
                  (unsigned long)h.last_impact_latency_us, (unsigned long)h.max_impact_latency_us,
                  (unsigned long)h.max_tick_us);
}

int formatMaint(char* buf, size_t n, const MaintState& m) {
  int k = snprintf(buf, n,
                   "--- maintenance ---\n"
                   "status            %s\n"
                   "tests             %u (baseline from the first %d)\n"
                   "baseline zeta     %.4f +/- %.4f\n"
                   "recent (EWMA)     %.4f  (%.0f%% below baseline)\n"
                   "CUSUM             %.2f  (watch > %.1f, service > %.1f)\n"
                   "history          ",
                   maintStatusName((MaintStatus)m.status), (unsigned)m.tests, cfg::MAINT_BASELINE_TESTS,
                   (double)m.base_mean, (double)m.base_std, (double)m.ewma,
                   m.base_mean > 0 ? 100.0 * (double)(m.base_mean - m.ewma) / (double)m.base_mean : 0.0,
                   (double)m.cusum, (double)cfg::MAINT_CUSUM_WATCH, (double)cfg::MAINT_CUSUM_SERVICE);
  for (int i = 0; i < m.hist_len && k > 0 && (size_t)k < n; ++i)
    k += snprintf(buf + k, n - (size_t)k, " %.3f", (double)m.history[i]);
  if (k > 0 && (size_t)k < n) k += snprintf(buf + k, n - (size_t)k, "\n");
  return k;
}

int formatRaw(char* buf, size_t n, const ImuSample& s) {
  return snprintf(buf, n, "R,%lu,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f\n", (unsigned long)s.t_us, (double)s.ax,
                  (double)s.ay, (double)s.az, (double)s.gx, (double)s.gy, (double)s.gz);
}

int formatFeatures(char* buf, size_t n, uint32_t t_ms, const Features& f, const Classification& c, int servo) {
  return snprintf(buf, n, "F,%lu,%.3f,%.1f,%.3f,%.3f,%d,%.4f,%s,%.2f,%d\n", (unsigned long)t_ms,
                  (double)f.accel_mean, (double)f.jerk_max, (double)f.peak_to_peak,
                  (double)f.accel_rms_dev, f.oscillations, (double)f.gyro_std, surfaceName(c.surface),
                  (double)c.confidence, servo);
}

bool parseRawLine(const char* line, ImuSample* out) {
  if (line[0] != 'R' || line[1] != ',') return false;
  const char* p = line + 2;
  char* end = 0;
  unsigned long t = strtoul(p, &end, 10);
  if (end == p || *end != ',') return false;
  float v[6];
  p = end + 1;
  for (int i = 0; i < 6; ++i) {
    v[i] = strtof(p, &end);
    if (end == p) return false;
    if (i < 5 && *end != ',') return false;
    p = end + 1;
  }
  out->t_us = (uint32_t)t;
  out->ax = v[0]; out->ay = v[1]; out->az = v[2];
  out->gx = v[3]; out->gy = v[4]; out->gz = v[5];
  return true;
}
