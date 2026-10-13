// telemetry.h - text formats for the serial output and the dashboard.
//
// CSV lines (for the calibration / training tools) start with a letter and a comma:
//   R,t_us,ax,ay,az,gx,gy,gz                                  raw IMU sample (100 Hz)
//   F,t_ms,mean,jerk,p2p,rms,osc,gyro,surface,conf,servo      features + decision (10 Hz)
//   J,{...}                                                   JSON snapshot for the dashboard (10 Hz)
// Every other line is for humans; the tools ignore it.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "health.h"
#include "maintenance.h"
#include "types.h"

#define RAW_CSV_HEADER "# R,t_us,ax,ay,az,gx,gy,gz"
#define FEATURE_CSV_HEADER \
  "# F,t_ms,accel_mean,jerk_max,peak_to_peak,accel_rms_dev,oscillations,gyro_std,surface,confidence,servo"

// Everything the status line and the dashboard show, gathered in one place.
struct Snapshot {
  uint32_t now_ms;
  RunState state;
  const char* fault_reason;
  bool imu_fault;
  // classification
  bool ready;                // 1 s of data and no IMU fault
  const char* classifier;    // "ml" or "rules"
  float p[4];
  int top;                   // most likely surface
  int surface;               // debounced surface (4 = uncertain)
  float confidence, mutual_info, novelty;
  bool uncertain;
  Features f;
  // damping
  const char* curve_mode;
  float stiffness, gate, c_eff;
  float s_exp, s_risk, s_sky;  // response-curve terms (see response_curve.h)
  int servo;
  bool servo_manual;
  // drive
  const char* speed_mode;
  int motor_pwm;
  float fuzzy_u;
  DistanceReading dist;
  bool have_distance;
  float range_cm, closing_cms;
  // EKF
  float roll, pitch, vz, a_vert;
  // maintenance
  MaintStatus maint;
  float maint_zeta;
  int maint_tests;
  // counters
  uint32_t estops, faults, impacts;
};

int formatStatus(char* buf, size_t n, const Snapshot& s);
int formatJson(char* buf, size_t n, const Snapshot& s);
int formatHealth(char* buf, size_t n, const Health& h, uint32_t uptime_ms, const char* reset_reason);
int formatMaint(char* buf, size_t n, const MaintState& m);
int formatRaw(char* buf, size_t n, const ImuSample& s);
int formatFeatures(char* buf, size_t n, uint32_t t_ms, const Features& f, const Classification& c, int servo);

// Parses an "R,..." line. Returns false for any other line.
bool parseRawLine(const char* line, ImuSample* out);
