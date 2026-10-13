#include "scenario.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "core/classifier.h"
#include "core/features.h"
#include "core/imu_buffer.h"
#include "core/ml_features.h"

const char* domainName(Domain d) {
  switch (d) {
    case Domain::NOMINAL: return "nominal";
    case Domain::ROBUST: return "robust";
    case Domain::SHIFTED: return "shifted";
  }
  return "?";
}

bool parseDomain(const char* s, Domain* out) {
  for (Domain d : {Domain::NOMINAL, Domain::ROBUST, Domain::SHIFTED}) {
    if (std::string(s) == domainName(d)) {
      *out = d;
      return true;
    }
  }
  return false;
}

void speedRange(Domain domain, double* lo, double* hi) {
  switch (domain) {
    case Domain::NOMINAL: *lo = 0.15; *hi = 0.55; break;
    case Domain::ROBUST: *lo = 0.12; *hi = 0.65; break;
    case Domain::SHIFTED: *lo = 0.35; *hi = 0.72; break;
  }
}

Track randomTrack(unsigned seed, const TrackOptions& opt) {
  std::mt19937 rng(seed * 7919u + 13u);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  auto range = [&](double lo, double hi) { return lo + (hi - lo) * u(rng); };

  // Per-domain road: stone sizes (gravel band, cycles/m), gravel roughness, bump/pothole size.
  double g_lo = 20, g_hi = 80, rms_lo = 0.0010, rms_hi = 0.0030, ev_lo = 0.010, ev_hi = 0.030;
  if (opt.domain == Domain::ROBUST) {
    g_lo = range(12, 25);
    g_hi = range(50, 95);
    rms_lo = 0.0008, rms_hi = 0.0035, ev_lo = 0.008, ev_hi = 0.032;
  } else if (opt.domain == Domain::SHIFTED) {
    g_lo = 10, g_hi = 45;  // bigger stones
    rms_lo = 0.0020, rms_hi = 0.0038, ev_lo = 0.015, ev_hi = 0.036;
  }
  Track track(seed, g_lo, g_hi);

  track.addSmooth(2.0);  // run-up while the buffer fills
  while (track.length() < opt.length_m) {
    double r = u(rng);
    if (r < 0.18) {
      track.addSmooth(range(1.0, 3.0));
    } else if (r < 0.40) {
      track.addGravel(range(2.0, 5.0), range(rms_lo, rms_hi));
      track.addSmooth(range(0.8, 1.5));
    } else if (r < 0.64) {
      track.addBreaker(range(ev_lo, ev_hi), range(0.06, 0.15));
      track.addSmooth(range(1.0, 1.6));
    } else if (r < 0.88) {
      track.addPothole(range(ev_lo, ev_hi), range(0.05, 0.12));
      track.addSmooth(range(1.0, 1.6));
    } else if (opt.other && r < 0.94) {
      if (u(rng) < 0.5) track.addCobble(range(1.5, 3.0), range(0.04, 0.09), range(0.003, 0.008));
      else track.addWashboard(range(1.5, 3.0), range(0.06, 0.25), range(0.002, 0.006));
      track.addSmooth(range(0.8, 1.5));
    } else if (opt.rumble) {
      track.addRumble(range(1.5, 3.0));
      track.addSmooth(range(0.8, 1.5));
    }
  }
  track.addSmooth(2.0);
  return track;
}

VehicleParams randomVehicle(std::mt19937& rng, Domain domain) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  auto range = [&](double lo, double hi) { return lo + (hi - lo) * u(rng); };
  VehicleParams p;
  if (domain == Domain::NOMINAL) {
    p.mass *= range(0.85, 1.15);
    p.k_axle *= range(0.9, 1.1);
    p.wear = range(0.6, 1.0);
    p.vmax_at_255 = range(0.5, 0.7);
  } else if (domain == Domain::ROBUST) {
    p.mass *= range(0.7, 1.4);
    p.k_axle *= range(0.8, 1.25);
    double damper = range(0.8, 1.2);
    p.c_soft *= damper;
    p.c_stiff *= damper;
    p.wear = range(0.5, 1.0);
    p.vmax_at_255 = range(0.45, 0.85);
    p.accel_noise = range(0.02, 0.10);
    p.gyro_noise = range(0.002, 0.008);
    p.accel_bias = range(0.0, 0.15);
    p.gyro_bias = range(0.0, 0.04);
    double f = u(rng);  // MPU6050 digital low-pass settings that a build might use
    p.filter_hz = f < 0.5 ? 21.0 : (f < 0.8 ? 44.0 : 10.0);
    p.mount_tilt_deg = range(0.0, 8.0);
    p.vib_scale = range(0.5, 2.0);
  } else {  // SHIFTED: a different cart, partly outside the ROBUST ranges
    p.mass *= range(1.35, 1.5);
    p.k_axle *= range(0.8, 0.9);
    p.c_soft *= 1.1;
    p.c_stiff *= 1.1;
    p.wear = range(0.7, 1.0);
    p.vmax_at_255 = 0.9;
    p.accel_noise = 0.12;
    p.gyro_noise = 0.006;
    p.filter_hz = 44.0;
    p.mount_tilt_deg = 10.0;
    p.vib_scale = 1.8;
  }
  return p;
}

// ---------------------------------------------------------------- labels

void WindowLabeler::record(double t, double front_x, double rear_x, double speed) {
  if (!hist_.empty()) {
    const Rec& prev = hist_.back();
    for (const TrackFeature& f : track_.features()) {
      if (f.kind != TrackKind::BREAKER && f.kind != TrackKind::POTHOLE) continue;
      if (prev.xf < f.x0 && front_x >= f.x0) entries_.push_back(Entry{t, f.kind});
      if (prev.xr < f.x0 && rear_x >= f.x0) entries_.push_back(Entry{t, f.kind});
    }
  }
  hist_.push_back(Rec{t, front_x, rear_x, speed});
}

int WindowLabeler::label(double t) const {
  const double w0 = t - 1.0;
  // Events: a wheel hit a breaker/pothole well inside the window -> that class.
  // Hits right at the window edges are ambiguous (only part of the response is visible).
  int event = -1;
  bool ambiguous = false;
  for (const Entry& e : entries_) {
    if (e.t > t || e.t <= t - 1.3) continue;
    if (e.t > t - min_delay_ || e.t < t - 0.9) {
      ambiguous = true;
      continue;
    }
    int cls = e.kind == TrackKind::BREAKER ? SURFACE_BREAKER : SURFACE_POTHOLE;
    if (event >= 0 && event != cls) ambiguous = true;
    event = cls;
  }
  if (ambiguous) return LABEL_AMBIGUOUS;

  // Base surface and speed over the window.
  int n = 0, gravel = 0, rumble = 0, other = 0;
  double speed = 0;
  for (auto it = hist_.rbegin(); it != hist_.rend() && it->t > w0; ++it) {
    if (it->t > t) continue;
    ++n;
    speed += it->v;
    for (double x : {it->xf, it->xr}) {
      int b = track_.baseAt(x);
      if (b == SURFACE_GRAVEL) ++gravel;
      if (b == -1) ++rumble;
      if (b == -2) ++other;
    }
  }
  if (n == 0 || speed / n < 0.1) return LABEL_AMBIGUOUS;  // too slow: no vibration to read
  if (event >= 0) return (gravel + rumble + other) == 0 ? event : LABEL_AMBIGUOUS;
  double fg = gravel / (2.0 * n), fr = rumble / (2.0 * n), fo = other / (2.0 * n);
  if (fr >= 0.9) return LABEL_OOD;
  if (fo >= 0.9) return LABEL_OTHER;
  if (fo > 0.1) return LABEL_AMBIGUOUS;
  if (fg >= 0.9) return SURFACE_GRAVEL;
  if (fg + fr <= 0.1) return SURFACE_SMOOTH;
  return LABEL_AMBIGUOUS;
}

// ---------------------------------------------------------------- dataset

// Signal-level augmentation of one window: scale the vibration (deviation from
// the window's mean) and the rotation rates, and add sensor noise.
static void augmentWindow(const ImuBuffer& in, ImuBuffer& out, std::mt19937& rng) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::normal_distribution<double> g(0.0, 1.0);
  const double ka = 0.7 + 0.65 * u(rng), kg = 0.7 + 0.65 * u(rng);
  const double na = 0.06 * u(rng), ng = 0.004 * u(rng);
  double m[3] = {0, 0, 0};
  for (int i = 0; i < in.size(); ++i) {
    m[0] += in.at(i).ax;
    m[1] += in.at(i).ay;
    m[2] += in.at(i).az;
  }
  for (double& v : m) v /= in.size();
  out.clear();
  for (int i = 0; i < in.size(); ++i) {
    ImuSample s = in.at(i);
    s.ax = (float)(m[0] + ka * (s.ax - m[0]) + na * g(rng));
    s.ay = (float)(m[1] + ka * (s.ay - m[1]) + na * g(rng));
    s.az = (float)(m[2] + ka * (s.az - m[2]) + na * g(rng));
    s.gx = (float)(kg * s.gx + ng * g(rng));
    s.gy = (float)(kg * s.gy + ng * g(rng));
    s.gz = (float)(kg * s.gz + ng * g(rng));
    out.push(s);
  }
}

long writeDataset(const DatasetOptions& opt, FILE* out) {
  std::fprintf(out, "run,t,label,speed,servo,pwm,aug");
  for (int i = 0; i < ML_FEATURE_COUNT; ++i) std::fprintf(out, ",%s", mlFeatureName(i));
  std::fprintf(out, ",rule_pred\n");

  const ClassifierThresholds rules = defaultThresholds();
  double v_lo, v_hi;
  speedRange(opt.domain, &v_lo, &v_hi);
  long rows = 0;
  for (int run = 0; run < opt.runs; ++run) {
    unsigned seed = opt.seed * 100003u + (unsigned)run;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    TrackOptions to;
    to.length_m = opt.track_m;
    to.domain = opt.domain;
    to.other = opt.other;
    Track track = randomTrack(seed, to);
    Vehicle car(track, randomVehicle(rng, opt.domain), seed);
    WindowLabeler labeler(track);
    ImuBuffer buf, aug_buf;

    // Speed and servo change at random so the model learns to ignore the servo
    // and to use speed only as context.
    double t = 0, next_speed = 0, next_servo = 0;
    int pwm = 0, servo = 90;
    const double dt = 0.001;
    long step = 0;
    while (car.rearX() < track.length() - 1.0 && t < 600.0) {
      if (t >= next_speed) {
        double v = v_lo + (v_hi - v_lo) * u(rng);
        pwm = (int)std::min(255.0, v / car.params().vmax_at_255 * 255.0);
        next_speed = t + 3.0 + 5.0 * u(rng);
      }
      if (t >= next_servo) {
        servo = 45 + (int)(105 * u(rng));
        next_servo = t + 1.0 + 3.0 * u(rng);
      }
      car.step(dt, pwm, false, servo);
      t += dt;
      ++step;
      if (step % 10) continue;  // 100 Hz IMU

      ImuSample s;
      car.readImu(s);
      s.t_us = (uint32_t)(step * 1000);
      buf.push(s);
      labeler.record(t, car.frontX(), car.rearX(), car.truth().v);
      if (step % 100 || !buf.full()) continue;  // 10 Hz windows

      const int label = labeler.label(t);
      const int copies = (opt.augment && u(rng) < 0.5) ? 2 : 1;
      for (int c = 0; c < copies; ++c) {
        const ImuBuffer* src = &buf;
        if (c == 1) {
          augmentWindow(buf, aug_buf, rng);
          src = &aug_buf;
        }
        float f[ML_FEATURE_COUNT];
        extractMlFeatures(*src, f);
        Classification rc = classify(extractFeatures(*src), rules);
        std::fprintf(out, "%d,%.2f,%d,%.3f,%.0f,%d,%d", run, t, label, car.truth().v, car.truth().servo_actual, pwm, c);
        for (int i = 0; i < ML_FEATURE_COUNT; ++i) std::fprintf(out, ",%.5g", f[i]);
        std::fprintf(out, ",%d\n", rc.surface == SURFACE_UNKNOWN ? -1 : (int)rc.surface);
        ++rows;
      }
    }
  }
  return rows;
}
