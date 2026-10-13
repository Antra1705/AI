#include "scenario.h"

#include <algorithm>
#include <cmath>

#include "core/classifier.h"
#include "core/features.h"
#include "core/imu_buffer.h"
#include "core/ml_features.h"

Track randomTrack(unsigned seed, const TrackOptions& opt) {
  Track track(seed);
  std::mt19937 rng(seed * 7919u + 13u);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  auto range = [&](double lo, double hi) { return lo + (hi - lo) * u(rng); };

  track.addSmooth(2.0);  // run-up while the buffer fills
  while (track.length() < opt.length_m) {
    double r = u(rng);
    if (r < 0.18) {
      track.addSmooth(range(1.0, 3.0));
    } else if (r < 0.40) {
      track.addGravel(range(2.0, 5.0), range(0.0010, 0.0030));
      track.addSmooth(range(0.8, 1.5));
    } else if (r < 0.64) {
      track.addBreaker(range(0.010, 0.030), range(0.06, 0.15));
      track.addSmooth(range(1.0, 1.6));
    } else if (r < 0.88) {
      track.addPothole(range(0.010, 0.030), range(0.05, 0.12));
      track.addSmooth(range(1.0, 1.6));
    } else if (opt.rumble) {
      track.addRumble(range(1.5, 3.0));
      track.addSmooth(range(0.8, 1.5));
    }
  }
  track.addSmooth(2.0);
  return track;
}

VehicleParams randomVehicle(std::mt19937& rng) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  VehicleParams p;
  p.mass *= 0.85 + 0.3 * u(rng);
  p.k_axle *= 0.9 + 0.2 * u(rng);
  p.wear = 0.6 + 0.4 * u(rng);
  p.vmax_at_255 = 0.5 + 0.2 * u(rng);
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
    if (e.t > t - 0.08 || e.t < t - 0.9) {
      ambiguous = true;
      continue;
    }
    int cls = e.kind == TrackKind::BREAKER ? SURFACE_BREAKER : SURFACE_POTHOLE;
    if (event >= 0 && event != cls) ambiguous = true;
    event = cls;
  }
  if (ambiguous) return LABEL_AMBIGUOUS;

  // Base surface and speed over the window.
  int n = 0, gravel = 0, rumble = 0;
  double speed = 0;
  for (auto it = hist_.rbegin(); it != hist_.rend() && it->t > w0; ++it) {
    if (it->t > t) continue;
    ++n;
    speed += it->v;
    for (double x : {it->xf, it->xr}) {
      int b = track_.baseAt(x);
      if (b == SURFACE_GRAVEL) ++gravel;
      if (b < 0) ++rumble;
    }
  }
  if (n == 0 || speed / n < 0.1) return LABEL_AMBIGUOUS;  // too slow: no vibration to read
  if (event >= 0) return (gravel + rumble) == 0 ? event : LABEL_AMBIGUOUS;
  double fg = gravel / (2.0 * n), fr = rumble / (2.0 * n);
  if (fr >= 0.9) return LABEL_OOD;
  if (fg >= 0.9) return SURFACE_GRAVEL;
  if (fg + fr <= 0.1) return SURFACE_SMOOTH;
  return LABEL_AMBIGUOUS;
}

// ---------------------------------------------------------------- dataset

long writeDataset(const DatasetOptions& opt, FILE* out) {
  std::fprintf(out, "run,t,label,speed,servo");
  for (int i = 0; i < ML_FEATURE_COUNT; ++i) std::fprintf(out, ",%s", mlFeatureName(i));
  std::fprintf(out, ",rule_pred\n");

  const ClassifierThresholds rules = defaultThresholds();
  long rows = 0;
  for (int run = 0; run < opt.runs; ++run) {
    unsigned seed = opt.seed * 100003u + (unsigned)run;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    TrackOptions to;
    to.length_m = opt.track_m;
    Track track = randomTrack(seed, to);
    Vehicle car(track, randomVehicle(rng), seed);
    WindowLabeler labeler(track);
    ImuBuffer buf;

    // Speed and servo change at random so the model learns to ignore both.
    double t = 0, next_speed = 0, next_servo = 0;
    int pwm = 0, servo = 90;
    const double dt = 0.001;
    long step = 0;
    while (car.rearX() < track.length() - 1.0 && t < 600.0) {
      if (t >= next_speed) {
        double v = 0.15 + 0.40 * u(rng);
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

      float f[ML_FEATURE_COUNT];
      extractMlFeatures(buf, f);
      Classification rc = classify(extractFeatures(buf), rules);
      std::fprintf(out, "%d,%.2f,%d,%.3f,%.0f", run, t, labeler.label(t), car.truth().v, car.truth().servo_actual);
      for (int i = 0; i < ML_FEATURE_COUNT; ++i) std::fprintf(out, ",%.5g", f[i]);
      std::fprintf(out, ",%d\n", rc.surface == SURFACE_UNKNOWN ? -1 : (int)rc.surface);
      ++rows;
    }
  }
  return rows;
}
