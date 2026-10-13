#include "eval.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "config.h"
#include "core/ekf.h"
#include "scenario.h"
#include "sim_cart.h"

namespace {

const unsigned EVAL_SEED = 900000;  // never used by 'sim dataset', so tests are on unseen tracks

struct Config {
  const char* name;
  bool ml, graded;
  int fixed_servo;  // >= 0: servo locked (passive suspension)
};

void configure(SimCart& c, const Config& cfgn, SpeedMode speed) {
  c.app.setClassifierMl(cfgn.ml);
  c.app.setCurveGraded(cfgn.graded);
  c.app.setSpeedMode(speed);
}

struct EventEntry {
  TrackKind kind;
  double x0;
};

std::vector<EventEntry> eventsOf(const Track& t) {
  std::vector<EventEntry> v;
  for (const TrackFeature& f : t.features())
    if (f.kind == TrackKind::BREAKER || f.kind == TrackKind::POTHOLE) v.push_back(EventEntry{f.kind, f.x0});
  return v;
}

double mean(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return v.empty() ? 0 : s / v.size();
}

}  // namespace

// ---------------------------------------------------------------- road

int evalRoad(int runs, Domain domain, int classify_ms) {
  const Config configs[] = {
      {"Passive soft (servo fixed 45)", false, false, 45},
      {"Passive stiff (servo fixed 150)", false, false, 150},
      {"SRA: rules + fixed angles", false, false, -1},
      {"Neural + fixed angles", true, false, -1},
      {"Neural + graded curve (deployed)", true, true, -1},
  };
  const int NC = sizeof configs / sizeof configs[0];

  struct Metrics {
    double smooth_sq = 0, gravel_sq = 0;
    long smooth_n = 0, gravel_n = 0;
    std::vector<double> event_peak, pothole_reaction, detect;
    int detect_miss = 0;
    int bottom_outs = 0, events = 0, missed_stiff = 0;
    int bottom_at_events = 0;  // within 1.5 s of a wheel reaching a breaker/pothole; the rest is on gravel
    double servo_travel = 0, distance = 0;
    int confusion[4][4] = {{0}};
    long uncertain = 0, scored = 0;
  } m[NC];

  for (int run = 0; run < runs; ++run) {
    unsigned seed = EVAL_SEED + (unsigned)run;
    TrackOptions to;
    to.length_m = 40;
    to.rumble = false;
    to.domain = domain;
    Track track = randomTrack(seed, to);
    std::mt19937 rng(seed);
    VehicleParams vp = randomVehicle(rng, domain);
    std::vector<EventEntry> events = eventsOf(track);

    for (int k = 0; k < NC; ++k) {
      SimCart cart(track, vp, seed);
      configure(cart, configs[k], SpeedMode::STEP);  // equal speed for all: compare damping only
      if (classify_ms > 0) cart.app.setClassifyPeriod((uint32_t)classify_ms);
      cart.begin();
      if (configs[k].fixed_servo >= 0) cart.command("servo " + std::to_string(configs[k].fixed_servo));
      cart.command("start");
      Metrics& M = m[k];
      WindowLabeler lab(cart.track);
      size_t next_event = 0;
      double last_event_t = -10, active_entry = -1, active_peak = 0;
      bool active_pothole = false, reacted = false, detected = false;
      int last_servo = cart.rig.servo.angle;
      int last_bottoms = 0;
      long step = 0;

      cart.on_ms = [&] {
        ++step;
        const VehicleTruth& tr = cart.car.truth();
        double t = cart.time();
        int servo = cart.rig.servo.angle;
        M.servo_travel += std::abs(servo - last_servo);
        last_servo = servo;
        if (next_event < events.size() && cart.car.frontX() >= events[next_event].x0) {
          if (active_entry >= 0) M.event_peak.push_back(active_peak);
          active_entry = t;
          active_peak = 0;
          active_pothole = events[next_event].kind == TrackKind::POTHOLE;
          reacted = false;
          detected = false;
          ++M.events;
          ++next_event;
          last_event_t = t;
        }
        if (active_entry >= 0) {
          double since = t - active_entry;
          if (since < 1.0) active_peak = std::max(active_peak, std::fabs(tr.heave_acc));
          // Reaction is measured on the servo's actual (slewing) position, not the command.
          if (active_pothole && !reacted && tr.servo_actual >= 140 && since < 0.5) {
            M.pothole_reaction.push_back(since * 1000.0);
            reacted = true;
          }
          // Detection: the classifier first puts > 50% on breaker + pothole.
          if (!detected && configs[k].fixed_servo < 0) {
            const Decision& d = cart.app.decision();
            if (d.valid && d.p[SURFACE_BREAKER] + d.p[SURFACE_POTHOLE] > 0.5f && since < 0.5) {
              M.detect.push_back(since * 1000.0);
              detected = true;
            } else if (since >= 0.5) {
              ++M.detect_miss;
              detected = true;
            }
          }
          if (since >= 0.5 && active_pothole && !reacted) {
            ++M.missed_stiff;
            reacted = true;
          }
          if (since >= 1.0) {
            M.event_peak.push_back(active_peak);
            active_entry = -1;
          }
        }
        if (cart.car.bottomOuts() != last_bottoms) {
          if (t - last_event_t <= 1.5) M.bottom_at_events += cart.car.bottomOuts() - last_bottoms;
          last_bottoms = cart.car.bottomOuts();
        }
        if (step % 10 == 0) lab.record(t, cart.car.frontX(), cart.car.rearX(), tr.v);
        if (t - last_event_t > 1.5 && tr.v > 0.1) {
          int bf = track.baseAt(cart.car.frontX()), br = track.baseAt(cart.car.rearX());
          if (bf == SURFACE_GRAVEL && br == SURFACE_GRAVEL) {
            M.gravel_sq += tr.heave_acc * tr.heave_acc;
            ++M.gravel_n;
          } else if (bf == SURFACE_SMOOTH && br == SURFACE_SMOOTH) {
            M.smooth_sq += tr.heave_acc * tr.heave_acc;
            ++M.smooth_n;
          }
        }
        if (step % 100 == 0 && cart.app.featuresValid() && configs[k].fixed_servo < 0) {
          int label = lab.label(t);
          if (label >= 0 && label < 4) {
            M.confusion[label][cart.app.decision().top]++;
            M.uncertain += cart.app.decision().uncertain;
            ++M.scored;
          }
        }
      };
      cart.runUntil([&] { return cart.car.rearX() > track.length() - 1.0; }, 400);
      M.bottom_outs += cart.car.bottomOuts();
      M.distance += cart.car.truth().x;
    }
  }

  std::printf("Closed-loop ride comparison (%s carts and roads): %d random tracks x 40 m (unseen in\n"
              "training), same cars, same speed (PWM 200) for every strategy so only the damping differs.\n"
              "Classifier runs every %d ms. Lower is better for every column except accuracy.\n\n",
              domainName(domain), runs, classify_ms > 0 ? classify_ms : (int)cfg::CLASSIFY_PERIOD_MS);
  std::printf("%-34s %8s %8s %9s %8s %9s %10s %8s\n", "Strategy", "smooth", "gravel", "event", "bottom", "servo",
              "pothole", "class.");
  std::printf("%-34s %8s %8s %9s %8s %9s %10s %8s\n", "", "RMS m/s2", "RMS m/s2", "peak m/s2", "outs/evt", "deg/m",
              "react ms", "acc.");
  for (int k = 0; k < NC; ++k) {
    const Metrics& M = m[k];
    int ok = 0, tot = 0;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        tot += M.confusion[i][j];
        if (i == j) ok += M.confusion[i][j];
      }
    char react[24], acc[16];
    if (configs[k].fixed_servo >= 0) {
      std::snprintf(react, sizeof react, "-");
      std::snprintf(acc, sizeof acc, "-");
    } else {
      std::snprintf(react, sizeof react, "%.0f (%d miss)", mean(M.pothole_reaction), M.missed_stiff);
      std::snprintf(acc, sizeof acc, "%.1f%%", tot ? 100.0 * ok / tot : 0.0);
    }
    std::printf("%-34s %8.3f %8.3f %9.2f %8.2f %9.1f %10s %8s\n", configs[k].name,
                std::sqrt(M.smooth_sq / std::max(1L, M.smooth_n)), std::sqrt(M.gravel_sq / std::max(1L, M.gravel_n)),
                mean(M.event_peak), (double)M.bottom_outs / std::max(1, M.events), M.servo_travel / std::max(1.0, M.distance),
                react, acc);
  }
  std::printf("\nsmooth/gravel RMS: body vertical acceleration away from events (comfort).\n"
              "event peak: largest body acceleration within 1 s of a breaker/pothole (protection).\n"
              "bottom outs/evt: suspension hitting its travel limit per breaker/pothole.\n"
              "servo deg/m: total servo movement per metre driven (actuator wear).\n"
              "pothole react: front wheel reaches the pothole -> servo physically at >= 140 deg\n"
              "               (miss = not within 0.5 s).\n");
  std::printf("\nWhere the bottom-outs happen (per breaker/pothole): at the event itself (within 1.5 s) /\n"
              "elsewhere (on gravel):\n");
  for (int k = 0; k < NC; ++k) {
    const Metrics& M = m[k];
    std::printf("  %-34s %5.2f at events, %5.2f elsewhere\n", configs[k].name,
                (double)M.bottom_at_events / std::max(1, M.events),
                (double)(M.bottom_outs - M.bottom_at_events) / std::max(1, M.events));
  }
  std::printf("\nImpact detection (front wheel reaches a breaker or pothole -> classifier puts > 50%% on\n"
              "breaker + pothole; miss = not within 0.5 s):\n");
  for (int k = 0; k < NC; ++k) {
    if (configs[k].fixed_servo >= 0) continue;
    std::vector<double> d = m[k].detect;
    std::sort(d.begin(), d.end());
    std::printf("  %-34s avg %4.0f ms, median %4.0f ms, 90th pct %4.0f ms, %d missed of %zu\n", configs[k].name, mean(d),
                d.empty() ? 0.0 : d[d.size() / 2], d.empty() ? 0.0 : d[(size_t)(0.9 * (d.size() - 1))],
                m[k].detect_miss, d.size() + (size_t)m[k].detect_miss);
  }
  const Metrics& deployed = m[NC - 1];
  int ok = 0, tot = 0;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      tot += deployed.confusion[i][j];
      if (i == j) ok += deployed.confusion[i][j];
    }
  std::printf("\nDeployed classifier in closed loop: %.1f%% correct, %.1f%% of windows flagged uncertain.\n",
              tot ? 100.0 * ok / tot : 0.0, deployed.scored ? 100.0 * deployed.uncertain / deployed.scored : 0.0);
  return 0;
}

// ---------------------------------------------------------------- obstacle

int evalObstacle() {
  struct Mode {
    const char* name;
    SpeedMode mode;
  } modes[] = {{"SRA steps (200/127/0)", SpeedMode::STEP},
               {"Type-1 fuzzy", SpeedMode::FUZZY_T1},
               {"Interval Type-2 fuzzy", SpeedMode::FUZZY_IT2}};
  const double cruises[] = {0.45, 0.7, 0.9};  // m/s at PWM 200
  const double brake = 3.0;

  std::printf("Obstacle approach: wall 4 m ahead, 20 runs per cell, noisy sonar (1 cm + 3%% dropped echoes),\n"
              "braking %.1f m/s^2. 'gap' is the worst closest approach; 'hard stops' counts emergency\n"
              "brakings (< 10 cm); 'decel' is the peak deceleration; 'speed flips' counts how often the\n"
              "speed command changed direction (jerky driving).\n\n", brake);
  std::printf("%-24s %-10s %9s %11s %10s %12s %10s\n", "Mode", "cruise", "gap cm", "hard stops", "decel", "speed flips",
              "time s");
  for (double cruise : cruises) {
    for (const Mode& md : modes) {
      double worst_gap = 1e9, peak_decel = 0, flips = 0, t_stop = 0;
      int estops = 0, contacts = 0;
      for (unsigned seed = 0; seed < 20; ++seed) {
        Track t(seed + 50);
        t.addSmooth(20);
        VehicleParams vp;
        vp.vmax_at_255 = cruise * 255.0 / cfg::SPEED_NORMAL;
        vp.brake_decel = brake;
        SimCart cart(t, vp, seed + 50);
        cart.wall_x = 4.0;
        cart.sonar_noise_cm = 1.0;
        std::mt19937 drop(seed);
        auto base = cart.rig.distance.source;
        cart.rig.distance.source = [&, base](uint64_t now) {
          if (std::uniform_real_distribution<double>(0, 1)(drop) < 0.03)
            return DistanceReading{DistanceStatus::NO_RESPONSE, 0.0f};
          return base(now);
        };
        configure(cart, Config{"", true, true, -1}, md.mode);
        cart.begin();
        cart.command("start");
        double prev_v = 0, prev_dir = 0, stopped_at = -1;
        int last_pwm = 0;
        cart.on_ms = [&] {
          double v = cart.car.truth().v;
          peak_decel = std::max(peak_decel, (prev_v - v) / 0.001);
          prev_v = v;
          int pwm = cart.rig.motors.pwm;
          if (pwm != last_pwm) {
            double dir = pwm > last_pwm ? 1 : -1;
            if (prev_dir != 0 && dir != prev_dir) flips += 1;
            prev_dir = dir;
            last_pwm = pwm;
          }
          worst_gap = std::min(worst_gap, cart.gapCm());
          if (stopped_at < 0 && cart.time() > 2 && v < 0.005 && cart.gapCm() < 60) stopped_at = cart.time();
        };
        cart.runFor(20.0);
        estops += (int)cart.app.health().estops;
        contacts += cart.gapCm() <= 0;
        t_stop += stopped_at > 0 ? stopped_at : 20.0;
      }
      std::printf("%-24s %-10s %9.1f %11d %8.1f/s2 %12.1f %10.1f%s\n", md.name,
                  (std::to_string(cruise).substr(0, 4) + " m/s").c_str(), worst_gap, estops, peak_decel, flips / 20.0,
                  t_stop / 20.0, contacts ? "  CONTACT" : "");
    }
    std::printf("\n");
  }
  std::printf("The hard 10 cm emergency stop stays active in every mode; fuzzy control only shapes the\n"
              "approach. Real braking and top speed must be measured on the cart (see BRD R-3).\n");
  return 0;
}

// ---------------------------------------------------------------- faults

int evalFaults() {
  bool all_ok = true;
  auto line = [&](const char* name, bool pass, const std::string& detail) {
    std::printf("%-46s %-5s %s\n", name, pass ? "PASS" : "FAIL", detail.c_str());
    all_ok = all_ok && pass;
  };
  auto drive = [](SimCart& c, double s) {
    c.begin();
    c.command("start");
    c.runFor(s);
  };
  auto brakeDelay = [](SimCart& c, double t0) {
    for (auto& h : c.rig.motors.history)
      if (h.first >= (uint64_t)(t0 * 1e6) && h.second == 0) return (h.first * 1e-6 - t0) * 1000.0;
    return -1.0;
  };
  Track t(7);
  t.addSmooth(60);
  VehicleParams vp;

  {
    SimCart c(t, vp, 1);
    c.imu_unplugged = true;
    drive(c, 1.0);
    line("IMU missing at power-on: start refused", c.app.state() == RunState::IDLE && c.rig.motors.braked,
         c.rig.console.contains("Cannot start") ? "('Cannot start: IMU not responding')" : "");
  }
  {
    SimCart c(t, vp, 2);
    drive(c, 3.0);
    c.imu_unplugged = true;
    double t0 = c.time();
    c.runFor(1.0);
    double ms = brakeDelay(c, t0);
    line("IMU unplugged while driving: stops", c.app.state() == RunState::FAULT && ms >= 0 && ms <= 200,
         "stopped after " + std::to_string((int)ms) + " ms (BRD T6: <= 200)");
    line("  ... suspension returns to neutral", c.rig.servo.angle == cfg::SERVO_NEUTRAL, "");
  }
  {
    SimCart c(t, vp, 3);
    drive(c, 3.0);
    c.ultrasonic_unplugged = true;
    double t0 = c.time();
    c.runFor(1.0);
    double ms = brakeDelay(c, t0);
    line("Ultrasonic unplugged while driving: stops", c.app.state() == RunState::FAULT && ms >= 0 && ms <= 200,
         "stopped after " + std::to_string((int)ms) + " ms (BRD T6: <= 200)");
  }
  {
    SimCart c(t, vp, 4);
    drive(c, 6.0);
    line("Open space (nothing within 4 m): keeps driving",
         c.app.state() == RunState::RUNNING && c.rig.motors.pwm > 150, "");
  }
  {
    SimCart c(t, vp, 5);
    drive(c, 2.0);
    c.wall_x = c.car.truth().x + 0.10 + 0.01;  // something appears 1 cm in front
    c.runFor(0.2);
    line("Object suddenly < 2 cm away: emergency stop", c.app.state() == RunState::ESTOP && c.rig.motors.braked,
         "(SRA version would have driven at full speed)");
  }
  {
    SimCart c(t, vp, 6);
    std::mt19937 drop(6);
    auto base = c.rig.distance.source;
    c.rig.distance.source = [&, base](uint64_t now) {
      if (std::uniform_real_distribution<double>(0, 1)(drop) < 0.1) return DistanceReading{DistanceStatus::NO_RESPONSE, 0.0f};
      return base(now);
    };
    drive(c, 8.0);
    line("10% dropped echoes: no false fault", c.app.state() == RunState::RUNNING, "");
  }
  {
    SimCart c(t, vp, 7);
    drive(c, 2.0);
    c.imu_unplugged = true;
    c.runFor(0.5);
    c.imu_unplugged = false;
    c.runFor(1.5);
    bool recovered = !c.app.imuFault() && c.app.state() == RunState::FAULT;
    c.command("start");
    c.runFor(2.0);
    line("IMU reconnects: fault stays until 'start'", recovered && c.app.state() == RunState::RUNNING, "");
  }
  std::printf("\nFail-safe checks: %s\n", all_ok ? "PASS" : "FAIL");
  return all_ok ? 0 : 1;
}

// ---------------------------------------------------------------- EKF

static void evalEkfOnce(int runs, bool with_bias);

int evalEkf(int runs) {
  evalEkfOnce(runs, true);
  std::printf("\nSame drives with a perfectly calibrated accelerometer (no bias), so only vibration\n"
              "and acceleration errors remain:\n\n");
  evalEkfOnce(runs, false);
  std::printf("\nWith bias, every filter sits near the ~0.5 deg floor that an uncalibrated accelerometer\n"
              "bias causes (0.08 m/s^2 -> 0.47 deg); calibrate the MPU6050 offsets on the cart.\n");
  return 0;
}

static void evalEkfOnce(int runs, bool with_bias) {
  // Same IMU samples into four estimators; compare with the simulator's true attitude.
  struct Est {
    explicit Est(const char* n) : name(n) {}
    const char* name;
    double sq_smooth = 0, sq_rough = 0, vz_sq = 0;
    long n_smooth = 0, n_rough = 0, n_vz = 0;
  };
  Est est[4] = {Est("EKF, noise follows classifier (deployed)"), Est("EKF, fixed noise tuned for smooth road"),
                Est("EKF, fixed noise tuned for rough road"), Est("Accelerometer tilt only (no filter)")};
  for (int run = 0; run < runs; ++run) {
    unsigned seed = EVAL_SEED + 500 + (unsigned)run;
    TrackOptions to;
    to.length_m = 40;
    to.rumble = false;
    Track track = randomTrack(seed, to);
    std::mt19937 rng(seed);
    VehicleParams vp = randomVehicle(rng);
    vp.mount_tilt_deg = 5.0;
    if (!with_bias) vp.accel_bias = 0.0;
    SimCart cart(track, vp, seed);
    cart.begin();
    cart.command("start");
    Ekf fixed_smooth, fixed_rough;
    long step = 0;
    // The App reads the IMU at the start of every 10th millisecond. Compare all
    // estimators on that same sample and against the truth at that same instant.
    ImuSample prev_s;
    VehicleTruth prev_truth = cart.car.truth();
    double prev_front = cart.car.frontX();
    cart.car.readImu(prev_s);
    cart.on_ms = [&] {
      ++step;
      if (step % 10 == 1) {
        const ImuSample& s = prev_s;
        const VehicleTruth& tr = prev_truth;
        const float g[3] = {s.gx, s.gy, s.gz}, a[3] = {s.ax, s.ay, s.az};
        fixed_smooth.step(g, a, 0.01f, cfg::EKF_R_SMOOTH);
        fixed_rough.step(g, a, 0.01f, cfg::EKF_R_GRAVEL);
        if (cart.time() >= 3.0) {  // let everything converge
          double acc_pitch = std::atan2(-s.ax, std::sqrt(s.ay * s.ay + s.az * s.az));
          double acc_roll = std::atan2(s.ay, s.az);
          double e[4][2] = {{cart.app.ekf().pitch() - tr.pitch, cart.app.ekf().roll() - tr.roll},
                            {fixed_smooth.pitch() - tr.pitch, fixed_smooth.roll() - tr.roll},
                            {fixed_rough.pitch() - tr.pitch, fixed_rough.roll() - tr.roll},
                            {acc_pitch - tr.pitch, acc_roll - tr.roll}};
          bool rough = track.baseAt(prev_front) != SURFACE_SMOOTH || cart.app.decision().top != SURFACE_SMOOTH;
          for (int k = 0; k < 4; ++k) {
            double sq = e[k][0] * e[k][0] + e[k][1] * e[k][1];
            if (rough) {
              est[k].sq_rough += sq;
              ++est[k].n_rough;
            } else {
              est[k].sq_smooth += sq;
              ++est[k].n_smooth;
            }
          }
          double dv = cart.app.ekf().vz() - tr.vz;
          est[0].vz_sq += dv * dv;
          ++est[0].n_vz;
        }
      }
      cart.car.readImu(prev_s);
      prev_truth = cart.car.truth();
      prev_front = cart.car.frontX();
    };
    cart.runUntil([&] { return cart.car.rearX() > track.length() - 1.0; }, 400);
  }
  if (with_bias)
    std::printf("Attitude estimation over %d random drives (sensor mounted 5 deg crooked, gyro and\n"
                "accelerometer bias, vibration). RMS error of roll+pitch vs. the true attitude, degrees.\n\n", runs);
  std::printf("%-44s %12s %12s\n", "Estimator", "smooth road", "rough road");
  for (int k = 0; k < 4; ++k)
    std::printf("%-44s %12.3f %12.3f\n", est[k].name, 57.2958 * std::sqrt(est[k].sq_smooth / std::max(1L, est[k].n_smooth)),
                57.2958 * std::sqrt(est[k].sq_rough / std::max(1L, est[k].n_rough)));
  std::printf("Vertical body velocity (used by the damping curve): RMS error %.3f m/s.\n",
              std::sqrt(est[0].vz_sq / std::max(1L, est[0].n_vz)));
}

// ---------------------------------------------------------------- maintenance

static double runMaintTest(double wear, unsigned seed, FakeStorage& store, MaintStatus* status, bool* ok) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1, 1);
  Track t(seed);
  t.addSmooth(2.0);
  t.addBreaker(0.015 * (1 + 0.1 * u(rng)), 0.10 * (1 + 0.1 * u(rng)));  // same bump, placed by hand
  t.addSmooth(15);
  VehicleParams vp;
  vp.wear = wear;
  vp.mass *= 1 + 0.05 * u(rng);  // a little load variation day to day
  SimCart cart(t, vp, seed);
  cart.rig.storage.blobs = store.blobs;
  cart.begin();
  cart.command("maint test");
  cart.runFor(0.2);
  cart.runUntil([&] { return cart.app.state() == RunState::IDLE; }, 30);
  store.blobs = cart.rig.storage.blobs;
  *status = cart.app.maintenance().status();
  *ok = cart.rig.console.contains("Maintenance: zeta=");
  return cart.app.maintenance().state().last_zeta;
}

int evalMaintenance() {
  std::printf("Predictive maintenance: one reference-bump test per day for 40 days.\n"
              "Wear: healthy for 10 days, then damping falls 2%% per day (shock oil leaking).\n\n");
  std::printf("%5s %8s %8s %10s\n", "day", "damping", "zeta", "status");
  FakeStorage store;
  int first_watch = -1, first_service = -1;
  for (int day = 1; day <= 40; ++day) {
    double wear = day <= 10 ? 1.0 : std::max(0.3, 1.0 - 0.02 * (day - 10));
    MaintStatus st;
    bool ok;
    double z = runMaintTest(wear, 4000 + day, store, &st, &ok);
    if (st == MaintStatus::WATCH && first_watch < 0) first_watch = day;
    if (st == MaintStatus::SERVICE && first_service < 0) first_service = day;
    if (day <= 12 || day % 3 == 0 || st != MaintStatus::HEALTHY)
      std::printf("%5d %7.0f%% %8.4f %10s%s\n", day, wear * 100, z, maintStatusName(st), ok ? "" : "  (test failed)");
  }
  std::printf("\nFirst WATCH on day %d (damping %.0f%%), first SERVICE on day %d (damping %.0f%%).\n", first_watch,
              100 * (first_watch <= 10 ? 1.0 : 1.0 - 0.02 * (first_watch - 10)), first_service,
              100 * (first_service <= 10 ? 1.0 : 1.0 - 0.02 * (first_service - 10)));

  FakeStorage healthy;
  int false_alarms = 0;
  for (int day = 1; day <= 40; ++day) {
    MaintStatus st;
    bool ok;
    runMaintTest(1.0, 8000 + day, healthy, &st, &ok);
    false_alarms += st == MaintStatus::WATCH || st == MaintStatus::SERVICE;
  }
  std::printf("Healthy shocks for 40 days: %d false alarms.\n", false_alarms);
  return 0;
}

// ---------------------------------------------------------------- uncertainty

int evalUncertainty(int runs) {
  long flagged[2] = {0, 0}, total[2] = {0, 0};
  double servo_sum[2] = {0, 0}, pwm_sum[2] = {0, 0};
  long rules_gravel = 0, rules_total = 0;
  for (int run = 0; run < runs; ++run) {
    unsigned seed = EVAL_SEED + 700 + (unsigned)run;
    Track t(seed);
    t.addSmooth(3);
    t.addRumble(4);
    t.addSmooth(3);
    t.addGravel(4, 0.002);
    t.addSmooth(3);
    t.addRumble(4);
    t.addSmooth(3);
    for (int mode = 0; mode < 2; ++mode) {
      SimCart cart(t, VehicleParams(), seed);
      cart.app.setClassifierMl(mode == 0);
      cart.begin();
      cart.command("start");
      WindowLabeler lab(cart.track);
      long step = 0;
      cart.on_ms = [&] {
        ++step;
        if (step % 10 == 0) lab.record(cart.time(), cart.car.frontX(), cart.car.rearX(), cart.car.truth().v);
        if (step % 100 || !cart.app.featuresValid()) return;
        int label = lab.label(cart.time());
        if (label < 0) return;
        int k = label == LABEL_OOD ? 1 : 0;
        if (mode == 0) {
          ++total[k];
          flagged[k] += cart.app.decision().uncertain;
          servo_sum[k] += cart.rig.servo.angle;
          pwm_sum[k] += cart.rig.motors.pwm;
        } else if (k == 1) {
          ++rules_total;
          rules_gravel += cart.app.decision().top == SURFACE_GRAVEL;
        }
      };
      cart.runUntil([&] { return cart.car.rearX() > t.length() - 1.0; }, 200);
    }
  }
  std::printf("Unseen surface: rumble strips (never in the training data), %d drives.\n\n", runs);
  std::printf("%-30s %12s %12s %12s\n", "", "flagged", "mean servo", "mean PWM");
  const char* names[2] = {"Known surfaces", "Rumble strip (unseen)"};
  for (int k = 0; k < 2; ++k)
    std::printf("%-30s %11.1f%% %12.1f %12.1f\n", names[k], total[k] ? 100.0 * flagged[k] / total[k] : 0.0,
                total[k] ? servo_sum[k] / total[k] : 0.0, total[k] ? pwm_sum[k] / total[k] : 0.0);
  double fk = total[0] ? 100.0 * flagged[0] / total[0] : 0.0, fo = total[1] ? 100.0 * flagged[1] / total[1] : 0.0;
  std::printf("\nRule-based classifier on the rumble strip: calls it gravel %.0f%% of the time and has no\n"
              "uncertainty signal at all.\n",
              rules_total ? 100.0 * rules_gravel / rules_total : 0.0);
  std::printf("Ensemble: flags %.1f%% of rumble-strip windows vs %.1f%% on known surfaces%s\n", fo, fk,
              fo > 2 * fk + 1 ? " - it notices the new surface." :
              " - it does NOT reliably notice this new surface;\nit mostly treats it like gravel. See docs/model_report.md for the offline numbers.");
  return 0;
}

// ---------------------------------------------------------------- record

int recordSurface(const char* name, double seconds, unsigned seed) {
  Surface s;
  if (!parseSurface(name, &s) || s == SURFACE_UNKNOWN) {
    if (std::string(name) == "bump") s = SURFACE_BREAKER;
    else {
      std::fprintf(stderr, "surface must be smooth, gravel, breaker or pothole\n");
      return 2;
    }
  }
  Track t(seed);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0, 1);
  t.addSmooth(1.5);
  double len = seconds * 0.4 + 2;
  while (t.length() < len) {
    if (s == SURFACE_SMOOTH) t.addSmooth(5);
    if (s == SURFACE_GRAVEL) t.addGravel(5, 0.001 + 0.002 * u(rng));
    if (s == SURFACE_BREAKER) {
      t.addBreaker(0.01 + 0.02 * u(rng), 0.06 + 0.09 * u(rng));
      t.addSmooth(1.2);
    }
    if (s == SURFACE_POTHOLE) {
      t.addPothole(0.01 + 0.02 * u(rng), 0.05 + 0.07 * u(rng));
      t.addSmooth(1.2);
    }
  }
  SimCart cart(t, VehicleParams(), seed);
  cart.begin();
  cart.command("log raw");
  cart.command("start");
  cart.runFor(seconds);
  const std::string& out = cart.rig.console.output;
  size_t pos = 0;
  while (pos < out.size()) {
    size_t end = out.find('\n', pos);
    if (end == std::string::npos) end = out.size();
    if (out.compare(pos, 2, "R,") == 0 || out.compare(pos, 2, "# ") == 0)
      std::fwrite(out.data() + pos, 1, end - pos + 1, stdout);
    pos = end + 1;
  }
  return 0;
}

// ---------------------------------------------------------------- tune

// Characterise the actuator: for each surface, drive with the servo locked at
// several stiffness settings and measure comfort and protection. The best
// setting per surface becomes STIFF_<CLASS> in config.h (the curve's targets).
int evalTune(Domain domain) {
  const bool shifted = domain == Domain::SHIFTED;
  const double S[] = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
  const int NS = 6, SEEDS = 4;
  const char* names[4] = {"smooth", "gravel", "breaker", "pothole"};
  double best_s[4];
  std::printf("Damping characterisation: servo locked at each stiffness, %d cars per cell, PWM 200.\n"
              "smooth/gravel cost = RMS body accel (m/s^2) + 3 x bottom-outs per metre\n"
              "breaker/pothole cost = mean peak body accel / 10 + bottom-outs per event\n\n", SEEDS);
  std::printf("%-8s", "surface");
  for (double s : S) std::printf("   s=%.1f (%3.0f deg)  ", s, 45 + 105 * s);
  std::printf("\n");
  for (int cls = 0; cls < 4; ++cls) {
    double best = 1e18;
    best_s[cls] = 0;
    std::printf("%-8s", names[cls]);
    for (int k = 0; k < NS; ++k) {
      double cost_sum = 0;
      for (int seed = 0; seed < SEEDS; ++seed) {
        unsigned sd = 7000 + 100 * cls + seed;
        std::mt19937 rng(sd);
        std::uniform_real_distribution<double> u(0, 1);
        Track t(sd, shifted ? 10.0 : 20.0, shifted ? 45.0 : 80.0);
        t.addSmooth(2.0);
        int events = 0;
        double surface_m = 20.0;
        if (cls == 0) t.addSmooth(surface_m);
        if (cls == 1) t.addGravel(surface_m, shifted ? 0.002 + 0.0018 * u(rng) : 0.001 + 0.002 * u(rng));
        if (cls >= 2) {
          for (int e = 0; e < 8; ++e) {
            double h = shifted ? 0.015 + 0.021 * u(rng) : 0.01 + 0.02 * u(rng);
            if (cls == 2) t.addBreaker(h, 0.06 + 0.09 * u(rng));
            else t.addPothole(h, 0.05 + 0.07 * u(rng));
            t.addSmooth(1.5);
            ++events;
          }
        }
        t.addSmooth(2.0);
        VehicleParams vp = randomVehicle(rng, domain);
        vp.wear = 1.0;
        SimCart cart(t, vp, sd);
        cart.app.setSpeedMode(SpeedMode::STEP);
        cart.begin();
        cart.command("servo " + std::to_string((int)(45 + 105 * S[k] + 0.5)));
        cart.command("start");
        double sq = 0, peak_sum = 0, peak = 0;
        long n = 0;
        size_t next_ev = 0;
        double ev_start = -1;
        std::vector<EventEntry> evs = eventsOf(t);
        cart.on_ms = [&] {
          const VehicleTruth& tr = cart.car.truth();
          double fx = cart.car.frontX();
          if (cls < 2 && fx > 3.0 && cart.car.rearX() < 2.0 + surface_m) {
            sq += tr.heave_acc * tr.heave_acc;
            ++n;
          }
          if (next_ev < evs.size() && fx >= evs[next_ev].x0) {
            if (ev_start >= 0) peak_sum += peak;
            ev_start = cart.time();
            peak = 0;
            ++next_ev;
          }
          if (ev_start >= 0) peak = std::max(peak, std::fabs(tr.heave_acc));
        };
        cart.runUntil([&] { return cart.car.rearX() > t.length() - 1.0; }, 300);
        if (ev_start >= 0) peak_sum += peak;
        double bottoms = cart.car.bottomOuts();
        double cost = cls < 2 ? std::sqrt(sq / std::max(1L, n)) + 3.0 * bottoms / surface_m
                              : peak_sum / std::max(1, events) / 10.0 + bottoms / std::max(1, events);
        cost_sum += cost;
      }
      double c = cost_sum / SEEDS;
      std::printf("   %-19.3f", c);
      if (c < best) {
        best = c;
        best_s[cls] = S[k];
      }
    }
    std::printf("\n");
  }
  std::printf("\nSuggested curve targets (config.h):\n");
  const char* keys[4] = {"STIFF_SMOOTH", "STIFF_GRAVEL", "STIFF_BREAKER", "STIFF_POTHOLE"};
  for (int c = 0; c < 4; ++c) std::printf("constexpr float %-13s = %.1ff;\n", keys[c], best_s[c]);
  return 0;
}
