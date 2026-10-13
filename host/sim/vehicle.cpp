#include "vehicle.h"

#include <algorithm>
#include <cmath>

static const double PI = 3.14159265358979323846;
static const double G = 9.81;

// ---------------------------------------------------------------- Track

static void makeWaves(std::mt19937& rng, std::vector<double>& k, int n, double f_lo, double f_hi) {
  // Log-spaced spatial frequencies (cycles per metre) between f_lo and f_hi.
  std::uniform_real_distribution<double> jitter(0.9, 1.1);
  for (int i = 0; i < n; ++i) {
    double f = f_lo * std::pow(f_hi / f_lo, (i + 0.5) / n) * jitter(rng);
    k.push_back(2.0 * PI * f);
  }
}

Track::Track(unsigned seed) : rng_(seed) {
  std::uniform_real_distribution<double> phase(0.0, 2.0 * PI);
  auto build = [&](std::vector<Wave>& out, int n, double f_lo, double f_hi, double rms, double slope) {
    std::vector<double> k;
    makeWaves(rng_, k, n, f_lo, f_hi);
    double sum_sq = 0.0;
    std::vector<double> amp;
    for (double ki : k) {
      double a = std::pow(ki, -slope);  // rougher at long wavelengths, like real roads
      amp.push_back(a);
      sum_sq += 0.5 * a * a;
    }
    double scale = rms / std::sqrt(sum_sq);
    for (size_t i = 0; i < k.size(); ++i) out.push_back(Wave{k[i], amp[i] * scale, phase(rng_)});
  };
  build(micro_, 12, 0.5, 15.0, 0.0003, 0.5);  // 0.3 mm RMS everywhere
  build(micro_lat_, 8, 0.5, 15.0, 0.0002, 0.5);
  // Gravel: stones of 1.2-5 cm (20-80 cycles per metre), i.e. high-frequency
  // vibration well above the cart's ~4 Hz bounce resonance at normal speeds.
  build(gravel_, 40, 20.0, 80.0, 1.0, 0.3);   // unit RMS, scaled per region
  build(gravel_lat_, 30, 20.0, 80.0, 1.0, 0.3);
}

void Track::addSmooth(double len) { end_ += len; }

void Track::addGravel(double len, double rms_m) {
  features_.push_back(TrackFeature{TrackKind::GRAVEL, end_, end_ + len, rms_m});
  end_ += len;
}

void Track::addBreaker(double height_m, double length_m) {
  features_.push_back(TrackFeature{TrackKind::BREAKER, end_, end_ + length_m, height_m});
  end_ += length_m;
}

void Track::addPothole(double depth_m, double length_m) {
  features_.push_back(TrackFeature{TrackKind::POTHOLE, end_, end_ + length_m, depth_m});
  end_ += length_m;
}

void Track::addRumble(double len) {
  features_.push_back(TrackFeature{TrackKind::RUMBLE, end_, end_ + len, 0.004});
  end_ += len;
}

double Track::sumWaves(const std::vector<Wave>& w, double x) const {
  double s = 0.0;
  for (const Wave& wv : w) s += wv.amp * std::sin(wv.k * x + wv.phase);
  return s;
}

double Track::edgeTaper(double x, double x0, double x1, double ramp) {
  if (x <= x0 || x >= x1) return 0.0;
  double a = std::min((x - x0) / ramp, 1.0);
  double b = std::min((x1 - x) / ramp, 1.0);
  return std::min(a, b);
}

double Track::elevation(double x) const {
  double r = sumWaves(micro_, x);
  for (const TrackFeature& f : features_) {
    if (x < f.x0 || x > f.x1) continue;
    double u = (x - f.x0) / (f.x1 - f.x0);
    switch (f.kind) {
      case TrackKind::GRAVEL:
        r += f.size * edgeTaper(x, f.x0, f.x1, 0.05) * sumWaves(gravel_, x);
        break;
      case TrackKind::BREAKER:  // raised cosine hump
        r += f.size * 0.5 * (1.0 - std::cos(2.0 * PI * u));
        break;
      case TrackKind::POTHOLE:  // dip with ~1.5 cm edges (the wheel rounds off sharp corners)
        r -= f.size * edgeTaper(x, f.x0, f.x1, 0.015);
        break;
      case TrackKind::RUMBLE: {  // ridges every 6 cm, 2 cm wide
        double p = std::fmod(x - f.x0, 0.06);
        if (p < 0.02) r += f.size * edgeTaper(x, f.x0, f.x1, 0.05) * 0.5 * (1.0 - std::cos(2.0 * PI * p / 0.02));
        break;
      }
    }
  }
  return r;
}

double Track::lateral(double x) const {
  double r = sumWaves(micro_lat_, x);
  for (const TrackFeature& f : features_) {
    if (x < f.x0 || x > f.x1) continue;
    if (f.kind == TrackKind::GRAVEL) r += 0.7 * f.size * edgeTaper(x, f.x0, f.x1, 0.05) * sumWaves(gravel_lat_, x);
  }
  return r;
}

int Track::baseAt(double x) const {
  for (const TrackFeature& f : features_) {
    if (x < f.x0 || x > f.x1) continue;
    if (f.kind == TrackKind::GRAVEL) return SURFACE_GRAVEL;
    if (f.kind == TrackKind::RUMBLE) return -1;
  }
  return SURFACE_SMOOTH;
}

// ---------------------------------------------------------------- rotation helpers

static void rotX(double a, double m[3][3]) {
  double c = std::cos(a), s = std::sin(a);
  double r[3][3] = {{1, 0, 0}, {0, c, -s}, {0, s, c}};
  std::copy(&r[0][0], &r[0][0] + 9, &m[0][0]);
}
static void rotY(double a, double m[3][3]) {
  double c = std::cos(a), s = std::sin(a);
  double r[3][3] = {{c, 0, s}, {0, 1, 0}, {-s, 0, c}};
  std::copy(&r[0][0], &r[0][0] + 9, &m[0][0]);
}
static void mul(const double a[3][3], const double b[3][3], double out[3][3]) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      out[i][j] = 0;
      for (int k = 0; k < 3; ++k) out[i][j] += a[i][k] * b[k][j];
    }
}
static void mulTv(const double m[3][3], const double v[3], double out[3]) {  // m^T v
  for (int i = 0; i < 3; ++i) out[i] = m[0][i] * v[0] + m[1][i] * v[1] + m[2][i] * v[2];
}

// ---------------------------------------------------------------- Vehicle

Vehicle::Vehicle(const Track& track, const VehicleParams& p, unsigned seed) : track_(track), p_(p), rng_(seed) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int i = 0; i < 3; ++i) {
    accel_bias_[i] = p_.accel_bias * u(rng_);
    gyro_bias_[i] = p_.gyro_bias * u(rng_);
  }
  double tilt = p_.mount_tilt_deg * PI / 180.0;
  double rx[3][3], ry[3][3];
  rotX(tilt * u(rng_), rx);
  rotY(tilt * u(rng_), ry);
  mul(ry, rx, mount_);
  for (double& f : filt_) f = 0.0;
  truth_ = VehicleTruth();
  // Start at rest on the road under the wheels.
  z_ = 0.5 * (track_.elevation(frontX()) + track_.elevation(rearX()));
}

void Vehicle::step(double dt, int pwm, bool braked, int servo_cmd) {
  // --- servo: slews toward the commanded angle at a finite rate
  double cmd = servo_cmd < 0 ? 90.0 : (double)servo_cmd;
  double max_move = p_.servo_slew * dt;
  servo_angle_ += std::max(-max_move, std::min(max_move, cmd - servo_angle_));
  double s = std::max(0.0, std::min(1.0, (servo_angle_ - 45.0) / 105.0));
  double c = p_.wear * (p_.c_soft + s * (p_.c_stiff - p_.c_soft));

  // --- longitudinal motion
  double v = truth_.v;
  double v_prev = v;
  if (braked || pwm < p_.motor_min_pwm) {
    double decel = braked ? p_.brake_decel : 0.6;  // stalled motors still drag
    v = std::max(0.0, v - decel * dt);
  } else {
    double target = p_.vmax_at_255 * pwm / 255.0;
    v += (target - v) * dt / p_.motor_tau;
  }
  truth_.x += v * dt;
  truth_.v = v;
  truth_.a_long = (v - v_prev) / dt;

  // --- road under each wheel and its rate of change
  const double a = p_.half_wheelbase, h = 1e-4;
  double xf = frontX(), xr = rearX();
  double rf = track_.elevation(xf), rr = track_.elevation(xr);
  double rfd = (track_.elevation(xf + h) - track_.elevation(xf - h)) / (2 * h) * v;
  double rrd = (track_.elevation(xr + h) - track_.elevation(xr - h)) / (2 * h) * v;
  double lat = 0.5 * (track_.lateral(xf) + track_.lateral(xr)) / 0.12;  // 12 cm track width -> roll input, rad
  double latd = 0.5 * ((track_.lateral(xf + h) - track_.lateral(xf - h)) +
                       (track_.lateral(xr + h) - track_.lateral(xr - h))) / (2 * h) * v / 0.12;

  // --- suspension forces (spring + damper + bump stop)
  auto force = [&](double defl, double defl_rate) {
    double f = p_.k_axle * defl + c * defl_rate;
    double over = std::fabs(defl) - p_.travel;
    if (over > 0) f += 30.0 * p_.k_axle * over * (defl > 0 ? 1 : -1);
    return f;
  };
  double df = rf - (z_ + a * th_), dr = rr - (z_ - a * th_);
  double Ff = force(df, rfd - (zd_ + a * thd_));
  double Fr = force(dr, rrd - (zd_ - a * thd_));
  bool bottomed = std::fabs(df) > p_.travel || std::fabs(dr) > p_.travel;
  if (bottomed && !bottomed_) ++bottom_outs_;
  bottomed_ = bottomed;

  const double m = p_.mass;
  const double Iy = m * 0.07 * 0.07;
  const double Ix = m * 0.05 * 0.05;
  // Roll: 5 Hz mode whose damping ratio follows the heave damping ratio.
  const double w_roll = 2 * PI * 5.0;
  const double k_roll = Ix * w_roll * w_roll;
  const double zeta = c / std::sqrt(2.0 * p_.k_axle * m);
  const double c_roll = 2.0 * zeta * Ix * w_roll;
  double zdd = (Ff + Fr) / m;
  double thdd = (a * Ff - a * Fr) / Iy;
  double phdd = (k_roll * (lat - ph_) + c_roll * (latd - phd_)) / Ix;

  // semi-implicit Euler
  zd_ += zdd * dt;  z_ += zd_ * dt;
  thd_ += thdd * dt; th_ += thd_ * dt;
  phd_ += phdd * dt; ph_ += phd_ * dt;

  truth_.heave_acc = zdd;
  truth_.vz = zd_;
  truth_.defl_front = df;
  truth_.defl_rear = dr;
  truth_.servo_actual = servo_angle_;
  truth_.stiffness = s;

  // --- sensor: specific force and body rates, rotated into the sensor frame
  double Rb[3][3], rx[3][3], ry[3][3], Rs[3][3];
  rotY(th_, ry);
  rotX(ph_, rx);
  mul(ry, rx, Rb);       // world <- body
  mul(Rb, mount_, Rs);   // world <- sensor
  truth_.pitch = -std::asin(std::max(-1.0, std::min(1.0, Rs[2][0])));
  truth_.roll = std::atan2(Rs[2][1], Rs[2][2]);

  double wheel_hz = v / (2 * PI * 0.033);
  double vib = 0.05 * (v / 0.5) * (std::sin(2 * PI * wheel_hz * t_) + 0.5 * std::sin(4 * PI * wheel_hz * t_ + 1.0));
  double f_world[3] = {truth_.a_long + 0.3 * vib, 0.0, zdd + G + vib};
  double f_sensor[3];
  mulTv(Rs, f_world, f_sensor);
  double w_body[3] = {phd_, std::cos(ph_) * thd_, -std::sin(ph_) * thd_};
  double w_sensor[3];
  mulTv(mount_, w_body, w_sensor);

  double raw[6];
  for (int i = 0; i < 3; ++i) {
    raw[i] = f_sensor[i] + accel_bias_[i] + p_.accel_noise * gauss_(rng_);
    raw[3 + i] = w_sensor[i] + gyro_bias_[i] + p_.gyro_noise * gauss_(rng_);
  }
  double alpha = 1.0 - std::exp(-dt * 2 * PI * p_.filter_hz);
  for (int i = 0; i < 6; ++i) filt_[i] = filt_init_ ? filt_[i] + alpha * (raw[i] - filt_[i]) : raw[i];
  filt_init_ = true;
  t_ += dt;
}

void Vehicle::readImu(ImuSample& s) const {
  s.ax = (float)filt_[0]; s.ay = (float)filt_[1]; s.az = (float)filt_[2];
  s.gx = (float)filt_[3]; s.gy = (float)filt_[4]; s.gz = (float)filt_[5];
}
