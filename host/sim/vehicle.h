// vehicle.h - physical model of the test cart for the simulator.
//
//  Track    road height profile r(x): micro-roughness everywhere, plus gravel
//           regions, speed breakers, potholes and (for uncertainty tests only)
//           rumble strips, a surface the classifier never sees in training.
//  Vehicle  half-car: body heave z and pitch theta on a front and a rear
//           spring-damper (plus a roll mode), driven by the road under each
//           wheel. The damper coefficient follows the servo angle, so the
//           suspension control has a real, measurable effect. 'wear' scales
//           the damping down to model worn shocks (predictive maintenance).
//  Sensor   MPU6050 model: specific force + body rates in the sensor frame,
//           mounting tilt, bias, noise and the 21 Hz digital low-pass filter.
//
// All parameters are plausible for a ~1 kg cart, not measured. They make the
// simulation physically consistent; real numbers come from the hardware.
#pragma once
#include <cstdint>
#include <random>
#include <vector>

#include "core/types.h"

enum class TrackKind { GRAVEL, BREAKER, POTHOLE, RUMBLE };

struct TrackFeature {
  TrackKind kind;
  double x0, x1;    // start / end position, m
  double size;      // gravel/rumble: roughness height m; breaker: height m; pothole: depth m
};

class Track {
 public:
  explicit Track(unsigned seed);
  void addSmooth(double len);
  void addGravel(double len, double rms_m);
  void addBreaker(double height_m, double length_m);
  void addPothole(double depth_m, double length_m);
  void addRumble(double len);
  // Live mode: make [from, to] plain smooth road with one reference bump at bump_at
  // (removes any feature overlapping that stretch).
  void placeReferenceBump(double from, double to, double bump_at, double height_m, double length_m);

  double length() const { return end_; }
  double elevation(double x) const;  // m
  double lateral(double x) const;    // left-right height difference, m (drives roll)
  const std::vector<TrackFeature>& features() const { return features_; }

  // Base surface under position x: SMOOTH, GRAVEL, or -1 for rumble strip.
  int baseAt(double x) const;

 private:
  struct Wave {
    double k, amp, phase;  // spatial frequency rad/m, amplitude m, phase
  };
  double sumWaves(const std::vector<Wave>& w, double x) const;
  static double edgeTaper(double x, double x0, double x1, double ramp);

  std::mt19937 rng_;
  std::vector<Wave> micro_, gravel_, gravel_lat_, micro_lat_;
  std::vector<TrackFeature> features_;
  double end_ = 0.0;
};

struct VehicleParams {
  double mass = 1.2;            // kg, sprung
  double half_wheelbase = 0.08; // m
  double k_axle = 379.0;        // N/m per axle -> ~4 Hz heave
  double c_soft = 3.6;          // N s/m per axle at servo 45 deg  (zeta ~0.12)
  double c_stiff = 27.0;        // N s/m per axle at servo 150 deg (zeta ~0.9)
  double travel = 0.012;        // m suspension travel before the bump stop
  double wear = 1.0;            // 1 = healthy, < 1 = worn shocks (damping scaled)
  double vmax_at_255 = 0.6;     // m/s top speed at full PWM
  double motor_tau = 0.15;      // s
  double brake_decel = 3.0;     // m/s^2
  int motor_min_pwm = 40;       // below this the motors stall
  double servo_slew = 300.0;    // deg/s, MG90S under load
  // sensor
  double accel_noise = 0.05, gyro_noise = 0.004;
  double accel_bias = 0.08, gyro_bias = 0.02;  // max magnitude, randomised per run
  double mount_tilt_deg = 3.0;                 // max, randomised per run
  double filter_hz = 21.0;
};

struct VehicleTruth {
  double x, v, a_long;   // position m, speed m/s, longitudinal accel
  double heave_acc;      // body vertical acceleration (m/s^2, gravity excluded)
  double vz;             // body vertical velocity m/s
  double roll, pitch;    // sensor-frame attitude (rad), what the EKF should estimate
  double defl_front, defl_rear;  // suspension deflection m
  double servo_actual;   // deg
  double stiffness;      // 0..1 from the actual servo angle
};

class Vehicle {
 public:
  Vehicle(const Track& track, const VehicleParams& p, unsigned seed);

  // Advance by dt seconds. pwm/braked come from the motor driver, servo_cmd
  // is the last commanded angle (-1 = never commanded, treated as 90).
  void step(double dt, int pwm, bool braked, int servo_cmd);

  // Current filtered IMU output (what the MPU6050 would return).
  void readImu(ImuSample& s) const;

  const VehicleTruth& truth() const { return truth_; }
  double frontX() const { return truth_.x + p_.half_wheelbase; }
  double rearX() const { return truth_.x - p_.half_wheelbase; }
  int bottomOuts() const { return bottom_outs_; }
  VehicleParams& params() { return p_; }

 private:
  const Track& track_;
  VehicleParams p_;
  std::mt19937 rng_;
  std::normal_distribution<double> gauss_{0.0, 1.0};

  double z_ = 0, zd_ = 0, th_ = 0, thd_ = 0, ph_ = 0, phd_ = 0;
  double servo_angle_ = 90.0;
  double accel_bias_[3], gyro_bias_[3];
  double mount_[3][3];        // body -> sensor rotation
  double filt_[6];            // filtered sensor output
  bool filt_init_ = false;
  double t_ = 0.0;
  bool bottomed_ = false;
  int bottom_outs_ = 0;
  VehicleTruth truth_;
};
