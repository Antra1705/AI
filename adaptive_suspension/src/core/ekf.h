// ekf.h - Extended Kalman Filter: IMU + classifier confidence -> physical state (#2).
//
// State x = [roll, pitch, gyro bias x, gyro bias y, vertical velocity]
//   roll/pitch      rad, attitude of the sensor
//   gyro biases     rad/s, slowly drifting offsets of the gyro
//   vertical vel.   m/s, body heave velocity (used by the damping curve: a
//                   bouncing body asks for more damping, "skyhook" style)
//
// Predict (100 Hz): integrate bias-corrected gyro rates through the nonlinear
// Euler-angle kinematics; integrate gravity-removed vertical acceleration
// (with a small leak so it cannot drift away).
// Update 1 - gravity: the accelerometer points along gravity when the body is
// not accelerating: h(x) = [-g sin(pitch), g sin(roll)cos(pitch), g cos(roll)cos(pitch)].
// Update 2 - zero mean vertical velocity: the cart stays on the ground, so on
// average vz = 0 (weak pseudo-measurement).
//
// Classifier link: on rough surfaces the accelerometer is dominated by
// vibration, not gravity. The gravity-update noise R is multiplied by the
// expected roughness under the classifier's probabilities,
//   r_scale = sum_c p(c) * R_FACTOR[c]   (smooth 1 ... pothole 100),
// so the filter trusts the accelerometer only when the classifier says the
// road is smooth, and coasts on the gyro otherwise. R also grows when |a|
// differs from 1 g (the body is clearly accelerating).
#pragma once
#include <stdint.h>

enum { EKF_N = 5, EKF_ROLL = 0, EKF_PITCH = 1, EKF_BGX = 2, EKF_BGY = 3, EKF_VZ = 4 };

class Ekf {
 public:
  Ekf() { reset(); }
  void reset();

  // Initialise attitude from one accelerometer sample (m/s^2).
  void initFromAccel(const float f[3]);
  bool initialized() const { return init_; }

  // gyro rad/s, accel m/s^2, dt s. r_scale from the classifier (1 = trust accel fully).
  void step(const float gyro[3], const float accel[3], float dt, float r_scale);

  float roll() const { return x_[EKF_ROLL]; }
  float pitch() const { return x_[EKF_PITCH]; }
  float vz() const { return x_[EKF_VZ]; }
  float biasX() const { return x_[EKF_BGX]; }
  float biasY() const { return x_[EKF_BGY]; }
  float verticalAccel() const { return az_world_; }  // gravity removed, m/s^2
  float variance(int i) const { return P_[i][i]; }
  const float* state() const { return x_; }

  // Model functions, public so the tests can check the Jacobians numerically.
  static void process(const float x[EKF_N], const float gyro[3], const float accel[3], float xdot[EKF_N]);
  static void processJacobian(const float x[EKF_N], const float gyro[3], const float accel[3], float A[EKF_N][EKF_N]);
  static void measure(const float x[EKF_N], float h[3]);
  static void measureJacobian(const float x[EKF_N], float H[3][EKF_N]);

 private:
  void gravityUpdate(const float accel[3], float r_scale);
  void zeroVelocityUpdate();

  float x_[EKF_N];
  float P_[EKF_N][EKF_N];
  float az_world_ = 0.0f;
  bool init_ = false;
};

// Classifier probabilities -> measurement-noise multiplier for the gravity update.
float ekfNoiseScale(const float p[4], bool classifier_ready);
