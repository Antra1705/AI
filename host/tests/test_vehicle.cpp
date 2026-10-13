// Sanity checks of the simulator's physics, so later results rest on something solid.
#include <algorithm>
#include <cmath>

#include "test.h"
#include "vehicle.h"

TEST(vehicle_rests_at_one_g_on_flat_road) {
  Track track(1);
  track.addSmooth(10);
  VehicleParams p;
  p.accel_noise = 0;
  p.gyro_noise = 0;
  p.accel_bias = 0;
  p.gyro_bias = 0;
  p.mount_tilt_deg = 0;
  Vehicle v(track, p, 1);
  for (int i = 0; i < 2000; ++i) v.step(0.001, 0, true, 90);
  ImuSample s;
  v.readImu(s);
  CHECK_NEAR(std::sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az), 9.81, 0.05);
  CHECK_NEAR(v.truth().v, 0.0, 1e-9);
}

TEST(vehicle_drives_at_pwm_speed) {
  Track track(1);
  track.addSmooth(50);
  VehicleParams p;
  Vehicle v(track, p, 1);
  for (int i = 0; i < 3000; ++i) v.step(0.001, 255, false, 90);
  CHECK_NEAR(v.truth().v, p.vmax_at_255, 0.01);
  for (int i = 0; i < 3000; ++i) v.step(0.001, 0, true, 90);
  CHECK_NEAR(v.truth().v, 0.0, 1e-9);
}

TEST(vehicle_servo_slews_at_finite_rate) {
  Track track(1);
  track.addSmooth(10);
  VehicleParams p;
  Vehicle v(track, p, 1);
  v.step(0.1, 0, true, 150);  // 0.1 s at 300 deg/s = 30 deg max
  CHECK_NEAR(v.truth().servo_actual, 120.0, 1e-6);
}

// Response after a 2 cm speed breaker: peak in the first 0.3 s after the rear
// wheel leaves it, and the leftover bounce from 0.3 s to 1 s.
struct Bounce {
  double peak, late_rms;
  int bottom_outs;
};

static Bounce bounceAfterBreaker(int servo, double wear) {
  Track track(2);
  track.addSmooth(1.0);
  track.addBreaker(0.02, 0.10);
  track.addSmooth(3.0);
  VehicleParams p;
  p.wear = wear;
  Vehicle v(track, p, 2);
  Bounce b = {0, 0, 0};
  double t_leave = -1, sum = 0;
  int n = 0;
  for (int i = 0; i < 8000; ++i) {
    v.step(0.001, 170, false, servo);
    double t = i * 0.001;
    if (t_leave < 0 && v.rearX() > 1.10) t_leave = t;
    if (t_leave < 0) continue;
    double a = v.truth().heave_acc;
    if (t - t_leave < 0.3) {
      b.peak = std::max(b.peak, std::fabs(a));
    } else if (t - t_leave < 1.0) {
      sum += a * a;
      ++n;
    }
  }
  b.late_rms = std::sqrt(sum / n);
  b.bottom_outs = v.bottomOuts();
  return b;
}

// Stiffer damping must tame the breaker: the whole premise of adaptive suspension.
TEST(vehicle_stiff_damping_protects_on_breaker) {
  Bounce soft = bounceAfterBreaker(45, 1.0);
  Bounce stiff = bounceAfterBreaker(150, 1.0);
  CHECK(stiff.peak < soft.peak);
  CHECK(stiff.late_rms < soft.late_rms);
  CHECK(soft.bottom_outs > 0);
  CHECK_EQ(stiff.bottom_outs, 0);
}

// Worn shocks keep bouncing longer and bottom out more often.
TEST(vehicle_worn_shocks_bounce_longer) {
  CHECK(bounceAfterBreaker(150, 0.4).late_rms > 1.5 * bounceAfterBreaker(150, 1.0).late_rms);
  CHECK(bounceAfterBreaker(45, 0.4).bottom_outs > bounceAfterBreaker(45, 1.0).bottom_outs);
}

TEST(vehicle_pothole_without_damping_bottoms_out) {
  Track track(3);
  track.addSmooth(1.0);
  track.addPothole(0.03, 0.10);
  track.addSmooth(2.0);
  VehicleParams p;
  Vehicle v(track, p, 3);
  for (int i = 0; i < 6000; ++i) v.step(0.001, 255, false, 45);
  CHECK(v.bottomOuts() > 0);
}

TEST(vehicle_gravel_vibrates_more_than_smooth) {
  auto rms = [](bool gravel) {
    Track track(4);
    if (gravel) track.addGravel(10, 0.003);
    else track.addSmooth(10);
    VehicleParams p;
    Vehicle v(track, p, 4);
    double sum = 0;
    int n = 0;
    for (int i = 0; i < 8000; ++i) {
      v.step(0.001, 170, false, 45);
      if (i > 2000) {
        sum += v.truth().heave_acc * v.truth().heave_acc;
        ++n;
      }
    }
    return std::sqrt(sum / n);
  };
  CHECK(rms(true) > 5 * rms(false));
}

TEST(vehicle_truth_pitch_matches_static_tilt) {
  Track track(5);
  track.addSmooth(10);
  VehicleParams p;
  p.accel_noise = 0;
  p.gyro_noise = 0;
  p.accel_bias = 0;
  p.mount_tilt_deg = 5;
  Vehicle v(track, p, 5);
  for (int i = 0; i < 1000; ++i) v.step(0.001, 0, true, 90);
  ImuSample s;
  v.readImu(s);
  // Accelerometer-derived angles equal the reported sensor-frame attitude.
  double pitch = std::atan2(-s.ax, std::sqrt(s.ay * s.ay + s.az * s.az));
  double roll = std::atan2(s.ay, s.az);
  CHECK_NEAR(pitch, v.truth().pitch, 0.002);
  CHECK_NEAR(roll, v.truth().roll, 0.002);
}
