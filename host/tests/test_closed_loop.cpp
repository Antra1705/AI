// Closed-loop tests: the real App (neural classifier, graded damping curve,
// Type-2 fuzzy speed, EKF, maintenance) driving the simulated cart.
#include <algorithm>
#include <cmath>

#include "config.h"
#include "sim_cart.h"
#include "test.h"

static VehicleParams healthy() {
  VehicleParams p;
  p.wear = 1.0;
  return p;
}

TEST(closed_loop_smooth_road_target_and_fast) {
  Track t(11);
  t.addSmooth(40);
  SimCart cart(t, healthy(), 11);
  cart.begin();
  cart.command("start");
  cart.runFor(6.0);
  CHECK(cart.app.state() == RunState::RUNNING);
  CHECK_EQ(cart.app.surfaceState(), SURFACE_SMOOTH);
  CHECK(!cart.app.decision().uncertain);
  // smooth-road target from 'sim tune' (STIFF_SMOOTH), plus a little body-motion damping
  CHECK_NEAR(cart.rig.servo.angle, stiffnessToAngle(cfg::STIFF_SMOOTH), 12);
  CHECK(cart.rig.motors.pwm >= 180);  // near full speed on a clear, smooth road
}

TEST(closed_loop_gravel_goes_soft) {
  Track t(12);
  t.addSmooth(3);
  t.addGravel(12, 0.002);
  SimCart cart(t, healthy(), 12);
  cart.begin();
  cart.command("start");
  CHECK(cart.runUntil([&] { return cart.car.rearX() > 4.5; }, 30));
  cart.runFor(1.5);
  CHECK_EQ(cart.app.decision().top, SURFACE_GRAVEL);
  CHECK(cart.rig.servo.angle <= stiffnessToAngle(cfg::STIFF_GRAVEL) + 25);  // soft: isolates stone vibration
  CHECK(cart.rig.motors.pwm < 200);  // rough road: fuzzy controller slows down
}

TEST(closed_loop_pothole_stiffens_quickly) {
  Track t(13);
  t.addSmooth(4);
  t.addPothole(0.025, 0.10);
  t.addSmooth(6);
  SimCart cart(t, healthy(), 13);
  cart.begin();
  cart.command("start");
  CHECK(cart.runUntil([&] { return cart.car.frontX() >= 4.0; }, 30));
  double t_hit = cart.time();
  int max_servo = 0;
  double t_stiff = -1;
  cart.on_ms = [&] {
    max_servo = std::max(max_servo, cart.rig.servo.angle);
    if (t_stiff < 0 && cart.rig.servo.angle >= 140) t_stiff = cart.time();
  };
  cart.runFor(1.0);
  CHECK(max_servo >= 140);
  // Classifier path: up to 100 ms to see it + servo slew from the smooth setting.
  // (The one-sample reflex covers only very hard hits.)
  CHECK(t_stiff > 0 && t_stiff - t_hit < 0.25);
}

TEST(closed_loop_obstacle_smooth_stop_without_contact) {
  Track t(14);
  t.addSmooth(20);
  SimCart cart(t, healthy(), 14);
  cart.wall_x = 6.0;
  cart.begin();
  cart.command("start");
  bool saw_approach = false;
  double min_gap = 1e9;
  cart.on_ms = [&] {
    saw_approach |= cart.app.state() == RunState::APPROACH;
    min_gap = std::min(min_gap, cart.gapCm());
  };
  cart.runFor(25.0);
  CHECK(saw_approach);
  CHECK(min_gap > 0.0);
  CHECK(cart.rig.motors.braked);
}

TEST(closed_loop_maintenance_test_records_damping) {
  Track t(15);
  t.addSmooth(2);
  t.addBreaker(0.015, 0.10);  // 1.5 cm reference bump
  t.addSmooth(15);
  SimCart cart(t, healthy(), 15);
  cart.begin();
  cart.command("maint test");
  cart.runFor(0.2);
  CHECK(cart.app.state() == RunState::MAINT);
  CHECK_EQ(cart.rig.servo.angle, cfg::MAINT_TEST_ANGLE);
  CHECK(cart.runUntil([&] { return cart.app.state() == RunState::IDLE; }, 25));
  CHECK_EQ(cart.app.maintenance().state().tests, 1);
  float z = cart.app.maintenance().state().last_zeta;
  CHECK(z > 0.03f && z < 0.4f);
  CHECK(cart.rig.storage.blobs.count("maint") == 1);  // persisted
}

TEST(closed_loop_ekf_tracks_attitude) {
  Track t(16);
  t.addSmooth(30);
  VehicleParams p = healthy();
  p.mount_tilt_deg = 4.0;  // sensor mounted crooked
  SimCart cart(t, p, 16);
  cart.begin();
  cart.command("start");
  cart.runFor(8.0);
  CHECK_NEAR(cart.app.ekf().pitch(), cart.car.truth().pitch, 0.02);
  CHECK_NEAR(cart.app.ekf().roll(), cart.car.truth().roll, 0.02);
}

TEST(closed_loop_json_stream) {
  Track t(17);
  t.addSmooth(10);
  SimCart cart(t, healthy(), 17);
  cart.begin();
  cart.command("log json");
  cart.runFor(1.0);
  int n = cart.rig.console.countLinesStartingWith("J,{");
  CHECK(n >= 9 && n <= 11);
}
