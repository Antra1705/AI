// Integration tests: the real App driven by the fake hardware.
//
// These check the state machine, safety and command logic. They run the App in
// its simple modes (rule-based classifier, fixed angles, fixed speed steps) so
// the expected outputs are exact. The neural classifier, graded curve and fuzzy
// speed control are tested in closed loop on the simulated cart
// (test_closed_loop.cpp), where the sensor signals are realistic.
#include "config.h"
#include "core/app.h"
#include "fake_hal.h"
#include "test.h"

static const uint64_t MS = 1000;

static DistanceReading at(float cm) { return DistanceReading{DistanceStatus::OK, cm}; }

static void simpleModes(App& app) {
  app.setClassifierMl(false);
  app.setCurveGraded(false);
  app.setSpeedMode(SpeedMode::STEP);
}

TEST(app_boots_idle_with_motors_off) {
  FakeRig rig;
  App app(rig.hal);
  simpleModes(app);
  app.begin();
  CHECK_EQ(rig.servo.angle, cfg::SERVO_NEUTRAL);  // safe position at power-on
  rig.run(app, 2000 * MS);
  CHECK(app.state() == RunState::IDLE);
  CHECK(rig.motors.braked);
  // Suspension adapts even while parked, so it can be tested by pushing the cart by hand.
  CHECK_EQ(rig.servo.angle, cfg::SERVO_SMOOTH);
  CHECK(rig.console.contains("System ready"));
  CHECK_EQ(rig.platform.watchdog_timeout_ms, cfg::WATCHDOG_TIMEOUT_MS);
  CHECK(rig.platform.feeds > 1000);
}

TEST(app_start_drives_at_normal_speed_when_clear) {
  FakeRig rig;
  App app(rig.hal);
  simpleModes(app);
  app.begin();
  rig.console.input.push_back("start");
  rig.run(app, 500 * MS);
  CHECK(app.state() == RunState::RUNNING);
  CHECK_EQ(rig.motors.pwm, cfg::SPEED_NORMAL);
  rig.console.input.push_back("stop");
  rig.run(app, 10 * MS);
  CHECK(app.state() == RunState::IDLE);
  CHECK(rig.motors.braked);
}

// BR-1: missing IMU at power-on -> clear error and driving refused.
TEST(app_refuses_to_start_without_imu) {
  FakeRig rig;
  rig.imu.present = false;
  App app(rig.hal);
  simpleModes(app);
  app.begin();
  CHECK(rig.console.contains("MPU6050 not responding"));
  rig.console.input.push_back("start");
  rig.run(app, 500 * MS);
  CHECK(app.state() == RunState::IDLE);
  CHECK(rig.motors.braked);
  CHECK(rig.console.contains("Cannot start"));
  // Plugging it back in is noticed by the periodic retry.
  rig.imu.present = true;
  rig.run(app, 1500 * MS);
  CHECK(!app.imuFault());
  CHECK(rig.console.contains("IMU recovered"));
}

// BR-3: exactly 100 samples per second, even across the micros() wrap.
TEST(app_samples_imu_at_100hz_across_timer_wrap) {
  FakeRig rig;
  rig.clock.t_us = 0xFFFFFFFFull - 3000 * MS;  // wraps 3 s into the run
  App app(rig.hal);
  simpleModes(app);
  app.begin();
  rig.run(app, 1000 * MS);  // settle
  int before = rig.imu.reads;
  rig.run(app, 6000 * MS, 250);
  CHECK_EQ(rig.imu.reads - before, 600);
  CHECK_EQ(app.health().imu_overruns, 0u);
}

TEST(app_obstacle_slows_then_stops_then_resumes) {
  FakeRig rig;
  float wall = 100.0f;
  rig.distance.source = [&](uint64_t) { return at(wall); };
  App app(rig.hal);
  simpleModes(app);
  app.begin();
  rig.console.input.push_back("start");
  rig.run(app, 300 * MS);
  CHECK_EQ(rig.motors.pwm, cfg::SPEED_NORMAL);
  wall = 15.0f;
  rig.run(app, 100 * MS);
  CHECK_EQ(rig.motors.pwm, cfg::SPEED_SLOW);
  wall = 8.0f;
  uint64_t t_close = rig.clock.t_us;
  rig.run(app, 100 * MS);
  CHECK(app.state() == RunState::ESTOP);
  CHECK(rig.motors.braked);
  // Reaction time: first brake after the wall came within 10 cm.
  uint64_t t_brake = 0;
  for (auto& h : rig.motors.history)
    if (h.first >= t_close && h.second == 0) { t_brake = h.first; break; }
  CHECK(t_brake > 0 && t_brake - t_close <= (cfg::DIST_PERIOD_MS + 5) * MS);
  wall = 50.0f;
  rig.run(app, 300 * MS);
  CHECK(app.state() == RunState::APPROACH);  // driving again, obstacle still within 60 cm
  CHECK_EQ(rig.motors.pwm, cfg::SPEED_NORMAL);
  CHECK_EQ(app.health().estops, 1u);
}

// BR-8: unplugging the IMU while driving stops the cart.
TEST(app_imu_dropout_faults_and_stops) {
  FakeRig rig;
  App app(rig.hal);
  simpleModes(app);
  app.begin();
  rig.console.input.push_back("start");
  rig.run(app, 2000 * MS);
  CHECK_EQ(rig.motors.pwm, cfg::SPEED_NORMAL);
  rig.imu.present = false;
  uint64_t t_fail = rig.clock.t_us;
  rig.run(app, 200 * MS);
  CHECK(app.state() == RunState::FAULT);
  CHECK(rig.motors.braked);
  CHECK_EQ(rig.servo.angle, cfg::SERVO_NEUTRAL);
  CHECK(rig.motors.history.back().first - t_fail <= 60 * MS);  // 5 failed reads = 50 ms
  // Fault stays latched after the IMU comes back, until the operator restarts.
  rig.imu.present = true;
  rig.run(app, 1500 * MS);
  CHECK(app.state() == RunState::FAULT);
  rig.console.input.push_back("start");
  rig.run(app, 100 * MS);
  CHECK(app.state() == RunState::RUNNING);
}

// BR-8 / I-6: a silent distance sensor stops the cart; open space does not.
TEST(app_distance_sensor_silence_faults) {
  FakeRig rig;
  bool unplugged = false;
  rig.distance.source = [&](uint64_t) {
    return unplugged ? DistanceReading{DistanceStatus::NO_RESPONSE, 0}
                     : DistanceReading{DistanceStatus::OUT_OF_RANGE, 0};
  };
  App app(rig.hal);
  simpleModes(app);
  app.begin();
  rig.console.input.push_back("start");
  rig.run(app, 1000 * MS);
  CHECK(app.state() == RunState::RUNNING);  // open space is fine
  CHECK_EQ(rig.motors.pwm, cfg::SPEED_NORMAL);
  unplugged = true;
  uint64_t t_fail = rig.clock.t_us;
  rig.run(app, 300 * MS);
  CHECK(app.state() == RunState::FAULT);
  CHECK(rig.motors.braked);
  CHECK(rig.motors.history.back().first - t_fail <= 200 * MS);  // BRD T6: within 200 ms
}

// BR-6: a sharp impact stiffens the suspension within one IMU sample.
TEST(app_impact_fast_path) {
  FakeRig rig;
  uint64_t impact_at = 3000 * MS;
  rig.imu.source = [&](uint64_t t, ImuSample& s) {
    if (t >= impact_at && t < impact_at + 30 * MS) s.az = 9.81f + 45.0f;  // hard hit: > IMPACT_JERK
  };
  App app(rig.hal);
  simpleModes(app);
  app.begin();
  rig.run(app, 2900 * MS);
  CHECK_EQ(rig.servo.angle, cfg::SERVO_SMOOTH);  // smooth road so far
  rig.run(app, 200 * MS);
  CHECK_EQ(rig.servo.angle, cfg::SERVO_IMPACT);
  uint64_t t_cmd = 0;
  for (auto& h : rig.servo.history)
    if (h.first >= impact_at && h.second == cfg::SERVO_IMPACT) { t_cmd = h.first; break; }
  CHECK(t_cmd > 0 && t_cmd - impact_at <= 10 * MS);
  CHECK_EQ(app.health().impacts, 1u);
  // Window clears 1 s after the impact; then the servo relaxes back to smooth.
  rig.run(app, 2000 * MS);
  CHECK_EQ(rig.servo.angle, cfg::SERVO_SMOOTH);
}

TEST(app_manual_servo_override) {
  FakeRig rig;
  App app(rig.hal);
  simpleModes(app);
  app.begin();
  rig.console.input.push_back("servo 120");
  rig.run(app, 2000 * MS);
  CHECK_EQ(rig.servo.angle, 120);
  rig.console.input.push_back("servo auto");
  rig.run(app, 1000 * MS);
  CHECK_EQ(rig.servo.angle, cfg::SERVO_SMOOTH);
}

TEST(app_manual_motor_pulse) {
  FakeRig rig;
  App app(rig.hal);
  simpleModes(app);
  app.begin();
  rig.run(app, 100 * MS);
  rig.console.input.push_back("motor 150 500");
  rig.run(app, 200 * MS);
  CHECK_EQ(rig.motors.pwm, 150);
  rig.run(app, 400 * MS);
  CHECK(rig.motors.braked);
  CHECK(app.state() == RunState::IDLE);
}

TEST(app_log_modes) {
  FakeRig rig;
  App app(rig.hal);
  simpleModes(app);
  app.begin();
  rig.console.input.push_back("log raw");
  rig.run(app, 1000 * MS);
  int raw = rig.console.countLinesStartingWith("R,");
  CHECK(raw >= 99 && raw <= 101);
  CHECK_EQ(rig.console.countLinesStartingWith("["), 0);  // status lines muted while logging
  rig.console.input.push_back("log features");
  rig.run(app, 1000 * MS);
  int feat = rig.console.countLinesStartingWith("F,");
  CHECK(feat >= 9 && feat <= 11);
}

TEST(app_health_command) {
  FakeRig rig;
  App app(rig.hal);
  simpleModes(app);
  app.begin();
  rig.console.input.push_back("health");
  rig.run(app, 10 * MS);
  CHECK(rig.console.contains("--- health ---"));
  CHECK(rig.console.contains("reset reason      SIMULATED"));
}
