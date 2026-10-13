// sim_cart.h - closed-loop harness: the real App driving the simulated cart.
//
// Every simulated millisecond: App::tick() (reads the simulated IMU and
// ultrasonic sensor, writes servo/motor commands), then the vehicle physics
// advance using those commands. Used by the simulator scenarios and by tests.
#pragma once
#include <functional>
#include <random>
#include <string>

#include "core/app.h"
#include "fake_hal.h"
#include "vehicle.h"

class SimCart {
 public:
  SimCart(const Track& track, const VehicleParams& params, unsigned seed);
  SimCart(const SimCart&) = delete;
  SimCart& operator=(const SimCart&) = delete;

  void begin() {
    rig.imu.present = !imu_unplugged;
    app.begin();
  }
  void command(const std::string& line) { rig.console.input.push_back(line); }
  // Advance by 'seconds' of simulated time; on_ms (if set) runs after every step.
  void runFor(double seconds);
  // Run until pred() is true or max_seconds pass. Returns whether pred() became true.
  bool runUntil(const std::function<bool()>& pred, double max_seconds);

  double time() const { return (double)rig.clock.t_us * 1e-6; }
  double gapCm() const;  // distance from the front bumper to the wall (cm); large if none

  Track track;  // live mode may lay a reference bump ahead of the cart
  FakeRig rig;
  Vehicle car;
  App app;

  double wall_x = -1.0;         // wall position along the track (m); < 0 = no wall
  bool imu_unplugged = false;
  bool ultrasonic_unplugged = false;
  double sonar_noise_cm = 0.3;
  std::function<void()> on_ms;  // per-millisecond hook for measurements

 private:
  std::mt19937 rng_;
  std::normal_distribution<double> gauss_{0.0, 1.0};
};
