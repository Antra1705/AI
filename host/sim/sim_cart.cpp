#include "sim_cart.h"

#include "config.h"

static const double BUMPER_M = 0.10;  // sensor sits 10 cm ahead of the body centre

SimCart::SimCart(const Track& t, const VehicleParams& params, unsigned seed)
    : track(t), car(track, params, seed), app(rig.hal), rng_(seed * 31u + 7u) {
  rig.imu.source = [this](uint64_t, ImuSample& s) { car.readImu(s); };
  rig.imu.present = true;
  rig.distance.source = [this](uint64_t) {
    if (ultrasonic_unplugged) return DistanceReading{DistanceStatus::NO_RESPONSE, 0.0f};
    double d = gapCm() + sonar_noise_cm * gauss_(rng_);
    if (d > cfg::DIST_MAX_CM) return DistanceReading{DistanceStatus::OUT_OF_RANGE, 0.0f};
    if (d < cfg::DIST_MIN_CM) return DistanceReading{DistanceStatus::TOO_CLOSE, 0.0f};
    return DistanceReading{DistanceStatus::OK, (float)d};
  };
}

double SimCart::gapCm() const {
  if (wall_x < 0) return 1e6;
  return (wall_x - (car.truth().x + BUMPER_M)) * 100.0;
}

void SimCart::runFor(double seconds) {
  uint64_t steps = (uint64_t)(seconds * 1000.0 + 0.5);
  for (uint64_t i = 0; i < steps; ++i) {
    rig.imu.present = !imu_unplugged;
    app.tick();
    car.step(0.001, rig.motors.pwm, rig.motors.braked, rig.servo.angle);
    rig.clock.advanceUs(1000);
    if (on_ms) on_ms();
  }
}

bool SimCart::runUntil(const std::function<bool()>& pred, double max_seconds) {
  double end = time() + max_seconds;
  while (time() < end) {
    runFor(0.001);
    if (pred()) return true;
  }
  return false;
}
