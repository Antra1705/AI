#include "app.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>

#include "../config.h"
#include "features.h"

namespace {

// True once 'now' has reached 'deadline'; correct across timer wrap-around.
bool reached(uint32_t now, uint32_t deadline) { return (int32_t)(now - deadline) >= 0; }

// Advances a periodic deadline. If the loop fell a whole period behind, it
// restarts from 'now' instead of firing repeatedly to catch up.
bool advance(uint32_t& deadline, uint32_t period, uint32_t now) {
  deadline += period;
  if (reached(now, deadline)) {
    deadline = now + period;
    return true;  // overrun
  }
  return false;
}

const char* MAINT_KEY = "maint";

}  // namespace

App::App(const Hal& hal) : hal_(hal), model_(deployedModel()), thresholds_(defaultThresholds()) {
  for (int i = 0; i < ML_FEATURE_COUNT; ++i) ml_features_[i] = 0.0f;
  use_ml_ = cfg::DEFAULT_CLASSIFIER_ML && model_.members > 0;
  graded_ = cfg::DEFAULT_CURVE_GRADED;
  speed_mode_ = (SpeedMode)cfg::DEFAULT_SPEED_MODE;
}

void App::say(const char* fmt, ...) {
  char buf[200];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof buf, fmt, args);
  va_end(args);
  hal_.console.print(buf);
  hal_.console.print("\n");
}

void App::setCurveGraded(bool graded) {
  graded_ = graded;
  int a = servo_angle_ < 0 ? cfg::SERVO_NEUTRAL : servo_angle_;
  shaper_.reset((float)a);
  servo_policy_.reset(a);
}

Classification App::classification() const {
  Classification c;
  c.surface = (!decision_.valid || decision_.uncertain) ? SURFACE_UNKNOWN : (Surface)decision_.top;
  c.confidence = decision_.confidence;
  return c;
}

void App::begin() {
  const uint32_t now_us = hal_.clock.micros();
  const uint32_t now_ms = hal_.clock.millis();

  hal_.console.print("\n");
  say("=== Adaptive Suspension v" FIRMWARE_VERSION " ===");
  say("Reset reason: %s", hal_.platform.resetReason());
  say("Classifier: %s", model_.members > 0 ? model_.info : "rule-based (no trained model)");

  // Safe outputs first (BR-1, BR-2): motors braked, suspension neutral.
  motor_pwm_ = -1;
  last_motor_ms_ = now_ms;
  applyMotors(now_ms);
  shaper_.reset((float)cfg::SERVO_NEUTRAL);
  servo_policy_.reset(cfg::SERVO_NEUTRAL);
  writeServo(cfg::SERVO_NEUTRAL);

  hal_.platform.watchdogBegin(cfg::WATCHDOG_TIMEOUT_MS);

  MaintState saved;
  if (hal_.storage.load(MAINT_KEY, &saved, sizeof saved) && maint_.load(saved)) {
    say("Maintenance record loaded: %s, %u tests", maintStatusName(maint_.status()), (unsigned)saved.tests);
  }

  if (hal_.imu.begin()) {
    say("OK  MPU6050 initialized");
  } else {
    imu_fault_ = true;
    ++health_.imu_faults;
    say("ERROR  MPU6050 not responding - check wiring. Driving is disabled.");
  }
  last_imu_retry_ms_ = now_ms;

  next_imu_us_ = now_us;
  next_dist_ms_ = now_ms;
  next_control_ms_ = now_ms + cfg::CONTROL_PERIOD_MS;
  next_status_ms_ = now_ms + cfg::STATUS_PERIOD_MS;

  if (imu_fault_) {
    say("System NOT ready. Fix the fault, then type 'start'. Type 'help' for commands.");
  } else {
    say("System ready. State: IDLE. Type 'start' to drive, 'help' for commands.");
  }
}

void App::tick() {
  const uint32_t t0 = hal_.clock.micros();
  const uint32_t now_ms = hal_.clock.millis();
  hal_.platform.watchdogFeed();

  char line[64];
  if (hal_.console.readLine(line, sizeof line)) handleCommand(parseCommand(line), now_ms);

  if (reached(t0, next_imu_us_)) {
    if (advance(next_imu_us_, cfg::IMU_PERIOD_US, t0)) ++health_.imu_overruns;
    imuStep(t0, now_ms);
  }

  distanceStep(now_ms);

  if (manual_motor_ && reached(now_ms, manual_until_ms_)) {
    manual_motor_ = false;
    say("Manual motor pulse finished");
    applyMotors(now_ms);
  }

  if (reached(now_ms, next_control_ms_)) {
    advance(next_control_ms_, cfg::CONTROL_PERIOD_MS, now_ms);
    controlStep(now_ms);
  }

  if (reached(now_ms, next_status_ms_)) {
    advance(next_status_ms_, cfg::STATUS_PERIOD_MS, now_ms);
    if (log_mode_ == LogMode::OFF) printStatus();
  }

  const uint32_t took = hal_.clock.micros() - t0;
  if (took > health_.max_tick_us) health_.max_tick_us = took;
}

// ---------------------------------------------------------------- IMU (100 Hz)

void App::imuStep(uint32_t now_us, uint32_t now_ms) {
  if (imu_fault_) {
    if (now_ms - last_imu_retry_ms_ >= cfg::IMU_RETRY_MS) {
      last_imu_retry_ms_ = now_ms;
      if (hal_.imu.begin()) {
        imu_fault_ = false;
        imu_fail_count_ = 0;
        buffer_.clear();
        ekf_.reset();
        have_last_imu_ = false;
        say("IMU recovered");
        updateSafety(now_ms);
      }
    }
    return;
  }

  ImuSample s;
  if (!hal_.imu.read(s)) {
    ++health_.imu_read_errors;
    if (++imu_fail_count_ >= cfg::IMU_FAIL_LIMIT) {
      imu_fault_ = true;
      ++health_.imu_faults;
      last_imu_retry_ms_ = now_ms;
      buffer_.clear();
      decision_.valid = false;
      say("FAULT  IMU stopped responding");
      if (manual_servo_ < 0) {
        shaper_.reset((float)cfg::SERVO_NEUTRAL);
        servo_policy_.reset(cfg::SERVO_NEUTRAL);
        writeServo(cfg::SERVO_NEUTRAL);
      }
      if (maint_phase_ != MaintPhase::NONE) finishMaintTest(false, "IMU fault");
      updateSafety(now_ms);
    }
    return;
  }
  imu_fail_count_ = 0;
  s.t_us = now_us;

  float dt = have_last_imu_ ? (float)(now_us - last_imu_us_) * 1e-6f : (float)cfg::IMU_PERIOD_US * 1e-6f;
  if (dt <= 0.0f || dt > 0.1f) dt = (float)cfg::IMU_PERIOD_US * 1e-6f;
  last_imu_us_ = now_us;
  have_last_imu_ = true;

  // EKF: trust in the accelerometer as a gravity sensor follows the road class.
  const float gyro[3] = {s.gx, s.gy, s.gz};
  const float accel[3] = {s.ax, s.ay, s.az};
  ekf_.step(gyro, accel, dt, ekfNoiseScale(decision_.p, decision_.valid));

  // Impact reflex: one very sharp jump stiffens the suspension now, without
  // waiting for the next classification (BR-6).
  const bool auto_servo = manual_servo_ < 0 && supervisor_.state() != RunState::MAINT;
  if (buffer_.size() > 0 && auto_servo) {
    float jerk = instantJerk(buffer_.newest(), s);
    if (jerk > cfg::IMPACT_JERK) {
      bool was_holding = graded_ ? shaper_.holding() : servo_policy_.holding();
      if (graded_) {
        shaper_.impact(now_ms);
        writeServo(cfg::SERVO_IMPACT);
      } else {
        servo_policy_.impact(now_ms);
        writeServo(servo_policy_.angle());
      }
      uint32_t latency = hal_.clock.micros() - s.t_us;
      health_.last_impact_latency_us = latency;
      if (latency > health_.max_impact_latency_us) health_.max_impact_latency_us = latency;
      if (!was_holding) {
        ++health_.impacts;
        say("Impact: jerk=%.0f m/s^3 -> servo %d", (double)jerk, cfg::SERVO_IMPACT);
      }
    }
  }

  buffer_.push(s);

  // Maintenance reference test: wait for the bump, then record the response.
  if (maint_phase_ == MaintPhase::WAIT_BUMP && fabsf(ekf_.verticalAccel()) > cfg::MAINT_TRIGGER) {
    maint_phase_ = MaintPhase::RECORD;
    analyzer_.start();
    say("Maintenance: reference bump detected, recording");
  }
  if (maint_phase_ == MaintPhase::RECORD && analyzer_.add(ekf_.verticalAccel())) {
    DampingEstimate e = analyzer_.analyze((float)cfg::IMU_PERIOD_US * 1e-6f);
    if (e.ok) {
      MaintStatus st = maint_.addTest(e.zeta);
      hal_.storage.save(MAINT_KEY, &maint_.state(), sizeof(MaintState));
      say("Maintenance: zeta=%.4f (bounce %.1f Hz, %d peaks) -> %s", (double)e.zeta, (double)e.freq_hz,
          e.peaks_used, maintStatusName(st));
      finishMaintTest(true, "");
    } else {
      finishMaintTest(false, "response too weak or noisy to analyse");
    }
  }

  // Graded mode: the shaper moves the servo smoothly toward the curve's target.
  if (graded_ && auto_servo && !imu_fault_) writeServo(shaper_.update(dt, now_ms));

  if (log_mode_ == LogMode::RAW) {
    char buf[96];
    formatRaw(buf, sizeof buf, s);
    hal_.console.print(buf);
  }
}

// ---------------------------------------------------------------- distance (25 Hz)

void App::distanceStep(uint32_t now_ms) {
  DistanceReading r;
  if (dist_in_flight_ && hal_.distance.poll(r)) {
    dist_in_flight_ = false;
    handleDistance(r, now_ms);
  }
  if (reached(now_ms, next_dist_ms_)) {
    advance(next_dist_ms_, cfg::DIST_PERIOD_MS, now_ms);
    if (dist_in_flight_) {  // previous measurement never finished
      DistanceReading none = {DistanceStatus::NO_RESPONSE, 0.0f};
      handleDistance(none, now_ms);
    }
    hal_.distance.trigger();
    dist_in_flight_ = true;
  }
}

void App::handleDistance(const DistanceReading& r, uint32_t now_ms) {
  if (r.status == DistanceStatus::NO_RESPONSE) ++health_.dist_no_response;
  const float dt = have_distance_ ? (float)(now_ms - last_dist_ms_) * 1e-3f : 0.0f;
  last_dist_ms_ = now_ms;
  last_distance_ = r;
  have_distance_ = true;
  obstacle_.update(r);

  switch (r.status) {
    case DistanceStatus::OK: range_.update(r.cm, dt); break;
    case DistanceStatus::TOO_CLOSE: range_.update(1.0f, dt); break;
    case DistanceStatus::OUT_OF_RANGE: range_.update(-1.0f, dt); break;
    case DistanceStatus::NO_RESPONSE: break;  // keep the last estimate
  }
  if (!range_.valid()) {
    near_ = false;
  } else if (range_.distance() < cfg::APPROACH_CM) {
    near_ = true;
  } else if (range_.distance() > cfg::APPROACH_EXIT_CM) {
    near_ = false;
  }
  updateSafety(now_ms);  // act on an obstacle immediately, not at the next control cycle
}

// ---------------------------------------------------------------- control (10 Hz)

void App::classify() {
  const bool ready = !imu_fault_ && buffer_.full();
  decision_.valid = ready;
  if (!ready) {
    for (int c = 0; c < 4; ++c) decision_.p[c] = 0.25f;
    decision_.top = SURFACE_SMOOTH;
    decision_.confidence = 0.0f;
    decision_.mutual_info = 0.0f;
    decision_.novelty = 0.0f;
    decision_.uncertain = true;
    return;
  }
  features_ = extractFeatures(buffer_);
  extractMlFeatures(buffer_, ml_features_);

  if (use_ml_ && model_.members > 0) {
    MlPrediction p = mlPredict(model_, ml_features_);
    for (int c = 0; c < 4; ++c) decision_.p[c] = p.p[c];
    decision_.top = p.top;
    decision_.confidence = p.confidence;
    decision_.mutual_info = p.mutual_info;
    decision_.novelty = p.novelty;
    decision_.uncertain = p.uncertain;
  } else {
    // Rule-based fallback: the winning class gets its confidence, the rest is spread.
    Classification c = ::classify(features_, thresholds_);
    if (c.surface == SURFACE_UNKNOWN) {
      for (int k = 0; k < 4; ++k) decision_.p[k] = 0.25f;
      decision_.top = SURFACE_SMOOTH;
      decision_.confidence = 0.25f;
    } else {
      for (int k = 0; k < 4; ++k) decision_.p[k] = (k == c.surface) ? c.confidence : (1.0f - c.confidence) / 3.0f;
      decision_.top = c.surface;
      decision_.confidence = c.confidence;
    }
    decision_.mutual_info = 0.0f;
    decision_.novelty = 0.0f;
    decision_.uncertain = decision_.confidence <= cfg::CONFIDENCE_GATE;
  }
}

void App::controlStep(uint32_t now_ms) {
  classify();
  tracker_.update(decision_.top, decision_.uncertain || !decision_.valid);

  CurveInput in;
  in.valid = decision_.valid;
  for (int c = 0; c < 4; ++c) in.p[c] = decision_.p[c];
  in.confidence = decision_.confidence;
  in.mutual_info = decision_.mutual_info;
  in.novelty = decision_.novelty;
  in.vz = ekf_.vz();
  CurveThresholds th = {use_ml_ ? model_.mi_threshold : 0.0f, use_ml_ ? model_.novelty_threshold : 0.0f};
  curve_ = responseCurve(in, th);

  if (manual_servo_ >= 0) {
    // operator override: leave the servo alone
  } else if (supervisor_.state() == RunState::MAINT) {
    shaper_.reset((float)cfg::MAINT_TEST_ANGLE);
    writeServo(cfg::MAINT_TEST_ANGLE);
  } else if (imu_fault_) {
    shaper_.reset((float)cfg::SERVO_NEUTRAL);
    servo_policy_.reset(cfg::SERVO_NEUTRAL);
    writeServo(cfg::SERVO_NEUTRAL);
  } else if (graded_) {
    shaper_.setTarget(curve_.angle);  // the 100 Hz IMU step moves the servo toward it
  } else {
    servo_policy_.update(servoAngleFor(classification()), now_ms);
    writeServo(servo_policy_.angle());
  }

  if (maint_phase_ != MaintPhase::NONE && reached(now_ms, maint_deadline_ms_)) {
    finishMaintTest(false, "no reference bump found before the timeout");
  }

  updateSafety(now_ms);

  if (log_mode_ == LogMode::FEATURES && decision_.valid) {
    char buf[128];
    formatFeatures(buf, sizeof buf, now_ms, features_, classification(), servo_angle_);
    hal_.console.print(buf);
  } else if (log_mode_ == LogMode::JSON) {
    char buf[1100];
    formatJson(buf, sizeof buf, snapshot());
    hal_.console.print(buf);
  }
}

// ---------------------------------------------------------------- safety + outputs

void App::updateSafety(uint32_t now_ms) {
  const RunState before = supervisor_.state();
  supervisor_.update(imu_fault_, obstacle_.sensorFault(), obstacle_.emergency(), near_);
  const RunState after = supervisor_.state();
  if (after != before) {
    if (after == RunState::ESTOP) ++health_.estops;
    if (after == RunState::FAULT) ++health_.faults;
    if (after == RunState::FAULT) {
      say("State %s -> %s (%s)", runStateName(before), runStateName(after), supervisor_.faultReason());
    } else {
      say("State %s -> %s", runStateName(before), runStateName(after));
    }
    if (before == RunState::MAINT && maint_phase_ != MaintPhase::NONE) finishMaintTest(false, "interrupted");
  }
  if (manual_motor_ && obstacle_.sensorFault()) {
    manual_motor_ = false;
    say("Manual motor pulse cancelled: distance sensor fault");
  }
  applyMotors(now_ms);
}

void App::applyMotors(uint32_t now_ms) {
  float dt = (float)(now_ms - last_motor_ms_) * 1e-3f;
  last_motor_ms_ = now_ms;
  if (dt < 0.0f || dt > 1.0f) dt = 0.0f;

  const RunState st = supervisor_.state();
  const bool driving = st == RunState::RUNNING || st == RunState::APPROACH;
  int target = 0;
  bool ramp = speed_mode_ != SpeedMode::STEP;
  fuzzy_u_ = 0.0f;

  if (driving) {
    if (speed_mode_ == SpeedMode::STEP) {
      target = obstacle_.speedLimit();  // SRA: 200 / 127 under 20 cm / 0 under 10 cm
      fuzzy_u_ = (float)target / cfg::SPEED_NORMAL;
    } else {
      float d = range_.valid() ? range_.distance() : cfg::DIST_MAX_CM + 1.0f;
      float rough = decision_.valid ? 1.0f - decision_.p[SURFACE_SMOOTH] : 0.5f;
      FuzzyResult f = fuzzySpeed(d, range_.closingSpeed(), rough, speed_mode_);
      fuzzy_u_ = f.u;
      target = f.u < cfg::FUZZY_STOP_U
                   ? 0
                   : (int)((float)cfg::MOTOR_MIN_PWM + f.u * (float)(cfg::SPEED_NORMAL - cfg::MOTOR_MIN_PWM) + 0.5f);
      // Approaching an obstacle: only slow down, never re-accelerate on sensor
      // noise, unless the obstacle really moved away.
      if (st == RunState::APPROACH) {
        if (!approach_active_ || d > approach_ref_cm_ + cfg::APPROACH_REARM_CM) {
          approach_active_ = true;
          approach_cap_ = target;
          approach_ref_cm_ = d;
        } else {
          if (target < approach_cap_) approach_cap_ = target;
          if (d < approach_ref_cm_) approach_ref_cm_ = d;
          target = approach_cap_;
        }
      } else {
        approach_active_ = false;
      }
      int cap = obstacle_.safetyCap();
      if (target > cap) target = cap;
    }
  } else if (st == RunState::MAINT) {
    int cap = obstacle_.safetyCap();
    target = cfg::MAINT_TEST_PWM < cap ? cfg::MAINT_TEST_PWM : cap;
    ramp = false;
  } else if (manual_motor_ && st == RunState::IDLE) {
    int limit = obstacle_.speedLimit();
    target = manual_pwm_ < limit ? manual_pwm_ : limit;
    ramp = false;
  }

  // Ramp speed changes (smooth deceleration); a stop for safety is immediate.
  if (target == 0 || !ramp) {
    pwm_ramped_ = (float)target;
  } else {
    float diff = (float)target - pwm_ramped_;
    float up = cfg::PWM_RAMP_UP * dt, down = cfg::PWM_RAMP_DOWN * dt;
    if (diff > up) diff = up;
    if (diff < -down) diff = -down;
    pwm_ramped_ += diff;
    if (pwm_ramped_ < (float)cfg::MOTOR_MIN_PWM) pwm_ramped_ = (float)cfg::MOTOR_MIN_PWM;  // start above stall
  }
  int pwm = (int)(pwm_ramped_ + 0.5f);
  if (pwm == motor_pwm_) return;
  if (pwm > 0) {
    hal_.motors.drive(pwm, true);
  } else {
    hal_.motors.brake();
  }
  motor_pwm_ = pwm;
}

void App::writeServo(int angle) {
  if (angle == servo_angle_) return;
  hal_.servo.writeAngle(angle);
  servo_angle_ = angle;
}

// ---------------------------------------------------------------- maintenance

void App::startMaintTest(uint32_t now_ms) {
  if (!supervisor_.startMaint(imu_fault_, obstacle_.sensorFault())) {
    say("Cannot start maintenance test: %s", supervisor_.faultReason());
    return;
  }
  maint_phase_ = MaintPhase::WAIT_BUMP;
  maint_deadline_ms_ = now_ms + cfg::MAINT_TIMEOUT_MS;
  shaper_.reset((float)cfg::MAINT_TEST_ANGLE);
  writeServo(cfg::MAINT_TEST_ANGLE);
  say("Maintenance test: driving slowly with soft suspension; waiting for the reference bump");
  updateSafety(now_ms);
}

void App::finishMaintTest(bool ok, const char* why) {
  maint_phase_ = MaintPhase::NONE;
  if (!ok) say("Maintenance test aborted: %s", why);
  if (supervisor_.state() == RunState::MAINT) supervisor_.stop();
  applyMotors(hal_.clock.millis());
}

// ---------------------------------------------------------------- commands

void App::handleCommand(const Command& c, uint32_t now_ms) {
  switch (c.type) {
    case CommandType::NONE:
      break;
    case CommandType::HELP:
      printHelp();
      break;
    case CommandType::START:
      manual_motor_ = false;
      if (maint_phase_ != MaintPhase::NONE) finishMaintTest(false, "start pressed");
      if (supervisor_.start(imu_fault_, obstacle_.sensorFault())) {
        say("Starting");
        updateSafety(now_ms);
      } else {
        say("Cannot start: %s", supervisor_.faultReason());
      }
      break;
    case CommandType::STOP:
      manual_motor_ = false;
      if (maint_phase_ != MaintPhase::NONE) finishMaintTest(false, "stop pressed");
      supervisor_.stop();
      say("Stopped (IDLE)");
      updateSafety(now_ms);
      break;
    case CommandType::STATUS:
      printStatus();
      break;
    case CommandType::HEALTH: {
      char buf[512];
      formatHealth(buf, sizeof buf, health_, now_ms, hal_.platform.resetReason());
      hal_.console.print(buf);
      break;
    }
    case CommandType::LOG_RAW:
      log_mode_ = LogMode::RAW;
      say(RAW_CSV_HEADER);
      break;
    case CommandType::LOG_FEATURES:
      log_mode_ = LogMode::FEATURES;
      say(FEATURE_CSV_HEADER);
      break;
    case CommandType::LOG_JSON:
      log_mode_ = LogMode::JSON;
      break;
    case CommandType::LOG_OFF:
      log_mode_ = LogMode::OFF;
      say("Logging off");
      break;
    case CommandType::SERVO_SET:
      manual_servo_ = c.a;
      writeServo(c.a);
      say("Servo manual: %d deg ('servo auto' to resume)", c.a);
      break;
    case CommandType::SERVO_AUTO:
      manual_servo_ = -1;
      shaper_.reset((float)(servo_angle_ < 0 ? cfg::SERVO_NEUTRAL : servo_angle_));
      servo_policy_.reset(cfg::SERVO_NEUTRAL);
      say("Servo automatic");
      break;
    case CommandType::MOTOR_PULSE:
      if (supervisor_.state() != RunState::IDLE) {
        say("'motor' only works in IDLE - type 'stop' first");
        break;
      }
      if (obstacle_.sensorFault()) {
        say("Cannot run motor: distance sensor not responding");
        break;
      }
      manual_motor_ = true;
      manual_pwm_ = c.a;
      manual_until_ms_ = now_ms + (uint32_t)c.b;
      say("Motor pulse: pwm %d for %d ms (obstacle limits still apply)", c.a, c.b);
      applyMotors(now_ms);
      break;
    case CommandType::MAINT_STATUS: {
      char buf[600];
      formatMaint(buf, sizeof buf, maint_.state());
      hal_.console.print(buf);
      break;
    }
    case CommandType::MAINT_TEST:
      startMaintTest(now_ms);
      break;
    case CommandType::MAINT_RESET:
      maint_.reset();
      hal_.storage.save(MAINT_KEY, &maint_.state(), sizeof(MaintState));
      say("Maintenance baseline cleared: the next %d tests form a new baseline", cfg::MAINT_BASELINE_TESTS);
      break;
    case CommandType::MODE_ML:
      if (model_.members == 0) {
        say("No trained model in this build - staying on rules");
      } else {
        use_ml_ = true;
        say("Classifier: neural ensemble");
      }
      break;
    case CommandType::MODE_RULES:
      use_ml_ = false;
      say("Classifier: rule-based fallback");
      break;
    case CommandType::CURVE_GRADED:
      setCurveGraded(true);
      say("Damping: graded confidence curve");
      break;
    case CommandType::CURVE_DISCRETE:
      setCurveGraded(false);
      say("Damping: discrete angles (SRA)");
      break;
    case CommandType::SPEED_IT2:
    case CommandType::SPEED_T1:
    case CommandType::SPEED_STEP:
      speed_mode_ = c.type == CommandType::SPEED_IT2  ? SpeedMode::FUZZY_IT2
                    : c.type == CommandType::SPEED_T1 ? SpeedMode::FUZZY_T1
                                                      : SpeedMode::STEP;
      say("Obstacle speed control: %s", speedModeName(speed_mode_));
      break;
    case CommandType::INVALID:
      say("Unknown command. Type 'help'.");
      break;
  }
}

// ---------------------------------------------------------------- output

Snapshot App::snapshot() const {
  Snapshot s;
  s.now_ms = hal_.clock.millis();
  s.state = supervisor_.state();
  s.fault_reason = supervisor_.faultReason();
  s.imu_fault = imu_fault_;
  s.ready = decision_.valid;
  s.classifier = use_ml_ ? "ml" : "rules";
  for (int c = 0; c < 4; ++c) s.p[c] = decision_.p[c];
  s.top = decision_.top;
  s.surface = tracker_.state();
  s.confidence = decision_.confidence;
  s.mutual_info = decision_.mutual_info;
  s.novelty = decision_.novelty;
  s.uncertain = decision_.uncertain;
  s.f = features_;
  s.curve_mode = graded_ ? "graded" : "discrete";
  s.stiffness = curve_.s;
  s.gate = curve_.gate;
  s.c_eff = curve_.c_eff;
  s.s_exp = curve_.s_exp;
  s.s_risk = curve_.s_risk;
  s.s_sky = curve_.s_sky;
  s.servo = servo_angle_;
  s.servo_manual = manual_servo_ >= 0;
  s.speed_mode = speedModeName(speed_mode_);
  s.motor_pwm = motor_pwm_ < 0 ? 0 : motor_pwm_;
  s.fuzzy_u = fuzzy_u_;
  s.dist = last_distance_;
  s.have_distance = have_distance_;
  s.range_cm = range_.valid() ? range_.distance() : -1.0f;
  s.closing_cms = range_.closingSpeed();
  s.roll = ekf_.roll();
  s.pitch = ekf_.pitch();
  s.vz = ekf_.vz();
  s.a_vert = ekf_.verticalAccel();
  s.maint = maint_.status();
  s.maint_zeta = maint_.state().last_zeta;
  s.maint_tests = maint_.state().tests;
  s.estops = health_.estops;
  s.faults = health_.faults;
  s.impacts = health_.impacts;
  return s;
}

void App::printStatus() {
  char buf[320];
  formatStatus(buf, sizeof buf, snapshot());
  hal_.console.print(buf);
}

void App::printHelp() {
  hal_.console.print(
      "Commands:\n"
      "  start / stop            drive / stop (the cart never starts on its own)\n"
      "  status | health         one status line | fault counters and timing\n"
      "  log raw|features|json   CSV streams for calibration/training, JSON for the dashboard\n"
      "  log off                 back to status lines\n"
      "  servo <0-180> | auto    manual servo angle / automatic again\n"
      "  motor <pwm> [ms]        short motor test in IDLE (default 1000 ms)\n"
      "  maint | maint test      maintenance status | run the reference-bump test\n"
      "  maint reset             forget the baseline (after replacing the shocks)\n"
      "  mode ml|rules           neural ensemble / rule-based classifier\n"
      "  curve graded|discrete   confidence curve / fixed angles per class\n"
      "  speed it2|t1|step       Type-2 fuzzy / Type-1 fuzzy / fixed steps near obstacles\n");
}
