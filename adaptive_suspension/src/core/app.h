// app.h - the main control loop. Hardware-independent: it only uses the Hal
// interfaces, so the exact same code runs in the simulator and on the ESP32.
//
// tick() must be called as often as possible (from Arduino loop()). It never
// blocks; each job runs when its own timer is due:
//   every 10 ms   read IMU -> buffer, EKF step, impact reflex, servo shaping
//   every 40 ms   distance -> range tracker, emergency stop, APPROACH state
//   every 100 ms  features -> classifier (+ uncertainty) -> response curve,
//                 fuzzy speed, state machine, maintenance test
//   every 500 ms  status line (or JSON / CSV streams when logging)
#pragma once
#include <stdint.h>

#include "classifier.h"
#include "command.h"
#include "ekf.h"
#include "fuzzy_speed.h"
#include "hal.h"
#include "health.h"
#include "imu_buffer.h"
#include "maintenance.h"
#include "ml_features.h"
#include "ml_model.h"
#include "obstacle_monitor.h"
#include "response_curve.h"
#include "servo_policy.h"
#include "supervisor.h"
#include "telemetry.h"
#include "types.h"

// The classifier's answer in one form, whichever classifier produced it.
struct Decision {
  bool valid;
  float p[4];
  int top;
  float confidence, mutual_info, novelty;
  bool uncertain;
};

class App {
 public:
  explicit App(const Hal& hal);

  void begin();
  void tick();

  // Read-only views, used by tests, the simulator and the dashboard.
  RunState state() const { return supervisor_.state(); }
  const Decision& decision() const { return decision_; }
  Classification classification() const;  // top class + confidence (UNKNOWN when uncertain)
  const Features& features() const { return features_; }
  bool featuresValid() const { return decision_.valid; }
  const CurveOutput& curve() const { return curve_; }
  const Ekf& ekf() const { return ekf_; }
  const Health& health() const { return health_; }
  const MaintenanceMonitor& maintenance() const { return maint_; }
  int servoAngle() const { return servo_angle_; }
  int motorPwm() const { return motor_pwm_; }
  bool imuFault() const { return imu_fault_; }
  int surfaceState() const { return tracker_.state(); }
  float fuzzyU() const { return fuzzy_u_; }
  Snapshot snapshot() const;

  // Runtime modes (also settable by commands).
  void setClassifierMl(bool ml) { use_ml_ = ml && model_.members > 0; }
  void setCurveGraded(bool graded);
  void setSpeedMode(SpeedMode m) { speed_mode_ = m; }
  bool classifierMl() const { return use_ml_; }
  bool curveGraded() const { return graded_; }
  SpeedMode speedMode() const { return speed_mode_; }

 private:
  enum class LogMode { OFF, RAW, FEATURES, JSON };
  enum class MaintPhase { NONE, WAIT_BUMP, RECORD };

  void imuStep(uint32_t now_us, uint32_t now_ms);
  void distanceStep(uint32_t now_ms);
  void handleDistance(const DistanceReading& r, uint32_t now_ms);
  void controlStep(uint32_t now_ms);
  void classify();
  void handleCommand(const Command& c, uint32_t now_ms);
  void updateSafety(uint32_t now_ms);
  void applyMotors(uint32_t now_ms);
  void writeServo(int angle);
  void startMaintTest(uint32_t now_ms);
  void finishMaintTest(bool ok, const char* why);
  void printStatus();
  void printHelp();
  void say(const char* fmt, ...) __attribute__((format(printf, 2, 3)));

  Hal hal_;
  const MlEnsemble& model_;
  ClassifierThresholds thresholds_;
  ImuBuffer buffer_;
  ServoPolicy servo_policy_;   // discrete (SRA) mode
  ActuatorShaper shaper_;      // graded mode
  ObstacleMonitor obstacle_;
  RangeTracker range_;
  Supervisor supervisor_;
  Health health_;
  Ekf ekf_;
  SurfaceTracker tracker_;
  MaintenanceMonitor maint_;
  DampingAnalyzer analyzer_;

  bool use_ml_ = true;
  bool graded_ = true;
  SpeedMode speed_mode_ = SpeedMode::FUZZY_IT2;

  Features features_ = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0};
  float ml_features_[ML_FEATURE_COUNT];
  Decision decision_ = {false, {0.25f, 0.25f, 0.25f, 0.25f}, 0, 0.0f, 0.0f, 0.0f, true};
  CurveOutput curve_ = {0.43f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 90.0f};
  float fuzzy_u_ = 0.0f;

  bool imu_fault_ = false;
  int imu_fail_count_ = 0;
  uint32_t last_imu_retry_ms_ = 0;
  uint32_t last_imu_us_ = 0;
  bool have_last_imu_ = false;

  DistanceReading last_distance_ = {DistanceStatus::NO_RESPONSE, 0.0f};
  bool have_distance_ = false;
  bool dist_in_flight_ = false;
  uint32_t last_dist_ms_ = 0;
  bool near_ = false;

  uint32_t next_imu_us_ = 0;
  uint32_t next_dist_ms_ = 0;
  uint32_t next_control_ms_ = 0;
  uint32_t next_status_ms_ = 0;

  int servo_angle_ = -1;   // last angle written (-1 = never)
  int manual_servo_ = -1;  // operator override, -1 = automatic
  int motor_pwm_ = -1;     // last PWM written (-1 = never)
  float pwm_ramped_ = 0.0f;
  bool approach_active_ = false;  // monotone deceleration while approaching an obstacle
  int approach_cap_ = 0;
  float approach_ref_cm_ = 0.0f;
  uint32_t last_motor_ms_ = 0;

  bool manual_motor_ = false;
  int manual_pwm_ = 0;
  uint32_t manual_until_ms_ = 0;

  MaintPhase maint_phase_ = MaintPhase::NONE;
  uint32_t maint_deadline_ms_ = 0;

  LogMode log_mode_ = LogMode::OFF;
};
