// config.h - every threshold, speed, angle and timing value in one place (BR-12).
//
// Values marked CALIBRATE are starting points. Replace them with the numbers
// printed by tools/calibrate.py once real data from the cart is available.
//
// Pin numbers are intentionally NOT here yet: they belong to the hardware
// drivers, which are added once the hardware is available.
#pragma once
#include <stdint.h>

#define FIRMWARE_VERSION "0.2.0"

namespace cfg {

// ---------------- Timing ----------------
constexpr uint32_t IMU_PERIOD_US       = 10000;  // 100 Hz IMU sampling (BR-3)
constexpr int      IMU_BUFFER_SIZE     = 100;    // 1 second of samples
constexpr uint32_t DIST_PERIOD_MS      = 40;     // distance check 25x per second (fixes I-10)
constexpr uint32_t CONTROL_PERIOD_MS   = 100;    // classify + servo decision at 10 Hz
constexpr uint32_t STATUS_PERIOD_MS    = 500;    // status line at 2 Hz (BR-10)
constexpr uint32_t WATCHDOG_TIMEOUT_MS = 2000;   // reset if the loop stalls (BR-9)

// ---------------- Fault detection ----------------
constexpr int      IMU_FAIL_LIMIT  = 5;     // consecutive failed reads (50 ms) -> IMU fault
constexpr uint32_t IMU_RETRY_MS    = 1000;  // how often to try reconnecting a faulted IMU
constexpr int      DIST_FAIL_LIMIT = 3;     // consecutive no-response readings -> sensor fault

// ---------------- Distance / obstacle (BR-7, BR-8) ----------------
constexpr float DIST_MIN_CM            = 2.0f;    // HC-SR04 lower limit
constexpr float DIST_MAX_CM            = 400.0f;  // HC-SR04 upper limit
constexpr float SLOW_CM                = 20.0f;   // below this: slow speed
constexpr float ESTOP_CM               = 10.0f;   // below this: emergency stop
constexpr float ESTOP_RELEASE_CM       = 15.0f;   // must be clear beyond this to release...
constexpr int   ESTOP_RELEASE_READINGS = 3;       // ...for this many readings in a row
constexpr bool  ESTOP_AUTO_RESUME      = true;    // resume driving once the path is clear

// ---------------- Motors ----------------
constexpr int      SPEED_NORMAL        = 200;   // PWM 0-255
constexpr int      SPEED_SLOW          = 127;
constexpr uint32_t MANUAL_PULSE_MAX_MS = 3000;  // longest 'motor' test command

// ---------------- Servo (higher angle = stiffer suspension) ----------------
constexpr int SERVO_SMOOTH  = 45;
constexpr int SERVO_GRAVEL  = 110;
constexpr int SERVO_BREAKER    = 150;
constexpr int SERVO_POTHOLE = 150;
constexpr int SERVO_NEUTRAL = 90;
constexpr int SERVO_IMPACT  = 150;               // fast-path angle on a sudden impact
constexpr int SOFTEN_CONFIRM_CYCLES = 3;         // softer setting must be asked for 3 cycles in a row
constexpr uint32_t IMPACT_HOLD_MS   = 1000;      // stay stiff this long after an impact

// ---------------- Rule-based classifier (fallback; CALIBRATE) ----------------
// Jerk is in m/s^3 (the SRA's per-sample thresholds x100, fixes I-1).
// Values from tools/calibrate.py --dataset on SIMULATED drives (host: sim dataset).
// The surfaces overlap in these 4 features, so this is only the fallback and the
// baseline; the neural network (ml_model.h) is the main classifier.
constexpr float CONFIDENCE_GATE   = 0.60f;  // discrete mode: servo moves only above this (SRA REQ-4)
constexpr float CONF_MIN          = 0.55f;  // rule confidence right at a threshold
constexpr float CONF_MAX          = 0.95f;  // rule confidence well past a threshold
constexpr float CONF_UNKNOWN      = 0.50f;

constexpr float POTHOLE_JERK      = 173.0f;  // CALIBRATE  m/s^3
constexpr float BREAKER_P2P       = 1.09f;   // CALIBRATE  m/s^2 peak-to-peak
constexpr float GRAVEL_RMS        = 0.7f;    // CALIBRATE  m/s^2 sustained vibration
constexpr int   GRAVEL_OSC        = 13;      // CALIBRATE  swings per second
constexpr float SMOOTH_JERK       = 16.3f;   // CALIBRATE  m/s^3
constexpr float SMOOTH_P2P        = 1.09f;   // CALIBRATE  m/s^2
constexpr float OSC_BAND          = 0.3f;    // m/s^2; a swing must cross +/- this to count (also an ML feature)

// One-sample impact reflex. Deliberately above ~99% of gravel jerks (simulated
// gravel p99 ~2500 m/s^3): gravel wants soft damping, so the reflex must not
// fire there. It catches only the hardest hits; everything else goes through
// the classifier and the curve's protective floor within ~100 ms.
constexpr float IMPACT_JERK       = 3000.0f;  // CALIBRATE  m/s^3

// ---------------- Extended Kalman Filter (ekf.h) ----------------
constexpr float EKF_GYRO_NOISE    = 0.01f;    // rad/s, gyro noise driving attitude uncertainty
constexpr float EKF_BIAS_WALK     = 0.0005f;  // rad/s per sqrt(s), gyro bias drift
constexpr float EKF_ACCEL_NOISE   = 1.0f;     // m/s^2, accelerometer noise on smooth road
constexpr float EKF_DYNAMIC_ACCEL = 1.0f;     // m/s^2; R doubles when | |a| - g | reaches this
constexpr float EKF_ZERO_VZ_NOISE = 2.0f;     // m/s, weak "average vertical velocity is 0" measurement (~0.7 s drift correction)
constexpr float EKF_VZ_LEAK       = 0.5f;     // 1/s, leak on the vertical-velocity integrator
// Gravity-update noise multiplier per road class (the classifier link). Kept mild:
// 'sim ekf' shows strong multipliers (30-100x) throw away useful data, because gravel
// vibration averages out; only impacts briefly corrupt the gravity reading.
constexpr float EKF_R_SMOOTH  = 1.0f;
constexpr float EKF_R_GRAVEL  = 2.0f;
constexpr float EKF_R_BREAKER = 4.0f;
constexpr float EKF_R_POTHOLE = 8.0f;
constexpr float EKF_R_UNKNOWN = 1.0f;      // classifier not ready yet

// ---------------- Graded damping curve (response_curve.h) ----------------
constexpr int   SERVO_SOFT_ANGLE  = 45;     // s = 0
constexpr int   SERVO_STIFF_ANGLE = 150;    // s = 1
// Target stiffness per class, from 'sim tune' (damping characterisation on the
// simulated cart; CALIBRATE on the real one): gravel is high-frequency
// vibration that soft damping isolates; breakers/potholes need stiff damping
// to avoid bottoming out; smooth road is flat from 0.6 upward.
constexpr float STIFF_SMOOTH      = 0.6f;
constexpr float STIFF_GRAVEL      = 0.0f;
constexpr float STIFF_BREAKER     = 1.0f;
constexpr float STIFF_POTHOLE     = 1.0f;
constexpr float STIFF_SAFE        = 0.43f;  // = 90 deg: used when the classifier is not trusted
constexpr float CURVE_C_LOW       = 0.55f;  // effective confidence where trust starts
constexpr float CURVE_C_HIGH      = 0.90f;  // ... and where it is full
constexpr float CURVE_RISK_GAIN   = 0.9f;   // floor = gain * P(breaker or pothole)
constexpr float CURVE_SKY_GAIN    = 0.15f;  // extra stiffness when the body bounces (EKF vz)
constexpr float CURVE_SKY_VREF    = 0.05f;  // m/s of vertical velocity for the full extra
constexpr float SLEW_UP_DEG_S     = 300.0f; // MG90S under load (~60 deg / 0.2 s): stiffen at full speed
constexpr float SLEW_DOWN_DEG_S   = 60.0f;  // soften slowly, avoids pumping and servo wear
constexpr int   SERVO_DEADBAND_DEG = 2;     // ignore smaller changes (jitter, wear)
constexpr int   SURFACE_CONFIRM   = 2;      // windows in a row before the reported surface changes

// ---------------- Speed control near obstacles (fuzzy_speed.h) ----------------
constexpr float FUZZY_LOWER_HEIGHT = 0.9f;  // IT2: height of the lower membership curves
constexpr float RANGE_ALPHA       = 0.5f;   // alpha-beta filter on distance readings
constexpr float RANGE_BETA        = 0.08f;  // low: smooth closing-speed estimate from noisy echoes
constexpr int   MOTOR_MIN_PWM     = 60;     // below this the TT motors stall: speed factor maps onto 60..SPEED_NORMAL
constexpr float PWM_RAMP_UP       = 400.0f; // PWM per second when speeding up
constexpr float PWM_RAMP_DOWN     = 800.0f; // PWM per second when slowing (emergency stop is instant)
constexpr float APPROACH_CM       = 60.0f;  // enter the obstacle-proximity state below this
constexpr float APPROACH_EXIT_CM  = 70.0f;  // ... and leave it above this
constexpr float FUZZY_STOP_U      = 0.1f;   // speed factor below this = stop (gentle stop before the hard 10 cm brake)
constexpr float APPROACH_REARM_CM = 10.0f;  // while approaching, speed only rises again if the obstacle moves this much further away

// ---------------- Predictive maintenance (maintenance.h) ----------------
constexpr int      MAINT_TEST_ANGLE      = 45;     // servo locked soft during the reference test
constexpr int      MAINT_TEST_PWM        = 110;    // slow, repeatable speed over the reference bump
constexpr float    MAINT_TRIGGER         = 1.0f;   // m/s^2 vertical accel that marks the bump hit (smooth-road noise ~0.2)
constexpr uint32_t MAINT_TIMEOUT_MS      = 20000;  // give up if no bump is found
constexpr float    MAINT_MIN_PEAK        = 1.0f;   // m/s^2; a weaker hit is not usable
constexpr float    MAINT_FORCED_FRACTION = 0.5f;   // free decay starts after the last sample above this x peak
constexpr float    MAINT_NOISE_FLOOR     = 0.12f;  // m/s^2; smaller peaks are ignored
constexpr int      MAINT_MIN_PEAKS       = 3;
constexpr int      MAINT_BASELINE_TESTS  = 5;      // tests that form the healthy baseline
constexpr float    MAINT_MIN_REL_STD     = 0.05f;  // baseline spread floor, fraction of the mean
constexpr float    MAINT_EWMA_LAMBDA     = 0.3f;
constexpr float    MAINT_CUSUM_K         = 0.5f;   // slack, in baseline standard deviations
constexpr float    MAINT_CUSUM_WATCH     = 3.0f;
constexpr float    MAINT_CUSUM_SERVICE   = 6.0f;
constexpr float    MAINT_DROP_WATCH      = 0.15f;  // EWMA 15 % below baseline
constexpr float    MAINT_DROP_SERVICE    = 0.30f;
constexpr int      MAINT_HISTORY         = 16;

// ---------------- Default modes (switchable at runtime: mode / curve / speed) ----------------
constexpr bool DEFAULT_CLASSIFIER_ML = true;  // neural ensemble (falls back to rules if no model)
constexpr bool DEFAULT_CURVE_GRADED  = true;  // graded confidence curve (false = SRA fixed angles)
constexpr int  DEFAULT_SPEED_MODE    = 2;     // 0 = SRA steps, 1 = Type-1 fuzzy, 2 = interval Type-2 fuzzy

}  // namespace cfg
