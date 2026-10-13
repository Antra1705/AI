// types.h - data passed between the modules.
#pragma once
#include <stdint.h>

// One IMU reading. Acceleration in m/s^2, rotation in rad/s.
struct ImuSample {
  uint32_t t_us;  // when it was taken (micros(); wraps every ~71 minutes)
  float ax, ay, az;
  float gx, gy, gz;
};

enum Surface : uint8_t {
  SURFACE_SMOOTH = 0,
  SURFACE_GRAVEL = 1,
  SURFACE_BREAKER = 2,
  SURFACE_POTHOLE = 3,
  SURFACE_UNKNOWN = 4,
  SURFACE_COUNT = 5
};

// Computed over the last 1 second of IMU samples.
struct Features {
  float accel_mean;     // mean |a|, m/s^2 (about 9.81 when level)
  float jerk_max;       // largest |da/dt| between consecutive samples, m/s^3
  float peak_to_peak;   // max |a| - min |a|, m/s^2
  float accel_rms_dev;  // RMS of (|a| - mean), m/s^2: sustained vibration
  float gyro_std;       // standard deviation of |gyro|, rad/s
  int oscillations;     // swings of |a| across +/- OSC_BAND around the mean
};

struct Classification {
  Surface surface;
  float confidence;  // 0..1
};

enum class DistanceStatus : uint8_t {
  OK,            // cm is valid (2-400)
  TOO_CLOSE,     // echo came back but under 2 cm
  OUT_OF_RANGE,  // sensor answered but nothing within 400 cm: path is clear
  NO_RESPONSE    // sensor did not answer at all: possible fault
};

struct DistanceReading {
  DistanceStatus status;
  float cm;
};

enum class RunState : uint8_t {
  IDLE,      // motors off, waiting for 'start'
  RUNNING,   // driving, path clear
  APPROACH,  // driving with an obstacle ahead (< APPROACH_CM): speed moderated
  ESTOP,     // obstacle under 10 cm, motors braked
  FAULT,     // a sensor failed, motors braked until the operator restarts
  MAINT      // predictive-maintenance reference test
};

const char* surfaceName(Surface s);
const char* runStateName(RunState s);
const char* distanceStatusName(DistanceStatus s);
// Case-insensitive name -> Surface. Returns false if the name is not recognised.
bool parseSurface(const char* name, Surface* out);
