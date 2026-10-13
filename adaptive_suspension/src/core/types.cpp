#include "types.h"

#include <ctype.h>

const char* surfaceName(Surface s) {
  switch (s) {
    case SURFACE_SMOOTH: return "SMOOTH";
    case SURFACE_GRAVEL: return "GRAVEL";
    case SURFACE_BREAKER: return "BREAKER";
    case SURFACE_POTHOLE: return "POTHOLE";
    default: return "UNKNOWN";
  }
}

const char* runStateName(RunState s) {
  switch (s) {
    case RunState::IDLE: return "IDLE";
    case RunState::RUNNING: return "RUNNING";
    case RunState::APPROACH: return "APPROACH";
    case RunState::ESTOP: return "ESTOP";
    case RunState::FAULT: return "FAULT";
    case RunState::MAINT: return "MAINT";
  }
  return "?";
}

const char* distanceStatusName(DistanceStatus s) {
  switch (s) {
    case DistanceStatus::OK: return "OK";
    case DistanceStatus::TOO_CLOSE: return "TOO_CLOSE";
    case DistanceStatus::OUT_OF_RANGE: return "OUT_OF_RANGE";
    case DistanceStatus::NO_RESPONSE: return "NO_RESPONSE";
  }
  return "?";
}

static bool equalsIgnoreCase(const char* a, const char* b) {
  while (*a && *b) {
    if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return false;
    ++a;
    ++b;
  }
  return *a == *b;
}

bool parseSurface(const char* name, Surface* out) {
  for (int i = 0; i < SURFACE_COUNT; ++i) {
    Surface s = static_cast<Surface>(i);
    if (equalsIgnoreCase(name, surfaceName(s))) {
      *out = s;
      return true;
    }
  }
  return false;
}
