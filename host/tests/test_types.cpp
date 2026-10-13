#include <cstring>

#include "core/types.h"
#include "test.h"

TEST(surface_names_round_trip) {
  for (int i = 0; i < SURFACE_COUNT; ++i) {
    Surface s = static_cast<Surface>(i);
    Surface parsed = SURFACE_UNKNOWN;
    CHECK(parseSurface(surfaceName(s), &parsed));
    CHECK_EQ(parsed, s);
  }
}

TEST(parse_surface_ignores_case_and_rejects_junk) {
  Surface s = SURFACE_UNKNOWN;
  CHECK(parseSurface("gravel", &s));
  CHECK_EQ(s, SURFACE_GRAVEL);
  CHECK(!parseSurface("gravelx", &s));
  CHECK(!parseSurface("", &s));
}

TEST(state_names) {
  CHECK(std::strcmp(runStateName(RunState::ESTOP), "ESTOP") == 0);
  CHECK(std::strcmp(distanceStatusName(DistanceStatus::NO_RESPONSE), "NO_RESPONSE") == 0);
}
