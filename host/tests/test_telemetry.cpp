#include <cstring>

#include "core/telemetry.h"
#include "test.h"

TEST(raw_line_round_trip) {
  ImuSample s = {123456789u, 0.123f, -1.5f, 9.81f, 0.0012f, -0.25f, 3.1416f};
  char buf[128];
  formatRaw(buf, sizeof buf, s);
  ImuSample p;
  CHECK(parseRawLine(buf, &p));
  CHECK_EQ(p.t_us, s.t_us);
  CHECK_NEAR(p.ax, s.ax, 1e-3);
  CHECK_NEAR(p.ay, s.ay, 1e-3);
  CHECK_NEAR(p.az, s.az, 1e-3);
  CHECK_NEAR(p.gx, s.gx, 1e-4);
  CHECK_NEAR(p.gy, s.gy, 1e-4);
  CHECK_NEAR(p.gz, s.gz, 1e-4);
}

TEST(raw_line_rejects_other_lines) {
  ImuSample p;
  CHECK(!parseRawLine("F,100,9.8,1,1,1,1,1,SMOOTH,0.9,45", &p));
  CHECK(!parseRawLine("# R,t_us,ax", &p));
  CHECK(!parseRawLine("R,12,1,2,3", &p));  // too few fields
  CHECK(!parseRawLine("hello", &p));
}

TEST(feature_line_format) {
  Features f = {9.81f, 12.0f, 0.3f, 0.05f, 0.01f, 2};
  Classification c = {SURFACE_SMOOTH, 0.91f};
  char buf[160];
  formatFeatures(buf, sizeof buf, 1500, f, c, 45);
  CHECK(std::strcmp(buf, "F,1500,9.810,12.0,0.300,0.050,2,0.0100,SMOOTH,0.91,45\n") == 0);
}

static Snapshot blank() {
  Snapshot s;
  std::memset(&s, 0, sizeof s);
  s.fault_reason = "";
  s.classifier = "ml";
  s.curve_mode = "graded";
  s.speed_mode = "fuzzy-it2";
  return s;
}

TEST(status_line_shows_fault_reason) {
  Snapshot s = blank();
  s.state = RunState::FAULT;
  s.fault_reason = "IMU not responding";
  s.imu_fault = true;
  s.servo = 90;
  char buf[320];
  formatStatus(buf, sizeof buf, s);
  CHECK(std::strstr(buf, "FAULT") != nullptr);
  CHECK(std::strstr(buf, "IMU-FAULT (IMU not responding)") != nullptr);
}

TEST(json_line_is_well_formed) {
  Snapshot s = blank();
  s.state = RunState::APPROACH;
  s.ready = true;
  s.p[0] = 0.9f;
  s.surface = 0;
  s.have_distance = true;
  s.dist = DistanceReading{DistanceStatus::OK, 42.0f};
  char buf[1100];
  int n = formatJson(buf, sizeof buf, s);
  CHECK(n > 0 && n < (int)sizeof buf);
  CHECK(std::strncmp(buf, "J,{", 3) == 0);
  CHECK(buf[n - 2] == '}' && buf[n - 1] == '\n');
  CHECK(std::strstr(buf, "\"state\":\"APPROACH\"") != nullptr);
  CHECK(std::strstr(buf, "\"surface\":\"SMOOTH\"") != nullptr);
  CHECK(std::strstr(buf, "\"dist\":42.0") != nullptr);
  int depth = 0;  // balanced braces and brackets
  for (int i = 2; i < n; ++i) depth += (buf[i] == '{' || buf[i] == '[') - (buf[i] == '}' || buf[i] == ']');
  CHECK_EQ(depth, 0);
}
