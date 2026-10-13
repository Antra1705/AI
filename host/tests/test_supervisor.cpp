#include "core/supervisor.h"
#include "test.h"

TEST(supervisor_starts_idle) {
  Supervisor s;
  CHECK(s.state() == RunState::IDLE);
  s.update(false, false, false, false);
  CHECK(s.state() == RunState::IDLE);  // never starts by itself
}

TEST(supervisor_refuses_start_with_fault) {
  Supervisor s;
  CHECK(!s.start(true, false));
  CHECK(s.state() == RunState::IDLE);
  CHECK(!s.start(false, true));
  CHECK(s.state() == RunState::IDLE);
  CHECK(s.start(false, false));
  CHECK(s.state() == RunState::RUNNING);
}

TEST(supervisor_estop_and_resume) {
  Supervisor s;
  s.start(false, false);
  s.update(false, false, true, true);
  CHECK(s.state() == RunState::ESTOP);
  s.update(false, false, false, false);
  CHECK(s.state() == RunState::RUNNING);
}

TEST(supervisor_approach_state_with_obstacle_ahead) {
  Supervisor s;
  s.start(false, false);
  s.update(false, false, false, true);
  CHECK(s.state() == RunState::APPROACH);
  s.update(false, false, false, true);
  CHECK(s.state() == RunState::APPROACH);
  s.update(false, false, false, false);
  CHECK(s.state() == RunState::RUNNING);
  s.update(false, false, false, true);
  s.update(false, false, true, true);
  CHECK(s.state() == RunState::ESTOP);
  s.update(false, false, false, true);  // released but obstacle still within range
  CHECK(s.state() == RunState::APPROACH);
}

TEST(supervisor_fault_latches_until_operator) {
  Supervisor s;
  s.start(false, false);
  s.update(true, false, false, false);
  CHECK(s.state() == RunState::FAULT);
  s.update(false, false, false, false);  // fault went away by itself
  CHECK(s.state() == RunState::FAULT);   // still latched
  CHECK(s.start(false, false));
  CHECK(s.state() == RunState::RUNNING);
}

TEST(supervisor_fault_beats_estop) {
  Supervisor s;
  s.start(false, false);
  s.update(false, true, true, true);
  CHECK(s.state() == RunState::FAULT);
}

TEST(supervisor_stop_from_anywhere) {
  Supervisor s;
  s.start(false, false);
  s.update(false, false, true, true);
  s.stop();
  CHECK(s.state() == RunState::IDLE);
}

TEST(supervisor_maintenance_test) {
  Supervisor s;
  CHECK(s.startMaint(false, false));
  CHECK(s.state() == RunState::MAINT);
  s.update(false, false, false, true);  // obstacle ahead does not change the test
  CHECK(s.state() == RunState::MAINT);
  s.update(false, false, true, true);   // ...but an emergency stops it
  CHECK(s.state() == RunState::ESTOP);
  s.update(false, false, false, false);
  CHECK(s.state() == RunState::IDLE);   // and it does not resume driving on its own
  s.start(false, false);
  CHECK(!s.startMaint(false, false));   // only from IDLE
}
