// supervisor.h - run state machine.
//
//   IDLE     --start (no faults)-->        RUNNING
//   IDLE     --maint test (no faults)-->   MAINT
//   RUNNING  --obstacle < APPROACH_CM-->   APPROACH  --clear again (> APPROACH_EXIT_CM)--> RUNNING
//   RUNNING/APPROACH/MAINT --obstacle < 10 cm--> ESTOP
//   ESTOP    --path clear-->  RUNNING (if ESTOP_AUTO_RESUME), or IDLE if it came from MAINT
//   any moving state --sensor fault--> FAULT  (stays until the operator sends start/stop)
//   any      --stop-->  IDLE
// The cart never starts moving on its own after power-up (BR-2).
// New states (e.g. the fall/theft/crash states of the earlier project) slot in
// here: add them to RunState and give them a branch in update().
#pragma once
#include "../config.h"
#include "types.h"

class Supervisor {
 public:
  RunState state() const { return state_; }
  const char* faultReason() const { return reason_; }

  bool start(bool imu_fault, bool dist_fault) { return enter(RunState::RUNNING, imu_fault, dist_fault); }

  bool startMaint(bool imu_fault, bool dist_fault) { return startTest(RunState::MAINT, imu_fault, dist_fault); }

  void stop() {
    state_ = RunState::IDLE;
    reason_ = "";
  }

  void update(bool imu_fault, bool dist_fault, bool emergency, bool near) {
    if (state_ == RunState::IDLE || state_ == RunState::FAULT) return;
    if (imu_fault || dist_fault) {
      state_ = RunState::FAULT;
      reason_ = imu_fault ? "IMU not responding" : "distance sensor not responding";
      return;
    }
    if (emergency) {
      if (state_ != RunState::ESTOP) {
        resume_ = state_ == RunState::MAINT ? RunState::IDLE : RunState::RUNNING;
        state_ = RunState::ESTOP;
      }
      return;
    }
    if (state_ == RunState::ESTOP) {
      if (!cfg::ESTOP_AUTO_RESUME) return;
      state_ = resume_;
      if (state_ != RunState::RUNNING) return;
    }
    if (state_ == RunState::RUNNING && near) state_ = RunState::APPROACH;
    else if (state_ == RunState::APPROACH && !near) state_ = RunState::RUNNING;
  }

 private:
  bool startTest(RunState s, bool imu_fault, bool dist_fault) {
    if (state_ != RunState::IDLE) {
      reason_ = "tests only start from IDLE";
      return false;
    }
    return enter(s, imu_fault, dist_fault);
  }

  bool enter(RunState s, bool imu_fault, bool dist_fault) {
    if (imu_fault || dist_fault) {
      reason_ = imu_fault ? "IMU not responding" : "distance sensor not responding";
      return false;
    }
    reason_ = "";
    state_ = s;
    return true;
  }

  RunState state_ = RunState::IDLE;
  RunState resume_ = RunState::RUNNING;
  const char* reason_ = "";
};
