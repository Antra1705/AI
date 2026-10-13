// health.h - counters for the 'health' command (BR-11).
#pragma once
#include <stdint.h>

struct Health {
  uint32_t imu_read_errors = 0;        // single failed IMU reads
  uint32_t imu_faults = 0;             // times the IMU was declared faulty
  uint32_t imu_overruns = 0;           // times the loop fell a whole IMU period behind
  uint32_t dist_no_response = 0;       // distance readings with no answer
  uint32_t faults = 0;                 // entries into FAULT state
  uint32_t estops = 0;                 // entries into ESTOP state
  uint32_t impacts = 0;                // fast-path impacts detected
  uint32_t max_tick_us = 0;            // slowest single loop pass
  uint32_t last_impact_latency_us = 0; // sample time -> servo command
  uint32_t max_impact_latency_us = 0;
};
