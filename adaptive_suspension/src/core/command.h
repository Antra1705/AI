// command.h - operator commands (serial monitor, or the dashboard).
//
//   start | stop | status | health | help
//   log raw | log features | log json | log off
//   servo <0-180> | servo auto
//   motor <pwm 1-255> [ms]          (IDLE only; default 1000 ms)
//   maint | maint test | maint reset
//   mode ml | mode rules            classifier: neural ensemble / rule-based fallback
//   curve graded | curve discrete   damping: confidence curve / SRA fixed angles
//   speed it2 | speed t1 | speed step   obstacle speed: Type-2 fuzzy / Type-1 / SRA steps
#pragma once

enum class CommandType {
  NONE,  // empty line
  HELP,
  START,
  STOP,
  STATUS,
  HEALTH,
  LOG_RAW,
  LOG_FEATURES,
  LOG_JSON,
  LOG_OFF,
  SERVO_SET,
  SERVO_AUTO,
  MOTOR_PULSE,
  MAINT_STATUS,
  MAINT_TEST,
  MAINT_RESET,
  MODE_ML,
  MODE_RULES,
  CURVE_GRADED,
  CURVE_DISCRETE,
  SPEED_IT2,
  SPEED_T1,
  SPEED_STEP,
  INVALID
};

struct Command {
  CommandType type;
  int a;  // servo angle, or motor pwm
  int b;  // motor duration ms
};

Command parseCommand(const char* line);
