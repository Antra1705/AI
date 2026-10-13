#include "command.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "../config.h"

namespace {

const int MAX_TOKENS = 4;
const int MAX_LINE = 64;

// Splits a lower-cased copy of line into whitespace-separated tokens.
int tokenize(const char* line, char* storage, char* tokens[]) {
  int len = 0;
  while (line[len] && len < MAX_LINE - 1) {
    storage[len] = (char)tolower((unsigned char)line[len]);
    ++len;
  }
  storage[len] = '\0';

  int n = 0;
  char* p = storage;
  while (*p && n < MAX_TOKENS) {
    while (*p && isspace((unsigned char)*p)) *p++ = '\0';
    if (!*p) break;
    tokens[n++] = p;
    while (*p && !isspace((unsigned char)*p)) ++p;
  }
  // Anything beyond MAX_TOKENS makes the command invalid.
  while (*p && isspace((unsigned char)*p)) ++p;
  return *p ? MAX_TOKENS + 1 : n;
}

bool parseInt(const char* s, int lo, int hi, int* out) {
  char* end = 0;
  long v = strtol(s, &end, 10);
  if (end == s || *end != '\0' || v < lo || v > hi) return false;
  *out = (int)v;
  return true;
}

Command make(CommandType t, int a = 0, int b = 0) {
  Command c;
  c.type = t;
  c.a = a;
  c.b = b;
  return c;
}

}  // namespace

Command parseCommand(const char* line) {
  char storage[MAX_LINE];
  char* tok[MAX_TOKENS];
  int n = tokenize(line, storage, tok);
  if (n == 0) return make(CommandType::NONE);
  if (n > MAX_TOKENS) return make(CommandType::INVALID);

  const char* cmd = tok[0];
  if (n == 1) {
    if (!strcmp(cmd, "start")) return make(CommandType::START);
    if (!strcmp(cmd, "stop")) return make(CommandType::STOP);
    if (!strcmp(cmd, "status")) return make(CommandType::STATUS);
    if (!strcmp(cmd, "health")) return make(CommandType::HEALTH);
    if (!strcmp(cmd, "help") || !strcmp(cmd, "?")) return make(CommandType::HELP);
    if (!strcmp(cmd, "maint")) return make(CommandType::MAINT_STATUS);
    return make(CommandType::INVALID);
  }

  if (n == 2) {
    struct Word {
      const char* a;
      const char* b;
      CommandType t;
    };
    static const Word words[] = {
        {"maint", "test", CommandType::MAINT_TEST},     {"maint", "reset", CommandType::MAINT_RESET},
        {"maint", "status", CommandType::MAINT_STATUS}, {"mode", "ml", CommandType::MODE_ML},
        {"mode", "rules", CommandType::MODE_RULES},     {"curve", "graded", CommandType::CURVE_GRADED},
        {"curve", "discrete", CommandType::CURVE_DISCRETE}, {"speed", "it2", CommandType::SPEED_IT2},
        {"speed", "t1", CommandType::SPEED_T1},         {"speed", "step", CommandType::SPEED_STEP},
        {"log", "json", CommandType::LOG_JSON},
    };
    for (const Word& w : words)
      if (!strcmp(cmd, w.a) && !strcmp(tok[1], w.b)) return make(w.t);
  }

  if (!strcmp(cmd, "log") && n == 2) {
    if (!strcmp(tok[1], "raw")) return make(CommandType::LOG_RAW);
    if (!strcmp(tok[1], "features")) return make(CommandType::LOG_FEATURES);
    if (!strcmp(tok[1], "off")) return make(CommandType::LOG_OFF);
  }

  if (!strcmp(cmd, "servo") && n == 2) {
    if (!strcmp(tok[1], "auto")) return make(CommandType::SERVO_AUTO);
    int angle;
    if (parseInt(tok[1], 0, 180, &angle)) return make(CommandType::SERVO_SET, angle);
  }

  if (!strcmp(cmd, "motor") && (n == 2 || n == 3)) {
    int pwm, ms = 1000;
    if (parseInt(tok[1], 1, 255, &pwm) &&
        (n == 2 || parseInt(tok[2], 1, (int)cfg::MANUAL_PULSE_MAX_MS, &ms))) {
      return make(CommandType::MOTOR_PULSE, pwm, ms);
    }
  }

  return make(CommandType::INVALID);
}
