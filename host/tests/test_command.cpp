#include "config.h"
#include "core/command.h"
#include "test.h"

TEST(command_simple_words) {
  CHECK(parseCommand("start").type == CommandType::START);
  CHECK(parseCommand("  STOP  ").type == CommandType::STOP);
  CHECK(parseCommand("status").type == CommandType::STATUS);
  CHECK(parseCommand("health").type == CommandType::HEALTH);
  CHECK(parseCommand("help").type == CommandType::HELP);
  CHECK(parseCommand("").type == CommandType::NONE);
  CHECK(parseCommand("   ").type == CommandType::NONE);
}

TEST(command_log_modes) {
  CHECK(parseCommand("log raw").type == CommandType::LOG_RAW);
  CHECK(parseCommand("log features").type == CommandType::LOG_FEATURES);
  CHECK(parseCommand("log off").type == CommandType::LOG_OFF);
  CHECK(parseCommand("log").type == CommandType::INVALID);
  CHECK(parseCommand("log banana").type == CommandType::INVALID);
}

TEST(command_servo) {
  Command c = parseCommand("servo 120");
  CHECK(c.type == CommandType::SERVO_SET);
  CHECK_EQ(c.a, 120);
  CHECK(parseCommand("servo auto").type == CommandType::SERVO_AUTO);
  CHECK(parseCommand("servo 181").type == CommandType::INVALID);
  CHECK(parseCommand("servo -1").type == CommandType::INVALID);
  CHECK(parseCommand("servo 12x").type == CommandType::INVALID);
}

TEST(command_motor) {
  Command c = parseCommand("motor 150");
  CHECK(c.type == CommandType::MOTOR_PULSE);
  CHECK_EQ(c.a, 150);
  CHECK_EQ(c.b, 1000);
  c = parseCommand("motor 200 2500");
  CHECK_EQ(c.b, 2500);
  CHECK(parseCommand("motor 0").type == CommandType::INVALID);
  CHECK(parseCommand("motor 256").type == CommandType::INVALID);
  CHECK(parseCommand("motor 100 999999").type == CommandType::INVALID);
}

TEST(command_rejects_junk) {
  CHECK(parseCommand("fly").type == CommandType::INVALID);
  CHECK(parseCommand("start now").type == CommandType::INVALID);
  CHECK(parseCommand("motor 100 200 300 400").type == CommandType::INVALID);
}
