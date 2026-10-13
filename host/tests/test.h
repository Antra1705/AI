// test.h - minimal unit test framework (no external dependencies).
#pragma once
#include <cmath>
#include <cstdio>
#include <vector>

struct TestCase {
  const char* name;
  void (*fn)();
};

inline std::vector<TestCase>& testRegistry() {
  static std::vector<TestCase> tests;
  return tests;
}

inline int& testFailures() {
  static int failures = 0;
  return failures;
}

struct TestRegistrar {
  TestRegistrar(const char* name, void (*fn)()) { testRegistry().push_back(TestCase{name, fn}); }
};

#define TEST(name)                                         \
  static void name();                                      \
  static TestRegistrar registrar_##name(#name, &name);     \
  static void name()

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("    FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
      ++testFailures();                                                      \
    }                                                                        \
  } while (0)

#define CHECK_EQ(a, b)                                                               \
  do {                                                                               \
    double va_ = (double)(a), vb_ = (double)(b);                                     \
    if (!(va_ == vb_)) {                                                             \
      std::printf("    FAIL %s:%d  %s == %s  (%g vs %g)\n", __FILE__, __LINE__, #a, #b, \
                  va_, vb_);                                                         \
      ++testFailures();                                                              \
    }                                                                                \
  } while (0)

#define CHECK_NEAR(a, b, eps)                                                             \
  do {                                                                                    \
    double va_ = (double)(a), vb_ = (double)(b);                                          \
    if (std::fabs(va_ - vb_) > (eps)) {                                                   \
      std::printf("    FAIL %s:%d  %s ~= %s  (%g vs %g, eps %g)\n", __FILE__, __LINE__, #a, \
                  #b, va_, vb_, (double)(eps));                                           \
      ++testFailures();                                                                   \
    }                                                                                     \
  } while (0)
