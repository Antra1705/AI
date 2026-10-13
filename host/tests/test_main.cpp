#include "test.h"

int main() {
  int failed_tests = 0;
  for (const TestCase& t : testRegistry()) {
    int before = testFailures();
    t.fn();
    bool ok = testFailures() == before;
    if (!ok) ++failed_tests;
    std::printf("[%s] %s\n", ok ? " OK " : "FAIL", t.name);
  }
  std::printf("\n%zu tests, %d failed\n", testRegistry().size(), failed_tests);
  return failed_tests == 0 ? 0 : 1;
}
