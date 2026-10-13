// The ESP32 serves its own copy of the dashboard; it must match host/dashboard/index.html.
#include <cstdio>
#include <cstring>
#include <string>

#include "esp32/dashboard_html.h"
#include "test.h"

TEST(esp32_dashboard_copy_is_up_to_date) {
  FILE* f = std::fopen("dashboard/index.html", "rb");  // tests run from host/
  CHECK(f != nullptr);
  if (!f) return;
  std::string html;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) html.append(buf, n);
  std::fclose(f);
  // If this fails: python3 tools/embed_dashboard.py
  CHECK(html == std::string(DASHBOARD_HTML));
}
