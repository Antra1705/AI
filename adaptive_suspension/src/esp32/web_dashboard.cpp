// web_dashboard.cpp - the cart's own WiFi network and dashboard (BR-21).
//
// Runs in a separate FreeRTOS task on core 0 (WiFi's core); the control loop runs
// on core 1. The two only exchange data through a FreeRTOS ring buffer (output
// lines) and queue (commands), so a slow phone can never delay the suspension.
#ifdef ARDUINO_ARCH_ESP32

#include <WiFi.h>

#include <atomic>
#include <freertos/FreeRTOS.h>
#include <freertos/ringbuf.h>
#include <freertos/task.h>

#include "board_config.h"
#include "dashboard_html.h"
#include "esp32_hal.h"

#if __has_include(<esp_arduino_version.h>)
#include <esp_arduino_version.h>
#endif
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
#define ACCEPT_CLIENT(server) (server).accept()
#else
#define ACCEPT_CLIENT(server) (server).available()
#endif

namespace {

RingbufHandle_t g_lines = nullptr;  // control loop -> web task
std::atomic<int> g_streams{0};      // browsers currently connected to /events
EspConsole* g_console = nullptr;

struct Stream {
  WiFiClient client;
  bool used = false;
};
Stream g_stream[board::WEB_MAX_STREAMS];

bool readHttpLine(WiFiClient& c, char* buf, int size, uint32_t timeout_ms) {
  int n = 0;
  const uint32_t t0 = millis();
  while (millis() - t0 < timeout_ms) {
    const int ch = c.read();
    if (ch < 0) {
      if (!c.connected()) return false;
      vTaskDelay(1);
      continue;
    }
    if (ch == '\r') continue;
    if (ch == '\n') {
      buf[n] = '\0';
      return true;
    }
    if (n < size - 1) buf[n++] = (char)ch;
  }
  return false;
}

bool sendAll(WiFiClient& c, const char* data, size_t len) {
  while (len > 0) {
    const size_t w = c.write((const uint8_t*)data, len > 1024 ? 1024 : len);
    if (w == 0) return false;
    data += w;
    len -= w;
  }
  return true;
}

bool sendText(WiFiClient& c, const char* text) { return sendAll(c, text, strlen(text)); }

void closeStream(Stream& s) {
  s.client.stop();
  s.used = false;
  --g_streams;
}

// One SSE message (or keep-alive comment) to every connected browser.
void forward(const char* line, size_t len, bool comment) {
  for (Stream& s : g_stream) {
    if (!s.used) continue;
    if (!s.client.connected()) {
      closeStream(s);
      continue;
    }
    bool ok;
    if (comment) ok = sendText(s.client, ": ping\n\n");
    else ok = sendAll(s.client, "data: ", 6) && sendAll(s.client, line, len) && sendAll(s.client, "\n\n", 2);
    if (!ok) closeStream(s);
  }
}

void handleClient(WiFiClient c) {
  char req[160], hdr[160];
  if (!readHttpLine(c, req, sizeof req, 1000)) {
    c.stop();
    return;
  }
  int content_length = 0;
  while (readHttpLine(c, hdr, sizeof hdr, 1000) && hdr[0]) {
    if (strncasecmp(hdr, "Content-Length:", 15) == 0) content_length = atoi(hdr + 15);
  }

  if (strncmp(req, "GET / ", 6) == 0 || strncmp(req, "GET /index.html", 15) == 0) {
    const size_t len = sizeof DASHBOARD_HTML - 1;
    char head[160];
    snprintf(head, sizeof head,
             "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: %u\r\n"
             "Connection: close\r\n\r\n",
             (unsigned)len);
    sendText(c, head);
    sendAll(c, DASHBOARD_HTML, len);
    c.stop();
    return;
  }

  if (strncmp(req, "GET /events", 11) == 0) {
    for (Stream& s : g_stream) {
      if (s.used) continue;
      sendText(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                  "Connection: keep-alive\r\n\r\n");
      s.client = c;
      s.used = true;
      ++g_streams;
      g_console->inject("log json");  // the dashboard reads the JSON snapshots
      return;
    }
    sendText(c, "HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\ntoo many dashboards open\n");
    c.stop();
    return;
  }

  if (strncmp(req, "POST /cmd", 9) == 0) {
    char body[80];
    int n = 0;
    const uint32_t t0 = millis();
    while (n < content_length && n < (int)sizeof body - 1 && millis() - t0 < 1000) {
      const int ch = c.read();
      if (ch < 0) {
        vTaskDelay(1);
        continue;
      }
      if (ch == '\n' || ch == '\r') break;
      body[n++] = (char)ch;
    }
    body[n] = '\0';
    if (n > 0) {
      g_console->inject(body);
      char echo[84];
      const int len = snprintf(echo, sizeof echo, "> %s", body);
      if (g_lines) xRingbufferSend(g_lines, echo, (size_t)len, 0);
    }
    sendText(c, "HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n");
    c.stop();
    return;
  }

  sendText(c, "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n");
  c.stop();
}

void webTask(void*) {
  WiFiServer server(board::WEB_PORT);
  server.begin();
  server.setNoDelay(true);
  uint32_t last_ping = millis();
  for (;;) {
    WiFiClient c = ACCEPT_CLIENT(server);
    if (c) handleClient(c);
    size_t size = 0;
    char* item = (char*)xRingbufferReceive(g_lines, &size, pdMS_TO_TICKS(10));
    if (item) {
      forward(item, size, false);
      vRingbufferReturnItem(g_lines, item);
    }
    if (millis() - last_ping > 10000) {
      last_ping = millis();
      forward(nullptr, 0, true);
    }
  }
}

}  // namespace

void WebDashboard::begin(EspConsole& console) {
  g_console = &console;
  g_lines = xRingbufferCreate(16384, RINGBUF_TYPE_NOSPLIT);
  WiFi.mode(WIFI_AP);
  WiFi.softAP(board::WIFI_AP_SSID, board::WIFI_AP_PASSWORD);
  WiFi.setSleep(false);
  char msg[128];
  snprintf(msg, sizeof msg, "WiFi: join \"%s\" (password \"%s\"), then open http://%s\n", board::WIFI_AP_SSID,
           board::WIFI_AP_PASSWORD, WiFi.softAPIP().toString().c_str());
  console.print(msg);
  xTaskCreatePinnedToCore(webTask, "web", 8192, nullptr, 1, nullptr, 0);
}

bool WebDashboard::streaming() const { return g_streams > 0; }

void WebDashboard::publish(const char* line, int len) {
  if (g_lines) xRingbufferSend(g_lines, line, (size_t)len, 0);  // full -> dropped, never waits
}

#endif  // ARDUINO_ARCH_ESP32
