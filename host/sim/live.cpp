// Live mode: runs the simulated cart in real time for the dashboard.
//
// stdin   operator commands, exactly as typed into the cart's serial monitor
//         ("start", "stop", "maint test", "speed t1", ...), plus simulator controls:
//           sim wall [m]     put an obstacle m metres ahead (default 1.5)
//           sim clear        remove the obstacle
//           sim wear <0-1>   shock condition (1 = healthy)
//           sim imu off|on   unplug / replug the IMU
//           sim sonar off|on unplug / replug the ultrasonic sensor
//           sim auto on|off  obstacles appear by themselves every ~25 s (default on)
//         'maint test' also lays a smooth stretch with the 1.5 cm reference bump just
//         ahead of the cart (on the real cart you drive it over your reference bump).
// stdout  everything the cart prints (J,{...} JSON at 10 Hz, event lines), plus
//         S,{...} lines with the simulator's ground truth for the dashboard.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include "eval.h"
#include "scenario.h"
#include "sim_cart.h"

namespace {

const char* trueSurface(const Track& t, double x) {
  for (const TrackFeature& f : t.features()) {
    if (x < f.x0 || x > f.x1) continue;
    switch (f.kind) {
      case TrackKind::BREAKER: return "BREAKER";
      case TrackKind::POTHOLE: return "POTHOLE";
      case TrackKind::GRAVEL: return "GRAVEL";
      case TrackKind::RUMBLE: return "RUMBLE (unseen)";
    }
  }
  return "SMOOTH";
}

}  // namespace

int runLive(double speed) {
  std::mutex mu;
  std::deque<std::string> pending;
  std::atomic<bool> quit(false);
  std::thread reader([&] {
    std::string line;
    while (std::getline(std::cin, line)) {
      std::lock_guard<std::mutex> lock(mu);
      pending.push_back(line);
    }
    quit = true;
  });
  reader.detach();

  TrackOptions to;
  to.length_m = 3000;
  Track track = randomTrack(424242, to);
  SimCart cart(track, VehicleParams(), 424242);
  cart.begin();
  cart.command("log json");

  bool auto_obstacles = true;
  double next_auto = 20.0, stopped_since = -1;
  size_t printed = 0;
  auto start = std::chrono::steady_clock::now();
  long tick = 0;

  while (!quit) {
    {
      std::lock_guard<std::mutex> lock(mu);
      while (!pending.empty()) {
        std::string line = pending.front();
        pending.pop_front();
        if (line.compare(0, 4, "sim ") != 0) {
          if (line == "maint test" && cart.app.state() == RunState::IDLE) {
            double from = cart.car.frontX() + 0.05;
            cart.track.placeReferenceBump(from, from + 6.0, from + 1.2, 0.015, 0.10);
            cart.wall_x = -1;
            std::printf("# sim: reference bump placed 1.2 m ahead\n");
          }
          cart.command(line);
          continue;
        }
        std::istringstream in(line.substr(4));
        std::string what, arg;
        in >> what >> arg;
        if (what == "wall") cart.wall_x = cart.car.truth().x + 0.10 + (arg.empty() ? 1.5 : std::atof(arg.c_str()));
        else if (what == "clear") cart.wall_x = -1;
        else if (what == "wear" && !arg.empty()) cart.car.params().wear = std::atof(arg.c_str());
        else if (what == "imu") cart.imu_unplugged = arg == "off";
        else if (what == "sonar") cart.ultrasonic_unplugged = arg == "off";
        else if (what == "auto") auto_obstacles = arg != "off";
        std::printf("# sim: %s\n", line.c_str());
      }
    }

    cart.runFor(0.01);
    ++tick;

    // Obstacles that appear and are taken away again, so the approach logic shows up on its own.
    double t = cart.time();
    if (cart.app.state() == RunState::MAINT) next_auto = std::max(next_auto, t + 10.0);  // no obstacles during the test
    if (auto_obstacles && cart.wall_x < 0 && t > next_auto && cart.car.truth().v > 0.1) {
      cart.wall_x = cart.car.truth().x + 0.10 + 1.6;
      std::printf("# sim: obstacle placed 1.6 m ahead\n");
    }
    if (cart.wall_x >= 0) {
      if (cart.car.truth().v < 0.005) {
        if (stopped_since < 0) stopped_since = t;
        if (t - stopped_since > 3.0 && auto_obstacles) {
          cart.wall_x = -1;
          stopped_since = -1;
          next_auto = t + 20.0 + 10.0 * ((tick * 7919) % 100) / 100.0;
          std::printf("# sim: obstacle removed\n");
        }
      } else {
        stopped_since = -1;
      }
    }

    if (tick % 10 == 0) {
      const VehicleTruth& tr = cart.car.truth();
      std::printf("S,{\"t\":%.2f,\"x\":%.2f,\"v\":%.3f,\"true_surface\":\"%s\",\"heave\":%.3f,\"defl\":%.4f,"
                  "\"wear\":%.2f,\"gap\":%.1f,\"wall\":%d,\"true_pitch\":%.2f,\"true_roll\":%.2f,\"true_vz\":%.4f,"
                  "\"bottom_outs\":%d}\n",
                  t, tr.x, tr.v, trueSurface(cart.track, cart.car.frontX()), tr.heave_acc,
                  std::max(std::fabs(tr.defl_front), std::fabs(tr.defl_rear)), cart.car.params().wear,
                  cart.wall_x < 0 ? -1.0 : cart.gapCm(), cart.wall_x < 0 ? 0 : 1, tr.pitch * 57.2958,
                  tr.roll * 57.2958, tr.vz, cart.car.bottomOuts());
    }
    std::string& out = cart.rig.console.output;
    if (out.size() > printed) {
      std::fwrite(out.data() + printed, 1, out.size() - printed, stdout);
      printed = out.size();
    }
    if (printed > (1u << 20)) {  // keep memory bounded
      out.clear();
      printed = 0;
    }
    std::fflush(stdout);

    // Real-time pacing.
    auto target = start + std::chrono::microseconds((long long)(t * 1e6 / speed));
    std::this_thread::sleep_until(target);
  }
  return 0;
}
