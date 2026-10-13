// sim - runs the real control code (core/App) on a simulated cart.
//
// Evaluation (closed loop, physics model in vehicle.h):
//   sim tune                 damping characterisation per surface -> curve targets for config.h
//   sim road [runs=8]        damping strategies: comfort, protection, servo wear, reaction time
//   sim obstacle             SRA steps vs Type-1 vs interval Type-2 fuzzy near obstacles
//   sim faults               sensor failures while driving
//   sim ekf [runs=5]         classifier-adaptive EKF vs fixed-noise EKF vs accelerometer only
//   sim maint                predictive maintenance over 40 simulated days of shock wear
//   sim uncertain [runs=5]   behaviour on a surface never seen in training
//   sim all                  everything above (used for docs/simulation_results.md)
// Data:
//   sim dataset [runs=80] [seed=1]                 labelled feature windows for host/train
//   sim record <surface> <seconds> [seed]          raw IMU log, like 'log raw' on the cart
// Dashboard:
//   sim live [--speed 1.0]                         real-time loop; commands on stdin, telemetry on stdout
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "eval.h"
#include "scenario.h"

static void usage() {
  std::fprintf(stderr,
               "usage:\n"
               "  sim tune | road [runs] | obstacle | faults | ekf [runs] | maint | uncertain [runs] | all\n"
               "  sim dataset [runs] [seed]\n"
               "  sim record <smooth|gravel|breaker|pothole> <seconds> [seed]\n"
               "  sim live [--speed 1.0]\n");
}

static int argInt(int argc, char** argv, int i, int def) { return argc > i ? std::atoi(argv[i]) : def; }

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  std::string cmd = argv[1];
  if (cmd == "tune") return evalTune();
  if (cmd == "road") return evalRoad(argInt(argc, argv, 2, 8));
  if (cmd == "obstacle") return evalObstacle();
  if (cmd == "faults") return evalFaults();
  if (cmd == "ekf") return evalEkf(argInt(argc, argv, 2, 5));
  if (cmd == "maint") return evalMaintenance();
  if (cmd == "uncertain") return evalUncertainty(argInt(argc, argv, 2, 5));
  if (cmd == "all") {
    int rc = 0;
    const char* rule = "\n================================================================\n";
    std::printf("%s1. Ride and damping%s", rule, rule);
    rc |= evalRoad(argInt(argc, argv, 2, 8));
    std::printf("%s2. Obstacle speed control%s", rule, rule);
    rc |= evalObstacle();
    std::printf("%s3. Fail-safe behaviour%s", rule, rule);
    rc |= evalFaults();
    std::printf("%s4. State estimation (EKF)%s", rule, rule);
    rc |= evalEkf(5);
    std::printf("%s5. Predictive maintenance%s", rule, rule);
    rc |= evalMaintenance();
    std::printf("%s6. Uncertainty on an unseen surface%s", rule, rule);
    rc |= evalUncertainty(5);
    return rc;
  }
  if (cmd == "dataset") {
    DatasetOptions o;
    o.runs = argInt(argc, argv, 2, 80);
    o.seed = (unsigned)argInt(argc, argv, 3, 1);
    long rows = writeDataset(o, stdout);
    std::fprintf(stderr, "%ld windows\n", rows);
    return 0;
  }
  if (cmd == "record" && argc >= 4) return recordSurface(argv[2], std::atof(argv[3]), (unsigned)argInt(argc, argv, 4, 1));
  if (cmd == "live") {
    double speed = 1.0;
    if (argc >= 4 && !std::strcmp(argv[2], "--speed")) speed = std::atof(argv[3]);
    return runLive(speed > 0 ? speed : 1.0);
  }
  usage();
  return 2;
}
