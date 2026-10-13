// replay - runs a recorded raw IMU log through the SAME code as the cart.
//
//   replay <raw.csv> [--expect surface] [--set name=value ...] [--ml] [--quiet]
//       Prints "F,..." lines (like the cart's 'log features') and a summary.
//       --ml      classify with the neural ensemble instead of the rules
//       --set     try rule thresholds: pothole_jerk breaker_p2p gravel_rms gravel_osc smooth_jerk smooth_p2p
//
//   replay <raw.csv> --label <surface> [--run N] [--event-p2p 2.0] [--header]
//       Prints training rows in the same format as 'sim dataset', so recordings
//       from the real cart can train the model (host/build/bin/train).
//       smooth/gravel recordings: every window gets the label.
//       breaker/pothole recordings: only windows whose peak-to-peak exceeds
//       --event-p2p get the label (the rest is road between the events, -1).
//       Give each recording its own --run number; train splits by run.
//
// Input: "R,t_us,ax,ay,az,gx,gy,gz" lines (from 'log raw' or 'sim record'); other lines ignored.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "config.h"
#include "core/classifier.h"
#include "core/features.h"
#include "core/imu_buffer.h"
#include "core/ml_features.h"
#include "core/ml_model.h"
#include "core/telemetry.h"

static bool applySetting(ClassifierThresholds& t, const char* arg) {
  const char* eq = std::strchr(arg, '=');
  if (!eq) return false;
  std::string name(arg, eq);
  double v = std::atof(eq + 1);
  if (name == "pothole_jerk") t.pothole_jerk = (float)v;
  else if (name == "breaker_p2p" || name == "bump_p2p") t.breaker_p2p = (float)v;
  else if (name == "gravel_rms") t.gravel_rms = (float)v;
  else if (name == "gravel_osc") t.gravel_osc = (int)v;
  else if (name == "smooth_jerk") t.smooth_jerk = (float)v;
  else if (name == "smooth_p2p") t.smooth_p2p = (float)v;
  else return false;
  return true;
}

static bool surfaceArg(const char* s, Surface* out) {
  if (!std::strcmp(s, "bump")) {
    *out = SURFACE_BREAKER;
    return true;
  }
  return parseSurface(s, out) && *out != SURFACE_UNKNOWN;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: replay <raw.csv> [--expect surface] [--set name=value ...] [--ml] [--quiet]\n"
                 "       replay <raw.csv> --label surface [--run N] [--event-p2p 2.0] [--header]\n");
    return 2;
  }
  ClassifierThresholds th = defaultThresholds();
  bool have_expect = false, quiet = false, use_ml = false, dataset = false, header = false;
  Surface expect = SURFACE_UNKNOWN, label = SURFACE_UNKNOWN;
  int run = 0;
  float event_p2p = 2.0f;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    bool has_next = i + 1 < argc;
    if (a == "--expect" && has_next) {
      if (!surfaceArg(argv[++i], &expect)) return std::fprintf(stderr, "unknown surface '%s'\n", argv[i]), 2;
      have_expect = true;
    } else if (a == "--set" && has_next) {
      if (!applySetting(th, argv[++i])) return std::fprintf(stderr, "bad setting '%s'\n", argv[i]), 2;
    } else if (a == "--label" && has_next) {
      if (!surfaceArg(argv[++i], &label)) return std::fprintf(stderr, "unknown surface '%s'\n", argv[i]), 2;
      dataset = true;
    } else if (a == "--run" && has_next) {
      run = std::atoi(argv[++i]);
    } else if (a == "--event-p2p" && has_next) {
      event_p2p = (float)std::atof(argv[++i]);
    } else if (a == "--header") {
      header = true;
    } else if (a == "--ml") {
      use_ml = true;
    } else if (a == "--quiet") {
      quiet = true;
    } else {
      std::fprintf(stderr, "unknown option '%s'\n", argv[i]);
      return 2;
    }
  }
  const MlEnsemble& model = deployedModel();
  if (use_ml && model.members == 0) {
    std::fprintf(stderr, "this build has no trained model; using the rules\n");
    use_ml = false;
  }

  FILE* in = std::fopen(argv[1], "r");
  if (!in) {
    std::perror(argv[1]);
    return 1;
  }

  ImuBuffer buf;
  int counts[SURFACE_COUNT] = {};
  int windows = 0, samples = 0, labelled = 0;
  uint64_t t64 = 0;  // unwrapped sample time
  uint32_t last_t = 0;
  uint64_t next_window = 0;
  char line[256];
  if (dataset && header) {
    std::printf("run,t,label,speed,servo");
    for (int k = 0; k < ML_FEATURE_COUNT; ++k) std::printf(",%s", mlFeatureName(k));
    std::printf(",rule_pred\n");
  } else if (!dataset && !quiet) {
    std::printf("%s\n", FEATURE_CSV_HEADER);
  }
  while (std::fgets(line, sizeof line, in)) {
    ImuSample s;
    if (!parseRawLine(line, &s)) continue;
    t64 = samples == 0 ? 0 : t64 + (uint32_t)(s.t_us - last_t);  // survives micros() wrap
    last_t = s.t_us;
    ++samples;
    buf.push(s);
    if (!buf.full()) continue;
    if (next_window == 0) next_window = t64;
    if (t64 < next_window) continue;
    next_window += cfg::CONTROL_PERIOD_MS * 1000ull;

    Features f = extractFeatures(buf);
    Classification c = classify(f, th);
    float mf[ML_FEATURE_COUNT];
    extractMlFeatures(buf, mf);
    if (use_ml) {
      MlPrediction p = mlPredict(model, mf);
      c.surface = p.uncertain ? SURFACE_UNKNOWN : (Surface)p.top;
      c.confidence = p.confidence;
    }
    ++counts[c.surface];
    ++windows;

    if (dataset) {
      bool event_class = label == SURFACE_BREAKER || label == SURFACE_POTHOLE;
      int lab = (!event_class || f.peak_to_peak > event_p2p) ? (int)label : -1;
      labelled += lab >= 0;
      Classification rc = classify(f, th);
      std::printf("%d,%.2f,%d,-1,-1", run, (double)t64 * 1e-6, lab);
      for (int k = 0; k < ML_FEATURE_COUNT; ++k) std::printf(",%.5g", (double)mf[k]);
      std::printf(",%d\n", rc.surface == SURFACE_UNKNOWN ? -1 : (int)rc.surface);
    } else if (!quiet) {
      char out[160];
      formatFeatures(out, sizeof out, (uint32_t)(t64 / 1000), f, c, servoAngleFor(c));
      std::fputs(out, stdout);
    }
  }
  std::fclose(in);

  std::fprintf(stderr, "\n%d samples, %d windows (1 s window every %lu ms), classifier: %s\n", samples, windows,
               (unsigned long)cfg::CONTROL_PERIOD_MS, use_ml ? "neural ensemble" : "rules");
  if (dataset) {
    std::fprintf(stderr, "%d windows labelled %s (run %d)\n", labelled, surfaceName(label), run);
    return samples > 0 ? 0 : 1;
  }
  for (int i = 0; i < SURFACE_COUNT; ++i) {
    if (counts[i] == 0) continue;
    std::fprintf(stderr, "  %-8s %5d  (%.1f%%)\n", surfaceName((Surface)i), counts[i],
                 windows ? 100.0 * counts[i] / windows : 0.0);
  }
  if (have_expect && windows > 0) {
    std::fprintf(stderr, "Classified as %s: %.1f%%\n", surfaceName(expect), 100.0 * counts[expect] / windows);
    if (expect == SURFACE_BREAKER || expect == SURFACE_POTHOLE)
      std::fprintf(stderr, "(event recordings are mostly smooth road between events, so a low %% is normal)\n");
  }
  return samples > 0 ? 0 : 1;
}
