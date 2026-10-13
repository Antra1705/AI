#!/usr/bin/env python3
"""Suggest thresholds for the rule-based classifier (the simple fallback).

From recordings of each surface (serial output of the cart, or `sim record`):
    python3 tools/calibrate.py smooth=smooth.csv gravel=gravel.csv breaker=breaker.csv pothole=pothole.csv

From a labelled dataset (output of `sim dataset`, or of `replay --label`), using
only the training runs (the same split the neural network uses):
    python3 tools/calibrate.py --dataset host/data/dataset.csv

Recording files may hold 'log features' output ("F,..." lines), used directly,
or 'log raw' output ("R,..." lines), which is first run through
host/build/bin/replay so the feature code is exactly the cart's.

Recording tips:
  - smooth / gravel: drive only on that surface for 30-60 s.
  - breaker / pothole: drive over it repeatedly with smooth road in between;
    only the windows containing an event are used.

Only the Python standard library is needed.
"""

import csv
import math
import os
import subprocess
import sys

SURFACES = ["smooth", "gravel", "breaker", "pothole"]
ALIASES = {"bump": "breaker"}
FEATURES = ["jerk_max", "peak_to_peak", "accel_rms_dev", "oscillations"]
REPLAY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "host", "build", "bin", "replay")


def percentile(values, p):
    if not values:
        return float("nan")
    v = sorted(values)
    k = (len(v) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


# ---------------------------------------------------------------- loading

def parse_feature_lines(lines):
    """'F,t_ms,mean,jerk,p2p,rms,osc,gyro,...' lines -> list of feature dicts."""
    rows = []
    for line in lines:
        if not line.startswith("F,"):
            continue
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        try:
            rows.append({"jerk_max": float(parts[3]), "peak_to_peak": float(parts[4]),
                         "accel_rms_dev": float(parts[5]), "oscillations": float(parts[6])})
        except ValueError:
            continue
    return rows


def load_recording(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    rows = parse_feature_lines(lines)
    if rows:
        return rows
    if not any(line.startswith("R,") for line in lines):
        sys.exit(f"{path}: no 'F,' or 'R,' lines found")
    if not os.path.exists(REPLAY):
        sys.exit(f"{path} holds raw data; build the replay tool first:  make -C host replay")
    out = subprocess.run([REPLAY, path], capture_output=True, text=True, check=True).stdout
    return parse_feature_lines(out.splitlines())


def load_dataset(path):
    """Dataset rows (training runs only) -> {surface: [feature dicts]}. Event windows are labelled directly."""
    data = {s: [] for s in SURFACES}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            label = int(r["label"])
            if label < 0 or label > 3 or int(r["run"]) % 20 >= 14:
                continue
            data[SURFACES[label]].append({
                "jerk_max": math.exp(float(r["log_jerk_max"])) - 1.0,
                "peak_to_peak": math.exp(float(r["log_p2p"])) - 1e-3,
                "accel_rms_dev": math.exp(float(r["log_rms"])) - 1e-3,
                "oscillations": float(r["swings"]),
            })
    return data


# ---------------------------------------------------------------- thresholds

def c_float(v):
    """Format as a valid C++ float literal, e.g. 636 -> 636.0f."""
    text = f"{v:.3g}"
    if "." not in text and "e" not in text:
        text += ".0"
    return text + "f"


def split_point(below, above, name, warnings, integer=False):
    """Threshold halfway between the class that must stay under it and the one that must go over."""
    if below >= above:
        warnings.append(f"{name}: classes overlap (must stay under {below:.3g}, must exceed {above:.3g}); "
                        "the rule-based classifier will confuse them")
        value = above
    else:
        value = (below + above) / 2.0
    if integer:
        value = math.ceil(value)
    return value


def suggest(data):
    def p(s, feat, q):
        return percentile([r[feat] for r in data[s]], q)

    warnings = []
    th = {
        # gravel is checked before pothole, so this only separates potholes from smooth/breakers
        "pothole_jerk": split_point(max(p("smooth", "jerk_max", 99), p("breaker", "jerk_max", 95)),
                                    p("pothole", "jerk_max", 10), "pothole_jerk", warnings),
        "gravel_rms": split_point(p("smooth", "accel_rms_dev", 99), p("gravel", "accel_rms_dev", 5),
                                  "gravel_rms", warnings),
        "gravel_osc": split_point(max(p("breaker", "oscillations", 90), p("pothole", "oscillations", 90)),
                                  p("gravel", "oscillations", 10),
                                  "gravel_osc", warnings, integer=True),
        "breaker_p2p": split_point(p("smooth", "peak_to_peak", 99), p("breaker", "peak_to_peak", 5),
                                   "breaker_p2p", warnings),
        "smooth_jerk": split_point(p("smooth", "jerk_max", 99),
                                   min(p("gravel", "jerk_max", 5), p("breaker", "jerk_max", 5)),
                                   "smooth_jerk", warnings),
        "smooth_p2p": split_point(p("smooth", "peak_to_peak", 99),
                                  min(p("gravel", "peak_to_peak", 5), p("breaker", "peak_to_peak", 5)),
                                  "smooth_p2p", warnings),
    }
    return th, warnings


def main(argv):
    if len(argv) == 2 and argv[0] == "--dataset":
        data = load_dataset(argv[1])
        source = argv[1]
    else:
        files = {}
        for arg in argv:
            if "=" not in arg:
                sys.exit(__doc__)
            name, path = arg.split("=", 1)
            name = ALIASES.get(name, name)
            if name not in SURFACES:
                sys.exit(f"unknown surface '{name}' (use {', '.join(SURFACES)})")
            files[name] = path
        missing = [s for s in SURFACES if s not in files]
        if missing:
            sys.exit(f"need recordings for all four surfaces; missing: {', '.join(missing)}")
        data = {s: load_recording(p) for s, p in files.items()}
        # Event recordings contain smooth road between events: keep only event windows.
        smooth_hi = percentile([r["peak_to_peak"] for r in data["smooth"]], 99)
        cut = max(3 * smooth_hi, smooth_hi + 0.5)
        for s in ("breaker", "pothole"):
            events = [r for r in data[s] if r["peak_to_peak"] > cut]
            if len(events) < 5:
                sys.exit(f"{s}: found only {len(events)} event windows (p2p > {cut:.2f}); drive over it more times")
            data[s] = events
        source = ", ".join(files.values())

    for s, rows in data.items():
        if len(rows) < 20:
            sys.exit(f"{s}: only {len(rows)} windows - need more data")

    print(f"Source: {source}")
    print("Windows used: " + ", ".join(f"{s} {len(data[s])}" for s in SURFACES))
    print()
    print(f"{'feature':<15}{'surface':<9}{'p1':>9}{'p5':>9}{'p50':>9}{'p95':>9}{'p99':>9}")
    for feat in FEATURES:
        for s in SURFACES:
            v = [r[feat] for r in data[s]]
            print(f"{feat:<15}{s:<9}" + "".join(f"{percentile(v, q):9.3g}" for q in (1, 5, 50, 95, 99)))
        print()

    th, warnings = suggest(data)
    names = {"pothole_jerk": "POTHOLE_JERK", "breaker_p2p": "BREAKER_P2P", "gravel_rms": "GRAVEL_RMS",
             "gravel_osc": "GRAVEL_OSC", "smooth_jerk": "SMOOTH_JERK", "smooth_p2p": "SMOOTH_P2P"}
    print("Suggested thresholds (paste into adaptive_suspension/src/config.h):")
    print()
    for key in ("pothole_jerk", "breaker_p2p", "gravel_rms", "gravel_osc", "smooth_jerk", "smooth_p2p"):
        v = th[key]
        if isinstance(v, int):
            print(f"constexpr int   {names[key]:<12} = {v};")
        else:
            print(f"constexpr float {names[key]:<12} = {c_float(v)};")
    print(f"constexpr float {'IMPACT_JERK':<12} = {c_float(th['pothole_jerk'])};")
    if warnings:
        print()
        print("WARNINGS (expected: fixed thresholds cannot separate overlapping surfaces;")
        print("the neural-network classifier is the main classifier, these rules are the fallback):")
        for w in warnings:
            print("  - " + w)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
