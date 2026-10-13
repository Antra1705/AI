#!/usr/bin/env python3
"""Report figures from the simulation results (needs matplotlib, e.g. the project .venv).

    .venv/bin/python tools/make_figures.py        # writes docs/figures/*.png

Reads docs/simulation_results.txt (host: sim all + sim tune) and config.h, so the
figures always match the numbers in the text.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "docs", "figures")
RESULTS = open(os.path.join(ROOT, "docs", "simulation_results.txt")).read()
CONFIG = open(os.path.join(ROOT, "adaptive_suspension", "src", "config.h")).read()

# Validated categorical order (light mode), chart chrome and status colours.
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIM = "#c3c2b7"
STATUS = {"LEARNING": MUTED, "HEALTHY": "#0ca30c", "WATCH": "#fab219", "SERVICE": "#d03b3b"}
CLASSES = ["Smooth", "Gravel", "Speed breaker", "Pothole"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def cfg(name):
    return float(re.search(r"\b%s\s*=\s*(-?[0-9.]+)" % name, CONFIG).group(1))


def section(title):
    i = RESULTS.index(title)
    body = RESULTS.index("\n====", i) + 1        # the ==== line under the title
    body = RESULTS.index("\n", body) + 1          # first line of the section
    j = RESULTS.find("\n====", body)              # start of the next section
    return RESULTS[i:j if j > 0 else None]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote docs/figures/" + name)


# ---------------------------------------------------------------- 1. response curve

def response_curve():
    targets = [cfg("STIFF_SMOOTH"), cfg("STIFF_GRAVEL"), cfg("STIFF_BREAKER"), cfg("STIFF_POTHOLE")]
    safe, lo, hi, risk = cfg("STIFF_SAFE"), cfg("CURVE_C_LOW"), cfg("CURVE_C_HIGH"), cfg("CURVE_RISK_GAIN")

    def stiffness(cls, c):
        p = [(1 - c) / 3] * 4
        p[cls] = c
        s_exp = sum(pi * t for pi, t in zip(p, targets))
        x = min(1, max(0, (c - lo) / (hi - lo)))
        g = x * x * (3 - 2 * x)
        return min(1, max(safe + g * (s_exp - safe), risk * (p[2] + p[3])))

    cs = [0.25 + i * 0.0075 for i in range(101)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(safe, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(0.995, safe + 0.02, "safe middle (90°)", color=INK2, fontsize=9, ha="right")
    # Breaker and pothole share the same target, so they are one curve.
    series = [(0, "Smooth"), (1, "Gravel"), (3, "Speed breaker / pothole")]
    for color, (k, name) in zip(S, series):
        ys = [stiffness(k, c) for c in cs]
        ax.plot(cs, ys, color=color, lw=2, label=name)
        ax.text(1.005, ys[-1], name, color=INK, fontsize=9, va="center")
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("confidence in the predicted surface")
    ax.set_ylabel("commanded stiffness (0 = 45°, 1 = 150°)")
    ax.set_title("Graded response: stiffness moves smoothly with confidence", loc="left")
    ax.legend(loc="lower left", fontsize=9)
    save(fig, "response_curve.png")


# ---------------------------------------------------------------- 2. damping characterisation

def damping_tune():
    sec = section("0. Damping characterisation")
    rows = {}
    for name in ["smooth", "gravel", "breaker", "pothole"]:
        m = re.search(r"^%s\s+(.+)$" % name, sec, re.M)
        rows[name] = [float(v) for v in m.group(1).split()]
    stiff = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    chosen = [cfg("STIFF_SMOOTH"), cfg("STIFF_GRAVEL"), cfg("STIFF_BREAKER"), cfg("STIFF_POTHOLE")]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for k, (ax, name) in enumerate(zip(axes, rows)):
        ax.plot(stiff, rows[name], color=S[0], lw=2, marker="o", ms=4)
        i = stiff.index(chosen[k])
        ax.plot([stiff[i]], [rows[name][i]], marker="o", ms=9, color=S[0], mec=SURFACE, mew=2)
        ax.annotate("chosen", (stiff[i], rows[name][i]), textcoords="offset points",
                    xytext=(8, 8) if stiff[i] < 0.5 else (-6, 9), ha="left" if stiff[i] < 0.5 else "right",
                    fontsize=8, color=INK)
        ax.set_title(CLASSES[k], loc="left")
        ax.set_xticks([0, 0.5, 1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("stiffness")
    axes[0].set_ylabel("cost (lower is better)")
    fig.suptitle("Damping characterisation: each surface wants a different setting", x=0.01, ha="left",
                 fontweight="bold", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "damping_characterisation.png")


# ---------------------------------------------------------------- 3. ride comparison

def ride():
    sec = section("1. Ride and damping")
    names, gravel, peak, bottoms = [], [], [], []
    for line in sec.splitlines():
        m = re.match(r"^(Passive soft|Passive stiff|SRA|Neural)(.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            names.append((m.group(1) + m.group(2)).strip().replace(" (deployed)", "\n(deployed)"))
            gravel.append(float(m.group(4)))
            peak.append(float(m.group(5)))
            bottoms.append(float(m.group(6)))
    colors = [S[0] if "deployed" in n else DIM for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, vals, title in zip(axes, [gravel, peak, bottoms],
                               ["Gravel vibration (RMS m/s²)", "Peak jolt at events (m/s²)", "Bottom-outs per event"]):
        y = range(len(names))
        ax.barh(y, vals, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v, yi, " %.2f" % v if v < 10 else " %.1f" % v, va="center", fontsize=8, color=INK)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.25)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=9, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle("Closed-loop ride comparison (lower is better); deployed design highlighted", x=0.01, ha="left",
                 fontweight="bold", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "ride_comparison.png")


# ---------------------------------------------------------------- 4. obstacle approach

def obstacle():
    sec = section("2. Obstacle speed control")
    modes = ["SRA steps (200/127/0)", "Type-1 fuzzy", "Interval Type-2 fuzzy"]
    data = {m: [] for m in modes}
    speeds = []
    for line in sec.splitlines():
        for m in modes:
            if line.startswith(m):
                rest = line[len(m):].split()
                if m == modes[0]:
                    speeds.append(rest[0])
                data[m].append(float(rest[2]))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w = 0.26
    for k, m in enumerate(modes):
        xs = [i + (k - 1) * w for i in range(len(speeds))]
        ax.bar(xs, data[m], width=w - 0.03, color=S[k], label=m)
        for x, v in zip(xs, data[m]):
            if v >= 0:
                ax.text(x, v + 0.5, "%.0f" % v, ha="center", fontsize=8, color=INK)
            else:
                ax.text(x + w / 2 + 0.02, v / 2, "%.0f\n(contact)" % v, ha="left", va="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK2, lw=1)
    ax.text(len(speeds) - 0.55, -1.2, "wall", color=INK2, fontsize=9, ha="right", va="top")
    ax.axhline(10, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(len(speeds) - 0.55, 10.4, "emergency-brake distance", color=INK2, fontsize=9, ha="right")
    ax.set_ylim(min(-6, min(min(v) for v in data.values()) - 2), 21)
    ax.set_xticks(range(len(speeds)))
    ax.set_xticklabels(["cruise %s m/s" % s for s in speeds])
    ax.set_ylabel("worst closest gap to the wall (cm)")
    ax.set_title("Obstacle approach: fuzzy control stops before the wall", loc="left", pad=28)
    ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    save(fig, "obstacle_approach.png")


# ---------------------------------------------------------------- 5. maintenance drift

def maintenance():
    sec = section("5. Predictive maintenance")
    days, damping, zeta, status = [], [], [], []
    for m in re.finditer(r"^\s+(\d+)\s+(\d+)%\s+([\d.]+)\s+(\w+)", sec, re.M):
        days.append(int(m.group(1)))
        damping.append(int(m.group(2)))
        zeta.append(float(m.group(3)))
        status.append(m.group(4))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    ax1.plot(days, damping, color=S[0], lw=2)
    ax1.set_ylabel("true damping (%)")
    ax1.set_title("Predictive maintenance: shock wear detected from the bounce", loc="left")
    for st in ["LEARNING", "HEALTHY", "WATCH", "SERVICE"]:
        xs = [d for d, s in zip(days, status) if s == st]
        ys = [z for z, s in zip(zeta, status) if s == st]
        marker = {"LEARNING": "o", "HEALTHY": "o", "WATCH": "^", "SERVICE": "s"}[st]
        ax2.scatter(xs, ys, color=STATUS[st], marker=marker, s=34, label=st.capitalize(), edgecolors=SURFACE,
                    linewidths=1, zorder=3)
    ax2.set_ylabel("measured damping ratio ζ")
    ax2.set_xlabel("day (one reference-bump test per day)")
    ax2.set_ylim(min(zeta) * 0.85, max(zeta) * 1.18)
    first = {s: next((d for d, x in zip(days, status) if x == s), None) for s in ("WATCH", "SERVICE")}
    for s, d in first.items():
        if d:
            for ax in (ax1, ax2):
                ax.axvline(d, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
            ax2.text(d + 0.3, max(zeta) * 1.15, "first %s\n(day %d, %d%%)" % (s.lower(), d, damping[days.index(d)]),
                     fontsize=8, color=INK, va="top")
    ax2.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    save(fig, "maintenance_drift.png")


if __name__ == "__main__":
    response_curve()
    damping_tune()
    ride()
    obstacle()
    maintenance()
