#include "fuzzy_speed.h"

#include "../config.h"

namespace {

const float INF = 1e9f;

// Trapezoid membership: 0 below a, rising to 1 at b, 1 until c, falling to 0 at d.
struct Trap {
  float a, b, c, d;
};

float trap(const Trap& t, float x) {
  if (x <= t.a || x >= t.d) return (x <= t.a && t.a <= -INF) || (x >= t.d && t.d >= INF) ? 1.0f : 0.0f;
  if (x < t.b) return (x - t.a) / (t.b - t.a);
  if (x <= t.c) return 1.0f;
  return (t.d - x) / (t.d - t.c);
}

// Interval membership: the upper curve has wider feet, the lower curve narrower
// feet and a slightly lower top. 'blur' is the half-width of the footprint.
void it2(const Trap& t, float x, float blur, float* lo, float* up) {
  Trap u = {t.a <= -INF ? t.a : t.a - blur, t.b, t.c, t.d >= INF ? t.d : t.d + blur};
  Trap l = {t.a <= -INF ? t.a : t.a + blur, t.b, t.c, t.d >= INF ? t.d : t.d - blur};
  if (l.a > l.b) l.a = l.b;
  if (l.d < l.c) l.d = l.c;
  *up = trap(u, x);
  *lo = blur > 0.0f ? cfg::FUZZY_LOWER_HEIGHT * trap(l, x) : trap(t, x);
  if (blur <= 0.0f) *up = *lo;
}

// Membership functions (principal curves) and footprint half-widths.
const Trap D_NEAR = {-INF, -INF, 12.0f, 30.0f};
const Trap D_MED = {15.0f, 35.0f, 50.0f, 80.0f};
const Trap D_FAR = {55.0f, 90.0f, INF, INF};
const Trap V_SLOW = {-INF, -INF, 10.0f, 30.0f};
const Trap V_FAST = {15.0f, 40.0f, INF, INF};
const Trap R_SMOOTH = {-INF, -INF, 0.2f, 0.5f};
const Trap R_ROUGH = {0.3f, 0.6f, INF, INF};
const float BLUR_D = 4.0f;   // cm: ultrasonic noise and edge uncertainty
const float BLUR_V = 5.0f;   // cm/s
const float BLUR_R = 0.08f;

struct Rule {
  int d, v, r;   // 0/1/2 = near/medium/far; 0/1 = slow/fast; 0/1 = smooth/rough
  float y;       // consequent speed factor
};

// Sorted by consequent (needed by the type reduction).
const Rule RULES[12] = {
    {0, 1, 0, 0.00f}, {0, 1, 1, 0.00f}, {0, 0, 1, 0.00f}, {0, 0, 0, 0.00f},
    {1, 1, 1, 0.30f}, {1, 1, 0, 0.35f}, {1, 0, 1, 0.45f}, {1, 0, 0, 0.55f},
    {2, 0, 1, 0.75f}, {2, 1, 1, 0.75f}, {2, 0, 0, 1.00f}, {2, 1, 0, 1.00f},
};

float minf(float a, float b) { return a < b ? a : b; }

}  // namespace

FuzzyResult fuzzySpeed(float distance_cm, float closing_cms, float rough, SpeedMode mode, float blur) {
  FuzzyResult res = {0.0f, 0.0f, 0.0f, 0.0f};
  if (mode == SpeedMode::FUZZY_T1) blur = 0.0f;

  float dl[3], du[3], vl[2], vu[2], rl[2], ru[2];
  const Trap* D[3] = {&D_NEAR, &D_MED, &D_FAR};
  const Trap* V[2] = {&V_SLOW, &V_FAST};
  const Trap* R[2] = {&R_SMOOTH, &R_ROUGH};
  for (int i = 0; i < 3; ++i) it2(*D[i], distance_cm, BLUR_D * blur, &dl[i], &du[i]);
  for (int i = 0; i < 2; ++i) it2(*V[i], closing_cms, BLUR_V * blur, &vl[i], &vu[i]);
  for (int i = 0; i < 2; ++i) it2(*R[i], rough, BLUR_R * blur, &rl[i], &ru[i]);

  float fl[12], fu[12], y[12];
  float total_up = 0.0f;
  for (int k = 0; k < 12; ++k) {
    const Rule& r = RULES[k];
    fl[k] = minf(dl[r.d], minf(vl[r.v], rl[r.r]));
    fu[k] = minf(du[r.d], minf(vu[r.v], ru[r.r]));
    y[k] = r.y;
    total_up += fu[k];
  }
  res.firing = total_up;
  if (total_up <= 1e-6f) return res;  // nothing fired: stop (safe)

  // Center-of-sets type reduction. With consequents sorted ascending:
  //   y_l = min over switch L of (sum_{k<=L} fu*y + sum_{k>L} fl*y) / (sum_{k<=L} fu + sum_{k>L} fl)
  //   y_r = max over switch R of (sum_{k<=R} fl*y + sum_{k>R} fu*y) / (sum_{k<=R} fl + sum_{k>R} fu)
  // Trying every switch point gives the exact Karnik-Mendel result.
  float y_l = 1e9f, y_r = -1e9f;
  for (int L = -1; L < 12; ++L) {
    float num_l = 0, den_l = 0, num_r = 0, den_r = 0;
    for (int k = 0; k < 12; ++k) {
      float wl = k <= L ? fu[k] : fl[k];
      float wr = k <= L ? fl[k] : fu[k];
      num_l += wl * y[k];
      den_l += wl;
      num_r += wr * y[k];
      den_r += wr;
    }
    if (den_l > 1e-9f && num_l / den_l < y_l) y_l = num_l / den_l;
    if (den_r > 1e-9f && num_r / den_r > y_r) y_r = num_r / den_r;
  }
  if (y_l > y_r) {  // can happen only when every lower firing is zero on one side
    float t = y_l;
    y_l = y_r;
    y_r = t;
  }
  res.y_l = y_l;
  res.y_r = y_r;
  res.u = 0.5f * (y_l + y_r);
  return res;
}

const char* speedModeName(SpeedMode m) {
  switch (m) {
    case SpeedMode::STEP: return "step";
    case SpeedMode::FUZZY_T1: return "fuzzy-t1";
    case SpeedMode::FUZZY_IT2: return "fuzzy-it2";
  }
  return "?";
}

void RangeTracker::update(float cm, float dt_s) {
  if (cm < 0.0f) {  // path clear
    valid_ = false;
    v_ = 0.0f;
    d_ = cfg::DIST_MAX_CM + 1.0f;
    return;
  }
  if (!valid_ || dt_s <= 0.0f) {
    d_ = cm;
    v_ = 0.0f;
    valid_ = true;
    return;
  }
  // Alpha-beta filter: predict, then correct with the residual.
  float pred = d_ - v_ * dt_s;
  float resid = cm - pred;
  d_ = pred + cfg::RANGE_ALPHA * resid;
  v_ = v_ - cfg::RANGE_BETA * resid / dt_s;
}
