#include <cmath>

#include "config.h"
#include "core/ekf.h"
#include "test.h"

static const float G = 9.81f;

static void gravityIn(float roll, float pitch, float f[3]) {
  f[0] = -G * std::sin(pitch);
  f[1] = G * std::sin(roll) * std::cos(pitch);
  f[2] = G * std::cos(roll) * std::cos(pitch);
}

TEST(ekf_jacobians_match_numerical_derivatives) {
  const float x[EKF_N] = {0.2f, -0.3f, 0.01f, -0.02f, 0.05f};
  const float w[3] = {0.3f, -0.4f, 0.5f};
  const float f[3] = {1.0f, -2.0f, 9.0f};
  float A[EKF_N][EKF_N], H[3][EKF_N];
  Ekf::processJacobian(x, w, f, A);
  Ekf::measureJacobian(x, H);
  const float h = 1e-3f;
  for (int j = 0; j < EKF_N; ++j) {
    float xp[EKF_N], xm[EKF_N];
    for (int k = 0; k < EKF_N; ++k) xp[k] = xm[k] = x[k];
    xp[j] += h;
    xm[j] -= h;
    float dp[EKF_N], dm[EKF_N], hp[3], hm[3];
    Ekf::process(xp, w, f, dp);
    Ekf::process(xm, w, f, dm);
    Ekf::measure(xp, hp);
    Ekf::measure(xm, hm);
    for (int i = 0; i < EKF_N; ++i) CHECK_NEAR(A[i][j], (dp[i] - dm[i]) / (2 * h), 2e-3);
    for (int i = 0; i < 3; ++i) CHECK_NEAR(H[i][j], (hp[i] - hm[i]) / (2 * h), 2e-3);
  }
}

TEST(ekf_initialises_from_gravity) {
  Ekf e;
  float f[3];
  gravityIn(0.1f, -0.2f, f);
  e.initFromAccel(f);
  CHECK_NEAR(e.roll(), 0.1, 1e-5);
  CHECK_NEAR(e.pitch(), -0.2, 1e-5);
}

TEST(ekf_learns_gyro_bias_when_still) {
  Ekf e;
  float f[3];
  gravityIn(0.05f, 0.08f, f);
  const float gyro[3] = {0.02f, -0.015f, 0.0f};  // pure bias, the cart is not rotating
  for (int i = 0; i < 6000; ++i) e.step(gyro, f, 0.01f, 1.0f);  // 60 s
  CHECK_NEAR(e.biasX(), 0.02, 0.003);
  CHECK_NEAR(e.biasY(), -0.015, 0.003);
  CHECK_NEAR(e.roll(), 0.05, 0.01);
  CHECK_NEAR(e.pitch(), 0.08, 0.01);
  CHECK_NEAR(e.vz(), 0.0, 0.01);
}

TEST(ekf_tracks_a_slow_pitch_change) {
  Ekf e;
  float f[3];
  gravityIn(0, 0, f);
  const float rate = 0.1f;  // rad/s for 2 s
  const float still[3] = {0, 0, 0};
  for (int i = 0; i < 100; ++i) e.step(still, f, 0.01f, 1.0f);
  for (int i = 1; i <= 200; ++i) {
    gravityIn(0, rate * i * 0.01f, f);
    const float w[3] = {0, rate, 0};
    e.step(w, f, 0.01f, 1.0f);
  }
  CHECK_NEAR(e.pitch(), 0.2, 0.01);
}

// The classifier link's mechanism: a larger noise scale means the filter is
// pulled less by an accelerometer reading that is not gravity. (Whether that
// helps overall is measured in closed loop by 'sim ekf'.)
TEST(ekf_noise_scale_reduces_pull_of_a_bad_reading) {
  auto pull = [](float r_scale) {
    Ekf e;
    float f[3];
    gravityIn(0, 0, f);
    e.initFromAccel(f);
    const float w[3] = {0, 0, 0};
    for (int i = 0; i < 200; ++i) e.step(w, f, 0.01f, r_scale);
    const float bump[3] = {3.0f, 0.0f, 9.81f};  // a jolt along x, not a real tilt
    e.step(w, bump, 0.01f, r_scale);
    return std::fabs(e.pitch());
  };
  CHECK(pull(cfg::EKF_R_GRAVEL) < pull(cfg::EKF_R_SMOOTH));
  CHECK(pull(cfg::EKF_R_POTHOLE) < 0.7f * pull(cfg::EKF_R_SMOOTH));  // gain falls ~1/sqrt(r)
}

TEST(ekf_noise_scale_from_probabilities) {
  const float smooth[4] = {1, 0, 0, 0}, pothole[4] = {0, 0, 0, 1}, mix[4] = {0.5f, 0.5f, 0, 0};
  CHECK_NEAR(ekfNoiseScale(smooth, true), cfg::EKF_R_SMOOTH, 1e-6);
  CHECK_NEAR(ekfNoiseScale(pothole, true), cfg::EKF_R_POTHOLE, 1e-6);
  CHECK_NEAR(ekfNoiseScale(mix, true), 0.5 * (cfg::EKF_R_SMOOTH + cfg::EKF_R_GRAVEL), 1e-5);
  CHECK_NEAR(ekfNoiseScale(smooth, false), cfg::EKF_R_UNKNOWN, 1e-6);
}

TEST(ekf_vertical_velocity_from_a_bounce) {
  Ekf e;
  float f[3];
  gravityIn(0, 0, f);
  e.initFromAccel(f);
  // 0.1 s of +5 m/s^2 upward acceleration -> about +0.5 m/s before the leak/pseudo-measurement pull it back
  for (int i = 0; i < 10; ++i) {
    const float fa[3] = {0, 0, G + 5.0f}, w[3] = {0, 0, 0};
    e.step(w, fa, 0.01f, 1.0f);
  }
  CHECK(e.vz() > 0.2f);
  for (int i = 0; i < 1000; ++i) {
    const float w[3] = {0, 0, 0};
    e.step(w, f, 0.01f, 1.0f);
  }
  CHECK_NEAR(e.vz(), 0.0, 0.02);
}
