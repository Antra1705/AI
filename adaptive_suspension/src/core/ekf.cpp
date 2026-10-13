#include "ekf.h"

#include <math.h>

#include "../config.h"

namespace {

const float G = 9.81f;
const int N = EKF_N;

void matMul(const float a[N][N], const float b[N][N], float out[N][N]) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.0f;
      for (int k = 0; k < N; ++k) s += a[i][k] * b[k][j];
      out[i][j] = s;
    }
}

void symmetrize(float P[N][N]) {
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j) {
      float m = 0.5f * (P[i][j] + P[j][i]);
      P[i][j] = m;
      P[j][i] = m;
    }
}

bool invertS(const float (&S)[1][1], float (&Si)[1][1]) {
  if (fabsf(S[0][0]) < 1e-12f) return false;
  Si[0][0] = 1.0f / S[0][0];
  return true;
}

bool invertS(const float (&S)[3][3], float (&Si)[3][3]) {
  float det = S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1]) - S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0]) +
              S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]);
  if (fabsf(det) < 1e-12f) return false;
  float id = 1.0f / det;
  Si[0][0] = (S[1][1] * S[2][2] - S[1][2] * S[2][1]) * id;
  Si[0][1] = (S[0][2] * S[2][1] - S[0][1] * S[2][2]) * id;
  Si[0][2] = (S[0][1] * S[1][2] - S[0][2] * S[1][1]) * id;
  Si[1][0] = (S[1][2] * S[2][0] - S[1][0] * S[2][2]) * id;
  Si[1][1] = (S[0][0] * S[2][2] - S[0][2] * S[2][0]) * id;
  Si[1][2] = (S[0][2] * S[1][0] - S[0][0] * S[1][2]) * id;
  Si[2][0] = (S[1][0] * S[2][1] - S[1][1] * S[2][0]) * id;
  Si[2][1] = (S[0][1] * S[2][0] - S[0][0] * S[2][1]) * id;
  Si[2][2] = (S[0][0] * S[1][1] - S[0][1] * S[1][0]) * id;
  return true;
}

// Generic Kalman update with an M-dimensional measurement (M = 1 or 3), Joseph form.
template <int M>
void kalmanUpdate(float x[N], float P[N][N], const float H[M][N], const float innov[M], const float R[M]) {
  float PHt[N][M];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) {
      float s = 0.0f;
      for (int k = 0; k < N; ++k) s += P[i][k] * H[j][k];
      PHt[i][j] = s;
    }
  float S[M][M];
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < M; ++j) {
      float s = (i == j) ? R[i] : 0.0f;
      for (int k = 0; k < N; ++k) s += H[i][k] * PHt[k][j];
      S[i][j] = s;
    }
  float Si[M][M];
  if (!invertS(S, Si)) return;
  float K[N][M];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) {
      float s = 0.0f;
      for (int k = 0; k < M; ++k) s += PHt[i][k] * Si[k][j];
      K[i][j] = s;
    }
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) x[i] += K[i][j] * innov[j];

  // Joseph form: P = (I - KH) P (I - KH)^T + K R K^T  (stays symmetric positive definite)
  float IKH[N][N];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      float s = (i == j) ? 1.0f : 0.0f;
      for (int k = 0; k < M; ++k) s -= K[i][k] * H[k][j];
      IKH[i][j] = s;
    }
  float T[N][N], IKHt[N][N], Pn[N][N];
  matMul(IKH, P, T);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) IKHt[i][j] = IKH[j][i];
  matMul(T, IKHt, Pn);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.0f;
      for (int k = 0; k < M; ++k) s += K[i][k] * R[k] * K[j][k];
      P[i][j] = Pn[i][j] + s;
    }
  symmetrize(P);
}

}  // namespace

void Ekf::reset() {
  for (int i = 0; i < N; ++i) {
    x_[i] = 0.0f;
    for (int j = 0; j < N; ++j) P_[i][j] = 0.0f;
  }
  P_[EKF_ROLL][EKF_ROLL] = 0.1f;
  P_[EKF_PITCH][EKF_PITCH] = 0.1f;
  P_[EKF_BGX][EKF_BGX] = 0.03f * 0.03f;
  P_[EKF_BGY][EKF_BGY] = 0.03f * 0.03f;
  P_[EKF_VZ][EKF_VZ] = 0.01f;
  az_world_ = 0.0f;
  init_ = false;
}

void Ekf::initFromAccel(const float f[3]) {
  reset();
  x_[EKF_ROLL] = atan2f(f[1], f[2]);
  x_[EKF_PITCH] = atan2f(-f[0], sqrtf(f[1] * f[1] + f[2] * f[2]));
  init_ = true;
}

void Ekf::process(const float x[N], const float w[3], const float f[3], float xd[N]) {
  const float sr = sinf(x[EKF_ROLL]), cr = cosf(x[EKF_ROLL]);
  const float sp = sinf(x[EKF_PITCH]), cp = cosf(x[EKF_PITCH]), tp = sp / cp;
  const float wx = w[0] - x[EKF_BGX], wy = w[1] - x[EKF_BGY], wz = w[2];
  xd[EKF_ROLL] = wx + sr * tp * wy + cr * tp * wz;
  xd[EKF_PITCH] = cr * wy - sr * wz;
  xd[EKF_BGX] = 0.0f;
  xd[EKF_BGY] = 0.0f;
  const float az = -sp * f[0] + sr * cp * f[1] + cr * cp * f[2] - G;
  xd[EKF_VZ] = az - cfg::EKF_VZ_LEAK * x[EKF_VZ];
}

void Ekf::processJacobian(const float x[N], const float w[3], const float f[3], float A[N][N]) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) A[i][j] = 0.0f;
  const float sr = sinf(x[EKF_ROLL]), cr = cosf(x[EKF_ROLL]);
  const float sp = sinf(x[EKF_PITCH]), cp = cosf(x[EKF_PITCH]), tp = sp / cp;
  const float wy = w[1] - x[EKF_BGY], wz = w[2];
  A[EKF_ROLL][EKF_ROLL] = cr * tp * wy - sr * tp * wz;
  A[EKF_ROLL][EKF_PITCH] = (sr * wy + cr * wz) / (cp * cp);
  A[EKF_ROLL][EKF_BGX] = -1.0f;
  A[EKF_ROLL][EKF_BGY] = -sr * tp;
  A[EKF_PITCH][EKF_ROLL] = -sr * wy - cr * wz;
  A[EKF_PITCH][EKF_BGY] = -cr;
  A[EKF_VZ][EKF_ROLL] = cr * cp * f[1] - sr * cp * f[2];
  A[EKF_VZ][EKF_PITCH] = -cp * f[0] - sr * sp * f[1] - cr * sp * f[2];
  A[EKF_VZ][EKF_VZ] = -cfg::EKF_VZ_LEAK;
}

void Ekf::measure(const float x[N], float h[3]) {
  const float sr = sinf(x[EKF_ROLL]), cr = cosf(x[EKF_ROLL]);
  const float sp = sinf(x[EKF_PITCH]), cp = cosf(x[EKF_PITCH]);
  h[0] = -G * sp;
  h[1] = G * sr * cp;
  h[2] = G * cr * cp;
}

void Ekf::measureJacobian(const float x[N], float H[3][N]) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < N; ++j) H[i][j] = 0.0f;
  const float sr = sinf(x[EKF_ROLL]), cr = cosf(x[EKF_ROLL]);
  const float sp = sinf(x[EKF_PITCH]), cp = cosf(x[EKF_PITCH]);
  H[0][EKF_PITCH] = -G * cp;
  H[1][EKF_ROLL] = G * cr * cp;
  H[1][EKF_PITCH] = -G * sr * sp;
  H[2][EKF_ROLL] = -G * sr * cp;
  H[2][EKF_PITCH] = -G * cr * sp;
}

void Ekf::step(const float gyro[3], const float accel[3], float dt, float r_scale) {
  if (!init_) {
    initFromAccel(accel);
    return;
  }
  // --- predict
  float xd[N], A[N][N];
  process(x_, gyro, accel, xd);
  processJacobian(x_, gyro, accel, A);
  az_world_ = xd[EKF_VZ] + cfg::EKF_VZ_LEAK * x_[EKF_VZ];
  for (int i = 0; i < N; ++i) x_[i] += xd[i] * dt;
  const float lim = 1.4f;  // keep pitch away from +/-90 deg where Euler angles break down
  if (x_[EKF_PITCH] > lim) x_[EKF_PITCH] = lim;
  if (x_[EKF_PITCH] < -lim) x_[EKF_PITCH] = -lim;

  float F[N][N], FP[N][N], Ft[N][N], Pn[N][N];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) F[i][j] = (i == j ? 1.0f : 0.0f) + A[i][j] * dt;
  matMul(F, P_, FP);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) Ft[i][j] = F[j][i];
  matMul(FP, Ft, Pn);
  const float q[N] = {cfg::EKF_GYRO_NOISE * cfg::EKF_GYRO_NOISE, cfg::EKF_GYRO_NOISE * cfg::EKF_GYRO_NOISE,
                      cfg::EKF_BIAS_WALK * cfg::EKF_BIAS_WALK, cfg::EKF_BIAS_WALK * cfg::EKF_BIAS_WALK,
                      cfg::EKF_ACCEL_NOISE * cfg::EKF_ACCEL_NOISE * r_scale};
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) P_[i][j] = Pn[i][j] + (i == j ? q[i] * dt : 0.0f);
  symmetrize(P_);

  // --- updates
  gravityUpdate(accel, r_scale);
  zeroVelocityUpdate();
}

void Ekf::gravityUpdate(const float f[3], float r_scale) {
  float h[3], H[3][N];
  measure(x_, h);
  measureJacobian(x_, H);
  // Inflate R when |a| is far from 1 g: the body is accelerating, so the
  // accelerometer is not measuring gravity alone.
  const float mag = sqrtf(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
  const float dyn = (mag - G) / cfg::EKF_DYNAMIC_ACCEL;
  const float var = cfg::EKF_ACCEL_NOISE * cfg::EKF_ACCEL_NOISE * r_scale * (1.0f + dyn * dyn);
  const float R[3] = {var, var, var};
  const float innov[3] = {f[0] - h[0], f[1] - h[1], f[2] - h[2]};
  kalmanUpdate<3>(x_, P_, H, innov, R);
}

void Ekf::zeroVelocityUpdate() {
  const float H[1][N] = {{0.0f, 0.0f, 0.0f, 0.0f, 1.0f}};
  const float innov[1] = {-x_[EKF_VZ]};
  const float R[1] = {cfg::EKF_ZERO_VZ_NOISE * cfg::EKF_ZERO_VZ_NOISE};
  kalmanUpdate<1>(x_, P_, H, innov, R);
}

float ekfNoiseScale(const float p[4], bool classifier_ready) {
  if (!classifier_ready) return cfg::EKF_R_UNKNOWN;
  const float factor[4] = {cfg::EKF_R_SMOOTH, cfg::EKF_R_GRAVEL, cfg::EKF_R_BREAKER, cfg::EKF_R_POTHOLE};
  float s = 0.0f;
  for (int c = 0; c < 4; ++c) s += p[c] * factor[c];
  return s;
}
