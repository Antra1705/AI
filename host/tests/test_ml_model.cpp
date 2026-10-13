#include <cmath>

#include "core/ml_model.h"
#include "model_vectors.h"
#include "test.h"

TEST(quantize_rounds_half_away_and_clamps) {
  CHECK_EQ(quantizeS8(0.5f, 1.0f), 1);
  CHECK_EQ(quantizeS8(-0.5f, 1.0f), -1);
  CHECK_EQ(quantizeS8(0.49f, 1.0f), 0);
  CHECK_EQ(quantizeS8(1000.0f, 1.0f), 127);
  CHECK_EQ(quantizeS8(-1000.0f, 1.0f), -127);
  CHECK_EQ(quantizeS8(2.6f, 0.5f), 5);
}

TEST(qnet_hand_computed) {
  // 2 -> 2 -> 2 -> 4 network small enough to check by hand.
  static const int8_t w0[4] = {10, 0, 0, -10};
  static const int32_t b0[2] = {0, 0};
  static const int8_t w1[4] = {1, 0, 0, 1};
  static const int32_t b1[2] = {0, 0};
  static const int8_t w2[8] = {1, 0, 0, 1, 0, 0, 0, 0};
  static const int32_t b2[4] = {0, 0, 0, 5};
  QNet n = {{{2, 2, w0, b0, 0.1f, 0.01f}, {2, 2, w1, b1, 0.01f, 1.0f}, {2, 4, w2, b2, 0.01f, 1.0f}}};
  float x[2] = {1.0f, 1.0f}, z[4];
  qnetForward(n, x, z);
  // layer 0: q(x) = 10, 10 -> [100, -100] * 0.001 = [0.1, -0.1] -> ReLU [0.1, 0]
  // layer 1: q = [10, 0] -> [10, 0] * 0.01 = [0.1, 0]
  // layer 2: q = [10, 0] -> [10, 0, 0, 5] * 0.01
  CHECK_NEAR(z[0], 0.10, 1e-6);
  CHECK_NEAR(z[1], 0.0, 1e-6);
  CHECK_NEAR(z[3], 0.05, 1e-6);
}

TEST(no_model_is_uncertain) {
  MlEnsemble empty = {0, 0, 0, 0, 1.0f, 0, 0, 0, 0, 0, ""};
  float f[ML_FEATURE_COUNT] = {0};
  MlPrediction p = mlPredict(empty, f);
  CHECK(!p.valid);
  CHECK(p.uncertain);
  CHECK_NEAR(p.confidence, 0.25, 1e-6);
}

TEST(deployed_model_is_present) {
  const MlEnsemble& m = deployedModel();
  CHECK(m.members >= 1);
  CHECK(m.temperature > 0.0f);
  CHECK(m.conf_threshold > 0.25f && m.conf_threshold < 1.0f);
}

// The int8 code that runs on the ESP32 must reproduce the trainer's own
// (independently written, fake-quantized) forward pass on real test windows.
TEST(deployed_model_matches_trainer) {
  const MlEnsemble& m = deployedModel();
  for (int k = 0; k < kVectorCount; ++k) {
    MlPrediction p = mlPredict(m, kVectorFeatures[k]);
    for (int c = 0; c < ML_CLASSES; ++c) CHECK_NEAR(p.p[c], kVectorProbs[k][c], 5e-3);
    CHECK_NEAR(p.mutual_info, kVectorMI[k], 5e-3);
    CHECK_NEAR(p.novelty, kVectorNovelty[k], 1e-3 * kVectorNovelty[k] + 1e-3);
    float sum = 0;
    for (int c = 0; c < ML_CLASSES; ++c) sum += p.p[c];
    CHECK_NEAR(sum, 1.0, 1e-5);
  }
}

TEST(identical_members_have_zero_disagreement) {
  const MlEnsemble& m = deployedModel();
  MlEnsemble twins = m;
  QNet same[3] = {m.nets[0], m.nets[0], m.nets[0]};
  twins.members = 3;
  twins.nets = same;
  MlPrediction p = mlPredict(twins, kVectorFeatures[0]);
  CHECK_NEAR(p.mutual_info, 0.0, 1e-5);
}
