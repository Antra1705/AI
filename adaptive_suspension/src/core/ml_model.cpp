#include "ml_model.h"

#include <math.h>

int8_t quantizeS8(float x, float scale) {
  float v = x / scale;
  v = v >= 0.0f ? floorf(v + 0.5f) : ceilf(v - 0.5f);
  if (v > 127.0f) v = 127.0f;
  if (v < -127.0f) v = -127.0f;
  return (int8_t)v;
}

void qnetForward(const QNet& net, const float x[], float logits[ML_CLASSES]) {
  float act[ML_MAX_WIDTH];
  float next[ML_MAX_WIDTH];
  for (int i = 0; i < net.layer[0].in; ++i) act[i] = x[i];

  for (int l = 0; l < ML_LAYERS; ++l) {
    const QLayer& L = net.layer[l];
    int8_t q[ML_MAX_WIDTH];
    for (int i = 0; i < L.in; ++i) q[i] = quantizeS8(act[i], L.in_scale);
    const float rescale = L.in_scale * L.w_scale;
    for (int o = 0; o < L.out; ++o) {
      int32_t acc = L.b[o];
      const int8_t* row = L.w + o * L.in;
      for (int i = 0; i < L.in; ++i) acc += (int32_t)row[i] * (int32_t)q[i];
      float y = (float)acc * rescale;
      next[o] = (l < ML_LAYERS - 1 && y < 0.0f) ? 0.0f : y;  // ReLU on hidden layers
    }
    for (int o = 0; o < L.out; ++o) act[o] = next[o];
  }
  for (int c = 0; c < ML_CLASSES; ++c) logits[c] = act[c];
}

static float entropyOf(const float p[ML_CLASSES]) {
  float h = 0.0f;
  for (int c = 0; c < ML_CLASSES; ++c)
    if (p[c] > 1e-7f) h -= p[c] * logf(p[c]);
  return h;
}

float mlNovelty(const MlEnsemble& m, const float x[ML_FEATURE_COUNT]) {
  if (!m.class_means || !m.inv_cov) return 0.0f;
  float best = 1e30f;
  for (int c = 0; c < ML_CLASSES; ++c) {
    float d[ML_FEATURE_COUNT];
    for (int i = 0; i < ML_FEATURE_COUNT; ++i) d[i] = x[i] - m.class_means[c * ML_FEATURE_COUNT + i];
    float q = 0.0f;
    for (int i = 0; i < ML_FEATURE_COUNT; ++i) {
      float row = 0.0f;
      const float* ic = m.inv_cov + c * ML_FEATURE_COUNT * ML_FEATURE_COUNT;
      for (int j = 0; j < ML_FEATURE_COUNT; ++j) row += ic[i * ML_FEATURE_COUNT + j] * d[j];
      q += d[i] * row;
    }
    if (q < best) best = q;
  }
  return best;
}

MlPrediction mlPredict(const MlEnsemble& m, const float features[ML_FEATURE_COUNT]) {
  MlPrediction r;
  r.valid = m.members > 0;
  for (int c = 0; c < ML_CLASSES; ++c) r.p[c] = 1.0f / (float)ML_CLASSES;
  r.top = 0;
  r.confidence = 1.0f / (float)ML_CLASSES;
  r.entropy = logf((float)ML_CLASSES);
  r.mutual_info = 0.0f;
  r.novelty = 0.0f;
  r.uncertain = true;
  if (!r.valid) return r;

  float x[ML_FEATURE_COUNT];
  for (int i = 0; i < ML_FEATURE_COUNT; ++i) x[i] = (features[i] - m.feat_mean[i]) / m.feat_std[i];

  float mean_p[ML_CLASSES] = {0.0f, 0.0f, 0.0f, 0.0f};
  float mean_member_entropy = 0.0f;
  for (int k = 0; k < m.members; ++k) {
    float z[ML_CLASSES], p[ML_CLASSES];
    qnetForward(m.nets[k], x, z);
    float zmax = z[0];
    for (int c = 1; c < ML_CLASSES; ++c)
      if (z[c] > zmax) zmax = z[c];
    float sum = 0.0f;
    for (int c = 0; c < ML_CLASSES; ++c) {
      p[c] = expf((z[c] - zmax) / m.temperature);
      sum += p[c];
    }
    for (int c = 0; c < ML_CLASSES; ++c) {
      p[c] /= sum;
      mean_p[c] += p[c] / (float)m.members;
    }
    mean_member_entropy += entropyOf(p) / (float)m.members;
  }

  for (int c = 0; c < ML_CLASSES; ++c) {
    r.p[c] = mean_p[c];
    if (mean_p[c] > mean_p[r.top]) r.top = c;
  }
  r.confidence = mean_p[r.top];
  r.entropy = entropyOf(mean_p);
  r.mutual_info = r.entropy - mean_member_entropy;
  if (r.mutual_info < 0.0f) r.mutual_info = 0.0f;
  r.novelty = mlNovelty(m, x);
  r.uncertain = r.confidence < m.conf_threshold || r.mutual_info > m.mi_threshold ||
                (m.novelty_threshold > 0.0f && r.novelty > m.novelty_threshold);
  return r;
}
