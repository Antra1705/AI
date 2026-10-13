// train - trains the int8 road-surface classifier ensemble (no ML libraries).
//
//   train <dataset.csv> [--members 5] [--hidden 16] [--epochs 30] [--qat-epochs 12] [--oe 0.5]
//         [--model ../adaptive_suspension/src/core/model_data.cpp]
//         [--vectors tests/model_vectors.h] [--report ../docs/model_report.md]
//
// Pipeline per ensemble member (bootstrap resample + different initial weights):
//   1. float training (Adam, class-weighted cross-entropy)
//   2. calibrate activation ranges -> int8 input scales per layer
//   3. quantization-aware fine-tuning: weights and activations are rounded to
//      int8 in the forward pass, gradients pass straight through (STE)
// Then for the whole ensemble:
//   4. temperature scaling fitted on the validation runs
//   5. uncertainty thresholds (confidence, mutual information) from validation
//   6. test-set report: float vs post-training quantization vs QAT, single vs
//      ensemble, rule-based baseline, calibration error, out-of-distribution detection
//   7. export model_data.cpp (int8 weights) + test vectors for the parity test
//
// Data is split by run (whole drives), never by window, so neighbouring,
// overlapping windows cannot leak between training and test.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "config.h"
#include "core/classifier.h"
#include "core/ml_model.h"

static const int F = ML_FEATURE_COUNT;
static const int C = ML_CLASSES;
static const char* CLASS_NAMES[C] = {"smooth", "gravel", "breaker", "pothole"};

// ---------------------------------------------------------------- data

struct Sample {
  int run, label, rule_pred, aug;
  float f[F];  // raw features
  float x[F];  // standardized
};

struct Data {
  std::vector<Sample> all;
  std::vector<int> train, val, test;          // labelled windows 0..3
  std::vector<int> train_other;               // "other" surfaces (label 5), outlier exposure
  std::vector<int> test_ood, test_ambiguous;  // for uncertainty evaluation
  float mean[F], std[F];
  float class_mean[C * F];  // class means of standardized features
  float inv_cov[C * F * F]; // inverse covariance per class
};

// Gauss-Jordan inverse of an n x n matrix (row-major). Returns false if singular.
static bool invert(std::vector<double> a, int n, std::vector<double>& inv) {
  inv.assign(n * n, 0.0);
  for (int i = 0; i < n; ++i) inv[i * n + i] = 1.0;
  for (int col = 0; col < n; ++col) {
    int piv = col;
    for (int r = col + 1; r < n; ++r)
      if (std::fabs(a[r * n + col]) > std::fabs(a[piv * n + col])) piv = r;
    if (std::fabs(a[piv * n + col]) < 1e-12) return false;
    for (int k = 0; k < n; ++k) {
      std::swap(a[col * n + k], a[piv * n + k]);
      std::swap(inv[col * n + k], inv[piv * n + k]);
    }
    double d = a[col * n + col];
    for (int k = 0; k < n; ++k) {
      a[col * n + k] /= d;
      inv[col * n + k] /= d;
    }
    for (int r = 0; r < n; ++r) {
      if (r == col) continue;
      double f = a[r * n + col];
      for (int k = 0; k < n; ++k) {
        a[r * n + k] -= f * a[col * n + k];
        inv[r * n + k] -= f * inv[col * n + k];
      }
    }
  }
  return true;
}

// Class-conditional Gaussian, one covariance per class, fitted on training windows.
static void fitNoveltyModel(Data& d) {
  double sum[C][F] = {{0}};
  int n[C] = {0};
  for (int i : d.train) {
    const Sample& s = d.all[i];
    for (int k = 0; k < F; ++k) sum[s.label][k] += s.x[k];
    n[s.label]++;
  }
  for (int c = 0; c < C; ++c)
    for (int k = 0; k < F; ++k) d.class_mean[c * F + k] = (float)(sum[c][k] / std::max(n[c], 1));
  for (int c = 0; c < C; ++c) {
    std::vector<double> cov(F * F, 0.0);
    for (int i : d.train) {
      const Sample& s = d.all[i];
      if (s.label != c) continue;
      double dv[F];
      for (int k = 0; k < F; ++k) dv[k] = s.x[k] - d.class_mean[c * F + k];
      for (int a = 0; a < F; ++a)
        for (int b = 0; b < F; ++b) cov[a * F + b] += dv[a] * dv[b];
    }
    for (int a = 0; a < F; ++a) {
      for (int b = 0; b < F; ++b) cov[a * F + b] /= std::max(n[c] - 1, 1);
      cov[a * F + a] += 1e-3;  // ridge for numerical safety
    }
    std::vector<double> inv;
    if (!invert(cov, F, inv)) {
      std::fprintf(stderr, "covariance of class %d not invertible\n", c);
      std::exit(1);
    }
    for (int k = 0; k < F * F; ++k) d.inv_cov[c * F * F + k] = (float)inv[k];
  }
}

static int splitOf(int run) {
  int r = run % 20;
  return r < 14 ? 0 : (r < 17 ? 1 : 2);  // 70 / 15 / 15 % of runs
}

static bool loadData(const char* path, Data& d) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  std::getline(in, line);
  std::vector<std::string> cols;
  {
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, ',')) cols.push_back(c);
  }
  auto col = [&](const std::string& name) {
    for (size_t i = 0; i < cols.size(); ++i)
      if (cols[i] == name) return (int)i;
    return -1;
  };
  int c_run = col("run"), c_label = col("label"), c_rule = col("rule_pred"), c_f0 = col(mlFeatureName(0));
  int c_aug = col("aug");
  if (c_run < 0 || c_label < 0 || c_f0 < 0) return false;

  std::vector<double> v(cols.size());
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string cell;
    size_t i = 0;
    while (std::getline(ss, cell, ',') && i < v.size()) v[i++] = std::atof(cell.c_str());
    if (i < v.size()) continue;
    Sample s;
    s.run = (int)v[c_run];
    s.label = (int)v[c_label];
    s.rule_pred = c_rule >= 0 ? (int)v[c_rule] : -1;
    s.aug = c_aug >= 0 ? (int)v[c_aug] : 0;
    for (int k = 0; k < F; ++k) s.f[k] = (float)v[c_f0 + k];
    d.all.push_back(s);
  }

  for (int i = 0; i < (int)d.all.size(); ++i) {
    const Sample& s = d.all[i];
    int sp = splitOf(s.run);
    if (s.aug && sp != 0) continue;  // augmented copies are for training only
    if (s.label >= 0 && s.label < C) {
      (sp == 0 ? d.train : sp == 1 ? d.val : d.test).push_back(i);
    } else if (s.label == 5 && sp == 0) {
      d.train_other.push_back(i);
    } else if (sp == 2) {
      (s.label == 4 ? d.test_ood : d.test_ambiguous).push_back(i);
    }
  }
  for (int k = 0; k < F; ++k) {
    double m = 0, m2 = 0;
    for (int i : d.train) m += d.all[i].f[k];
    m /= d.train.size();
    for (int i : d.train) m2 += (d.all[i].f[k] - m) * (d.all[i].f[k] - m);
    d.mean[k] = (float)m;
    d.std[k] = (float)std::max(std::sqrt(m2 / d.train.size()), 1e-3);
  }
  for (Sample& s : d.all)
    for (int k = 0; k < F; ++k) s.x[k] = (s.f[k] - d.mean[k]) / d.std[k];
  if (d.train.empty() || d.val.empty() || d.test.empty()) return false;
  fitNoveltyModel(d);
  return true;
}

// ---------------------------------------------------------------- network

struct Layer {
  int in = 0, out = 0;
  std::vector<float> w, b, gw, gb, mw, vw, mb, vb;
  std::vector<float> wq, bq;  // weights/bias used in the forward pass (fake-quantized in QAT)
  float in_scale = 0.0f;      // int8 scale of this layer's input
  float w_scale = 1.0f;
};

struct Net {
  Layer L[ML_LAYERS];
  bool qat = false;
};

static void initNet(Net& n, int hidden, std::mt19937& rng) {
  int sizes[ML_LAYERS + 1] = {F, hidden, hidden, C};
  for (int l = 0; l < ML_LAYERS; ++l) {
    Layer& L = n.L[l];
    L.in = sizes[l];
    L.out = sizes[l + 1];
    float lim = std::sqrt(6.0f / L.in);
    std::uniform_real_distribution<float> u(-lim, lim);
    L.w.resize(L.in * L.out);
    for (float& w : L.w) w = u(rng);
    L.b.assign(L.out, 0.0f);
    L.gw.assign(L.w.size(), 0.0f);
    L.gb.assign(L.out, 0.0f);
    L.mw.assign(L.w.size(), 0.0f);
    L.vw.assign(L.w.size(), 0.0f);
    L.mb.assign(L.out, 0.0f);
    L.vb.assign(L.out, 0.0f);
  }
}

// Prepare the forward-pass weights: plain copy, or rounded to int8 for QAT.
static void prepareWeights(Net& n) {
  for (Layer& L : n.L) {
    L.wq = L.w;
    L.bq = L.b;
    if (!n.qat) continue;
    float mx = 1e-8f;
    for (float w : L.w) mx = std::max(mx, std::fabs(w));
    L.w_scale = mx / 127.0f;
    for (size_t i = 0; i < L.w.size(); ++i) L.wq[i] = quantizeS8(L.w[i], L.w_scale) * L.w_scale;
    float bs = L.in_scale * L.w_scale;
    for (int o = 0; o < L.out; ++o) L.bq[o] = (float)std::lround(L.b[o] / bs) * bs;
  }
}

struct Trace {
  float in[ML_LAYERS][ML_MAX_WIDTH];    // layer inputs (after quantization)
  bool pass[ML_LAYERS][ML_MAX_WIDTH];   // false where quantization clipped (no gradient)
  float z[ML_LAYERS][ML_MAX_WIDTH];     // pre-activations
};

static void forward(const Net& n, const float* x, Trace& t, float logits[C]) {
  float a[ML_MAX_WIDTH];
  for (int i = 0; i < F; ++i) a[i] = x[i];
  for (int l = 0; l < ML_LAYERS; ++l) {
    const Layer& L = n.L[l];
    for (int i = 0; i < L.in; ++i) {
      if (n.qat) {
        t.in[l][i] = quantizeS8(a[i], L.in_scale) * L.in_scale;
        t.pass[l][i] = std::fabs(a[i]) <= 127.0f * L.in_scale;
      } else {
        t.in[l][i] = a[i];
        t.pass[l][i] = true;
      }
    }
    for (int o = 0; o < L.out; ++o) {
      float z = L.bq[o];
      for (int i = 0; i < L.in; ++i) z += L.wq[o * L.in + i] * t.in[l][i];
      t.z[l][o] = z;
      a[o] = (l < ML_LAYERS - 1) ? std::max(z, 0.0f) : z;
    }
  }
  for (int c = 0; c < C; ++c) logits[c] = a[c];
}

static void softmax(const float z[C], float p[C], float temperature = 1.0f) {
  float mx = *std::max_element(z, z + C), s = 0;
  for (int c = 0; c < C; ++c) s += (p[c] = std::exp((z[c] - mx) / temperature));
  for (int c = 0; c < C; ++c) p[c] /= s;
}

// Backward pass for one sample; accumulates gradients. Returns weighted loss.
static float backward(Net& n, const Trace& t, const float logits[C], int label, float weight) {
  float p[C];
  softmax(logits, p);
  float dz[ML_MAX_WIDTH];
  // label == C: an "other" surface - the target is the uniform distribution
  // (outlier exposure), so the model learns to be unsure on unfamiliar road.
  for (int c = 0; c < C; ++c) dz[c] = weight * (p[c] - (label == C ? 1.0f / C : (c == label ? 1.0f : 0.0f)));
  for (int l = ML_LAYERS - 1; l >= 0; --l) {
    Layer& L = n.L[l];
    float da[ML_MAX_WIDTH] = {0};
    for (int o = 0; o < L.out; ++o) {
      L.gb[o] += dz[o];
      for (int i = 0; i < L.in; ++i) {
        L.gw[o * L.in + i] += dz[o] * t.in[l][i];
        da[i] += L.wq[o * L.in + i] * dz[o];
      }
    }
    if (l == 0) break;
    for (int i = 0; i < L.in; ++i) dz[i] = (t.pass[l][i] && t.z[l - 1][i] > 0.0f) ? da[i] : 0.0f;
  }
  if (label == C) {
    float mean_logp = 0;
    for (int c = 0; c < C; ++c) mean_logp += std::log(std::max(p[c], 1e-7f)) / C;
    return -weight * mean_logp;
  }
  return -weight * std::log(std::max(p[label], 1e-7f));
}

static void adamStep(Net& n, float lr, int step, float batch_weight) {
  const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, decay = 1e-4f;
  float c1 = 1.0f - std::pow(b1, (float)step), c2 = 1.0f - std::pow(b2, (float)step);
  auto upd = [&](std::vector<float>& w, std::vector<float>& g, std::vector<float>& m, std::vector<float>& v, bool wd) {
    for (size_t i = 0; i < w.size(); ++i) {
      float gi = g[i] / batch_weight + (wd ? decay * w[i] : 0.0f);
      m[i] = b1 * m[i] + (1 - b1) * gi;
      v[i] = b2 * v[i] + (1 - b2) * gi * gi;
      w[i] -= lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + eps);
      g[i] = 0.0f;
    }
  };
  for (Layer& L : n.L) {
    upd(L.w, L.gw, L.mw, L.vw, true);
    upd(L.b, L.gb, L.mb, L.vb, false);
  }
}

static void trainEpochs(Net& n, const Data& d, const std::vector<int>& idx, const float cw[C + 1], int epochs,
                        float lr0, std::mt19937& rng, const char* tag) {
  std::vector<int> order = idx;
  int step = 0;
  for (int e = 0; e < epochs; ++e) {
    std::shuffle(order.begin(), order.end(), rng);
    float lr = lr0 * 0.5f * (1.0f + std::cos(3.14159265f * e / epochs));  // cosine decay
    double loss = 0, wsum_all = 0;
    for (size_t start = 0; start < order.size(); start += 64) {
      prepareWeights(n);
      size_t end = std::min(order.size(), start + 64);
      float wsum = 0;
      for (size_t k = start; k < end; ++k) {
        const Sample& s = d.all[order[k]];
        Trace t;
        float z[C];
        forward(n, s.x, t, z);
        const int lab = s.label < C ? s.label : C;
        loss += backward(n, t, z, lab, cw[lab]);
        wsum += cw[lab];
      }
      wsum_all += wsum;
      adamStep(n, lr, ++step, wsum);
    }
    if (e == epochs - 1) std::fprintf(stderr, "    %s: %d epochs, final loss %.4f\n", tag, epochs, loss / wsum_all);
  }
  prepareWeights(n);
}

// Set each layer's int8 input scale from the 99.99th percentile of |input|.
static void calibrateScales(Net& n, const Data& d, const std::vector<int>& idx) {
  bool was = n.qat;
  n.qat = false;
  prepareWeights(n);
  std::vector<float> mags[ML_LAYERS];
  for (size_t k = 0; k < idx.size(); k += 2) {
    Trace t;
    float z[C];
    forward(n, d.all[idx[k]].x, t, z);
    for (int l = 0; l < ML_LAYERS; ++l)
      for (int i = 0; i < n.L[l].in; ++i) mags[l].push_back(std::fabs(t.in[l][i]));
  }
  for (int l = 0; l < ML_LAYERS; ++l) {
    std::vector<float>& m = mags[l];
    size_t q = (size_t)(0.9999 * (m.size() - 1));
    std::nth_element(m.begin(), m.begin() + q, m.end());
    n.L[l].in_scale = std::max(m[q], 1e-3f) / 127.0f;
  }
  n.qat = was;
  prepareWeights(n);
}

// ---------------------------------------------------------------- export to int8

struct Exported {
  std::vector<int8_t> w[ML_LAYERS];
  std::vector<int32_t> b[ML_LAYERS];
  QNet net;
};

static void exportNet(const Net& n, Exported& e) {
  for (int l = 0; l < ML_LAYERS; ++l) {
    const Layer& L = n.L[l];
    float mx = 1e-8f;
    for (float w : L.w) mx = std::max(mx, std::fabs(w));
    float ws = mx / 127.0f;
    e.w[l].resize(L.w.size());
    for (size_t i = 0; i < L.w.size(); ++i) e.w[l][i] = quantizeS8(L.w[i], ws);
    e.b[l].resize(L.out);
    for (int o = 0; o < L.out; ++o) e.b[l][o] = (int32_t)std::lround(L.b[o] / (L.in_scale * ws));
    e.net.layer[l] = QLayer{L.in, L.out, e.w[l].data(), e.b[l].data(), L.in_scale, ws};
  }
}

// ---------------------------------------------------------------- metrics

struct Scores {
  double acc = 0, macro_f1 = 0;
  double recall[C] = {0};
  int confusion[C][C] = {{0}};
};

static Scores score(const std::vector<int>& truth, const std::vector<int>& pred) {
  Scores s;
  int tp[C] = {0}, fp[C] = {0}, fn[C] = {0}, n_ok = 0;
  for (size_t i = 0; i < truth.size(); ++i) {
    int t = truth[i], p = pred[i];
    if (p >= 0 && p < C) s.confusion[t][p]++;
    if (t == p) {
      ++n_ok;
      ++tp[t];
    } else {
      ++fn[t];
      if (p >= 0 && p < C) ++fp[p];
    }
  }
  s.acc = (double)n_ok / truth.size();
  for (int c = 0; c < C; ++c) {
    double prec = tp[c] + fp[c] ? (double)tp[c] / (tp[c] + fp[c]) : 0;
    double rec = tp[c] + fn[c] ? (double)tp[c] / (tp[c] + fn[c]) : 0;
    s.recall[c] = rec;
    s.macro_f1 += (prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0) / C;
  }
  return s;
}

static double auroc(const std::vector<double>& neg, const std::vector<double>& pos) {
  // Probability that a random positive (OOD) scores higher than a random negative.
  std::vector<std::pair<double, int>> all;
  for (double v : neg) all.push_back({v, 0});
  for (double v : pos) all.push_back({v, 1});
  std::sort(all.begin(), all.end());
  double rank_sum = 0;
  for (size_t i = 0; i < all.size();) {
    size_t j = i;
    while (j < all.size() && all[j].first == all[i].first) ++j;
    double avg_rank = (i + j + 1) / 2.0;
    for (size_t k = i; k < j; ++k)
      if (all[k].second) rank_sum += avg_rank;
    i = j;
  }
  double np = pos.size(), nn = neg.size();
  return (rank_sum - np * (np + 1) / 2) / (np * nn);
}

static double ece(const std::vector<double>& conf, const std::vector<int>& correct) {
  const int B = 15;
  double sum_c[B] = {0}, sum_a[B] = {0};
  int cnt[B] = {0};
  for (size_t i = 0; i < conf.size(); ++i) {
    int b = std::min(B - 1, (int)(conf[i] * B));
    sum_c[b] += conf[i];
    sum_a[b] += correct[i];
    ++cnt[b];
  }
  double e = 0;
  for (int b = 0; b < B; ++b)
    if (cnt[b]) e += std::fabs(sum_a[b] / cnt[b] - sum_c[b] / cnt[b]) * cnt[b] / conf.size();
  return e;
}

static double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[(size_t)(p * (v.size() - 1))];
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: train <dataset.csv> [--members N] [--hidden N] [--epochs N] [--qat-epochs N]\n"
                         "             [--model FILE] [--vectors FILE] [--report FILE]\n");
    return 2;
  }
  int members = 5, hidden = 16, epochs = 30, qat_epochs = 12;
  float oe = 0.5f;
  std::string model_path = "../adaptive_suspension/src/core/model_data.cpp";
  std::string vectors_path = "tests/model_vectors.h";
  std::string report_path = "../docs/model_report.md";
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "--members") members = std::atoi(argv[i + 1]);
    else if (k == "--hidden") hidden = std::atoi(argv[i + 1]);
    else if (k == "--epochs") epochs = std::atoi(argv[i + 1]);
    else if (k == "--qat-epochs") qat_epochs = std::atoi(argv[i + 1]);
    else if (k == "--oe") oe = (float)std::atof(argv[i + 1]);
    else if (k == "--model") model_path = argv[i + 1];
    else if (k == "--vectors") vectors_path = argv[i + 1];
    else if (k == "--report") report_path = argv[i + 1];
    else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  if (hidden > ML_MAX_WIDTH || members < 1) {
    std::fprintf(stderr, "hidden must be <= %d, members >= 1\n", ML_MAX_WIDTH);
    return 2;
  }

  Data d;
  if (!loadData(argv[1], d)) {
    std::fprintf(stderr, "could not load %s\n", argv[1]);
    return 1;
  }
  int count[C] = {0};
  for (int i : d.train) count[d.all[i].label]++;
  float cw[C + 1];
  for (int c = 0; c < C; ++c) cw[c] = std::min(20.0f, (float)d.train.size() / (C * std::max(count[c], 1)));
  cw[C] = oe;  // outlier-exposure weight for "other" surfaces
  std::fprintf(stderr, "windows: train %zu, val %zu, test %zu, test-OOD %zu, test-ambiguous %zu\n", d.train.size(),
               d.val.size(), d.test.size(), d.test_ood.size(), d.test_ambiguous.size());

  // --- train the ensemble
  std::vector<Net> float_nets(members), qat_nets(members);
  for (int m = 0; m < members; ++m) {
    std::fprintf(stderr, "member %d/%d\n", m + 1, members);
    std::mt19937 rng(1234 + 97 * m);
    std::vector<int> boot;
    std::vector<int> pool = d.train;
    if (oe > 0) pool.insert(pool.end(), d.train_other.begin(), d.train_other.end());
    std::uniform_int_distribution<size_t> pick(0, pool.size() - 1);
    for (size_t i = 0; i < pool.size(); ++i) boot.push_back(pool[pick(rng)]);
    Net& n = float_nets[m];
    initNet(n, hidden, rng);
    trainEpochs(n, d, boot, cw, epochs, 3e-3f, rng, "float");
    calibrateScales(n, d, d.train);
    qat_nets[m] = n;
    qat_nets[m].qat = true;
    prepareWeights(qat_nets[m]);
    trainEpochs(qat_nets[m], d, boot, cw, qat_epochs, 5e-4f, rng, "QAT");
  }

  // --- int8 exports: QAT (deployed) and post-training quantization (comparison)
  std::vector<Exported> qat_x(members), ptq_x(members);
  std::vector<QNet> qat_q(members), ptq_q(members);
  for (int m = 0; m < members; ++m) {
    exportNet(qat_nets[m], qat_x[m]);
    exportNet(float_nets[m], ptq_x[m]);
    qat_q[m] = qat_x[m].net;
    ptq_q[m] = ptq_x[m].net;
  }
  auto ensembleOf = [&](const std::vector<QNet>& nets, int n_members, float T) {
    MlEnsemble e = {n_members, nets.data(), d.mean, d.std, T, 0, 0, d.class_mean, d.inv_cov, 0, ""};
    return e;
  };

  // --- temperature scaling on validation runs
  float best_T = 1.0f;
  double best_nll = 1e30;
  for (float T = 0.5f; T <= 5.0f; T += 0.05f) {
    MlEnsemble e = ensembleOf(qat_q, members, T);
    double nll = 0;
    for (int i : d.val) nll -= std::log(std::max(mlPredict(e, d.all[i].f).p[d.all[i].label], 1e-7f));
    if (nll < best_nll) {
      best_nll = nll;
      best_T = T;
    }
  }
  MlEnsemble deployed = ensembleOf(qat_q, members, best_T);

  // --- uncertainty thresholds from correctly classified validation windows
  std::vector<double> val_conf_ok, val_mi_ok, val_novelty;
  for (int i : d.val) {
    MlPrediction p = mlPredict(deployed, d.all[i].f);
    val_novelty.push_back(p.novelty);
    if (p.top == d.all[i].label) {
      val_conf_ok.push_back(p.confidence);
      val_mi_ok.push_back(p.mutual_info);
    }
  }
  float conf_thr = (float)std::max(0.5, percentile(val_conf_ok, 0.025));
  float mi_thr = (float)std::max(1e-3, percentile(val_mi_ok, 0.975));
  float nov_thr = (float)percentile(val_novelty, 0.99);
  deployed.conf_threshold = conf_thr;
  deployed.mi_threshold = mi_thr;
  deployed.novelty_threshold = nov_thr;
  auto uncertain = [&](const MlPrediction& p) { return p.uncertain; };

  // --- test-set evaluation
  std::vector<int> truth;
  for (int i : d.test) truth.push_back(d.all[i].label);
  auto predictAll = [&](std::function<int(const Sample&)> f) {
    std::vector<int> out;
    for (int i : d.test) out.push_back(f(d.all[i]));
    return out;
  };
  Scores s_rule = score(truth, predictAll([&](const Sample& s) {
    Features ft = {9.81f, std::exp(s.f[4]) - 1.0f, std::exp(s.f[1]) - 1e-3f, std::exp(s.f[0]) - 1e-3f, 0.0f, (int)s.f[6]};
    Classification c = classify(ft, defaultThresholds());
    return c.surface == SURFACE_UNKNOWN ? -1 : (int)c.surface;
  }));
  Scores s_float = score(truth, predictAll([&](const Sample& s) {
    Net n = float_nets[0];
    n.qat = false;
    prepareWeights(n);
    Trace t;
    float z[C];
    forward(n, s.x, t, z);
    return (int)(std::max_element(z, z + C) - z);
  }));
  Scores s_ptq = score(truth, predictAll([&](const Sample& s) {
    float z[C];
    qnetForward(ptq_q[0], s.x, z);
    return (int)(std::max_element(z, z + C) - z);
  }));
  Scores s_qat = score(truth, predictAll([&](const Sample& s) {
    float z[C];
    qnetForward(qat_q[0], s.x, z);
    return (int)(std::max_element(z, z + C) - z);
  }));
  MlEnsemble ptq_ens = ensembleOf(ptq_q, members, best_T);
  Scores s_ptq_ens = score(truth, predictAll([&](const Sample& s) { return mlPredict(ptq_ens, s.f).top; }));
  Scores s_ens = score(truth, predictAll([&](const Sample& s) { return mlPredict(deployed, s.f).top; }));

  // calibration
  std::vector<double> conf_raw, conf_cal;
  std::vector<int> ok_raw, ok_cal;
  MlEnsemble raw = ensembleOf(qat_q, members, 1.0f);
  for (int i : d.test) {
    MlPrediction a = mlPredict(raw, d.all[i].f), b = mlPredict(deployed, d.all[i].f);
    conf_raw.push_back(a.confidence);
    ok_raw.push_back(a.top == d.all[i].label);
    conf_cal.push_back(b.confidence);
    ok_cal.push_back(b.top == d.all[i].label);
  }

  // uncertainty: selective accuracy and out-of-distribution detection
  int accepted = 0, accepted_ok = 0, rejected = 0, rejected_ok = 0;
  std::vector<double> mi_in, mi_ood, inv_conf_in, inv_conf_ood, single_inv_conf_in, single_inv_conf_ood, nov_in,
      nov_ood;
  MlEnsemble single = ensembleOf(qat_q, 1, best_T);
  for (int i : d.test) {
    MlPrediction p = mlPredict(deployed, d.all[i].f);
    bool ok = p.top == d.all[i].label;
    if (uncertain(p)) {
      ++rejected;
      rejected_ok += ok;
    } else {
      ++accepted;
      accepted_ok += ok;
    }
    mi_in.push_back(p.mutual_info);
    nov_in.push_back(p.novelty);
    inv_conf_in.push_back(1 - p.confidence);
    single_inv_conf_in.push_back(1 - mlPredict(single, d.all[i].f).confidence);
  }
  int ood_flagged = 0, amb_flagged = 0, single_ood_flagged = 0;
  for (int i : d.test_ood) {
    MlPrediction p = mlPredict(deployed, d.all[i].f);
    ood_flagged += uncertain(p);
    mi_ood.push_back(p.mutual_info);
    nov_ood.push_back(p.novelty);
    inv_conf_ood.push_back(1 - p.confidence);
    MlPrediction ps = mlPredict(single, d.all[i].f);
    single_ood_flagged += ps.confidence < conf_thr;
    single_inv_conf_ood.push_back(1 - ps.confidence);
  }
  for (int i : d.test_ambiguous) amb_flagged += uncertain(mlPredict(deployed, d.all[i].f));
  std::vector<double> combo_in, combo_ood;  // the deployed rule as one score
  for (size_t k = 0; k < mi_in.size(); ++k)
    combo_in.push_back(std::max(std::max(mi_in[k] / mi_thr, inv_conf_in[k] / (1 - conf_thr)), nov_in[k] / nov_thr));
  for (size_t k = 0; k < mi_ood.size(); ++k)
    combo_ood.push_back(std::max(std::max(mi_ood[k] / mi_thr, inv_conf_ood[k] / (1 - conf_thr)), nov_ood[k] / nov_thr));

  // --- write model_data.cpp
  {
    FILE* f = std::fopen(model_path.c_str(), "w");
    if (!f) {
      std::perror(model_path.c_str());
      return 1;
    }
    std::fprintf(f, "// GENERATED by host/train from %s - do not edit by hand.\n", argv[1]);
    std::fprintf(f, "// %d-member int8 ensemble, %d-%d-%d-%d, QAT. Test accuracy %.1f%%, macro-F1 %.3f.\n", members, F,
                 hidden, hidden, C, 100 * s_ens.acc, s_ens.macro_f1);
    std::fprintf(f, "#include \"ml_model.h\"\n\nnamespace {\n\n");
    std::fprintf(f, "const float kMean[%d] = {", F);
    for (int k = 0; k < F; ++k) std::fprintf(f, "%s%#.9gf", k ? ", " : "", d.mean[k]);
    std::fprintf(f, "};\nconst float kStd[%d] = {", F);
    for (int k = 0; k < F; ++k) std::fprintf(f, "%s%#.9gf", k ? ", " : "", d.std[k]);
    std::fprintf(f, "};\nconst float kClassMeans[%d] = {", C * F);
    for (int k = 0; k < C * F; ++k) std::fprintf(f, "%s%#.9gf", k ? ", " : "", d.class_mean[k]);
    std::fprintf(f, "};\nconst float kInvCov[%d] = {", C * F * F);
    for (int k = 0; k < C * F * F; ++k) std::fprintf(f, "%s%#.9gf", k ? ", " : "", d.inv_cov[k]);
    std::fprintf(f, "};\n\n");
    for (int m = 0; m < members; ++m)
      for (int l = 0; l < ML_LAYERS; ++l) {
        std::fprintf(f, "const int8_t kW%d_%d[%zu] = {", m, l, qat_x[m].w[l].size());
        for (size_t i = 0; i < qat_x[m].w[l].size(); ++i) std::fprintf(f, "%s%d", i ? "," : "", qat_x[m].w[l][i]);
        std::fprintf(f, "};\nconst int32_t kB%d_%d[%zu] = {", m, l, qat_x[m].b[l].size());
        for (size_t i = 0; i < qat_x[m].b[l].size(); ++i) std::fprintf(f, "%s%d", i ? "," : "", qat_x[m].b[l][i]);
        std::fprintf(f, "};\n");
      }
    std::fprintf(f, "\nconst QNet kNets[%d] = {\n", members);
    for (int m = 0; m < members; ++m) {
      std::fprintf(f, "    {{");
      for (int l = 0; l < ML_LAYERS; ++l) {
        const QLayer& L = qat_q[m].layer[l];
        std::fprintf(f, "%s{%d, %d, kW%d_%d, kB%d_%d, %#.9gf, %#.9gf}", l ? ", " : "", L.in, L.out, m, l, m, l,
                     L.in_scale, L.w_scale);
      }
      std::fprintf(f, "}},\n");
    }
    std::fprintf(f, "};\n\nconst MlEnsemble kModel = {%d, kNets, kMean, kStd, %#.4gf, %#.6gf, %#.6gf,\n", members,
                 best_T, mi_thr, conf_thr);
    std::fprintf(f, "                             kClassMeans, kInvCov, %#.6gf,\n", nov_thr);
    std::fprintf(f, "                             \"%d x MLP %d-%d-%d-%d int8 QAT, test acc %.1f%%\"};\n\n", members, F,
                 hidden, hidden, C, 100 * s_ens.acc);
    std::fprintf(f, "}  // namespace\n\nconst MlEnsemble& deployedModel() { return kModel; }\n");
    std::fclose(f);
  }

  // --- parity test vectors: trainer's own fake-quant forward, independent of qnetForward
  {
    FILE* f = std::fopen(vectors_path.c_str(), "w");
    if (f) {
      std::fprintf(f, "// GENERATED by host/train. Expected ensemble outputs computed by the trainer's\n"
                      "// fake-quantized forward pass; the on-device int8 code must reproduce them.\n#pragma once\n\n");
      std::vector<int> pick;
      for (int c = 0; c < C; ++c) {
        int n = 0;
        for (int i : d.test)
          if (d.all[i].label == c && n < 8 && (std::rand() % 7 == 0)) {
            pick.push_back(i);
            ++n;
          }
      }
      for (size_t k = 0; k < std::min<size_t>(6, d.test_ood.size()); ++k) pick.push_back(d.test_ood[k * 7 % d.test_ood.size()]);
      std::fprintf(f, "static const int kVectorCount = %zu;\nstatic const float kVectorFeatures[%zu][%d] = {\n", pick.size(),
                   pick.size(), F);
      for (int i : pick) {
        std::fprintf(f, "    {");
        for (int k = 0; k < F; ++k) std::fprintf(f, "%s%#.9gf", k ? ", " : "", d.all[i].f[k]);
        std::fprintf(f, "},\n");
      }
      std::fprintf(f, "};\nstatic const float kVectorProbs[%zu][%d] = {\n", pick.size(), C);
      std::vector<float> mis;
      for (int i : pick) {
        float mean_p[C] = {0}, member_h = 0;
        for (int m = 0; m < members; ++m) {
          Trace t;
          float z[C], p[C];
          forward(qat_nets[m], d.all[i].x, t, z);
          softmax(z, p, best_T);
          float h = 0;
          for (int c = 0; c < C; ++c) {
            mean_p[c] += p[c] / members;
            if (p[c] > 1e-7f) h -= p[c] * std::log(p[c]);
          }
          member_h += h / members;
        }
        float h = 0;
        for (int c = 0; c < C; ++c)
          if (mean_p[c] > 1e-7f) h -= mean_p[c] * std::log(mean_p[c]);
        mis.push_back(std::max(0.0f, h - member_h));
        std::fprintf(f, "    {%#.7gf, %#.7gf, %#.7gf, %#.7gf},\n", mean_p[0], mean_p[1], mean_p[2], mean_p[3]);
      }
      std::fprintf(f, "};\nstatic const float kVectorMI[%zu] = {", pick.size());
      for (size_t k = 0; k < mis.size(); ++k) std::fprintf(f, "%s%#.7gf", k ? ", " : "", mis[k]);
      std::fprintf(f, "};\nstatic const float kVectorNovelty[%zu] = {", pick.size());
      for (size_t k = 0; k < pick.size(); ++k) {
        double best = 1e300;  // double-precision reference, independent of mlNovelty()
        for (int c = 0; c < C; ++c) {
          double q = 0;
          for (int a = 0; a < F; ++a)
            for (int b = 0; b < F; ++b)
              q += (d.all[pick[k]].x[a] - d.class_mean[c * F + a]) * d.inv_cov[c * F * F + a * F + b] *
                   (d.all[pick[k]].x[b] - d.class_mean[c * F + b]);
          best = std::min(best, q);
        }
        std::fprintf(f, "%s%#.7gf", k ? ", " : "", best);
      }
      std::fprintf(f, "};\n");
      std::fclose(f);
    }
  }

  // --- report
  {
    FILE* f = std::fopen(report_path.c_str(), "w");
    FILE* outs[2] = {stdout, f};
    for (FILE* o : outs) {
      if (!o) continue;
      std::fprintf(o, "# Road-surface model report\n\n");
      std::fprintf(o, "Generated by `host/build/bin/train %s`. Data: simulated drives (host `sim dataset`),\n"
                      "split by run: train %zu / validation %zu / test %zu windows; test also has %zu rumble-strip\n"
                      "windows (a surface never seen in training) and %zu ambiguous windows.\n\n",
                   argv[1], d.train.size(), d.val.size(), d.test.size(), d.test_ood.size(), d.test_ambiguous.size());
      std::fprintf(o, "Model: %d x MLP %d-%d-%d-%d, int8 weights and activations, %d bytes of weights in total.\n\n",
                   members, F, hidden, hidden, C, members * (F * hidden + hidden * hidden + hidden * C));
      std::fprintf(o, "## Accuracy on the test runs\n\n| Model | Accuracy | Macro-F1 | smooth | gravel | breaker | pothole |\n"
                      "|---|---|---|---|---|---|---|\n");
      auto row = [&](const char* name, const Scores& s) {
        std::fprintf(o, "| %s | %.1f%% | %.3f | %.1f%% | %.1f%% | %.1f%% | %.1f%% |\n", name, 100 * s.acc, s.macro_f1,
                     100 * s.recall[0], 100 * s.recall[1], 100 * s.recall[2], 100 * s.recall[3]);
      };
      row("Rule-based (config.h thresholds)", s_rule);
      row("1 MLP, float32", s_float);
      row("1 MLP, int8 post-training quantization", s_ptq);
      row("1 MLP, int8 quantization-aware training", s_qat);
      row("Ensemble, int8 post-training quantization", s_ptq_ens);
      row("**Ensemble, int8 QAT (deployed)**", s_ens);
      std::fprintf(o, "\nPer-class columns are recall. Rule-based 'unknown' answers count as wrong.\n\n");
      std::fprintf(o, "Confusion matrix of the deployed model (rows = true, columns = predicted):\n\n|  | ");
      for (int c = 0; c < C; ++c) std::fprintf(o, "%s | ", CLASS_NAMES[c]);
      std::fprintf(o, "\n|---|---|---|---|---|\n");
      for (int t = 0; t < C; ++t) {
        std::fprintf(o, "| %s | ", CLASS_NAMES[t]);
        for (int p = 0; p < C; ++p) std::fprintf(o, "%d | ", s_ens.confusion[t][p]);
        std::fprintf(o, "\n");
      }
      std::fprintf(o, "\n## Calibration\n\nTemperature fitted on validation runs: T = %.2f.\n"
                      "Expected calibration error on test: %.3f before, %.3f after temperature scaling.\n",
                   best_T, ece(conf_raw, ok_raw), ece(conf_cal, ok_cal));
      std::fprintf(o, "\n## Uncertainty and safety fallback\n\n"
                      "A window is flagged *uncertain* when confidence < %.3f or mutual information > %.4f\n"
                      "(2.5th / 97.5th percentiles of correctly classified validation windows), or novelty\n"
                      "(squared Mahalanobis distance to the nearest class) > %.1f (99th percentile of validation).\n\n",
                   conf_thr, mi_thr, nov_thr);
      std::fprintf(o, "| | Value |\n|---|---|\n");
      std::fprintf(o, "| Test windows accepted (coverage) | %.1f%% |\n", 100.0 * accepted / d.test.size());
      std::fprintf(o, "| Accuracy on accepted windows | %.1f%% |\n", accepted ? 100.0 * accepted_ok / accepted : 0);
      std::fprintf(o, "| Accuracy on flagged windows | %.1f%% |\n", rejected ? 100.0 * rejected_ok / rejected : 0);
      std::fprintf(o, "| Rumble-strip (unseen surface) windows flagged | %.1f%% |\n",
                   d.test_ood.empty() ? 0 : 100.0 * ood_flagged / d.test_ood.size());
      std::fprintf(o, "| Rumble-strip windows flagged by 1 model + confidence gate only | %.1f%% |\n",
                   d.test_ood.empty() ? 0 : 100.0 * single_ood_flagged / d.test_ood.size());
      std::fprintf(o, "| Ambiguous (transition) windows flagged | %.1f%% |\n",
                   d.test_ambiguous.empty() ? 0 : 100.0 * amb_flagged / d.test_ambiguous.size());
      if (!d.test_ood.empty()) {
        std::fprintf(o, "| AUROC unseen-surface detection, ensemble mutual information | %.3f |\n", auroc(mi_in, mi_ood));
        std::fprintf(o, "| AUROC, ensemble 1 - confidence | %.3f |\n", auroc(inv_conf_in, inv_conf_ood));
        std::fprintf(o, "| AUROC, novelty (Mahalanobis) | %.3f |\n", auroc(nov_in, nov_ood));
        std::fprintf(o, "| AUROC, deployed rule (all three signals) | %.3f |\n", auroc(combo_in, combo_ood));
        std::fprintf(o, "| AUROC, single model 1 - confidence (simplified #8) | %.3f |\n",
                     auroc(single_inv_conf_in, single_inv_conf_ood));
      }
      std::fprintf(o, "\nAll numbers come from simulated data. Retrain on recordings from the real cart\n"
                      "(log raw -> replay -> dataset) before trusting them.\n");
    }
    if (f) std::fclose(f);
  }
  std::fprintf(stderr, "wrote %s, %s, %s\n", model_path.c_str(), vectors_path.c_str(), report_path.c_str());
  return 0;
}
