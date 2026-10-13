#!/usr/bin/env python3
"""Train the road-surface classifier with PyTorch and export it for the ESP32.

This is the readable reference pipeline. It produces exactly the same artefacts
as the dependency-free C++ trainer (host/train) and both are checked by the same
parity test against the on-device int8 code.

    .venv/bin/python tools/train_torch.py host/data/dataset.csv
    make -C host test          # parity test: on-device int8 code == this script's model

Pipeline, per ensemble member (bootstrap resample, different seed):
  1. float training: Adam, class-weighted cross-entropy, cosine learning rate
  2. calibrate the int8 input scale of every layer (99.99th percentile of |input|)
  3. quantization-aware fine-tuning: weights, biases and activations are rounded
     to int8 in the forward pass; gradients pass straight through (STE), except
     where an activation was clipped
Then for the ensemble:
  4. temperature scaling on the validation runs
  5. uncertainty thresholds (confidence, ensemble disagreement, Mahalanobis novelty)
  6. report: rules vs float vs post-training quantization vs QAT, calibration,
     unseen-surface detection
  7. export: adaptive_suspension/src/core/model_data.cpp + host/tests/model_vectors.h

Data is split by run (whole drives), never by window.
"""

import argparse
import csv
import math
import os
import re

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CLASSES = ["smooth", "gravel", "breaker", "pothole"]
C = 4
F = 16
LABEL_OOD = 4


# ---------------------------------------------------------------- data

def load(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader]
    col = {name: i for i, name in enumerate(header)}
    f0 = col["log_rms"]
    data = np.array([[float(v) for v in r] for r in rows], dtype=np.float64)
    runs = data[:, col["run"]].astype(int)
    labels = data[:, col["label"]].astype(int)
    feats = data[:, f0:f0 + F].astype(np.float32)
    return runs, labels, feats


def split_of(run):
    r = run % 20
    return np.where(r < 14, 0, np.where(r < 17, 1, 2))  # 70 / 15 / 15 % of runs


# ---------------------------------------------------------------- int8 helpers (identical rules to quantizeS8 on the ESP32)

def round_half_away(x):
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def quantize_s8(x, scale):
    return torch.clamp(round_half_away(x / scale), -127, 127)


def np_quantize_s8(x, scale):
    v = np.float32(x) / np.float32(scale)
    v = np.where(v >= 0, np.floor(v + np.float32(0.5)), np.ceil(v - np.float32(0.5)))
    return np.clip(v, -127, 127).astype(np.int32)


# ---------------------------------------------------------------- model

class Mlp(torch.nn.Module):
    """16 -> H -> H -> 4, ReLU. In QAT mode every layer sees int8-rounded inputs and weights."""

    def __init__(self, hidden):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(F, hidden), torch.nn.Linear(hidden, hidden),
                                           torch.nn.Linear(hidden, C)])
        for layer in self.layers:  # He-uniform weights, zero bias (as the C++ trainer)
            torch.nn.init.uniform_(layer.weight, -math.sqrt(6.0 / layer.in_features), math.sqrt(6.0 / layer.in_features))
            torch.nn.init.zeros_(layer.bias)
        self.in_scale = [0.0, 0.0, 0.0]
        self.qat = False

    def forward(self, x, record=None):
        a = x
        for i, layer in enumerate(self.layers):
            if record is not None:
                record[i].append(a.detach().abs().flatten())
            if self.qat:
                s = self.in_scale[i]
                q = quantize_s8(a, s) * s
                inside = a.abs() <= 127 * s
                a = torch.where(inside, a + (q - a).detach(), q.detach())  # STE, no gradient where clipped
                ws = (layer.weight.abs().max() / 127).detach()
                w = layer.weight + (quantize_s8(layer.weight, ws) * ws - layer.weight).detach()
                bs = s * ws
                b = layer.bias + (round_half_away(layer.bias / bs) * bs - layer.bias).detach()
                z = a @ w.T + b
            else:
                z = layer(a)
            a = torch.relu(z) if i < 2 else z
        return a


def train_epochs(model, x, y, weights, idx, epochs, lr, seed):
    decay = [p for n, p in model.named_parameters() if n.endswith("weight")]
    rest = [p for n, p in model.named_parameters() if not n.endswith("weight")]
    opt = torch.optim.Adam([{"params": decay, "weight_decay": 1e-4}, {"params": rest, "weight_decay": 0.0}], lr=lr)
    gen = torch.Generator().manual_seed(seed)
    steps_per_epoch = math.ceil(len(idx) / 64)
    for epoch in range(epochs):
        for g in opt.param_groups:  # cosine decay, once per epoch
            g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * epoch / epochs))
        order = idx[torch.randperm(len(idx), generator=gen)]
        total, wsum = 0.0, 0.0
        for k in range(steps_per_epoch):
            b = order[k * 64:(k + 1) * 64]
            logits = model(x[b])
            ce = torch.nn.functional.cross_entropy(logits, y[b], reduction="none")
            w = weights[y[b]]
            loss = (w * ce).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float((w * ce).sum().detach())
            wsum += float(w.sum())
    return total / wsum


def calibrate_scales(model, x):
    model.qat = False
    record = [[], [], []]
    with torch.no_grad():
        model(x[::2], record=record)
    for i in range(3):
        mags = torch.cat(record[i]).numpy()
        model.in_scale[i] = float(np.float32(max(np.quantile(mags, 0.9999), 1e-3)) / np.float32(127))


def export_int8(model):
    """-> list of (in, out, int8 weights, int32 bias, in_scale, w_scale), as qnetForward expects."""
    out = []
    for i, layer in enumerate(model.layers):
        w = layer.weight.detach().numpy().astype(np.float32)
        b = layer.bias.detach().numpy().astype(np.float32)
        ws = np.float32(np.abs(w).max()) / np.float32(127)
        qw = np_quantize_s8(w, ws)
        bs = np.float64(np.float32(model.in_scale[i])) * np.float64(ws)
        qb = np.where(b / bs >= 0, np.floor(b / bs + 0.5), np.ceil(b / bs - 0.5)).astype(np.int64)
        out.append((layer.in_features, layer.out_features, qw, qb, np.float32(model.in_scale[i]), ws))
    return out


def int8_forward(net, x):
    """Batch version of qnetForward (the ESP32 code): int8 inputs, int32 sums, one float rescale."""
    a = x.astype(np.float32)
    for i, (_, _, qw, qb, s_in, ws) in enumerate(net):
        q = np_quantize_s8(a, s_in)
        acc = q.astype(np.int64) @ qw.T.astype(np.int64) + qb
        y = acc.astype(np.float32) * (np.float32(s_in) * np.float32(ws))
        a = np.maximum(y, 0) if i < 2 else y
    return a


# ---------------------------------------------------------------- ensemble outputs

def softmax(z, t=1.0):
    z = z / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def entropy(p):
    return -(np.where(p > 1e-7, p * np.log(np.maximum(p, 1e-12)), 0)).sum(axis=1)


def ensemble(logits_list, t):
    probs = [softmax(z, t) for z in logits_list]
    mean = np.mean(probs, axis=0)
    mi = np.maximum(entropy(mean) - np.mean([entropy(p) for p in probs], axis=0), 0)
    return mean, mi


def novelty(x, class_means, inv_cov):
    d = x[:, None, :] - class_means[None, :, :]
    return np.einsum("nci,ij,ncj->nc", d, inv_cov, d).min(axis=1)


def ece(conf, correct, bins=15):
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
               for b in range(bins) if (idx == b).any())


# ---------------------------------------------------------------- rule-based baseline (same rules as classifier.cpp)

def rule_thresholds():
    text = open(os.path.join(ROOT, "adaptive_suspension", "src", "config.h")).read()
    get = lambda name: float(re.search(r"\b%s\s*=\s*([0-9.]+)" % name, text).group(1))
    return {k: get(k) for k in ["POTHOLE_JERK", "BREAKER_P2P", "GRAVEL_RMS", "GRAVEL_OSC", "SMOOTH_JERK", "SMOOTH_P2P"]}


def rules(feats, t):
    rms, p2p = np.exp(feats[:, 0]) - 1e-3, np.exp(feats[:, 1]) - 1e-3
    jerk, osc = np.exp(feats[:, 4]) - 1.0, feats[:, 6]
    out = np.full(len(feats), -1)
    for i in range(len(feats)):
        if rms[i] > t["GRAVEL_RMS"] and osc[i] >= t["GRAVEL_OSC"]: out[i] = 1
        elif jerk[i] > t["POTHOLE_JERK"]: out[i] = 3
        elif p2p[i] > t["BREAKER_P2P"]: out[i] = 2
        elif jerk[i] < t["SMOOTH_JERK"] and p2p[i] < t["SMOOTH_P2P"]: out[i] = 0
    return out


# ---------------------------------------------------------------- export files

def lit(v):
    """C++ float literal with a guaranteed decimal point."""
    return "%#.9gf" % float(v)


def write_model(path, source, members, hidden, mean, std, class_means, inv_cov, nets, t, mi_thr, conf_thr, nov_thr, acc, f1):
    with open(path, "w") as f:
        f.write("// GENERATED by tools/train_torch.py from %s - do not edit by hand.\n" % source)
        f.write("// %d-member int8 ensemble, %d-%d-%d-%d, QAT. Test accuracy %.1f%%, macro-F1 %.3f.\n"
                % (members, F, hidden, hidden, C, 100 * acc, f1))
        f.write('#include "ml_model.h"\n\nnamespace {\n\n')
        f.write("const float kMean[%d] = {%s};\n" % (F, ", ".join(lit(v) for v in mean)))
        f.write("const float kStd[%d] = {%s};\n" % (F, ", ".join(lit(v) for v in std)))
        f.write("const float kClassMeans[%d] = {%s};\n" % (C * F, ", ".join(lit(v) for v in class_means.ravel())))
        f.write("const float kInvCov[%d] = {%s};\n\n" % (F * F, ", ".join(lit(v) for v in inv_cov.ravel())))
        for m, net in enumerate(nets):
            for l, (_, _, qw, qb, _, _) in enumerate(net):
                f.write("const int8_t kW%d_%d[%d] = {%s};\n" % (m, l, qw.size, ",".join(str(int(v)) for v in qw.ravel())))
                f.write("const int32_t kB%d_%d[%d] = {%s};\n" % (m, l, qb.size, ",".join(str(int(v)) for v in qb)))
        f.write("\nconst QNet kNets[%d] = {\n" % members)
        for m, net in enumerate(nets):
            layers = ", ".join("{%d, %d, kW%d_%d, kB%d_%d, %s, %s}" % (i, o, m, l, m, l, lit(s), lit(ws))
                               for l, (i, o, _, _, s, ws) in enumerate(net))
            f.write("    {{%s}},\n" % layers)
        f.write("};\n\nconst MlEnsemble kModel = {%d, kNets, kMean, kStd, %s, %s, %s,\n"
                % (members, lit(t), lit(mi_thr), lit(conf_thr)))
        f.write("                             kClassMeans, kInvCov, %s,\n" % lit(nov_thr))
        f.write('                             "%d x MLP %d-%d-%d-%d int8 QAT (PyTorch), test acc %.1f%%"};\n\n'
                % (members, F, hidden, hidden, C, 100 * acc))
        f.write("}  // namespace\n\nconst MlEnsemble& deployedModel() { return kModel; }\n")


def write_vectors(path, feats, probs, mi, nov):
    with open(path, "w") as f:
        f.write("// GENERATED by tools/train_torch.py. Expected ensemble outputs computed by PyTorch's\n"
                "// fake-quantized forward pass; the on-device int8 code must reproduce them.\n#pragma once\n\n")
        n = len(feats)
        f.write("static const int kVectorCount = %d;\nstatic const float kVectorFeatures[%d][%d] = {\n" % (n, n, F))
        for row in feats:
            f.write("    {%s},\n" % ", ".join(lit(v) for v in row))
        f.write("};\nstatic const float kVectorProbs[%d][%d] = {\n" % (n, C))
        for row in probs:
            f.write("    {%s},\n" % ", ".join(lit(v) for v in row))
        f.write("};\nstatic const float kVectorMI[%d] = {%s};\n" % (n, ", ".join(lit(v) for v in mi)))
        f.write("static const float kVectorNovelty[%d] = {%s};\n" % (n, ", ".join(lit(v) for v in nov)))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--qat-epochs", type=int, default=12)
    ap.add_argument("--model", default=os.path.join(ROOT, "adaptive_suspension", "src", "core", "model_data.cpp"))
    ap.add_argument("--vectors", default=os.path.join(ROOT, "host", "tests", "model_vectors.h"))
    ap.add_argument("--report", default=os.path.join(ROOT, "docs", "model_report.md"))
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    runs, labels, feats = load(args.dataset)
    split = split_of(runs)
    known = (labels >= 0) & (labels < C)
    tr, va, te = np.where(known & (split == 0))[0], np.where(known & (split == 1))[0], np.where(known & (split == 2))[0]
    te_ood = np.where((labels == LABEL_OOD) & (split == 2))[0]
    te_amb = np.where((labels < 0) & (split == 2))[0]
    print("windows: train %d, val %d, test %d, test-OOD %d, test-ambiguous %d" % (len(tr), len(va), len(te), len(te_ood), len(te_amb)))

    mean = feats[tr].mean(axis=0)
    std = np.maximum(feats[tr].std(axis=0), 1e-3).astype(np.float32)
    x = ((feats - mean) / std).astype(np.float32)

    # Novelty model: class-conditional Gaussian, shared covariance (LDA-style)
    class_means = np.stack([x[tr][labels[tr] == c].mean(axis=0) for c in range(C)]).astype(np.float32)
    d = x[tr] - class_means[labels[tr]]
    cov = d.T.astype(np.float64) @ d.astype(np.float64) / len(tr) + 1e-3 * np.eye(F)
    inv_cov = np.linalg.inv(cov).astype(np.float32)

    counts = np.bincount(labels[tr], minlength=C)
    weights = torch.tensor(np.minimum(20.0, len(tr) / (C * np.maximum(counts, 1))), dtype=torch.float32)
    xt, yt = torch.from_numpy(x), torch.from_numpy(np.where(known, labels, 0))

    float_nets, qat_nets, qat_models = [], [], []
    for m in range(args.members):
        torch.manual_seed(1234 + 97 * m)
        rng = np.random.default_rng(1234 + 97 * m)
        boot = torch.from_numpy(rng.choice(tr, size=len(tr), replace=True))
        model = Mlp(args.hidden)
        loss_f = train_epochs(model, xt, yt, weights, boot, args.epochs, 3e-3, 1234 + 97 * m)
        calibrate_scales(model, xt[torch.from_numpy(tr)])
        float_nets.append(export_int8(model))  # post-training quantization of the float model
        float_model_logits = model  # kept for the float baseline below
        if m == 0:
            with torch.no_grad():
                float_logits0 = float_model_logits(xt).numpy()
        model.qat = True
        loss_q = train_epochs(model, xt, yt, weights, boot, args.qat_epochs, 5e-4, 4321 + 97 * m)
        qat_nets.append(export_int8(model))
        qat_models.append(model)
        print("member %d/%d: float loss %.4f, QAT loss %.4f" % (m + 1, args.members, loss_f, loss_q))

    qat_logits = [int8_forward(net, x) for net in qat_nets]
    ptq_logits = [int8_forward(net, x) for net in float_nets]

    # Temperature scaling on validation runs
    best_t, best_nll = 1.0, float("inf")
    for t in np.arange(0.5, 5.0001, 0.05):
        p, _ = ensemble([z[va] for z in qat_logits], t)
        nll = -np.log(np.maximum(p[np.arange(len(va)), labels[va]], 1e-7)).sum()
        if nll < best_nll:
            best_t, best_nll = float(t), nll

    p_all, mi_all = ensemble(qat_logits, best_t)
    conf_all, top_all = p_all.max(axis=1), p_all.argmax(axis=1)
    nov_all = novelty(x, class_means, inv_cov)
    ok_va = top_all[va] == labels[va]
    conf_thr = max(0.5, float(np.quantile(conf_all[va][ok_va], 0.025, method="lower")))
    mi_thr = max(1e-3, float(np.quantile(mi_all[va][ok_va], 0.975, method="lower")))
    nov_thr = float(np.quantile(nov_all[va], 0.99, method="lower"))
    unsure = (conf_all < conf_thr) | (mi_all > mi_thr) | (nov_all > nov_thr)

    def scores(pred):
        y = labels[te]
        recall = [(pred[y == c] == c).mean() for c in range(C)]
        return (pred == y).mean(), f1_score(y, pred, labels=range(C), average="macro", zero_division=0), recall

    results = [
        ("Rule-based (config.h thresholds)", scores(rules(feats[te], rule_thresholds()))),
        ("1 MLP, float32", scores(float_logits0[te].argmax(axis=1))),
        ("1 MLP, int8 post-training quantization", scores(ptq_logits[0][te].argmax(axis=1))),
        ("1 MLP, int8 quantization-aware training", scores(qat_logits[0][te].argmax(axis=1))),
        ("Ensemble, int8 post-training quantization", scores(ensemble([z[te] for z in ptq_logits], best_t)[0].argmax(axis=1))),
        ("**Ensemble, int8 QAT (deployed)**", scores(top_all[te])),
    ]
    acc, f1, _ = results[-1][1]
    p_raw, _ = ensemble([z[te] for z in qat_logits], 1.0)
    single_conf, _ = ensemble([qat_logits[0]], best_t)
    single_conf = single_conf.max(axis=1)
    combo = np.maximum(np.maximum(mi_all / mi_thr, (1 - conf_all) / (1 - conf_thr)), nov_all / nov_thr)

    def auroc(score):
        if not len(te_ood):
            return float("nan")
        y = np.r_[np.zeros(len(te)), np.ones(len(te_ood))]
        return roc_auc_score(y, np.r_[score[te], score[te_ood]])

    lines = ["# Road-surface model report", "",
             "Generated by `tools/train_torch.py %s` (PyTorch). Data split by run: train %d / validation %d /"
             % (os.path.basename(args.dataset), len(tr), len(va)),
             "test %d windows; the test runs also have %d rumble-strip windows (a surface never seen in"
             % (len(te), len(te_ood)), "training) and %d ambiguous windows." % len(te_amb), "",
             "Model: %d x MLP %d-%d-%d-%d, int8 weights and activations, %d bytes of weights in total."
             % (args.members, F, args.hidden, args.hidden, C, args.members * (F * args.hidden + args.hidden ** 2 + args.hidden * C)),
             "", "## Accuracy on the test runs", "",
             "| Model | Accuracy | Macro-F1 | smooth | gravel | breaker | pothole |", "|---|---|---|---|---|---|---|"]
    for name, (a, f, r) in results:
        lines.append("| %s | %.1f%% | %.3f | %s |" % (name, 100 * a, f, " | ".join("%.1f%%" % (100 * v) for v in r)))
    lines += ["", "Per-class columns are recall. Rule-based 'unknown' answers count as wrong.", "",
              "Confusion matrix of the deployed model (rows = true, columns = predicted):", "",
              "|  | " + " | ".join(CLASSES) + " |", "|---|---|---|---|---|"]
    for c in range(C):
        row = [int(((labels[te] == c) & (top_all[te] == k)).sum()) for k in range(C)]
        lines.append("| %s | %s |" % (CLASSES[c], " | ".join(str(v) for v in row)))
    lines += ["", "## Calibration", "",
              "Temperature fitted on validation runs: T = %.2f." % best_t,
              "Expected calibration error on test: %.3f before, %.3f after temperature scaling."
              % (ece(p_raw.max(axis=1), p_raw.argmax(axis=1) == labels[te]), ece(conf_all[te], top_all[te] == labels[te])),
              "", "## Uncertainty and safety fallback", "",
              "A window is flagged *uncertain* when confidence < %.3f or mutual information > %.4f" % (conf_thr, mi_thr),
              "(2.5th / 97.5th percentiles of correctly classified validation windows), or novelty",
              "(squared Mahalanobis distance to the nearest class) > %.1f (99th percentile of validation)." % nov_thr,
              "", "| | Value |", "|---|---|",
              "| Test windows accepted (coverage) | %.1f%% |" % (100 * (~unsure[te]).mean()),
              "| Accuracy on accepted windows | %.1f%% |" % (100 * (top_all[te] == labels[te])[~unsure[te]].mean()),
              "| Accuracy on flagged windows | %.1f%% |" % (100 * (top_all[te] == labels[te])[unsure[te]].mean()),
              "| Rumble-strip (unseen surface) windows flagged | %.1f%% |" % (100 * unsure[te_ood].mean()),
              "| Rumble-strip windows flagged by 1 model + confidence gate only | %.1f%% |" % (100 * (single_conf[te_ood] < conf_thr).mean()),
              "| Ambiguous (transition) windows flagged | %.1f%% |" % (100 * unsure[te_amb].mean()),
              "| AUROC unseen-surface detection, ensemble mutual information | %.3f |" % auroc(mi_all),
              "| AUROC, ensemble 1 - confidence | %.3f |" % auroc(1 - conf_all),
              "| AUROC, novelty (Mahalanobis) | %.3f |" % auroc(nov_all),
              "| AUROC, deployed rule (all three signals) | %.3f |" % auroc(combo),
              "| AUROC, single model 1 - confidence (simplified #8) | %.3f |" % auroc(1 - single_conf),
              "", "All numbers come from simulated data. Retrain on recordings from the real cart",
              "(log raw -> replay --label -> this script) before trusting them.", ""]
    report = "\n".join(lines)
    print(report)

    write_model(args.model, os.path.basename(args.dataset), args.members, args.hidden, mean, std, class_means, inv_cov,
                qat_nets, best_t, mi_thr, conf_thr, nov_thr, acc, f1)

    # Parity vectors: PyTorch's own fake-quantized forward (independent of the int8 numpy path)
    rng = np.random.default_rng(7)
    pick = np.r_[np.concatenate([rng.choice(te[labels[te] == c], 8, replace=False) for c in range(C)]),
                 te_ood[:6]].astype(int)
    with torch.no_grad():
        zs = [mdl(xt[torch.from_numpy(pick)]).numpy() for mdl in qat_models]
    vp, vmi = ensemble(zs, best_t)
    write_vectors(args.vectors, feats[pick], vp, vmi, novelty(x[pick], class_means, inv_cov))
    with open(args.report, "w") as f:
        f.write(report)
    print("wrote %s, %s, %s" % (args.model, args.vectors, args.report))


if __name__ == "__main__":
    main()
