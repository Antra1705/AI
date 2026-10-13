#!/usr/bin/env python3
"""Evaluate the DEPLOYED model (adaptive_suspension/src/core/model_data.cpp) on datasets.

    .venv/bin/python tools/eval_model.py host/data/nominal_test.csv host/data/shifted.csv

Uses the same int8 arithmetic as the ESP32 (via train_torch's reference
functions). Prints accuracy, macro-F1, the confusion matrix, "impact recall"
(a true breaker or pothole recognised as breaker OR pothole - both get full
stiffness, so this is what the suspension depends on), uncertainty and, for
datasets with unseen surfaces, how well those are flagged.
"""

import os
import re
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_torch as tt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NAMES = ["smooth", "gravel", "breaker", "pothole"]


def floats(text, name):
    m = re.search(r"%s\[\d+\] = \{([^}]*)\}" % re.escape(name), text)
    return np.array([float(v.rstrip("f")) for v in m.group(1).split(",")], dtype=np.float32)


def ints(text, name):
    m = re.search(r"%s\[\d+\] = \{([^}]*)\}" % re.escape(name), text)
    return np.array([int(v) for v in m.group(1).split(",")], dtype=np.int64)


def load_model(path):
    text = open(path).read()
    mean, std = floats(text, "kMean"), floats(text, "kStd")
    f = len(mean)
    model = {
        "mean": mean, "std": std,
        "class_means": floats(text, "kClassMeans").reshape(4, f),
        "inv_cov": floats(text, "kInvCov").reshape(-1, f, f) if "kInvCov" in text else None,
        "nets": [],
    }
    for m in re.finditer(r"\{\{(\{[^}]*\}(?:, \{[^}]*\})*)\}\}", text):
        net = []
        for layer in re.finditer(r"\{(\d+), (\d+), kW(\d+)_(\d+), kB\d+_\d+, ([-0-9.e]+)f, ([-0-9.e]+)f\}", m.group(1)):
            i, o, mi, li = int(layer.group(1)), int(layer.group(2)), layer.group(3), layer.group(4)
            qw = ints(text, "kW%s_%s" % (mi, li)).reshape(o, i)
            qb = ints(text, "kB%s_%s" % (mi, li))
            net.append((i, o, qw, qb, np.float32(layer.group(5)), np.float32(layer.group(6))))
        model["nets"].append(net)
    head = re.search(r"kModel = \{(\d+), kNets, kMean, kStd, ([-0-9.e]+)f, ([-0-9.e]+)f, ([-0-9.e]+)f,\s*"
                     r"kClassMeans, (kInvCov\w*), ([-0-9.e]+)f", text)
    model["T"], model["mi_thr"], model["conf_thr"] = float(head.group(2)), float(head.group(3)), float(head.group(4))
    model["nov_thr"] = float(head.group(6))
    return model


def novelty(x, model):
    cm, ic = model["class_means"], model["inv_cov"]
    if ic.shape[0] == 1:
        return tt.novelty(x, cm, ic[0])
    d = x[:, None, :] - cm[None, :, :]
    return np.einsum("nci,cij,ncj->nc", d, ic, d).min(axis=1)  # per-class covariance


def predict(model, feats):
    x = ((feats - model["mean"]) / model["std"]).astype(np.float32)
    p, mi = tt.ensemble([tt.int8_forward(net, x) for net in model["nets"]], model["T"])
    nov = novelty(x, model)
    unsure = (p.max(axis=1) < model["conf_thr"]) | (mi > model["mi_thr"]) | (nov > model["nov_thr"])
    return p, mi, nov, unsure


def report(name, path, model):
    runs, labels, aug, feats, _ = tt.load(path)
    feats = feats[:, :len(model["mean"])]
    real = aug == 0
    known = real & (labels >= 0) & (labels < 4)
    p, mi, nov, unsure = predict(model, feats)
    y, pred = labels[known], p[known].argmax(axis=1)
    impact = np.isin(y, [2, 3])
    print("\n== %s (%d windows)" % (name, known.sum()))
    print("accuracy %.1f%%, macro-F1 %.3f, flagged uncertain %.1f%%"
          % (100 * (pred == y).mean(), f1_score(y, pred, labels=range(4), average="macro"), 100 * unsure[known].mean()))
    print("impact recall (true breaker/pothole called breaker OR pothole): %.1f%%"
          % (100 * np.isin(pred[impact], [2, 3]).mean()))
    print("confusion (rows = true, columns = predicted):")
    print("%10s" % "" + "".join("%9s" % n for n in NAMES))
    for c in range(4):
        print("%10s" % NAMES[c] + "".join("%9d" % ((y == c) & (pred == k)).sum() for k in range(4)))
    ood = real & (labels == tt.LABEL_OOD)
    if ood.any():
        score = np.r_[unsure[known].astype(float), unsure[ood].astype(float)]
        cont = np.maximum(np.maximum(mi / model["mi_thr"], (1 - p.max(axis=1)) / (1 - model["conf_thr"])), nov / model["nov_thr"])
        auroc = roc_auc_score(np.r_[np.zeros(known.sum()), np.ones(ood.sum())], np.r_[cont[known], cont[ood]])
        print("unseen surface (rumble strip): %.1f%% flagged, detection AUROC %.3f" % (100 * unsure[ood].mean(), auroc))
        del score


def main(argv):
    if not argv:
        sys.exit(__doc__)
    model = load_model(os.path.join(ROOT, "adaptive_suspension", "src", "core", "model_data.cpp"))
    print("deployed model: %d members, %d features" % (len(model["nets"]), len(model["mean"])))
    for path in argv:
        report(os.path.basename(path), path, model)


if __name__ == "__main__":
    main(sys.argv[1:])
