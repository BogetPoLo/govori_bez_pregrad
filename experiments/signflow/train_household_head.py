"""Fit a small classifier on frozen SignFlow-R scores from the Slovo signer split."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


OTHER = "__other__"
SUPPORTED = ("Комната", "Жарить", "Собака", "Аккуратный", "Комар", "Красивый")
FEATURE_COUNT = 267
FEATURE_SCALE = 267.0
MINIMUM_CONFIDENCE = 0.60


def read_scores(paths, classes):
    class_indices = {name.casefold(): index for index, name in enumerate(classes)}
    inputs, targets, splits, signers, clips = [], [], [], [], []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for word, clip, split, signer, *values in csv.reader(stream, delimiter="\t"):
                label = word.strip().casefold()
                if label not in class_indices:
                    raise ValueError(f"Unknown label {word!r} for {clip}")
                target = class_indices[label]
                if len(values) != FEATURE_COUNT or split not in ("train", "test"):
                    raise ValueError(f"Invalid scores for {clip}")
                inputs.append(values)
                targets.append(target)
                splits.append(split)
                signers.append(signer)
                clips.append(clip)
    if len(set(clips)) != len(clips):
        raise ValueError("Duplicate input clips")
    if {s for s, p in zip(signers, splits) if p == "train"} & {
        s for s, p in zip(signers, splits) if p == "test"
    }:
        raise ValueError("A signer is present in both splits")
    scores = np.asarray(inputs, dtype=np.float64)
    if not np.isfinite(scores).all() or not np.allclose(scores.sum(axis=1), 1.0, atol=0.001):
        raise ValueError("Expected finite SignFlow-R probabilities")
    return scores, np.asarray(targets), np.asarray(splits), clips


def probabilities(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def fit(inputs, targets, class_count):
    y = np.eye(class_count)[targets]
    counts = np.bincount(targets, minlength=class_count)
    if min(counts) < 3:
        raise ValueError("Need at least three training examples per class")
    sample_weight = len(targets) / (class_count * counts[targets])
    weights = np.zeros((inputs.shape[1], class_count))
    bias = np.zeros(class_count)
    for _ in range(1800):
        error = (probabilities(inputs @ weights + bias) - y) * sample_weight[:, None]
        weights -= 0.12 * (inputs.T @ error / len(targets) + 0.018 * weights)
        bias -= 0.12 * error.mean(axis=0)
    return weights, bias


def evaluate(probability, targets, classes, reject_unknown):
    best = probability.argmax(axis=1)
    predictions = np.where(probability.max(axis=1) >= MINIMUM_CONFIDENCE, best, 0) if reject_unknown else best
    counts = Counter(zip(targets.tolist(), predictions.tolist()))
    return {
        name: {
            "samples": int((targets == index).sum()),
            "correct": counts[index, index],
            "wrong": {classes[j]: counts[index, j] for j in range(len(classes))
                      if j != index and counts[index, j]},
        }
        for index, name in enumerate(classes)
    }


def train(paths, model_path, words, require_other=False):
    labels = (OTHER, *words)
    if not words or "" in words or len({word.casefold() for word in labels}) != len(labels):
        raise ValueError("Expected nonempty and unique catalog words")
    scores, targets, splits, _ = read_scores(paths, labels)
    has_other = require_other or 0 in targets
    classes = labels if has_other else words
    targets = targets if has_other else targets - 1
    inputs = np.log1p(FEATURE_SCALE * scores)
    training = splits == "train"
    testing = ~training
    for split, rows, minimum in (("train", training, 3), ("test", testing, 1)):
        counts = np.bincount(targets[rows], minlength=len(classes))
        missing = [name for name, count in zip(classes, counts) if count < minimum]
        if missing:
            raise ValueError(f"Need at least {minimum} {split} examples for: {', '.join(missing)}")
    weights, bias = fit(inputs[training], targets[training], len(classes))
    test_logits = inputs[testing] @ weights + bias
    test_targets = targets[testing]
    result = evaluate(probabilities(test_logits), test_targets, classes, has_other)
    offset = int(has_other)
    known = test_targets >= offset
    guesses = test_logits[known, offset:].argmax(axis=1) + offset
    expected = test_targets[known]
    report = {
        "trainSize": int(training.sum()),
        "test": result,
        "appTop1": {
            "correct": int((guesses == expected).sum()),
            "total": int(known.sum()),
            "words": {
                name: {
                    "correct": int(((guesses == target) & (expected == target)).sum()),
                    "total": int((expected == target).sum()),
                }
                for target, name in enumerate(classes) if target >= offset
            },
        },
    }
    if not has_other:
        # Preserve the app's reserved null row when training on catalog gestures only.
        weights = np.column_stack((np.zeros(weights.shape[0]), weights))
        bias = np.insert(bias, 0, 0.0)
    with model_path.open("rb") as model:
        model_hash = hashlib.file_digest(model, "sha256").hexdigest()
    artifact = {
        "recipe": "log1p-signflow-probabilities-v1",
        "modelSha256": model_hash,
        "minimumConfidence": MINIMUM_CONFIDENCE,
        "labels": [None, *words],
        "bias": bias.round(8).tolist(),
        "weights": weights.T.round(8).tolist(),
    }
    return artifact, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, nargs="+",
                        help="TSV rows: label, clip ID, train/test, signer ID, 267 ONNX probabilities")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, help="JSON catalog whose words the classifier must cover")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    words = SUPPORTED
    if args.catalog:
        records = json.loads(args.catalog.read_text(encoding="utf-8"))
        words = tuple(record["word"].strip() for record in records)
    artifact, report = train(args.scores, args.model, words, require_other=args.catalog is None)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
