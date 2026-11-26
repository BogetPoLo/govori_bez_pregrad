"""Run the desktop control clip using an explicitly supplied preprocessing recipe."""
import argparse
import hashlib
import json
import math
from array import array
from pathlib import Path
import subprocess
import sys
import tempfile


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_frames(clip, width, height, *, letterbox=False):
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(clip),
    ])
    timestamps = [
        round(float(frame["best_effort_timestamp_time"]) * 1_000_000_000)
        for frame in json.loads(probe)["frames"]
    ]
    if letterbox:
        filters = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=bilinear,"
            f"format=rgb24,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x727272"
        )
    else:
        filters = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=bilinear,"
            f"format=rgb24,crop={width}:{height}"
        )
    pixels = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(clip), "-map", "0:v:0", "-vf", filters,
        "-fps_mode", "passthrough", "-f", "rawvideo", "-",
    ])
    stride = width * height * 3
    if len(pixels) != len(timestamps) * stride:
        raise ValueError("Decoded frames and timestamps do not match")
    frames = memoryview(pixels)
    return timestamps, [frames[i:i + stride] for i in range(0, len(pixels), stride)]


def pack_tensor(timestamps, frames, recipe):
    count = recipe["frames"]
    if count < 1 or not frames:
        raise ValueError("No frames to sample")
    start, end = timestamps[0], timestamps[-1]
    selected = []
    for index in range(count):
        target = start + (end - start) * index // max(count - 1, 1)
        nearest = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target))
        selected.append(frames[nearest])

    layout = recipe["layout"]
    if layout in ("N1CTHW", "NCTHW"):
        planes = ((channel, frame) for channel in range(3) for frame in selected)
        batch = [1, 1] if layout == "N1CTHW" else [1]
        shape = batch + [3, count, recipe["height"], recipe["width"]]
    elif layout == "NTCHW":
        planes = ((channel, frame) for frame in selected for channel in range(3))
        shape = [1, count, 3, recipe["height"], recipe["width"]]
    else:
        raise ValueError(f"Unsupported layout {layout}")

    values = array("f")
    for channel, frame in planes:
        mean, std = recipe["mean"][channel], recipe["std"][channel]
        values.extend((intensity - mean) / std for intensity in frame[channel::3])
    if sys.byteorder != "little":
        values.byteswap()
    return values, shape


def household_prediction(scores, classifier):
    features = [math.log1p(267.0 * score) for score in scores]
    logits = [
        bias + sum(weight * feature for weight, feature in zip(row, features, strict=True))
        for row, bias in zip(classifier["weights"], classifier["bias"], strict=True)
    ]
    index = max(range(len(logits)), key=logits.__getitem__)
    scaled = [math.exp(logit - logits[index]) for logit in logits]
    total = sum(scaled)
    candidates = [
        {"label": label, "confidence": value / total}
        for label, value in zip(classifier["labels"], scaled, strict=True)
        if label is not None
    ]
    candidates.sort(key=lambda candidate: candidate["confidence"], reverse=True)
    count = 1 if candidates[0]["confidence"] == 1.0 else 3
    approximate = (
        classifier["labels"][index] is None
        or scaled[index] / total < classifier["minimumConfidence"]
    )
    return candidates[:count], approximate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ort-jar", type=Path, required=True)
    parser.add_argument("--java", default="java")
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--expected")
    parser.add_argument("--save-tensor", type=Path)
    parser.add_argument("--save-scores", type=Path)
    args = parser.parse_args()

    java = [args.java, "-cp", str(args.ort_jar), str(Path(__file__).with_name("OnnxProbe.java"))]
    model_hash = sha256(args.model)
    print(f"Model SHA-256: {model_hash}", file=sys.stderr)
    if not args.clip:
        subprocess.run([*java, str(args.model)], check=True)
        return
    if not args.manifest:
        parser.error("--manifest is required for a video run")

    recipe = json.loads(args.manifest.read_text(encoding="utf-8"))
    slovo = recipe["recipe"] == "rgb-letterbox-bilinear-uniform-v1"
    if not slovo and (
        recipe["recipe"] != "rgb-center-crop-bilinear-uniform-v1" or recipe["modelSha256"] != model_hash
    ):
        raise ValueError("The recipe does not match this model")
    timestamps, frames = extract_frames(args.clip, recipe["width"], recipe["height"], letterbox=slovo)
    values, shape = pack_tensor(timestamps, frames, recipe)
    tensor = values.tobytes()
    if args.save_tensor:
        args.save_tensor.write_bytes(tensor)
    with tempfile.TemporaryDirectory(prefix="signflow-") as directory:
        path = Path(directory) / "clip.f32"
        path.write_bytes(tensor)
        output = subprocess.check_output([
            *java, str(args.model), str(path), ",".join(map(str, shape)),
        ], text=True)
    scores = json.loads(output)
    class_count = 1001 if slovo else 267
    if len(scores) != class_count or not all(math.isfinite(score) for score in scores):
        raise ValueError(f"Expected {class_count} finite class scores")
    if args.save_scores:
        args.save_scores.write_text(json.dumps(scores), encoding="utf-8")
    index = max(range(len(scores)), key=scores.__getitem__)
    report = {"clipSha256": sha256(args.clip), "decodedFrames": len(frames),
              "shape": shape, "index": index, "score": scores[index]}
    if slovo:
        catalog = args.catalog or Path(__file__).resolve().parents[2] / "app/src/main/assets/phrases/catalog.json"
        entries = json.loads(catalog.read_text(encoding="utf-8"))
        entry = max(entries, key=lambda entry: scores[entry["classIndex"]])
        index = entry["classIndex"]
        report.update(label=entry["word"], index=index, score=scores[index])
    elif args.labels:
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        if sha256(args.labels) != recipe["labelsSha256"] or len(labels) != 267:
            raise ValueError("The label map does not match this recipe")
        report["label"] = labels[index]
    elif "householdClassifier" in recipe or "prototype" in recipe:
        prototype = recipe.get("prototype")
        classifier = None
        if "householdClassifier" in recipe:
            path = args.manifest.with_name(recipe["householdClassifier"])
            classifier = json.loads(path.read_text(encoding="utf-8"))
            if classifier["modelSha256"] != model_hash:
                raise ValueError("The household classifier does not match this model")
        if prototype is not None:
            catalog = args.catalog or Path(__file__).with_name("legacy") / "catalog.json"
            words = {row["word"].casefold() for row in json.loads(catalog.read_text(encoding="utf-8"))}
            if classifier is not None:
                trained_words = {word.casefold() for word in classifier["labels"] if word is not None}
                if not words <= trained_words:
                    classifier = None
        if classifier is not None:
            candidates, approximate = household_prediction(scores, classifier)
            report.update(candidates[0], candidates=candidates, approximate=approximate or len(frames) < recipe["frames"])
        else:
            labels = {
                entry["classIndex"]: entry["label"]
                for entry in prototype.get("labels", [prototype])
                if entry["label"].casefold() in words
            }
            report["label"] = labels.get(index) if scores[index] >= prototype["minimumScore"] else None
    print(json.dumps(report, ensure_ascii=False))
    if args.expected and report.get("label") != args.expected:
        raise SystemExit("Expected label was not reproduced")


if __name__ == "__main__":
    main()
