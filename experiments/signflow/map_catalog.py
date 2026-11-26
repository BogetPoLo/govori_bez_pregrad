"""Check direct SignFlow-R class mappings for the app's labeled reference clips."""

import argparse
import csv
import json
import math
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from control import extract_frames, pack_tensor, sha256


RECOGNITION = Path(__file__).with_name("legacy")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
CLASS_COUNT = 267


def write_scores(directory, clips, model, recipe, jar, java):
    tensors = directory / "tensors.tsv"
    with tensors.open("w", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        for number, (word, clip) in enumerate(clips):
            timestamps, frames = extract_frames(clip, recipe["width"], recipe["height"])
            values, shape = pack_tensor(timestamps, frames, recipe)
            tensor = directory / f"{number}.f32"
            tensor.write_bytes(values.tobytes())
            writer.writerow([word, str(number), str(tensor), "test", "reference"])
    scores = directory / "scores.tsv"
    with scores.open("w", encoding="utf-8") as stream:
        subprocess.run([
            java, "-cp", str(jar), str(Path(__file__).with_name("OnnxProbe.java")),
            str(model), str(tensors), ",".join(map(str, shape)), "--batch",
        ], stdout=stream, check=True)
    return scores


def checked_winners(path, words):
    names = {word.casefold(): word for word in words}
    winners = defaultdict(list)
    clips = set()
    with path.open(encoding="utf-8") as stream:
        for label, clip, _, _, *columns in csv.reader(stream, delimiter="\t"):
            word = names.get(label.strip().casefold())
            if word is None or clip in clips:
                raise ValueError(f"Unexpected label or repeated clip: {label}, {clip}")
            scores = list(map(float, columns))
            if (len(scores) != CLASS_COUNT or not all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
                    or not math.isclose(sum(scores), 1.0, abs_tol=0.001)):
                raise ValueError(f"Expected {CLASS_COUNT} probabilities from SignFlow-R for clip {clip}")
            clips.add(clip)
            index = max(range(CLASS_COUNT), key=scores.__getitem__)
            winners[word].append((index, scores[index]))
    return winners


def map_labels(words, prototype, winners):
    minimum = prototype["minimumScore"]
    existing = {entry["label"]: entry["classIndex"] for entry in prototype["labels"]}
    candidates = {}
    details = {}
    for word in words:
        results = winners.get(word, [])
        indices = {index for index, _ in results}
        if word in existing:
            details[word] = {"status": "existing", "classIndex": existing[word]}
        elif not results:
            details[word] = {"status": "missing_clips"}
        elif len(indices) != 1 or min(score for _, score in results) < minimum:
            details[word] = {"status": "uncertain", "predictions": results}
        else:
            candidates[word] = results[0][0]
            details[word] = {"status": "candidate", "clips": len(results), "classIndex": results[0][0]}

    counts = Counter([*existing.values(), *candidates.values()])
    for word, index in candidates.items():
        if counts[index] > 1:
            details[word]["status"] = "index_conflict"
        else:
            existing[word] = index
            details[word]["status"] = "added"
    labels = [{"classIndex": existing[word], "label": word} for word in words if word in existing]
    return labels, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--videos", type=Path, help="Reference files named <catalog-slug>.mp4 or in <catalog-slug>/")
    source.add_argument("--scores", type=Path, help="Labeled batch scores already produced by OnnxProbe.java")
    parser.add_argument("--ort-jar", type=Path, help="Desktop ONNX Runtime JAR, required for videos")
    parser.add_argument("--java", default="java")
    parser.add_argument("--catalog", type=Path, default=RECOGNITION / "catalog.json")
    parser.add_argument("--output", required=True, type=Path, help="Path for the candidate recognition manifest")
    args = parser.parse_args()

    model = RECOGNITION / "SignFlow-R.onnx"
    manifest = json.loads((RECOGNITION / "manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    words = [row["word"] for row in catalog]
    if sha256(model) != manifest["modelSha256"]:
        raise ValueError("The SignFlow-R file does not match the legacy manifest")

    if args.videos:
        clips = []
        for row in catalog:
            slug = row["signflow"]
            candidates = [*args.videos.glob(f"{slug}.*"), *(args.videos / slug).glob("*")]
            clips.extend(
                (row["word"], clip) for clip in sorted(candidates)
                if clip.is_file() and clip.suffix.lower() in VIDEO_EXTENSIONS
            )
        if not clips:
            parser.error("No videos matched the approved catalog")
        if not args.ort_jar or not args.ort_jar.is_file():
            parser.error("--ort-jar must point to the desktop ONNX Runtime JAR")
        with tempfile.TemporaryDirectory(prefix="signflow-map-") as temporary:
            path = write_scores(Path(temporary), clips, model, manifest, args.ort_jar, args.java)
            winners = checked_winners(path, words)
    else:
        winners = checked_winners(args.scores, words)
    if not winners:
        parser.error("No labeled model scores were supplied")

    prototype = manifest["prototype"]
    prototype["labels"], report = map_labels(words, prototype, winners)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "covered": len(prototype["labels"]), "total": len(words), "words": report,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
