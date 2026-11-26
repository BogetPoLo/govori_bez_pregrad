"""Extract the approved catalog's clips from the official Slovo archive."""

import argparse
import csv
import io
import json
import re
import shutil
import zipfile
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from urllib.request import Request, urlopen


ARCHIVE = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo.zip"
CATALOG = Path(__file__).resolve().parents[2] / "app/src/main/assets/phrases/catalog.json"


class RemoteArchive(io.RawIOBase):
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.position = 0
        self.etag = None
        _, self.size = self.get_range(0, 0)

    def get_range(self, start, end):
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self.etag:
            headers["If-Match"] = self.etag
        with urlopen(Request(self.url, headers=headers), timeout=20) as response:
            content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                         response.headers.get("Content-Range", ""))
            if response.status != 206 or not content_range:
                raise OSError("The server does not support partial downloads. Use a local ZIP.")
            first, last, total = map(int, content_range.groups())
            if (first, last) != (start, end) or (hasattr(self, "size") and self.size != total):
                raise OSError("Unexpected archive range returned by the server")
            etag = response.headers.get("ETag")
            if self.etag and etag and etag != self.etag:
                raise OSError("Archive changed during download")
            self.etag = etag or self.etag
            data = response.read(end - start + 1)
            if len(data) != end - start + 1:
                raise OSError("Incomplete archive range")
            return data, total

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        if base + offset < 0:
            raise OSError("Cannot seek before the archive")
        self.position = base + offset
        return self.position

    def readinto(self, buffer):
        size = min(len(buffer), self.size - self.position, 1 << 20)
        if size <= 0:
            return 0
        data, _ = self.get_range(self.position, self.position + size - 1)
        buffer[:size] = data
        self.position += size
        return size


def normalized(word):
    return word.strip().casefold().rstrip(".!?").replace("ё", "е")


def extract(archive, entries, output, per_word):
    annotations = [file for file in archive.infolist() if Path(file.filename).name == "annotations.csv"]
    if len(annotations) != 1:
        raise ValueError("Expected one annotations.csv in the trimmed Slovo archive")
    videos = {}
    for member in archive.infolist():
        path = Path(member.filename)
        if path.suffix.lower() != ".mp4":
            continue
        if path.stem in videos:
            raise ValueError(f"Repeated video ID in archive: {path.stem}")
        videos[path.stem] = member

    matches = defaultdict(list)
    with archive.open(annotations[0]) as source, io.TextIOWrapper(source, encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["attachment_id"] in videos:
                matches[normalized(row["text"])].append(row)

    report = {}
    for entry in entries:
        word = entry["word"]
        files = []
        signers = set()
        for row in matches[normalized(word)]:
            signer = row["user_id"]
            if signer in signers:
                continue
            signers.add(signer)
            member = videos[row["attachment_id"]]
            relative = Path(entry["signflow"]) / Path(member.filename).name
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".part")
            try:
                with archive.open(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            files.append({"file": relative.as_posix(), "signer": signer, "train": row["train"]})
            if len(files) == per_word:
                break
        report[word] = files
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=ARCHIVE, help="Local trimmed Slovo ZIP or its official HTTPS URL")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-word", type=int, default=3)
    args = parser.parse_args()
    if args.per_word < 1:
        parser.error("--per-word must be at least 1")
    entries = json.loads(CATALOG.read_text(encoding="utf-8"))
    try:
        with ExitStack() as stack:
            path = args.archive
            if path.startswith("https://"):
                path = stack.enter_context(io.BufferedReader(RemoteArchive(path), buffer_size=1 << 20))
            archive = stack.enter_context(zipfile.ZipFile(path))
            report = extract(archive, entries, args.output, args.per_word)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Slovo extraction failed: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
