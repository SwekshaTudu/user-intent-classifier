"""Download dataset-provided SRTs and enrich annotation CSVs (no ML dependencies)."""

import argparse
import csv
import io
import json
import os
import re
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = "aintropy-ai/video-mme-sample10"
ROOT = Path(__file__).resolve().parents[1]
SUBTITLE_FIELDS = {"Subtitle URL", "Subtitle text", "Subtitle status"}
TIMESTAMP = re.compile(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}")


def request_bytes(url):
    """Retry transient failures, but preserve missing-file and authorization errors."""
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "intent-subtitle-fetcher/1.0"}), timeout=45) as response:
                return response.read(), response.headers
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def available_subtitles(revision):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{quote(revision, safe='')}/subtitles?limit=1000"
    paths = set()
    while url:
        body, headers = request_bytes(url)
        paths.update(item["path"] for item in json.loads(body) if item["type"] == "file")
        next_page = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
        url = next_page.group(1) if next_page else None
    return paths


def validate_srt(body):
    text = body.decode("utf-8-sig").replace("\r\n", "\n")
    if not TIMESTAMP.search(text):
        raise ValueError("Response is not a timestamped SRT subtitle file")
    return text


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def fetch_one(video_id, paths, revision, cache_dir):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/subtitles/{video_id}.srt"
    if f"subtitles/{video_id}.srt" not in paths:
        return {"url": url, "text": "", "status": "Unavailable: no subtitle file in dataset"}
    cached = cache_dir / f"{video_id}.srt"
    try:
        if cached.exists():
            try:
                text = validate_srt(cached.read_bytes())
                return {"url": url, "text": text, "status": "Downloaded"}
            except (ValueError, UnicodeError):
                pass  # A corrupt cache entry is fetched again.
        body, _ = request_bytes(url)
        text = validate_srt(body)
        atomic_write(cached, text)
        return {"url": url, "text": text, "status": "Downloaded"}
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, UnicodeError) as exc:
        # Even a 404 here is a failed fetch: the pinned repository listing said it exists.
        return {"url": url, "text": "", "status": "Download failed", "error": str(exc)}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if not fields or not {"Video ID", "Question ID", "Question"}.issubset(fields):
        raise ValueError(f"Missing required CSV columns: {path}")
    for row in rows:
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", row["Video ID"]):
            raise ValueError(f"Invalid video ID: {row['Video ID']!r}")
    return fields, rows


def enrich_rows(fields, rows, subtitles):
    fields = fields + [key for key in ["Subtitle URL", "Subtitle text", "Subtitle status"] if key not in fields]
    result = []
    for row in rows:
        updated = dict(row)
        subtitle = subtitles[row["Video ID"]]
        updated["Subtitle URL"] = subtitle["url"]
        updated["Subtitle status"] = subtitle["status"]
        # Preserve any existing context when a network refresh fails.
        if subtitle["status"] != "Download failed" or not row.get("Subtitle text"):
            updated["Subtitle text"] = subtitle["text"]
        result.append(updated)
    return fields, result


def save_csv(path, fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    text = "\ufeff" + stream.getvalue()
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return path
    if path.exists():
        backup = path.with_suffix(path.suffix + ".before_subtitles.bak")
        if not backup.exists():
            atomic_write(backup, path.read_text(encoding="utf-8"))
    try:
        atomic_write(path, text)
        return path
    except PermissionError:
        # Excel may lock the original; keep it intact and provide a usable output.
        alternate = path.with_name(path.stem + "_with_subtitles.csv")
        atomic_write(alternate, text)
        print(f"File locked; saved updated copy: {alternate}", flush=True)
        return alternate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", type=Path, help="CSV paths; defaults to the three active split files")
    parser.add_argument("--revision", default="main", help="Hugging Face revision; pinned to its commit for this run")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16")
    manifest_path = ROOT / "data/splits/split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() and not args.csv else None
    files = args.csv or ([manifest_path.parent / manifest["splits"][name]["annotation_file"] for name in ["train", "validation", "test"]] if manifest else [ROOT / "data/model" / f"{name}.csv" for name in ["train", "validation", "test"]])
    loaded = {path.resolve(): read_csv(path) for path in files}
    info, _ = request_bytes(f"https://huggingface.co/api/datasets/{REPO}/revision/{quote(args.revision, safe='')}")
    revision = json.loads(info)["sha"]
    paths = available_subtitles(revision)
    video_ids = sorted({r["Video ID"] for _, rows in loaded.values() for r in rows})
    cache_dir = ROOT / "data/subtitles" / revision
    subtitles = {}
    print(f"Fetching subtitle context for {len(video_ids)} unique videos at revision {revision}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_one, vid, paths, revision, cache_dir): vid for vid in video_ids}
        for future in as_completed(futures):
            subtitles[futures[future]] = future.result()
            if len(subtitles) % 10 == 0:
                print(f"Checked {len(subtitles)}/{len(video_ids)} videos", flush=True)
    outputs = {}
    for path, (fields, rows) in loaded.items():
        new_fields, updated = enrich_rows(fields, rows, subtitles)
        saved = save_csv(path, new_fields, updated)
        saved_fields, checked = read_csv(saved)
        assert saved_fields == new_fields and checked == updated
        assert [{k:v for k,v in r.items() if k not in SUBTITLE_FIELDS} for r in rows] == [{k:v for k,v in r.items() if k not in SUBTITLE_FIELDS} for r in checked]
        outputs[str(path)] = str(saved)
        print(f"Updated {saved.name}: {dict(Counter(r['Subtitle status'] for r in checked))}", flush=True)
        if manifest is not None:
            for item in manifest["splits"].values():
                if path.name == item["annotation_file"]:
                    item["annotation_file"] = saved.name
    report = {"dataset":REPO, "revision":revision, "unique_videos":len(video_ids), "status_counts":dict(Counter(s["status"] for s in subtitles.values())), "errors":{vid:s["error"] for vid,s in subtitles.items() if "error" in s}, "outputs":outputs, "notes":"Timestamped SRT text is annotation context only. Labels and review status are unchanged; transcripts have not been reviewed."}
    atomic_write(ROOT / "data/subtitles/download_report.json", json.dumps(report, indent=2) + "\n")
    if manifest is not None:
        manifest["subtitle_revision"] = revision
        atomic_write(manifest_path, json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report["status_counts"], indent=2), flush=True)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
