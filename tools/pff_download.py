"""Download the PFF FC World Cup 2022 data from Google Drive. Not part of the submission.

    python -I tools/pff_download.py --out RAW [--workers 2]

Source: https://drive.google.com/drive/folders/1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa (PFF FC, no
licence text, docs/plan/calibration/PFF_REFERENCES.md). The tool writes this layout into RAW:

    RAW/tracking/<id>.jsonl.bz2   64 files, broadcast tracking (folder "Tracking Data")
    RAW/events/<id>.json          64 files, the current event version (folder "Event Data",
                                  top level, June 3, 2025; the dated subfolders are older)
    RAW/metadata/<id>.json        64 files
    RAW/rosters/<id>.json         64 files
    RAW/competitions.csv, RAW/players.csv
    RAW/MANIFEST.md5              "md5 ./path" of every file above, sorted by path

Listing. The tool reads each folder from
`https://drive.google.com/embeddedfolderview?id=<FOLDER_ID>`. That page has one
`<div class="flip-entry" id="entry-<FILE_ID>">` with a `flip-entry-title` for each file.

Download. The tool fetches each file with curl from
`https://drive.usercontent.google.com/download?id=<FILE_ID>&export=download&confirm=t`.
A `.bz2` file must start with `BZh`, a `.json` file must parse, and a `.csv` file must not be an
HTML page. The tool tries a file up to five times, with a pause of 15, 30, 60 and 120 s. A file
that is already present and valid is not fetched again. The downloaded files are untrusted data:
the tool reads them only to check them. Run it with `python -I`.

Python 3.9, the standard library and curl.
"""

import argparse
import concurrent.futures
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import time

ROOT_FOLDER = "1_a_q1e9CXeEPJ3GdCv_3-rNO3gPqacfa"
# Subfolder ids of the root folder (listing of 9 November 2026).
FOLDERS = {
    "tracking": "1-ug2vouBfYazPy8z5lZlgd9abAAJSA5q",    # Tracking Data
    "events": "1A7ZvTNzVZ715PDATIhaGA9fPb1KfFmPx",      # Event Data (top level = June 3, 2025)
    "metadata": "1vTCH5W9RMlfWmucTHbrPJ1LrkKnpTJw0",    # Metadata
    "rosters": "1Rp8tpK-MIJt3oSMjufgX6swa9VZZNRy7",     # Rosters
}
# The file names that each folder must give, and the files of the root folder.
PATTERNS = {"tracking": r"^\d+\.jsonl\.bz2$", "events": r"^\d+\.json$",
            "metadata": r"^\d+\.json$", "rosters": r"^\d+\.json$"}
ROOT_FILES = ("competitions.csv", "players.csv")
LIST_URL = "https://drive.google.com/embeddedfolderview?id={}"
FILE_URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
ENTRY = re.compile(r'<div class="flip-entry" id="entry-([^"]+)".*?'
                   r'<div class="flip-entry-title">([^<]*)</div>', re.S)
ATTEMPTS = 5
FIRST_PAUSE_S = 15
MATCHES = 64


def listing(folder_id):
    """[(file id, title)] of a public Drive folder, in page order."""
    page = subprocess.run(["curl", "-sfL", LIST_URL.format(folder_id)], check=True,
                          stdout=subprocess.PIPE).stdout.decode("utf-8")
    return [(fid, html.unescape(title)) for fid, title in ENTRY.findall(page)]


def valid(path):
    """True if the file exists, is not empty and has the expected content type."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    with open(path, "rb") as handle:
        head = handle.read(512)
    if path.endswith(".bz2"):
        return head[:3] == b"BZh"
    if path.endswith(".json"):
        try:
            with open(path, "rb") as handle:
                json.load(handle)
            return True
        except ValueError:
            return False
    if path.endswith(".csv"):
        return b"<html" not in head.lower() and b"," in head
    return True


def fetch(file_id, out):
    """Download one file to `out` with retries. Return None on success or an error text."""
    if valid(out):
        return None
    tmp = out + ".part"
    for attempt in range(ATTEMPTS):
        code = subprocess.call(["curl", "-sL", "--max-time", "1800", FILE_URL.format(file_id),
                                "-o", tmp])
        if code == 0 and valid(tmp):
            os.replace(tmp, out)
            return None
        if os.path.exists(tmp):
            os.remove(tmp)
        if attempt + 1 < ATTEMPTS:
            time.sleep(FIRST_PAUSE_S * 2 ** attempt)
    return "{}: failed after {} attempts (curl {})".format(out, ATTEMPTS, code)


def plan(out_dir):
    """[(file id, output path)] of every file to download. SystemExit if a listing is short."""
    jobs = []
    for name, folder in sorted(FOLDERS.items()):
        files = [(fid, title) for fid, title in listing(folder)
                 if re.match(PATTERNS[name], title)]
        if len(files) != MATCHES:
            raise SystemExit("{}: {} files, expected {}".format(name, len(files), MATCHES))
        jobs += [(fid, os.path.join(out_dir, name, title)) for fid, title in files]
    root = dict((title, fid) for fid, title in listing(ROOT_FOLDER))
    for title in ROOT_FILES:
        if title not in root:
            raise SystemExit("{} is not in the root folder".format(title))
        jobs.append((root[title], os.path.join(out_dir, title)))
    return sorted(jobs, key=lambda job: job[1])


def manifest(out_dir):
    """Write MANIFEST.md5 ("md5 ./relative/path", sorted by path). Return its path."""
    lines = []
    for folder, _, names in os.walk(out_dir):
        for name in names:
            path = os.path.join(folder, name)
            rel = "./" + os.path.relpath(path, out_dir)
            if name == "MANIFEST.md5" or name.endswith(".part"):
                continue
            digest = hashlib.md5()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(block)
            lines.append("{} {}\n".format(digest.hexdigest(), rel))
    target = os.path.join(out_dir, "MANIFEST.md5")
    with open(target, "w") as handle:
        handle.writelines(sorted(lines, key=lambda line: line.split(" ", 1)[1]))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="raw data folder (outside the repository)")
    parser.add_argument("--workers", type=int, default=2, help="parallel downloads (default 2)")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(os.path.expanduser(args.out))
    for name in FOLDERS:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    jobs = plan(out_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        errors = [e for e in pool.map(lambda job: fetch(*job), jobs) if e]
    for error in errors:
        print(error)
    print("{} files, {} failed, manifest {}".format(len(jobs), len(errors), manifest(out_dir)))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
