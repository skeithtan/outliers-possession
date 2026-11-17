"""The leak guard of the public bundle (site/data/SCHEMA.md Section 5).

    python -m tools.viz.guard site/data/public

The guard refuses a public object that could rebuild the tracking or our full-match labels.
`build_site.write_bundle()` calls assert_no_coordinates() on each object before it writes it.
The command checks the written files and exits 1 on a leak. The Pages workflow runs it without
numpy, so this module imports only the standard library.
"""

import argparse
import fnmatch
import json
import os
import re
import sys

# Keys that name a position, a velocity or a distance, compared in lower case (SCHEMA.md 5).
FORBIDDEN_KEYS = frozenset([
    "x", "y", "px", "py", "bx", "by", "x_raw", "y_raw", "vx", "vy", "cx", "cy", "xy", "pos",
    "positions", "coords", "dh", "da", "v", "dv_3", "meanv", "speed", "dmin", "sep"])
# A list with more numeric leaves than this is a series, not a summary (SCHEMA.md 5).
MAX_NUMERIC_LEAVES = 20000
# A base64 or hex string longer than this is a blob (SCHEMA.md 5: 4 KB).
MAX_BLOB_CHARS = 4096
BLOB_PATTERN = re.compile(r"^[A-Za-z0-9+/=\r\n]+$")
# Keys whose integer value is a frame (SCHEMA.md 5). The list adds the frame keys that
# tools.forensics.analyse() writes: Smart Tagging rows, sandwich cases and kickoff rows.
FRAME_KEYS = frozenset([
    "frame", "start", "stop", "our_frame", "gt_frame", "start_frame", "end_frame",
    "gt_d_start", "gt_d_stop", "goals_end", "frame_start", "frame_stop"])
# Keys that hold an exclusive end frame. The value can equal the window stop.
STOP_KEYS = frozenset(["stop", "end_frame", "gt_d_stop", "frame_stop"])
# Tables whose rows start with a frame (SCHEMA.md 2.5). Both elements of a stoppage row are
# frames.
ROW_TABLES = frozenset(["events", "owner_events", "onsets", "restarts", "stoppages"])
PAIR_TABLES = frozenset(["stoppages"])
# Label letters of an RLE run (SCHEMA.md Section 1).
RUN_LETTERS = frozenset(["H", "A", "D", "X"])
# Files that the window rule does not cover (SCHEMA.md 5). A manifest can only narrow this list.
DEFAULT_EXEMPT = ("manifest.json", "minute_bins.json", "labels_baseline_*.json", "scores.json",
                  "sanity.json", "ladder.json", "cycles.json", "config.json")
# Files that are never exempt (SCHEMA.md 5).
NEVER_EXEMPT = ("labels_window.json", "window_detail.json", "transitions.json")
# The public window is 5 minutes, 7500 frames at 25 fps (docs/challenge/EVALUATION.md). A
# manifest window that is longer is refused.
MAX_WINDOW_FRAMES = 7500
MANIFEST = "manifest.json"


class LeakError(ValueError):
    """A public object holds data that the organiser rules keep out of a public repository."""


def is_exempt(where, exempt):
    """True if the file name of `where` matches a pattern of `exempt` and of DEFAULT_EXEMPT."""
    name = os.path.basename(where)
    if name in NEVER_EXEMPT:
        return False
    allowed = any(fnmatch.fnmatch(name, pattern) for pattern in DEFAULT_EXEMPT)
    return allowed and any(fnmatch.fnmatch(name, pattern) for pattern in exempt)


def _is_int(value):
    """True for an int that is not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    """True for an int or a float that is not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_run(value):
    """True for an RLE run `[int, int, letter]`."""
    return (isinstance(value, list) and len(value) == 3 and _is_int(value[0]) and
            _is_int(value[1]) and value[2] in RUN_LETTERS)


class _Checker(object):
    """One walk over one object. Collects the problems as strings."""

    def __init__(self, where, window, exempt, n_frames):
        self.where = where
        self.window = window
        self.n_frames = n_frames
        self.window_rule = window is not None and not is_exempt(where, exempt)
        self.problems = []

    def problem(self, path, text):
        self.problems.append("{}:{}: {}".format(self.where, path or "/", text))

    def frame(self, path, value, stop=False):
        """Check one frame integer against the window rule."""
        if not self.window_rule or not _is_int(value):
            return
        lo, hi = self.window
        inside = lo <= value <= hi if stop else lo <= value < hi
        if not inside:
            self.problem(path, "frame {} outside the public window [{}, {})".format(value, lo, hi))

    def walk(self, obj, path="", key=None):
        """Walk `obj`. Return the number of numeric leaves under it."""
        if isinstance(obj, dict):
            leaves = 0
            for name, value in obj.items():
                lower = str(name).lower()
                child = "{}/{}".format(path, name)
                if lower in FORBIDDEN_KEYS:
                    self.problem(child, "forbidden key {!r}".format(name))
                if lower in FRAME_KEYS:
                    self.frame(child, value, stop=lower in STOP_KEYS)
                leaves += self.walk(value, child, lower)
            return leaves
        if isinstance(obj, list):
            if self.n_frames is not None and len(obj) == self.n_frames:
                self.problem(path, "list of length n_frames ({})".format(self.n_frames))
            if _is_run(obj):
                self.frame(path + "[0]", obj[0])
                self.frame(path + "[1]", obj[1], stop=True)
            leaves = 0
            for index, item in enumerate(obj):
                child = "{}[{}]".format(path, index)
                if key in ROW_TABLES and isinstance(item, list) and item:
                    self.frame(child + "[0]", item[0], stop=False)
                    if key in PAIR_TABLES and len(item) > 1:
                        self.frame(child + "[1]", item[1], stop=True)
                leaves += self.walk(item, child, None)
            if leaves > MAX_NUMERIC_LEAVES:
                self.problem(path, "{} numeric leaves, more than {}".format(
                    leaves, MAX_NUMERIC_LEAVES))
                return 0  # report the longest list once, not each parent again
            return leaves
        if isinstance(obj, str):
            if len(obj) > MAX_BLOB_CHARS and BLOB_PATTERN.match(obj):
                self.problem(path, "base64 or hex string of {} characters".format(len(obj)))
            return 0
        return 1 if _is_number(obj) else 0


def problems(obj, where, window=None, exempt=DEFAULT_EXEMPT, n_frames=None):
    """The list of leaks in `obj`, as strings. An empty list means the object passes.

    `where` is the file name. `window` is the half-open public window `(start, stop)`.
    `exempt` is the exempt list of the manifest. `n_frames` is the match length.
    """
    checker = _Checker(where, window, exempt, n_frames)
    checker.walk(obj)
    return checker.problems


def assert_no_coordinates(obj, where, window=None, exempt=DEFAULT_EXEMPT, n_frames=None):
    """Raise LeakError if `obj` holds a position, a series, a blob or a frame outside the window."""
    found = problems(obj, where, window, exempt, n_frames)
    if found:
        raise LeakError("refusing {}:\n  {}".format(where, "\n  ".join(found[:20])))


def manifest_limits(manifest):
    """(window, exempt, n_frames) from a public manifest. Raise LeakError if it is unsafe."""
    try:
        window = (int(manifest["public_window"]["start"]), int(manifest["public_window"]["stop"]))
        exempt = list(manifest["exempt"])
        n_frames = int(manifest["n_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LeakError("manifest.json: missing or bad key {}".format(exc))
    if not 0 < window[1] - window[0] <= MAX_WINDOW_FRAMES:
        raise LeakError("manifest.json: public window {} is longer than {} frames".format(
            window, MAX_WINDOW_FRAMES))
    for pattern in exempt:
        if not any(pattern == allowed for allowed in DEFAULT_EXEMPT):
            raise LeakError("manifest.json: exempt pattern {!r} is not in the guard's list".format(
                pattern))
    return window, exempt, n_frames


def check_dir(folder):
    """All leaks of the public bundle in `folder`, as strings. Read the limits from the manifest."""
    found = []
    path = os.path.join(folder, MANIFEST)
    if not os.path.isfile(path):
        return ["{}: no manifest.json".format(folder)]
    try:
        with open(path, "rb") as handle:
            manifest = json.loads(handle.read().decode("utf-8"))
        window, exempt, n_frames = manifest_limits(manifest)
    except (LeakError, ValueError) as exc:
        return [str(exc)]
    listed = set(manifest.get("files", {}))
    for root, dirs, names in os.walk(folder):
        dirs.sort()
        for name in sorted(names):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, folder).replace(os.sep, "/")
            if not name.endswith(".json"):
                found.append("{}: not a JSON file, the public bundle holds JSON only".format(rel))
                continue
            if rel != MANIFEST and rel not in listed:
                found.append("{}: not listed in manifest.json files".format(rel))
            try:
                with open(full, "rb") as handle:
                    obj = json.loads(handle.read().decode("utf-8"))
            except ValueError as exc:
                found.append("{}: not valid JSON ({})".format(rel, exc))
                continue
            found += problems(obj, rel, window, exempt, n_frames)
    return found


def main(argv=None):
    """CLI entry point. Return 0 if the folder passes, 1 on a leak, 2 on a usage error."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="public bundle folder, for example site/data/public")
    args = parser.parse_args(argv)
    if not os.path.isdir(args.folder):
        print("guard: {} is not a folder".format(args.folder), file=sys.stderr)
        return 2
    found = check_dir(args.folder)
    if found:
        print("guard: {} leaks in {}".format(len(found), args.folder))
        for line in found[:50]:
            print("  " + line)
        return 1
    print("guard: {} passes".format(args.folder))
    return 0


if __name__ == "__main__":
    sys.exit(main())
