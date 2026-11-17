"""A second, finer SkillCorner reference from the per-frame `possession.group` field. Not part
of the submission.

    python tools/skillcorner_fine.py build --raw-root RAW --refs-root REFS --cache CACHE \
        --out OUT [--games SC-... ...]

The existing SkillCorner reference (tools/skillcorner_bundle.py, docs/plan/calibration/
SKILLCORNER_REFERENCES.md) labels a frame by the SkillCorner phase of play that covers it. A
phase can contain a loss of the ball (column `team_possession_loss_in_phase`), so that reference
merges brief losses: it is the coarse view. The tracking file carries a per-frame field
`possession` {player_id, group} that the coarse reference reads only for its team check. This
tool builds the fine view from that field and changes nothing else:

- Frame grid, halves, video blocks and the 10 Hz to 25 Hz mapping: skillcorner_bundle.build()
  unchanged (nearest tracking frame, round(0.4 * v); X outside the blocks of the halves).
- Dead ball: exactly the coarse reference's `D` frames (inside a half and outside every phase).
- Graded mask: exactly the coarse reference's non-X frames.
- Live frames (inside a phase): `H` if `possession.group` is "home team", `A` if "away team".
- Live frames where the field is empty (rule HOLD): the last team the field named earlier in the
  same live stretch (a maximal run of consecutive live frames; a stretch ends at every `D` or `X`
  frame). At the head of a stretch, before the field names any team, the frame takes the team of
  its phase (rule HEAD). Both are counted per match in the build file.

So the fine reference differs from the coarse one only on live frames, where the per-frame field
names the other team than the phase. Every check of the coarse build is repeated: the rebuilt
coarse labels must equal the reference CSV on disk and the cached `raw` labels, frame for frame.

Outputs, never in the existing reference folders nor in the training cache:

    OUT/refs/SC-<id>_reference_possgroup.csv   `frame,label` (H, A, D, X), one row per video frame
    OUT/refs/SC-<id>_possgroup_build.json      counts, checks, input checksums
    OUT/cache_possgroup/labels/SC-<id>.npz     raw (the fine codes), ctl2 (its 2 s controlled
                                               view, build.py CONTROL_FRAMES), phases (the coarse
                                               raw codes, for audit)
    OUT/cache_possgroup/labels/<other>.npz     symlinks to CACHE/labels (other providers)
    OUT/cache_possgroup/feats                  symlink to CACHE/feats

OUT/cache_possgroup is therefore a build.py-shaped cache in which the SkillCorner labels are the
fine view (the extra view "possgroup") and everything else is the cache as built. The degraded
SkillCorner features in CACHE were built with the coarse labels (degrade.py reads them); this
tool does not rebuild them.

SkillCorner Open Data, github.com/SkillCorner/opendata, MIT licence, copyright SkillCorner.
Deterministic, no network. Python 3.9, numpy and the standard library.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from tools import skillcorner_bundle as scb  # noqa: E402

H, A, D, X = scb.H, scb.A, scb.D, scb.X
VIEW = "possgroup"
CONTROL_FRAMES = 50          # tools/learned_spike/build.py CONTROL_FRAMES (the 2 s view)


def fine_source_labels(coarse10, group):
    """(fine 10 Hz labels int8, counts dict) from the coarse 10 Hz labels and the group field.

    coarse10: int8 (F,) of H, A, D, X (skillcorner_bundle.source_labels). group: int8 (F,) of H,
    A, or X for an empty field (skillcorner_bundle.read_tracking). Live = coarse in (H, A).
    Rule HOLD and rule HEAD as in the module docstring. D and X frames are copied unchanged.
    """
    coarse10 = np.asarray(coarse10, dtype=np.int8)
    group = np.asarray(group, dtype=np.int8)
    if coarse10.shape != group.shape:
        raise ValueError("coarse labels and group field differ in length")
    if not np.isin(group, (H, A, X)).all():
        raise ValueError("group field holds a value other than H, A, X")
    live = np.isin(coarse10, (H, A))
    out = coarse10.copy()
    held = head = 0
    stretches = 0
    n = len(coarse10)
    i = 0
    while i < n:
        if not live[i]:
            i += 1
            continue
        j = i
        while j < n and live[j]:
            j += 1
        stretches += 1
        last = X
        for k in range(i, j):
            g = group[k]
            if g != X:
                last = g
                out[k] = g
            elif last != X:
                out[k] = last
                held += 1
            else:
                out[k] = coarse10[k]
                head += 1
        i = j
    n_live = int(live.sum())
    counts = {
        "live_frames": n_live,
        "live_stretches": stretches,
        "group_named_frames": int((live & (group != X)).sum()),
        "group_empty_frames": int((live & (group == X)).sum()),
        "held_frames": held,             # rule HOLD
        "head_frames": head,             # rule HEAD (phase team)
        "fine_differs_from_coarse_frames": int((out != coarse10).sum()),
        "group_named_outside_live_frames": int((~live & (coarse10 != X) & (group != X)).sum()),
    }
    if n_live:
        counts["held_share_of_live"] = round(held / float(n_live), 6)
        counts["head_share_of_live"] = round(head / float(n_live), 6)
        counts["differs_share_of_live"] = round(counts["fine_differs_from_coarse_frames"]
                                                / float(n_live), 6)
    return out, counts


def to_video(labels10, n_frames, blocks):
    """25 Hz labels: the nearest tracking frame (skillcorner_bundle.build), X outside the blocks."""
    nearest = np.minimum(scb.nearest_source(n_frames), len(labels10) - 1)
    out = np.asarray(labels10, dtype=np.int8)[nearest]
    in_block = np.zeros(n_frames, dtype=bool)
    for first, last in blocks.values():
        in_block[first:last + 1] = True
    out[~in_block] = X
    return out


def control_view(codes):
    """The 2 s controlled view of a code vector (build.py load_codes)."""
    import possession_metrics as pm
    from analysis import control_reference as cref
    labels = np.asarray(pm.LABELS, dtype=object)[np.asarray(codes, dtype=np.int8)]
    return pm.label_codes(cref.control_labels(labels, CONTROL_FRAMES)).astype(np.int8)


def read_csv_codes(path):
    import possession_metrics as pm
    return pm.label_codes(pm.read_label_csv(path)).astype(np.int8)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def build_match(match_dir, refs_dir, cache, out):
    """Convert one match. Return the build summary dict. Raises on any check failure."""
    match_id = os.path.basename(os.path.normpath(match_dir))
    stem = scb.STEM_PREFIX + match_id
    names = scb.file_names(match_id)
    inputs = {k: os.path.join(match_dir, v) for k, v in names.items()}
    match = scb.read_match(inputs["match"])
    tracking = scb.read_tracking(inputs["tracking"], match["team_of_player"])
    phases = scb.read_phases(inputs["phases"])
    spans = scb.half_spans(tracking["period"])
    blocks = {h: scb.video_block(a, b) for h, (a, b) in spans.items()}
    n_frames = blocks[max(blocks)][1] + 1
    coarse10 = scb.source_labels(tracking["frames"].size, spans, phases,
                                 match["home"]["team_id"], match["away"]["team_id"])
    coarse = to_video(coarse10, n_frames, blocks)
    # checks: the rebuilt coarse labels equal the reference on disk and the cached labels
    ref_csv = os.path.join(refs_dir, stem + "_reference.csv")
    on_disk = read_csv_codes(ref_csv)
    if not np.array_equal(on_disk, coarse):
        raise ValueError("{}: rebuilt coarse labels differ from {}".format(stem, ref_csv))
    with np.load(os.path.join(cache, "labels", stem + ".npz")) as lab:
        cached = lab["raw"].astype(np.int8)
    if not np.array_equal(cached, coarse):
        raise ValueError("{}: rebuilt coarse labels differ from the cached raw labels".format(stem))
    fine10, counts = fine_source_labels(coarse10, tracking["group"])
    fine = to_video(fine10, n_frames, blocks)
    # invariants of the fine view
    assert np.array_equal(fine == X, coarse == X), "graded mask changed"
    assert np.array_equal(fine == D, coarse == D), "dead-ball frames changed"
    os.makedirs(os.path.join(out, "refs"), exist_ok=True)
    lab_dir = os.path.join(out, "cache_possgroup", "labels")
    os.makedirs(lab_dir, exist_ok=True)
    csv_path = os.path.join(out, "refs", stem + "_reference_possgroup.csv")
    scb.events_gt.write_csv(csv_path, fine)
    ctl2 = control_view(fine)
    npz_path = os.path.join(lab_dir, stem + ".npz")
    # write a new file and rename it over the path: never through a symlink into CACHE
    tmp = os.path.join(lab_dir, stem + ".tmp.npz")
    np.savez_compressed(tmp, raw=fine, ctl2=ctl2, phases=coarse)
    os.replace(tmp, npz_path)
    summary = {
        "stem": stem, "view": VIEW, "match_id": match_id, "title": match["title"],
        "source": {"repository": "github.com/SkillCorner/opendata", "commit": scb.SOURCE_COMMIT,
                   "licence": "MIT, copyright SkillCorner"},
        "rules": {"live": "frames inside a phase (coarse H or A)",
                  "HOLD": "empty field in live play: last team named earlier in the same live "
                          "stretch",
                  "HEAD": "empty field before any named team in the stretch: the phase team"},
        "counts_10hz": counts,
        "video": {"n_frames": int(n_frames),
                  "graded": int((coarse != X).sum()),
                  "dead": int((coarse == D).sum()),
                  "live": int(np.isin(coarse, (H, A)).sum()),
                  "fine_differs_from_coarse": int((fine != coarse).sum())},
        "checks": {"coarse_equals_reference_csv": True, "coarse_equals_cache_raw": True,
                   "graded_mask_unchanged": True, "dead_ball_unchanged": True},
        "inputs": {k: {"file": os.path.basename(p), "sha256": sha256(p)}
                   for k, p in sorted(inputs.items())},
        "coarse_reference_sha256": sha256(ref_csv),
        "outputs": {"reference": {"file": os.path.basename(csv_path), "sha256": sha256(csv_path)}},
    }
    with open(os.path.join(out, "refs", stem + "_possgroup_build.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def link_cache(cache, out, fine_games):
    """Symlink CACHE/feats and every CACHE label that the fine view does not replace."""
    root = os.path.join(out, "cache_possgroup")
    lab_dir = os.path.join(root, "labels")
    os.makedirs(lab_dir, exist_ok=True)
    feats = os.path.join(root, "feats")
    if not os.path.lexists(feats):
        os.symlink(os.path.join(os.path.abspath(cache), "feats"), feats)
    for name in sorted(os.listdir(os.path.join(cache, "labels"))):
        if not name.endswith(".npz") or name[:-4] in fine_games:
            continue
        dst = os.path.join(lab_dir, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(os.path.abspath(cache), "labels", name), dst)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw-root", required=True, help="folder of SkillCorner match folders <id>/")
    b.add_argument("--refs-root", required=True, help="refs/skillcorner (coarse references)")
    b.add_argument("--cache", required=True, help="build.py cache (read only)")
    b.add_argument("--out", required=True, help="new folder (never the cache or refs)")
    b.add_argument("--games", nargs="*", default=None)
    args = parser.parse_args(argv)
    out = os.path.abspath(args.out)
    for protected in (args.cache, args.refs_root, args.raw_root):
        p = os.path.abspath(protected)
        if out == p or out.startswith(p + os.sep):
            raise SystemExit("--out must lie outside {}".format(p))
    from tools.learned_spike import folds
    games = args.games or list(folds.SC)
    for game in games:
        if folds.provider(game) != "skillcorner":
            raise SystemExit("{} is not a SkillCorner match".format(game))
        folds.check(game)
    for game in games:
        mid = game[len(scb.STEM_PREFIX):]
        s = build_match(os.path.join(args.raw_root, mid), os.path.join(args.refs_root, mid),
                        args.cache, out)
        c = s["counts_10hz"]
        print("{}: live {} differs {:.3f} held {:.3f} head {:.4f}".format(
            game, c["live_frames"], c["differs_share_of_live"], c["held_share_of_live"],
            c["head_share_of_live"]), flush=True)
    link_cache(args.cache, out, set(games))
    return 0


if __name__ == "__main__":
    sys.exit(main())
