"""Build the local replay bundle of the visualizer (site/data/SCHEMA.md Section 3).

    python -m tools.viz.build_replay --version c8_bump --results evaluation/results/c8_bump.csv
                                     [--game BAR-ATH] [--out site/data/local]
                                     [--allow-md5-mismatch]

The local bundle holds the real tracking positions and our full-match labels. Never commit it.
The exporter refuses an output folder that git would commit. It accepts a folder outside the
repository or a folder that `git check-ignore` accepts, for example `site/data/local`.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

from tools.viz import quant, rle
from tools.viz.build_site import BuildError, mask_runs, serialize, table_rows

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(ROOT, "site", "data", "local")
# Schema version of both bundles (SCHEMA.md 2.1).
SCHEMA_VERSION = 1
TEAMS = ("H", "A")
# Bits of the `flags` array, from bit 0 (SCHEMA.md 3.3).
FLAG_BITS = ("stoppage", "visible", "airborne_suspect", "ghost", "teleport", "graded")
# Scales of the frame arrays (SCHEMA.md 3.3): probabilities in 1/255 steps, lengths and speeds
# in centimetres and cm/s, log evidence in centinats.
PROBABILITY_SCALE = 1.0 / 255
CM = quant.CM
# Decimals of an x or y coordinate in the JSON files (SCHEMA.md 3.4).
XY_DECIMALS = 2
CONFIDENCE_DECIMALS = 3
# Kinematic trace columns of transitions_kin.json and their forensics names (SCHEMA.md 3.4).
TRACE_COLUMNS = (("off", "off"), ("ours", "ours"), ("gt", "gt"), ("speed", "v"),
                 ("dv3", "dv_3"), ("d_h", "dH"), ("d_a", "dA"))
REFUSAL = "refusing to write positions into a path git would commit"


class RefusedPath(RuntimeError):
    """The output folder is inside the repository and git does not ignore it."""


def check_out_dir(out, root=ROOT):
    """Return the absolute output folder, or raise RefusedPath.

    A folder outside the repository is accepted. A folder inside it is accepted only if
    `git check-ignore` matches a file in it. git must be available in both cases inside the tree.
    """
    out = os.path.realpath(os.path.abspath(out))
    repo = os.path.realpath(root)
    if os.path.commonpath([out, repo]) != repo:
        return out
    probe = os.path.join(os.path.relpath(out, repo), "manifest.json")
    try:
        result = subprocess.run(["git", "check-ignore", "-q", probe], cwd=repo,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        raise RefusedPath("{}: git is not available to check {}".format(REFUSAL, out))
    if result.returncode != 0:
        raise RefusedPath("{}: {}".format(REFUSAL, out))
    return out


def _metres(array, length):
    """Normalised 0..1 coordinates as float64 metres."""
    return np.asarray(array, dtype=np.float64) * float(length)


def halves(blocks, n_frames):
    """`[(half, frame_start, frame_stop)]`: the split at the first frame of block 2 (SCHEMA 3.1)."""
    split = int(blocks[1][1])
    return [(1, 0, split), (2, split, int(n_frames))]


def positions_arrays(tactical, ball, pitch, lo, hi):
    """(name, int16 array) pairs of positions_h<k> for frames [lo, hi) (SCHEMA.md 3.2)."""
    length, width = pitch
    out = [("px", quant.encode_i16(_metres(tactical["px"][lo:hi], length), CM, "px")),
           ("py", quant.encode_i16(_metres(tactical["py"][lo:hi], width), CM, "py"))]
    for name, key in (("bx", "x"), ("by", "y"), ("bx_raw", "x_raw"), ("by_raw", "y_raw")):
        out.append((name, quant.encode_i16(ball[key][lo:hi], CM, name)[:, None]))
    return out


def flags(inputs, parts, blocks, n_frames):
    """uint8 bit field of SCHEMA.md 3.3 over the whole match."""
    ball = inputs["ball"]
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    bits = {"stoppage": parts["stoppage"], "visible": ball["visible"],
            "airborne_suspect": ball["airborne_suspect"], "ghost": inputs["ghost"],
            "teleport": ball["teleport"], "graded": graded}
    out = np.zeros(n_frames, dtype=np.uint8)
    for bit, name in enumerate(FLAG_BITS):
        out |= (np.asarray(bits[name], dtype=bool).astype(np.uint8) << bit)
    return out


def frame_arrays(inputs, parts, flag_array, lo, hi):
    """(name, array, scale) triples of frames_h<k> for frames [lo, hi) (SCHEMA.md 3.3)."""
    ball, feats = inputs["ball"], inputs["feats"]
    cut = slice(lo, hi)
    return [
        ("owner", quant.encode_i8(parts["owner"][cut], "owner"), 1),
        ("p_dead", quant.encode_u8(parts["p_dead"][cut], PROBABILITY_SCALE, "p_dead"),
         PROBABILITY_SCALE),
        ("e_dead", quant.encode_u8(inputs["e_dead"][cut], PROBABILITY_SCALE, "e_dead"),
         PROBABILITY_SCALE),
        ("flags", flag_array[cut], 1),
        ("slot_h", quant.encode_i8(feats["nearest_slot_H"][cut], "slot_h"), 1),
        ("slot_a", quant.encode_i8(feats["nearest_slot_A"][cut], "slot_a"), 1),
        ("d_h", quant.encode_u16(feats["dH"][cut], CM, "d_h"), CM),
        ("d_a", quant.encode_u16(feats["dA"][cut], CM, "d_a"), CM),
        ("speed", quant.encode_u16(ball["speed"][cut], CM, "speed"), CM),
        ("vx", quant.encode_i16(ball["vx"][cut], CM, "vx"), CM),
        ("vy", quant.encode_i16(ball["vy"][cut], CM, "vy"), CM),
        ("dv3", quant.encode_i16(ball["dv_3"][cut], CM, "dv3"), CM),
        ("meanv", quant.encode_u16(feats["meanv_all"][cut], CM, "meanv"), CM),
        ("sep", quant.encode_u16(feats["sep"][cut], CM, "sep"), CM),
        ("log_e", quant.encode_i16(parts["log_e"][cut], CM, "log_e"), CM)]


def _xy(ball, frame):
    """[x, y] of the cleaned ball at `frame` in metres, None where missing."""
    values = []
    for key in ("x", "y"):
        value = float(ball[key][int(frame)])
        values.append(round(value, XY_DECIMALS) if np.isfinite(value) else None)
    return values


def _team(value):
    return value if value in TEAMS else None


def _confidence(value):
    value = float(value)
    return round(value, CONFIDENCE_DECIMALS) if np.isfinite(value) else None


def json_files(codes, inputs, parts, tactical, forensics_result, n_frames):
    """{file name: object} of the JSON files of SCHEMA.md 3.4."""
    ball = inputs["ball"]
    events = [[int(f), str(k), _team(t), _confidence(c)] + _xy(ball, f)
              for f, k, t, c in table_rows(parts["events"], ("frame", "kind", "team",
                                                             "confidence"))]
    onsets = [[int(f), str(m), _confidence(c)] + _xy(ball, f)
              for f, m, c in table_rows(parts["onsets"], ("frame", "mechanism", "confidence"))]
    restarts = [[int(f), _team(t), str(k), _confidence(c), bool(ok)] + _xy(ball, f)
                for f, t, k, c, ok in table_rows(parts["restarts"], ("frame", "team", "kind",
                                                                    "confidence", "confirmed"))]

    def trace(rows):
        return [{name: row[key] for name, key in TRACE_COLUMNS} for row in rows]

    team_of_slot = [TEAMS[int(t)] for t in tactical["team_of_slot"]]
    visible = np.isfinite(np.asarray(tactical["px"])).sum(axis=0)
    return {
        "labels_ours.json": {"name": "ours", "n_frames": int(n_frames), "runs": rle.encode(codes)},
        "events_xy.json": {"events": events},
        "deadball_xy.json": {"onsets": onsets, "restarts": restarts,
                             "stoppages": mask_runs(parts["stoppage"])},
        "transitions_kin.json": {
            "gt_transitions": [{"gt_frame": rec["gt_frame"], "labels_trace": trace(rec["trace"])}
                               for rec in forensics_result["gt_transitions"]],
            "spurious": [{"our_frame": rec["our_frame"], "labels_trace": trace(rec["trace"])}
                         for rec in forensics_result["spurious"]]},
        "slots.json": {"team_of_slot": team_of_slot,
                       "visible_counts": [int(n) for n in visible]}}


def build(args):
    """Check the output folder, run the engine and write the local bundle. Return the exit code."""
    try:
        out = check_out_dir(args.out)
    except RefusedPath as exc:
        print("build_replay: {}".format(exc), file=sys.stderr)
        return 1
    try:
        written = write_replay(args.version, args.results, out, args.game,
                               args.allow_md5_mismatch)
    except (BuildError, ValueError, OverflowError, FileNotFoundError) as exc:
        print("build_replay: {}".format(exc), file=sys.stderr)
        return 1
    print("wrote {} files, {} bytes, to {}".format(len(written), sum(written.values()), out))
    return 0


def write_replay(version, results_path, out, game=None, allow_mismatch=False):
    """Write the local bundle to `out` and return {file name: bytes}.

    The caller must pass a folder that check_out_dir() accepted.
    """
    from engine import data, possession
    from engine.config import Config
    from tools import _common as cm, forensics
    from tools.viz import gitinfo
    from tools.viz._engine_state import engine_state

    if cm.result_name(results_path) != version:
        raise BuildError("--results {} does not name version {}".format(results_path, version))
    cfg = Config.default()
    game = data.resolve_game(game)
    codes, inputs, parts, md5 = engine_state(cfg, game, results_path, allow_mismatch)
    tactical = possession.engine_tactical(cfg, game)
    pitch = data.pitch_m(game)
    n_frames, blocks = int(inputs["n_frames"]), inputs["blocks"]
    result = forensics.analyse(results_path, tactical=tactical, tags=inputs["tagging"], game=game)
    flag_array = flags(inputs, parts, blocks, n_frames)

    payloads = {}
    half_rows = []
    for half, lo, hi in halves(blocks, n_frames):
        stem_p, stem_f = "positions_h{}".format(half), "frames_h{}".format(half)
        blob, header = quant.pack(positions_arrays(tactical, inputs["ball"], pitch, lo, hi))
        payloads[stem_p + ".bin"] = blob
        payloads[stem_p + ".json"] = serialize({
            "dtype": "int16", "scale": CM, "sentinel": quant.I16_SENTINEL,
            "frame_start": lo, "frame_stop": hi, "n_slots": int(tactical["px"].shape[1]),
            "arrays": {name: {"offset": e["offset"], "shape": e["shape"]}
                       for name, e in header.items()}})
        triples = frame_arrays(inputs, parts, flag_array, lo, hi)
        blob, header = quant.pack([(name, array) for name, array, _ in triples])
        scales = {name: scale for name, _, scale in triples}
        for name, entry in header.items():
            entry["scale"] = scales[name]
        payloads[stem_f + ".bin"] = blob
        payloads[stem_f + ".json"] = serialize({
            "frame_start": lo, "frame_stop": hi, "n_slots": int(tactical["px"].shape[1]),
            "arrays": header})
        half_rows.append({"half": half, "frame_start": lo, "frame_stop": hi,
                          "files": {"positions": stem_p + ".json", "positions_bin": stem_p + ".bin",
                                    "frames": stem_f + ".json", "frames_bin": stem_f + ".bin"}})
    for name, obj in json_files(codes, inputs, parts, tactical, result, n_frames).items():
        payloads[name] = serialize(obj)

    manifest = {"schema": SCHEMA_VERSION, "version": version, "md5": md5,
                "config_hash": cfg.config_hash(), "git": gitinfo.head(), "game": game,
                "n_frames": n_frames, "fps": int(round(inputs["fps"])),
                "pitch_m": [float(v) for v in pitch], "halves": half_rows,
                "files": {name: len(payload) for name, payload in sorted(payloads.items())}}
    payloads["manifest.json"] = serialize(manifest)

    if not os.path.isdir(out):
        os.makedirs(out)
    for name in sorted(payloads):
        with open(os.path.join(out, name), "wb") as handle:
            handle.write(payloads[name])
    return {name: len(payload) for name, payload in sorted(payloads.items())}


def main(argv=None):
    """CLI entry point. Return 0 on success, 1 on a refused build, 2 on a usage error."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="version name, for example c8_bump")
    parser.add_argument("--results", required=True, help="results CSV of that version")
    parser.add_argument("--game", default=None, help="match of games.py (default: games.GAME)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output folder (default: %(default)s)")
    parser.add_argument("--allow-md5-mismatch", action="store_true",
                        help="accept a results file that the engine does not reproduce")
    return build(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
