"""Apply a frozen gap model to the default engine labels of one match (P10-C, P10-D).

    python -m tools.gap_models.apply --model <artefact.json> --kind owner --out <csv>
    python -m tools.gap_models.apply --model <artefact.json> --kind dead --out <csv>
                                     [--game name] [--config path]

The tool reads the artefact first and refuses a missing or malformed file. It then runs the
default engine once with `possession.run(..., return_parts=True)` and applies the correction
of `--kind`. It writes the CSV with `engine.submission` and prints the scalar audit entries.

This tool is for experiments only. The submission path is `python -m engine.run`, and
`engine/` loads no learned artefact.
"""

import argparse
import sys

from engine import data, possession, submission
from engine import run as engine_run
from engine.config import Config
from tools.gap_models import gap_dead, gap_owner

KINDS = ("owner", "dead")
LOADERS = {"owner": gap_owner.load_artifact, "dead": gap_dead.load_artifact}


def correct(kind, artefact, cfg=None, game=None, inputs=None):
    """Return the corrected labels and the audit of one gap model on the default engine.

    `inputs` is an optional `possession.prepare()` result for the same `cfg` and match.
    The owner correction receives the detection inputs, the decoder parts and the raw
    event table under "raw_events". The dead correction receives the inputs and the
    parts separately.
    """
    if kind not in KINDS:
        raise ValueError("kind must be one of {}".format(", ".join(KINDS)))
    cfg = cfg or Config.default()
    if inputs is None:
        inputs = possession.prepare(cfg, game=game)
    labels, parts = possession.run(cfg, inputs=inputs, return_parts=True, game=game)
    if kind == "dead":
        return gap_dead.apply(labels, inputs, parts, artefact, cfg, game=game)
    combined = dict(inputs)
    combined.update(parts)
    combined["raw_events"] = inputs["events"]
    return gap_owner.apply(labels, combined, artefact, cfg, game=inputs["game"])


def audit_summary(audit):
    """Return the scalar entries of an audit. Lists and dicts hold interval coordinates."""
    return {key: value for key, value in sorted(audit.items())
            if isinstance(value, (bool, int, str))}


def main(argv=None):
    """CLI entry point. Return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="frozen artefact JSON")
    parser.add_argument("--kind", required=True, choices=KINDS,
                        help="owner (P10-C) or dead (P10-D)")
    parser.add_argument("--out", required=True, help="CSV to write")
    parser.add_argument("--game", default=None, choices=sorted(data.games.GAMES),
                        help="match of games.py (default: games.GAME)")
    parser.add_argument("--config", default=None, help="config file")
    args = parser.parse_args(argv)
    game = data.resolve_game(args.game)
    try:
        artefact = LOADERS[args.kind](args.model)
    except (OSError, ValueError) as error:
        parser.error("cannot use the {} artefact {}: {}".format(args.kind, args.model, error))
    cfg = engine_run._resolve_config(args.config)

    labels, audit = correct(args.kind, artefact, cfg, game=game)
    path = submission.write_submission(labels, args.out, game=game)
    if game == data.games.GAME:
        print(submission.validate(path).rstrip())
    else:
        print("validator skipped: it checks {} only".format(data.games.GAME))
    print()
    print(submission.format_summary(submission.summarize(labels, game=game)))
    print()
    for key, value in audit_summary(audit).items():
        print("{:<32} {}".format(key, value))
    print("wrote {}".format(path))
    print("md5   {}".format(submission.md5_of(path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
