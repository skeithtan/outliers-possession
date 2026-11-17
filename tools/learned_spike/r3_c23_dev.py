"""c23 labels of the 8 development matches (raw view), for the R3 interim kill check.

    (engine venv) python tools/learned_spike/r3_c23_dev.py --engine-repo DIR --ext DIR --out DIR

DIR of --engine-repo is a checkout whose engine/ tree is c23 (tag `submission`, engine tree
241e70d6...; its BAR-ATH md5 is 7844cad8...). The runs use that checkout's
tools/analysis/external_sweep.py with the default config, one at a time, and copy each CSV to
OUT/<game>.csv. Development matches by default; `--games` also takes the 20 SkillCorner matches
(the unseen-provider probe of r3_diag.py, no-skillcorner); nothing else (folds.check()). Not part
of the submission. Python 3.9.

Round 4 (step C): `--data-root DIR --view NAME` runs the DFL development matches on the bundles of
tools/still_ball.py under DIR (MTAH_DATA_DIR=DIR) instead of the raw view, with the run cache in
<ext>/views/NAME; `--jobs N` runs N engine processes at a time. DFL development matches only in
that mode.

Round 4 (test T-P, the owner's decision of 13 November 2026: PFF as a test only): `--games-set
pff` runs c23 on the 43 open PFF matches (folds.PFF_TRAIN, copied in PFF_OPEN below and checked
by the tests); the 4 pilot and 15 sealed PFF matches are not in that list and are refused.
`--games-set sc` is the 20 SkillCorner matches.
"""
import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys

DEV = ["METRICA-S1", "METRICA-S2", "METRICA-S3", "DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH",
       "DFL-J03WOY", "DFL-J03WPY"]
# folds.SC (the 20 SkillCorner matches of SKILLCORNER_REFERENCES.md)
SC = ["SC-1874553", "SC-1886347", "SC-1899585", "SC-1925299", "SC-1927964", "SC-1953632",
      "SC-1959846", "SC-1986691", "SC-1996435", "SC-1996436", "SC-2006229", "SC-2006363",
      "SC-2007448", "SC-2007721", "SC-2010085", "SC-2011166", "SC-2013725", "SC-2015213",
      "SC-2016236", "SC-2017461"]
C23_ENGINE_TREE = "241e70d6a0b29944f1ea25db2e9202aebdda3f1c"
DFL_DEV = [g for g in DEV if g.startswith("DFL-")]
# folds.PFF_TRAIN: the 43 open PFF matches (test only in round 4). Pilot and sealed never here.
PFF_OPEN = [
    "PFF-3814", "PFF-3815", "PFF-3816", "PFF-3818", "PFF-3819", "PFF-3820", "PFF-3822",
    "PFF-3823", "PFF-3824", "PFF-3826", "PFF-3827", "PFF-3828", "PFF-3830", "PFF-3831",
    "PFF-3832", "PFF-3834", "PFF-3835", "PFF-3836", "PFF-3838", "PFF-3839", "PFF-3841",
    "PFF-3843", "PFF-3844", "PFF-3846", "PFF-3848", "PFF-3849", "PFF-3850", "PFF-3852",
    "PFF-3853", "PFF-3854", "PFF-3856", "PFF-3857", "PFF-3858", "PFF-10503", "PFF-10504",
    "PFF-10505", "PFF-10507", "PFF-10508", "PFF-10509", "PFF-10511", "PFF-10513", "PFF-10514",
    "PFF-10516",
]


def _run(job):
    from analysis import external_sweep as xs
    return job["game"], xs.run_engine(job)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--engine-repo", required=True)
    p.add_argument("--ext", required=True, help="external_sweep cache (engine npz, runs)")
    p.add_argument("--out", required=True)
    p.add_argument("--games", nargs="+", default=DEV)
    p.add_argument("--data-root", default=None, help="R4: root of the still-ball DFL bundles")
    p.add_argument("--view", default="raw", help="R4: run-cache view name with --data-root")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--games-set", choices=("pff", "sc"), default=None,
                   help="R4 T-P: the 43 open PFF matches, or the 20 SkillCorner matches")
    a = p.parse_args(argv)
    if a.games_set is not None:
        a.games = list(PFF_OPEN) if a.games_set == "pff" else list(SC)
    if a.data_root is not None:
        bad = [g for g in a.games if g not in DFL_DEV]
        if bad or a.view == "raw":
            raise SystemExit("--data-root: DFL development matches and a view name other than "
                             "raw only ({})".format(bad))
        a.data_root = os.path.abspath(a.data_root)
    for game in a.games:      # no folds import: this process imports the c23 checkout's tools
        if game not in DEV and game not in SC and game not in PFF_OPEN:
            raise SystemExit("{}: development, SkillCorner or open PFF matches only".format(
                game))
    tree = subprocess.check_output(["git", "-C", a.engine_repo, "rev-parse", "HEAD:engine"]).decode().strip()
    dirty = subprocess.check_output(["git", "-C", a.engine_repo, "status", "--porcelain", "--", "engine", "games.py"]).decode().strip()
    if tree != C23_ENGINE_TREE or dirty:
        raise SystemExit("engine tree {} (dirty={}) is not c23 {}".format(tree, bool(dirty), C23_ENGINE_TREE))
    sys.path[:0] = [a.engine_repo, os.path.join(a.engine_repo, "tools")]
    from analysis import external_sweep as xs
    roots = [os.path.abspath(os.path.expanduser(r)) for r in
             (xs.DEFAULT_REFS_ROOT, xs.DEFAULT_SAMPLE_ROOT, xs.DEFAULT_MTAH_ROOT)]
    resolved, _ = xs.resolve_config({})
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for game in a.games:
        root = a.data_root or xs.data_root(game, "raw", *roots)
        jobs.append({"game": game, "overrides": {}, "data_dir": root,
                     "key": xs.run_key(game, resolved, root, xs.engine_source_hash(), a.view),
                     "where": xs.view_dir(os.path.abspath(a.ext), a.view)})
    if a.jobs > 1:
        with multiprocessing.get_context("fork").Pool(a.jobs) as pool:
            done = pool.map(_run, jobs)
    else:
        done = [_run(j) for j in jobs]
    for game, path in done:
        shutil.copyfile(path, os.path.join(a.out, game + ".csv"))
        print(game, file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
