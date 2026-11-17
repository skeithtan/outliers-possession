"""K1 measurement: which team does each contact rule give an impulse maximum? Not part of the submission.

    python tools/analysis/kink_attribution.py --view matched --out /tmp/k1_matched.json
        [--games G1 G2 ...] [--jobs N] [--cache DIR] [--refs-root DIR] [--sample-root DIR]
        [--mtah-root DIR]

The tool compares four rules for the team distances of a bump contact (engine/events.py):

- `window`: the C8 rule, the nearest distance of each team over [c - h, c + h].
- `kink`: the distance of each team to the kink contact point (events.kink_points and
  events.kink_team_distances), the window rule where the fit has no point.
- `min`: per team, the smaller of the two (CONTACT_POINT "kink_min").
- `mean`: per team, the mean of the two, the window rule where the fit has no point.

For each rule a contact gets a team when the nearer team is within R_TOUCH and the contact is
not contested (the other team within R_TOUCH and less than CONTESTED_MARGIN_M further), the
rule of _bump_contacts. The proxy for the team that played the ball is the reference label 12
frames after the maximum (half the impulse window, h). The proxy is wrong for a deflection that
does not change the owner and for a ball lost at once, the same way for every rule, so the
comparison between rules is the measurement, not the agreement rate itself.

The contacts are the impulse maxima of the default engine on alive frames (|J| >= DV_CONTACT,
window_maxima). The tool reads the engine through possession.prepare() and the events module. It
never runs on the competition match or on a sealed match. It prints one JSON of counts per match
and per rule, and the per-provider totals. Deterministic. Python 3.9.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
ANALYSIS = os.path.join(TOOLS, "analysis")
for _path in (ROOT, TOOLS, ANALYSIS, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

RULES = ("window", "kink", "min", "mean")
# The proxy reads the reference this many frames after the impulse maximum: h of the C8 rule.
PROXY_LAG = 12


def team_of(d_home, d_away, cfg):
    """0 (H), 1 (A) or None: the team of a contact under the _bump_contacts rule."""
    near = min(d_home, d_away)
    if not near <= cfg.R_TOUCH:
        return None
    far = max(d_home, d_away)
    if far <= cfg.R_TOUCH and (far - near) < cfg.CONTESTED_MARGIN_M:
        return None
    return 0 if d_home <= d_away else 1


def child(game, view, data_cache, ref_path):
    """Counts of one match and view, as a dict. Runs inside the subprocess of the match."""
    import numpy as np
    from engine import data, events, possession
    from engine.config import Config
    import possession_metrics as pm

    data.DEFAULT_CACHE_DIR = data_cache
    cfg = Config.default()
    inputs = possession.prepare(cfg, game)
    tactical = possession.engine_tactical(cfg, game)
    ball, feats, e_dead = inputs["ball"], inputs["feats"], inputs["e_dead"]
    alive = ~(np.isfinite(e_dead) & (e_dead > cfg.EVENT_SUPPRESS_DEAD))
    h = int(cfg.CONTACT_HALF_WINDOW)
    size = events.impulse(ball, h)
    with np.errstate(invalid="ignore"):
        strong = np.isfinite(size) & (size >= cfg.DV_CONTACT) & alive
    centres = events.window_maxima(size, h, np.flatnonzero(strong))
    points = events.kink_points(ball, centres, cfg)
    kink = events.kink_team_distances(points, events.match_players(tactical), cfg)
    ref = pm.label_codes(pm.read_label_csv(ref_path))
    n = ref.shape[0]
    d_h = np.where(np.isfinite(feats["dH"]), feats["dH"], np.inf)
    d_a = np.where(np.isfinite(feats["dA"]), feats["dA"], np.inf)
    counts = {rule: {"attributed": 0, "agree": 0} for rule in RULES}
    pair = {"both": 0, "disagree": 0, "window_agrees": 0, "min_agrees": 0}
    fitted = 0
    for index, c in enumerate(centres):
        c = int(c)
        proxy = int(ref[min(n - 1, c + PROXY_LAG)])
        if proxy not in (pm.H, pm.A):
            continue
        lo, hi = max(0, c - h), min(n, c + h + 1)
        window = (float(np.min(d_h[lo:hi])), float(np.min(d_a[lo:hi])))
        has_point = bool(np.isfinite(points[0][index]))
        fitted += int(has_point)
        point = (float(kink["H"][index]), float(kink["A"][index]))
        by_rule = {
            "window": window,
            "kink": point if has_point else window,
            "min": (min(window[0], point[0]), min(window[1], point[1])),
            "mean": (((window[0] + point[0]) / 2.0, (window[1] + point[1]) / 2.0)
                     if has_point else window),
        }
        teams = {}
        for rule in RULES:
            team = team_of(by_rule[rule][0], by_rule[rule][1], cfg)
            teams[rule] = team
            if team is not None:
                counts[rule]["attributed"] += 1
                counts[rule]["agree"] += int(team == proxy)
        if teams["window"] is not None and teams["min"] is not None:
            pair["both"] += 1
            if teams["window"] != teams["min"]:
                pair["disagree"] += 1
                pair["window_agrees"] += int(teams["window"] == proxy)
                pair["min_agrees"] += int(teams["min"] == proxy)
    return {"game": game, "view": view, "maxima": int(centres.size), "fitted": fitted,
            "rules": counts, "window_vs_min": pair}


def run_child(job):
    """Run one match in a subprocess with its MTAH_DATA_DIR. Return the parsed JSON."""
    env = dict(os.environ, MTAH_DATA_DIR=job["data_dir"])
    command = [sys.executable, os.path.abspath(__file__), "--child", "--game", job["game"],
               "--view", job["view"], "--data-cache", job["data_cache"], "--ref", job["ref"]]
    done = subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=subprocess.PIPE)
    return json.loads(done.stdout.decode("utf-8").strip().splitlines()[-1])


def totals(results):
    """{group: {rule: {attributed, agree, accuracy}}, ...} over the matches of each provider."""
    out = {}
    for group, prefix in (("metrica", "METRICA-"), ("dfl", "DFL-"), ("all", "")):
        rules = {rule: {"attributed": 0, "agree": 0} for rule in RULES}
        pair = {"both": 0, "disagree": 0, "window_agrees": 0, "min_agrees": 0}
        for result in results:
            if not result["game"].startswith(prefix):
                continue
            for rule in RULES:
                for key in ("attributed", "agree"):
                    rules[rule][key] += result["rules"][rule][key]
            for key in pair:
                pair[key] += result["window_vs_min"][key]
        for rule in RULES:
            att = rules[rule]["attributed"]
            rules[rule]["accuracy"] = rules[rule]["agree"] / float(att) if att else None
        out[group] = {"rules": rules, "window_vs_min": pair}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--view", default="matched", choices=("raw", "matched", "sigma95"))
    parser.add_argument("--games", nargs="*", default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--refs-root", default=None)
    parser.add_argument("--sample-root", default=None)
    parser.add_argument("--mtah-root", default=None)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--game", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--data-cache", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ref", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        print(json.dumps(child(args.game, args.view, args.data_cache, args.ref), sort_keys=True))
        return 0

    import external_sweep as es
    games = es.check_games(args.games or list(es.OPEN_GAMES))
    refs_root = os.path.expanduser(args.refs_root or es.DEFAULT_REFS_ROOT)
    sample_root = os.path.expanduser(args.sample_root or es.DEFAULT_SAMPLE_ROOT)
    mtah_root = os.path.expanduser(args.mtah_root or es.DEFAULT_MTAH_ROOT)
    cache = os.path.abspath(os.path.expanduser(args.cache or es.DEFAULT_CACHE))
    jobs = []
    for game in games:
        ref, _ = es.prepare_reference(game, refs_root, sample_root, cache)
        jobs.append({"game": game, "view": args.view, "ref": ref,
                     "data_dir": es.data_root(game, args.view, refs_root, sample_root, mtah_root),
                     "data_cache": os.path.join(es.view_dir(cache, args.view), "engine_data")})
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(run_child, jobs))
    report = {"view": args.view, "proxy_lag": PROXY_LAG, "matches": results,
              "totals": totals(results)}
    text = json.dumps(report, sort_keys=True, indent=1)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    for group in ("metrica", "dfl"):
        block = report["totals"][group]
        line = " | ".join("{} {}/{} = {:.3f}".format(rule, block["rules"][rule]["agree"],
                                                     block["rules"][rule]["attributed"],
                                                     block["rules"][rule]["accuracy"] or 0.0)
                          for rule in RULES)
        pair = block["window_vs_min"]
        print("{} {}: {} | window and min disagree on {}: window right {}, min right {}".format(
            args.view, group, line, pair["disagree"], pair["window_agrees"], pair["min_agrees"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
