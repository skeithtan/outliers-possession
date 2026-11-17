"""Apply the pre-registered selection rule of COMPLIANCE_PROTOCOL.md to external sweep tables.

    python tools/analysis/apply_selection.py --field NAME --sweep RAW.json MATCHED.json
        [--grids docs/plan/calibration/grids.json] [--tie-value X] [--tie-candidate C]
        [--out OUT.json]

The tool reads the tables of tools/analysis/external_sweep.py. It reads no match data and runs
no engine. It applies the rules mechanically. The shipped value of the field has no role.

Inputs:

- `--sweep` takes one JSON table per view. The tool reads the view of each table from its
  "view" key. A bucket a field needs its primary view and the other view (raw and matched).
- `--grids` gives the pre-registered entry of the field: bucket, judge, view, grid, tie_to,
  inert_fallback, tie. The grid of each table must equal this grid, in the same order. The tool
  applies the "appendix_8nov" key of grids.json. A field in "removed_by_precedence" gets the
  decision "removed_by_precedence" and no value. A field in "judge_changes" takes the new judge
  and view. A mixed view, "mixed: DFL matched + Metrica raw", reads the DFL scores from the
  matched table and the Metrica scores from the raw table. A pooled view, "pooled: raw +
  matched", makes each pair of a judging match and a view one judging row, "<match>@<view>",
  with the judge weight of the match shared equally by its views. A "recorded" provider view
  gets its o_m in the details. It selects nothing. When the recorded provider is the provider
  that does not judge, the guard of step 7 does not apply. The details give its result for
  information.
- `--tie-value` gives the current value of the tie_to field X. Without it, the tool resolves X
  from the base overrides of the primary table on top of engine/config.py.

Bucket a rule (COMPLIANCE_PROTOCOL.md, Section 1). S(m, g) is the full-match spellF1 of match m
at grid value g, in the primary view. The tool applies these steps in this order:

1. Inert. If max S(m, .) minus min S(m, .) is under 0.002 on every match of the table, in the
   primary view and in the other view, the decision is "inert". The field then takes its
   pre-registered fallback. The tool does not select a value.
2. Judge. "metrica" uses the three Metrica matches with equal weights. "dfl" uses the five DFL
   matches with equal weights. "all" uses the eight matches, 1/6 for each Metrica match and 1/10
   for each DFL match. The tool keeps the weights as exact fractions.
3. Per-match plateau. P_m holds every grid value g with S(m, g) >= max S(m, .) - 0.002. The tool
   allows 1e-9 of floating point rounding in this test. P_m can have gaps. o_m is the mean of
   the first and the last grid index of P_m, so o_m can be a half index.
   Grid edge (protocol appendix, 8 Nov 2026). If o_m is the first or the last grid index on more
   than half of the judging weight, the decision is "inconclusive". The field takes its
   inert_fallback of grids.json. A field without one takes the grid centre under the snapping-tie
   rule of step 6. The tool tests the grid edge before tie_to.
4. tie_to. The judged plateau is the set of grid values that are in P_m for a weighted majority
   of the judging matches. A weighted majority is a weight sum strictly above 1/2. If the field
   has tie_to X and the judged plateau contains the current value of X, the decision is
   "tie_to X" and the value is the current value of X. A value of X that is not a grid value
   cannot be in the judged plateau. The guard does not apply to a tie_to decision. The details
   record the guard test of that value for information.
5. Weighted median. Sort the o_m. The median is the smallest o_m at which the cumulative weight
   reaches 1/2 or more. If the cumulative weight is exactly 1/2 there, the median is the mean of
   that o_m and the next larger o_m.
6. Snap. The selected index is the grid index nearest to the median. On a snapping tie (the
   median is exactly halfway between two indices), the tool takes the index nearer to the grid
   centre index (n - 1) / 2. If both are equally near (an even grid and a median at the centre),
   the tool takes the lower index.
7. Guard. For judge "metrica" or "dfl", M(g) is the equal-weight mean of S(., g) over the
   matches of the other provider. The selected value passes when M(selected) >= max M - 0.005,
   with the same 1e-9 rounding allowance. If it fails, the tool moves to the grid value nearest
   in grid index that passes. A tie between two such values goes to the index nearer to the grid
   centre, then to the lower index. For judge "all" there is no other provider and no guard.

Bucket c rule (judge "metrica", views raw and matched). POSSESSION_RULE has two candidates.
CONTACT_DETECTOR and OWNER_RELEASE_KINDS take their candidates from the "bucket_c_record" key of
"appendix_8nov" in grids.json. The tool applies steps 3, 5 and 6 to the Metrica matches in each
view, with the candidates in their grid order as the index. A snapping tie gives no winner in
that view. A candidate is non-negative on a match when its paired 5 to 95 % interval against
every other candidate has an upper end at or above zero. The winner is the candidate that wins
in both views and is non-negative on the three Metrica matches in both views. Otherwise the
decision is the tie candidate. grids.json gives the tie as text, so the caller names the tie
candidate with `--tie-candidate` (a JSON literal or plain text). The base run of each table
must be one of the candidates, because the tool reads the paired intervals against the base.
With three or more candidates the tables give no interval between two candidates that are not
the base. If such a candidate wins, the tool stops and says so.

Output. A JSON object {field, decision, selected, o_m, details}. "decision" is "value",
"inert", "inconclusive", "tie_to" or "removed_by_precedence". "o_m" maps each judging match to its o_m in the primary view. "details"
holds the plateaus, the weights, the median, the snap, the guard and the inert ranges.
Deterministic. Python 3.9.
"""

import argparse
import json
import os
import sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_GRIDS = os.path.join(ROOT, "docs", "plan", "calibration", "grids.json")

TOLERANCE = 0.002          # per-match plateau and inert range (COMPLIANCE_PROTOCOL.md, 1)
GUARD = 0.005              # guard on the provider that does not judge (COMPLIANCE_PROTOCOL.md, 1)
EPS = 1e-9                 # floating point allowance in the two comparisons above
VIEWS = ("raw", "matched")
PREFIX = {"metrica": "METRICA-", "dfl": "DFL-"}


# -- the rule on plain numbers ---------------------------------------------------------------

def provider(game):
    """The provider of a match name: "metrica" or "dfl"."""
    for name, prefix in PREFIX.items():
        if game.startswith(prefix):
            return name
    raise ValueError("{}: no provider".format(game))


def judge_weights(judge, games):
    """{game: Fraction}: the judging matches and their weights for judge metrica, dfl or all."""
    if judge in PREFIX:
        chosen = [g for g in games if provider(g) == judge]
        if not chosen:
            raise ValueError("judge {} has no match in {}".format(judge, games))
        return {g: Fraction(1, len(chosen)) for g in chosen}
    if judge == "all":
        counts = {}
        for g in games:
            counts[provider(g)] = counts.get(provider(g), 0) + 1
        return {g: Fraction(1, len(counts) * counts[provider(g)]) for g in games}
    raise ValueError("unknown judge {}".format(judge))


def plateau(spells, tolerance=TOLERANCE):
    """The grid indices of P_m: S(m, g) >= max S(m, .) - tolerance."""
    top = max(spells)
    return [i for i, s in enumerate(spells) if s >= top - tolerance - EPS]


def midpoint(indices):
    """o_m: the mean of the first and the last index, as a Fraction (a whole or a half index)."""
    return Fraction(indices[0] + indices[-1], 2)


def weighted_median(points, weights):
    """The weighted median of {game: o_m} with {game: weight} (module docstring, step 5)."""
    order = sorted(set(points.values()))
    total = sum(weights[g] for g in points)
    cumulative = Fraction(0)
    for k, value in enumerate(order):
        cumulative += sum(weights[g] for g in points if points[g] == value)
        if 2 * cumulative >= total:
            if 2 * cumulative == total and k + 1 < len(order):
                return (value + order[k + 1]) / 2, True
            return value, False
    raise ValueError("empty median")


def toward_centre(candidates, n):
    """The index of `candidates` nearest to the grid centre (n - 1) / 2, then the lowest."""
    centre = Fraction(n - 1, 2)
    return min(candidates, key=lambda i: (abs(Fraction(i) - centre), i))


def snap(median, n):
    """(grid index nearest to `median`, True on a snapping tie). A tie goes toward the centre."""
    low = int(median)          # median >= 0, so int() is the floor
    if median == low:
        return low, False
    if median - low == Fraction(1, 2):
        return toward_centre([low, low + 1], n), True
    return (low if median - low < Fraction(1, 2) else low + 1), False


def guard(selected, other_means):
    """(index that passes the guard, best M, passed at first) (module docstring, step 7)."""
    best = max(other_means)
    ok = [i for i, m in enumerate(other_means) if m >= best - GUARD - EPS]
    if selected in ok:
        return selected, best, True
    distance = min(abs(i - selected) for i in ok)
    nearest = [i for i in ok if abs(i - selected) == distance]
    return toward_centre(nearest, len(other_means)), best, False


def score_range(spells):
    """max minus min of a list of spellF1 values."""
    return max(spells) - min(spells)


def select_grid(field, spec, values, views, primary, tie_value=None):
    """The bucket a decision of one field.

    `values` is the grid. `views` is {view: {game: [S(m, g) for g in values]}}. `primary` names
    the primary view, or maps each provider to its view ({"dfl": "matched", "metrica": "raw"}).
    `tie_value` is the current value of the tie_to field, if it has one.
    """
    if isinstance(primary, dict):
        games = sorted(next(iter(views.values())))
        table = {g: views[primary[provider(g)]][g] for g in games}
    elif isinstance(primary, list):
        # Pooled views: each (match, view) pair is one judging row, "<match>@<view>".
        games = sorted(next(iter(views.values())))
        table = {"{}@{}".format(g, v): views[v][g] for g in games for v in primary}
    else:
        table = views[primary]
        games = sorted(table)
    n = len(values)
    for view, per_game in views.items():
        if sorted(per_game) != games:
            raise ValueError("view {} has matches {}, the primary view has {}".format(
                view, sorted(per_game), games))
        for g in games:
            if len(per_game[g]) != n:
                raise ValueError("view {} match {} has {} scores for {} grid values".format(
                    view, g, len(per_game[g]), n))
    ranges = {view: {g: score_range(views[view][g]) for g in games} for view in sorted(views)}
    inert = (set(VIEWS) <= set(views)
             and all(r < TOLERANCE for per in ranges.values() for r in per.values()))
    details = {"primary_view": primary, "views": sorted(views), "grid": list(values),
               "judge": spec["judge"], "ranges": ranges, "inert": bool(inert),
               "inert_needs_both_views": not set(VIEWS) <= set(views)}
    out = {"field": field, "o_m": {}, "details": details}
    if inert:
        out.update(decision="inert", selected=None,
                   fallback=spec.get("inert_fallback", "no pre-registered fallback"))
        return out

    weights = judge_weights(spec["judge"], sorted(table))
    plateaus = {g: plateau(table[g]) for g in weights}
    o_m = {g: midpoint(plateaus[g]) for g in weights}
    share = [sum((weights[g] for g in weights if i in plateaus[g]), Fraction(0))
             for i in range(n)]
    judged = [i for i in range(n) if 2 * share[i] > 1]
    median, exact_half = weighted_median(o_m, weights)
    index, snap_tie = snap(median, n)
    details.update(
        weights={g: str(w) for g, w in sorted(weights.items())},
        plateaus={g: {"indices": plateaus[g], "values": [values[i] for i in plateaus[g]],
                      "best": max(table[g])} for g in sorted(plateaus)},
        plateau_share=[float(s) for s in share],
        judged_plateau=[values[i] for i in judged],
        weighted_median_index=float(median), median_exact_half=bool(exact_half),
        snapped_index=index, snap_tie=bool(snap_tie))
    out["o_m"] = {g: float(o) for g, o in sorted(o_m.items())}

    edge = sum((weights[g] for g in weights if o_m[g] in (0, n - 1)), Fraction(0))
    details["edge_weight"] = float(edge)
    details["edge"] = bool(2 * edge > 1)
    if details["edge"]:
        fallback = spec.get("inert_fallback")
        out["decision"] = "inconclusive"
        if fallback is not None:
            out.update(selected=None, fallback=fallback)
        else:
            centre = toward_centre([(n - 1) // 2, n // 2], n)
            out.update(selected=values[centre], selected_index=centre,
                       fallback="grid centre under the snapping-tie rule")
        return out

    other = None if spec["judge"] == "all" else [p for p in PREFIX if p != spec["judge"]][0]
    other_means = None
    if other is not None:
        members = [g for g in sorted(table) if provider(g) == other]
        other_means = [sum(table[g][i] for g in members) / len(members) for i in range(n)]
        details["guard_provider"] = other
        details["guard_means"] = other_means

    tie_to = spec.get("tie_to")
    if tie_to is not None:
        in_grid = [i for i, v in enumerate(values) if v == tie_value]
        hit = bool(in_grid) and in_grid[0] in judged
        details["tie_to"] = {"field": tie_to, "current_value": tie_value,
                             "in_grid": bool(in_grid), "in_judged_plateau": hit}
        if hit:
            if other_means is not None:
                passed = other_means[in_grid[0]] >= max(other_means) - GUARD - EPS
                details["tie_to"]["guard_passed_info"] = bool(passed)
            out.update(decision="tie_to", tie_to=tie_to, selected=tie_value)
            return out

    recorded_only = spec.get("recorded", "").split()[:1]
    if other_means is not None and [other] == [r.lower() for r in recorded_only]:
        final, best, passed = guard(index, other_means)
        details["guard"] = "none: the other provider is recorded only"
        details["guard_info"] = {"best": best, "passed": bool(passed), "would_move_to": final}
    elif other_means is not None:
        final, best, passed = guard(index, other_means)
        details.update(guard_best=best, guard_passed=bool(passed), guard_index=final)
        index = final
    else:
        details["guard"] = "none: judge all"
    out.update(decision="value", selected=values[index], selected_index=index)
    return out


def select_mechanism(field, spec, values, views, base_index, tie_candidate):
    """The bucket c decision of a field with two or more candidates (module docstring).

    `views` is {view: {game: entry}}. An entry is {"spells": [S per candidate], "versus":
    [(low, high) or None per candidate]}: the paired 5 to 95 % interval of each candidate minus
    the base candidate `base_index` on that match (None for the base). The two-candidate form
    {"spells", "low", "high"} gives the interval of the other candidate.
    """
    if len(values) < 2:
        raise ValueError("the bucket c rule compares two or more candidates")
    n = len(values)
    winners, per_view = {}, {}
    nonneg = [True] * n
    for view in sorted(views):
        table = {g: _versus_entry(e, base_index, n) for g, e in views[view].items()}
        weights = judge_weights(spec["judge"], sorted(table))
        plateaus = {g: plateau(table[g]["spells"]) for g in weights}
        o_m = {g: midpoint(plateaus[g]) for g in weights}
        median, exact_half = weighted_median(o_m, weights)
        tie = median.denominator != 1
        winners[view] = None if tie else int(median)
        signs = {}
        for g in weights:
            ok = _non_negative(table[g]["versus"], base_index)
            nonneg = [_and3(nonneg[k], ok[k]) for k in range(n)]
            signs[g] = {_label(values[k]): ok[k] for k in range(n)}
        per_view[view] = {"o_m": {g: float(o) for g, o in sorted(o_m.items())},
                          "plateaus": {g: [values[i] for i in plateaus[g]] for g in plateaus},
                          "weighted_median_index": float(median),
                          "median_exact_half": bool(exact_half),
                          "winner": None if tie else values[int(median)],
                          "non_negative": signs,
                          "spells": {g: table[g]["spells"] for g in sorted(table)}}
    agreed = set(winners.values())
    if tie_candidate not in values:
        raise ValueError("the tie candidate {!r} is not in {}".format(tie_candidate, values))
    winner = agreed.pop() if len(agreed) == 1 else None
    if winner is not None and nonneg[winner] is None:
        raise SystemExit("{}: the winner {!r} needs its paired intervals against every other "
                         "candidate; the tables give them against the base only".format(
                             field, values[winner]))
    if winner is not None and nonneg[winner]:
        chosen = values[winner]
        decision = "value"
    else:
        chosen = tie_candidate
        decision = "tie"
    return {"field": field, "decision": decision, "selected": chosen,
            "o_m": {view: per_view[view]["o_m"] for view in sorted(per_view)},
            "details": {"views": per_view,
                        "non_negative_all": {_label(values[k]): nonneg[k] for k in range(n)},
                        "judge": spec["judge"], "tie_text": spec.get("tie"),
                        "tie_candidate": tie_candidate,
                        "base_candidate": values[base_index]}}


def _and3(a, b):
    """Three-valued AND: False wins, then None (unknown), then True."""
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return True


def _label(value):
    """A candidate as a JSON object key: text stays, other values become their JSON literal."""
    return value if isinstance(value, str) else json.dumps(value)


def _versus_entry(entry, base_index, n):
    """The {"spells", "versus"} form of one match entry of select_mechanism()."""
    if "versus" in entry:
        return entry
    versus = [None] * n
    versus[1 - base_index] = (entry["low"], entry["high"])
    return {"spells": entry["spells"], "versus": versus}


def _non_negative(versus, base_index):
    """[True, False or None per candidate]: is its paired interval against each other one >= 0?

    A candidate c is non-negative against d when the interval of c minus d has an upper end at
    or above zero. The tables give k minus the base only. So the base is non-negative against k
    when low(k) <= 0, and k is non-negative against the base when high(k) >= 0. The interval of
    two candidates that are not the base is unknown (None). A missing interval counts as zero.
    """
    n = len(versus)
    out = []
    for c in range(n):
        if c == base_index:
            out.append(all(versus[k] is None or versus[k][0] is None or versus[k][0] <= 0.0
                           for k in range(n) if k != base_index))
            continue
        low_high = versus[c]
        ok = low_high is None or low_high[1] is None or low_high[1] >= 0.0
        if not ok:
            out.append(False)
        elif n > 2:
            out.append(None)
        else:
            out.append(True)
    return out


# -- tables on disk --------------------------------------------------------------------------

def load_tables(paths):
    """{view: table} of external_sweep JSON tables. Two tables of one view raise."""
    tables = {}
    for path in paths:
        with open(path) as handle:
            table = json.load(handle)
        view = table.get("view", "raw")
        if view in tables:
            raise SystemExit("two tables of view {}".format(view))
        tables[view] = table
    return tables


def table_spells(table):
    """{game: [S(m, g) for each grid value]} of one sweep table."""
    return {g: [row["games"][g]["spell_f1"] for row in table["rows"]] for g in table["games"]}


def resolve_tie_value(field, table):
    """The current value of `field` under the base overrides of `table` (engine/config.py)."""
    for path in (ROOT, os.path.join(ROOT, "tools", "analysis"), os.path.join(ROOT, "tools"),
                 os.path.join(ROOT, "evaluation")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import external_sweep  # noqa: E402
    resolved, _ = external_sweep.resolve_config(table.get("base_overrides", {}))
    return resolved[field]


def check_grid(field, spec, tables):
    """Raise SystemExit when a table is not the pre-registered grid of the field."""
    for view, table in tables.items():
        if table.get("field") != field:
            raise SystemExit("table of view {} is field {}, not {}".format(
                view, table.get("field"), field))
        if list(table["values"]) != list(spec["grid"]):
            raise SystemExit("{} view {}: grid {} is not the pre-registered grid {}".format(
                field, view, table["values"], spec["grid"]))


def parse_view(text):
    """A view of grids.json as the primary argument of select_grid().

    "raw" and "matched" stay text. "mixed: DFL matched + Metrica raw" becomes a dict
    {provider: view}. "pooled: raw + matched" becomes the list ["raw", "matched"].
    """
    if text.startswith("pooled:"):
        out = [part.strip() for part in text[len("pooled:"):].split("+")]
        if not out or not set(out) <= set(VIEWS) or len(set(out)) != len(out):
            raise ValueError("cannot read the view {!r}".format(text))
        return out
    if not text.startswith("mixed:"):
        return text
    out = {}
    for part in text[len("mixed:"):].split("+"):
        name, view = part.split()
        out[name.lower()] = view
    if sorted(out) != sorted(PREFIX) or not set(out.values()) <= set(VIEWS):
        raise ValueError("cannot read the view {!r}".format(text))
    return out


def effective_spec(field, grids, superseded=None):
    """The grids.json entry of `field` with the appendix_8nov judge changes applied.

    Return None when the field is in appendix_8nov "removed_by_precedence". `superseded` is an
    earlier judge_changes dict, to apply instead of the current one (a record of an old judge).
    """
    appendix = grids.get("appendix_8nov", {})
    if field in appendix.get("removed_by_precedence", []):
        return None
    spec = dict(grids["fields"][field])
    spec.update(appendix.get("bucket_c_record", {}).get(field, {}))
    change = (superseded or {}).get(field) if superseded is not None else \
        appendix.get("judge_changes", {}).get(field)
    if change:
        spec.update(judge=change["judge"], view=change["view"],
                    registered_judge=grids["fields"][field].get("judge"),
                    registered_view=grids["fields"][field].get("view"))
        if "recorded" in change:
            spec["recorded"] = change["recorded"]
    return spec


def recorded_view(spec, views, values):
    """The o_m of a recorded-only provider view ("DFL raw"), for information. It selects nothing."""
    name, view = spec["recorded"].split()
    games = [g for g in sorted(views[view]) if provider(g) == name.lower()]
    o_m = {g: float(midpoint(plateau(views[view][g]))) for g in games}
    return {"provider": name.lower(), "view": view, "o_m": o_m}


def apply(field, spec, tables, tie_value=None, tie_candidate=None):
    """The decision of `field` from its effective grids.json entry and its {view: table}."""
    if spec is None:
        return {"field": field, "decision": "removed_by_precedence", "selected": None}
    check_grid(field, spec, tables)
    values = list(spec["grid"])
    if spec["bucket"] == "a":
        primary = parse_view(spec["view"])
        if isinstance(primary, dict):
            needed = set(primary.values())
        elif isinstance(primary, list):
            needed = set(primary)
        else:
            needed = {primary}
        missing = needed - set(tables)
        if missing:
            raise SystemExit("{}: the primary view {} has no table".format(field, missing))
        if spec.get("tie_to") and tie_value is None:
            tie_value = resolve_tie_value(spec["tie_to"], tables[sorted(needed)[0]])
        views = {view: table_spells(t) for view, t in tables.items()}
        out = select_grid(field, spec, values, views, primary, tie_value)
        if spec.get("recorded"):
            out["details"]["recorded"] = recorded_view(spec, views, values)
        for key in ("registered_judge", "registered_view"):
            if key in spec:
                out["details"][key] = spec[key]
    elif spec["bucket"] == "c":
        views, base_index = {}, None
        for view, table in tables.items():
            hashes = table["config_hashes"]
            if table["base_config_hash"] not in hashes:
                raise SystemExit("{} view {}: the base run is not a candidate".format(field, view))
            index = hashes.index(table["base_config_hash"])
            if base_index is not None and index != base_index:
                raise SystemExit("{}: the base candidate differs between views".format(field))
            base_index = index
            spells = table_spells(table)
            views[view] = {g: {"spells": spells[g],
                               "versus": [None if k == index else
                                          (row["games"][g]["versus_base"]["low"],
                                           row["games"][g]["versus_base"]["high"])
                                          for k, row in enumerate(table["rows"])]}
                           for g in table["games"]}
        out = select_mechanism(field, spec, values, views, base_index, tie_candidate)
    else:
        raise SystemExit("{}: bucket {} has no grid rule".format(field, spec["bucket"]))
    out["sweeps"] = {view: {"engine_source": t.get("engine_source"),
                            "base_overrides": t.get("base_overrides", {}),
                            "base_config_hash": t.get("base_config_hash"),
                            "no_effect": t.get("no_effect")}
                     for view, t in sorted(tables.items())}
    return out


def main(argv=None):
    """CLI entry point. Return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--field", required=True, help="the field to decide")
    parser.add_argument("--sweep", nargs="+", required=True,
                        help="external_sweep JSON tables of the field, one per view")
    parser.add_argument("--grids", default=DEFAULT_GRIDS, help="the pre-registered grids.json")
    parser.add_argument("--tie-value", help="current value of the tie_to field, a JSON literal")
    parser.add_argument("--tie-candidate", help="bucket c: the candidate the tie text names")
    parser.add_argument("--out", help="write the decision JSON here (default: stdout only)")
    args = parser.parse_args(argv)
    with open(args.grids) as handle:
        grids = json.load(handle)
    if args.field not in grids["fields"]:
        parser.error("{} is not in {}".format(args.field, args.grids))
    spec = effective_spec(args.field, grids)
    tie_value = None if args.tie_value is None else json.loads(args.tie_value)
    tables = load_tables(args.sweep) if spec is not None else {}
    tie_candidate = args.tie_candidate
    if tie_candidate is not None:
        try:
            tie_candidate = json.loads(tie_candidate)
        except ValueError:
            pass
    out = apply(args.field, spec, tables, tie_value, tie_candidate)
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
