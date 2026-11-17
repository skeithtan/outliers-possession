"""The matches, views and folds of the learned stage. Not part of the submission.

This module implements Section 3 of docs/plan/calibration/LEARNED_STAGE_PROTOCOL.md. It has
no dependency outside the standard library, so the fold tests run without torch.

The protocol uses 28 external matches. OPEN holds the eight open references in the protocol
order. SC holds the 20 SkillCorner matches of SKILLCORNER_REFERENCES.md, sorted by name.

Fold k (k = 0 to 7) has these parts:

- test: OPEN[k] and every SC[j] with j mod 8 = k.
- validation: the first match of OPEN after OPEN[k], in cyclic order, with a provider that
  differs from the provider of OPEN[k]. Also SC[(k + 1) mod 8].
- training: the other matches.

Round 2 (LEARNED_STAGE_PROTOCOL_R2.md Section 1, change A) adds the 43 PFF training matches
PFF_TRAIN to the training set of every fold and of the final models. They are never in a
validation pair or a test set. Their only view is raw. The sealed PFF set PFF_SEALED (the veto
V1) and the pilot set PFF_PILOT are never training, validation or cache data:
assert_not_sealed() raises AssertionError for them, and fold(), train_games() and the training
loaders call it.

check() refuses the sealed DFL matches, the competition match and every name outside the 28.
check_train() also accepts the 43 PFF training matches, and nothing else. check_sealed_v1() is
the only gate that accepts a sealed PFF match, for the V1 tool of evaluate.py.
views() gives the training views of a match: SkillCorner has the raw and degraded views only,
PFF the raw view only. crop_shares() gives the share of training crops per provider that the
uniform match-then-view sampling implies. final_steps() gives the step count of the final models
from the best steps of the fold models.

Python 3.9.
"""

OPEN = ("METRICA-S1", "METRICA-S2", "METRICA-S3",
        "DFL-J03WMX", "DFL-J03WN1", "DFL-J03WOH", "DFL-J03WOY", "DFL-J03WPY")
SC = tuple(sorted((
    "SC-1874553", "SC-1886347", "SC-1899585", "SC-1925299", "SC-1927964",
    "SC-1953632", "SC-1959846", "SC-1986691", "SC-1996435", "SC-1996436",
    "SC-2006229", "SC-2006363", "SC-2007448", "SC-2007721", "SC-2010085",
    "SC-2011166", "SC-2013725", "SC-2015213", "SC-2016236", "SC-2017461",
)))
ALL = OPEN + SC
N_FOLDS = 8
SEALED = ("DFL-J03WQQ", "DFL-J03WR9")
COMPETITION = "BAR-ATH"     # games.GAME. This module does not import games.py.
FULL_VIEWS = ("raw", "matched", "sigma95", "degraded")
SC_VIEWS = ("raw", "degraded")
PFF_VIEWS = ("raw",)        # PFF is real broadcast tracking: no degraded view (R2 Section 1)
STEP_ROUND = 200

# PFF FC World Cup 2022 (LEARNED_STAGE_PROTOCOL_R2.md Section 1, PFF_TRANSFER_TEST.md Section 9).
# The pilot set P: c22 was scored on it. Round 2 does not use it at all.
PFF_PILOT = ("PFF-3812", "PFF-3840", "PFF-10502", "PFF-10517")
# The sealed set S: positions 0, 4, 8, ... of U \ P sorted by numeric id. The veto V1 only.
PFF_SEALED = (
    "PFF-3813", "PFF-3817", "PFF-3821", "PFF-3825", "PFF-3829", "PFF-3833", "PFF-3837",
    "PFF-3842", "PFF-3847", "PFF-3851", "PFF-3855", "PFF-3859", "PFF-10506", "PFF-10510",
    "PFF-10515",
)
# The 43 PFF training matches: U minus P minus S, sorted by numeric id.
PFF_TRAIN = (
    "PFF-3814", "PFF-3815", "PFF-3816", "PFF-3818", "PFF-3819", "PFF-3820", "PFF-3822",
    "PFF-3823", "PFF-3824", "PFF-3826", "PFF-3827", "PFF-3828", "PFF-3830", "PFF-3831",
    "PFF-3832", "PFF-3834", "PFF-3835", "PFF-3836", "PFF-3838", "PFF-3839", "PFF-3841",
    "PFF-3843", "PFF-3844", "PFF-3846", "PFF-3848", "PFF-3849", "PFF-3850", "PFF-3852",
    "PFF-3853", "PFF-3854", "PFF-3856", "PFF-3857", "PFF-3858", "PFF-10503", "PFF-10504",
    "PFF-10505", "PFF-10507", "PFF-10508", "PFF-10509", "PFF-10511", "PFF-10513", "PFF-10514",
    "PFF-10516",
)
# Label-free intake exclusions (PFF_REFERENCES.md): no games.py entry, never used.
PFF_EXCLUDED = ("PFF-3845", "PFF-10512")
# The round 2 training set of the final models: the 28 external matches and the 43 PFF.
TRAIN_ALL = ALL + PFF_TRAIN
# Never training, validation or cache data for training (assert_not_sealed()). The sealed DFL
# matches and BAR-ATH stay refused by check() with SystemExit, as in round 1.
NEVER_TRAIN = frozenset(PFF_SEALED + PFF_PILOT + PFF_EXCLUDED)


def provider(game):
    """The provider of a match: "metrica", "dfl" or "skillcorner"."""
    if game.startswith("METRICA-"):
        return "metrica"
    if game.startswith("DFL-"):
        return "dfl"
    if game.startswith("SC-"):
        return "skillcorner"
    if game.startswith("PFF-"):
        return "pff"
    raise ValueError("{}: unknown provider".format(game))


def assert_not_sealed(names):
    """Raise AssertionError if a name is a sealed, pilot or excluded PFF match.

    This is the code assertion of LEARNED_STAGE_PROTOCOL_R2.md Section 1: S and P never enter
    any training, validation or cache used by training. Return the names as a list.
    """
    names = list(names)
    bad = sorted(set(names) & NEVER_TRAIN)
    assert not bad, "never training, validation or training-cache data: {}".format(bad)
    return names


def check(game):
    """Raise SystemExit if `game` is sealed, the competition match or not one of the 28."""
    if game in SEALED or game == COMPETITION:
        raise SystemExit("{} is forbidden for the learned stage (sealed or competition "
                         "match)".format(game))
    if game not in ALL:
        raise SystemExit("{} is not one of the 28 external matches of the protocol".format(game))
    return game


def check_all(names):
    """check() on each name. Return the names as a list."""
    return [check(g) for g in names]


def check_train(game):
    """Return `game` if it may be training data (the 28 or the 43 PFF). Else raise.

    A sealed or pilot PFF match raises AssertionError (assert_not_sealed()). Any other name
    outside TRAIN_ALL raises SystemExit.
    """
    assert_not_sealed([game])
    if game in PFF_TRAIN:
        return game
    return check(game)


def check_train_all(names):
    """check_train() on each name. Return the names as a list."""
    return [check_train(g) for g in names]


def check_sealed_v1(game):
    """Return `game` if it is in the sealed PFF set S. The V1 tool of evaluate.py only."""
    if game not in PFF_SEALED:
        raise SystemExit("{} is not in the sealed PFF set S".format(game))
    return game


def views(game):
    """The training views of a match (protocol Section 2, R2 Section 1 for PFF)."""
    check_train(game)
    p = provider(game)
    if p == "pff":
        return PFF_VIEWS
    return SC_VIEWS if p == "skillcorner" else FULL_VIEWS


def open_validation(k):
    """The first match of OPEN after OPEN[k], in cyclic order, of another provider."""
    test = OPEN[k]
    order = OPEN[k + 1:] + OPEN[:k]
    for game in order:
        if provider(game) != provider(test):
            return game
    raise ValueError("no match of another provider in OPEN")


def fold(k, pff=True):
    """{"k", "test", "val", "train"} of fold k. pff=False (R3b) drops the PFF training matches.

    test and val come from the 28 external matches, in the order of ALL. train holds the other
    external matches in the order of ALL, then the 43 PFF training matches (R2 Section 1).
    """
    if not 0 <= k < N_FOLDS:
        raise ValueError("fold {} is outside 0..{}".format(k, N_FOLDS - 1))
    test = [OPEN[k]] + [g for j, g in enumerate(SC) if j % N_FOLDS == k]
    val = [open_validation(k), SC[(k + 1) % N_FOLDS]]
    held = set(test) | set(val)
    train = [g for g in ALL if g not in held] + (list(PFF_TRAIN) if pff else [])
    assert_not_sealed(test + val + train)
    assert not set(test + val) & set(PFF_TRAIN), "a PFF match in a validation pair or test set"
    return {"k": k, "test": check_all(test), "val": check_all(val),
            "train": check_train_all(train)}


def folds():
    """The eight folds, in order."""
    return [fold(k) for k in range(N_FOLDS)]


def train_games(pff=True):
    """The training matches of the final models: TRAIN_ALL (R2 Section 1); ALL without PFF."""
    return check_train_all(assert_not_sealed(TRAIN_ALL if pff else ALL))


def crop_shares(train):
    """{provider: share of the training crops} under uniform match-then-view sampling.

    A crop draws a training match uniformly, then one of its views (train.sample_batch()), so
    the share of a provider is its number of training matches over the total. The views do
    not change it.
    """
    train = check_train_all(train)
    counts = {}
    for game in train:
        counts[provider(game)] = counts.get(provider(game), 0) + 1
    return {p: counts[p] / float(len(train)) for p in sorted(counts)}


def final_steps(best_steps):
    """The median of the fold best steps, rounded to the nearest STEP_ROUND, at least STEP_ROUND.

    The median of an even count is the mean of the two middle values. A half goes up.
    """
    values = sorted(int(s) for s in best_steps)
    if not values:
        raise ValueError("no best steps")
    mid = len(values) // 2
    med = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0
    return max(STEP_ROUND, int(med / STEP_ROUND + 0.5) * STEP_ROUND)
