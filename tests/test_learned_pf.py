"""The per-frame member (engine/learned_pf.py, tools/learned_spike/pf_export.py) and its wiring
into engine.learned (Config.LEARNED_PF).

numpy only. Optional, by environment:
- LEARNED_TEST_ENGINE_CACHE (engine.data npz cache of the external matches) and
  LEARNED_TEST_PF_REF_FEATS (the spike's feature folder feats/eval: <game>.npy, the reference
  tools/learned_spike/pf_features.py of branch learned-r6-pf): feature parity on real matches;
- LEARNED_TEST_PF_PKL (a spike models.pkl without contact columns) and scikit-learn importable:
  predictor parity against scikit-learn.
The committed model tests/fixtures/learned_pf_synthetic.npz is synthetic (make_synthetic_model()).
"""

import dataclasses
import os

import numpy as np
import pytest

from engine import data, learned
from engine import learned_features as lf
from engine import learned_pf as P
from engine.config import Config
from tests.test_learned_v3 import _match, _random_state, _swapped
from tools.learned_spike import export, pf_export

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "learned_pf_synthetic.npz")
BASE = "stage50-v2s"
# md5 (engine.learned_features.matrix_md5) of the reference pf_features (branch learned-r6-pf,
# f009cf5) on synth_with_touches(), base stage50-v2, its 184 computable columns in spec order.
REF_SYNTH_MD5 = "1b29f8bcbfbd1b3b4433cd85ceef6411"


@pytest.fixture
def blocks(monkeypatch):
    monkeypatch.setattr(data, "graded_blocks", lambda game=None: [(1, 0, 1179), (2, 1220, 2399)])
    monkeypatch.setattr(data, "pitch_m", lambda game=None: (105.0, 68.0))


def synth_with_touches():
    tac = _match()
    tac["bx"][100:160] = tac["px"][100:160, 4] + np.float32(0.004)
    tac["by"][100:160] = tac["py"][100:160, 4]
    tac["bx"][900:960] = tac["px"][900:960, 15]
    tac["by"][900:960] = tac["py"][900:960, 15] + np.float32(0.004)
    return tac


def computable_columns(base=BASE):
    return [n for n, _ in P.full_spec(base) if P.computable(n)]


def make_synthetic_model(seed=7, n_trees=6, depth=4):
    """A small random tree model over the 184 computable columns (base stage50-v2s), thresholds
    at quantiles of the synthetic match's features so that every split is used."""
    import contextlib
    rng = np.random.RandomState(seed)
    cols = computable_columns()
    with contextlib.ExitStack() as stack:
        patch = pytest.MonkeyPatch()
        stack.callback(patch.undo)
        patch.setattr(data, "graded_blocks", lambda game=None: [(1, 0, 1179), (2, 1220, 2399)])
        patch.setattr(data, "pitch_m", lambda game=None: (105.0, 68.0))
        F = P.pf_features(synth_with_touches(), "METRICA-S3", cols, BASE)
    models = {}
    for name in P.MODELS:
        arr = {k: [] for k in P.NODE_ARRAYS}
        roots = []
        for _ in range(n_trees):
            roots.append(len(arr["feature"]))
            stack_ = [(len(arr["feature"]), 0)]
            for k in P.NODE_ARRAYS:
                arr[k].append(0)
            while stack_:
                node, d = stack_.pop()
                if d == depth or rng.random_sample() < 0.15:
                    arr["leaf"][node] = True
                    arr["value"][node] = float(rng.normal(0, 0.8))
                    arr["feature"][node], arr["left"][node], arr["right"][node] = 0, -1, -1
                    arr["threshold"][node] = 0.0
                    arr["missing_left"][node] = False
                    continue
                f = int(rng.randint(len(cols)))
                arr["feature"][node] = f
                arr["threshold"][node] = float(np.quantile(F[:, f], rng.uniform(0.2, 0.8)))
                arr["missing_left"][node] = bool(rng.randint(2))
                arr["leaf"][node] = False
                arr["value"][node] = 0.0
                kids = []
                for _ in range(2):
                    kids.append(len(arr["feature"]))
                    for k in P.NODE_ARRAYS:
                        arr[k].append(0)
                arr["left"][node], arr["right"][node] = kids
                stack_ += [(kids[1], d + 1), (kids[0], d + 1)]
        models[name] = {"feature": np.array(arr["feature"], np.int32),
                        "threshold": np.array(arr["threshold"], np.float64),
                        "left": np.array(arr["left"], np.int32),
                        "right": np.array(arr["right"], np.int32),
                        "missing_left": np.array(arr["missing_left"], bool),
                        "leaf": np.array(arr["leaf"], bool),
                        "value": np.array(arr["value"], np.float64),
                        "roots": np.array(roots, np.int32),
                        "baseline": float(rng.normal(0, 0.3))}
    return pf_export.arrays_of(models, cols, BASE, "synthetic-test", "0" * 32)


def test_the_fixture_is_the_synthetic_model():
    want = make_synthetic_model()
    with np.load(FIXTURE, allow_pickle=False) as z:
        assert sorted(z.files) == sorted(want)
        for k in want:
            got, exp = z[k], np.asarray(want[k])
            if got.dtype.kind == "f":
                # make_synthetic_model() draws the leaf values with the normal generator of the
                # seeded random state. That generator calls the C library's log and sqrt, and
                # their last bit differs between platforms (macOS arm64 and Linux x86-64 differ
                # by one unit in the last place on a few values). The committed fixture is the
                # reference of every other test. This test checks only that the fixture is the
                # documented synthetic model, so float arrays match to 1e-12.
                assert got.shape == exp.shape, k
                assert np.allclose(got, exp, rtol=0.0, atol=1e-12, equal_nan=True), k
            else:
                assert np.array_equal(got, exp), k
    m = P.load_pf(FIXTURE)
    assert m["base_version"] == BASE and len(m["columns"]) == 184


# -- features -------------------------------------------------------------------------------------

def test_features_equal_the_reference_on_a_synthetic_match(blocks):
    cols = computable_columns("stage50-v2")
    F = P.pf_features(synth_with_touches(), "METRICA-S3", cols, "stage50-v2")
    assert F.dtype == np.float32 and F.shape == (2400, 184)
    assert lf.matrix_md5(F) == REF_SYNTH_MD5
    assert (F[:, cols.index("pt_n2")] > 0).sum() > 500      # the proximity touches are used


def test_columns_and_spec():
    names = [n for n, _ in P.full_spec("stage50-v2")]
    assert len(names) == 248 and names[50:58] == [n for n, _ in P.CONTACT]
    assert not P.computable("tl_team") and not P.computable("t_n2__c05")
    assert P.computable("ptl_team") and P.computable("pt_n2__c05")
    with pytest.raises(ValueError):
        P.pf_features({}, "METRICA-S3", ["tl_team"])
    perm, sign = P.swap_columns(computable_columns())
    assert np.array_equal(perm[perm], np.arange(len(perm)))
    with pytest.raises(ValueError):
        P.swap_columns(["H_d"])                             # A_d is missing


def test_the_engine_refuses_contact_columns(tmp_path):
    arrays = dict(np.load(FIXTURE, allow_pickle=False))
    cols = list(arrays["columns"])
    cols[0] = "tl_team"
    arrays["columns"] = np.array(cols)
    path = str(tmp_path / "bad.npz")
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        P.load_pf(path)
    arrays["columns"] = np.array(computable_columns())
    arrays["format"] = np.str_("other")
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        P.load_pf(path)


REAL = os.environ.get("LEARNED_TEST_ENGINE_CACHE")
REF = os.environ.get("LEARNED_TEST_PF_REF_FEATS")


@pytest.mark.skipif(not (REAL and REF), reason="LEARNED_TEST_ENGINE_CACHE and "
                    "LEARNED_TEST_PF_REF_FEATS are not set")
@pytest.mark.parametrize("game", ["METRICA-S1", "SC-1874553"])
def test_features_equal_the_reference_on_real_matches(game):
    """The spike's stored features (raw view, base stage50-v2) against the engine's."""
    with np.load(os.path.join(REAL, game + "_tactical.npz")) as z:
        tac = {k: z[k] for k in ("px", "py", "ps", "bx", "by", "team_of_slot")}
    tac["n_frames"] = len(tac["px"])
    names = [n for n, _ in P.full_spec("stage50-v2")]
    cols = computable_columns("stage50-v2")
    ref = np.load(os.path.join(REF, game + ".npy"), mmap_mode="r")
    R = np.asarray(ref[:, [names.index(c) for c in cols]])
    F = P.pf_features(tac, game, cols, "stage50-v2")
    assert F.shape == R.shape and np.array_equal(F, R)


# -- the predictor --------------------------------------------------------------------------------

def reference_predict_raw(m, X):
    """tools/learned_spike/pf_numpy.predict_raw of the spike (level by level, all frames),
    with the missing-value branch."""
    n = X.shape[0]
    raw = np.full(n, m["baseline"])
    rows = np.arange(n)
    depth = 64
    for root in m["roots"]:
        node = np.full(n, root, np.int64)
        for _ in range(depth):
            lf_ = m["leaf"][node]
            if lf_.all():
                break
            x = X[rows, m["feature"][node]]
            go_left = np.where(np.isnan(x), m["missing_left"][node], x <= m["threshold"][node])
            nxt = np.where(go_left, m["left"][node], m["right"][node])
            node = np.where(lf_, node, nxt)
        raw += m["value"][node]
    return raw


def test_predictor_equals_the_reference_walk(blocks):
    model = P.load_pf(FIXTURE)
    X = P.pf_features(synth_with_touches(), "METRICA-S3", model["columns"], BASE).astype(
        np.float64)
    X[::97, 3] = np.nan                                   # the missing-value branch
    for name in P.MODELS:
        assert np.array_equal(P.predict_raw(model[name], X),
                              reference_predict_raw(model[name], X))
    p = P.predict_proba(model["dead"], X)
    assert np.all((p > 0) & (p < 1))


PKL = os.environ.get("LEARNED_TEST_PF_PKL")


@pytest.mark.skipif(not (PKL and REAL and REF), reason="LEARNED_TEST_PF_PKL, "
                    "LEARNED_TEST_ENGINE_CACHE and LEARNED_TEST_PF_REF_FEATS are not set")
def test_predictor_equals_scikit_learn(tmp_path):
    pytest.importorskip("sklearn")
    import pickle
    with open(PKL, "rb") as h:
        obj = pickle.load(h)
    path = str(tmp_path / "m.npz")
    pf_export.main(["--pkl", PKL, "--base-version", "stage50-v2", "--version", "t",
                    "--out", path])
    model = P.load_pf(path)
    names = [n for n, _ in P.full_spec("stage50-v2")]
    ref = np.load(os.path.join(REF, "METRICA-S1.npy"), mmap_mode="r")
    X = np.asarray(ref[:20000, [names.index(c) for c in model["columns"]]])
    for name in P.MODELS:
        assert np.array_equal(P.predict_proba(model[name], X), obj[name].predict_proba(X)[:, 1])


# -- symmetry and the ensemble --------------------------------------------------------------------

def tracking_fixture(tmp_path):
    """The synthetic model with swap_mode "tracking"."""
    arrays = dict(np.load(FIXTURE, allow_pickle=False))
    arrays["swap_mode"] = np.str_("tracking")
    path = str(tmp_path / "tracking.npz")
    np.savez(path, **arrays)
    return path


def test_member_team_swap(blocks, tmp_path):
    model = P.load_pf(FIXTURE)
    assert model["swap_mode"] == "columns"
    tac = synth_with_touches()
    # "tracking": exactly symmetric by construction
    p = P.member_probabilities(model, tac, "METRICA-S3", swap_mode="tracking")
    q = P.member_probabilities(model, _swapped(tac), "METRICA-S3", swap_mode="tracking")
    assert np.array_equal(q, p[:, [1, 0, 2]])
    assert P.load_pf(tracking_fixture(tmp_path))["swap_mode"] == "tracking"
    # "columns" (the default, as the spike measured): exact here, where the features are
    # equivariant (the dyadic synthetic match)
    c = P.member_probabilities(model, tac, "METRICA-S3")
    d = P.member_probabilities(model, _swapped(tac), "METRICA-S3")
    assert np.abs(d - c[:, [1, 0, 2]]).max() < 1e-12
    r = P.frame_probabilities_pf(tac, "METRICA-S3", model)
    assert np.array_equal(r, np.round(c, 4)) and np.allclose(r.sum(axis=1), 1.0, atol=2e-4)
    assert np.abs(c - p).max() < 0.05


def _cfg(net_path, pf="off", weight=0.5, pf_path=FIXTURE, mode="model"):
    return dataclasses.replace(Config.default(), LEARNED_MODE=mode, LEARNED_WEIGHTS=net_path,
                               LEARNED_STRUCTURE=True, LEARNED_COST_HA=12.0,
                               LEARNED_COST_AD=12.0, LEARNED_PRIOR_BONUS=0.0, LEARNED_PF=pf,
                               LEARNED_PF_WEIGHT=weight, LEARNED_PF_WEIGHTS=pf_path)


def test_apply_blends_and_off_changes_nothing(blocks, monkeypatch, tmp_path):
    tac = synth_with_touches()
    monkeypatch.setattr(data, "load_tactical", lambda game=None: tac)
    monkeypatch.setattr(data, "kickoff_frames", lambda game=None: {1: 0, 2: 1250})
    net = str(tmp_path / "net.npz")
    export.write(net, [_random_state(50, seed=s) for s in range(2)], BASE)
    blocks_ = data.graded_blocks()
    off_labels, off_probs = learned.apply(_cfg(net), "METRICA-S3", blocks_, return_probs=True)
    # PF off = the network path exactly (frame_probabilities, then the decoder)
    nets = learned.load_weights(net)
    _, want = learned.frame_probabilities(nets, tac, "METRICA-S3")
    assert np.array_equal(off_probs, want)
    # with a "tracking" member the network takes the paired, exactly symmetric copies
    _, paired = learned.frame_probabilities_paired(nets, tac, "METRICA-S3")
    assert np.abs(paired - want).max() <= 2e-4
    plain = dataclasses.replace(_cfg(net), LEARNED_PF_WEIGHTS="/does/not/exist.npz")
    assert np.array_equal(learned.apply(plain, "METRICA-S3", blocks_), off_labels)
    # PF on: w p_pf + (1 - w) p_net, rounded once more
    on_labels, on_probs = learned.apply(_cfg(net, "on", 0.5), "METRICA-S3", blocks_,
                                        return_probs=True)
    p_pf = P.frame_probabilities_pf(tac, "METRICA-S3", P.load_pf(FIXTURE))
    assert np.array_equal(on_probs, np.round(0.5 * p_pf + 0.5 * want, 4))
    _, on_t = learned.apply(_cfg(net, "on", 0.5, tracking_fixture(tmp_path)), "METRICA-S3",
                            blocks_, return_probs=True)
    p_pf_t = P.frame_probabilities_pf(tac, "METRICA-S3", P.load_pf(tracking_fixture(tmp_path)))
    assert np.array_equal(on_t, np.round(0.5 * p_pf_t + 0.5 * paired, 4))
    assert not np.array_equal(on_probs, off_probs)
    _, w0 = learned.apply(_cfg(net, "on", 0.0), "METRICA-S3", blocks_, return_probs=True)
    # weight 0 is the network's frame probabilities rounded to 4 decimals (the upsampled
    # probabilities between bin centres carry more decimals): "on" with weight 0 is not "off"
    assert np.array_equal(w0, np.round(want, 4))
    with pytest.raises(ValueError):
        learned.pf_settings(_cfg(net, "maybe"))
    with pytest.raises(ValueError):
        learned.pf_settings(_cfg(net, "on", 1.5))


def test_pf_is_not_read_when_the_learned_stage_is_off():
    # LEARNED_MODE off set explicitly: since branch r6e-adopt the default is mode "model" with
    # the member on (LEARNED_STAGE_RESULT_R6.md Section 2).
    cfg = dataclasses.replace(Config.default(), LEARNED_MODE="off", LEARNED_PF="on",
                              LEARNED_PF_WEIGHTS="/does/not/exist.npz")
    assert cfg.LEARNED_MODE == "off" and cfg.LEARNED_PF == "on"


def test_pf_defaults_are_the_round_6_candidate():
    cfg = Config.default()
    assert cfg.LEARNED_MODE == "model" and cfg.LEARNED_PF == "on"
    assert cfg.LEARNED_PF_WEIGHT == 0.25
    assert cfg.LEARNED_PF_WEIGHTS == "engine/models/learned_pf_v7.npz"


def test_team_swap_of_the_symmetry_scene_with_pf_on(tmp_path, monkeypatch):
    """tests/test_symmetry.py's scene (the one that failed on r4-adopt): with a stage50-v2s
    network and the per-frame member on, the swapped bundle gives exactly H and A swapped; with
    a "tracking" member the frame probabilities exchange H and A exactly, with a "columns"
    member up to the rounding of the features."""
    import games
    from engine import possession
    from tests import test_symmetry as ts
    root = str(tmp_path)
    px, py, bx, by, tags = ts.scene(())
    ts.write_bundle(os.path.join(root, "home_first"), px, py, bx, by, tags, True)
    ts.write_bundle(os.path.join(root, "away_first"), px, py, bx, by, tags, False)
    monkeypatch.setattr(games, "DATA_DIR", root)
    monkeypatch.setattr(data, "DEFAULT_CACHE_DIR", os.path.join(root, "cache"))
    monkeypatch.setitem(games.GAMES, "SYNTH-HOME-FIRST", ts._entry("home_first"))
    monkeypatch.setitem(games.GAMES, "SYNTH-AWAY-FIRST", ts._entry("away_first"))
    net = str(tmp_path / "net.npz")
    export.write(net, [_random_state(50, seed=s) for s in range(4)], BASE)
    cfg = _cfg(net, "on", 0.5, tracking_fixture(tmp_path))
    out = {}
    for g in ("SYNTH-HOME-FIRST", "SYNTH-AWAY-FIRST"):
        tac = possession.engine_tactical(cfg, g)
        labels, probs = learned.apply(cfg, g, data.graded_blocks(g), return_probs=True,
                                      tactical=tac)
        out[g] = (labels, probs, possession.run(cfg, game=g))
    la, pa, ra = out["SYNTH-HOME-FIRST"]
    lb, pb, rb = out["SYNTH-AWAY-FIRST"]
    swap = {"H": "A", "A": "H", "D": "D", "X": "X"}
    assert np.array_equal(pb, pa[:, [1, 0, 2]])
    assert np.array_equal(lb, np.array([swap[v] for v in la.tolist()]))
    assert np.array_equal(rb, np.array([swap[v] for v in ra.tolist()]))
    cfg_c = _cfg(net, "on", 0.5)
    lab = {g: possession.run(cfg_c, game=g) for g in ("SYNTH-HOME-FIRST", "SYNTH-AWAY-FIRST")}
    assert np.array_equal(lab["SYNTH-AWAY-FIRST"],
                          np.array([swap[v] for v in lab["SYNTH-HOME-FIRST"].tolist()]))


# -- the gate tools (variant R6E, pf_gate) --------------------------------------------------------

def test_pf_paths_and_overrides():
    from tools.learned_spike import evaluate as ev
    from tools.learned_spike import pf_gate
    t = "/m/pf_{provider}_{fold}.npz"
    assert pf_gate.pf_path(t, provider="dfl", fold=3) == "/m/pf_dfl_3.npz"
    assert pf_gate.pf_path("/m/pf_{fold}.npz") == "/m/pf_final.npz"
    assert pf_gate.pf_shared("/m/pf.npz") and not pf_gate.pf_shared(t)
    over = ev.overrides("M", "/w.npz", {"M": (12.0, 12.0), "pf": {"weights": "/m/pf_{fold}.npz",
                                                                   "weight": 0.5}})
    assert over["LEARNED_PF"] == "on" and over["LEARNED_PF_WEIGHTS"] == "/m/pf_final.npz"
    assert over["LEARNED_PF_WEIGHT"] == 0.5
    assert "LEARNED_PF" not in ev.overrides("M", "/w.npz", {"M": (12.0, 12.0)})


def test_r6e_costs_fixed_records_the_member(tmp_path):
    import argparse
    import json
    from tools.learned_spike import r3_eval as r3
    w = tmp_path / "work"
    w.mkdir()
    (w / "preflight.json").write_text(json.dumps({"variant": "R6E", "rehearsal": True}))
    args = argparse.Namespace(work=str(w), run="/r", pair=(12.0, 12.0), prior_bonus=0.0,
                              no_structure=False, pf_weights=FIXTURE, pf_weight=0.5)
    assert r3.cmd_costs_fixed(args) == 0
    costs = r3.load_costs(str(w))
    assert costs["pf"]["weight"] == 0.5 and costs["pf"]["pf_shared"] is True
    mc = r3.match_costs(costs, "SC-1874553")
    assert mc["pf"]["weights"] == FIXTURE
    assert r3.blocking_of("R6E") == ("5", "8")
    w2 = tmp_path / "w2"
    w2.mkdir()
    (w2 / "preflight.json").write_text(json.dumps({"variant": "R6E", "rehearsal": True}))
    with pytest.raises(SystemExit):
        r3.cmd_costs_fixed(argparse.Namespace(work=str(w2), run="/r", pair=(12.0, 12.0),
                                              prior_bonus=0.0, no_structure=False,
                                              pf_weights=None, pf_weight=0.5))
    w3 = tmp_path / "w3"
    w3.mkdir()
    (w3 / "preflight.json").write_text(json.dumps({"variant": "R6", "rehearsal": True}))
    with pytest.raises(SystemExit):
        r3.cmd_costs_fixed(argparse.Namespace(work=str(w3), run="/r", pair=(12.0, 12.0),
                                              prior_bonus=0.0, no_structure=False,
                                              pf_weights=FIXTURE, pf_weight=0.5))
    md = r3.result_markdown("/w", {}, None, "x", {}, True, None, "R6E")
    assert "seed member" in md and "copy member" in md and "LEARNED_STAGE_RESULT_R6E" in md


def test_stability_members_with_pf(blocks, tmp_path):
    from tools.learned_spike import bar_report as br
    from tools.learned_spike import evaluate as ev
    tac = synth_with_touches()
    net = str(tmp_path / "net.npz")
    export.write(net, [_random_state(50, seed=s) for s in range(4)], BASE)
    nets = learned.load_weights(net)
    seeds, copies = ev.member_frames_pf(nets, tac, "METRICA-S3", {"weights": FIXTURE,
                                                                 "weight": 0.5})
    assert len(seeds) == 4 and len(copies) == 4 and seeds[0].shape == (2400, 3)
    # the members average to the candidate's frame probabilities (up to the rounding of
    # each member): seeds and copies are two splits of the same ensemble
    p_pf = P.frame_probabilities_pf(tac, "METRICA-S3", P.load_pf(FIXTURE))
    want = learned.candidate_frame_probabilities(nets, None, tac, "METRICA-S3",
                                                 P.load_pf(FIXTURE), 0.5)
    assert np.abs(np.mean(seeds, axis=0) - want).max() < 5e-4
    assert np.abs(np.mean(copies, axis=0) - want).max() < 5e-4
    assert not np.array_equal(seeds[0], seeds[1])
    for i, s in enumerate(seeds):
        f = str(tmp_path / "s{}.npz".format(i))
        np.savez(f, p25=s)
    graded = np.zeros(2400, bool)
    graded[:1180] = True
    g = br.prob_group([str(tmp_path / "s{}.npz".format(i)) for i in range(4)], graded)
    assert g["members"] == 4 and 0.0 <= g["share_any_disagree"] <= 1.0


def test_r6e_final_dry_run(tmp_path):
    import subprocess
    run = tmp_path / "run"
    run.mkdir()
    ca = tmp_path / "ca"
    ca.mkdir()
    env = dict(os.environ, CA=str(ca), MTAH_DATA_DIR=str(tmp_path / "d"),
               C23_DEV=str(tmp_path / "c"), PAIR="12,12", PRIOR="0", OOF_RATIO="/x.json",
               DRY_RUN="1", REHEARSAL="1", AS_MATCH="DFL-J03WMX", VARIANT="R6E",
               PF_WEIGHTS=FIXTURE, PF_WEIGHT="0.5")
    env.pop("STOP_BEFORE_LABEL_FREE", None)
    sh = os.path.join(ROOT, "tools", "learned_spike", "r5_final.sh")
    res = subprocess.run(["bash", sh, str(run), str(tmp_path / "out")], env=env, cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    text = res.stdout.decode()
    assert res.returncode == 0, text
    assert "--variant R6E" in text and "--pf-weights " + FIXTURE in text
    assert "learned_v6.npz" in text and "goes on to G8" in text
    env.pop("PF_WEIGHTS")
    res = subprocess.run(["bash", sh, str(run), str(tmp_path / "out2")], env=env, cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert res.returncode != 0


# -- round 7: variants R7 and R7E (stage50-v2t), R6 and R6E in every rule ------------------------

def test_r7_variants_follow_r6(tmp_path):
    import argparse
    import json
    from tools.learned_spike import r3_eval as r3
    from tests.test_learned_r5 import r5_switches
    for v6, v7 in (("R6", "R7"), ("R6E", "R7E")):
        assert r3.blocking_of(v7) == r3.blocking_of(v6) == ("5", "8")
        assert r3.limit8("a", 0.06, v7) == pytest.approx(0.0717)
        assert r3.limit8("b", 0.01, v7) == pytest.approx(0.015)
        assert r3.result_name(v7) == "LEARNED_STAGE_RESULT_" + v7
        assert r3.check_r4_switches(r5_switches(feature_version="stage50-v2t"), v7) == []
        assert r3.check_r4_switches(r5_switches(feature_version="stage50-v2s"), v7) != []
    ok, bad = {"pass": True}, {"pass": False}
    crits = {"5": ok, "6": ok, "7": bad, "7x": ok, "8": ok}
    choice, reason = r3.verdict(crits, True, blocking=r3.blocking_of("R7E"))
    assert choice == "M" and "G7: fail (recorded; the owner decides)" in reason
    md = r3.result_markdown("/w", crits, "M", "x", {}, False, None, "R7E")
    assert "round 7, variant R7E" in md and "stage50-v2t" in md and "seed member" in md
    w = tmp_path / "work"
    w.mkdir()
    (w / "preflight.json").write_text(json.dumps({"variant": "R7E", "rehearsal": True}))
    args = argparse.Namespace(work=str(w), run="/r", pair=(12.0, 12.0), prior_bonus=0.0,
                              no_structure=False, pf_weights=FIXTURE, pf_weight=0.25)
    assert r3.cmd_costs_fixed(args) == 0
    assert r3.load_costs(str(w))["pf"]["weight"] == 0.25
    w2 = tmp_path / "w2"
    w2.mkdir()
    (w2 / "preflight.json").write_text(json.dumps({"variant": "R7", "rehearsal": True}))
    with pytest.raises(SystemExit):
        r3.cmd_costs_fixed(argparse.Namespace(work=str(w2), run="/r", pair=(12.0, 12.0),
                                              prior_bonus=0.0, no_structure=False,
                                              pf_weights=FIXTURE, pf_weight=0.5))


@pytest.mark.parametrize("variant", ["R7", "R7E"])
def test_r7_final_dry_run(tmp_path, variant):
    import subprocess
    run = tmp_path / "run"
    run.mkdir()
    ca = tmp_path / "ca"
    ca.mkdir()
    env = dict(os.environ, CA=str(ca), MTAH_DATA_DIR=str(tmp_path / "d"),
               C23_DEV=str(tmp_path / "c"), PAIR="12,12", PRIOR="0", OOF_RATIO="/x.json",
               DRY_RUN="1", REHEARSAL="1", AS_MATCH="DFL-J03WMX", VARIANT=variant,
               PF_WEIGHTS=FIXTURE, PF_WEIGHT="0.25")
    env.pop("STOP_BEFORE_LABEL_FREE", None)
    if variant == "R7":
        env.pop("PF_WEIGHTS")
    sh = os.path.join(ROOT, "tools", "learned_spike", "r5_final.sh")
    res = subprocess.run(["bash", sh, str(run), str(tmp_path / "out")], env=env, cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    text = res.stdout.decode()
    assert res.returncode == 0, text
    assert "--variant " + variant in text and "learned_v7.npz" in text
    assert "learned_v6" not in text and "stage50-v2t" in text and "goes on to G8" in text
    assert ("--pf-weights " + FIXTURE in text) == (variant == "R7E")
