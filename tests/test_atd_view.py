"""tools/atd_view.py: the ATD-like smoothing view of a bundle, on synthetic data (P9-A4)."""

import json

import numpy as np
import pytest

import games
from engine import data
from tools import atd_view as av

N_FRAMES = 200
HALVES = {1: (1, 100), 2: (101, 200)}
PITCH = (100.0, 50.0)
GAME = "FAKE-VIEW"
REL_META, REL_RAW = "fake/view_atd.xml", "fake/view_atd.txt"

META = """<?xml version="1.0" encoding="utf-8"?>
<main><Metadata>
<GlobalConfig><FrameRate>25</FrameRate></GlobalConfig>
<Teams><Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name></Team></Teams>
<Players><Player id="P1" teamId="T1"/><Player id="P2" teamId="T2"/></Players>
</Metadata></main>
"""


def write_bundle(root, bx, by, px, py):
    """A two-slot bundle in the default channel layout (x, y, s for players and ball)."""
    (root / "fake").mkdir(parents=True, exist_ok=True)
    (root / REL_META).write_text(META)

    def num(v):
        return "NaN" if not np.isfinite(v) else "{:.5f}".format(v)

    lines = []
    for i in range(N_FRAMES):
        players = ";".join("{},{},1.000".format(num(px[i, s]), num(py[i, s])) for s in range(2))
        lines.append("{}:{}:{},{},NaN".format(i + 1, players, num(bx[i]), num(by[i])))
    (root / REL_RAW).write_text("\n".join(lines) + "\n")


@pytest.fixture
def fake_game(monkeypatch):
    entry = dict(games.GAMES["METRICA-S1"], n_video_frames=N_FRAMES, halves=HALVES,
                 offsets={1: 1, 2: 1}, pitch=PITCH,
                 assets={"atd_meta": REL_META, "atd_raw": REL_RAW})
    monkeypatch.setitem(games.GAMES, GAME, entry)
    return GAME


def flat_tracks():
    bx = np.full(N_FRAMES, 0.5)
    by = np.full(N_FRAMES, 0.5)
    px = np.column_stack([np.linspace(0.1, 0.4, N_FRAMES), np.full(N_FRAMES, 0.7)])
    py = np.column_stack([np.full(N_FRAMES, 0.3), np.linspace(0.2, 0.9, N_FRAMES)])
    return bx, by, px, py


def build(tmp_path, game, bx, by, px, py, players=False, sigma=av.ATD_SIGMA_FRAMES):
    write_bundle(tmp_path / "in", bx, by, px, py)
    av.build_view(game, str(tmp_path / "in"), str(tmp_path / "out"), sigma, players)
    raw = data.load_tactical(game, data_dir=str(tmp_path / "in"), use_cache=False)
    view = data.load_tactical(game, data_dir=str(tmp_path / "out"), use_cache=False)
    return raw, view


def test_an_impulse_becomes_the_kernel():
    values = np.zeros(301)
    values[150] = 1.0
    out = av.smooth_runs(values, np.ones(301, bool), np.zeros(301, bool), 9.5)
    lags = np.arange(301) - 150
    expected = np.exp(-0.5 * (lags / 9.5) ** 2) * (np.abs(lags) <= 38)
    expected /= expected.sum()
    assert np.allclose(out, expected, atol=1e-12)
    assert abs(out.sum() - 1.0) < 1e-9


def test_a_constant_run_stays_constant_up_to_its_edges():
    values = np.full(50, 3.25)
    out = av.smooth_runs(values, np.ones(50, bool), np.zeros(50, bool), 9.5)
    assert np.allclose(out, 3.25, atol=1e-12)


def test_a_gap_and_a_break_are_not_bridged():
    values = np.concatenate([np.zeros(60), np.full(60, 10.0)])
    visible = np.ones(120, bool)
    visible[58:61] = False                                  # a gap between the two levels
    out = av.smooth_runs(values, visible, np.zeros(120, bool), 9.5)
    assert np.allclose(out[:58], 0.0) and np.allclose(out[61:], 10.0)
    assert np.array_equal(out[58:61], values[58:61])        # missing frames keep their value
    breaks = np.zeros(120, bool)
    breaks[60] = True                                       # a block edge, no gap
    out = av.smooth_runs(values, np.ones(120, bool), breaks, 9.5)
    assert np.allclose(out[:60], 0.0) and np.allclose(out[60:], 10.0)


def test_a_one_frame_jump_is_smoothed_like_any_displacement():
    values = np.concatenate([np.zeros(100), np.full(100, 6.0)])
    out = av.smooth_runs(values, np.ones(200, bool), np.zeros(200, bool), 9.5)
    assert 2.5 < out[99] < 3.0 and 3.0 < out[100] < 3.5
    assert np.max(np.abs(np.diff(out))) < 0.3


def test_the_view_smooths_the_ball_in_metres_and_keeps_the_rest(tmp_path, fake_game):
    bx, by, px, py = flat_tracks()
    bx[40] = 0.6                                            # a 10 m one-frame spike in x
    bx[120:130] = np.nan                                    # a gap in half 2
    raw, view = build(tmp_path, fake_game, bx, by, px, py)
    kernel = av.gaussian_kernel(9.5)
    assert abs(view["bx"][40] - (0.5 + 0.1 / kernel.sum())) < 2e-5
    assert np.allclose(view["by"], raw["by"], equal_nan=True, atol=1e-5)
    assert np.array_equal(np.isnan(view["bx"]), np.isnan(raw["bx"]))
    assert np.allclose(view["bx"][100:120], 0.5, atol=1e-6)  # the spike stays in half 1
    assert np.allclose(view["bx"][130:], 0.5, atol=1e-6)
    for key in ("px", "py", "ps"):
        assert np.array_equal(view[key], raw[key])
    assert view["team_names"] == raw["team_names"]
    assert (tmp_path / "out" / REL_META).read_bytes() == (tmp_path / "in" / REL_META).read_bytes()
    info = json.loads((tmp_path / "out" / (REL_RAW + av.VIEW_SUFFIX)).read_text())
    assert info["sigma_frames"] == 9.5 and info["players"] is False
    # The ball speed channel keeps its text.
    first = (tmp_path / "out" / REL_RAW).read_text().splitlines()[0]
    assert first.split(":")[2].split(",")[2] == "NaN"


def test_players_are_untouched_when_off_and_smoothed_when_on(tmp_path, fake_game):
    bx, by, px, py = flat_tracks()
    px[70, 0] = px[70, 0] + 0.05                            # a 5 m spike on slot 0
    py[30:35, 1] = np.nan
    raw, view = build(tmp_path, fake_game, bx, by, px, py, players=True)
    assert view["px"][70, 0] < raw["px"][70, 0] - 0.04
    assert np.array_equal(np.isnan(view["py"]), np.isnan(raw["py"]))
    assert np.array_equal(view["ps"], raw["ps"])
    # A straight line at constant speed is unchanged more than one kernel radius from a run end.
    assert np.allclose(view["py"][140:160, 1], raw["py"][140:160, 1], atol=2e-5)


def test_the_view_is_deterministic(tmp_path, fake_game):
    bx, by, px, py = flat_tracks()
    bx[10:20] = np.linspace(0.5, 0.7, 10)
    write_bundle(tmp_path / "in", bx, by, px, py)
    av.build_view(fake_game, str(tmp_path / "in"), str(tmp_path / "a"))
    av.build_view(fake_game, str(tmp_path / "in"), str(tmp_path / "b"))
    for rel in (REL_RAW, REL_META, REL_RAW + av.VIEW_SUFFIX):
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()


def test_sealed_matches_are_refused_before_reading_assets(tmp_path, monkeypatch):
    def forbidden_read(*args, **kwargs):
        pytest.fail("A sealed match must be refused before the tool reads its assets.")

    monkeypatch.setattr(av.data, "asset_paths", forbidden_read)
    for game in av.SEALED_GAMES:
        assert game in games.GAMES
        with pytest.raises(SystemExit, match="Pass --allow-sealed"):
            av.build_view(game, str(tmp_path), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("game", av.SEALED_GAMES)
def test_sealed_override_builds_only_the_supplied_synthetic_bundle(
        tmp_path, monkeypatch, fake_game, game):
    monkeypatch.setitem(games.GAMES, game, games.GAMES[fake_game])
    write_bundle(tmp_path / "in", *flat_tracks())
    summary = av.build_view(game, str(tmp_path / "in"), str(tmp_path / "out"),
                            sigma=3, allow_sealed=True)
    manifest = json.loads((tmp_path / "out" / (REL_RAW + av.VIEW_SUFFIX)).read_text())
    assert manifest == summary
    assert manifest["game"] == game
    assert manifest["sealed_opening"] is True
    assert manifest["sigma_frames"] == 3
    assert (tmp_path / "out" / REL_META).read_bytes() == (tmp_path / "in" / REL_META).read_bytes()


def test_override_does_not_change_an_open_view(tmp_path, fake_game):
    write_bundle(tmp_path / "in", *flat_tracks())
    av.build_view(fake_game, str(tmp_path / "in"), str(tmp_path / "default"))
    av.build_view(fake_game, str(tmp_path / "in"), str(tmp_path / "allowed"),
                  allow_sealed=True)
    for rel in (REL_RAW, REL_META, REL_RAW + av.VIEW_SUFFIX):
        assert (tmp_path / "default" / rel).read_bytes() == (tmp_path / "allowed" / rel).read_bytes()


@pytest.mark.parametrize("allow_sealed", [False, True])
def test_cli_passes_the_explicit_sealed_flag(tmp_path, monkeypatch, capsys, allow_sealed):
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {"test_only": True}

    monkeypatch.setattr(av, "build_view", capture)
    args = ["--game", av.SEALED_GAMES[0], "--in-root", str(tmp_path / "in"),
            "--out-root", str(tmp_path / "out"), "--sigma", "3", "--players", "off"]
    if allow_sealed:
        args.append("--allow-sealed")
    av.main(args)
    assert calls == [((av.SEALED_GAMES[0], str(tmp_path / "in"), str(tmp_path / "out"),
                       3.0, False), {"allow_sealed": allow_sealed})]
    assert json.loads(capsys.readouterr().out) == {"test_only": True}
