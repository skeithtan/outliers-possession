"""P8-R1 restart timing on synthetic speed ramps. No match data or reference labels."""

import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.config import Config
from tools import p8_restart as timing


def inputs(speed, frames, blocks=None, visible=None, run_id=None, kinds=None, fps=25.0):
    """Build the estimator's inputs from synthetic speeds and candidate frames."""
    speed = np.asarray(speed, dtype=np.float64)
    n_frames = speed.size
    return {
        "ball": {"speed": speed, "visible": (np.ones(n_frames, bool) if visible is None
                                             else np.asarray(visible, bool)),
                 "run_id": (np.zeros(n_frames, int) if run_id is None
                            else np.asarray(run_id)), "fps": fps},
        "blocks": [(1, 0, n_frames - 1)] if blocks is None else blocks,
        "restarts": pd.DataFrame({"frame": frames,
                                   "kind": ["kick"] * len(frames) if kinds is None else kinds}),
    }


def ramp(center=40, width=3.0, n_frames=120):
    """A smooth speed step with one symmetric acceleration peak."""
    return 20.0 / (1.0 + np.exp(-(np.arange(n_frames) - center) / width))


def test_symmetric_smoothed_step_recovers_bump_center():
    estimate = timing.estimate_restart_timing(inputs(ramp(), [28]))
    assert estimate["delay_frames"] == 12
    assert estimate["eligible_kicks"] == 1
    assert not estimate["fallback"]


def test_earliest_impulse_tie_wins():
    speed = np.zeros(40)
    speed[11:14] = [1, 2, 3]
    speed[14:] = 3
    cfg = dataclasses.replace(Config.default(), CONTACT_HALF_WINDOW=1)
    estimate = timing.estimate_restart_timing(inputs(speed, [10]), cfg)
    assert estimate["delay_frames"] == 1


@pytest.mark.parametrize("offsets", [(1, 2), (2, 3)])
def test_even_sample_median_uses_ties_to_even(offsets):
    speed = ramp(center=40)
    speed = np.concatenate([speed, ramp(center=40)])
    cfg = dataclasses.replace(Config.default(), CONTACT_HALF_WINDOW=1)
    frames = [40 - offsets[0], 160 - offsets[1]]
    report = timing.estimate_restart_timing(inputs(speed, frames), cfg)
    assert report["delay_frames"] == 2
    assert report["offset_percentiles_frames"]["p50"] == sum(offsets) / 2


def test_duplicate_candidates_do_not_weight_the_median():
    single = inputs(ramp(), [28])
    duplicate = inputs(ramp(), [28, 28, 28])
    before = duplicate["restarts"].copy(deep=True)
    result = timing.estimate_restart_timing(duplicate)
    assert result["delay_frames"] == timing.estimate_restart_timing(single)["delay_frames"]
    assert result["eligible_kicks"] == 1
    assert result["duplicate_kick_rows"] == 2
    pd.testing.assert_frame_equal(before, duplicate["restarts"])


def test_unconfirmed_candidates_are_included_and_reappearances_are_excluded():
    source = inputs(ramp(), [28, 35], kinds=["kick", "reappear"])
    source["restarts"]["confirmed"] = False
    result = timing.estimate_restart_timing(source)
    assert result["eligible_kicks"] == result["kick_rows"] == 1
    assert result["delay_frames"] == 12


@pytest.mark.parametrize("invalid", ["gap", "run_id", "speed", "block"])
def test_impulse_does_not_bridge_missing_support_or_block_edges(invalid):
    source = inputs(ramp(), [28])
    if invalid == "gap":
        source["ball"]["visible"][31] = False
    elif invalid == "run_id":
        source["ball"]["run_id"][31:] = 1
    elif invalid == "speed":
        source["ball"]["speed"][31] = np.nan
    else:
        source["blocks"] = [(1, 0, 30), (2, 31, 119)]
    cfg = dataclasses.replace(Config.default(), DELTA_RESTART_KICK=7)
    result = timing.estimate_restart_timing(source, cfg)
    assert result["delay_frames"] == 7
    assert result["fallback"]
    assert result["rejected"]["no_impulse_support"] == 1


def test_noncontiguous_reused_run_id_is_not_a_bridge():
    source = inputs(ramp(), [28])
    source["ball"]["run_id"][31] = -1
    result = timing.estimate_restart_timing(source)
    assert result["fallback"]


def test_ungraded_and_invalid_candidates_are_reported():
    source = inputs(ramp(), [0, 20, 28], blocks=[(1, 10, 100)])
    source["ball"]["visible"][20] = False
    result = timing.estimate_restart_timing(source)
    assert result["rejected"]["outside_graded"] == 1
    assert result["rejected"]["invalid_ball"] == 1
    assert result["eligible_kicks"] == 1


def test_empty_samples_keep_legacy_config_and_expose_no_percentiles():
    cfg = dataclasses.replace(Config.default(), DELTA_RESTART_KICK=9)
    candidate, report = timing.candidate_config(inputs([], []), cfg)
    assert candidate == cfg
    assert report["fallback"]
    assert report["offset_percentiles_frames"] is None


def test_only_restart_delay_changes_in_candidate_config():
    cfg = dataclasses.replace(Config.default(), DELTA_RESTART_KICK=3)
    candidate, report = timing.candidate_config(inputs(ramp(), [28]), cfg)
    assert candidate.DELTA_RESTART_KICK == 12
    assert dataclasses.replace(candidate, DELTA_RESTART_KICK=3) == cfg
    assert report["baseline_delay_frames"] == 3


def test_peak_end_uses_first_qualifying_peak_and_bounded_fallback():
    assert timing._peak_end(np.array([0, 0.4, 0.3, 3, 2, 5]), 0, 5) == 3
    assert timing._peak_end(np.array([0, 1, 2, 3, 4]), 0, 3) == 3
    assert timing._peak_end(np.array([2, 2, 2]), 0, 2) == 0


def test_horizon_uses_fps_and_does_not_search_later_rises():
    speed = np.zeros(100)
    speed[15:] = np.linspace(0, 50, 85)
    cfg = dataclasses.replace(Config.default(), CONTACT_HALF_WINDOW=1)
    short = timing.estimate_restart_timing(inputs(speed, [5], fps=1.0), cfg)
    assert short["delay_frames"] == 0


@pytest.mark.parametrize("game", timing.SEALED_GAMES)
def test_runner_refuses_sealed_matches_before_loading_data(game, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("sealed match reached the data loader")
    monkeypatch.setattr(timing.possession, "prepare", forbidden)
    with pytest.raises(ValueError, match="sealed"):
        timing.run_comparison(game, tmp_path)


def test_runner_refuses_repository_output():
    with pytest.raises(ValueError, match="outside the repository"):
        timing._external_output(timing.ROOT / "site" / "data" / "public")


def test_estimator_is_team_symmetric_and_input_order_invariant():
    source = inputs(ramp(), [28, 30])
    source["restarts"]["team"] = ["H", "A"]
    swapped = dict(source, restarts=source["restarts"].iloc[::-1].copy())
    swapped["restarts"]["team"] = ["H", "A"]
    assert timing.estimate_restart_timing(source) == timing.estimate_restart_timing(swapped)


@pytest.mark.parametrize("game", ["BAR-ATH", "METRICA-S3"])
def test_cli_data_root_reaches_imported_asset_resolver(game, tmp_path, monkeypatch):
    old_root = str(tmp_path / "original")
    requested = tmp_path / "requested"
    cache = tmp_path / "cache"
    monkeypatch.setattr(timing.data.games, "DATA_DIR", old_root)
    monkeypatch.setenv("MTAH_DATA_DIR", old_root)
    previous_cache = timing.data.DEFAULT_CACHE_DIR
    observed = {}

    def capture(match, out_dir, cfg, estimate_only):
        observed["paths"] = timing.data.asset_paths(match)
        observed["env"] = os.environ["MTAH_DATA_DIR"]
        observed["cache"] = timing.data.DEFAULT_CACHE_DIR
        return {"game": match}

    monkeypatch.setattr(timing, "run_comparison", capture)
    assert timing.main(["--game", game, "--data-root", str(requested),
                        "--cache-dir", str(cache), "--out-dir", str(tmp_path / "out")]) == 0
    assert observed["paths"]
    for path in observed["paths"].values():
        assert Path(path).relative_to(requested)
    assert observed["env"] == str(requested)
    assert observed["cache"] == str(cache)
    assert timing.data.games.DATA_DIR == old_root
    assert os.environ["MTAH_DATA_DIR"] == old_root
    assert timing.data.DEFAULT_CACHE_DIR == previous_cache


def test_cli_refuses_a_data_root_without_a_cache_dir(tmp_path, monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("the comparison ran without a separate cache directory")

    monkeypatch.setattr(timing, "run_comparison", refuse)
    with pytest.raises(SystemExit) as error:
        timing.main(["--game", "METRICA-S3", "--data-root", str(tmp_path / "data"),
                     "--out-dir", str(tmp_path / "out")])
    assert error.value.code == 2
    assert "--data-root requires --cache-dir" in capsys.readouterr().err


def test_cli_restores_input_roots_after_failure(tmp_path, monkeypatch):
    old_root = timing.data.games.DATA_DIR
    old_cache = timing.data.DEFAULT_CACHE_DIR
    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate load failure")

    monkeypatch.setattr(timing, "run_comparison", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        timing.main(["--game", "METRICA-S3", "--data-root", str(tmp_path / "data"),
                     "--cache-dir", str(tmp_path / "cache"), "--out-dir", str(tmp_path)])
    assert "MTAH_DATA_DIR" not in os.environ
    assert timing.data.games.DATA_DIR == old_root
    assert timing.data.DEFAULT_CACHE_DIR == old_cache


def test_full_config_snapshots_preserve_base_overrides(tmp_path, monkeypatch):
    cfg = dataclasses.replace(Config.default(), DELTA_RESTART_KICK=4, DV_CONTACT=7.5,
                              FLIGHT_ASSIGNMENT=True, OWNER_RELEASE_KINDS=("release",))
    monkeypatch.setattr(timing.possession, "prepare", lambda *args, **kwargs: inputs(ramp(), [28]))
    report = timing.run_comparison("METRICA-S3", tmp_path, cfg, estimate_only=True)
    baseline = timing.config.load(str(tmp_path / "baseline_config.json"))
    candidate = timing.config.load(str(tmp_path / "candidate_config.json"))
    assert baseline == cfg
    assert candidate == dataclasses.replace(cfg, DELTA_RESTART_KICK=12)
    assert baseline.config_hash() == report["baseline_config_hash"]
    assert candidate.config_hash() == report["candidate_config_hash"]
