"""Tests of tools/analysis/control_reference.py, the controlled view of a reference."""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from analysis import control_reference as cref  # noqa: E402


def _run(text, min_frames):
    return "".join(cref.control_labels(np.array(list(text), dtype=object), min_frames))


def test_short_run_goes_to_the_team_before_it():
    assert _run("HHHHAAHHHH", 3) == "HHHHHHHHHH"


def test_long_run_is_a_change_of_possession():
    assert _run("HHHHAAAHHH", 3) == "HHHHAAAHHH"


def test_first_run_of_an_alive_stretch_never_changes():
    # The one-frame A run starts the stretch, so it stays. The H run is long enough to follow it.
    assert _run("DDAHHHHDDH", 3) == "DDAHHHHDDH"
    assert _run("DDAHAAADDH", 3) == "DDAAAAADDH"


def test_dead_and_abstained_frames_never_change():
    assert _run("XXHHDAADHHAX", 3) == "XXHHDAADHHHX"


def test_two_short_runs_in_a_row_both_go_back():
    assert _run("HHHHAAHAAHHH", 3) == "HHHHHHHHHHHH"


def test_zero_minimum_is_the_identity():
    assert _run("HAHADHA", 0) == "HAHADHA"


def test_file_round_trip_keeps_header_and_frames(tmp_path):
    src = tmp_path / "ref.csv"
    pd.DataFrame({"frame": range(8), "label": list("HHAHHDDA")}).to_csv(src, index=False)
    dst = cref.write_control_reference(str(src), str(tmp_path / "out.csv"), seconds=0.08)
    out = pd.read_csv(dst)
    assert list(out.columns) == ["frame", "label"]
    assert out["frame"].tolist() == list(range(8))
    assert "".join(out["label"]) == "HHHHHDDA"
