"""The gap model CLI in `tools/gap_models/` routes an explicit artefact and refuses a bad one."""

import json

import numpy as np
import pytest

from engine import possession, run
from tools.gap_models import apply as gap_apply
from tools.gap_models import gap_dead, gap_owner


def _quiet_writer(monkeypatch):
    """Replace the CSV writer and the printed checks with stubs."""
    monkeypatch.setattr(gap_apply.submission, "write_submission", lambda labels, path, **kw: path)
    monkeypatch.setattr(gap_apply.submission, "validate", lambda path: "OK")
    monkeypatch.setattr(gap_apply.submission, "summarize", lambda labels, **kw: {})
    monkeypatch.setattr(gap_apply.submission, "format_summary", lambda summary: "summary")
    monkeypatch.setattr(gap_apply.submission, "md5_of", lambda path: "checksum")


def _no_engine(monkeypatch):
    """Make any engine call fail, to show that a refusal happens before detection."""
    def refuse(*args, **kwargs):
        raise AssertionError("the engine ran before the artefact was checked")

    monkeypatch.setattr(possession, "prepare", refuse)
    monkeypatch.setattr(possession, "run", refuse)


@pytest.mark.parametrize("option", ("--gap-owner-model", "--gap-dead-model"))
def test_engine_run_has_no_gap_model_flag(option, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        run.main(["--out", str(tmp_path / "result.csv"), option, "model.json"])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_owner_kind_routes_the_artefact_to_the_owner_correction(monkeypatch, tmp_path):
    artefact = {"owner": True}
    inputs = {"game": "BAR-ATH", "events": "raw events", "tactical": "players"}
    parts = {"events": "decoded events", "stoppage": "stoppage"}
    baseline = np.array(["H", "A"])
    calls = []
    monkeypatch.setattr(gap_owner, "load_artifact", lambda path: artefact)
    monkeypatch.setitem(gap_apply.LOADERS, "owner", gap_owner.load_artifact)
    monkeypatch.setattr(possession, "prepare", lambda cfg, game: inputs)

    def decode(cfg, **kwargs):
        assert kwargs["inputs"] is inputs and kwargs["return_parts"]
        return baseline, parts

    def apply(labels, combined, model, cfg, game):
        assert labels is baseline and model is artefact and game == "BAR-ATH"
        assert combined["raw_events"] == "raw events" and combined["events"] == "decoded events"
        assert combined["tactical"] == "players" and combined["stoppage"] == "stoppage"
        calls.append("owner")
        return labels.copy(), {"enabled": True, "changed_frames": 0, "segments": [1]}

    monkeypatch.setattr(possession, "run", decode)
    monkeypatch.setattr(gap_owner, "apply", apply)
    monkeypatch.setattr(gap_dead, "apply", lambda *args, **kwargs: pytest.fail("dead ran"))
    _quiet_writer(monkeypatch)
    arguments = ["--model", "owner.json", "--kind", "owner", "--out", str(tmp_path / "o.csv")]
    assert gap_apply.main(arguments) == 0
    assert calls == ["owner"]


def test_dead_kind_uses_the_same_prepared_baseline(monkeypatch, tmp_path):
    inputs, parts, artefact = {"game": "BAR-ATH"}, {"part": True}, {"dead": True}
    baseline = np.array(["H", "D", "H"])
    calls = []
    monkeypatch.setattr(gap_dead, "load_artifact", lambda path: artefact)
    monkeypatch.setitem(gap_apply.LOADERS, "dead", gap_dead.load_artifact)
    monkeypatch.setattr(possession, "prepare", lambda cfg, game: inputs)

    def decode(cfg, **kwargs):
        assert kwargs["inputs"] is inputs and kwargs["return_parts"]
        return baseline, parts

    def apply(labels, prepared, decoded, model, cfg, game):
        assert labels is baseline and prepared is inputs and decoded is parts
        assert model is artefact
        calls.append(game)
        return labels.copy(), {}

    monkeypatch.setattr(possession, "run", decode)
    monkeypatch.setattr(gap_dead, "apply", apply)
    monkeypatch.setattr(gap_owner, "apply", lambda *args, **kwargs: pytest.fail("owner ran"))
    _quiet_writer(monkeypatch)
    arguments = ["--model", "dead.json", "--kind", "dead", "--out", str(tmp_path / "d.csv")]
    assert gap_apply.main(arguments) == 0
    assert calls == [run.data.games.GAME]


@pytest.mark.parametrize("kind", gap_apply.KINDS)
def test_missing_artefact_is_refused_before_detection(kind, monkeypatch, tmp_path, capsys):
    _no_engine(monkeypatch)
    with pytest.raises(SystemExit) as error:
        gap_apply.main(["--model", str(tmp_path / "absent.json"), "--kind", kind,
                        "--out", str(tmp_path / "result.csv")])
    assert error.value.code == 2
    assert "cannot use the {} artefact".format(kind) in capsys.readouterr().err
    assert not (tmp_path / "result.csv").exists()


@pytest.mark.parametrize("kind", gap_apply.KINDS)
@pytest.mark.parametrize("text", ("{not json", json.dumps({"format": "unknown"}),
                                  '{"format": NaN}'))
def test_malformed_artefact_is_refused_before_detection(kind, text, monkeypatch, tmp_path,
                                                        capsys):
    _no_engine(monkeypatch)
    model = tmp_path / "model.json"
    model.write_text(text)
    with pytest.raises(SystemExit) as error:
        gap_apply.main(["--model", str(model), "--kind", kind,
                        "--out", str(tmp_path / "result.csv")])
    assert error.value.code == 2
    assert "cannot use the {} artefact".format(kind) in capsys.readouterr().err


def test_kind_is_required(tmp_path):
    with pytest.raises(SystemExit) as error:
        gap_apply.main(["--model", "model.json", "--out", str(tmp_path / "result.csv")])
    assert error.value.code == 2


def test_unknown_kind_is_rejected_by_the_helper():
    with pytest.raises(ValueError, match="kind must be one of"):
        gap_apply.correct("both", {}, inputs={})


def test_audit_summary_drops_interval_coordinates():
    audit = {"enabled": True, "changed_frames": 3, "model_sha256": "a" * 64,
             "segments": [{"start": 1}], "protected_by_reason": {"ghost": 1}}
    assert gap_apply.audit_summary(audit) == {"changed_frames": 3, "enabled": True,
                                              "model_sha256": "a" * 64}
