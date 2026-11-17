"""Display labels, scene captions and reduced motion need no match data or browser packages."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from engine.config import Config


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def run_js(script, *args):
    result = subprocess.run(
        [NODE, "--input-type=module", "-e", script, ROOT.as_uri(), *args],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def layout(items, left=0, right=100):
    return run_js("""
        const {edgeLabelRows} = await import(process.argv[1] + '/site/js/util/chart-labels.js');
        const items = JSON.parse(process.argv[2]);
        const before = JSON.stringify(items);
        const rows = edgeLabelRows(items, +process.argv[3], +process.argv[4], 4);
        if (before !== JSON.stringify(items)) throw new Error('input was changed');
        console.log(JSON.stringify(rows));
    """, json.dumps(items), str(left), str(right))


@pytest.mark.parametrize("side", [-1, 1])
def test_crowded_edge_labels_keep_every_value_without_overlap(side):
    items = [{"x": x, "width": width, "side": side, "value": value}
             for x, width, value in [(48, 30, -407), (50, 30, -371),
                                     (52, 42, -1000), (95, 24, -81)]]
    rows = layout(items)
    assert {row["value"] for row in rows} == {item["value"] for item in items}
    assert len(rows) == len(items)
    for index, a in enumerate(rows):
        assert 0 <= a["x"] - a["width"] / 2
        assert a["x"] + a["width"] / 2 <= 100
        for b in rows[index + 1:]:
            if a["row"] == b["row"]:
                assert b["x"] - b["width"] / 2 >= a["x"] + a["width"] / 2 + 4


def test_edge_labels_use_separate_rows_for_each_sign_and_reuse_free_space():
    rows = layout([
        {"x": 85, "width": 20, "side": 1},
        {"x": -10, "width": 20, "side": 1},
        {"x": -10, "width": 20, "side": -1},
    ])
    assert [row["row"] for row in rows] == [0, 0, 0]
    assert [row["x"] for row in rows] == [10, 10, 85]
    assert layout([]) == []


def test_reduced_motion_helpers_finish_counts_and_scroll_without_animation():
    result = run_js("""
        const query = {matches: true};
        globalThis.window = {matchMedia: () => query};
        globalThis.requestAnimationFrame = () => {throw new Error('animated count');};
        const {countUp, dur, scrollToEl} = await import(process.argv[1] + '/site/js/util/dom.js');
        const element = {textContent: ''}, scrolls = [];
        const target = {scrollIntoView: (value) => scrolls.push(value)};
        await countUp(element, 13, {format: String});
        const reducedDuration = dur(900);
        scrollToEl(target);
        query.matches = false;
        scrollToEl(target, 'center');
        console.log(JSON.stringify({text: element.textContent, reducedDuration,
            fullDuration: dur(900), scrolls}));
    """)
    assert result == {"text": "13", "reducedDuration": 0, "fullDuration": 900,
                      "scrolls": [{"behavior": "auto", "block": "start"},
                                  {"behavior": "smooth", "block": "center"}]}


@pytest.mark.parametrize("reduced, expected", [(True, 1), (False, 0.5)])
def test_scroll_step_finishes_immediately_only_with_reduced_motion(reduced, expected):
    result = run_js("""
        const query = {matches: JSON.parse(process.argv[2])};
        globalThis.window = {innerHeight: 1000, scrollY: 0, matchMedia: () => query};
        const section = {id: 'problem', dataset: {}};
        const step = {dataset: {step: '3'}, closest: () => section,
            classList: {add() {}, remove() {}},
            getBoundingClientRect: () => ({top: 250, bottom: 750, height: 500})};
        globalThis.document = {documentElement: {scrollHeight: 2000},
            getElementById: () => null,
            querySelectorAll: (selector) => selector.includes('.step[') ? [step] : []};
        let update;
        globalThis.requestAnimationFrame = (fn) => {update = fn; return 1;};
        globalThis.IntersectionObserver = class {observe() {} disconnect() {}};
        const {state} = await import(process.argv[1] + '/site/js/state.js');
        const {refresh} = await import(process.argv[1] + '/site/js/scroller.js');
        refresh();
        update();
        console.log(JSON.stringify(state.get().activeStep));
    """, json.dumps(reduced))
    assert result == {"section": "problem", "step": 3, "progress": expected}


@pytest.mark.parametrize("lag, expected_onset", [(0, 69), (21, 90)])
def test_out_lag_scene_explains_loss_before_timing_without_moving_the_onset(lag, expected_onset):
    config = Config().as_dict()
    config["DELTA_OUT_LAG"] = lag
    result = run_js("""
        const {createScenes} = await import(process.argv[1] + '/site/js/synthetic.js');
        const scene = createScenes({config: JSON.parse(process.argv[2]),
            pitchM: [105, 68], fps: 25}).get('out-lag');
        const onset = scene.onsets[0].frame;
        const before = scene.frameAt(onset - 1), at = scene.frameAt(onset);
        console.log(JSON.stringify({onset, lastSeen: scene.lastSeen.frame,
            beats: scene.beats, captions: scene.beats.map(b => scene.caption(b.from)),
            before: {stoppage: before.stoppage, label: before.labels.ours},
            at: {stoppage: at.stoppage, label: at.labels.ours,
                checklist: at.insets.find(x => x.type === 'checklist').items,
                overlays: at.overlays},
            end: scene.frameAt(scene.n - 1).insets.find(x => x.type === 'checklist').items,
            marks: scene.timeline.marks.filter(m => m.label === 'onset out')}));
    """, json.dumps(config))
    assert result["onset"] == expected_onset
    assert result["lastSeen"] == 69
    beats = result["beats"]
    assert all(a["from"] < b["from"] for a, b in zip(beats, beats[1:]))
    assert result["captions"] == [beat["text"] for beat in beats]
    assert beats[1]["text"].startswith("Last sighting:")
    assert beats[-1]["text"].startswith("Kept because the players stop within")
    assert result["before"]["stoppage"] is False
    assert result["before"]["label"] != "D"
    assert result["at"]["stoppage"] is True
    assert result["at"]["label"] == "D"
    assert [mark["i"] for mark in result["marks"]] == [expected_onset]
    assert result["at"]["checklist"][1]["ok"] is True
    assert result["at"]["checklist"][2]["ok"] is None
    assert result["end"][2]["ok"] is True
    if lag == 0:
        assert "no added delay" in beats[2]["text"]
        assert "players confirm" in beats[2]["text"]
        assert "no added delay" in result["at"]["checklist"][1]["text"]
        assert not any(overlay.get("text", "").startswith("lag ") for overlay in result["at"]["overlays"])
    else:
        assert "starts 21 frames after the last sighting" in beats[2]["text"]
        assert "last seen + 21 frames" in result["at"]["checklist"][1]["text"]
