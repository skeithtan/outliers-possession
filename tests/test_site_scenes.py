"""The JavaScript port of the C15 shape clause and the C18 reach clause in site/js/synthetic.js.

The tests run the site module in Node and compare it with the engine on the synthetic redirect
and reach scenes. They skip if Node is not installed. They need no match data. The scene builds
from the committed `site/data/public/config.json`, which an older bundle may lack the C18
constants in. The port then reads the engine defaults.
"""

import dataclasses
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from engine import events, tracks
from engine.config import Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")
PITCH = (105.0, 68.0)
FPS = 25
N_SLOTS = 46        # NS in site/js/synthetic.js
C15_FIELDS = ["REDIRECT_SHAPE_CONFIRM", "SHAPE_LAG_FRAMES", "SHAPE_CONFIRM_MS",
              "SHAPE_HALF_FRAMES", "SHAPE_MIN_PLAYERS"]
C18_FIELDS = ["REACH_SHAPE_CONFIRM", "R_REACH_M"]
SCENES = ["redirect", "redirect-flat", "redirect-reach", "redirect-reach-flat"]

SCRIPT = """
import fs from 'fs';
import { pathToFileURL } from 'url';
const root = process.argv[1];
const { createScenes } = await import(pathToFileURL(root + '/site/js/synthetic.js').href);
const config = JSON.parse(fs.readFileSync(root + '/site/data/public/config.json', 'utf8'));
const drop = JSON.parse(process.argv[2]);
const set = JSON.parse(process.argv[3]);
for (const sec of config.sections) {
  sec.fields = sec.fields.filter((f) => drop.indexOf(f.name) < 0);
  for (const f of sec.fields) if (f.name in set) f.value = set[f.name];
}
// a constant that the committed bundle lacks (an older bundle) is added by the override
const have = new Set(config.sections.flatMap((sec) => sec.fields.map((f) => f.name)));
for (const name of Object.keys(set)) if (!have.has(name)) config.sections[0].fields.push({ name, value: set[name] });
const lib = createScenes({ config, pitchM: [105, 68], fps: 25 });
const out = {};
const arr = (a) => Array.from(a);
for (const id of ['redirect', 'redirect-flat', 'redirect-reach', 'redirect-reach-flat']) {
  const s = lib.get(id);
  const b = s.ball, f = s.f;
  out[id] = { n: s.n, events: s.events.map((e) => [e.frame, e.kind, e.team, e.note || '']),
    info: s.redirectInfo, px: arr(s.px), py: arr(s.py), team: arr(s.team), alive: arr(s.alive),
    shape: s.shape ? { H: arr(s.shape.H), A: arr(s.shape.A) } : null,
    contacts: s.contacts.map((c) => [c.frame, c.kind, c.team, c.reach ? 1 : 0, c.d1]),
    ball: { vx: arr(b.vx), vy: arr(b.vy), speed: arr(b.speed), runId: arr(b.runId), vis: arr(b.vis) },
    f: { dH: arr(f.dH), dA: arr(f.dA), nearH: arr(f.nearH), nearA: arr(f.nearA), relH: arr(f.relH), relA: arr(f.relA) } };
}
console.log(JSON.stringify(out));
"""


def build(drop=(), **overrides):
    """The four redirect and reach scenes of the site, built in Node with a changed bundle config."""
    result = subprocess.run(
        [NODE, "--input-type=module", "-e", SCRIPT, ROOT, json.dumps(list(drop)),
         json.dumps(overrides)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def tactical_of(scene):
    """The tactical dict of tracks.team_forward_velocity for one scene (one block)."""
    n = scene["n"]
    px = array(scene["px"]).reshape(n, N_SLOTS)
    py = array(scene["py"]).reshape(n, N_SLOTS)
    team = np.array(scene["team"])
    cols = np.flatnonzero(team != 255)
    return {"px": px[:, cols] / PITCH[0], "py": py[:, cols] / PITCH[1],
            "team_of_slot": team.astype(np.int8)[cols],
            "ps": np.full((n, cols.size), np.nan), "fps": FPS}


def engine_rows(scene, cfg):
    """Sorted (frame, kind, team) rows of events.detect_events on the exported scene."""
    n = scene["n"]
    shape = tracks.team_forward_velocity(tactical_of(scene), [(1, 0, n - 1)], PITCH, cfg, fps=FPS)
    b, f = scene["ball"], scene["f"]
    ball = {"visible": np.array(b["vis"], dtype=bool), "speed": array(b["speed"]),
            "vx": array(b["vx"]), "vy": array(b["vy"]), "run_id": np.array(b["runId"], dtype=np.int64)}
    feats = {"dH": array(f["dH"]), "dA": array(f["dA"]),
             "nearest_slot_H": np.array(f["nearH"], dtype=np.int64),
             "nearest_slot_A": np.array(f["nearA"], dtype=np.int64),
             "relH": array(f["relH"]), "relA": array(f["relA"])}
    e_dead = 1.0 - np.array(scene["alive"], dtype=np.float64)
    table = events.detect_events(ball, feats, None, None, cfg, e_dead=e_dead, shape=shape)
    return sorted((int(r.frame), r.kind, r.team) for r in table.itertuples())


def port_rows(scene):
    return sorted((e[0], e[1], e[2]) for e in scene["events"])


def redirect_rows(scene):
    return [e for e in scene["events"] if e[3] == events.REDIRECT_NOTE]


def array(values):
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64)


@needs_node
def test_shape_response_confirms_a_redirect_with_no_next_contact():
    scene = build()["redirect"]
    rows = redirect_rows(scene)
    assert [(r[1], r[2]) for r in rows] == [("control", "A")]
    assert scene["info"]["response"] >= Config.default().SHAPE_CONFIRM_MS


@needs_node
def test_a_flat_shape_leaves_the_redirect_without_a_row():
    scene = build()["redirect-flat"]
    assert redirect_rows(scene) == []
    assert scene["info"]["response"] < Config.default().SHAPE_CONFIRM_MS


@needs_node
def test_redirect_shape_confirm_false_gives_the_c8_rule():
    both = build(REDIRECT_SHAPE_CONFIRM=False)
    assert redirect_rows(both["redirect"]) == [] and redirect_rows(both["redirect-flat"]) == []
    assert both["redirect"]["shape"] is None


@needs_node
def test_a_bundle_without_the_c15_and_c18_fields_uses_the_engine_defaults():
    assert build(drop=C15_FIELDS + C18_FIELDS) == build()


@needs_node
def test_forward_velocity_and_response_match_the_engine():
    cfg = Config.default()
    for scene in build().values():
        n = scene["n"]
        shape = tracks.team_forward_velocity(tactical_of(scene), [(1, 0, n - 1)], PITCH, cfg,
                                             fps=FPS)
        for key in ("H", "A"):
            ours = array(scene["shape"][key])
            assert np.array_equal(np.isnan(ours), np.isnan(shape[key]))
            assert np.nanmax(np.abs(ours - shape[key])) < 1e-9
        info = scene["info"]
        value = events.shape_response(shape, info["frame"], "A", cfg.SHAPE_LAG_FRAMES)
        assert abs(value - info["response"]) < 1e-9


@needs_node
def test_reach_contact_in_the_reach_scene_only_with_a_shape_response():
    scenes = build()
    cfg = Config.default()
    reach, flat = scenes["redirect-reach"], scenes["redirect-reach-flat"]
    assert [(r[1], r[2]) for r in redirect_rows(reach)] == [("control", "A")]
    assert redirect_rows(flat) == []
    for scene in (reach, flat):
        (contact,) = [c for c in scene["contacts"] if c[0] == scene["info"]["frame"]]
        assert cfg.R_TOUCH < contact[4] <= cfg.R_REACH_M
    assert [c[3] for c in reach["contacts"]] == [1] and [c[3] for c in flat["contacts"]] == [0]
    assert reach["info"]["response"] >= cfg.SHAPE_CONFIRM_MS > flat["info"]["response"]


@needs_node
def test_reach_shape_confirm_false_drops_the_reach_contact_and_keeps_the_other_scenes():
    base, off = build(), build(REACH_SHAPE_CONFIRM=False)
    assert redirect_rows(off["redirect-reach"]) == []
    assert [c[1] for c in off["redirect-reach"]["contacts"]] == ["none"]
    for name in ("redirect", "redirect-flat", "redirect-reach-flat"):
        assert off[name]["events"] == base[name]["events"]


@needs_node
@pytest.mark.parametrize("over", [{}, {"REACH_SHAPE_CONFIRM": False},
                                  {"REDIRECT_SHAPE_CONFIRM": False}, {"R_REACH_M": 4.0},
                                  {"SHAPE_CONFIRM_MS": 3.5}])
def test_event_rows_of_the_scenes_equal_the_engine_rows(over):
    cfg = dataclasses.replace(Config.default(), **over)
    for name, scene in build(**over).items():
        assert port_rows(scene) == engine_rows(scene, cfg), name
