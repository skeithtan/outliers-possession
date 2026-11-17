# Visualizer data schema (the hand-off contract)

This file is the contract between the exporters in `tools/viz/` and the site in `site/`. The
exporters write the two bundles below. The site reads them. Change this file first, then the code.
It is a working contract and is not bound by the sentence limits of our writing guide.

## 0. Which bundle is committed and which is not

| bundle | path | built by | git |
|---|---|---|---|
| public | `site/data/public/` | `python -m tools.viz.build_site` | committed, below 500 KB |
| local | `site/data/local/` | `python -m tools.viz.build_replay` | gitignored, never committed, about 45 MB |

Two organiser rules decide the split. Both come from the data licence and bind every public file.

1. The organisers' `RULES.md`: "Do not publish derived data that would reconstruct it. Frame-level
   tracking exports, clipped video, and similar." The public bundle holds no positions, no
   velocities, no distances and no per-frame kinematic series.
2. `evaluation/results/README.md`: "A submission is a complete set of labels for the match, and a
   correct one is a copy of the ground truth. Nothing that belongs in this folder belongs in a
   public repository". A complete label set does not belong in a public repository. The public
   bundle holds no full-match labels of ours and nothing that rebuilds them (full-match stoppage
   runs, owner events, per-frame owner).

The public bundle therefore holds frame-level detail only inside the public window
(`11185..18684`, half-open `[11185, 18685)`), per-minute aggregates for the whole match, and the
full-match baseline labels (the organisers publish those files themselves). The local bundle holds
the real tracking replay and our full-match labels. It lives on one laptop. Teammates rebuild it
with one command. `tools/viz/build_replay.py` refuses to write it to a path that git would commit.

## 1. Conventions

- A frame is a video frame. Row `i` of any per-frame array is video frame `i`.
  `video_frame = atd_frame_id - 1`.
- An interval is half-open `[start, stop)`. Text that names graded blocks as `11185..90651` uses
  inclusive ends. The JSON always uses half-open ends: `[11185, 90652)` and `[110808, 190198)`.
- Label letters in JSON are `H`, `A`, `D`, `X`. Label codes in binary files and in decoded arrays
  are `H=0`, `A=1`, `D=2`, `X=3`.
- Time is in seconds, `fps` is 25. Lengths are in metres in the local bundle. Pitch size comes
  from `engine.data.pitch_m(game)`, which reads `games.GAMES["BAR-ATH"]["pitch"]`, and is written
  to the manifest as `pitch_m`. For BAR-ATH it is `[105.3, 68.0]`. The 105.0 of `games.py` is
  the Metrica entry, not this match.
- Normalised 0..1 coordinates are never used in these bundles.
- Binary files are little-endian. JSON files have sorted keys, compact separators
  (`separators=(",", ":")`), UTF-8, and one trailing newline. Rebuilding on the same commit gives
  byte-identical files.
- No wall clock is written. Dates come from `git log -1 --format=%cI`.
- A confidence is a float in `[0, 1]` rounded to 3 decimals.
- A team is the letter `H` or `A`, or `null` when unknown.
- `null` means "not applicable or unknown". A missing key is a schema error.
- RLE `runs` (used by every label file) are lists `[start, stop, letter]` sorted by `start`, with
  `stop` of one run equal to or below `start` of the next. A frame in no run is `X`. Adjacent runs
  never share a letter.

## 2. Public bundle: `site/data/public/`

Every file listed in `manifest.json` `files` is fetched by `data.js` `loadPublic()`.

### 2.1 `manifest.json`

| key | type | meaning |
|---|---|---|
| `schema` | int | schema version, `1` |
| `version` | string | name of the current best version, for example `c8_bump` |
| `md5` | string | md5 of the results file of `version`, 32 hex characters |
| `config_hash` | string | hash of `Config.default().as_dict()`, 10 hex characters |
| `git` | object | `{commit: string, commit_date: string (ISO 8601 from git), tag: string or null, dirty: bool}`, see below |
| `game` | string | `BAR-ATH` |
| `n_frames` | int | 199319 |
| `fps` | int | 25 |
| `pitch_m` | `[number, number]` | pitch length and width in metres, from `engine.data.pitch_m(game)`: `[105.3, 68.0]` |
| `team_names` | `{H: string, A: string}` | display names of the two teams |
| `graded_blocks` | `[[int, int], [int, int]]` | the two half-open graded blocks |
| `kickoffs` | `[{half: 1\|2, frame: int}]` | kickoff frames |
| `public_window` | `{start: int, stop: int}` | `11185`, `18685` |
| `exempt` | `[string]` | file name patterns that the window rule does not cover (Section 5) |
| `team` | `[string]` | five team member names from the README "## Team" list |
| `repro` | object | `{command: string, python: string, pins: {package: version}, n_tests: int}` |
| `files` | `{string: int}` | file name to size in bytes, for every public file except `manifest.json` |

The guard reads `public_window`, `exempt` and `n_frames` from this file.

`git.commit` is the newest commit that changed a file outside `site/data/public/` and
`site/data/local/` (`git log -1 -- . ':(exclude)site/data/public' ...`). A commit that adds only
a rebuilt bundle does not move it, so a rebuild after that commit gives the same bytes.
`git.dirty` is true when `git status` shows a change outside the two bundle folders.
`git.tag` is the first tag, in sorted order, that points at `git.commit`, or `null`.
`repro.pins` holds every `name==version` line of `requirements.txt`, `requirements-dev.txt` and
`requirements-starter.txt`. `repro.n_tests` counts the `def test_` functions in
`tests/test_*.py`. A parametrised test counts once, so pytest reports a larger number.

### 2.2 `labels_window.json`

Frame-level labels inside the public window only. Every run has `start >= 11185` and
`stop <= 18685`.

| key | type | meaning |
|---|---|---|
| `window` | `{start: int, stop: int}` | same as the manifest |
| `ours` | RLE runs | our labels for `version` |
| `gt_public` | RLE runs | the public ground truth for the window |
| `baseline_smooth50` | RLE runs | baseline label file `smooth50`, clipped |
| `baseline_smooth125` | RLE runs | baseline label file `smooth125`, clipped |

### 2.3 `labels_baseline_smooth50.json` and `labels_baseline_smooth125.json`

Full-match labels of the two organiser baselines. They are already public.

| key | type | meaning |
|---|---|---|
| `name` | string | `baseline_smooth50` or `baseline_smooth125` |
| `n_frames` | int | 199319 |
| `runs` | RLE runs | covers `[0, n_frames)`, `X` outside the graded blocks |

### 2.4 `minute_bins.json`

Per-minute aggregates for the whole match. See Section 4 for the bin definition.

| key | type | meaning |
|---|---|---|
| `bin_frames` | int | 1500 |
| `n_bins` | int | 106 |
| `ours` | list of bin rows | one row per bin, in frame order |
| `baseline_smooth50` | list of bin rows | same bins for the baseline |

A bin row has these fields:

| field | type | meaning |
|---|---|---|
| `start`, `stop` | int | half-open frame range of the bin |
| `half` | 1 or 2 | half of the bin |
| `share_h`, `share_a`, `share_d`, `share_x` | float | fraction of the bin's frames with that label, 4 decimals, sum is 1 within rounding |
| `n_possession_starts` | int | `H` or `A` spells that start inside the bin |
| `n_d_segments` | int | `D` segments that start inside the bin |
| `events_by_kind` | `{kind: int}` | count of events with frame inside the bin |
| `onsets_by_mechanism` | `{mechanism: int}` | count of dead-ball onsets with frame inside the bin |
| `restarts_by_kind` | `{kind: int}` | count of restarts with frame inside the bin |

The three count objects are `{}` for the baseline rows, because the baseline has no events.
Kind and mechanism names are the strings that the engine writes: event kinds `release`,
`control`, `dribble`, `touch`, `contested`; onset mechanisms `out`, `pursuit_stop`,
`collective_stop`; restart kinds `kick`, `reappear`. A count object holds only the names that
occur in the bin. `restarts_by_kind` counts confirmed restart candidates only. The counts use the
same frames as Section 2.5.

### 2.5 `window_detail.json`

Detector output inside the window only. Every frame lies in `[11185, 18685)`.

| key | row shape | meaning |
|---|---|---|
| `events` | `[frame, kind, team, confidence]` | ball contacts and touches |
| `owner_events` | `[frame, team, kind, confidence]` | events that change the owner |
| `onsets` | `[frame, mechanism, confidence]` | dead-ball onsets |
| `restarts` | `[frame, team, kind, confidence, confirmed]` | restarts, `confirmed` is a bool |
| `stoppages` | `[start, stop]` | half-open stoppage runs |

`frame` is int, `kind` and `mechanism` are strings, `team` is `H`, `A` or `null`.

The frames are the frames that the decoder uses, from `possession.run(return_parts=True)`: the
event table after `DELTA_SWITCH`, the onsets after `DELTA_DEAD`, the restart candidates after
`DELTA_RESTART_KICK` or `DELTA_RESTART_REAPPEAR` (values in `config.json`), and the stoppage mask
after `shift_stoppage`. With `c8_bump` the event and onset offsets are 0. A kick restart sits 12
frames after the start of its speed rise, a reappear restart 10 frames after the reappearance.
`restarts` holds every candidate, confirmed or not. `owner_events` are the `(frame, team)` rows
of `parts["owner_events"]`, with `kind` and `confidence` taken from the event row at that frame.
`kind` and `confidence` are `null` when no event row matches.

### 2.6 `scores.json`

A map from version name to a whitelist of keys from `evaluation/reports/<name>.json`. It holds
one entry for each ladder version that has a report. Keys per entry:

| key | type | meaning |
|---|---|---|
| `spell_f1_mean` | float | the headline `spellF1` |
| `macro_f1` | float | macro F1 over the frame classes |
| `spell_f1` | `{iou: {tp, fp, fn, f1}}` | per IoU threshold, with counts |
| `transitions` | `{tp, fp, fn, total}`-style object | transition matching at 12 and 25 frames |
| `bootstrap` | `{lo: float, hi: float}` | bootstrap bounds, a record only |
| `per_class` | `{letter: {precision, recall, f1}}` | per class |
| `confusion` | 4x4 int matrix | rows GT, columns ours, order `H A D X` |
| `n_spells`, `duration_s` | int, object | spell counts and duration statistics |

The exporter copies these keys unchanged from the report. If the report names them differently,
the exporter keeps the report names and `data.js` documents the mapping.

What the reports really hold, and what the exporter writes:

- `spell_f1` is keyed by the IoU as a string (`"0.3"` to `"0.7"`). Each value is
  `{f1, precision, recall, n_matched, n_missed, n_spurious}`. `n_matched` is tp, `n_spurious`
  fp, `n_missed` fn.
- `transitions` is keyed by the tolerance as a string (`"12"`, `"25"`). Each value is
  `{f1, precision, recall, n_gt, n_pred, n_matched, n_missed, n_spurious, n_ambiguous,
  mean_latency, median_latency, mean_abs_latency, p90_abs_latency}`.
- `per_class` is keyed `H`, `A`, `D`, each `{precision, recall, f1, support}`.
- `confusion` is the 4 x 4 matrix of the report, order `H A D X`.
- `bootstrap` is derived: `{lo: spell_bootstrap_low, hi: spell_bootstrap_high}`. The reports of
  `run_evaluation.py` run without draws, so both are `null` today. The ladder carries the
  1000-draw bounds from the ledger instead (Section 2.10).
- `n_spells` is derived: `n_pred_spells`. `duration_s` is derived:
  `{mean: mean_spell_duration_s, gt_mean: gt_mean_spell_duration_s}`.
- These report keys are also copied: `spell_f1_at_headline`, `spell_iou_headline`, `taus`,
  `n_gt_spells`, `n_pred_spells`, `n_gt_segments`, `n_pred_segments`, `spell_count_error`,
  `mean_spell_duration_s`, `gt_mean_spell_duration_s`, `bootstrap_low`, `bootstrap_high`,
  `bootstrap_n`, `spell_bootstrap_low`, `spell_bootstrap_high`, `accuracy`, `kappa`,
  `alive_dead_f1`, `fragmentation`, `share_err`, `pred_home_share`, `gt_home_share`.

The entries are the ladder versions (Section 2.10), the rejected versions, the four organiser
baselines `baseline_raw`, `baseline_smooth25`, `baseline_smooth50`, `baseline_smooth125`, and
`version`. A name without a report in `evaluation/reports/` is skipped with a warning. `version`
must have a report, or the exporter refuses to build.

### 2.7 `transitions.json`

Output of `tools.forensics.analyse()` for the current version, with the kinematic columns
stripped.

`forensics.analyse()` returns one record per transition, so the traces, tags and events sit
inside each record, not at the top level. Top-level keys:

| key | type | meaning |
|---|---|---|
| `name` | string | the version |
| `tol` | int | matching tolerance in frames, 25 |
| `trace_half_width` | int | half-width of each trace in frames, 50 |
| `reappear_lookback` | int | frames searched before a restart for a missing ball, 30 |
| `reports_stale` | bool | true if the scorer reports disagree with a fresh match |
| `gt_transitions` | list of GT records | one record per GT transition, in frame order |
| `spurious` | list of spurious records | our transitions with no GT transition |
| `latency_by_kind` | `{kind: object}` | see below |
| `kickoffs` | list | kickoff verdicts inside the window |
| `sandwiches` | list | GT `T D T` cases: `{team, gt_d_start, gt_d_stop, separate, our_frames: {H, A, D}, our_d_share_of_gt_d}` |
| `summary` | object | the scorer numbers that forensics prints: `{spell_f1_mean, macro_f1, f1_D, fragmentation, n_gt_spells, n_pred_spells, per_iou, transitions}` |

A GT record: `{gt_frame, from, to, kind, status, our_frame, our_from_to, latency,
ball_missing_at_f_minus_1, labels_trace, tags, events}`. `kind` is one of `H<->A`,
`alive->D`, `D->alive (kick)`, `D->alive (reappear)`. `status` is `matched` or `missed`.
`our_frame`, `our_from_to` (`"H->A"`) and `latency` (frames, negative means early) describe our
nearest transition into the same label, and are `null` when there is none.

A spurious record: `{our_frame, from, to, gt_label_at_frame, nearest_gt_transition, labels_trace,
tags, events}`. `nearest_gt_transition` is `{frame, from_to, distance}`, `distance` in frames.

In both records:

- `labels_trace` is `[{off: int, ours: string, gt: string}]`, every 3 frames over
  `-50..+50`, `off` relative to the record frame. Rows outside the window are dropped. The
  kinematic columns stay in the local bundle.
- `tags` is `[{code, team, type, start_frame, end_frame}]`, the Smart Tagging rows that overlap
  the trace, with both ends clipped to the window.
- `events` is `[[frame, kind, team, confidence]]`, the engine events of Section 2.5 within
  `trace_half_width` frames of the record frame.

`latency_by_kind` is keyed by the four kinds. Each value is `{n_gt, n_matched, matched_mean,
matched_median, nearest_n, nearest_mean, nearest_median, nearest_latencies}`. The `matched_*`
numbers cover matched transitions only. The `nearest_*` numbers cover every GT transition with a
nearest transition of ours. Means and medians are `null` when there is no case.

A kickoff row is `{kind, frame, rule, team, ours, goals_end, goals_team}`. Only the kickoffs
inside the window are kept. The distance of the taker to the centre mark is dropped.

All frames in this file are inside the public window.

### 2.8 `sanity.json`

| key | type | meaning |
|---|---|---|
| `ours` | object | `tools.sanity.sanity_stats()` for the current version |
| `baseline_smooth50` | object | the same for the baseline |
| `limits` | object | `VETO_*` values from `Config`: `{possessions: [180, 450], d_share: [0.18, 0.55], min_d_segments: 55, max_graded_x: 0}` |
| `watched` | object | `WATCH_*` values of `tools/sanity.py`: `{possessions: [230, 330], d_segments: [85, 150], min_tag_agreement: 0.85}` |

`ours` and `baseline_smooth50` have the keys of `sanity_stats()`: `halves` (keyed `"1"`, `"2"`),
`total`, `tags`, `set_pieces`, `gaps`, `d_block_end`, `vetoes`, `violations`, `watched`,
`tag_phase_codes`, `tag_codes_present`, `tag_codes_missing`. Three lists of full-match frame rows
are removed and their counts kept: `gaps.first` (our label changes inside long ball gaps),
`set_pieces.without_d` (set-piece tags with no `D` nearby) and `d_block_end.segments` (`D`
segments that end at a block end).

### 2.9 `config.json`

| key | type | meaning |
|---|---|---|
| `config_hash` | string | as in the manifest |
| `n_fields` | int | number of constants, 156 |
| `sections` | list | `{title: string, fields: [field]}` in source order |
| `classes` | object | `{counts: {E, F, U, D, P: int}, definitions: {letter: string}}` |

A field is `{name: string, value: number\|string\|bool\|null\|list, type: string, comment: string,
sources: [string], class: string\|null}`. Values come from `Config.default().as_dict()`. Comments
and sources come from `engine/config.py`.

`class` is the origin class of the field in our provenance inventory, the machine-readable record
of the source of every setting, which stays in the private repository: `E` selected on external labelled matches, `F`
fixed by physics or the Laws, `U` a label-free statistic of the unlabelled match, `D` a default
never varied against labels, `P` inferred from the public labels. A field that the inventory does
not list has `class: null`, and `classes.counts` does not count it. `classes.definitions` copies
the `class_definitions` of the inventory. The site counts the classes and filters the full table
by class. It does not show `sources`.

### 2.10 `ladder.json` and `cycles.json`

The page does not show these two files. The build of the technical document reads them.

`ladder.json`:

| key | type | meaning |
|---|---|---|
| `rows` | list | in PROGRESS "Current best" order |
| `rejected` | list | rejected versions, for example `c4` |

A ladder row is `{version: string, cycle: string, date: string, spell_f1: float, macro_f1: float,
possessions: int, d_share: float, md5: string, note: string, status: "kept"\|"rejected"\|"baseline"}`.

Details of a ladder row:

- `status` is `baseline` for `baseline_smooth50`, the first row, `kept` for a version with a
  `keep` verdict in the ledger, and `rejected` in `rejected`.
- `cycle` is the verdict tag of the ledger (`M1`, `P2`, `C8`), `baseline` for the baseline, or
  `C<n>` from a rejected name `c<n>_...`.
- `d_share` is a fraction (`0.498`), not a percentage.
- `md5` is the full md5 when the ledger note or the committed file gives it, else the 8-character
  prefix of the note (`c4_restart_anchor`), else `null`.
- Extra keys: `label` (the first cell of the PROGRESS row), `git`, `config_hash` (`null` for the
  baseline), `mean_latency` (frames, at tolerance 25), `d_segments` (`null` for a rejected row),
  and `bootstrap` (`{lo, hi}` of spellF1 from the ledger row "p5 bootstrap (1000)", or `null`).
- Numbers come from the ledger row. A number that the ledger leaves out comes from the PROGRESS
  row.

`cycles.json` is a list of `{id: string, title: string, hypothesis: string}`, where `hypothesis`
is the first sentence of the hypothesis in our lessons file of the private repository. Each entry also has `date`
and `status` (the text in the parentheses of the heading, for example `1 Nov` and `kept`).

### 2.11 `external_summary.json`

The summary numbers of the external validation, from our file of external numbers, which the
build of the technical document writes. Every number is a score or a difference of scores
on another match, rounded to 6 decimals, so that a value shown with 3 decimals matches the
technical document. The file holds no frame, no position and no label, and
no number of this match: `build_site.external_summary()` refuses a match whose id does not start
with `METRICA-` or `DFL-`. The name differs from `external.json` because the PDF build of the
technical document keys every JSON file by its stem and reads the full `external.json` too.

A view is one of `own` (the provider's own tracking, `raw` in `external.json`) and `atd` (the
same tracking with the ball smoothed like this match's ball, `matched` in `external.json`). An
interval is `{mean, low, high}`: a paired difference with its 5 to 95% bootstrap interval.

| key | type | meaning |
|---|---|---|
| `summary` | object | `{metrica, dfl, all}`, each `{own, atd}` of a group summary |
| `matches` | list | one row for each open match, sorted by `game` |
| `sealed` | list | one row for each sealed match, sorted by `game` |
| `ablations` | list | the component switch-offs, in the order of `external.json` |
| `oracle` | object | `{own, atd}`, each `{metrica, dfl, all}` of an oracle row |

- A group summary is `{engine_spell_f1, engine_macro_f1, baseline_spell_f1, baseline_macro_f1,
  engine_minus_baseline: interval, recall_by_duration: [{bucket, n, recalled, recall}]}`.
  `baseline` is `baseline_smooth50` run on the external match. `recall_by_duration` is the share
  of reference possessions that the engine finds at IoU 0.5, by length bucket.
- A match row is `{game, provider: "metrica"|"dfl", own, atd}`, each view `{engine_spell_f1,
  engine_macro_f1, baseline_spell_f1, engine_minus_baseline: interval, reference_possessions}`.
- A sealed row is `{game, own, atd}`, each view `{engine_spell_f1, engine_macro_f1}`. The sealed
  matches were opened once, on the engine before the fitted contact point. They have no baseline
  run.
- An ablation row is `{field, group, label, own, atd}`, each view `{mean, low, high, metrica,
  dfl}`: the change in mean spellF1 when the component `field` is switched off. A negative change
  means the component helps. `metrica` and `dfl` are the provider means.
- An oracle row is `{engine, reference_teams, reference_dead_ball}`: the mean spellF1 of the
  engine, of the reference teams on our dead-ball mask, and of the reference dead-ball mask with
  our teams.

## 3. Local bundle: `site/data/local/`

The site probes this folder at start. A missing folder is normal and means public mode.
`manifest.json` of the local bundle carries the same identity fields as the public one:
`schema`, `version`, `md5`, `config_hash`, `git`, `game`, `n_frames`, `fps`, `pitch_m`. The site
refuses the local bundle and shows a banner if its `md5` differs from the public manifest.

Extra keys of the local manifest:

| key | type | meaning |
|---|---|---|
| `halves` | `[{half: int, frame_start: int, frame_stop: int, files: {positions: string, positions_bin: string, frames: string, frames_bin: string}}]` | the per-half split, see Section 3.1 |
| `files` | `{string: int}` | name to size in bytes |

### 3.1 Per-half split

Each binary pair covers one half. The split is at frame 110808.

| half | `frame_start` | `frame_stop` |
|---|---|---|
| 1 | 0 | 110808 |
| 2 | 110808 | 199319 |

Row `i` of a half's array is frame `frame_start + i`. The graded flag (Section 3.3) tells which
rows are graded.

### 3.2 Binary layout (`positions_h1`, `positions_h2`)

Each pair is a `.bin` file and a `.json` sidecar header with the same stem
(`positions_h1.bin`, `positions_h1.json`).

- The `.bin` file is a struct of arrays. Each array is stored whole, row-major, one after
  another. Frame is the slow axis, slot is the fast axis.
- All elements are `int16` little-endian. Values are centimetres. Decoded value in metres is
  `value * scale`, with `scale = 0.01`.
- The sentinel `-32768` marks a missing value (a NaN in the engine). The site must test for it
  before scaling.
- Every array starts at an even byte offset. The exporter pads with zero bytes. `Int16Array`
  views need this.
- The sidecar JSON header:

| key | type | meaning |
|---|---|---|
| `dtype` | string | `int16` |
| `scale` | number | `0.01` |
| `sentinel` | int | `-32768` |
| `frame_start`, `frame_stop` | int | half-open frame range of this file |
| `n_slots` | int | 46 |
| `arrays` | `{name: {offset: int, shape: [int, int]}}` | byte offset into the `.bin` and the shape |

Arrays:

| name | shape | meaning |
|---|---|---|
| `px`, `py` | `[n, 46]` | player x and y in metres, from `engine_tactical()` so the C9 goalkeeper guard is applied |
| `bx`, `by` | `[n, 1]` | cleaned ball x and y |
| `bx_raw`, `by_raw` | `[n, 1]` | raw ball x and y before cleaning |

`n = frame_stop - frame_start`. The slot-to-team map is in `slots.json`. The total size of both
halves is about 37 MB.

### 3.3 Binary layout (`frames_h1`, `frames_h2`)

Per-frame engine internals. The layout rules of Section 3.2 apply, with one change: each entry
in `arrays` also carries its own `dtype` and `scale`, because the arrays have different types.
The top-level `dtype` and `scale` are absent in these headers. The header keys are
`frame_start`, `frame_stop`, `n_slots`, `arrays`. Every array offset is even.

| name | dtype | shape | scale | meaning |
|---|---|---|---|---|
| `owner` | `int8` | `[n]` | 1 | owner team: `0` H, `1` A, `-1` none |
| `p_dead` | `uint8` | `[n]` | `1/255` | probability that the ball is dead |
| `e_dead` | `uint8` | `[n]` | `1/255` | dead-ball evidence score |
| `flags` | `uint8` | `[n]` | 1 | bit field, see below |
| `slot_h`, `slot_a` | `int8` | `[n]` | 1 | slot index of the nearest player of each team, `-1` none |
| `d_h`, `d_a` | `uint16` | `[n]` | `0.01` | distance in metres from the ball to those players, `65535` missing |
| `speed` | `uint16` | `[n]` | `0.01` | ball speed in m/s, `65535` missing |
| `vx`, `vy`, `dv3` | `int16` | `[n]` | `0.01` | ball velocity and 3-frame velocity change in m/s, `-32768` missing |
| `meanv`, `sep` | `uint16` | `[n]` | `0.01` | mean player speed in m/s and team separation in metres, `65535` missing |
| `log_e` | `int16` | `[n, 3]` | `0.01` | log evidence for `H`, `A`, `D` in nats, `-32768` missing |

Bits of `flags`, from bit 0:

| bit | name | meaning |
|---|---|---|
| 0 | `stoppage` | the engine marks the frame as inside a stoppage |
| 1 | `visible` | the ball is visible in the raw tracking |
| 2 | `airborne_suspect` | the engine suspects the ball is in the air |
| 3 | `ghost` | the ball position is a ghost |
| 4 | `teleport` | the cleaner removed a teleport at this frame |
| 5 | `graded` | the frame lies in a graded block |

The total size of both halves is about 6 MB.

### 3.4 JSON files of the local bundle

| file | content |
|---|---|
| `labels_ours.json` | full-match RLE of our labels: `{name, n_frames, runs}`, same shape as Section 2.3 |
| `events_xy.json` | `{events: [[frame, kind, team, confidence, x, y]]}` for the full match, x and y of the ball in metres, 2 decimals |
| `deadball_xy.json` | `{onsets: [[frame, mechanism, confidence, x, y]], restarts: [[frame, team, kind, confidence, confirmed, x, y]], stoppages: [[start, stop]]}` for the full match |
| `transitions_kin.json` | the forensics traces of Section 2.7 with the kinematic columns kept, in the same record order: `{gt_transitions: [{gt_frame, labels_trace}], spurious: [{our_frame, labels_trace}]}`, a trace row `{off, ours, gt, speed, dv3, d_h, d_a}` in m/s and metres, `null` where missing, unclipped |
| `slots.json` | `{team_of_slot: [string x 46], visible_counts: [int x 46]}`, `H` or `A` for each slot |

In `events_xy.json` and `deadball_xy.json` the frames and stoppages are the decoder frames of
Section 2.5, and x and y are the cleaned ball at that frame, `null` when the ball is missing.
The local `manifest.json` also carries the `git` object of Section 2.1.

## 4. Minute bins

A bin has 60 s, which is 1500 frames. Bins are aligned to the first graded frame of each half.
Half 1 bins start at 11185. Half 2 bins start at 110808. The last bin of a half can be short.
There are 53 bins in each half and 106 in all. The graded frames in half 1 are `11185..90651`
and in half 2 `110808..190197`. A bin never crosses halftime and never covers an ungraded frame.

- `share_h`, `share_a`, `share_d`, `share_x` are fractions of the bin's frames.
- `n_possession_starts` counts `H` or `A` spells that start inside the bin. A spell starts at the
  first frame of a maximal run of the same letter `H` or `A`, with `D` breaking a spell.
- `n_d_segments` counts `D` segments that start inside the bin.
- `events_by_kind`, `onsets_by_mechanism` and `restarts_by_kind` count rows whose `frame` lies in
  `[start, stop)`.

## 5. The guard: `tools/viz/guard.py`

`assert_no_coordinates(obj, where, window, exempt)` runs on every public object before it is
written. `python -m tools.viz.guard site/data/public` runs on the written files and exits 1 on a
leak. The Pages workflow runs it. The guard uses only the standard library.

It refuses:

- Forbidden keys, compared in lower case: `x`, `y`, `px`, `py`, `bx`, `by`, `x_raw`, `y_raw`,
  `vx`, `vy`, `cx`, `cy`, `xy`, `pos`, `positions`, `coords`, `dh`, `da`, `v`, `dv_3`, `meanv`,
  `speed`, `dmin`, `sep`.
- Any list of more than 20,000 numeric leaves.
- Any list whose length equals `n_frames`.
- Any base64 or hex string longer than 4 KB.
- The window rule: any integer that denotes a frame must lie in `[11185, 18685)`. An integer
  denotes a frame when it sits under the keys `frame`, `start`, `stop`, `our_frame`, `gt_frame`,
  `start_frame`, `end_frame`, `gt_d_start`, `gt_d_stop`, `goals_end`, `frame_start`,
  `frame_stop`, as the first or second element of an RLE run, or as the first element of a row
  of `events`, `owner_events`, `onsets`, `restarts` and `stoppages` (both elements of a stoppage
  row). An exclusive end (`stop`, `end_frame`, `gt_d_stop`, `frame_stop`, the second element of
  a run or of a stoppage row) can equal `18685`.
- A file that is not JSON, and a JSON file that `manifest.json` `files` does not list.
- A manifest whose `public_window` is longer than 7500 frames (5 minutes), or whose `exempt`
  list holds a pattern that is not in the table below.

Exempt files are not subject to the window rule. They still pass every other check.

| exempt file | reason |
|---|---|
| `manifest.json` | holds the graded blocks and kickoffs |
| `minute_bins.json` | per-minute aggregates for the whole match |
| `labels_baseline_smooth50.json`, `labels_baseline_smooth125.json` | the organisers already publish them |
| `scores.json` | numbers from the scorer reports |
| `sanity.json` | summary statistics |
| `ladder.json`, `cycles.json` | history of the project |
| `config.json` | engine constants, some of them frame counts |

`external_summary.json` is not exempt. It holds no frame key, so the window rule finds nothing in
it, and it still passes every other check.

The window and the exempt list come from `manifest.json`. `labels_window.json`,
`window_detail.json` and `transitions.json` are never exempt.
