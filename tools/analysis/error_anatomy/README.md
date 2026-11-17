# Error anatomy of the team sequence

These scripts split the spellF1 loss of the default engine on the eight open references into
error classes. They give each class its exact Shapley share of the loss. The write-up and the
numbers are in a research note of our private repository.

The scripts are diagnostic. They decide nothing and set no value. They were written on
8 November 2026, and they are not reviewed to the standard of `engine/`. They have no unit tests.

## Rules

- The scripts refuse BAR-ATH and the two sealed DFL matches (`common.FORBIDDEN`).
- They read no file under `evaluation/gt/`.
- They write only to the work folder `$ERROR_ANATOMY_WORK`. The default is
  `/tmp/error_anatomy`. Never set it to a folder inside the repository.

## Run

Run the scripts from the repository root in this order. `q4.py` writes the episodes that the later
scripts read, and `q3_detector.py` writes the table that `q_absorbed.py` reads. The engine run takes about 2 minutes
with 6 jobs. The other steps take about 2 minutes together.

```bash
source ~/.venvs/barca/bin/activate
export ERROR_ANATOMY_WORK=/tmp/error_anatomy
python tools/analysis/error_anatomy/refs.py
python tools/analysis/error_anatomy/run_parts.py --views matched raw --jobs 6
python tools/analysis/error_anatomy/q123.py
python tools/analysis/error_anatomy/q4.py
python tools/analysis/error_anatomy/q13_context.py
python tools/analysis/error_anatomy/q3_detector.py
python tools/analysis/error_anatomy/q_attr.py
python tools/analysis/error_anatomy/q_plaus.py
python tools/analysis/error_anatomy/q_absorbed.py
python tools/analysis/error_anatomy/q56.py
python tools/analysis/error_anatomy/report_tables.py
```

`run_parts.py` runs the engine with the configuration of the checkout. To compare two engine
versions, run the sequence once per version with a separate work folder.

## What each script does

| script | output in `$ERROR_ANATOMY_WORK/out` | content |
|---|---|---|
| `refs.py` | `refs/` | the eight open references, md5-checked by `external_sweep.prepare_reference` |
| `run_parts.py` | `runs/<view>/<game>.npz`, `_tables.pkl` | engine labels, the Viterbi path before `min_durations`, owner track, emissions, ball and event tables |
| `q123.py` | `q123.json`, `changes_<view>.csv` | reference owner changes and our detection at ±5 to ±50 frames, our unmatched changes |
| `q4.py` | `q4.json`, `episodes_<view>.csv` | episodes and their 12 classes, exact Shapley shares, IoU 0.5 accounting |
| `q13_context.py` | `q13_context.json`, `ref_context_<view>.csv`, `our_context_<view>.csv` | context of each reference hand-over and each of our changes |
| `q3_detector.py` | `q3_detector.json`, `handover_detector_<view>.csv`, `absorbed_detector_<view>.csv` | what the ball showed at missed hand-overs (the detector ladder) |
| `q_attr.py` | `q_attr.json` | attribution windows of strong contacts at hand-overs |
| `q_plaus.py` | `q_plaus.json` | nearest-team plausibility inside absorbed possessions |
| `q_absorbed.py` | `q_absorbed.json` | detector ladder at absorbed starts |
| `q56.py` | `q56.json` | possession durations, recall by duration, ball missing at hand-overs |
| `report_tables.py` | `tables.md` | the tables of the write-up |

## The episode classes (`q4.py`)

Every graded frame where our label differs from the reference belongs to one episode: a maximal
run with a constant (ours, reference) pair. The start and the end of the episode give its class.

| class | meaning |
|---|---|
| `T_late` | the reference changed team, and we followed later |
| `T_early` | we changed team before the reference |
| `T_missed_absorbed` | the reference changed team and changed back before we followed |
| `T_missed_to_dead` | the reference changed team, and we did not follow before a stoppage or the run end |
| `T_spurious` | we changed team, and the reference did not |
| `T_swap` | both changed, to different teams |
| `T_restart_wrong` | a wrong team from our restart, or from both restarts |
| `T_restart_after_missedD` | the reference restarted with the other team while we stayed alive through its stoppage |
| `D_onset`, `D_restart` | the timing of a stoppage edge |
| `D_spurious` | a stoppage that we added, or our `D` over a whole alive stretch of the reference |
| `D_missed` | a stoppage that we missed, or an alive island inside one |

To fix a class, set the frames of its episodes to the reference. If you fix every class, spellF1
is 1. The script asserts this. The Shapley value of a class is its mean marginal gain over all
4,096 subsets of the 12 classes, for each match.
