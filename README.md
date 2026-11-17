# Barça possession, More than a Hack 2027

This repository is our team's entry for Phase 1 of
[More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027). The engine
labels every frame of FC Barcelona against Athletic Club with one of four labels:

- `H`: the home team has possession
- `A`: the away team has possession
- `D`: dead ball
- `X`: no answer

The engine computes the probability of `H`, `A` and `D` for each frame with two learned models.
The first is a small recurrent neural network at 5 frames a second. The second is a per-frame model
of gradient-boosted trees. The engine averages the two, with weights 0.75 and 0.25. A Viterbi
decoder then turns the average into labels. The decoder gives the first 1 s after the kickoff of
each half to the team that takes the kickoff (Law 8). Both models trained offline on labelled external
matches from three providers: Metrica sample games, Sportec Open DFL (DFL) and SkillCorner. No label
and no video of this match trained them.
PFF FC's World Cup 2022 matches are test data only.

The engine loads the two models from two weights files in `engine/models/` and never trains. It
evaluates both models in `numpy`, so it needs only `numpy` and `pandas`. It uses no randomness.

The rule engine is the engine of physical rules that the two models replace. It finds stoppages
from the movement of the ball and the players, and changes the team when a player of the other
team plays the ball. It is the reference for every gain that we report. The engine keeps
only its ball cleaning, its goalkeeper-swap correction and its dead ball before each kickoff (the
kickoff frames come from `games.py`). When you switch the learned models off, the rule engine
writes the whole output.

The organisers rank entries on `spellF1`. The baseline to beat scores **0.2433** on the full match.

The technical document explains the method, the assets we used and dropped, the possession rules,
the limitations and the reproduction. It is short, because the organisers ask for a short
document. We submit it with the entry as `The Outliers.pdf`. It is also attached to the
[release of the tag `submission`](https://github.com/skeithtan/outliers-possession/releases/tag/submission).

## The public repository and the private repository

The public repository is
[github.com/skeithtan/outliers-possession](https://github.com/skeithtan/outliers-possession). We
develop in a private repository and copy the submitted state to the public
one. The copy holds the code, the two weights files, the organisers' scorer and starter code, the
tests and the pinned requirements. It leaves out the commit history, our documents,
our records and our working notes. `python -m tools.export_public` makes the copy and checks it.

Some comments in the code name our records and our working rules. Those files stay in the private
repository.

The engine source hash identifies the engine in both repositories. It is the sha1 of
`engine/*.py` and `games.py`, and `python -m tools.export_public --engine-hash` prints it. The
engine has the hash `5ea4241b504e56d69707177b80c74bd807db5a73`.

## Reproduce the submitted file

The submitted file comes from the engine with the two learned models, on `main`. The submitted
commit on `main` is tagged `submission`: run `git checkout submission`.

You need Python 3.9 to 3.12 (numpy 2.0.2 has no wheel for 3.13) and the organisers' match data
bundle. We tested the engine on Python 3.9.6.

1. Put the data bundle in `data/`. The engine reads `data/tactical/`.
2. If the bundle is in another folder, point `MTAH_DATA_DIR` at the folder that holds `tactical/`:

   ```bash
   export MTAH_DATA_DIR=/path/to/mtah2027
   ```

3. Install the pinned packages and run the engine from the repository root:

   ```bash
   pip install -r requirements.txt && python -m engine.run --out "The Outliers.csv"
   ```

4. Compare the md5 that the command prints with `d7f167a2401387bd8339d1bcbb292f2c`, the
   submitted file.

The command writes the CSV and runs the organisers' validator on it. It also prints a summary of
the full match with the sanity vetoes, and the md5 of the file.

The fresh-clone check on Python 3.9.6 and `numpy` 2.0.2 (macOS arm64), in a new environment
installed from `requirements.txt`, passed: both runs gave the md5 above. Linux
arm64 gave the same md5 on two runs, and emulated Linux x86-64 on two runs. On Intel and AMD
hardware, with AVX2 and AVX-512, under Linux, Windows and macOS, the engine wrote the same bytes
as the Mac on the three Metrica sample games. The data of this match stays on our machine. There
we reproduced the one arithmetic difference that we found between the two processor families, a
fused multiply-add in one interpolation. The md5 stayed the same. On
one thread the two models without the kickoff team rule take 54 s on this match on an Apple M1 (macOS).
The maximum resident memory is 3.6 GB. Timing depends on the machine.

The engine loads two weights files:

- `engine/models/learned_v7.npz`, the four networks, md5 `31e88976ea8da0169b15f07ff44ee17c`
- `engine/models/learned_pf_v7.npz`, the per-frame model, md5 `0b5e81aa00c0486057aa7eb790c98109`

The engine reads two files of the bundle, `BAR-ATH_tactical_atd.xml` and
`BAR-ATH_tactical_atd.txt`. The rule engine also reads `BAR-ATH_tactical_smart_tagging.xml`.
Neither opens the videos.

The md5 changes when the output bytes change.

### Switch the learned models off

To write the rule engine's labels instead, do these steps:

1. Write a configuration file `off.json` that holds `{"LEARNED_MODE": "off"}`.
2. Run `python -m engine.run --config off.json --out rules.csv`.
3. Compare the md5 with `7844cad862e38d55cf9a3fc29b18ae85`, the rule engine's output.

## Validate and score the file

1. Validate the exact file that you upload:

   ```bash
   python evaluation/validate_submission.py "The Outliers.csv"
   ```

2. To score a file on the public window, copy it into `evaluation/results/`.
3. Run the scorer. It scores every CSV in `evaluation/results/` and writes reports to
   `evaluation/reports/`:

   ```bash
   python evaluation/run_evaluation.py
   python evaluation/run_evaluation.py --bootstrap 1000          # add confidence intervals
   ```

The organisers publish labels for the first 5 minutes only. That window has 17 possessions. Use it
to check that the pipeline is correct. Do not use it to choose between versions. The reason is in
the organisers' [EVALUATION.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md).

## Run the engine on another match

To label another match, add an entry for it to `GAMES` in `games.py`. Then run
`python -m engine.run --game <name> --out <file>.csv`. The entry gives the frame count, the halves,
the kickoff anchors and the pitch size. Two fields are optional:
`assets` names the tracking files under `MTAH_DATA_DIR`, and `kickoff_from_anchor` takes the
first-half kickoff from the `kickoff` anchor. The `METRICA-S3` entry is an example.

Never write the CSV of another match into `evaluation/results/`. The scorer scores every file there
against the ground truth of this match.

## Team

- Salil Marath Ponmadom
- Samuel Keith Tan
- Gowri Umesh
- Shannon Savio D'Souza
- Saud Ahmad

## The organisers' documents

We built on the organisers' starter repository at commit `c073f29`. Their documents are in
[their repository](https://github.com/metrica-sports/more-than-a-hack-2027). Read them in this
order:

| | |
|---|---|
| [README.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/README.md) | the challenge on one screen |
| [CHALLENGE.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/CHALLENGE.md) | the problem, the four labels, and what counts as possession |
| [DATA.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md) | every asset, its format, and the frame alignment |
| [EVALUATION.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md) | how the organisers score us, and why. Read it before you tune anything |
| [SUBMISSION.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md) | the file format, the three deliverables, and how to submit |
| [RULES.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md) | what we can and cannot do |
| [FAQ.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/FAQ.md) | common questions |
| [OVERVIEW.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/OVERVIEW.md) | context, scoring summary and calendar |

## Our documents

The technical document holds the method and the evidence. The private repository holds our plan,
the source of every setting, the test plans and their results, and the ledger of every scored run.
It also holds our working rules, our writing guides, the status of each task and the lessons of
each improvement cycle. [tools/analysis/](tools/analysis/) holds the scripts that measured the
facts of our plan.

## Set up for development

You need Python 3.9 to 3.12 (numpy 2.0.2 has no wheel for 3.13). We tested 3.9.

1. Create the virtual environment and install the requirements:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
   ```

   To use the organisers' loaders and notebooks, also install `requirements-starter.txt`.
   To retrain the learned models, also install `torch` and `scikit-learn`. The engine does not
   need them. [tools/learned_spike/README.md](tools/learned_spike/README.md) gives the training
   code.

2. Download the match data bundle. The organisers gate it, and you need a registered account.

3. Unpack the bundle into `data/`, or point the repository at it:

   ```bash
   export MTAH_DATA_DIR=/path/to/mtah2027
   ```

4. Run the tests:

   ```bash
   python -m pytest -q tests
   ```

Never commit the match data. The organisers disqualify a team that publishes it. The `.gitignore`
file excludes `data/`, but check `git status` before every commit.

To rebuild the four organisers' baselines in `evaluation/results/`, run `python starter/baseline.py`.

## Rules

The full rules are in the organisers' [RULES.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md). These are the ones that bite:

- Never label a frame of this match by hand, not even to validate. A team that does is
  disqualified.
- Never upload the videos, the data or an extract of them to a third-party hosted service
  ([OVERVIEW.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/OVERVIEW.md). [RULES.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md) forbids redistribution of the
  data.)
- The repository must regenerate the submitted file exactly. Pin every dependency and hold every
  seed constant.
- The repository must be public at submission time. The deadline is **17 November 2026**.
- Submit through the form that the organisers emailed to registered participants. Send three
  things: the CSV named `The Outliers.csv`, the technical document named `The Outliers.pdf`, and the
  repository URL. One response per team counts, the last one before the deadline.

## Layout

```
games.py              match constants, asset paths, frame alignment
evaluation/           official scorer, validator, public ground truth, baseline submissions
starter/              organisers' loaders, baseline and notebooks
engine/               our submission pipeline
tools/                our forensics, sanity checks and analysis scripts
tests/                our unit and reproducibility tests
site/                 a web page about the engine (static files)
data/                 match data, never committed
```
