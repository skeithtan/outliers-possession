# tools/learned_spike: a learned sequence model against c22

This folder holds a time-boxed spike. It asks one question. Does a learned sequence model,
trained on our external references only, beat the engine `c22` on held-out external matches
under the same labels? The learned stage grew from it.

The engine never imports this folder. The training code here produced the two weights files that
the submitted engine loads: `engine/models/learned_v7.npz` (the network, md5
`31e88976ea8da0169b15f07ff44ee17c`) and `engine/models/learned_pf_v7.npz` (the per-frame model, md5
`0b5e81aa00c0486057aa7eb790c98109`). The engine evaluates both in numpy and
never trains. To retrain, you need torch and scikit-learn, which the engine does not need. The
spike never trains, tunes or scores on BAR-ATH, never reads `evaluation/gt/*` and never opens
the sealed matches DFL-J03WQQ and DFL-J03WR9. Every tool refuses them.

## Method

The model reads identity-free features of the tracking. It writes per-frame probabilities of
`H`, `A` and `D`. A Viterbi decoder turns the probabilities into labels.

- **Features** (`features.py`). There are 50 features per frame. They include ball visibility,
  position, speed and velocity, and the time since and until a sighting. They also include the
  nearest and second-nearest player of each team, counts within 1.5, 3 and 5 m, and the speed
  and relative velocity of the nearest player. Team centroids, spreads, the deepest player and
  the share of near-stationary players complete the set. The pitch turns per half so that the
  home team attacks +x. The tool averages the features over bins of 5 frames (5 fps).
- **Symmetries.** Each feature has a kind. So a mirrored copy (y to -y) and a swapped copy (home
  and away exchanged, pitch turned) are a column permutation and a sign change. Training draws
  one of the four copies per crop. Prediction averages the four copies.
- **Targets.** The references go through `control_labels(..., 50)`
  (`tools/analysis/control_reference.py`), the 2 s controlled view.
- **Network** (`model.py`). A linear layer, one residual temporal convolution (5 bins), a
  two-layer bidirectional GRU with 64 units and a linear head. It has 148,611 parameters.
- **Training** (`lomo.py`). Random 2-minute crops, batch 16, Adam with a one-cycle schedule,
  1,200 steps, seed 0. The loss is cross entropy with `X` masked. Frames within 1 s of a label
  change weigh 3 times more. The views are raw, matched, sigma 9.5 and one degraded copy
  (`degrade.py`: dead-ball blanking, live ball gaps, player track fragments).
- **Folds.** For each test match, one match of the other provider is the validation match. It
  picks the checkpoint (lowest loss) and the two switch costs (best spellF1 against its 2 s
  reference, grid 1 to 32 for H-A and 2 to 64 for alive-dead). The model trains on the other
  six matches. The test match is used once, at the end, on its raw view.
- **Decoder.** The tool interpolates the 5 fps probabilities to 25 fps. Viterbi runs over
  `H`, `A`, `D` in each graded block of games.py, with the costs of the fold.

## Results

LOMO on the eight open references, raw view of the test match. The c22 runs are the default
engine outputs of `external_sweep.py` (run keys in `results/summary_raw.json`). The interval is
the 5 to 95 % paired bootstrap of `tools/paired_summary.py` (1000 draws, seed 0).

| match | model, 2 s ref | c22, 2 s ref | model, raw ref | c22, raw ref |
|---|---|---|---|---|
| METRICA-S1 | 0.8634 | 0.7810 | 0.7479 | 0.6901 |
| METRICA-S2 | 0.9103 | 0.7595 | 0.7941 | 0.7081 |
| METRICA-S3 | 0.8600 | 0.7227 | 0.7328 | 0.6878 |
| DFL-J03WMX | 0.7764 | 0.6257 | 0.6112 | 0.5064 |
| DFL-J03WN1 | 0.7600 | 0.5239 | 0.5870 | 0.4253 |
| DFL-J03WOH | 0.8349 | 0.5479 | 0.6973 | 0.4539 |
| DFL-J03WOY | 0.8073 | 0.5200 | 0.6847 | 0.4243 |
| DFL-J03WPY | 0.7717 | 0.5152 | 0.5839 | 0.4098 |

| group | 2 s ref, model minus c22 [5 to 95 %] | raw ref, model minus c22 [5 to 95 %] |
|---|---|---|
| Metrica (3) | +0.124 [+0.099, +0.147] | +0.063 [+0.038, +0.087] |
| DFL (5) | +0.244 [+0.221, +0.266] | +0.189 [+0.169, +0.208] |
| all (8) | +0.199 [+0.182, +0.214] | +0.142 [+0.126, +0.157] |

The model wins on all eight matches and on both references. The documents in `results/` hold
the full tables, the stress view and the hybrids. SPIKE-LEARNED.md discusses them.

## Commands

Run everything with one script. It takes about one hour on an 8-core Mac.

```bash
tools/learned_spike/run_all.sh WORK TORCH_PYTHON ENGINE_PYTHON EXTCACHE
```

To make the torch venv, use these commands. Do not install torch into `~/.venvs/barca`.

```bash
python3 -m venv WORK/venv-torch
WORK/venv-torch/bin/pip install numpy==2.0.2 pandas==2.3.3 torch \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

The spike used torch 2.8.0 on Python 3.9.6.

| file | what it does |
|---|---|
| `features.py` | the 50 features, the symmetry tables, the 5 fps bins |
| `degrade.py` | the degraded training copy |
| `build.py` | the feature and label caches per match and view |
| `model.py` | the network, the symmetric prediction, the Viterbi decoder |
| `lomo.py` | one LOMO fold: train, choose the checkpoint and the costs, predict the test match |
| `summarise.py` | model against c22 on the 2 s and raw references, paired intervals |
| `stress.py` | the label-free ball-gap stress view, for c22 and for the model |
| `hybrid.py` | two hybrids of c22 and the model |
| `bar_sanity.py` | label-free numbers on BAR-ATH from the eight fold models |
| `run_all.sh` | the whole spike |

## The learned stage: folds and final ensemble

Our test plan of the learned stage sets its training. The tools below implement it on the 28 external matches (8 open references and 20 SkillCorner
matches). They score no test match: the scores come from `engine.run` with the fold weights.

| file | what it does |
|---|---|
| `folds.py` | the 28 matches, the eight folds, the views per match, the final step count (no torch) |
| `decoders.py` | ensemble rounding, upsampling, the M and H decoders over a cost grid, the tie rule (numpy) |
| `train.py` | the recipe: batches, loss, one-cycle Adam, checkpoint on the validation pair |
| `c22.py` | c22 labels (engine.run, default config) of external matches, cached by run key |
| `kfold.py` | one fold: four models, costs on the validation pair, then the test predictions |
| `final.py` | the final four models on all 28 matches, and the median fold costs |
| `run_stage.sh` | the whole stage: 32 fold models and 4 final models, 4 jobs at a time |

Run the whole stage with one command. Weights go to OUT, never into the repository.

```bash
tools/learned_spike/run_stage.sh CACHE EXT OUT TORCH_PYTHON ENGINE_PYTHON
```

CACHE is the `build.py` cache. EXT holds the engine npz cache of the 28 matches in
`views/raw/engine_data`, the Metrica references in `refs`, and the c22 runs. For a smoke test,
set `STEPS=50 EVAL_EVERY=25 FINAL_STEPS=20` before the command.

## To add a match

1. Add its games.py entry.
2. Teach `build.reference_path()` where its reference CSV is.
3. Add its name to `--games` of `build.py` and `lomo.py`.

## What the engine would need

This section is the plan that we wrote before the engine loaded a learned artefact. The team
agreed to the change. The submitted engine now loads two weights files and runs them in numpy.
The plan was:

1. **Numpy inference.** Export the weights to an `.npz` file. Write the linear layer, the
   convolution, the GRU cell and the softmax in numpy. The GRU is about 40 lines. Commit the
   weights (an ensemble of eight fold models in float32 is about 4.8 MB, one model is 0.6 MB).
2. **Byte reproducibility.** Inference with fixed weights and float64 numpy is deterministic on
   one machine. Across machines, BLAS order can change the last bits of a probability. Round
   the probabilities (for example to 6 decimals) before the decoder, and test the md5 on Linux
   x86 and on macOS arm64. Training randomness does not matter once the weights are committed.
3. **Run time.** On BAR-ATH the features take 1.5 s. Eight models with four symmetric copies
   take 42 s in torch on 4 threads. The decoder takes under 1 s. A numpy GRU loops over about 40,000 steps per
   direction and layer. We estimate under 30 s for one model and four copies. We did not measure it.
4. **Compliance.** Training uses external matches only. The switch costs come from the folds.
   No value comes from BAR-ATH. Organiser answer #11 asks to commit the weights and to
   regenerate the CSV byte for byte on CPU.

## Engine port (learned stage)

`engine/learned.py` runs the network in numpy for `python -m engine.run`
(`LEARNED_STAGE_PROTOCOL.md`). Two tools connect the spike to it:

- `export.py` writes torch `net.pt` files into one weights `.npz` for the engine.
- `feature_md5.py` checks that `engine/learned_features.py` and `features.py` give the same
  feature bytes on each external match.
