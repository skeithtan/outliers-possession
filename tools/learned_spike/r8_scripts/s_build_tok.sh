#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
cd ~/analysis/r8/wt-s; export MTAH_DATA_DIR=~/mtah2027/data OMP_NUM_THREADS=1
S971=${SCRATCH:?}
ALL=($(~/.venvs/barca-torch/bin/python -c "import sys; sys.path.insert(0,'.'); from tools.learned_spike import folds; print(' '.join(folds.ALL))"))
A=(); B=(); for i in "${!ALL[@]}"; do if (( i % 2 )); then B+=(${ALL[$i]}); else A+=(${ALL[$i]}); fi; done
for part in A B; do
  eval "G=(\"\${$part[@]}\")"
  nice -n 5 ~/.venvs/barca-torch/bin/python -m tools.learned_spike.set_tokens build --labels $S971/r5_work/ca2t --engine-cache $S971/r4_work/engine_data_raw --out ~/mtah2027/r8/s/tok --games ${G[@]} --aug-k 16 > ~/mtah2027/r8/s/logs/tok_$part.log 2>&1 &
done
wait; echo done > ~/mtah2027/r8/s/TOK_DONE
