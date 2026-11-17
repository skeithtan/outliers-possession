#!/bin/bash
# Round 8 arm N: the set folders that n_score.py reads, as symbolic links (nothing is copied).
#   $NOUT/sets/E2:   round 7's networks (E2's own): fold0-2, lo_dfl (AQCT/dfl), lo_skillcorner (ABCTJ/skillcorner), final
#   $NOUT/sets/K200, K400: the fine-tuned networks of n_finetune.sh ($NOUT/ft/<set>/K<k>/)
set -u
cd "$(dirname "$0")/../../.."
source tools/learned_spike/r8_scripts/n_env.sh
E=$NOUT/sets/E2
for k in 0 1 2; do mkdir -p $E/fold$k; for s in 0 1 2 3; do
  ln -sfn $R7T/folds/fold$k/net_s$s.pt $E/fold$k/net_s$s.pt; ln -sfn $R7T/folds/fold$k/model_s$s.json $E/fold$k/model_s$s.json; done; done
for p in dfl skillcorner; do mkdir -p $E/lo_$p; for s in 0 1 2 3; do
  ln -sfn $R7S/AQCTJ/$p/seed$s/net.pt $E/lo_$p/net_s$s.pt; ln -sfn $R7S/AQCTJ/$p/seed$s/model.json $E/lo_$p/model_s$s.json; done; done
mkdir -p $E/final; for s in 0 1 2 3; do ln -sfn $R7T/final/net_s$s.pt $E/final/net_s$s.pt; ln -sfn $R7T/final/final_s$s.json $E/final/model_s$s.json; done
for K in 200 400; do for set in fold0 fold1 fold2 lo_dfl lo_skillcorner final; do
  mkdir -p $NOUT/sets/K$K/$set
  for s in 0 1 2 3; do [ -f $NOUT/ft/$set/K$K/net_s$s.pt ] || continue
    ln -sfn $NOUT/ft/$set/K$K/net_s$s.pt $NOUT/sets/K$K/$set/net_s$s.pt; ln -sfn $NOUT/ft/$set/K$K/model_s$s.json $NOUT/sets/K$K/$set/model_s$s.json; done
done; done
# N2 (note 6.3): $NOUT/sets/N2 from $NOUT/ft2/<set>/K400
for set in fold0 fold1 fold2 lo_dfl lo_skillcorner final; do
  mkdir -p $NOUT/sets/N2/$set
  for s in 0 1 2 3; do [ -f $NOUT/ft2/$set/K400/net_s$s.pt ] || continue
    ln -sfn $NOUT/ft2/$set/K400/net_s$s.pt $NOUT/sets/N2/$set/net_s$s.pt; ln -sfn $NOUT/ft2/$set/K400/model_s$s.json $NOUT/sets/N2/$set/model_s$s.json; done
done
