#!/bin/bash
# Round 8 arm N: the fine-tunes of one queue, one call at a time (each call: the given seeds x K 200 and 400).
# usage: n_finetune.sh QUEUE SEED [SEED ...]   QUEUE: folds (fold0, fold1, fold2) | others (lo_dfl, lo_skillcorner, final)
# Inputs (read only): round 7's fold networks ($R7T/folds/fold<k>/net_s<s>.pt), left-out-provider networks
# ($R7S/AQCTJ/<provider>/seed<s>/net.pt), finals ($R7T/final/net_s<s>.pt). Outputs: $NOUT/ft/<set>/K<k>/net_s<s>.pt.
# Waits while the battery is under 35 % or the battery guard pauses; resumable (DONE_s<s> skips a job).
# N2 (protocol R8 note 6.3): FORCED_LOSS=team K_LIST=400 FT_DIR=ft2 (defaults: full, "200 400", ft = arm N).
set -u
cd "$(dirname "$0")/../../.."
source tools/learned_spike/r8_scripts/n_env.sh
Q=$1; shift; SEEDS="$*"
mkdir -p $NOUT/logs
case $Q in
  folds) SETS="fold0 fold1 fold2";;
  others) SETS="lo_dfl lo_skillcorner final";;
  *) SETS=$Q;;
esac
init_of() {  # set seed -> network path
  case $1 in
    fold*) echo $R7T/folds/$1/net_s$2.pt;;
    lo_dfl) echo $R7S/AQCTJ/dfl/seed$2/net.pt;;
    lo_skillcorner) echo $R7S/AQCTJ/skillcorner/seed$2/net.pt;;
    final) echo $R7T/final/net_s$2.pt;;
  esac
}
batt() { pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d %; }
for set in $SETS; do
  while [ "$(batt)" -lt 35 ] || ! grep -q "paused=0" "$GLOG.last" 2>/dev/null; do sleep 60; done
  INITS=""; for s in $SEEDS; do INITS="$INITS --init $(init_of $set $s)"; done
  echo "$(date '+%F %T') start $set seeds $SEEDS (battery $(batt) %)" >> $NOUT/logs/finetune_jobs.log
  nice -n 10 $TPY -m tools.learned_spike.finetune $INITS --steps ${K_LIST:-200 400} --forced-loss ${FORCED_LOSS:-full} \
    --out $NOUT/${FT_DIR:-ft}/$set --threads 2 $FT_SW > $NOUT/logs/${FT_DIR:-ft}_${set}_s$(echo $SEEDS | tr ' ' '_').log 2>&1
  echo "$(date '+%F %T') end $set seeds $SEEDS rc=$?" >> $NOUT/logs/finetune_jobs.log
done
