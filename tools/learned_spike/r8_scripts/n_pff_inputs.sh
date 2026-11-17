#!/bin/bash
# Round 8 arm N: the engine inputs of the 43 open PFF matches (n_pff.py inputs), once.
set -u
cd "$(dirname "$0")/../../.."
source tools/learned_spike/r8_scripts/n_env.sh
batt() { pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d %; }
while [ "$(batt)" -lt 35 ] || ! grep -q "paused=0" "$GLOG.last" 2>/dev/null; do sleep 60; done
MTAH_DATA_DIR=$HOME/mtah2027/refs nice -n 10 $TPY -m tools.learned_spike.r8_scripts.n_pff inputs --engine-cache $NOUT/pff_engine_data --out $NOUT/pff_inputs > $NOUT/logs/pff_inputs.log 2>&1
echo "$(date '+%F %T') pff inputs rc=$?" >> $NOUT/logs/finetune_jobs.log
