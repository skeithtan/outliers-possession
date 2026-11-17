#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# Arm S: the two left-out fits, one after the other, then the left-out predictions.
cd ~/analysis/r8/wt-s; export MTAH_DATA_DIR=~/mtah2027/data OMP_NUM_THREADS=4
S971=${SCRATCH:?}
O=~/mtah2027/r8/s; PY=~/.venvs/barca-torch/bin/python; STEPS=${STEPS:-800}
echo "$(date '+%F %T') start, commit $(git rev-parse HEAD), steps $STEPS" >> $O/logs/run_fits.log
for prov in metrica dfl; do
  until [ "$(pmset -g batt | grep -o '[0-9]*%' | tr -d %)" -ge 35 ]; do sleep 60; done
  $PY -m tools.learned_spike.set_train train --provider $prov --seed 0 --tokens $O/tok \
    --labels $S971/r5_work/cache_c_train2tj --sc-labels $S971/sc_fine/cache_possgroup \
    --out $O/fits --steps $STEPS --eval-every 100 --threads 4 > $O/logs/train_$prov.log 2>&1
  echo "$(date '+%F %T') train $prov exit $?" >> $O/logs/run_fits.log
  if [ $prov = metrica ]; then G="METRICA-S1 METRICA-S2 METRICA-S3"; else G="DFL-J03WMX DFL-J03WN1 DFL-J03WOH DFL-J03WOY DFL-J03WPY"; fi
  $PY -m tools.learned_spike.set_train predict --net $O/fits/$prov/seed0/net.pt --tokens $O/tok \
    --games $G --out $O/p5_set --threads 4 > $O/logs/predict_$prov.log 2>&1
  echo "$(date '+%F %T') predict $prov exit $?" >> $O/logs/run_fits.log
done
touch $O/FITS_DONE
