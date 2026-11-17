#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# STATUS: REHEARSED 13 Nov (RECIPE=AQCSJ, 30-step stage50-v2s networks, seeds 0-3, bonuses 0 1 2, scratch r5_work/reh6): exit 0; the rule said c23 ships (floor), as expected for 30-step models. With PF_WEIGHTS: the grid tool with --pf-weights was rehearsed through r4_screen score (same code path), not this script.
# Candidate network + per-frame member: set PF_WEIGHTS (the member's weights file, pf_export.py) and optionally
# PF_WEIGHT (0.5); every step then scores the blend as engine.learned decodes it (LEARNED_PF on). Unset: the network alone.
# Round 6 decoder costs and prior (round 5's costs_R5.sh for RECIPE=AQCSJ with the stage50-v2s inputs of r6_env.sh):
# r4_screen grid on the left-out-provider ensembles of seeds 0-3 at prior bonus 0, 1, 2 (in parallel, GRID_JOBS, default 3
# as round 4), then r4_choose costs --step3 mean2. Writes RUNDIR/costs/chosen.env (PAIR_HA PAIR_AD PRIOR, or C23_SHIPS=1)
# that gates_R5.sh, g3b_R5.sh and final_R5.sh read. Env: RECIPE RUNDIR CODE; optional GRID_BONUSES ("0 1 2").
CHAIN=costs_R6
source ${SCRATCH:?}/r5_work/scripts/r6_chain_common.sh
trap finish EXIT
set -u
C=$RUNDIR/costs; mkdir -p $C
BON=${GRID_BONUSES:-"0 1 2"}; GJ=${GRID_JOBS:-3}
say "grids at bonus $BON on $OUTROOT/$RECIPE seeds $SEEDS (${GJ} at a time)"
for p in metrica dfl skillcorner; do for s in $SEEDS; do
  [ -f "$OUTROOT/$RECIPE/$p/seed$s/DONE" ] || { say "missing left-out network $p seed $s (run full_R5.sh stage 1)"; exit 1; }
done; done
REH=${STEPS:+--rehearsal}
PFA=""; [ -n "${PF_WEIGHTS:-}" ] && PFA="--pf-weights $PF_WEIGHTS --pf-weight ${PF_WEIGHT:-0.5}"
export C TPY NICE THREADS OUTROOT RECIPE SEEDS SCORE_INPUTS2S SP REH PFA
grid() {
  B=$1; PB=""; [ "$B" != 0 ] && PB="--prior-bonus $B"
  nice -n $NICE $TPY -m tools.learned_spike.r4_screen grid --recipe-dir $OUTROOT/$RECIPE --seeds $SEEDS $SCORE_INPUTS2S \
    --loose-dir $SP/r3b_score/work/views/refs --threads $THREADS --out $C/grid_b$B $PB $REH $PFA > $C/grid_b$B.log 2>&1
  echo "grid b$B exit $?" >> $C/status
}
export -f grid
for B in $BON; do echo $B; done | xargs -P "$GJ" -L 1 bash -c 'grid $0'
for B in $BON; do [ -f $C/grid_b$B/grid.json ] || { say "grid b$B missing"; exit 1; }; done
G=""; for B in $BON; do G="$G --grid$B $C/grid_b$B/grid.json"; done
nice -n $NICE $EPY -m tools.learned_spike.r4_choose costs $G --step3 mean2 --out $C/choice \
  --title "Round 6 decoder costs and prior ($RECIPE; Section 3: 8.1 grid, 8.4 step 3)" > $C/choice.txt 2>&1 || { say "r4_choose failed"; exit 1; }
$EPY - "$C/choice.json" "$C/chosen.env" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as h:
    if r.get("decision", "").startswith("c23"):
        h.write("C23_SHIPS=1\nREASON='{}'\n".format(r.get("reason", "")))
    else:
        b = (r.get("prior") or {}).get("bonus", 0.0)
        h.write("PAIR_HA={:g}\nPAIR_AD={:g}\nPRIOR={:g}\nSTEP3={}\n".format(r["pair"][0], r["pair"][1], b, r["step3"]))
PY
say "decision: $(grep -m1 Decision $C/choice.md); $(tr '\n' ' ' < $C/chosen.env)"
