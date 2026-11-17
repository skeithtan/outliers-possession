# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# Round 6 environment (recipe AQCSJ): round 5's, plus the stage50-v2s inputs. Source it.
# A 2s variable already set in the environment is kept (scratch smoke runs only).
source ${SCRATCH:?}/r5_work/scripts/r5_env.sh
export CA2S=${CA2S:-$R5W/ca2s} C_VAL2S=${C_VAL2S:-$R5W/cache_c2s} C_TRAIN2S=${C_TRAIN2S:-$R5W/cache_c_train2s}
export AUG2S=${AUG2S:-$R5W/aug2s} C_TRAIN2SJ=${C_TRAIN2SJ:-$R5W/cache_c_train2sj} AUG2SJ=${AUG2SJ:-$R5W/aug2sj}
export DEFECT2S=${DEFECT2S:-$R5W/defect_eval_stage50-v2s}
export OUTROOT=${SCRATCH:?}/r6_work/screen
SCORE_INPUTS2S="--cache $CA2S --dfl-cache $C_VAL2S --sc-labels $PG --c23-dev $SP/r3_out/c23_dev --c23-dfl $SP/r4_work/c23_dfl_c --c23-sc $SP/sc_fine/c23_sc --engine-cache $SP/r4_work/engine_data_raw"
# Round 6 arm A2 (recipe AQCSL: J keeping every still-ball run). Added lines only.
export C_TRAIN2SJ100=${C_TRAIN2SJ100:-$R5W/cache_c_train2sj100} AUG2SJ100=${AUG2SJ100:-$R5W/aug2sj100}
