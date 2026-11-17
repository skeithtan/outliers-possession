# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# Round 7 environment (recipe AQCTJ: AQCSJ with feature version stage50-v2t). Sources r6_env.sh, adds the 2t inputs.
# A 2t variable already set in the environment is kept (scratch smoke runs only).
source ${SCRATCH:?}/r5_work/scripts/r6_env.sh
export CA2T=${CA2T:-$R5W/ca2t} C_VAL2T=${C_VAL2T:-$R5W/cache_c2t} C_TRAIN2T=${C_TRAIN2T:-$R5W/cache_c_train2t}
export AUG2T=${AUG2T:-$R5W/aug2t} C_TRAIN2TJ=${C_TRAIN2TJ:-$R5W/cache_c_train2tj} AUG2TJ=${AUG2TJ:-$R5W/aug2tj}
export DEFECT2T=${DEFECT2T:-$R5W/defect_eval_stage50-v2t}
export OUTROOT=${SCRATCH:?}/r7_work/screen
SCORE_INPUTS2T="--cache $CA2T --dfl-cache $C_VAL2T --sc-labels $PG --c23-dev $SP/r3_out/c23_dev --c23-dfl $SP/r4_work/c23_dfl_c --c23-sc $SP/sc_fine/c23_sc --engine-cache $SP/r4_work/engine_data_raw"
