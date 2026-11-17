# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# Round 8 arm N environment: round 7's inputs (r7_env.sh, read only) and arm N's output root.
# Source it. Every output of arm N goes under $NOUT (never into the round 7 folders).
source ${SCRATCH:?}/r5_work/scripts/r7_env.sh
export NOUT=${NOUT:-$HOME/mtah2027/r8/n}
export OUTROOT=$NOUT/screen_unused
export R7T=$SP/r7_train R7S=$SP/r7_work/screen R7M=$SP/r7_gates/members
export ENGINE_CACHE=$SP/r4_work/engine_data_raw
export MTAH_DATA_DIR=${MTAH_DATA_DIR:-$HOME/mtah2027/data}
# The training switches of recipe AQCTJ (r7_work/screen/AQCTJ/logs/launch.log, 14 Nov 05:03).
FT_SW="--target raw --mix sc_quarter --views raw --sc-alive-dead --cache $C_TRAIN2TJ --val-cache $C_VAL2T --aug-cache $AUG2TJ --dfl-still removed --feature-version stage50-v2t"
export C23_DEV_DIR=$SP/r3_out/c23_dev C23_DFL_DIR=$SP/r4_work/c23_dfl_c C23_SC_DIR=$SP/sc_fine/c23_sc
