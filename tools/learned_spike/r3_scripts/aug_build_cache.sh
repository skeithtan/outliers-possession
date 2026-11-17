#!/bin/bash
# SCRATCH and TORCH_SCRATCH name the local scratch folders of the run (outside the repository).
# aug views, 2 processes x 2 threads (commit 26d5ba2 + working tree)
cd ${SCRATCH:?}/r3aug
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
nice -n 10 ${TORCH_SCRATCH:?}/venv-torch/bin/python -m tools.learned_spike.aug_build --out ${SCRATCH:?}/r3aug_cache --labels-cache ${TORCH_SCRATCH:?}/train2/cache --engine-cache ${TORCH_SCRATCH:?}/train2/ext/views/raw/engine_data --k 16 --games METRICA-S1 METRICA-S3 DFL-J03WN1 DFL-J03WOY SC-1874553 SC-1899585 SC-1927964 SC-1959846 SC-1996435 SC-2006229 SC-2007448 SC-2010085 SC-2013725 SC-2016236  > ${SCRATCH:?}/r3aug_cache/logs/build_a.log 2>&1 &
nice -n 10 ${TORCH_SCRATCH:?}/venv-torch/bin/python -m tools.learned_spike.aug_build --out ${SCRATCH:?}/r3aug_cache --labels-cache ${TORCH_SCRATCH:?}/train2/cache --engine-cache ${TORCH_SCRATCH:?}/train2/ext/views/raw/engine_data --k 16 --games METRICA-S2 DFL-J03WMX DFL-J03WOH DFL-J03WPY SC-1886347 SC-1925299 SC-1953632 SC-1986691 SC-1996436 SC-2006363 SC-2007721 SC-2011166 SC-2015213 SC-2017461  > ${SCRATCH:?}/r3aug_cache/logs/build_b.log 2>&1 &
wait
${TORCH_SCRATCH:?}/venv-torch/bin/python -m tools.learned_spike.aug_build --out ${SCRATCH:?}/r3aug_cache --labels-cache ${TORCH_SCRATCH:?}/train2/cache --engine-cache ${TORCH_SCRATCH:?}/train2/ext/views/raw/engine_data --k 16 --manifest > ${SCRATCH:?}/r3aug_cache/logs/manifest.log 2>&1
echo done > ${SCRATCH:?}/r3aug_cache/BUILD_DONE
