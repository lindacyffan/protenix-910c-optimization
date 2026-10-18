#!/usr/bin/env bash
set -Eeo pipefail

readonly PROTENIX_ROOT="${PROTENIX_ROOT:?Set PROTENIX_ROOT to the Protenix checkout}"
readonly NPU_ID="${NPU_ID:-8}"
readonly OUTPUT_NAME="${OUTPUT_NAME:-output_910c_same_msa_fp32_tf32off_pse}"
readonly LOG_FILE="${LOG_FILE:-${OUTPUT_NAME}.log}"

source /usr/local/Ascend/cann-8.5.0/bin/setenv.bash

export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}"
export CUDA_VISIBLE_DEVICES="${NPU_ID}"
export TE_PARALLEL_COMPILER=0
export CANN_JIT_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export PROTENIX_DETERMINISTIC=true
export STATIC_CACHE_DISABLE=1
export AMP_MIXED_DISABLE=1
export APB_PSE_DISABLE=0
export APB_PSE_BROADCAST=copy
export SDPA_DISABLE=1
export SDPA_DEBUG=0
export PSE_SPLIT_MODE=pse_native
export TM_PATCH_DISABLE=0
export TA_PROJECTION_DISABLE=1
export TA_PROJECTION_PROF=0
export TA_PROJECTION_CONTIGUOUS=0
export FG_TRIATT_ENABLE=0
export FG_TRIATT_DISABLE=1
export FG_TRIATT_PRECISION=ieee
export TRIATT_KERNEL=torch
export TRIMUL_KERNEL=torch
export TRIATT_NPU_PATCH_DISABLE=0
export TRIATT_NPU_PATCH_MASK_ONLY=0
export TRIATT_NPU_LOOP=1
export TRIATT_NPU_BLOCK_M=32
export TRIATT_NPU_BLOCK_N=32
export PAIR_PROF_ENABLE=0
export FLAGOS_ENABLE_FLAGGEMS=0

cd "${PROTENIX_ROOT}"
python3 flagos_910c_launcher.py pred \
  -i examples/example_7wux_same_msa_910c.json \
  -o "${OUTPUT_NAME}" \
  -n protenix_base_default_v1.0.0 \
  -s 101 \
  -c 10 \
  -p 200 \
  -e 1 \
  -d fp32 \
  --use_msa true \
  --trimul_kernel torch \
  --triatt_kernel torch \
  --enable_cache false \
  --enable_fusion true \
  --enable_tf32 false \
  2>&1 | tee "${LOG_FILE}"
