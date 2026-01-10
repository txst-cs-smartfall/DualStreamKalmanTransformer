#!/bin/bash
# Train KGHF embed64 with _len_check > 1 (90.44% F1)
# Usage: ./scripts/best_exps/run_kghf_embed64_gt1.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="results/kghf_embed64_gt1_${TIMESTAMP}"
CONFIG="config/smartfallmm/kghf_embed64_gt1.yaml"

echo "=============================================="
echo "KGHF embed64 gt1 (Young Only, Strict Filter)"
echo "=============================================="
echo "Config:    ${CONFIG}"
echo "Output:    ${WORK_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "Host:      $(hostname)"
echo "=============================================="

mkdir -p ${WORK_DIR}

python main.py \
    --config ${CONFIG} \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}/train.log

echo ""
echo "=============================================="
echo "Training Complete"
echo "=============================================="

if [ -f "${WORK_DIR}/scores.csv" ]; then
    echo ""
    echo "Results:"
    grep "Average" ${WORK_DIR}/scores.csv
    echo ""
    echo "Expected: Test F1 ~90.44%"
fi

echo "Output saved to: ${WORK_DIR}"
