#!/bin/bash
# Train Transformer embed_dim=48 with old+young (90.43% F1)
# Usage: ./scripts/best_exps/run_transformer_e48.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="results/transformer_e48_${TIMESTAMP}"
CONFIG="config/smartfallmm/transformer_e48_old_young.yaml"

echo "=============================================="
echo "Transformer e48 (Old+Young Training)"
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
    echo "Expected: Test F1 ~90.43%"
fi

echo "Output saved to: ${WORK_DIR}"
