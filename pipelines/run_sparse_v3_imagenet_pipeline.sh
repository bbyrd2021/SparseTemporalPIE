#!/usr/bin/env bash
# run_sparse_v3_imagenet_pipeline.sh
# Full SparseTemporalPIE v3 training pipeline from ImageNet pretrained weights.
# Step 0 base training (70 epochs) → IL steps 2→14 (30 epochs each).
# Run from repo root: bash pipelines/run_sparse_v3_imagenet_pipeline.sh

set -e

WEIGHTS_IMAGENET="pre_train_weights/min_loss_pretrained_model_imagenet.pth"
OUTPUT_DIR="weights_sparse_v3_imagenet"
KEYPOINTS_DIR="/data/datasets/PIE/keypoints_pid"
DATA_PATH="/data/datasets/PIE"
DEVICE="cuda:0"

mkdir -p "$OUTPUT_DIR"
mkdir -p training_logs_sparse_v3_imagenet

echo "========================================================"
echo "  SparseTemporalPIE v3 — ImageNet init pipeline"
echo "  $(date)"
echo "========================================================"

echo ""
echo "--- Step 0: Base training (70 epochs) ---"
python scripts/sparsetemporalpie/train_SparseTemporalPIE_v3.py \
    --weights       "$WEIGHTS_IMAGENET" \
    --output-dir    "$OUTPUT_DIR" \
    --keypoints-dir "$KEYPOINTS_DIR" \
    --data-path     "$DATA_PATH" \
    --epochs        70 \
    --batch-size    16 \
    --lr            1e-4 \
    --weight-decay  1e-4 \
    --restart-period 10 \
    --device        "$DEVICE" \
    2>&1 | tee training_logs_sparse_v3_imagenet/step0.log

echo ""
echo "--- IL Steps 2→14 (30 epochs each) ---"
python scripts/sparsetemporalpie/pie_sparse_incremental_learning_v3.py \
    --weights       "$OUTPUT_DIR/best_sparse_v3_step0.pth" \
    --output-dir    "$OUTPUT_DIR" \
    --keypoints-dir "$KEYPOINTS_DIR" \
    --data-path     "$DATA_PATH" \
    --epochs        30 \
    --batch-size    16 \
    --lr            1e-4 \
    --weight-decay  1e-4 \
    --restart-period 7 \
    --start-step    2 \
    --device        "$DEVICE" \
    2>&1 | tee training_logs_sparse_v3_imagenet/il_steps.log

echo ""
echo "========================================================"
echo "  Pipeline complete: $(date)"
echo "  Weights in: $OUTPUT_DIR"
echo "========================================================"
