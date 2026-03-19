#!/bin/bash
# Full SparseTemporalPIE training pipeline for PIE dataset
#
# Prerequisites (run in order):
#   1. python scripts/preprocess/extract_frames.py --dataset pie
#   2. python scripts/preprocess/extract_keypoints.py --dataset pie --output-dir /data/datasets/PIE/keypoints_pid
#   3. bash run_sparse_pie_pipeline.sh

set -e

DATA_PATH="/data/datasets/PIE"
KEYPOINTS_DIR="/data/datasets/PIE/keypoints_pid"
WEIGHTS_DIR="weights_sparse"
LOG_DIR="training_logs_sparse"
DEVICE="cuda:0"
BACKBONE_WEIGHTS="weights_v8/model_8_PIE_IL_step14_new.pth"

mkdir -p $WEIGHTS_DIR $LOG_DIR

echo "========================================"
echo "  Step 0: Base training (50 epochs)"
echo "========================================"
python scripts/sparsetemporalpie/train_SparseTemporalPIE.py \
    --data-path       $DATA_PATH \
    --weights         $BACKBONE_WEIGHTS \
    --keypoints-dir   $KEYPOINTS_DIR \
    --output-dir      $WEIGHTS_DIR \
    --step            0 \
    --epochs          50 \
    --device          $DEVICE \
    2>&1 | tee $LOG_DIR/step0.log

echo "========================================"
echo "  Steps 2-14: Incremental learning"
echo "========================================"
python scripts/sparsetemporalpie/pie_sparse_incremental_learning.py \
    --data-path       $DATA_PATH \
    --weights         $WEIGHTS_DIR/best_sparse_step0.pth \
    --keypoints-dir   $KEYPOINTS_DIR \
    --output-dir      $WEIGHTS_DIR \
    --epochs          30 \
    --device          $DEVICE \
    2>&1 | tee $LOG_DIR/il_steps.log

echo "========================================"
echo "  Final Evaluation"
echo "========================================"
python scripts/sparsetemporalpie/test_SparseTemporalPIE.py \
    --data-path       $DATA_PATH \
    --weights         $WEIGHTS_DIR/best_sparse_step14.pth \
    --keypoints-dir   $KEYPOINTS_DIR \
    --device          $DEVICE \
    2>&1 | tee $LOG_DIR/evaluation.log

echo "Pipeline complete. Results in $LOG_DIR/evaluation.log"
