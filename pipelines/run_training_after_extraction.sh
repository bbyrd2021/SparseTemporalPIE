#!/bin/bash
# Wait for frame extraction to complete, then run full PIE training pipeline
# Usage: bash run_training_after_extraction.sh

set -e
cd /data/repos/EfficientPIE

DATA_PATH="/data/datasets/PIE"
PRETRAIN_WEIGHTS="pre_train_weights/min_loss_pretrained_model_imagenet.pth"
VERSION=8
BATCH_SIZE=32
BASE_EPOCHS=50
IL_EPOCHS=30
DEVICE="cuda:0"
LOG_DIR="./training_logs"
mkdir -p "$LOG_DIR"

# Wait for PIE extraction to finish
echo "[$(date)] Waiting for PIE frame extraction to complete..."
TOTAL_PIE_VIDEOS=$(find /data/datasets/PIE/set* -name "*.mp4" | wc -l)
echo "Total PIE videos to extract: $TOTAL_PIE_VIDEOS"

while true; do
    DONE=$(ls /data/datasets/PIE/images/*/ 2>/dev/null | grep -v "/$" | wc -l 2>/dev/null || echo 0)
    # Count video dirs across all sets
    DONE=$(find /data/datasets/PIE/images -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)
    echo "[$(date)] PIE videos extracted: $DONE/$TOTAL_PIE_VIDEOS"
    if [ "$DONE" -ge "$TOTAL_PIE_VIDEOS" ]; then
        echo "[$(date)] PIE extraction complete!"
        break
    fi
    sleep 120
done

# Phase 1: Base Training (step 0)
echo "[$(date)] === Starting PIE Base Training (step=0, $BASE_EPOCHS epochs) ==="
python scripts/efficientpie/train_EfficientPIE.py \
    --data-path "$DATA_PATH" \
    --weights "$PRETRAIN_WEIGHTS" \
    --batch_size $BATCH_SIZE \
    --epochs $BASE_EPOCHS \
    --step 0 \
    --version $VERSION \
    --device $DEVICE \
    2>&1 | tee "$LOG_DIR/step0.log"

echo "[$(date)] === Base training complete ==="

# Phase 2: IL steps 2->14
STEPS=(2 4 6 8 10 12 14)
PREV_WEIGHTS="weights_v${VERSION}/best_model_PIE_step0.pth"

for STEP in "${STEPS[@]}"; do
    echo "[$(date)] === IL Step $STEP (prev: $PREV_WEIGHTS) ==="
    python scripts/efficientpie/pie_domain_incremental_learning.py \
        --data-path "$DATA_PATH" \
        --prev_weights "$PREV_WEIGHTS" \
        --batch_size $BATCH_SIZE \
        --epochs $IL_EPOCHS \
        --step $STEP \
        --version $VERSION \
        --device $DEVICE \
        2>&1 | tee "$LOG_DIR/il_step${STEP}.log"

    PREV_WEIGHTS="weights_v${VERSION}/best_model_PIE_IL_step${STEP}_new.pth"
    echo "[$(date)] === IL Step $STEP complete ==="
done

# Evaluation
echo "[$(date)] === Running final evaluation ==="
python scripts/efficientpie/test_EfficientPIE.py \
    --data-path "$DATA_PATH" \
    --weights "weights_v${VERSION}/best_model_PIE_IL_step14_new.pth" \
    --device $DEVICE \
    2>&1 | tee "$LOG_DIR/final_eval.log"

echo "[$(date)] === Full PIE pipeline complete! Check $LOG_DIR/ for results ==="
