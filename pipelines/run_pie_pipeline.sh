#!/bin/bash
# EfficientPIE Full PIE Training Pipeline
# Phase 1: Base training (step 0) with ImageNet pre-trained weights
# Phase 2: Incremental learning steps 2, 4, 6, 8, 10, 12, 14

set -e

DATA_PATH="/data/datasets/PIE"
PRETRAIN_WEIGHTS="pre_train_weights/min_loss_pretrained_model_imagenet.pth"
VERSION=8
BATCH_SIZE=32
BASE_EPOCHS=50
IL_EPOCHS=30
DEVICE="cuda:0"

echo "=== Phase 1: Base Training (step 0) ==="
python scripts/efficientpie/train_EfficientPIE.py \
    --data-path "$DATA_PATH" \
    --weights "$PRETRAIN_WEIGHTS" \
    --batch_size $BATCH_SIZE \
    --epochs $BASE_EPOCHS \
    --step 0 \
    --version $VERSION \
    --device $DEVICE

# Phase 2: Incremental Learning
STEPS=(2 4 6 8 10 12 14)
PREV_WEIGHTS="weights_v${VERSION}/best_model_PIE_step0.pth"

for STEP in "${STEPS[@]}"; do
    echo "=== Phase 2: IL Step $STEP ==="
    python scripts/efficientpie/pie_domain_incremental_learning.py \
        --data-path "$DATA_PATH" \
        --prev_weights "$PREV_WEIGHTS" \
        --batch_size $BATCH_SIZE \
        --epochs $IL_EPOCHS \
        --step $STEP \
        --version $VERSION \
        --device $DEVICE

    PREV_WEIGHTS="weights_v${VERSION}/best_model_PIE_IL_step${STEP}_new.pth"
done

echo "=== Phase 3: Evaluation ==="
python scripts/efficientpie/test_EfficientPIE.py \
    --data-path "$DATA_PATH" \
    --weights "weights_v${VERSION}/best_model_PIE_IL_step14_new.pth" \
    --device $DEVICE

echo "=== PIE Pipeline Complete ==="
