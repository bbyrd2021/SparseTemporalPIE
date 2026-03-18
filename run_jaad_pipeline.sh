#!/bin/bash
# EfficientPIE Full JAAD Training Pipeline
# Phase 1: Base training with ImageNet pre-trained weights
# Phase 2: Incremental learning steps 2, 4, 6, 8, 10, 12, 14

set -e

DATA_PATH="/data/datasets/JAAD"
PRETRAIN_WEIGHTS="pre_train_weights/min_loss_pretrained_model_imagenet.pth"
VERSION=8
BATCH_SIZE=32
BASE_EPOCHS=50
IL_EPOCHS=30
DEVICE="cuda:0"

echo "=== Phase 1: Base Training (JAAD) ==="
python train_EfficientPIE_JAAD.py \
    --data-path "$DATA_PATH" \
    --weights "$PRETRAIN_WEIGHTS" \
    --batch_size $BATCH_SIZE \
    --epochs $BASE_EPOCHS \
    --device $DEVICE

# Phase 2: Incremental Learning
STEPS=(2 4 6 8 10 12 14)
PREV_WEIGHTS="weights/transfer_best_model_JAAD.pth"

for STEP in "${STEPS[@]}"; do
    echo "=== Phase 2: IL Step $STEP ==="
    python jaad_domain_incremental_learning.py \
        --data-path "$DATA_PATH" \
        --prev_weights "$PREV_WEIGHTS" \
        --batch_size $BATCH_SIZE \
        --epochs $IL_EPOCHS \
        --step $STEP \
        --version $VERSION \
        --device $DEVICE

    PREV_WEIGHTS="weights_v${VERSION}/best_model_JAAD_IL_step${STEP}_new.pth"
done

echo "=== Phase 3: Evaluation ==="
python test_EfficientPIE_JAAD.py \
    --data-path "$DATA_PATH" \
    --weights "weights_v${VERSION}/best_model_JAAD_IL_step14_new.pth" \
    --device $DEVICE

echo "=== JAAD Pipeline Complete ==="
