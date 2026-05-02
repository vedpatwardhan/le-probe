#!/bin/bash
# 🚀 SAE BATCH SWEEP ENGINE
# Role: Trains an SAE for every layer in the model automatically.

# 1. Configuration
ACTIVATIONS_DIR="activations_granular"
OUTPUT_DIR="sae_weights"
DICT_SIZE=4096
EPOCHS=10
MODEL_CKPT="gr1_reward_tuned_v2.ckpt"

mkdir -p $OUTPUT_DIR

# 2. Layer List (The Full Stack)
ENCODER_LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11)
PREDICTOR_LAYERS=(0 1 2 3 4 5)

echo "🔥 Starting Full-Stack SAE Sweep..."

# --- ENCODER SWEEP ---
# Layer 0 is an SAE (Dictionary)
echo "⚙️ Training SAE for encoder_L0..."
python train_transcoder.py \
    --dir $ACTIVATIONS_DIR \
    --source_layer encoder_L0 \
    --target_layer encoder_L0 \
    --output "$OUTPUT_DIR/encoder_L0_sae.pt" \
    --dict_size $DICT_SIZE \
    --epochs $EPOCHS

# Layers 1-11 are CLTs (Transitions)
for i in {1..11}; do
    PREV=$((i-1))
    SRC="encoder_L$PREV"
    TGT="encoder_L$i"
    echo "⚙️ Training CLT for $SRC ⮕ $TGT..."
    python train_transcoder.py \
        --dir $ACTIVATIONS_DIR \
        --source_layer $SRC \
        --target_layer $TGT \
        --output "$OUTPUT_DIR/${TGT}_clt.pt" \
        --dict_size $DICT_SIZE \
        --epochs $EPOCHS
done

# --- THE BRIDGE: Encoder -> Predictor ---
echo "⚙️ Training BRIDGE CLT: encoder_L11 ⮕ predictor_L0..."
python train_transcoder.py \
    --dir $ACTIVATIONS_DIR \
    --source_layer encoder_L11 \
    --target_layer predictor_L0 \
    --output "$OUTPUT_DIR/predictor_L0_clt.pt" \
    --dict_size $DICT_SIZE \
    --epochs $EPOCHS

# --- PREDICTOR SWEEP ---
for i in {1..5}; do
    PREV=$((i-1))
    SRC="predictor_L$PREV"
    TGT="predictor_L$i"
    echo "⚙️ Training CLT for $SRC ⮕ $TGT..."
    python train_transcoder.py \
        --dir $ACTIVATIONS_DIR \
        --source_layer $SRC \
        --target_layer $TGT \
        --output "$OUTPUT_DIR/${TGT}_clt.pt" \
        --dict_size $DICT_SIZE \
        --epochs $EPOCHS
done

echo "✨ Full-Stack Layered Sweep Complete! Weights stored in $OUTPUT_DIR"
