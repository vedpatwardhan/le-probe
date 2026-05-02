#!/bin/bash
# 🚀 SAE BATCH SWEEP ENGINE
# Role: Trains an SAE for every layer in the model automatically.

# 1. Configuration
ACTIVATIONS_DIR="activations_granular"
OUTPUT_DIR="transcoder_weights_granular"
DICT_SIZE=12288
EPOCHS=10

# 🚀 MODULE CONTROL (Set to false to skip sections)
TRAIN_ENCODER=true
TRAIN_BRIDGE=true
TRAIN_PREDICTOR=true

mkdir -p $OUTPUT_DIR

echo "🔥 Starting Layered SAE Sweep..."

# --- ENCODER SWEEP ---
if [ "$TRAIN_ENCODER" = true ]; then
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
fi

# --- THE BRIDGE: Encoder -> Predictor ---
if [ "$TRAIN_BRIDGE" = true ]; then
    echo "⚙️ Training BRIDGE CLT: encoder_L11 ⮕ predictor_L0..."
    python train_transcoder.py \
        --dir $ACTIVATIONS_DIR \
        --source_layer encoder_L11 \
        --target_layer predictor_L0 \
        --output "$OUTPUT_DIR/predictor_L0_clt.pt" \
        --dict_size $DICT_SIZE \
        --epochs $EPOCHS
fi

# --- PREDICTOR SWEEP (High-Fidelity) ---
if [ "$TRAIN_PREDICTOR" = true ]; then
    # The Predictor has 257x fewer tokens than the Encoder. 
    # We use a smaller batch size (512) to ensure enough gradient updates per epoch.
    for i in {1..5}; do
        PREV=$((i-1))
        SRC="predictor_L$PREV"
        TGT="predictor_L$i"
        echo "⚙️ Training High-Fidelity CLT for $SRC ⮕ $TGT..."
        python train_transcoder.py \
            --dir $ACTIVATIONS_DIR \
            --source_layer $SRC \
            --target_layer $TGT \
            --output "$OUTPUT_DIR/${TGT}_clt.pt" \
            --dict_size $DICT_SIZE \
            --batch_size 512 \
            --epochs $EPOCHS
    done
fi

echo "✨ Sweep Complete! Weights stored in $OUTPUT_DIR"
