#!/bin/bash

# datasets
DATASETS=("dtd" "eurosat" "gtsrb" "svhn")

# Base folder: if not specified, use the current directory
BASEFOLDER=${1:-$(pwd)}

for DATASET in "${DATASETS[@]}"
do
    TARGET_DIR="${BASEFOLDER}/checkpoints/ViT-B-16/${DATASET}"
    OUTPUT_PATH="${TARGET_DIR}/model.pt"

    # Create the target directory if it doesn't exist
    mkdir -p "$TARGET_DIR"

    URL="https://huggingface.co/Jackpepito/TransFusion_models/resolve/main/task_vectors/${DATASET}/model.pt"

    echo "Downloading $DATASET model into $OUTPUT_PATH"
    curl -L "$URL" --output "$OUTPUT_PATH"
done