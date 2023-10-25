#!/bin/bash

BASE_URL="https://huggingface.co/anayebi/mental-sim-models/resolve/main/"
TARGET_DIR="./trained_models/"

# Ensure the target directory exists
mkdir -p $TARGET_DIR

# List of model files to download
FILES=(
    "VC-1+CTRNN_k700.pt"
    "VC-1+CTRNN_physion.pt"
    "VC-1+LSTM_k700.pt"
    "VC-1+LSTM_physion.pt"
    "R3M+CTRNN_k700.pt"
    "R3M+CTRNN_physion.pt"
    "R3M+LSTM_k700.pt"
    "R3M+LSTM_physion.pt"
    "FitVid_physion_64x64.pt"
    "SVG_physion_128x128.pt"
    "CSWM_large_physion.pt"
)

# Download each file
for file in "${FILES[@]}"; do
    # Replace '+' with '%2B' for URL encoding
    encoded_file=$(echo $file | sed 's/+/%2B/g')
    curl -L "$BASE_URL$encoded_file" --output "$TARGET_DIR$file"
done

echo "Download complete."
