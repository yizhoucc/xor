#!/bin/bash
# Download Penn Treebank (PTB) dataset for language modeling
# Mikolov's preprocessed version with 10K vocabulary

set -e

DATA_DIR="data/ptb"
mkdir -p "$DATA_DIR"

BASE_URL="https://raw.githubusercontent.com/wojzaremba/lstm/master/data"

echo "Downloading PTB dataset..."
for split in train valid test; do
    echo "  Downloading ptb.${split}.txt..."
    curl -sL "${BASE_URL}/ptb.${split}.txt" -o "${DATA_DIR}/${split}.txt"
done

# Verify
for split in train valid test; do
    wc -l "${DATA_DIR}/${split}.txt"
done

echo "PTB dataset downloaded to ${DATA_DIR}/"
