#!/bin/bash
# Sync experiment results from cluster to local.
# Usage:
#   bash scripts/sync_results.sh          # sync all results (no .pth)
#   bash scripts/sync_results.sh --full   # sync everything including .pth models

CLUSTER="yizhouc3@mind.cs.cmu.edu"
REMOTE_DIR="~/xor/exp/"
LOCAL_DIR="./exp/"

if [[ "$1" == "--full" ]]; then
    echo "Full sync (including .pth model weights)..."
    rsync -avz --progress \
        "$CLUSTER:$REMOTE_DIR" "$LOCAL_DIR"
else
    echo "Syncing results only (no .pth)..."
    rsync -avz --progress \
        --include='*/' \
        --include='*.p' \
        --include='*.yaml' \
        --include='*.txt' \
        --include='COMPLETED' \
        --include='*_DONE' \
        --exclude='*.pth' \
        --exclude='in2cells.p' \
        "$CLUSTER:$REMOTE_DIR" "$LOCAL_DIR"
fi

echo "Done. Local exp/ size: $(du -sh ./exp/ | cut -f1)"
