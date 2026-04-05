#!/bin/bash
# Validate all experiment configs: build model + 1 forward pass
# Usage: bash scripts/validate_all.sh [config_dir]
# Run this BEFORE submitting to Slurm to catch errors early

CONFIG_DIR="${1:-config/experiments}"
FAILED=0
PASSED=0

for cfg in "$CONFIG_DIR"/*.yaml; do
    name=$(basename "$cfg" .yaml)
    printf "%-40s " "$name"
    output=$(python run.py -c "$cfg" --validate 2>&1)
    if echo "$output" | grep -q "Validation PASSED"; then
        echo "PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "FAILED"
        echo "$output" | grep -i "error\|exception" | tail -3 | sed 's/^/    /'
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "Results: $PASSED passed, $FAILED failed"
