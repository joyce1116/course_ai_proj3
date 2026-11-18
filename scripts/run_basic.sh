#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
GPU_ID="${GPU_ID:-2}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -u run.py train --config config/basic.json --gpu "$GPU_ID" "$@"
