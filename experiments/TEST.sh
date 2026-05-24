#!/usr/bin/env bash

find . -type d -name "__pycache__" -exec rm -rf {} +

set -e
cd "$(dirname "$0")/.."

rm -rf outputs/TEST_multi
uv run python -m src.main \
    --dataset TEST \
    --io-pairs "8,6;8,8" \
    --max-time 5

find . -type d -name "__pycache__" -exec rm -rf {} +
