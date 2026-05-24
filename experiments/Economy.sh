#!/usr/bin/env bash

find . -type d -name "__pycache__" -exec rm -rf {} +

set -e
cd "$(dirname "$0")/.."

rm -rf outputs/Economy_multi
uv run python -m src.main \
    --dataset Economy \
    --io-pairs "8,6;8,8;8,10;8,12"

find . -type d -name "__pycache__" -exec rm -rf {} +
