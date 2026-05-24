#!/usr/bin/env bash

find . -type d -name "__pycache__" -exec rm -rf {} +

set -e
cd "$(dirname "$0")/.."

rm -rf outputs/Environment_multi
uv run python -m src.main \
    --dataset Environment \
    --io-pairs "96,48;96,96;96,192;96,336"

find . -type d -name "__pycache__" -exec rm -rf {} +
