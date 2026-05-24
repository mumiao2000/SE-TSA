#!/usr/bin/env bash

find . -type d -name "__pycache__" -exec rm -rf {} +

set -e
cd "$(dirname "$0")/.."

rm -rf outputs/PublicHealth_multi
uv run python -m src.main \
    --dataset PublicHealth \
    --io-pairs "36,12;36,24;36,36;36,48"

find . -type d -name "__pycache__" -exec rm -rf {} +
