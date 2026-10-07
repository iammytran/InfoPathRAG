#!/usr/bin/env bash
#
# Download the Qwen2.5 generator + decomposer checkpoints into models/.
#
# * Qwen2.5-VL-7B-Instruct  → step0 (image summaries), step5 (sub-image
#                              detection), and the answer generator.
# * Qwen2.5-7B-Instruct     → query decomposer and modality estimator
#                              (smaller substitute for the 72B model used in
#                              the paper — fits in 1 GPU).
#
# Target paths must match the constants in src/models/mllm/*.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

hf download \
    Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir "$REPO_ROOT/models/Qwen2.5-VL-7B-Instruct"

# hf download \
#     Qwen/Qwen2.5-7B-Instruct \
#     --local-dir "$REPO_ROOT/models/Qwen2.5-7B-Instruct"
