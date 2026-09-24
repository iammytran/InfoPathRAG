#!/usr/bin/env bash
#
# Step 0 — generate VLM-based summaries for every image with Qwen2.5-VL-7B.
# Writes artifacts/<DS>/image_summaries/dev/<image_stem>.txt.
#
# Multi-GPU + resumable: the python helper shards across visible GPUs and
# skips images whose summary .txt already exists.
#
# Usage:
#   ./step0_summarize_images.sh                                  # all benchmarks
#   ./step0_summarize_images.sh -b "MP-DocVQA"                   # subset
#   ./step0_summarize_images.sh -b "MMCoQA" --start_idx 0 --end_idx 200
#   CUDA_VISIBLE_DEVICES=0,1,2 ./step0_summarize_images.sh -b "MMCoQA"
#
# Conda env: qwen2.5 (Qwen2.5-VL-7B-Instruct).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# source "$REPO_ROOT/scripts/_lilac_preamble.sh" lilac-qwen 18000

ALL_BENCHMARKS=(MP-DocVQA SlideVQA InfoVQA MultimodalQA MMCoQA)
BENCHMARKS=("${ALL_BENCHMARKS[@]}")
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--benchmarks) IFS=' ' read -r -a BENCHMARKS <<<"$2"; shift 2;;
        --start_idx) EXTRA+=(--start_idx "$2"); shift 2;;
        --end_idx)   EXTRA+=(--end_idx "$2"); shift 2;;
        --max_tokens) EXTRA+=(--max_tokens "$2"); shift 2;;
        --num_gpus)   EXTRA+=(--num_gpus "$2"); shift 2;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

echo "[step0] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader,nounits | sed 's/^/[step0][nvidia-smi] /'

for DS in "${BENCHMARKS[@]}"; do
    echo "==> step0 $DS"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 \
       -m src.lilac.lcg_constructor.image_parser.summarize_images \
        --target_data "$DS" "${EXTRA[@]}"
done
