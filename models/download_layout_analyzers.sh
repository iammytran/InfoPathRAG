#!/usr/bin/env bash
#
# Download checkpoints for the two supported layout analyzers used as an
# optional preprocessing step (see src/lilac/lcg_constructor/preprocessing/).
#
# 1) DocLayout-YOLO  (lightweight, layout-box detection)
#    https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench
#    Target: models/DocLayout-YOLO-DocStructBench/
#
# 2) MinerU          (heavier, full PDF/image → structured JSON pipeline)
#    https://github.com/opendatalab/mineru
#    MinerU manages its own model registry via its CLI; we install the
#    `mineru` Python package and trigger its `mineru-models-download` helper.
#    Models land under MinerU's own cache (default: ~/.cache/mineru or set
#    by env MINERU_MODEL_SOURCE / MINERU_MODELS_DIR).
#
# Usage:
#   ./models/download_layout_analyzers.sh                     # both
#   ./models/download_layout_analyzers.sh --only doclayout    # one
#   ./models/download_layout_analyzers.sh --only mineru       # one
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_ENABLE_HF_TRANSFER=1

ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2;;
        -h|--help) sed -n '2,22p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

# ── 1) DocLayout-YOLO ────────────────────────────────────────────────────────
if [[ -z "$ONLY" || "$ONLY" == "doclayout" ]]; then
    DEST="$REPO_ROOT/models/DocLayout-YOLO-DocStructBench"
    echo "==> DocLayout-YOLO  →  $DEST"
    hf download \
        juliozhao/DocLayout-YOLO-DocStructBench \
        --local-dir "$DEST"
fi

# ── 2) MinerU ────────────────────────────────────────────────────────────────
if [[ -z "$ONLY" || "$ONLY" == "mineru" ]]; then
    echo "==> MinerU"
    if ! command -v mineru-models-download >/dev/null 2>&1; then
        echo "    Installing mineru Python package (this may take a few minutes)…"
        pip install -q --upgrade "mineru[core]"
    fi
    echo "    Fetching MinerU model weights via mineru-models-download …"
    # `-s huggingface` chooses the HF source; `-m all` fetches the full set
    # (layout + OCR + formula + table). Trim to `-m layout` if you only need
    # layout analysis.
    mineru-models-download --source huggingface --tier basic
fi

echo
echo "✅  Done."