#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/workspace/InfoPathRAG/models/NV-Embed-v1"
CONFIG_PATHS=(
    "/workspace/InfoPathRAG/models/MM-Embed/config.json"
    "/workspace/InfoPathRAG/models/NV-Embed-v1/config.json"
)

python3 - "$MODEL_PATH" "${CONFIG_PATHS[@]}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

model_path = sys.argv[1]
config_paths = [Path(value) for value in sys.argv[2:]]

def update_fields(value):
    changed = 0
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"_name_or_path", "retriever"} and child != model_path:
                value[key] = model_path
                changed += 1
            else:
                changed += update_fields(child)
    elif isinstance(value, list):
        for child in value:
            changed += update_fields(child)
    return changed

for config_path in config_paths:
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)

    changed = update_fields(config)
    if changed == 0:
        print(f"unchanged: {config_path}")
        continue

    backup_path = config_path.with_name(config_path.name + ".bak")
    shutil.copy2(config_path, backup_path)
    with config_path.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=4, ensure_ascii=False)
        stream.write("\n")
    print(f"updated {changed} field(s): {config_path}")
    print(f"backup: {backup_path}")
PY
