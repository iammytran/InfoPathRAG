import json
import re
from pathlib import Path


_FACT_PAIR_PATTERN = re.compile(
    r'"ocr"\s*:\s*"((?:\\.|[^"\\])*)"\s*,\s*'
    r'"fact"\s*:\s*"((?:\\.|[^"\\])*)"'
)


def _decode_json_string(value: str) -> str:
    """Decode a JSON string value without requiring the surrounding JSON to be valid."""
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        # Keep recoverable text if a malformed escape occurs in an otherwise complete pair.
        return value.replace(r"\"", '"').replace(r"\\", "\\")


def extract_partial_facts_from_broken_json(raw_text: str) -> list[dict[str, str]]:
    """Extract complete ``ocr``/``fact`` pairs from valid or truncated JSON text."""
    facts: list[dict[str, str]] = []
    for ocr_value, fact_value in _FACT_PAIR_PATTERN.findall(raw_text):
        facts.append({
            "ocr": _decode_json_string(ocr_value),
            "fact": _decode_json_string(fact_value),
        })
    return facts


def deduplicate_facts(facts: list[str]) -> list[str]:
    """Remove exact duplicate facts while preserving their original order."""
    unique_facts: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        if fact not in seen:
            seen.add(fact)
            unique_facts.append(fact)
    return unique_facts


def process_raw_folder(
    raw_folder: str | Path,
    output_folder: str | Path,
) -> dict[str, list[str]]:
    """Process every JSON file and save each result as ``{"facts": [...]}``."""
    raw_path = Path(raw_folder)
    output_path = Path(output_folder)
    if not raw_path.is_dir():
        raise NotADirectoryError(f"Raw folder does not exist: {raw_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    processed: dict[str, list[str]] = {}
    for input_path in sorted(raw_path.glob("*.json")):
        try:
            raw_text = input_path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            print(f"Could not read {input_path}: {error}")
            continue

        extracted_facts = extract_partial_facts_from_broken_json(raw_text)
        facts = deduplicate_facts([item["fact"] for item in extracted_facts])
        processed[input_path.name] = facts
        output_path.joinpath(input_path.name).write_text(
            json.dumps({"facts": facts}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return processed

# # --- Thử nghiệm với đoạn JSON bị cụt của bạn ---
# broken_raw = """
# [
#     {
#         "ocr": "Threats in Q1 2020",
#         "fact": "The infographic highlights cybersecurity threats observed during the first quarter of 2020."
#     },
#     {
#         "ocr": ">737",
#         "fact": "There were over 737 instances of detected malware linked to COVID-19."
#     },
#     {
#         "ocr": "TREND",
#         "fact
# """

# extracted = extract_partial_facts_from_broken_json(broken_raw)
# print(extracted)

def main():
    raw_folder = "/workspace/LILaC/artifacts/InfoVQA/facts_each_tile/raw"
    folder = "/workspace/LILaC/artifacts/InfoVQA/facts_each_tile"

    process_raw_folder(raw_folder, folder)

if __name__ == "__main__":
    main()