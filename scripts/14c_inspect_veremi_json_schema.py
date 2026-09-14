import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXTRACTED_DIR = PROJECT_ROOT / "data" / "veremi" / "extracted"
RESULTS_DIR = PROJECT_ROOT / "results" / "veremi"
REPORT_FILE = RESULTS_DIR / "veremi_json_schema_inspection.txt"


MAX_FILES_TO_SCAN = 50
MAX_RECORDS_PER_FILE = 200
MAX_EXAMPLES_PER_TYPE = 3


def iter_json_records(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.endswith(","):
                    line = line[:-1]

                try:
                    obj = json.loads(line)

                    if isinstance(obj, dict):
                        yield obj
                    elif isinstance(obj, list):
                        for item in obj:
                            if isinstance(item, dict):
                                yield item

                except Exception:
                    continue

    except Exception as e:
        print(f"[WARNING] Could not read {path}: {e}")


def short_value(value, limit=300):
    text = repr(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def flatten_keys(obj, prefix=""):
    keys = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else str(k)
            keys.append(full_key)

            if isinstance(v, dict):
                keys.extend(flatten_keys(v, full_key))

    return keys


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("VeReMi JSON Schema Inspection")
    lines.append("=" * 80)

    json_files = sorted(EXTRACTED_DIR.rglob("*.json"))

    trace_gt_files = [p for p in json_files if "groundtruth" in p.name.lower()]
    trace_files = [
        p for p in json_files
        if "groundtruth" not in p.name.lower()
        and p.name.lower().startswith("tracejson")
    ]

    lines.append(f"Total JSON files: {len(json_files)}")
    lines.append(f"Ground-truth JSON files: {len(trace_gt_files)}")
    lines.append(f"Trace JSON files: {len(trace_files)}")

    type_counter = Counter()
    key_counter = Counter()
    flat_key_counter = Counter()
    examples_by_type = {}

    # Inspect ground truth first.
    lines.append("\n" + "=" * 80)
    lines.append("GROUND TRUTH EXAMPLES")
    lines.append("=" * 80)

    for gt_file in trace_gt_files[:3]:
        lines.append(f"\n[GROUND TRUTH FILE] {gt_file.relative_to(PROJECT_ROOT)}")

        count = 0
        for rec in iter_json_records(gt_file):
            lines.append(short_value(rec, limit=1200))
            count += 1
            if count >= 5:
                break

    # Inspect trace JSON records.
    lines.append("\n" + "=" * 80)
    lines.append("TRACE JSON RECORD EXAMPLES")
    lines.append("=" * 80)

    scanned_files = 0

    for path in trace_files:
        scanned_files += 1

        for i, rec in enumerate(iter_json_records(path)):
            if i >= MAX_RECORDS_PER_FILE:
                break

            rec_type = str(rec.get("type", "NO_TYPE"))
            type_counter[rec_type] += 1

            for k in rec.keys():
                key_counter[k] += 1

            for fk in flatten_keys(rec):
                flat_key_counter[fk] += 1

            if rec_type not in examples_by_type:
                examples_by_type[rec_type] = []

            if len(examples_by_type[rec_type]) < MAX_EXAMPLES_PER_TYPE:
                examples_by_type[rec_type].append((path, rec))

        if scanned_files >= MAX_FILES_TO_SCAN:
            break

    lines.append(f"\nScanned trace files: {scanned_files}")
    lines.append("\nRecord type counts from sample:")
    for t, c in type_counter.most_common():
        lines.append(f"type={t}: {c}")

    lines.append("\nTop-level keys:")
    for k, c in key_counter.most_common(50):
        lines.append(f"{k}: {c}")

    lines.append("\nFlattened/nested keys:")
    for k, c in flat_key_counter.most_common(100):
        lines.append(f"{k}: {c}")

    for rec_type, examples in examples_by_type.items():
        lines.append("\n" + "=" * 80)
        lines.append(f"EXAMPLES FOR RECORD TYPE {rec_type}")
        lines.append("=" * 80)

        for path, rec in examples:
            lines.append(f"\n[FILE] {path.relative_to(PROJECT_ROOT)}")
            lines.append(short_value(rec, limit=2000))

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\n" + "=" * 80)
    print(f"[OK] Schema report saved to: {REPORT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()