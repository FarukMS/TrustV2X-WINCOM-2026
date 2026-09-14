import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXTRACTED_DIR = PROJECT_ROOT / "data" / "veremi" / "extracted"
RESULTS_DIR = PROJECT_ROOT / "results" / "veremi"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "veremi_label_alignment_diagnostic.csv"

MAX_TRACE_RECORDS_PER_ARCHIVE = 30000


def iter_json_records(path):
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


def vector3(value):
    if isinstance(value, list):
        x = float(value[0]) if len(value) > 0 else 0.0
        y = float(value[1]) if len(value) > 1 else 0.0
        z = float(value[2]) if len(value) > 2 else 0.0
        return x, y, z

    return 0.0, 0.0, 0.0


def vector_distance(a, b):
    ax, ay, az = vector3(a)
    bx, by, bz = vector3(b)

    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def heading_angle_error(h1, h2):
    x1, y1, z1 = vector3(h1)
    x2, y2, z2 = vector3(h2)

    mag1 = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
    mag2 = math.sqrt(x2 * x2 + y2 * y2 + z2 * z2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    dot = (x1 * x2 + y1 * y2 + z1 * z2) / (mag1 * mag2)
    dot = max(-1.0, min(1.0, dot))

    return math.degrees(math.acos(dot))


def parse_attack_token_from_filename(path):
    match = re.search(r"-A(\d+)-", path.name)

    if match:
        return f"A{match.group(1)}"

    return "unknown"


def archive_roots():
    roots = []

    for folder in sorted(EXTRACTED_DIR.iterdir()):
        if folder.is_dir():
            roots.append(folder)

    return roots


def build_groundtruth_map(root):
    gt_map = {}

    gt_files = [
        p for p in root.rglob("*.json")
        if "groundtruth" in p.name.lower()
    ]

    for gt_file in gt_files:
        print(f"[GT] {root.name}: reading {gt_file.name}")

        for rec in iter_json_records(gt_file):
            if rec.get("type") != 4:
                continue

            message_id = rec.get("messageID")

            if message_id is None:
                continue

            gt_map[str(message_id)] = rec

    return gt_map, len(gt_files)


def is_trace_file(path):
    name = path.name.lower()
    return name.startswith("tracejson") and "groundtruth" not in name and name.endswith(".json")


def diagnose_archive(root):
    archive = root.name

    print("\n" + "=" * 80)
    print(f"[ARCHIVE] {archive}")
    print("=" * 80)

    gt_map, gt_file_count = build_groundtruth_map(root)

    print(f"[INFO] Ground-truth files: {gt_file_count}")
    print(f"[INFO] Ground-truth messageID entries: {len(gt_map):,}")

    trace_files = [p for p in root.rglob("*.json") if is_trace_file(p)]

    print(f"[INFO] Trace files: {len(trace_files):,}")

    rows = []
    counters = Counter()
    token_counter = Counter()
    error_label_counter = Counter()
    token_label_counter = Counter()

    for trace_file in trace_files:
        if counters["type3_seen"] >= MAX_TRACE_RECORDS_PER_ARCHIVE:
            break

        attack_token = parse_attack_token_from_filename(trace_file)
        token_counter[attack_token] += 1

        for rec in iter_json_records(trace_file):
            if rec.get("type") != 3:
                continue

            counters["type3_seen"] += 1

            message_id = rec.get("messageID")

            if message_id is None:
                counters["missing_message_id"] += 1
                continue

            gt = gt_map.get(str(message_id))

            if gt is None:
                counters["no_groundtruth_match"] += 1
                continue

            counters["matched_groundtruth"] += 1

            pos_error = vector_distance(rec.get("pos"), gt.get("pos"))
            speed_error = vector_distance(rec.get("spd"), gt.get("spd"))
            accel_error = vector_distance(rec.get("acl"), gt.get("acl"))
            heading_error = heading_angle_error(rec.get("hed"), gt.get("hed"))

            label_by_error = int(
                pos_error > 20.0
                or speed_error > 8.0
                or accel_error > 8.0
                or heading_error > 45.0
            )

            label_by_token = int(attack_token != "A0")

            error_label_counter[label_by_error] += 1
            token_label_counter[label_by_token] += 1

            rows.append(
                {
                    "archive": archive,
                    "trace_file": trace_file.name,
                    "attack_token": attack_token,
                    "message_id": message_id,
                    "sender": rec.get("sender"),
                    "sendTime": rec.get("sendTime"),
                    "rcvTime": rec.get("rcvTime"),
                    "position_error": pos_error,
                    "speed_error": speed_error,
                    "acceleration_error": accel_error,
                    "heading_error_degrees": heading_error,
                    "label_by_error": label_by_error,
                    "label_by_token": label_by_token,
                }
            )

            if counters["type3_seen"] >= MAX_TRACE_RECORDS_PER_ARCHIVE:
                break

    print(f"[INFO] Type-3 records seen: {counters['type3_seen']:,}")
    print(f"[INFO] Matched to ground truth: {counters['matched_groundtruth']:,}")
    print(f"[INFO] No ground-truth match: {counters['no_groundtruth_match']:,}")
    print(f"[INFO] Missing messageID: {counters['missing_message_id']:,}")
    print(f"[INFO] File attack-token counts: {dict(token_counter)}")
    print(f"[INFO] Label by physical error: {dict(error_label_counter)}")
    print(f"[INFO] Label by filename token: {dict(token_label_counter)}")

    return rows


def main():
    print("=" * 80)
    print("TrustV2X VeReMi Label Alignment Diagnostic")
    print("=" * 80)

    all_rows = []

    for root in archive_roots():
        all_rows.extend(diagnose_archive(root))

    if not all_rows:
        print("[ERROR] No matched diagnostic rows were created.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 80)
    print("[OK] Diagnostic complete")
    print("=" * 80)
    print(f"[OK] Output saved to: {OUTPUT_CSV}")
    print(f"[INFO] Diagnostic shape: {df.shape}")

    print("\nOverall label_by_error:")
    print(df["label_by_error"].value_counts())

    print("\nOverall label_by_token:")
    print(df["label_by_token"].value_counts())

    print("\nMean errors by archive:")
    print(
        df.groupby("archive")[
            [
                "position_error",
                "speed_error",
                "acceleration_error",
                "heading_error_degrees",
            ]
        ]
        .mean()
        .round(4)
    )

    print("\nLabel by physical error per archive:")
    print(pd.crosstab(df["archive"], df["label_by_error"]))

    print("\nLabel by filename token per archive:")
    print(pd.crosstab(df["archive"], df["label_by_token"]))


if __name__ == "__main__":
    main()