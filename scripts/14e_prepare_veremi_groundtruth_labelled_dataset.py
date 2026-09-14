import json
import math
import re
from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# TrustV2X - Step 14e
# Prepare Ground-Truth-Labelled VeReMi External Benchmark
# ============================================================
#
# Why this script exists:
# - VeReMi ground truth does not provide a simple label column.
# - Ground-truth type-4 records provide the true vehicle state.
# - Trace type-3 records provide received/claimed V2X messages.
# - We label each type-3 record by comparing it with its matching
#   type-4 ground-truth record using messageID.
#
# Output:
# data/veremi/processed/veremi_extension_trustv2x_external_benchmark.csv
#
# This does NOT modify the original SUMO 4x4 TrustV2X experiment.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXTRACTED_DIR = PROJECT_ROOT / "data" / "veremi" / "extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "veremi" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "veremi"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = PROCESSED_DIR / "veremi_extension_trustv2x_external_benchmark.csv"
SUMMARY_CSV = RESULTS_DIR / "veremi_groundtruth_labelled_preparation_summary.csv"

# Keep this manageable first. Increase later for a larger external benchmark.
MAX_ROWS_PER_ARCHIVE = 50_000
MAX_ROWS_TOTAL = 250_000

# Physical-error thresholds used to create external benchmark labels.
POSITION_ERROR_THRESHOLD = 20.0
SPEED_ERROR_THRESHOLD = 8.0
ACCELERATION_ERROR_THRESHOLD = 8.0
HEADING_ERROR_THRESHOLD_DEGREES = 45.0


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


def vector_magnitude(value):
    x, y, z = vector3(value)
    return math.sqrt(x * x + y * y + z * z)


def vector_distance(a, b):
    ax, ay, az = vector3(a)
    bx, by, bz = vector3(b)

    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def heading_degrees_from_vector(hed):
    hx, hy, _ = vector3(hed)

    if abs(hx) < 1e-12 and abs(hy) < 1e-12:
        return 0.0

    angle = math.degrees(math.atan2(hy, hx))

    if angle < 0:
        angle += 360.0

    return angle


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


def attack_name_from_archive(archive_name, label):
    if label == 0:
        return "normal"

    archive_lower = archive_name.lower()

    if "constpos" in archive_lower:
        return "constant_position"
    if "constspeed" in archive_lower:
        return "constant_speed"
    if "datareplay" in archive_lower:
        return "replay_attack"
    if "dosdisruptivesybil" in archive_lower:
        return "dos_sybil_attack"
    if "dosdisruptive" in archive_lower:
        return "dos_attack"

    return "attack"


def receiver_from_filename(path):
    # Example:
    # traceJSON-10005-10003-A0-28468-7.json
    # We keep the file stem as receiver/file-level context.
    return path.stem


def archive_roots():
    return [p for p in sorted(EXTRACTED_DIR.iterdir()) if p.is_dir()]


def is_groundtruth_file(path):
    return path.name.lower().startswith("tracegroundtruthjson") and path.suffix.lower() == ".json"


def is_trace_file(path):
    name = path.name.lower()
    return name.startswith("tracejson") and "groundtruth" not in name and path.suffix.lower() == ".json"


def build_groundtruth_map(root):
    gt_map = {}

    gt_files = [p for p in root.rglob("*.json") if is_groundtruth_file(p)]

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


def create_label_from_errors(position_error, speed_error, acceleration_error, heading_error):
    return int(
        position_error > POSITION_ERROR_THRESHOLD
        or speed_error > SPEED_ERROR_THRESHOLD
        or acceleration_error > ACCELERATION_ERROR_THRESHOLD
        or heading_error > HEADING_ERROR_THRESHOLD_DEGREES
    )


def convert_trace_record_to_row(rec, gt, archive_name, trace_file):
    pos_x, pos_y, pos_z = vector3(rec.get("pos"))
    spd_x, spd_y, spd_z = vector3(rec.get("spd"))
    acl_x, acl_y, acl_z = vector3(rec.get("acl"))

    gt_pos_x, gt_pos_y, gt_pos_z = vector3(gt.get("pos"))
    gt_spd_x, gt_spd_y, gt_spd_z = vector3(gt.get("spd"))
    gt_acl_x, gt_acl_y, gt_acl_z = vector3(gt.get("acl"))

    position_error = vector_distance(rec.get("pos"), gt.get("pos"))
    speed_error = vector_distance(rec.get("spd"), gt.get("spd"))
    acceleration_error = vector_distance(rec.get("acl"), gt.get("acl"))
    heading_error = heading_angle_error(rec.get("hed"), gt.get("hed"))

    label = create_label_from_errors(
        position_error=position_error,
        speed_error=speed_error,
        acceleration_error=acceleration_error,
        heading_error=heading_error,
    )

    attack_token = parse_attack_token_from_filename(trace_file)
    label_by_token = int(attack_token != "A0")

    return {
        "source_dataset": "VeReMi_Extension",
        "source_archive": archive_name,
        "source_file": trace_file.name,
        "message_type": str(rec.get("type")),
        "receiver_id": receiver_from_filename(trace_file),

        "vehicle_id": str(rec.get("sender")),
        "sender": str(rec.get("sender")),
        "senderPseudo": str(rec.get("senderPseudo")),
        "message_id": str(rec.get("messageID")),
        "attack_token": attack_token,

        "timestamp": float(rec.get("rcvTime", rec.get("sendTime", 0.0))),
        "rcvTime": float(rec.get("rcvTime", 0.0)),
        "sendTime": float(rec.get("sendTime", 0.0)),

        "x_position": pos_x,
        "y_position": pos_y,
        "z_position": pos_z,
        "speed_x": spd_x,
        "speed_y": spd_y,
        "speed_z": spd_z,
        "speed": vector_magnitude(rec.get("spd")),
        "acceleration_x": acl_x,
        "acceleration_y": acl_y,
        "acceleration_z": acl_z,
        "acceleration": vector_magnitude(rec.get("acl")),
        "heading": heading_degrees_from_vector(rec.get("hed")),

        # Ground-truth values are included for audit/analysis only.
        # Script 15 does not use them as model input features.
        "gt_x_position": gt_pos_x,
        "gt_y_position": gt_pos_y,
        "gt_z_position": gt_pos_z,
        "gt_speed_x": gt_spd_x,
        "gt_speed_y": gt_spd_y,
        "gt_speed_z": gt_spd_z,
        "gt_speed": vector_magnitude(gt.get("spd")),
        "gt_acceleration_x": gt_acl_x,
        "gt_acceleration_y": gt_acl_y,
        "gt_acceleration_z": gt_acl_z,
        "gt_acceleration": vector_magnitude(gt.get("acl")),
        "gt_heading": heading_degrees_from_vector(gt.get("hed")),

        # Labelling/audit columns.
        # Do not use these as model input features.
        "position_error": position_error,
        "speed_error": speed_error,
        "acceleration_error": acceleration_error,
        "heading_error_degrees": heading_error,
        "label_by_token": label_by_token,

        "attack_type": attack_name_from_archive(archive_name, label),
        "label": label,
    }


def process_archive(root):
    archive_name = root.name

    print("\n" + "=" * 80)
    print(f"[PROCESS] {archive_name}")
    print("=" * 80)

    gt_map, gt_file_count = build_groundtruth_map(root)
    trace_files = [p for p in root.rglob("*.json") if is_trace_file(p)]

    print(f"[INFO] Ground-truth files: {gt_file_count}")
    print(f"[INFO] Ground-truth messageID entries: {len(gt_map):,}")
    print(f"[INFO] Trace files: {len(trace_files):,}")

    rows = []
    counters = Counter()
    label_counter = Counter()
    token_label_counter = Counter()
    attack_counter = Counter()

    for trace_file in trace_files:
        if len(rows) >= MAX_ROWS_PER_ARCHIVE:
            break

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

            row = convert_trace_record_to_row(
                rec=rec,
                gt=gt,
                archive_name=archive_name,
                trace_file=trace_file,
            )

            rows.append(row)

            label_counter[row["label"]] += 1
            token_label_counter[row["label_by_token"]] += 1
            attack_counter[row["attack_type"]] += 1

            if len(rows) >= MAX_ROWS_PER_ARCHIVE:
                break

    print(f"[INFO] Type-3 records seen: {counters['type3_seen']:,}")
    print(f"[INFO] Matched to ground truth: {counters['matched_groundtruth']:,}")
    print(f"[INFO] No ground-truth match: {counters['no_groundtruth_match']:,}")
    print(f"[INFO] Missing messageID: {counters['missing_message_id']:,}")
    print(f"[INFO] Records kept: {len(rows):,}")
    print(f"[INFO] Label distribution: {dict(label_counter)}")
    print(f"[INFO] Token-label distribution: {dict(token_label_counter)}")
    print(f"[INFO] Attack distribution: {dict(attack_counter)}")

    summary = {
        "archive": archive_name,
        "groundtruth_files": gt_file_count,
        "groundtruth_message_id_entries": len(gt_map),
        "trace_files": len(trace_files),
        "type3_seen": counters["type3_seen"],
        "matched_groundtruth": counters["matched_groundtruth"],
        "no_groundtruth_match": counters["no_groundtruth_match"],
        "missing_message_id": counters["missing_message_id"],
        "records_kept": len(rows),
        "normal_records": label_counter[0],
        "attack_records": label_counter[1],
        "token_normal_records": token_label_counter[0],
        "token_attack_records": token_label_counter[1],
        "attack_counts": dict(attack_counter),
    }

    return rows, summary


def main():
    print("=" * 80)
    print("TrustV2X VeReMi Ground-Truth-Labelled Dataset Preparation")
    print("=" * 80)

    all_rows = []
    summaries = []

    for root in archive_roots():
        rows, summary = process_archive(root)
        all_rows.extend(rows)
        summaries.append(summary)

        if len(all_rows) >= MAX_ROWS_TOTAL:
            print(f"[INFO] Reached MAX_ROWS_TOTAL={MAX_ROWS_TOTAL:,}. Stopping.")
            break

    if not all_rows:
        print("[ERROR] No rows created. Check extraction and JSON structure.")
        return

    df = pd.DataFrame(all_rows)

    # Shuffle final dataset for later train/test splitting.
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame(summaries).to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 80)
    print("[OK] VeReMi external benchmark CSV created")
    print("=" * 80)
    print(f"[OK] Output CSV: {OUTPUT_CSV}")
    print(f"[OK] Summary CSV: {SUMMARY_CSV}")

    print("\nDataset shape:")
    print(df.shape)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nAttack type distribution:")
    print(df["attack_type"].value_counts())

    print("\nLabel by source archive:")
    print(pd.crosstab(df["source_archive"], df["label"]))

    print("\nMean physical errors by archive:")
    print(
        df.groupby("source_archive")[
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

    print("\nNow run:")
    print("python scripts/15_train_validate_veremi_external_benchmark.py")


if __name__ == "__main__":
    main()