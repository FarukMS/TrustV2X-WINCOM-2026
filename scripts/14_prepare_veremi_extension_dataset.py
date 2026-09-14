import json
import gzip
import zipfile
import math
import random
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# TrustV2X - Step 14
# Prepare VeReMi / VeReMi Extension Dataset
# ============================================================
#
# Purpose:
# - Extract VeReMi zip files from data/veremi/raw/
# - Inspect JSON/JSONlog files
# - Parse message logs and ground-truth files
# - Align messages with labels using messageID where possible
# - Convert VeReMi records into a TrustV2X-compatible CSV
#
# This script does NOT modify the original TrustV2X 4x4 experiment.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "veremi" / "raw"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "veremi" / "extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "veremi" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "veremi"

OUTPUT_CSV = PROCESSED_DIR / "veremi_extension_trustv2x_external_benchmark.csv"
SUMMARY_CSV = RESULTS_DIR / "veremi_preparation_summary.csv"

# Keep this moderate first. Increase later for full benchmark.
MAX_ROWS_TOTAL = 250_000
MAX_ROWS_PER_ARCHIVE = 60_000

# Type 3 is normally received BSM-style message data.
# If a file has no type field, records are still kept.
TARGET_MESSAGE_TYPES = {"3", 3, None, ""}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def ensure_dirs():
    for folder in [RAW_DIR, EXTRACTED_DIR, PROCESSED_DIR, RESULTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def safe_str(value):
    if value is None:
        return ""
    return str(value)


def to_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int_like(value, default=None):
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def get_first(record, keys, default=None):
    for key in keys:
        if key in record:
            return record[key]
    return default


def normalise_key(value):
    if value is None:
        return None
    return str(value).strip()


def unpack_vector(value):
    """
    Handles vector fields stored as:
    - list/tuple: [x, y, z]
    - dict: {"x":..., "y":..., "z":...}
    - scalar: treated as x only
    """
    if value is None:
        return 0.0, 0.0, 0.0

    if isinstance(value, (list, tuple)):
        x = to_float(value[0]) if len(value) > 0 else 0.0
        y = to_float(value[1]) if len(value) > 1 else 0.0
        z = to_float(value[2]) if len(value) > 2 else 0.0
        return x, y, z

    if isinstance(value, dict):
        x = to_float(get_first(value, ["x", "X", "0", 0], 0.0))
        y = to_float(get_first(value, ["y", "Y", "1", 1], 0.0))
        z = to_float(get_first(value, ["z", "Z", "2", 2], 0.0))
        return x, y, z

    return to_float(value), 0.0, 0.0


def extract_position(record):
    pos_value = get_first(
        record,
        [
            "pos",
            "position",
            "senderPos",
            "senderPosition",
            "claimedPosition",
            "claimPos",
        ],
    )

    if pos_value is not None:
        return unpack_vector(pos_value)

    return (
        to_float(get_first(record, ["pos_0", "x", "X", "x_position"], 0.0)),
        to_float(get_first(record, ["pos_1", "y", "Y", "y_position"], 0.0)),
        to_float(get_first(record, ["pos_2", "z", "Z", "z_position"], 0.0)),
    )


def extract_speed(record):
    spd_value = get_first(
        record,
        [
            "spd",
            "speed",
            "senderSpeed",
            "claimedSpeed",
            "velocity",
        ],
    )

    if spd_value is not None:
        sx, sy, sz = unpack_vector(spd_value)
    else:
        sx = to_float(get_first(record, ["spd_0", "speed_0", "vx"], 0.0))
        sy = to_float(get_first(record, ["spd_1", "speed_1", "vy"], 0.0))
        sz = to_float(get_first(record, ["spd_2", "speed_2", "vz"], 0.0))

    speed_mag = math.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    return sx, sy, sz, speed_mag


def extract_acceleration(record):
    acl_value = get_first(
        record,
        [
            "acl",
            "acceleration",
            "senderAcceleration",
            "claimedAcceleration",
        ],
    )

    if acl_value is not None:
        ax, ay, az = unpack_vector(acl_value)
    else:
        ax = to_float(get_first(record, ["acl_0", "acc_0", "ax"], 0.0))
        ay = to_float(get_first(record, ["acl_1", "acc_1", "ay"], 0.0))
        az = to_float(get_first(record, ["acl_2", "acc_2", "az"], 0.0))

    accel_mag = math.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    return ax, ay, az, accel_mag


def compute_heading_from_speed(speed_x, speed_y):
    if abs(speed_x) < 1e-9 and abs(speed_y) < 1e-9:
        return 0.0

    heading = math.degrees(math.atan2(speed_y, speed_x))

    if heading < 0:
        heading += 360.0

    return heading


def attack_value_to_label(value):
    """
    Returns:
    - 0 for benign/genuine
    - 1 for attack/misbehaviour
    - None if unknown
    """
    if value is None:
        return None

    text = str(value).strip().lower()

    if text in ["", "none", "nan", "null", "unknown"]:
        return None

    if text in ["0", "0.0", "normal", "benign", "genuine", "legitimate", "false"]:
        return 0

    return 1


def attack_name_from_source(source_name, attacker_type, label):
    if label == 0:
        return "normal"

    source_lower = source_name.lower()

    if "constpos" in source_lower or "constantposition" in source_lower:
        return "constant_position"
    if "constspeed" in source_lower:
        return "constant_speed"
    if "datareplay" in source_lower or "replay" in source_lower:
        return "replay_attack"
    if "dosdisruptive" in source_lower and "sybil" in source_lower:
        return "dos_sybil_attack"
    if "dos" in source_lower:
        return "dos_attack"
    if "sybil" in source_lower:
        return "sybil_attack"
    if "random" in source_lower:
        return "random_position"
    if "offset" in source_lower:
        return "offset_position"

    if attacker_type is not None:
        return f"attacker_type_{attacker_type}"

    return "attack"


def is_groundtruth_file(path):
    name = path.name.lower()
    return "ground" in name or "truth" in name or "gt" in name


def is_candidate_log_file(path):
    name = path.name.lower()

    if is_groundtruth_file(path):
        return False

    return (
        name.endswith(".json")
        or name.endswith(".jsonlog")
        or name.endswith(".jsonl")
        or name.endswith(".txt")
        or name.endswith(".gz")
    )


def open_text_file(path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")

    return open(path, "r", encoding="utf-8", errors="ignore")


def iter_json_records(path):
    """
    Reads JSON lines safely.
    Most VeReMi files are line-based JSON records.
    """
    try:
        with open_text_file(path) as f:
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

                except json.JSONDecodeError:
                    continue

    except Exception as e:
        print(f"[WARNING] Could not read {path}: {e}")


def extract_zip_files():
    zip_files = sorted(RAW_DIR.glob("*.zip"))

    if not zip_files:
        print("[ERROR] No .zip files found in data/veremi/raw/")
        print("Place VeReMi Extension zip files there first.")
        return []

    extracted_roots = []

    for zip_path in zip_files:
        target_dir = EXTRACTED_DIR / zip_path.stem
        extracted_roots.append(target_dir)

        if target_dir.exists() and any(target_dir.rglob("*")):
            print(f"[SKIP] Already extracted: {zip_path.name}")
            continue

        print(f"[EXTRACT] {zip_path.name} -> {target_dir}")

        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_dir)
        except Exception as e:
            print(f"[WARNING] Could not extract {zip_path.name}: {e}")

    return extracted_roots


def build_groundtruth_maps(root):
    """
    Builds lookup maps from ground-truth files.
    Primary key: messageID
    Secondary key: sender + sendTime
    """
    gt_by_message_id = {}
    gt_by_sender_time = {}
    gt_files = []

    for path in root.rglob("*"):
        if path.is_file() and is_groundtruth_file(path):
            gt_files.append(path)

    for gt_file in gt_files:
        print(f"[GT] Reading ground truth: {gt_file.name}")

        for rec in iter_json_records(gt_file):
            message_id = normalise_key(
                get_first(rec, ["messageID", "messageId", "msgID", "id", "message_id"])
            )

            sender = normalise_key(
                get_first(rec, ["sender", "senderID", "senderId", "vehicleID", "vehicleId"])
            )

            send_time = normalise_key(
                get_first(rec, ["sendTime", "time", "timestamp", "rcvTime"])
            )

            attacker_type = get_first(
                rec,
                [
                    "attackerType",
                    "AttackerType",
                    "attackType",
                    "AttackType",
                    "attacker_type",
                    "label",
                    "class",
                ],
            )

            if message_id is not None:
                gt_by_message_id[message_id] = attacker_type

            if sender is not None and send_time is not None:
                gt_by_sender_time[(sender, send_time)] = attacker_type

    return gt_by_message_id, gt_by_sender_time, len(gt_files)


def receiver_from_path(path):
    stem = path.stem

    for token in [".jsonlog", ".json", ".jsonl", ".txt"]:
        stem = stem.replace(token, "")

    return stem


def convert_record_to_row(record, archive_name, log_file, gt_by_message_id, gt_by_sender_time):
    message_type = get_first(record, ["type", "messageType", "msgType"], None)

    if message_type not in TARGET_MESSAGE_TYPES:
        return None

    message_id = normalise_key(
        get_first(record, ["messageID", "messageId", "msgID", "id", "message_id"])
    )

    sender = normalise_key(
        get_first(record, ["sender", "senderID", "senderId", "vehicleID", "vehicleId"])
    )

    send_time = normalise_key(
        get_first(record, ["sendTime", "send_time", "txTime", "timestamp"])
    )

    rcv_time = normalise_key(
        get_first(record, ["rcvTime", "receiveTime", "recvTime", "rxTime", "timestamp"])
    )

    attacker_type = get_first(
        record,
        [
            "attackerType",
            "AttackerType",
            "attackType",
            "AttackType",
            "attacker_type",
            "label",
            "class",
        ],
    )

    if attacker_type is None and message_id is not None:
        attacker_type = gt_by_message_id.get(message_id)

    if attacker_type is None and sender is not None and send_time is not None:
        attacker_type = gt_by_sender_time.get((sender, send_time))

    label = attack_value_to_label(attacker_type)

    # Keep only labelled rows for external benchmark reliability.
    if label is None:
        return None

    x, y, z = extract_position(record)
    speed_x, speed_y, speed_z, speed_mag = extract_speed(record)
    accel_x, accel_y, accel_z, accel_mag = extract_acceleration(record)

    heading = to_float(
        get_first(record, ["heading", "hed", "yaw", "direction"], None),
        default=compute_heading_from_speed(speed_x, speed_y),
    )

    timestamp = to_float(rcv_time, default=to_float(send_time, 0.0))

    rssi = to_float(get_first(record, ["rssi", "RSSI"], 0.0))

    attack_name = attack_name_from_source(archive_name, attacker_type, label)

    receiver_id = receiver_from_path(log_file)

    return {
        "source_dataset": "VeReMi_Extension",
        "source_archive": archive_name,
        "source_file": str(log_file.name),
        "message_type": safe_str(message_type),
        "receiver_id": receiver_id,
        "vehicle_id": safe_str(sender),
        "sender": safe_str(sender),
        "message_id": safe_str(message_id),
        "timestamp": timestamp,
        "rcvTime": to_float(rcv_time, 0.0),
        "sendTime": to_float(send_time, 0.0),
        "x_position": x,
        "y_position": y,
        "z_position": z,
        "speed_x": speed_x,
        "speed_y": speed_y,
        "speed_z": speed_z,
        "speed": speed_mag,
        "acceleration_x": accel_x,
        "acceleration_y": accel_y,
        "acceleration_z": accel_z,
        "acceleration": accel_mag,
        "heading": heading,
        "rssi": rssi,
        "attacker_type": safe_str(attacker_type),
        "attack_type": attack_name,
        "label": int(label),
    }


def process_archive_root(root):
    archive_name = root.name

    print("\n" + "=" * 70)
    print(f"[PROCESS] {archive_name}")
    print("=" * 70)

    gt_by_message_id, gt_by_sender_time, gt_file_count = build_groundtruth_maps(root)

    print(f"[INFO] Ground truth files: {gt_file_count}")
    print(f"[INFO] Ground truth messageID entries: {len(gt_by_message_id):,}")
    print(f"[INFO] Ground truth sender-time entries: {len(gt_by_sender_time):,}")

    log_files = [p for p in root.rglob("*") if p.is_file() and is_candidate_log_file(p)]

    print(f"[INFO] Candidate log files: {len(log_files):,}")

    archive_rows = []
    archive_counters = Counter()
    type_counter = Counter()
    label_counter = Counter()
    attack_counter = Counter()

    for log_file in log_files:
        if len(archive_rows) >= MAX_ROWS_PER_ARCHIVE:
            break

        for rec in iter_json_records(log_file):
            archive_counters["records_seen"] += 1

            msg_type = get_first(rec, ["type", "messageType", "msgType"], None)
            type_counter[safe_str(msg_type)] += 1

            row = convert_record_to_row(
                record=rec,
                archive_name=archive_name,
                log_file=log_file,
                gt_by_message_id=gt_by_message_id,
                gt_by_sender_time=gt_by_sender_time,
            )

            if row is None:
                archive_counters["records_skipped_unlabelled_or_non_target"] += 1
                continue

            archive_rows.append(row)
            label_counter[row["label"]] += 1
            attack_counter[row["attack_type"]] += 1

            if len(archive_rows) >= MAX_ROWS_PER_ARCHIVE:
                break

    summary = {
        "archive": archive_name,
        "groundtruth_files": gt_file_count,
        "groundtruth_message_id_entries": len(gt_by_message_id),
        "groundtruth_sender_time_entries": len(gt_by_sender_time),
        "candidate_log_files": len(log_files),
        "records_seen": archive_counters["records_seen"],
        "records_kept": len(archive_rows),
        "records_skipped": archive_counters["records_skipped_unlabelled_or_non_target"],
        "type_counts": dict(type_counter),
        "label_counts": dict(label_counter),
        "attack_counts": dict(attack_counter),
    }

    print(f"[INFO] Records seen: {archive_counters['records_seen']:,}")
    print(f"[INFO] Records kept: {len(archive_rows):,}")
    print(f"[INFO] Label counts: {dict(label_counter)}")
    print(f"[INFO] Attack counts: {dict(attack_counter)}")
    print(f"[INFO] Message type counts: {dict(type_counter)}")

    return archive_rows, summary


def balance_if_needed(df):
    """
    Keeps the initial external benchmark manageable and not dominated
    by one class. This is for the first benchmark pass only.
    """
    if len(df) <= MAX_ROWS_TOTAL:
        return df

    print(f"[INFO] Dataset has {len(df):,} rows. Sampling down to {MAX_ROWS_TOTAL:,} rows.")

    per_class = MAX_ROWS_TOTAL // 2

    normal = df[df["label"] == 0]
    attack = df[df["label"] == 1]

    normal_sample = normal.sample(
        n=min(per_class, len(normal)),
        random_state=RANDOM_SEED,
    )

    attack_sample = attack.sample(
        n=min(per_class, len(attack)),
        random_state=RANDOM_SEED,
    )

    sampled = pd.concat([normal_sample, attack_sample], ignore_index=True)

    remaining = MAX_ROWS_TOTAL - len(sampled)

    if remaining > 0:
        rest = df.drop(sampled.index, errors="ignore")
        if len(rest) > 0:
            sampled = pd.concat(
                [
                    sampled,
                    rest.sample(n=min(remaining, len(rest)), random_state=RANDOM_SEED),
                ],
                ignore_index=True,
            )

    return sampled.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def main():
    print("=" * 70)
    print("TrustV2X VeReMi Extension Dataset Preparation")
    print("=" * 70)

    ensure_dirs()

    extracted_roots = extract_zip_files()

    if not extracted_roots:
        return

    all_rows = []
    summaries = []

    for root in extracted_roots:
        rows, summary = process_archive_root(root)
        all_rows.extend(rows)
        summaries.append(summary)

        if len(all_rows) >= MAX_ROWS_TOTAL:
            print(f"[INFO] Reached MAX_ROWS_TOTAL={MAX_ROWS_TOTAL:,}. Stopping early.")
            break

    if not all_rows:
        print("\n[ERROR] No labelled VeReMi rows were created.")
        print("Possible causes:")
        print("- Downloaded files do not include ground-truth files.")
        print("- File structure differs from expected VeReMi JSON/JSONlog format.")
        print("- Message type filter removed all records.")
        print("\nNext action: send me the printed summary above or a screenshot of the folder structure.")
        return

    df = pd.DataFrame(all_rows)

    df = balance_if_needed(df)

    df.to_csv(OUTPUT_CSV, index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 70)
    print("[OK] VeReMi external benchmark dataset prepared")
    print("=" * 70)
    print(f"[OK] Output CSV: {OUTPUT_CSV}")
    print(f"[OK] Summary CSV: {SUMMARY_CSV}")

    print("\nDataset shape:")
    print(df.shape)

    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False))

    print("\nAttack type distribution:")
    print(df["attack_type"].value_counts(dropna=False).head(20))

    print("\nColumns:")
    print(list(df.columns))

    print("\nNext script:")
    print("scripts/15_train_validate_veremi_external_benchmark.py")


if __name__ == "__main__":
    main()