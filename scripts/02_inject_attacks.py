import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# TrustV2X - Step 02
# Build final 40,000-record 70/30 CAM misbehavior dataset
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "sumo_generated" / "normal_cam_messages.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "sumo_generated" / "trustv2x_attack_dataset.csv"

RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_MODEL_DIR = PROJECT_ROOT / "results" / "models"

SUMMARY_FILE = RESULTS_TABLE_DIR / "attack_injection_summary.csv"
PROFILE_FILE = RESULTS_TABLE_DIR / "dataset_profile.csv"

RANDOM_SEED = 42

TARGET_TOTAL_MESSAGES = 40000
TARGET_NORMAL_RATIO = 0.70

TARGET_NORMAL_MESSAGES = int(TARGET_TOTAL_MESSAGES * TARGET_NORMAL_RATIO)
TARGET_ATTACK_MESSAGES = TARGET_TOTAL_MESSAGES - TARGET_NORMAL_MESSAGES

ATTACK_TYPES = [
    "false_position",
    "gradual_position_drift",
    "speed_falsification",
    "low_speed_spoofing",
    "heading_falsification",
    "replay_attack",
    "sudden_jump",
    "ghost_vehicle",
    "sybil_attack",
    "intermittent_attack",
    "dos_flooding",
    "neighbour_inconsistency",
]

ATTACK_SAMPLE_SIZE_PER_TYPE = TARGET_ATTACK_MESSAGES // len(ATTACK_TYPES)

np.random.seed(RANDOM_SEED)


def clean_previous_outputs():
    """
    Remove old result files so the next ML/DL training run produces fresh tables,
    figures and models.
    """
    patterns = [
        RESULTS_TABLE_DIR / "ml_binary_results.csv",
        RESULTS_TABLE_DIR / "ml_classification_reports.txt",
        RESULTS_TABLE_DIR / "dl_binary_results.csv",
        RESULTS_TABLE_DIR / "dl_classification_reports.txt",
        RESULTS_TABLE_DIR / "trustv2x_scoring_results.csv",
        RESULTS_TABLE_DIR / "trustv2x_decision_summary.csv",
    ]

    for file_path in patterns:
        if file_path.exists():
            file_path.unlink()

    if RESULTS_FIGURE_DIR.exists():
        for fig in RESULTS_FIGURE_DIR.glob("*.png"):
            fig.unlink()

    if RESULTS_MODEL_DIR.exists():
        for model in RESULTS_MODEL_DIR.glob("*"):
            if model.is_file():
                model.unlink()

    print("[OK] Previous ML/DL result tables, figures and model files cleared.")


def load_normal_cam_messages():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run scripts/01_generate_sumo_cam.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "vehicle_id",
        "timestamp",
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "heading",
        "lane_id",
        "lane_position",
        "road_id",
        "neighbour_count",
        "message_type",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["label"] = 0
    df["attack_type"] = "normal"
    df["attack_severity"] = "none"
    df["is_injected"] = 0
    df["source"] = "SUMO"

    return df


def prepare_normal_subset(df):
    """
    Use the earliest 28,000 normal messages so we preserve natural temporal continuity
    rather than random sampling normal rows and breaking vehicle trajectories.
    """
    df = df.sort_values(["timestamp", "vehicle_id"]).reset_index(drop=True)

    if len(df) < TARGET_NORMAL_MESSAGES:
        raise ValueError(
            f"Only {len(df)} normal messages found, but {TARGET_NORMAL_MESSAGES} are required.\n"
            "Increase SIMULATION_END_TIME in 01_generate_sumo_cam.py and rerun it."
        )

    normal_subset = df.head(TARGET_NORMAL_MESSAGES).copy()

    normal_subset["label"] = 0
    normal_subset["attack_type"] = "normal"
    normal_subset["attack_severity"] = "none"
    normal_subset["is_injected"] = 0

    return normal_subset


def sample_rows(df, n):
    sample_size = min(n, len(df))
    return df.sample(n=sample_size, random_state=np.random.randint(0, 999999)).copy()


def assign_severity(n):
    return np.random.choice(
        ["low", "medium", "high"],
        size=n,
        p=[0.40, 0.40, 0.20]
    )


def severity_offset(severities, low_range, medium_range, high_range):
    values = []

    for sev in severities:
        if sev == "low":
            values.append(np.random.uniform(*low_range))
        elif sev == "medium":
            values.append(np.random.uniform(*medium_range))
        else:
            values.append(np.random.uniform(*high_range))

    return np.array(values)


def finalise_attack_df(attack_df, attack_type, severities):
    attack_df["label"] = 1
    attack_df["attack_type"] = attack_type
    attack_df["attack_severity"] = severities
    attack_df["is_injected"] = 1
    attack_df["source"] = "SUMO_injected"
    return attack_df


def inject_false_position(df):
    attack_df = sample_rows(df, ATTACK_SAMPLE_SIZE_PER_TYPE)
    severities = assign_severity(len(attack_df))

    offset = severity_offset(
        severities,
        low_range=(10, 35),
        medium_range=(35, 90),
        high_range=(90, 250),
    )

    signs_x = np.random.choice([-1, 1], size=len(attack_df))
    signs_y = np.random.choice([-1, 1], size=len(attack_df))

    attack_df["x_position"] = attack_df["x_position"] + signs_x * offset
    attack_df["y_position"] = attack_df["y_position"] + signs_y * offset * np.random.uniform(0.5, 1.5, len(attack_df))

    return finalise_attack_df(attack_df, "false_position", severities)


def inject_gradual_position_drift(df):
    attack_df = sample_rows(df, ATTACK_SAMPLE_SIZE_PER_TYPE)
    severities = assign_severity(len(attack_df))

    drift = severity_offset(
        severities,
        low_range=(2, 8),
        medium_range=(8, 20),
        high_range=(20, 45),
    )

    attack_df = attack_df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)
    attack_df["drift_step"] = attack_df.groupby("vehicle_id").cumcount() + 1

    attack_df["x_position"] = attack_df["x_position"] + drift * attack_df["drift_step"] * 0.15
    attack_df["y_position"] = attack_df["y_position"] + drift * attack_df["drift_step"] * 0.10

    attack_df.drop(columns=["drift_step"], inplace=True)

    return finalise_attack_df(attack_df, "gradual_position_drift", severities)


def inject_speed_falsification(df):
    attack_df = sample_rows(df, ATTACK_SAMPLE_SIZE_PER_TYPE)
    severities = assign_severity(len(attack_df))

    fake_speed = severity_offset(
        severities,
        low_range=(20, 35),
        medium_range=(35, 60),
        high_range=(60, 95),
    )

    attack_df["speed"] = fake_speed
    attack_df["acceleration"] = severity_offset(
        severities,
        low_range=(4, 8),
        medium_range=(8, 15),
        high_range=(15, 30),
    )

    return finalise_attack_df(attack_df, "speed_falsification", severities)


def inject_low_speed_spoofing(df):
    attack_df = sample_rows(df, ATTACK_SAMPLE_SIZE_PER_TYPE)
    severities = assign_severity(len(attack_df))

    multiplier = severity_offset(
        severities,
        low_range=(0.60, 0.85),
        medium_range=(0.30, 0.60),
        high_range=(0.05, 0.30),
    )

    attack_df["speed"] = attack_df["speed"] * multiplier
    attack_df["acceleration"] = attack_df["acceleration"] - np.random.uniform(1, 6, len(attack_df))

    return finalise_attack_df(attack_df, "low_speed_spoofing", severities)


def inject_heading_falsification(df):
    attack_df = sample_rows(df, ATTACK_SAMPLE_SIZE_PER_TYPE)
    severities = assign_severity(len(attack_df))

    heading_offset = severity_offset(
        severities,
        low_range=(15, 35),
        medium_range=(45, 100),
        high_range=(120, 180),
    )

    attack_df["heading"] = (attack_df["heading"] + heading_offset) % 360

    return finalise_attack_df(attack_df, "heading_falsification", severities)


def inject_replay_attack(df):
    attack_df = sample_rows(df, ATTACK_SAMPLE_SIZE_PER_TYPE)
    severities = assign_severity(len(attack_df))

    replay_delay = severity_offset(
        severities,
        low_range=(5, 15),
        medium_range=(15, 60),
        high_range=(60, 180),
    )

    attack_df["timestamp"] = attack_df["timestamp"] + replay_delay
    attack_df["inter_arrival_time"] = replay_delay

    return finalise_attack_df(attack_df, "replay_attack", severities)


def inject_sudden_jump(df):
    attack_df = sample_rows(df, ATTACK_SAMPLE_SIZE_PER_TYPE)
    severities = assign_severity(len(attack_df))

    jump = severity_offset(
        severities,
        low_range=(40, 80),
        medium_range=(80, 180),
        high_range=(180, 450),
    )

    attack_df["x_position"] = attack_df["x_position"] + np.random.choice([-1, 1], len(attack_df)) * jump
    attack_df["y_position"] = attack_df["y_position"] + np.random.choice([-1, 1], len(attack_df)) * jump

    return finalise_attack_df(attack_df, "sudden_jump", severities)


def inject_ghost_vehicle(df):
    attack_df = sample_rows(df, ATTACK_SAMPLE_SIZE_PER_TYPE)
    severities = assign_severity(len(attack_df))

    attack_df["vehicle_id"] = [
        f"ghost_{i}_{np.random.randint(10000, 99999)}" for i in range(len(attack_df))
    ]

    # Keep road/lane context but alter reported mobility to represent a phantom vehicle.
    attack_df["x_position"] = attack_df["x_position"] + np.random.uniform(-120, 120, len(attack_df))
    attack_df["y_position"] = attack_df["y_position"] + np.random.uniform(-120, 120, len(attack_df))
    attack_df["speed"] = np.random.uniform(0, 35, len(attack_df))
    attack_df["heading"] = np.random.uniform(0, 360, len(attack_df))
    attack_df["neighbour_count"] = np.random.randint(0, 3, len(attack_df))

    return finalise_attack_df(attack_df, "ghost_vehicle", severities)


def inject_sybil_attack(df):
    """
    Sybil attack:
    One physical vehicle appears as multiple identities with highly similar
    mobility state around the same timestamp.
    """
    base_df = sample_rows(df, math.ceil(ATTACK_SAMPLE_SIZE_PER_TYPE / 4))
    sybil_rows = []

    for _, row in base_df.iterrows():
        for clone_id in range(4):
            clone = row.copy()
            clone["vehicle_id"] = f"sybil_{row['vehicle_id']}_{clone_id}_{np.random.randint(1000, 9999)}"
            clone["x_position"] = clone["x_position"] + np.random.uniform(-5, 5)
            clone["y_position"] = clone["y_position"] + np.random.uniform(-5, 5)
            clone["speed"] = max(0, clone["speed"] + np.random.uniform(-1, 1))
            clone["heading"] = (clone["heading"] + np.random.uniform(-5, 5)) % 360
            sybil_rows.append(clone)

            if len(sybil_rows) >= ATTACK_SAMPLE_SIZE_PER_TYPE:
                break

        if len(sybil_rows) >= ATTACK_SAMPLE_SIZE_PER_TYPE:
            break

    attack_df = pd.DataFrame(sybil_rows)
    severities = assign_severity(len(attack_df))

    return finalise_attack_df(attack_df, "sybil_attack", severities)


def inject_intermittent_attack(df):
    """
    Intermittent attack:
    A vehicle behaves normally most of the time but occasionally manipulates
    CAM mobility values.
    """
    attack_df = sample_rows(df, ATTACK_SAMPLE_SIZE_PER_TYPE)
    severities = assign_severity(len(attack_df))

    position_drift = severity_offset(
        severities,
        low_range=(5, 15),
        medium_range=(15, 45),
        high_range=(45, 90),
    )

    speed_noise = severity_offset(
        severities,
        low_range=(2, 6),
        medium_range=(6, 15),
        high_range=(15, 30),
    )

    attack_df["x_position"] = attack_df["x_position"] + np.random.choice([-1, 1], len(attack_df)) * position_drift
    attack_df["y_position"] = attack_df["y_position"] + np.random.choice([-1, 1], len(attack_df)) * position_drift
    attack_df["speed"] = np.maximum(0, attack_df["speed"] + np.random.choice([-1, 1], len(attack_df)) * speed_noise)
    attack_df["heading"] = (attack_df["heading"] + np.random.uniform(-45, 45, len(attack_df))) % 360

    return finalise_attack_df(attack_df, "intermittent_attack", severities)


def inject_dos_flooding(df):
    """
    DoS flooding:
    A sender produces excessive CAM messages within a short time window.
    """
    base_df = sample_rows(df, math.ceil(ATTACK_SAMPLE_SIZE_PER_TYPE / 20))
    dos_rows = []

    for _, row in base_df.iterrows():
        for burst_id in range(20):
            burst = row.copy()
            burst["timestamp"] = row["timestamp"] + burst_id * 0.05
            burst["inter_arrival_time"] = 0.05
            burst["speed"] = max(0, row["speed"] + np.random.uniform(-0.3, 0.3))
            burst["heading"] = (row["heading"] + np.random.uniform(-1, 1)) % 360
            dos_rows.append(burst)

            if len(dos_rows) >= ATTACK_SAMPLE_SIZE_PER_TYPE:
                break

        if len(dos_rows) >= ATTACK_SAMPLE_SIZE_PER_TYPE:
            break

    attack_df = pd.DataFrame(dos_rows)
    severities = assign_severity(len(attack_df))

    return finalise_attack_df(attack_df, "dos_flooding", severities)


def inject_neighbour_inconsistency(df):
    attack_df = sample_rows(df, ATTACK_SAMPLE_SIZE_PER_TYPE)
    severities = assign_severity(len(attack_df))

    fake_neighbours = []

    for current_count in attack_df["neighbour_count"]:
        if current_count <= 2:
            fake_neighbours.append(np.random.randint(15, 45))
        else:
            fake_neighbours.append(0)

    attack_df["neighbour_count"] = fake_neighbours

    return finalise_attack_df(attack_df, "neighbour_inconsistency", severities)


def add_trustv2x_plausibility_features(df):
    """
    Add TrustV2X plausibility, temporal, identity and message-rate features.
    These features are used by the ML/DL models but do not directly expose
    the attack_type or label.
    """
    df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)

    df["prev_x_position"] = df.groupby("vehicle_id")["x_position"].shift(1)
    df["prev_y_position"] = df.groupby("vehicle_id")["y_position"].shift(1)
    df["prev_timestamp"] = df.groupby("vehicle_id")["timestamp"].shift(1)
    df["prev_heading"] = df.groupby("vehicle_id")["heading"].shift(1)
    df["prev_speed"] = df.groupby("vehicle_id")["speed"].shift(1)

    df["delta_t"] = df["timestamp"] - df["prev_timestamp"]
    df["delta_t"] = df["delta_t"].replace(0, np.nan)

    df["distance_from_previous"] = np.sqrt(
        (df["x_position"] - df["prev_x_position"]) ** 2
        + (df["y_position"] - df["prev_y_position"]) ** 2
    )

    df["estimated_speed_from_position"] = df["distance_from_previous"] / df["delta_t"]

    df["speed_position_error"] = np.abs(
        df["estimated_speed_from_position"] - df["speed"]
    )

    df["heading_change"] = np.abs(df["heading"] - df["prev_heading"])
    df["heading_change"] = df["heading_change"].apply(
        lambda x: min(x, 360 - x) if pd.notnull(x) else np.nan
    )

    df["speed_change"] = np.abs(df["speed"] - df["prev_speed"])

    # Message rate features for DoS/flooding behaviour
    df["time_bin_5s"] = np.floor(df["timestamp"] / 5).astype(int)
    df["message_rate_5s"] = df.groupby(["vehicle_id", "time_bin_5s"])["vehicle_id"].transform("count")

    # Identity similarity features for Sybil-like behaviour
    df["timestamp_int"] = np.floor(df["timestamp"]).astype(int)
    df["x_bin_20m"] = np.floor(df["x_position"] / 20).astype(int)
    df["y_bin_20m"] = np.floor(df["y_position"] / 20).astype(int)
    df["speed_bin_2ms"] = np.floor(df["speed"] / 2).astype(int)
    df["heading_bin_20deg"] = np.floor(df["heading"] / 20).astype(int)

    df["identity_cluster_count"] = df.groupby(
        [
            "timestamp_int",
            "x_bin_20m",
            "y_bin_20m",
            "speed_bin_2ms",
            "heading_bin_20deg",
        ]
    )["vehicle_id"].transform("nunique")

    df["sender_message_count"] = df.groupby("vehicle_id")["vehicle_id"].transform("count")

    # Rule flags
    df["flag_unrealistic_speed"] = (df["speed"] > 40).astype(int)
    df["flag_hard_acceleration"] = (np.abs(df["acceleration"]) > 8).astype(int)
    df["flag_large_position_jump"] = (df["distance_from_previous"] > 100).astype(int)
    df["flag_speed_position_mismatch"] = (df["speed_position_error"] > 30).astype(int)
    df["flag_heading_anomaly"] = (df["heading_change"] > 120).astype(int)
    df["flag_stale_or_irregular_time"] = ((df["delta_t"] <= 0) | (df["delta_t"] > 10)).astype(int)
    df["flag_dos_rate"] = (df["message_rate_5s"] > 8).astype(int)
    df["flag_possible_sybil"] = (df["identity_cluster_count"] > 2).astype(int)
    df["flag_low_sender_history"] = (df["sender_message_count"] <= 2).astype(int)
    df["flag_neighbour_anomaly"] = ((df["neighbour_count"] > 30) | (df["neighbour_count"] < 0)).astype(int)

    rule_flags = [
        "flag_unrealistic_speed",
        "flag_hard_acceleration",
        "flag_large_position_jump",
        "flag_speed_position_mismatch",
        "flag_heading_anomaly",
        "flag_stale_or_irregular_time",
        "flag_dos_rate",
        "flag_possible_sybil",
        "flag_low_sender_history",
        "flag_neighbour_anomaly",
    ]

    df["trustv2x_rule_violation_count"] = df[rule_flags].sum(axis=1)

    fill_columns = [
        "prev_x_position",
        "prev_y_position",
        "prev_timestamp",
        "prev_heading",
        "prev_speed",
        "delta_t",
        "distance_from_previous",
        "estimated_speed_from_position",
        "speed_position_error",
        "heading_change",
        "speed_change",
    ]

    df[fill_columns] = df[fill_columns].fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    # Shuffle only after feature engineering
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return df


def save_summaries(df):
    RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby(["attack_type", "attack_severity", "label"])
        .size()
        .reset_index(name="message_count")
        .sort_values(["label", "attack_type", "attack_severity"])
    )

    summary.to_csv(SUMMARY_FILE, index=False)

    profile = pd.DataFrame(
        [
            {
                "total_messages": len(df),
                "normal_messages": int((df["label"] == 0).sum()),
                "attack_messages": int((df["label"] == 1).sum()),
                "normal_ratio": round((df["label"] == 0).mean(), 4),
                "attack_ratio": round((df["label"] == 1).mean(), 4),
                "attack_types": df[df["label"] == 1]["attack_type"].nunique(),
                "features_columns": df.shape[1],
            }
        ]
    )

    profile.to_csv(PROFILE_FILE, index=False)

    print("\nDataset profile:")
    print(profile.to_string(index=False))

    print("\nAttack injection summary:")
    print(
        df["attack_type"]
        .value_counts()
        .to_string()
    )

    print(f"\n[OK] Summary saved to: {SUMMARY_FILE}")
    print(f"[OK] Profile saved to: {PROFILE_FILE}")


def main():
    print("=" * 70)
    print("TrustV2X - Final 40,000 Record Attack Dataset Generator")
    print("=" * 70)

    clean_previous_outputs()

    normal_df = load_normal_cam_messages()

    print(f"[INFO] Loaded normal CAM messages: {len(normal_df)}")

    normal_subset = prepare_normal_subset(normal_df)

    print(f"[INFO] Selected normal messages: {len(normal_subset)}")

    attack_dfs = [
        inject_false_position(normal_subset),
        inject_gradual_position_drift(normal_subset),
        inject_speed_falsification(normal_subset),
        inject_low_speed_spoofing(normal_subset),
        inject_heading_falsification(normal_subset),
        inject_replay_attack(normal_subset),
        inject_sudden_jump(normal_subset),
        inject_ghost_vehicle(normal_subset),
        inject_sybil_attack(normal_subset),
        inject_intermittent_attack(normal_subset),
        inject_dos_flooding(normal_subset),
        inject_neighbour_inconsistency(normal_subset),
    ]

    combined_df = pd.concat([normal_subset] + attack_dfs, ignore_index=True)

    combined_df = add_trustv2x_plausibility_features(combined_df)

    if len(combined_df) != TARGET_TOTAL_MESSAGES:
        raise RuntimeError(
            f"Dataset size mismatch. Expected {TARGET_TOTAL_MESSAGES}, got {len(combined_df)}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n[OK] Final attack dataset created: {OUTPUT_FILE}")
    print(f"[INFO] Total messages: {len(combined_df)}")
    print(f"[INFO] Normal messages: {(combined_df['label'] == 0).sum()}")
    print(f"[INFO] Attack messages: {(combined_df['label'] == 1).sum()}")

    save_summaries(combined_df)

    print("\nDone. Next step: update and rerun ML/DL training scripts.")


if __name__ == "__main__":
    main()