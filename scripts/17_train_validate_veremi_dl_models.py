import os
import time
import warnings
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# TrustV2X - Step 17
# VeReMi External Benchmark Deep Learning Evaluation
# ============================================================
#
# Models:
# - MLP
# - 1D-CNN
#
# Excluded:
# - LSTM is deliberately excluded because the current VeReMi
#   representation is tabular/message-level, not a proper
#   vehicle-sequence window representation.
#
# Input:
# data/veremi/processed/veremi_extension_trustv2x_external_benchmark.csv
#
# Outputs:
# results/veremi_dl/tables/veremi_external_dl_results.csv
# results/veremi_dl/tables/veremi_external_ml_dl_combined_results.csv
# results/veremi_dl/figures/
# results/veremi_dl/models/
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "veremi"
    / "processed"
    / "veremi_extension_trustv2x_external_benchmark.csv"
)

ML_RESULTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "veremi"
    / "tables"
    / "veremi_external_ml_results.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "veremi_dl"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

for folder in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


RANDOM_SEED = 42

# Keep this deadline-friendly.
# Increase to 150000 only if you have enough time.
MAX_ROWS_FOR_TRAINING = 100_000

EPOCHS = 10
BATCH_SIZE = 1024
VALIDATION_SPLIT = 0.15


def make_onehot_encoder():
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def to_numeric_safe(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    return df


def heading_difference(current, previous):
    diff = np.abs(current - previous)
    return np.minimum(diff, 360 - diff)


def sample_if_needed(df):
    if len(df) <= MAX_ROWS_FOR_TRAINING:
        return df.reset_index(drop=True)

    print(f"[INFO] Dataset has {len(df):,} rows.")
    print(f"[INFO] Sampling to {MAX_ROWS_FOR_TRAINING:,} rows for fast DL evaluation.")

    per_class = MAX_ROWS_FOR_TRAINING // 2

    normal_df = df[df["label"] == 0]
    attack_df = df[df["label"] == 1]

    normal_sample = normal_df.sample(
        n=min(per_class, len(normal_df)),
        random_state=RANDOM_SEED,
    )

    attack_sample = attack_df.sample(
        n=min(per_class, len(attack_df)),
        random_state=RANDOM_SEED,
    )

    sampled = pd.concat([normal_sample, attack_sample], ignore_index=False)

    remaining = MAX_ROWS_FOR_TRAINING - len(sampled)

    if remaining > 0:
        rest = df.drop(index=sampled.index, errors="ignore")
        extra = rest.sample(n=min(remaining, len(rest)), random_state=RANDOM_SEED)
        sampled = pd.concat([sampled, extra], ignore_index=False)

    return sampled.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def engineer_trustv2x_features(df):
    print("[INFO] Engineering TrustV2X-style VeReMi features...")

    df = df.copy()

    numeric_base = [
        "timestamp",
        "rcvTime",
        "sendTime",
        "x_position",
        "y_position",
        "z_position",
        "speed",
        "acceleration",
        "heading",
    ]

    optional_numeric = [
        "speed_x",
        "speed_y",
        "speed_z",
        "acceleration_x",
        "acceleration_y",
        "acceleration_z",
    ]

    df = to_numeric_safe(df, numeric_base + optional_numeric)

    if "vehicle_id" not in df.columns:
        df["vehicle_id"] = df.get("sender", "unknown").astype(str)

    df["vehicle_id"] = df["vehicle_id"].fillna("unknown").astype(str)

    if "message_type" not in df.columns:
        df["message_type"] = "3"

    df["message_type"] = df["message_type"].fillna("unknown").astype(str)

    df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)

    grouped = df.groupby("vehicle_id", sort=False)

    df["prev_timestamp"] = grouped["timestamp"].shift(1)
    df["prev_x_position"] = grouped["x_position"].shift(1)
    df["prev_y_position"] = grouped["y_position"].shift(1)
    df["prev_speed"] = grouped["speed"].shift(1)
    df["prev_heading"] = grouped["heading"].shift(1)

    df["delta_t"] = df["timestamp"] - df["prev_timestamp"]
    df.loc[df["delta_t"] <= 0, "delta_t"] = np.nan

    df["distance_from_previous"] = np.sqrt(
        (df["x_position"] - df["prev_x_position"]) ** 2
        + (df["y_position"] - df["prev_y_position"]) ** 2
    )

    df["estimated_speed_from_position"] = (
        df["distance_from_previous"] / df["delta_t"]
    )

    df["estimated_speed_from_position"] = df[
        "estimated_speed_from_position"
    ].replace([np.inf, -np.inf], np.nan)

    df["speed_position_error_observable"] = np.abs(
        df["estimated_speed_from_position"] - df["speed"]
    )

    df["heading_change"] = heading_difference(df["heading"], df["prev_heading"])
    df["speed_change"] = np.abs(df["speed"] - df["prev_speed"])

    df["time_bin_5s"] = np.floor(df["timestamp"] / 5).fillna(0).astype(int)

    df["message_rate_5s"] = (
        df.groupby(["vehicle_id", "time_bin_5s"])["timestamp"]
        .transform("count")
        .astype(float)
    )

    df["x_bin_20m"] = np.floor(df["x_position"] / 20).fillna(0).astype(int)
    df["y_bin_20m"] = np.floor(df["y_position"] / 20).fillna(0).astype(int)

    df["identity_cluster_count"] = (
        df.groupby(["time_bin_5s", "x_bin_20m", "y_bin_20m"])["vehicle_id"]
        .transform("nunique")
        .astype(float)
    )

    df["sender_message_count"] = grouped.cumcount() + 1

    df["flag_unrealistic_speed"] = (df["speed"] > 55).astype(int)
    df["flag_hard_acceleration"] = (df["acceleration"].abs() > 8).astype(int)

    df["flag_large_position_jump"] = (
        (df["distance_from_previous"] > 80) & (df["delta_t"] <= 2)
    ).astype(int)

    df["flag_speed_position_mismatch"] = (
        df["speed_position_error_observable"] > 25
    ).astype(int)

    df["flag_heading_anomaly"] = (
        (df["heading_change"] > 120) & (df["speed"] > 1)
    ).astype(int)

    df["flag_stale_or_irregular_time"] = (
        (df["delta_t"].isna()) | (df["delta_t"] > 10)
    ).astype(int)

    df["flag_dos_rate"] = (df["message_rate_5s"] > 25).astype(int)
    df["flag_possible_sybil"] = (df["identity_cluster_count"] > 4).astype(int)
    df["flag_low_sender_history"] = (df["sender_message_count"] < 3).astype(int)

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
    ]

    df["trustv2x_rule_violation_count"] = df[rule_flags].sum(axis=1)

    fill_cols = [
        "prev_timestamp",
        "prev_x_position",
        "prev_y_position",
        "prev_speed",
        "prev_heading",
        "delta_t",
        "distance_from_previous",
        "estimated_speed_from_position",
        "speed_position_error_observable",
        "heading_change",
        "speed_change",
        "message_rate_5s",
        "identity_cluster_count",
        "sender_message_count",
    ]

    for col in fill_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(0.0)

    return df


def get_feature_sets():
    raw_numeric = [
        "timestamp",
        "rcvTime",
        "sendTime",
        "x_position",
        "y_position",
        "z_position",
        "speed",
        "acceleration",
        "heading",
        "speed_x",
        "speed_y",
        "speed_z",
        "acceleration_x",
        "acceleration_y",
        "acceleration_z",
    ]

    trustv2x_continuous = raw_numeric + [
        "delta_t",
        "distance_from_previous",
        "estimated_speed_from_position",
        "speed_position_error_observable",
        "heading_change",
        "speed_change",
        "message_rate_5s",
        "identity_cluster_count",
        "sender_message_count",
    ]

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
        "trustv2x_rule_violation_count",
    ]

    return {
        "VeReMi_Raw_BSM_Features": {
            "numeric": raw_numeric,
            "categorical": ["message_type"],
        },
        "VeReMi_TrustV2X_No_Rule_Flags": {
            "numeric": trustv2x_continuous,
            "categorical": ["message_type"],
        },
        "VeReMi_TrustV2X_Full": {
            "numeric": trustv2x_continuous + rule_flags,
            "categorical": ["message_type"],
        },
    }


def make_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def import_tensorflow():
    try:
        import tensorflow as tf

        tf.random.set_seed(RANDOM_SEED)
        return tf
    except Exception as e:
        raise RuntimeError(
            "TensorFlow is required for Step 17. Install it with:\n"
            "pip install tensorflow\n\n"
            f"Original error: {e}"
        )


def build_mlp(tf, input_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_1d_cnn(tf, input_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim, 1)),
            tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def compute_metrics(y_true, y_pred, y_prob, prediction_time, test_size):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    output = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_detection_rate": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "false_alarm_rate": false_alarm_rate,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "prediction_time_seconds": prediction_time,
        "avg_latency_ms_per_message": (prediction_time / max(test_size, 1)) * 1000,
    }

    try:
        output["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        output["roc_auc"] = np.nan

    try:
        output["pr_auc"] = average_precision_score(y_true, y_prob)
    except Exception:
        output["pr_auc"] = np.nan

    return output


def train_dl_model(tf, model_name, X_train, y_train, X_test):
    input_dim = X_train.shape[1]

    if model_name == "MLP":
        model = build_mlp(tf, input_dim)
        X_train_model = X_train
        X_test_model = X_test

    elif model_name == "1D-CNN":
        model = build_1d_cnn(tf, input_dim)
        X_train_model = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_model = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    start_train = time.perf_counter()

    history = model.fit(
        X_train_model,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=0,
    )

    training_time = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_prob = model.predict(X_test_model, batch_size=BATCH_SIZE, verbose=0).ravel()
    prediction_time = time.perf_counter() - start_pred

    y_pred = (y_prob >= 0.5).astype(int)

    return model, history, y_pred, y_prob, training_time, prediction_time


def plot_confusion_matrix(cm, title, output_path):
    fig, ax = plt.subplots(figsize=(5, 4))

    image = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Attack"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Attack"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate_split(tf, df, split_name, train_idx, test_idx, feature_sets):
    print("\n" + "=" * 80)
    print(f"[SPLIT] {split_name}")
    print("=" * 80)

    y = df["label"].astype(int).values

    results = []

    for feature_set_name, fs in feature_sets.items():
        print(f"\n[FEATURE SET] {feature_set_name}")

        numeric_features = [c for c in fs["numeric"] if c in df.columns]
        categorical_features = [c for c in fs["categorical"] if c in df.columns]

        feature_columns = numeric_features + categorical_features

        X = df[feature_columns]

        X_train_raw = X.iloc[train_idx]
        X_test_raw = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        preprocessor = make_preprocessor(numeric_features, categorical_features)

        X_train = preprocessor.fit_transform(X_train_raw).astype(np.float32)
        X_test = preprocessor.transform(X_test_raw).astype(np.float32)

        print(f"[INFO] Train size: {len(y_train):,}, Test size: {len(y_test):,}")
        print(f"[INFO] Features after preprocessing: {X_train.shape[1]}")

        for model_name in ["MLP", "1D-CNN"]:
            print(f"\n[TRAIN] {split_name} | {feature_set_name} | {model_name}")

            model, history, y_pred, y_prob, training_time, prediction_time = train_dl_model(
                tf=tf,
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
            )

            metrics = compute_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                prediction_time=prediction_time,
                test_size=len(y_test),
            )

            result_row = {
                "split": split_name,
                "feature_set": feature_set_name,
                "model": model_name,
                "train_size": len(y_train),
                "test_size": len(y_test),
                "epochs_requested": EPOCHS,
                "epochs_trained": len(history.history.get("loss", [])),
                "training_time_seconds": training_time,
                **metrics,
            }

            results.append(result_row)

            safe_split = split_name.lower().replace(" ", "_")
            safe_feature = feature_set_name.lower().replace(" ", "_")
            safe_model = model_name.lower().replace("-", "_")

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

            plot_confusion_matrix(
                cm,
                f"{split_name} | {feature_set_name} | {model_name}",
                FIGURES_DIR / f"cm_{safe_split}_{safe_feature}_{safe_model}.png",
            )

            model.save(MODELS_DIR / f"{safe_split}_{safe_feature}_{safe_model}.keras")

            print(
                f"Accuracy={metrics['accuracy']:.4f} | "
                f"BalancedAcc={metrics['balanced_accuracy']:.4f} | "
                f"Precision={metrics['precision']:.4f} | "
                f"Recall={metrics['recall_detection_rate']:.4f} | "
                f"F1={metrics['f1_score']:.4f} | "
                f"FAR={metrics['false_alarm_rate']:.4f} | "
                f"MCC={metrics['mcc']:.4f} | "
                f"Latency={metrics['avg_latency_ms_per_message']:.6f} ms | "
                f"Epochs={result_row['epochs_trained']}"
            )

    return results


def save_dl_model_comparison_plot(results_df):
    plot_df = results_df[
        results_df["split"] == "Vehicle Disjoint Holdout"
    ].copy()

    plot_df["method"] = plot_df["feature_set"] + "\n" + plot_df["model"]

    plot_df = plot_df.sort_values("f1_score", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 7))

    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["f1_score"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["method"])
    ax.set_xlabel("F1-score")
    ax.set_title("VeReMi DL model comparison under vehicle-disjoint holdout")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_veremi_dl_vehicle_disjoint_f1.png", dpi=300)
    plt.close(fig)


def save_combined_ml_dl_plot(combined_df):
    plot_df = combined_df[
        (combined_df["split"] == "Vehicle Disjoint Holdout")
        & (
            combined_df["feature_set"].isin(
                [
                    "VeReMi_Raw_BSM_Features",
                    "VeReMi_TrustV2X_No_Rule_Flags",
                    "VeReMi_TrustV2X_Full",
                ]
            )
        )
    ].copy()

    plot_df = plot_df.sort_values("f1_score", ascending=True)
    plot_df["method"] = plot_df["feature_set"] + "\n" + plot_df["model"]

    fig, ax = plt.subplots(figsize=(12, 9))

    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["f1_score"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["method"])
    ax.set_xlabel("F1-score")
    ax.set_title("VeReMi ML and DL model comparison under vehicle-disjoint holdout")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_veremi_ml_dl_vehicle_disjoint_f1.png", dpi=300)
    plt.close(fig)


def main():
    print("=" * 80)
    print("TrustV2X - VeReMi External Benchmark DL Evaluation")
    print("=" * 80)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_CSV}\n\n"
            "Run Step 14e first."
        )

    tf = import_tensorflow()

    df = pd.read_csv(INPUT_CSV)

    print(f"[INFO] Loaded input CSV: {INPUT_CSV}")
    print(f"[INFO] Original shape: {df.shape}")

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)
    df = df[df["label"].isin([0, 1])].copy()

    print("\n[INFO] Label distribution before sampling:")
    print(df["label"].value_counts())

    if "attack_type" in df.columns:
        print("\n[INFO] Attack type distribution before sampling:")
        print(df["attack_type"].value_counts())

    leakage_columns = [
        "gt_x_position",
        "gt_y_position",
        "gt_z_position",
        "gt_speed_x",
        "gt_speed_y",
        "gt_speed_z",
        "gt_speed",
        "gt_acceleration_x",
        "gt_acceleration_y",
        "gt_acceleration_z",
        "gt_acceleration",
        "gt_heading",
        "position_error",
        "speed_error",
        "acceleration_error",
        "heading_error_degrees",
        "label_by_token",
        "attack_type",
        "attack_token",
        "source_archive",
        "source_file",
        "receiver_id",
        "message_id",
        "sender",
        "senderPseudo",
    ]

    print("\n[INFO] Leakage/audit columns present but NOT used as model features:")
    print([c for c in leakage_columns if c in df.columns])

    df = sample_if_needed(df)

    print(f"\n[INFO] Shape after optional sampling: {df.shape}")
    print("[INFO] Label distribution after sampling:")
    print(df["label"].value_counts())

    df = engineer_trustv2x_features(df)

    feature_sets = get_feature_sets()

    all_results = []

    y = df["label"].astype(int)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    all_results.extend(
        evaluate_split(
            tf=tf,
            df=df,
            split_name="Random Stratified Holdout",
            train_idx=train_idx,
            test_idx=test_idx,
            feature_sets=feature_sets,
        )
    )

    groups = df["vehicle_id"].fillna("unknown").astype(str)

    if groups.nunique() > 1:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=RANDOM_SEED,
        )

        group_train_idx, group_test_idx = next(splitter.split(df, y, groups=groups))

        all_results.extend(
            evaluate_split(
                tf=tf,
                df=df,
                split_name="Vehicle Disjoint Holdout",
                train_idx=group_train_idx,
                test_idx=group_test_idx,
                feature_sets=feature_sets,
            )
        )

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        ["split", "f1_score", "balanced_accuracy"],
        ascending=[True, False, False],
    )

    dl_results_path = TABLES_DIR / "veremi_external_dl_results.csv"
    results_df.to_csv(dl_results_path, index=False)

    save_dl_model_comparison_plot(results_df)

    combined_df = results_df.copy()
    combined_df["model_family"] = "DL"

    if ML_RESULTS_CSV.exists():
        ml_df = pd.read_csv(ML_RESULTS_CSV)
        ml_df["model_family"] = "ML"
        combined_df = pd.concat([ml_df, combined_df], ignore_index=True, sort=False)

        combined_path = TABLES_DIR / "veremi_external_ml_dl_combined_results.csv"
        combined_df.to_csv(combined_path, index=False)

        save_combined_ml_dl_plot(combined_df)

    print("\n" + "=" * 80)
    print("[OK] VeReMi DL evaluation complete")
    print("=" * 80)

    print(f"[OK] DL results saved to: {dl_results_path}")
    print(f"[OK] Figures saved to: {FIGURES_DIR}")
    print(f"[OK] Models saved to: {MODELS_DIR}")

    if ML_RESULTS_CSV.exists():
        print(f"[OK] Combined ML/DL results saved to: {combined_path}")

    display_columns = [
        "split",
        "feature_set",
        "model",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_detection_rate",
        "f1_score",
        "false_alarm_rate",
        "mcc",
        "roc_auc",
        "pr_auc",
        "avg_latency_ms_per_message",
        "epochs_trained",
    ]

    print("\nTop DL results:")
    print(results_df[display_columns].head(20).to_string(index=False))

    if ML_RESULTS_CSV.exists():
        print("\nTop combined ML/DL vehicle-disjoint results:")
        combined_vehicle = combined_df[
            combined_df["split"] == "Vehicle Disjoint Holdout"
        ].copy()

        combined_vehicle = combined_vehicle.sort_values(
            ["f1_score", "balanced_accuracy", "mcc"],
            ascending=[False, False, False],
        )

        combined_display = [
            "model_family",
            "feature_set",
            "model",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall_detection_rate",
            "f1_score",
            "false_alarm_rate",
            "mcc",
            "avg_latency_ms_per_message",
        ]

        print(combined_vehicle[combined_display].head(20).to_string(index=False))

    print("\nPaper note:")
    print(
        "LSTM is intentionally excluded because this experiment is message-level/tabular. "
        "A valid LSTM comparison would require explicit vehicle-level temporal windows."
    )


if __name__ == "__main__":
    main()