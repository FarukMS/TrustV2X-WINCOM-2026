import os
import time
import warnings
from pathlib import Path

# Force Matplotlib to use a non-GUI backend on Windows.
# This prevents Tkinter/Tcl crashes during multi-threaded ML training.
os.environ["MPLBACKEND"] = "Agg"

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
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
# TrustV2X - Step 15
# Train and Validate TrustV2X on VeReMi External Benchmark
# ============================================================
#
# Input:
# data/veremi/processed/veremi_extension_trustv2x_external_benchmark.csv
#
# Output:
# results/veremi/tables/veremi_external_ml_results.csv
# results/veremi/tables/veremi_external_classification_reports.txt
# results/veremi/tables/veremi_external_per_attack_results.csv
# results/veremi/figures/
# results/veremi/models/
#
# Important:
# - This script does NOT modify the original controlled SUMO 4x4 results.
# - Ground-truth/error columns are NOT used as model input features.
# - The model uses observable CAM/BSM-like fields and engineered TrustV2X features.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "veremi"
    / "processed"
    / "veremi_extension_trustv2x_external_benchmark.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "veremi"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

for folder in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


RANDOM_SEED = 42

# Your created VeReMi CSV has 250,000 rows.
# Keep first benchmark manageable. Increase later if required.
MAX_ROWS_FOR_TRAINING = 150_000

# Set True only if you want extra slow sklearn GradientBoosting.
RUN_SLOW_MODELS = False


def make_onehot_encoder():
    """
    Handles different scikit-learn versions.
    Newer versions use sparse_output=False.
    Older versions use sparse=False.
    """
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
    print(f"[INFO] Sampling to {MAX_ROWS_FOR_TRAINING:,} rows.")

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
        extra = rest.sample(
            n=min(remaining, len(rest)),
            random_state=RANDOM_SEED,
        )
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

    # Sort by vehicle and time before calculating temporal features.
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

    # Broad plausibility flags.
    # These are based only on observable message values, not ground truth.
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
            "description": "Raw VeReMi CAM/BSM-like fields only.",
        },
        "VeReMi_TrustV2X_No_Rule_Flags": {
            "numeric": trustv2x_continuous,
            "categorical": ["message_type"],
            "description": "TrustV2X continuous movement/timing features without binary rule flags.",
        },
        "VeReMi_TrustV2X_Full": {
            "numeric": trustv2x_continuous + rule_flags,
            "categorical": ["message_type"],
            "description": "Full TrustV2X feature set with continuous features and plausibility flags.",
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

def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=1,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=RANDOM_SEED,
        )

    except Exception as e:
        print(f"[WARNING] XGBoost not available: {e}")

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=220,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_SEED,
            verbose=-1,
        )

    except Exception as e:
        print(f"[WARNING] LightGBM not available: {e}")

    if RUN_SLOW_MODELS:
        try:
            from sklearn.ensemble import GradientBoostingClassifier

            models["Gradient Boosting"] = GradientBoostingClassifier(
                n_estimators=160,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_SEED,
            )

        except Exception as e:
            print(f"[WARNING] Gradient Boosting not available: {e}")

    return models


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


def evaluate_split(df, split_name, train_idx, test_idx, feature_sets, models):
    print("\n" + "=" * 80)
    print(f"[SPLIT] {split_name}")
    print("=" * 80)

    y = df["label"].astype(int)

    all_rows = []
    reports = []
    best_model_info = None

    for feature_set_name, fs in feature_sets.items():
        print(f"\n[FEATURE SET] {feature_set_name}")

        numeric_features = [c for c in fs["numeric"] if c in df.columns]
        categorical_features = [c for c in fs["categorical"] if c in df.columns]

        feature_columns = numeric_features + categorical_features

        X = df[feature_columns]

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        print(f"[INFO] Train size: {len(y_train):,}, Test size: {len(y_test):,}")
        print(f"[INFO] Features used: {len(feature_columns)}")

        for model_name, model in models.items():
            print(f"\n[TRAIN] {split_name} | {feature_set_name} | {model_name}")

            preprocessor = make_preprocessor(numeric_features, categorical_features)

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            start_train = time.perf_counter()
            pipeline.fit(X_train, y_train)
            training_time = time.perf_counter() - start_train

            start_pred = time.perf_counter()
            y_pred = pipeline.predict(X_test)
            prediction_time = time.perf_counter() - start_pred

            try:
                y_prob = pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = y_pred.astype(float)

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
                "feature_description": fs["description"],
                "model": model_name,
                "train_size": len(y_train),
                "test_size": len(y_test),
                "training_time_seconds": training_time,
                **metrics,
            }

            all_rows.append(result_row)

            report = classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["normal", "attack"],
                zero_division=0,
            )

            reports.append(
                f"\n{'=' * 100}\n"
                f"Split: {split_name}\n"
                f"Feature set: {feature_set_name}\n"
                f"Model: {model_name}\n"
                f"{'=' * 100}\n"
                f"{report}\n"
            )

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

            safe_split = split_name.lower().replace(" ", "_")
            safe_feature = feature_set_name.lower().replace(" ", "_")
            safe_model = model_name.lower().replace(" ", "_")

            cm_path = FIGURES_DIR / f"cm_{safe_split}_{safe_feature}_{safe_model}.png"

            plot_confusion_matrix(
                cm,
                f"{split_name} | {feature_set_name} | {model_name}",
                cm_path,
            )

            model_path = MODELS_DIR / f"{safe_split}_{safe_feature}_{safe_model}.joblib"
            joblib.dump(pipeline, model_path)

            print(
                f"Accuracy={metrics['accuracy']:.4f} | "
                f"BalancedAcc={metrics['balanced_accuracy']:.4f} | "
                f"Precision={metrics['precision']:.4f} | "
                f"Recall={metrics['recall_detection_rate']:.4f} | "
                f"F1={metrics['f1_score']:.4f} | "
                f"FAR={metrics['false_alarm_rate']:.4f} | "
                f"MCC={metrics['mcc']:.4f} | "
                f"Latency={metrics['avg_latency_ms_per_message']:.6f} ms"
            )

            if best_model_info is None or result_row["f1_score"] > best_model_info["row"]["f1_score"]:
                best_model_info = {
                    "row": result_row,
                    "test_index": X_test.index,
                    "y_test": y_test.copy(),
                    "y_pred": pd.Series(y_pred, index=X_test.index),
                    "y_prob": pd.Series(y_prob, index=X_test.index),
                }

    return all_rows, reports, best_model_info


def per_attack_analysis(df, best_model_info, split_name):
    if best_model_info is None:
        return None

    test_index = best_model_info["test_index"]
    y_pred = best_model_info["y_pred"]

    test_df = df.loc[test_index].copy()
    test_df["predicted_label"] = y_pred.values

    if "attack_type" not in test_df.columns:
        return None

    rows = []

    for attack_type, group in test_df.groupby("attack_type"):
        true_attack_count = int((group["label"] == 1).sum())
        true_normal_count = int((group["label"] == 0).sum())

        if true_attack_count > 0:
            detection_rate = (
                ((group["label"] == 1) & (group["predicted_label"] == 1)).sum()
                / true_attack_count
            )
        else:
            detection_rate = np.nan

        if true_normal_count > 0:
            false_alarm_rate = (
                ((group["label"] == 0) & (group["predicted_label"] == 1)).sum()
                / true_normal_count
            )
        else:
            false_alarm_rate = np.nan

        rows.append(
            {
                "split": split_name,
                "attack_type": attack_type,
                "samples": len(group),
                "normal_samples": true_normal_count,
                "attack_samples": true_attack_count,
                "detection_rate_for_attack_rows": detection_rate,
                "false_alarm_rate_for_normal_rows": false_alarm_rate,
            }
        )

    return pd.DataFrame(rows)


def save_summary_plot(results_df):
    top = results_df.copy()
    top["method"] = top["split"] + " | " + top["feature_set"] + " | " + top["model"]

    top = top.sort_values("f1_score", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(11, 7))

    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["f1_score"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["method"])
    ax.invert_yaxis()
    ax.set_xlabel("F1-score")
    ax.set_title("Top VeReMi External Benchmark Results")
    ax.set_xlim(0, 1.05)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "veremi_external_top_f1_results.png", dpi=300)
    plt.close(fig)


def main():
    print("=" * 80)
    print("TrustV2X - VeReMi External Benchmark Training and Validation")
    print("=" * 80)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_CSV}\n\n"
            "Run this first:\n"
            "python scripts/14e_prepare_veremi_groundtruth_labelled_dataset.py"
        )

    df = pd.read_csv(INPUT_CSV)

    print(f"[INFO] Loaded input CSV: {INPUT_CSV}")
    print(f"[INFO] Original shape: {df.shape}")

    if "label" not in df.columns:
        raise ValueError("Input CSV does not contain a 'label' column.")

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)
    df = df[df["label"].isin([0, 1])].copy()

    print("\n[INFO] Label distribution before sampling:")
    print(df["label"].value_counts())

    if "attack_type" in df.columns:
        print("\n[INFO] Attack type distribution before sampling:")
        print(df["attack_type"].value_counts())

    # Critical leakage check.
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
    present_leakage = [c for c in leakage_columns if c in df.columns]
    print(present_leakage)

    df = sample_if_needed(df)

    print(f"\n[INFO] Shape after optional sampling: {df.shape}")
    print("[INFO] Label distribution after sampling:")
    print(df["label"].value_counts())

    df = engineer_trustv2x_features(df)

    feature_sets = get_feature_sets()
    models = get_models()

    print("\n[INFO] Models to run:")
    print(list(models.keys()))

    all_results = []
    all_reports = []
    per_attack_tables = []

    y = df["label"].astype(int)

    # --------------------------------------------------------
    # Split 1: Random stratified holdout
    # --------------------------------------------------------
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    results, reports, best_model_info = evaluate_split(
        df=df,
        split_name="Random Stratified Holdout",
        train_idx=train_idx,
        test_idx=test_idx,
        feature_sets=feature_sets,
        models=models,
    )

    all_results.extend(results)
    all_reports.extend(reports)

    per_attack = per_attack_analysis(
        df=df,
        best_model_info=best_model_info,
        split_name="Random Stratified Holdout",
    )

    if per_attack is not None:
        per_attack_tables.append(per_attack)

    # --------------------------------------------------------
    # Split 2: Vehicle-disjoint holdout
    # --------------------------------------------------------
    groups = df["vehicle_id"].fillna("unknown").astype(str)

    if groups.nunique() > 1:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=RANDOM_SEED,
        )

        group_train_idx, group_test_idx = next(splitter.split(df, y, groups=groups))

        results, reports, best_model_info = evaluate_split(
            df=df,
            split_name="Vehicle Disjoint Holdout",
            train_idx=group_train_idx,
            test_idx=group_test_idx,
            feature_sets=feature_sets,
            models=models,
        )

        all_results.extend(results)
        all_reports.extend(reports)

        per_attack = per_attack_analysis(
            df=df,
            best_model_info=best_model_info,
            split_name="Vehicle Disjoint Holdout",
        )

        if per_attack is not None:
            per_attack_tables.append(per_attack)
    else:
        print("[WARNING] Vehicle-disjoint split skipped because only one vehicle group exists.")

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        ["split", "f1_score", "balanced_accuracy"],
        ascending=[True, False, False],
    )

    results_path = TABLES_DIR / "veremi_external_ml_results.csv"
    reports_path = TABLES_DIR / "veremi_external_classification_reports.txt"
    per_attack_path = TABLES_DIR / "veremi_external_per_attack_results.csv"

    results_df.to_csv(results_path, index=False)
    reports_path.write_text("\n".join(all_reports), encoding="utf-8")

    if per_attack_tables:
        per_attack_df = pd.concat(per_attack_tables, ignore_index=True)
        per_attack_df.to_csv(per_attack_path, index=False)

    save_summary_plot(results_df)

    print("\n" + "=" * 80)
    print("[OK] VeReMi external benchmark training complete")
    print("=" * 80)

    print(f"[OK] Results saved to: {results_path}")
    print(f"[OK] Reports saved to: {reports_path}")

    if per_attack_tables:
        print(f"[OK] Per-attack results saved to: {per_attack_path}")

    print(f"[OK] Figures saved to: {FIGURES_DIR}")
    print(f"[OK] Models saved to: {MODELS_DIR}")

    print("\nTop results:")
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
    ]

    print(results_df[display_columns].head(20).to_string(index=False))

    print("\nNext step:")
    print("Compare these VeReMi external benchmark results against the controlled SUMO 4x4 TrustV2X results.")


if __name__ == "__main__":
    main()