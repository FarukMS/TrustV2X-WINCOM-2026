import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================================
# TrustV2X - Step 06
# Ablation and Hard Validation Testing
# ============================================================

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "sumo_generated" / "trustv2x_attack_dataset.csv"

RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

RANDOM_SEED = 42
TEST_SIZE = 0.25


def ensure_directories():
    RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run 01_generate_sumo_cam.py and 02_inject_attacks.py first."
        )

    df = pd.read_csv(DATA_FILE)

    print("=" * 70)
    print("TrustV2X - Ablation and Hard Validation Testing")
    print("=" * 70)

    print(f"[INFO] Dataset loaded: {DATA_FILE}")
    print(f"[INFO] Dataset shape: {df.shape}")

    print("\n[INFO] Label distribution:")
    print(df["label"].value_counts())

    print("\n[INFO] Attack type distribution:")
    print(df["attack_type"].value_counts())

    return df


def available_columns(df, columns):
    return [c for c in columns if c in df.columns]


def build_feature_sets(df):
    categorical_features = [
        "lane_id",
        "road_id",
        "message_type",
    ]

    core_cam_numeric = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "heading",
        "lane_position",
        "neighbour_count",
    ]

    raw_timing_numeric = core_cam_numeric + [
        "timestamp",
        "message_sequence",
        "inter_arrival_time",
    ]

    continuous_plausibility_numeric = raw_timing_numeric + [
        "delta_t",
        "distance_from_previous",
        "estimated_speed_from_position",
        "speed_position_error",
        "heading_change",
        "speed_change",
        "message_rate_5s",
        "identity_cluster_count",
        "sender_message_count",
    ]

    binary_rule_flags = [
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
        "trustv2x_rule_violation_count",
    ]

    full_trustv2x_numeric = continuous_plausibility_numeric + binary_rule_flags

    timing_heavy_features = [
        "timestamp",
        "message_sequence",
        "inter_arrival_time",
        "delta_t",
        "message_rate_5s",
    ]

    no_timing_numeric = [
        c for c in full_trustv2x_numeric
        if c not in timing_heavy_features
    ]

    no_rule_numeric = [
        c for c in full_trustv2x_numeric
        if c not in binary_rule_flags
    ]

    hard_no_timing_no_rules_numeric = [
        c for c in full_trustv2x_numeric
        if c not in timing_heavy_features and c not in binary_rule_flags
    ]

    feature_sets = {
        "Core_CAM_Only": {
            "numeric": available_columns(df, core_cam_numeric),
            "categorical": available_columns(df, categorical_features),
            "description": "Only basic CAM mobility/context fields. No timing artefacts, no plausibility features, no rule flags.",
        },
        "Raw_CAM_With_Timing": {
            "numeric": available_columns(df, raw_timing_numeric),
            "categorical": available_columns(df, categorical_features),
            "description": "Raw CAM fields plus timestamp, message sequence and inter-arrival time.",
        },
        "TrustV2X_No_Rule_Flags": {
            "numeric": available_columns(df, no_rule_numeric),
            "categorical": available_columns(df, categorical_features),
            "description": "TrustV2X continuous features only. Binary rule flags removed.",
        },
        "TrustV2X_No_Timing_Heavy": {
            "numeric": available_columns(df, no_timing_numeric),
            "categorical": available_columns(df, categorical_features),
            "description": "Full TrustV2X features except dominant timing-heavy features.",
        },
        "TrustV2X_Hard_No_Timing_No_Rules": {
            "numeric": available_columns(df, hard_no_timing_no_rules_numeric),
            "categorical": available_columns(df, categorical_features),
            "description": "Hardest setting: removes timing-heavy features and binary rule flags.",
        },
        "TrustV2X_Full": {
            "numeric": available_columns(df, full_trustv2x_numeric),
            "categorical": available_columns(df, categorical_features),
            "description": "Full TrustV2X enhanced feature set.",
        },
    }

    return feature_sets


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_xgboost_pipeline(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", make_onehot_encoder(), categorical_features),
        ],
        remainder="drop",
    )

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return model


def false_alarm_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    if (fp + tn) == 0:
        return 0.0

    return fp / (fp + tn)


def safe_roc_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


def safe_pr_auc(y_true, y_prob):
    try:
        return average_precision_score(y_true, y_prob)
    except Exception:
        return np.nan


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_detection_rate": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "false_alarm_rate": round(false_alarm_rate(y_true, y_pred), 4),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
        "roc_auc": round(safe_roc_auc(y_true, y_prob), 4),
        "pr_auc": round(safe_pr_auc(y_true, y_prob), 4),
    }


def train_and_evaluate_feature_set(
    feature_set_name,
    feature_info,
    train_df,
    test_df,
):
    numeric_features = feature_info["numeric"]
    categorical_features = feature_info["categorical"]

    selected_features = numeric_features + categorical_features

    X_train = train_df[selected_features]
    y_train = train_df["label"].astype(int)

    X_test = test_df[selected_features]
    y_test = test_df["label"].astype(int)

    model = build_xgboost_pipeline(numeric_features, categorical_features)

    print("\n" + "=" * 70)
    print(f"[FEATURE SET] {feature_set_name}")
    print("=" * 70)
    print(f"[DESCRIPTION] {feature_info['description']}")
    print(f"[INFO] Numeric features: {numeric_features}")
    print(f"[INFO] Categorical features: {categorical_features}")
    print(f"[INFO] Total input columns before encoding: {len(selected_features)}")

    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_prob = model.predict_proba(X_test)[:, 1]
    pred_time = time.perf_counter() - start_pred

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)

    metrics["feature_set"] = feature_set_name
    metrics["description"] = feature_info["description"]
    metrics["numeric_feature_count"] = len(numeric_features)
    metrics["categorical_feature_count"] = len(categorical_features)
    metrics["total_input_columns_before_encoding"] = len(selected_features)
    metrics["train_size"] = len(train_df)
    metrics["test_size"] = len(test_df)
    metrics["training_time_seconds"] = round(train_time, 4)
    metrics["prediction_time_seconds"] = round(pred_time, 4)
    metrics["avg_latency_ms_per_message"] = round((pred_time / len(test_df)) * 1000, 6)

    print(
        f"[RESULT] {feature_set_name} | "
        f"Acc={metrics['accuracy']} | "
        f"BalAcc={metrics['balanced_accuracy']} | "
        f"Prec={metrics['precision']} | "
        f"Recall={metrics['recall_detection_rate']} | "
        f"F1={metrics['f1_score']} | "
        f"FAR={metrics['false_alarm_rate']} | "
        f"MCC={metrics['mcc']} | "
        f"ROC-AUC={metrics['roc_auc']} | "
        f"PR-AUC={metrics['pr_auc']} | "
        f"Latency={metrics['avg_latency_ms_per_message']} ms"
    )

    prediction_df = test_df.copy()
    prediction_df["feature_set"] = feature_set_name
    prediction_df["y_pred"] = y_pred
    prediction_df["y_prob_attack"] = y_prob

    return model, metrics, prediction_df


def per_attack_metrics(prediction_df):
    results = []

    feature_set_name = prediction_df["feature_set"].iloc[0]

    normal_df = prediction_df[prediction_df["label"] == 0]

    if len(normal_df) > 0:
        false_alerts = (normal_df["y_pred"] == 1).sum()

        results.append({
            "feature_set": feature_set_name,
            "attack_type": "normal",
            "samples": len(normal_df),
            "detected_as_attack": int(false_alerts),
            "missed_or_normal": int(len(normal_df) - false_alerts),
            "detection_rate": np.nan,
            "false_alarm_rate": round(false_alerts / len(normal_df), 4),
            "mean_attack_probability": round(normal_df["y_prob_attack"].mean(), 4),
        })

    for attack_type in sorted(prediction_df[prediction_df["label"] == 1]["attack_type"].unique()):
        attack_df = prediction_df[
            (prediction_df["label"] == 1)
            & (prediction_df["attack_type"] == attack_type)
        ]

        detected = (attack_df["y_pred"] == 1).sum()
        missed = (attack_df["y_pred"] == 0).sum()

        results.append({
            "feature_set": feature_set_name,
            "attack_type": attack_type,
            "samples": len(attack_df),
            "detected_as_attack": int(detected),
            "missed_or_normal": int(missed),
            "detection_rate": round(detected / len(attack_df), 4),
            "false_alarm_rate": np.nan,
            "mean_attack_probability": round(attack_df["y_prob_attack"].mean(), 4),
        })

    return results


def severity_metrics(prediction_df):
    results = []

    feature_set_name = prediction_df["feature_set"].iloc[0]

    attack_df = prediction_df[prediction_df["label"] == 1].copy()

    if "attack_severity" not in attack_df.columns:
        return results

    for severity in ["low", "medium", "high"]:
        sev_df = attack_df[attack_df["attack_severity"] == severity]

        if len(sev_df) == 0:
            continue

        detected = (sev_df["y_pred"] == 1).sum()
        missed = (sev_df["y_pred"] == 0).sum()

        results.append({
            "feature_set": feature_set_name,
            "attack_severity": severity,
            "samples": len(sev_df),
            "detected_as_attack": int(detected),
            "missed_or_normal": int(missed),
            "detection_rate": round(detected / len(sev_df), 4),
            "mean_attack_probability": round(sev_df["y_prob_attack"].mean(), 4),
        })

    return results


def plot_ablation_summary(summary_df):
    plot_df = summary_df.sort_values("f1_score", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    y_positions = np.arange(len(plot_df))

    ax.barh(y_positions - 0.2, plot_df["f1_score"], height=0.2, label="F1-score")
    ax.barh(y_positions, plot_df["recall_detection_rate"], height=0.2, label="Recall")
    ax.barh(y_positions + 0.2, plot_df["balanced_accuracy"], height=0.2, label="Balanced Accuracy")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["feature_set"])
    ax.set_xlabel("Score")
    ax.set_title("TrustV2X Ablation Comparison")
    ax.set_xlim(0, 1.05)
    ax.legend()

    plt.tight_layout()

    figure_path = RESULTS_FIGURE_DIR / "ablation_f1_recall_balanced_accuracy.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"[OK] Ablation summary figure saved to: {figure_path}")


def plot_far_summary(summary_df):
    plot_df = summary_df.sort_values("false_alarm_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(plot_df["feature_set"], plot_df["false_alarm_rate"])
    ax.set_xlabel("False Alarm Rate")
    ax.set_title("False Alarm Rate by Feature Set")

    plt.tight_layout()

    figure_path = RESULTS_FIGURE_DIR / "ablation_false_alarm_rate.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"[OK] FAR summary figure saved to: {figure_path}")


def plot_per_attack_heatmap(per_attack_df):
    attack_df = per_attack_df[per_attack_df["attack_type"] != "normal"].copy()

    if attack_df.empty:
        return

    pivot_df = attack_df.pivot(
        index="attack_type",
        columns="feature_set",
        values="detection_rate",
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    im = ax.imshow(pivot_df.values, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))

    ax.set_xticklabels(pivot_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot_df.index)

    ax.set_title("Per-Attack Detection Rate Across Feature Sets")

    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    figure_path = RESULTS_FIGURE_DIR / "ablation_per_attack_detection_heatmap.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"[OK] Per-attack heatmap saved to: {figure_path}")


def main():
    ensure_directories()

    df = load_dataset()

    feature_sets = build_feature_sets(df)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    print("\n[INFO] Common train/test split created")
    print(f"[INFO] Train shape: {train_df.shape}")
    print(f"[INFO] Test shape: {test_df.shape}")
    print(f"[INFO] Train attack ratio: {train_df['label'].mean():.4f}")
    print(f"[INFO] Test attack ratio: {test_df['label'].mean():.4f}")

    summary_results = []
    all_per_attack_results = []
    all_severity_results = []
    feature_count_rows = []

    for feature_set_name, feature_info in feature_sets.items():
        model, metrics, prediction_df = train_and_evaluate_feature_set(
            feature_set_name,
            feature_info,
            train_df,
            test_df,
        )

        summary_results.append(metrics)

        all_per_attack_results.extend(per_attack_metrics(prediction_df))
        all_severity_results.extend(severity_metrics(prediction_df))

        feature_count_rows.append({
            "feature_set": feature_set_name,
            "description": feature_info["description"],
            "numeric_features": ", ".join(feature_info["numeric"]),
            "categorical_features": ", ".join(feature_info["categorical"]),
            "numeric_feature_count": len(feature_info["numeric"]),
            "categorical_feature_count": len(feature_info["categorical"]),
            "total_input_columns_before_encoding": len(feature_info["numeric"]) + len(feature_info["categorical"]),
        })

    summary_df = pd.DataFrame(summary_results)
    summary_df = summary_df.sort_values("f1_score", ascending=False)

    per_attack_df = pd.DataFrame(all_per_attack_results)
    severity_df = pd.DataFrame(all_severity_results)
    feature_count_df = pd.DataFrame(feature_count_rows)

    summary_path = RESULTS_TABLE_DIR / "ablation_hard_test_summary.csv"
    per_attack_path = RESULTS_TABLE_DIR / "ablation_per_attack_results.csv"
    severity_path = RESULTS_TABLE_DIR / "ablation_severity_results.csv"
    feature_count_path = RESULTS_TABLE_DIR / "ablation_feature_set_definitions.csv"

    summary_df.to_csv(summary_path, index=False)
    per_attack_df.to_csv(per_attack_path, index=False)
    severity_df.to_csv(severity_path, index=False)
    feature_count_df.to_csv(feature_count_path, index=False)

    plot_ablation_summary(summary_df)
    plot_far_summary(summary_df)
    plot_per_attack_heatmap(per_attack_df)

    print("\n" + "=" * 70)
    print("[OK] Ablation and hard validation complete")
    print("=" * 70)

    print(f"[OK] Summary saved to: {summary_path}")
    print(f"[OK] Per-attack ablation saved to: {per_attack_path}")
    print(f"[OK] Severity ablation saved to: {severity_path}")
    print(f"[OK] Feature set definitions saved to: {feature_count_path}")

    print("\nAblation hard-test summary:")
    print(summary_df.to_string(index=False))

    print("\nNext interpretation guide:")
    print("- If TrustV2X_Full is best but No_Rule_Flags remains strong, the model is not only relying on binary rules.")
    print("- If No_Timing_Heavy remains strong, the model is not only relying on timing artefacts.")
    print("- If Core_CAM_Only is much weaker, the proposed TrustV2X feature engineering is justified.")
    print("- If performance collapses without timing features, the dataset may still contain timing artefact bias.")


if __name__ == "__main__":
    main()