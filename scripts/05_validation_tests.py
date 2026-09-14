import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================================
# TrustV2X - Step 05
# High-Level Validation Testing
# ============================================================

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "sumo_generated" / "trustv2x_attack_dataset.csv"

RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_MODEL_DIR = PROJECT_ROOT / "results" / "models"

RANDOM_SEED = 42
TEST_SIZE = 0.25


def ensure_directories():
    for folder in [RESULTS_TABLE_DIR, RESULTS_FIGURE_DIR, RESULTS_MODEL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run 01_generate_sumo_cam.py and 02_inject_attacks.py first."
        )

    df = pd.read_csv(DATA_FILE)

    print("=" * 70)
    print("TrustV2X - High-Level Validation Testing")
    print("=" * 70)

    print(f"[INFO] Dataset loaded: {DATA_FILE}")
    print(f"[INFO] Dataset shape: {df.shape}")

    print("\n[INFO] Label distribution:")
    print(df["label"].value_counts())

    print("\n[INFO] Attack type distribution:")
    print(df["attack_type"].value_counts())

    return df


def get_feature_columns(df):
    numeric_features = [
        "message_sequence",
        "timestamp",
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "heading",
        "lane_position",
        "neighbour_count",
        "inter_arrival_time",
        "delta_t",
        "distance_from_previous",
        "estimated_speed_from_position",
        "speed_position_error",
        "heading_change",
        "speed_change",
        "message_rate_5s",
        "identity_cluster_count",
        "sender_message_count",
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

    categorical_features = [
        "lane_id",
        "road_id",
        "message_type",
    ]

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    return numeric_features, categorical_features


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_xgboost_pipeline(df):
    numeric_features, categorical_features = get_feature_columns(df)

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


def false_alarm_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    if (fp + tn) == 0:
        return 0.0

    return fp / (fp + tn)


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


def fit_and_predict(train_df, test_df, experiment_name):
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"].astype(int)

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"].astype(int)

    model = build_xgboost_pipeline(train_df)

    print(f"\n[TRAINING] {experiment_name}")
    print(f"[INFO] Train shape: {train_df.shape}")
    print(f"[INFO] Test shape: {test_df.shape}")

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["experiment"] = experiment_name
    metrics["train_size"] = len(train_df)
    metrics["test_size"] = len(test_df)
    metrics["train_attack_ratio"] = round(train_df["label"].mean(), 4)
    metrics["test_attack_ratio"] = round(test_df["label"].mean(), 4)

    return model, y_test, y_pred, y_prob, test_df.copy(), metrics


def random_holdout_validation(df):
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    return fit_and_predict(train_df, test_df, "Random Stratified Holdout")


def per_attack_evaluation(test_df, y_pred, y_prob):
    results = []

    test_df = test_df.copy()
    test_df["y_pred"] = y_pred
    test_df["y_prob_attack"] = y_prob

    normal_df = test_df[test_df["label"] == 0]

    if len(normal_df) > 0:
        normal_false_alerts = (normal_df["y_pred"] == 1).sum()
        normal_far = normal_false_alerts / len(normal_df)

        results.append({
            "attack_type": "normal",
            "samples": len(normal_df),
            "detected_as_attack": int(normal_false_alerts),
            "missed_or_normal": int(len(normal_df) - normal_false_alerts),
            "detection_rate": np.nan,
            "false_alarm_rate": round(normal_far, 4),
            "mean_attack_probability": round(normal_df["y_prob_attack"].mean(), 4),
        })

    attack_types = sorted(test_df[test_df["label"] == 1]["attack_type"].unique())

    for attack_type in attack_types:
        attack_df = test_df[
            (test_df["label"] == 1)
            & (test_df["attack_type"] == attack_type)
        ]

        if len(attack_df) == 0:
            continue

        detected = (attack_df["y_pred"] == 1).sum()
        missed = (attack_df["y_pred"] == 0).sum()
        detection_rate = detected / len(attack_df)

        results.append({
            "attack_type": attack_type,
            "samples": len(attack_df),
            "detected_as_attack": int(detected),
            "missed_or_normal": int(missed),
            "detection_rate": round(detection_rate, 4),
            "false_alarm_rate": np.nan,
            "mean_attack_probability": round(attack_df["y_prob_attack"].mean(), 4),
        })

    result_df = pd.DataFrame(results)
    result_path = RESULTS_TABLE_DIR / "per_attack_validation_results.csv"
    result_df.to_csv(result_path, index=False)

    print(f"\n[OK] Per-attack validation saved to: {result_path}")
    print(result_df.to_string(index=False))

    return result_df


def severity_evaluation(test_df, y_pred, y_prob):
    results = []

    test_df = test_df.copy()
    test_df["y_pred"] = y_pred
    test_df["y_prob_attack"] = y_prob

    attack_df = test_df[test_df["label"] == 1].copy()

    if "attack_severity" not in attack_df.columns:
        print("[WARNING] attack_severity column not found.")
        return pd.DataFrame()

    for severity in ["low", "medium", "high"]:
        sev_df = attack_df[attack_df["attack_severity"] == severity]

        if len(sev_df) == 0:
            continue

        detected = (sev_df["y_pred"] == 1).sum()
        missed = (sev_df["y_pred"] == 0).sum()

        results.append({
            "attack_severity": severity,
            "samples": len(sev_df),
            "detected_as_attack": int(detected),
            "missed_or_normal": int(missed),
            "detection_rate": round(detected / len(sev_df), 4),
            "mean_attack_probability": round(sev_df["y_prob_attack"].mean(), 4),
        })

    result_df = pd.DataFrame(results)
    result_path = RESULTS_TABLE_DIR / "severity_validation_results.csv"
    result_df.to_csv(result_path, index=False)

    print(f"\n[OK] Severity validation saved to: {result_path}")
    print(result_df.to_string(index=False))

    return result_df


def unseen_vehicle_validation(df):
    if "vehicle_id" not in df.columns:
        print("[WARNING] vehicle_id column not found. Skipping unseen-vehicle validation.")
        return None

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    X = df.drop(columns=["label"])
    y = df["label"].astype(int)
    groups = df["vehicle_id"].astype(str)

    train_idx, test_idx = next(splitter.split(X, y, groups))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_vehicles = set(train_df["vehicle_id"].astype(str))
    test_vehicles = set(test_df["vehicle_id"].astype(str))
    overlap = train_vehicles.intersection(test_vehicles)

    model, y_test, y_pred, y_prob, test_df_out, metrics = fit_and_predict(
        train_df,
        test_df,
        "Unseen Vehicle Split",
    )

    metrics["train_unique_vehicles"] = len(train_vehicles)
    metrics["test_unique_vehicles"] = len(test_vehicles)
    metrics["vehicle_overlap"] = len(overlap)

    return metrics


def time_based_validation(df):
    if "timestamp" not in df.columns:
        print("[WARNING] timestamp column not found. Skipping time-based validation.")
        return None

    sorted_df = df.sort_values("timestamp").reset_index(drop=True)

    threshold = sorted_df["timestamp"].quantile(0.75)

    train_df = sorted_df[sorted_df["timestamp"] <= threshold].copy()
    test_df = sorted_df[sorted_df["timestamp"] > threshold].copy()

    if train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2:
        print("[WARNING] Time-based split does not contain both classes. Skipping.")
        return None

    model, y_test, y_pred, y_prob, test_df_out, metrics = fit_and_predict(
        train_df,
        test_df,
        "Time-Based Split",
    )

    metrics["timestamp_threshold"] = round(threshold, 3)
    metrics["train_time_min"] = round(train_df["timestamp"].min(), 3)
    metrics["train_time_max"] = round(train_df["timestamp"].max(), 3)
    metrics["test_time_min"] = round(test_df["timestamp"].min(), 3)
    metrics["test_time_max"] = round(test_df["timestamp"].max(), 3)

    return metrics


def rare_attack_ratio_validation(model, heldout_df):
    """
    Use the already trained random-holdout model and create deployment-like
    test subsets with lower attack rates.
    """
    results = []

    normal_df = heldout_df[heldout_df["label"] == 0].copy()
    attack_df = heldout_df[heldout_df["label"] == 1].copy()

    ratios = [
        ("Current 70/30", 0.30),
        ("Low attack 95/5", 0.05),
        ("Rare attack 99/1", 0.01),
    ]

    for scenario_name, attack_ratio in ratios:
        normal_ratio = 1 - attack_ratio

        max_total_from_normal = int(len(normal_df) / normal_ratio)
        max_total_from_attack = int(len(attack_df) / attack_ratio)

        total_size = min(max_total_from_normal, max_total_from_attack, len(heldout_df))

        n_attack = max(1, int(total_size * attack_ratio))
        n_normal = total_size - n_attack

        sampled_normal = normal_df.sample(
            n=min(n_normal, len(normal_df)),
            random_state=RANDOM_SEED,
        )

        sampled_attack = attack_df.sample(
            n=min(n_attack, len(attack_df)),
            random_state=RANDOM_SEED,
        )

        scenario_df = pd.concat([sampled_normal, sampled_attack], ignore_index=True)
        scenario_df = scenario_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        X_test = scenario_df.drop(columns=["label"])
        y_test = scenario_df["label"].astype(int)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["scenario"] = scenario_name
        metrics["test_size"] = len(scenario_df)
        metrics["normal_samples"] = int((scenario_df["label"] == 0).sum())
        metrics["attack_samples"] = int((scenario_df["label"] == 1).sum())
        metrics["attack_ratio"] = round(scenario_df["label"].mean(), 4)

        results.append(metrics)

    result_df = pd.DataFrame(results)
    result_path = RESULTS_TABLE_DIR / "rare_attack_ratio_validation_results.csv"
    result_df.to_csv(result_path, index=False)

    print(f"\n[OK] Rare attack-ratio validation saved to: {result_path}")
    print(result_df.to_string(index=False))

    return result_df


def feature_importance_analysis(model):
    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })

        importance_df = importance_df.sort_values(
            by="importance",
            ascending=False,
        ).reset_index(drop=True)

        importance_path = RESULTS_TABLE_DIR / "xgboost_feature_importance_top30.csv"
        importance_df.head(30).to_csv(importance_path, index=False)

        print(f"\n[OK] Feature importance saved to: {importance_path}")
        print(importance_df.head(30).to_string(index=False))

        plot_feature_importance(importance_df.head(20))

        return importance_df

    except Exception as e:
        print(f"[WARNING] Feature importance failed: {e}")
        return pd.DataFrame()


def plot_feature_importance(importance_df):
    fig, ax = plt.subplots(figsize=(8, 7))

    plot_df = importance_df.sort_values("importance", ascending=True)

    ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_title("Top TrustV2X Feature Importance - XGBoost")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    plt.tight_layout()

    figure_path = RESULTS_FIGURE_DIR / "xgboost_feature_importance_top20.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"[OK] Feature importance figure saved to: {figure_path}")


def plot_per_attack_results(per_attack_df):
    attack_df = per_attack_df[per_attack_df["attack_type"] != "normal"].copy()

    if attack_df.empty:
        return

    attack_df = attack_df.sort_values("detection_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(attack_df["attack_type"], attack_df["detection_rate"])
    ax.set_title("Per-Attack Detection Rate")
    ax.set_xlabel("Detection Rate")
    ax.set_ylabel("Attack Type")
    ax.set_xlim(0, 1.05)

    plt.tight_layout()

    figure_path = RESULTS_FIGURE_DIR / "per_attack_detection_rate.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"[OK] Per-attack detection figure saved to: {figure_path}")


def plot_severity_results(severity_df):
    if severity_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(severity_df["attack_severity"], severity_df["detection_rate"])
    ax.set_title("Detection Rate by Attack Severity")
    ax.set_xlabel("Attack Severity")
    ax.set_ylabel("Detection Rate")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    figure_path = RESULTS_FIGURE_DIR / "severity_detection_rate.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"[OK] Severity detection figure saved to: {figure_path}")


def main():
    ensure_directories()

    df = load_dataset()

    # ------------------------------------------------------------
    # 1. Random holdout validation
    # ------------------------------------------------------------
    random_model, y_test, y_pred, y_prob, test_df, random_metrics = random_holdout_validation(df)

    overall_results = [random_metrics]

    # ------------------------------------------------------------
    # 2. Per-attack validation
    # ------------------------------------------------------------
    per_attack_df = per_attack_evaluation(test_df, y_pred, y_prob)
    plot_per_attack_results(per_attack_df)

    # ------------------------------------------------------------
    # 3. Severity validation
    # ------------------------------------------------------------
    severity_df = severity_evaluation(test_df, y_pred, y_prob)
    plot_severity_results(severity_df)

    # ------------------------------------------------------------
    # 4. Unseen vehicle validation
    # ------------------------------------------------------------
    unseen_vehicle_metrics = unseen_vehicle_validation(df)

    if unseen_vehicle_metrics is not None:
        overall_results.append(unseen_vehicle_metrics)

    # ------------------------------------------------------------
    # 5. Time-based validation
    # ------------------------------------------------------------
    time_metrics = time_based_validation(df)

    if time_metrics is not None:
        overall_results.append(time_metrics)

    # ------------------------------------------------------------
    # 6. Rare attack-ratio validation
    # ------------------------------------------------------------
    rare_attack_ratio_validation(random_model, test_df)

    # ------------------------------------------------------------
    # 7. Feature importance
    # ------------------------------------------------------------
    feature_importance_analysis(random_model)

    # ------------------------------------------------------------
    # Save overall validation summary
    # ------------------------------------------------------------
    overall_df = pd.DataFrame(overall_results)

    overall_path = RESULTS_TABLE_DIR / "high_level_validation_summary.csv"
    overall_df.to_csv(overall_path, index=False)

    print("\n" + "=" * 70)
    print("[OK] High-level validation complete")
    print("=" * 70)

    print(f"[OK] Overall validation summary saved to: {overall_path}")

    print("\nOverall validation summary:")
    print(overall_df.to_string(index=False))

    print("\nGenerated validation outputs:")
    print(f"- {RESULTS_TABLE_DIR / 'high_level_validation_summary.csv'}")
    print(f"- {RESULTS_TABLE_DIR / 'per_attack_validation_results.csv'}")
    print(f"- {RESULTS_TABLE_DIR / 'severity_validation_results.csv'}")
    print(f"- {RESULTS_TABLE_DIR / 'rare_attack_ratio_validation_results.csv'}")
    print(f"- {RESULTS_TABLE_DIR / 'xgboost_feature_importance_top30.csv'}")
    print(f"- {RESULTS_FIGURE_DIR / 'per_attack_detection_rate.png'}")
    print(f"- {RESULTS_FIGURE_DIR / 'severity_detection_rate.png'}")
    print(f"- {RESULTS_FIGURE_DIR / 'xgboost_feature_importance_top20.png'}")

    print("\nNext step after this:")
    print("Review weak attack categories, then build TrustV2X scoring layer.")


if __name__ == "__main__":
    main()