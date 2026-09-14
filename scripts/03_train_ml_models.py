import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


# ============================================================
# TrustV2X - Step 03
# Train classical ML models for binary CAM misbehavior detection
# ============================================================

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "sumo_generated" / "trustv2x_attack_dataset.csv"

RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_MODEL_DIR = PROJECT_ROOT / "results" / "models"

RESULTS_FILE = RESULTS_TABLE_DIR / "ml_binary_results.csv"
REPORT_FILE = RESULTS_TABLE_DIR / "ml_classification_reports.txt"


RANDOM_SEED = 42
TEST_SIZE = 0.25


def ensure_directories():
    for folder in [RESULTS_TABLE_DIR, RESULTS_FIGURE_DIR, RESULTS_MODEL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run scripts/02_inject_attacks.py first."
        )

    df = pd.read_csv(DATA_FILE)

    print("=" * 70)
    print("TrustV2X - Classical ML Training")
    print("=" * 70)

    print(f"[INFO] Dataset loaded: {DATA_FILE}")
    print(f"[INFO] Dataset shape: {df.shape}")
    print("\n[INFO] Attack type distribution:")
    print(df["attack_type"].value_counts())

    return df

def build_feature_sets(df):
    """
    We train two versions:

    1. Raw_CAM_Features:
       Uses only direct CAM-like mobility and context fields.

    2. TrustV2X_Enhanced_Features:
       Uses raw CAM features plus plausibility, temporal, identity,
       message-rate and trust-rule indicators.
    """

    raw_numeric_features = [
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
    ]

    raw_categorical_features = [
        "lane_id",
        "road_id",
        "message_type",
    ]

    enhanced_numeric_features = raw_numeric_features + [
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

    feature_sets = {
        "Raw_CAM_Features": {
            "numeric": [c for c in raw_numeric_features if c in df.columns],
            "categorical": [c for c in raw_categorical_features if c in df.columns],
        },
        "TrustV2X_Enhanced_Features": {
            "numeric": [c for c in enhanced_numeric_features if c in df.columns],
            "categorical": [c for c in raw_categorical_features if c in df.columns],
        },
    }

    return feature_sets


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_SEED,
        ),
    }

    # Optional: XGBoost
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
        )
    except Exception as e:
        print(f"[WARNING] XGBoost not available. Skipping. Reason: {e}")

    # Optional: LightGBM
    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            verbose=-1,
        )
    except Exception as e:
        print(f"[WARNING] LightGBM not available. Skipping. Reason: {e}")

    return models


def create_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def false_alarm_rate(y_true, y_pred):
    """
    False Alarm Rate = FP / (FP + TN)
    Where:
    FP = normal messages wrongly predicted as attack
    TN = normal messages correctly predicted as normal
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def detection_rate(y_true, y_pred):
    """
    Detection Rate = TP / (TP + FN)
    Same as recall for the attack class.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def plot_confusion_matrix(cm, model_name, feature_set_name):
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm)

    ax.set_title(f"{model_name}\n{feature_set_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Attack"])
    ax.set_yticklabels(["Normal", "Attack"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    safe_model_name = model_name.lower().replace(" ", "_")
    safe_feature_name = feature_set_name.lower().replace(" ", "_")

    figure_path = RESULTS_FIGURE_DIR / f"cm_{safe_model_name}_{safe_feature_name}.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    return figure_path


def train_and_evaluate(df):
    feature_sets = build_feature_sets(df)
    models = get_models()

    y = df["label"].astype(int)

    all_results = []
    all_reports = []

    for feature_set_name, features in feature_sets.items():
        numeric_features = features["numeric"]
        categorical_features = features["categorical"]

        selected_features = numeric_features + categorical_features

        print("\n" + "=" * 70)
        print(f"[FEATURE SET] {feature_set_name}")
        print("=" * 70)
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")

        X = df[selected_features]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y,
        )

        for model_name, model in models.items():
            print(f"\n[TRAINING] {model_name} using {feature_set_name}")

            preprocessor = create_preprocessor(numeric_features, categorical_features)

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model),
                ]
            )

            start_train = time.perf_counter()
            pipeline.fit(X_train, y_train)
            train_time = time.perf_counter() - start_train

            start_pred = time.perf_counter()
            y_pred = pipeline.predict(X_test)
            pred_time = time.perf_counter() - start_pred

            avg_latency_ms = (pred_time / len(X_test)) * 1000

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            far = false_alarm_rate(y_test, y_pred)
            dr = detection_rate(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)
            cm_path = plot_confusion_matrix(cm, model_name, feature_set_name)

            result_row = {
                "feature_set": feature_set_name,
                "model": model_name,
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall_detection_rate": round(recall, 4),
                "f1_score": round(f1, 4),
                "false_alarm_rate": round(far, 4),
                "detection_rate": round(dr, 4),
                "training_time_seconds": round(train_time, 4),
                "prediction_time_seconds": round(pred_time, 4),
                "avg_latency_ms_per_message": round(avg_latency_ms, 6),
                "confusion_matrix_file": str(cm_path),
            }

            all_results.append(result_row)

            report = classification_report(
                y_test,
                y_pred,
                target_names=["Normal", "Attack"],
                zero_division=0,
            )

            report_text = (
                f"\n{'=' * 70}\n"
                f"Feature Set: {feature_set_name}\n"
                f"Model: {model_name}\n"
                f"{'=' * 70}\n"
                f"Confusion Matrix:\n{cm}\n\n"
                f"Classification Report:\n{report}\n"
            )

            all_reports.append(report_text)

            safe_model_name = model_name.lower().replace(" ", "_")
            safe_feature_name = feature_set_name.lower().replace(" ", "_")
            model_path = RESULTS_MODEL_DIR / f"{safe_model_name}_{safe_feature_name}.joblib"

            joblib.dump(pipeline, model_path)

            print(
                f"[RESULT] {model_name} | {feature_set_name} | "
                f"Acc={accuracy:.4f}, Prec={precision:.4f}, "
                f"Recall={recall:.4f}, F1={f1:.4f}, FAR={far:.4f}, "
                f"Latency={avg_latency_ms:.6f} ms"
            )

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(
        by=["feature_set", "f1_score", "accuracy"],
        ascending=[True, False, False],
    )

    results_df.to_csv(RESULTS_FILE, index=False)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(all_reports))

    print("\n" + "=" * 70)
    print("[OK] ML training complete")
    print("=" * 70)
    print(f"[OK] Results saved to: {RESULTS_FILE}")
    print(f"[OK] Reports saved to: {REPORT_FILE}")
    print(f"[OK] Models saved to: {RESULTS_MODEL_DIR}")

    print("\nTop results:")
    print(results_df.head(10).to_string(index=False))


def main():
    ensure_directories()
    df = load_dataset()
    train_and_evaluate(df)

    print("\nDone. Next step: train deep learning models using 04_train_dl_models.py")


if __name__ == "__main__":
    main()