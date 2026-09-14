import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================================
# TrustV2X - Step 07
# Trust Scoring and Decision Layer
# ============================================================

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "sumo_generated" / "trustv2x_attack_dataset.csv"

RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_MODEL_DIR = PROJECT_ROOT / "results" / "models"

SCORING_RESULTS_FILE = RESULTS_TABLE_DIR / "trustv2x_scoring_results.csv"
DECISION_SUMMARY_FILE = RESULTS_TABLE_DIR / "trustv2x_decision_summary.csv"
DECISION_BY_ATTACK_FILE = RESULTS_TABLE_DIR / "trustv2x_decision_by_attack_type.csv"
DECISION_BY_SEVERITY_FILE = RESULTS_TABLE_DIR / "trustv2x_decision_by_severity.csv"
TRUST_SCORE_PROFILE_FILE = RESULTS_TABLE_DIR / "trustv2x_score_profile.csv"

MODEL_FILE = RESULTS_MODEL_DIR / "trustv2x_xgboost_no_rule_flags_scoring_engine.joblib"

RANDOM_SEED = 42
TEST_SIZE = 0.25


# Decision thresholds
ACCEPT_THRESHOLD = 0.75
REJECT_THRESHOLD = 0.40


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
    print("TrustV2X - Trust Scoring and Decision Layer")
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


def get_no_rule_flag_features(df):
    """
    Main ML engine uses TrustV2X continuous features only.
    Binary rule flags are deliberately excluded from the ML model and used
    separately in the trust scoring layer.
    """

    numeric_features = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "heading",
        "lane_position",
        "neighbour_count",
        "timestamp",
        "message_sequence",
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
    ]

    categorical_features = [
        "lane_id",
        "road_id",
        "message_type",
    ]

    numeric_features = available_columns(df, numeric_features)
    categorical_features = available_columns(df, categorical_features)

    return numeric_features, categorical_features


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_scoring_model(df):
    numeric_features, categorical_features = get_no_rule_flag_features(df)

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

    selected_features = numeric_features + categorical_features

    return model, selected_features, numeric_features, categorical_features


def clip01(series):
    return np.clip(series, 0, 1)


def normalised_ratio(series, denominator):
    return clip01(series / denominator)


def calculate_time_gap_risk(delta_t):
    """
    CAM messages are expected to be temporally consistent.
    Very small or very large gaps are suspicious.
    First messages often have delta_t = 0 after feature construction, so those are not penalised.
    """
    delta_t = pd.Series(delta_t).fillna(0)

    risk = np.zeros(len(delta_t))

    risk = np.where(
        delta_t == 0,
        0,
        np.where(
            (delta_t < 0.2) | (delta_t > 5),
            1,
            np.minimum(np.abs(delta_t - 1) / 4, 1),
        ),
    )

    return risk


def calculate_plausibility_risk(df):
    """
    Plausibility risk is computed from continuous TrustV2X indicators.
    Higher value = less plausible.
    """

    speed_position_risk = normalised_ratio(
        df.get("speed_position_error", pd.Series(0, index=df.index)).fillna(0),
        60,
    )

    distance_jump_risk = normalised_ratio(
        df.get("distance_from_previous", pd.Series(0, index=df.index)).fillna(0),
        150,
    )

    heading_risk = normalised_ratio(
        df.get("heading_change", pd.Series(0, index=df.index)).fillna(0),
        180,
    )

    speed_change_risk = normalised_ratio(
        df.get("speed_change", pd.Series(0, index=df.index)).fillna(0),
        30,
    )

    message_rate = df.get("message_rate_5s", pd.Series(1, index=df.index)).fillna(1)
    message_rate_risk = clip01(np.maximum(message_rate - 5, 0) / 15)

    identity_cluster = df.get("identity_cluster_count", pd.Series(1, index=df.index)).fillna(1)
    identity_risk = clip01(np.maximum(identity_cluster - 1, 0) / 5)

    delta_t = df.get("delta_t", pd.Series(0, index=df.index)).fillna(0)
    time_gap_risk = calculate_time_gap_risk(delta_t)

    plausibility_risk = (
        0.25 * speed_position_risk
        + 0.20 * distance_jump_risk
        + 0.15 * heading_risk
        + 0.15 * speed_change_risk
        + 0.10 * message_rate_risk
        + 0.10 * identity_risk
        + 0.05 * time_gap_risk
    )

    plausibility_risk = clip01(plausibility_risk)

    return plausibility_risk


def calculate_rule_risk(df):
    rule_columns = [
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

    available_rule_columns = [c for c in rule_columns if c in df.columns]

    if "trustv2x_rule_violation_count" in df.columns:
        rule_count = df["trustv2x_rule_violation_count"].fillna(0)
    elif available_rule_columns:
        rule_count = df[available_rule_columns].fillna(0).sum(axis=1)
    else:
        rule_count = pd.Series(0, index=df.index)

    rule_risk = clip01(rule_count / 5)

    return rule_risk, rule_count


def calculate_trust_score(df, ml_attack_probability):
    """
    Final risk = weighted combination of:
    - ML attack probability
    - rule violation risk
    - plausibility risk

    Trust score = 1 - final risk.
    """

    scoring_df = df.copy()

    scoring_df["ml_attack_probability"] = ml_attack_probability

    scoring_df["plausibility_risk"] = calculate_plausibility_risk(scoring_df)
    scoring_df["plausibility_score"] = 1 - scoring_df["plausibility_risk"]

    scoring_df["rule_risk"], scoring_df["rule_violation_count"] = calculate_rule_risk(scoring_df)

    scoring_df["combined_risk_score"] = (
        0.65 * scoring_df["ml_attack_probability"]
        + 0.20 * scoring_df["rule_risk"]
        + 0.15 * scoring_df["plausibility_risk"]
    )

    scoring_df["combined_risk_score"] = clip01(scoring_df["combined_risk_score"])
    scoring_df["trust_score"] = 1 - scoring_df["combined_risk_score"]

    scoring_df["ml_confidence"] = np.abs(scoring_df["ml_attack_probability"] - 0.5) * 2

    scoring_df["trustv2x_decision"] = scoring_df["trust_score"].apply(make_decision)

    scoring_df["decision_binary_attack"] = scoring_df["trustv2x_decision"].apply(
        lambda x: 0 if x == "ACCEPT" else 1
    )

    return scoring_df


def make_decision(trust_score):
    if trust_score >= ACCEPT_THRESHOLD:
        return "ACCEPT"
    elif trust_score >= REJECT_THRESHOLD:
        return "FLAG_FOR_VERIFICATION"
    else:
        return "REJECT_ALERT"


def compute_decision_metrics(scoring_df):
    y_true = scoring_df["label"].astype(int)
    y_pred = scoring_df["decision_binary_attack"].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_detection_rate": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "false_alarm_rate": round(far, 4),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
        "true_normal_accept": int(tn),
        "false_alarm_normal_flagged_or_rejected": int(fp),
        "missed_attack_accepted": int(fn),
        "detected_attack_flagged_or_rejected": int(tp),
        "test_size": len(scoring_df),
    }

    return metrics


def save_decision_summaries(scoring_df, metrics):
    # Overall summary
    decision_counts = (
        scoring_df.groupby(["label", "attack_type", "trustv2x_decision"])
        .size()
        .reset_index(name="count")
        .sort_values(["label", "attack_type", "trustv2x_decision"])
    )

    decision_counts.to_csv(DECISION_SUMMARY_FILE, index=False)

    # Attack type summary
    by_attack = (
        scoring_df.groupby("attack_type")
        .agg(
            samples=("attack_type", "size"),
            true_label=("label", "first"),
            mean_ml_attack_probability=("ml_attack_probability", "mean"),
            mean_plausibility_score=("plausibility_score", "mean"),
            mean_rule_violation_count=("rule_violation_count", "mean"),
            mean_trust_score=("trust_score", "mean"),
            accept_count=("trustv2x_decision", lambda x: (x == "ACCEPT").sum()),
            flag_count=("trustv2x_decision", lambda x: (x == "FLAG_FOR_VERIFICATION").sum()),
            reject_alert_count=("trustv2x_decision", lambda x: (x == "REJECT_ALERT").sum()),
        )
        .reset_index()
    )

    by_attack["non_accept_rate"] = (
        (by_attack["flag_count"] + by_attack["reject_alert_count"]) / by_attack["samples"]
    ).round(4)

    by_attack["accept_rate"] = (by_attack["accept_count"] / by_attack["samples"]).round(4)

    numeric_round_cols = [
        "mean_ml_attack_probability",
        "mean_plausibility_score",
        "mean_rule_violation_count",
        "mean_trust_score",
    ]

    for col in numeric_round_cols:
        by_attack[col] = by_attack[col].round(4)

    by_attack.to_csv(DECISION_BY_ATTACK_FILE, index=False)

    # Severity summary
    if "attack_severity" in scoring_df.columns:
        severity_df = scoring_df[scoring_df["label"] == 1].copy()

        by_severity = (
            severity_df.groupby("attack_severity")
            .agg(
                samples=("attack_severity", "size"),
                mean_ml_attack_probability=("ml_attack_probability", "mean"),
                mean_plausibility_score=("plausibility_score", "mean"),
                mean_rule_violation_count=("rule_violation_count", "mean"),
                mean_trust_score=("trust_score", "mean"),
                accept_count=("trustv2x_decision", lambda x: (x == "ACCEPT").sum()),
                flag_count=("trustv2x_decision", lambda x: (x == "FLAG_FOR_VERIFICATION").sum()),
                reject_alert_count=("trustv2x_decision", lambda x: (x == "REJECT_ALERT").sum()),
            )
            .reset_index()
        )

        by_severity["non_accept_rate"] = (
            (by_severity["flag_count"] + by_severity["reject_alert_count"])
            / by_severity["samples"]
        ).round(4)

        by_severity["accept_rate"] = (
            by_severity["accept_count"] / by_severity["samples"]
        ).round(4)

        for col in numeric_round_cols:
            by_severity[col] = by_severity[col].round(4)

        by_severity.to_csv(DECISION_BY_SEVERITY_FILE, index=False)

    # Score profile
    score_profile = pd.DataFrame(
        [
            {
                **metrics,
                "mean_trust_score": round(scoring_df["trust_score"].mean(), 4),
                "normal_mean_trust_score": round(
                    scoring_df[scoring_df["label"] == 0]["trust_score"].mean(), 4
                ),
                "attack_mean_trust_score": round(
                    scoring_df[scoring_df["label"] == 1]["trust_score"].mean(), 4
                ),
                "normal_mean_ml_attack_probability": round(
                    scoring_df[scoring_df["label"] == 0]["ml_attack_probability"].mean(), 4
                ),
                "attack_mean_ml_attack_probability": round(
                    scoring_df[scoring_df["label"] == 1]["ml_attack_probability"].mean(), 4
                ),
                "accept_threshold": ACCEPT_THRESHOLD,
                "reject_threshold": REJECT_THRESHOLD,
            }
        ]
    )

    score_profile.to_csv(TRUST_SCORE_PROFILE_FILE, index=False)

    return decision_counts, by_attack


def plot_trust_score_distribution(scoring_df):
    fig, ax = plt.subplots(figsize=(7, 5))

    normal_scores = scoring_df[scoring_df["label"] == 0]["trust_score"]
    attack_scores = scoring_df[scoring_df["label"] == 1]["trust_score"]

    ax.hist(normal_scores, bins=30, alpha=0.6, label="Normal")
    ax.hist(attack_scores, bins=30, alpha=0.6, label="Attack")

    ax.axvline(ACCEPT_THRESHOLD, linestyle="--", label="Accept threshold")
    ax.axvline(REJECT_THRESHOLD, linestyle="--", label="Reject threshold")

    ax.set_title("TrustV2X Trust Score Distribution")
    ax.set_xlabel("Trust Score")
    ax.set_ylabel("Message Count")
    ax.legend()

    plt.tight_layout()

    figure_path = RESULTS_FIGURE_DIR / "trustv2x_trust_score_distribution.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"[OK] Trust score distribution saved to: {figure_path}")


def plot_decision_counts(scoring_df):
    decision_counts = scoring_df["trustv2x_decision"].value_counts().reset_index()
    decision_counts.columns = ["decision", "count"]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(decision_counts["decision"], decision_counts["count"])
    ax.set_title("TrustV2X Decision Counts")
    ax.set_xlabel("Decision")
    ax.set_ylabel("Message Count")
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()

    figure_path = RESULTS_FIGURE_DIR / "trustv2x_decision_counts.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"[OK] Decision count figure saved to: {figure_path}")


def plot_attack_type_trust_scores(scoring_df):
    plot_df = (
        scoring_df.groupby("attack_type")["trust_score"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(plot_df["attack_type"], plot_df["trust_score"])
    ax.set_title("Mean Trust Score by Attack Type")
    ax.set_xlabel("Mean Trust Score")
    ax.set_ylabel("Attack Type")
    ax.set_xlim(0, 1.05)

    plt.tight_layout()

    figure_path = RESULTS_FIGURE_DIR / "trustv2x_mean_trust_score_by_attack_type.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"[OK] Mean trust score by attack type saved to: {figure_path}")


def plot_decision_confusion_matrix(scoring_df):
    y_true = scoring_df["label"].astype(int)
    y_pred = scoring_df["decision_binary_attack"].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)

    ax.set_title("TrustV2X Decision Confusion Matrix")
    ax.set_xlabel("Decision Output")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Accept", "Flag/Reject"])
    ax.set_yticklabels(["Normal", "Attack"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    figure_path = RESULTS_FIGURE_DIR / "trustv2x_decision_confusion_matrix.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"[OK] Decision confusion matrix saved to: {figure_path}")


def main():
    ensure_directories()

    df = load_dataset()

    model, selected_features, numeric_features, categorical_features = build_scoring_model(df)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    print("\n[INFO] TrustV2X scoring engine configuration")
    print("[INFO] ML model: XGBoost")
    print("[INFO] ML input: TrustV2X continuous features only; rule flags excluded")
    print(f"[INFO] Numeric features: {numeric_features}")
    print(f"[INFO] Categorical features: {categorical_features}")
    print(f"[INFO] Train size: {len(train_df)}")
    print(f"[INFO] Held-out scoring/test size: {len(test_df)}")

    X_train = train_df[selected_features]
    y_train = train_df["label"].astype(int)

    X_test = test_df[selected_features]

    print("\n[TRAINING] TrustV2X XGBoost no-rule-flags scoring engine")
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    print(f"[OK] Scoring engine saved to: {MODEL_FILE}")

    print("\n[SCORING] Held-out test messages")
    ml_attack_probability = model.predict_proba(X_test)[:, 1]

    scoring_df = calculate_trust_score(test_df, ml_attack_probability)

    metrics = compute_decision_metrics(scoring_df)

    scoring_df.to_csv(SCORING_RESULTS_FILE, index=False)

    decision_counts, by_attack = save_decision_summaries(scoring_df, metrics)

    plot_trust_score_distribution(scoring_df)
    plot_decision_counts(scoring_df)
    plot_attack_type_trust_scores(scoring_df)
    plot_decision_confusion_matrix(scoring_df)

    print("\n" + "=" * 70)
    print("[OK] TrustV2X scoring complete")
    print("=" * 70)

    print(f"[OK] Full scoring results saved to: {SCORING_RESULTS_FILE}")
    print(f"[OK] Decision summary saved to: {DECISION_SUMMARY_FILE}")
    print(f"[OK] Decision by attack type saved to: {DECISION_BY_ATTACK_FILE}")
    print(f"[OK] Decision by severity saved to: {DECISION_BY_SEVERITY_FILE}")
    print(f"[OK] Trust score profile saved to: {TRUST_SCORE_PROFILE_FILE}")

    print("\nTrustV2X decision metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nDecision by attack type:")
    print(by_attack.to_string(index=False))

    print("\nDecision counts:")
    print(decision_counts.to_string(index=False))

    print("\nInterpretation:")
    print("- ACCEPT means the CAM-like message is trusted and can continue to C-ITS processing.")
    print("- FLAG_FOR_VERIFICATION means the message is suspicious and should be checked or down-weighted.")
    print("- REJECT_ALERT means the message is highly suspicious and should be rejected or escalated.")
    print("- For evaluation, FLAG_FOR_VERIFICATION and REJECT_ALERT are both treated as attack detections.")

    print("\nNext step after this:")
    print("Interpret TrustV2X scoring output, then move to explainability using SHAP/LIME-style feature interpretation.")


if __name__ == "__main__":
    main()