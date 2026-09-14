import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score


# ============================================================
# TrustV2X - Step 08
# Explainability and Decision Interpretation
# ============================================================

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCORING_RESULTS_FILE = PROJECT_ROOT / "results" / "tables" / "trustv2x_scoring_results.csv"
MODEL_FILE = PROJECT_ROOT / "results" / "models" / "trustv2x_xgboost_no_rule_flags_scoring_engine.joblib"

RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

GLOBAL_IMPORTANCE_FILE = RESULTS_TABLE_DIR / "explainability_xgboost_global_importance.csv"
PERMUTATION_IMPORTANCE_FILE = RESULTS_TABLE_DIR / "explainability_permutation_importance.csv"
ATTACK_PROFILE_FILE = RESULTS_TABLE_DIR / "explainability_attack_type_profile.csv"
DECISION_PROFILE_FILE = RESULTS_TABLE_DIR / "explainability_decision_profile.csv"
NORMAL_ATTACK_CONTRAST_FILE = RESULTS_TABLE_DIR / "explainability_normal_attack_feature_contrast.csv"
LOCAL_EXPLANATION_FILE = RESULTS_TABLE_DIR / "explainability_local_decision_examples.csv"
SHAP_GLOBAL_FILE = RESULTS_TABLE_DIR / "explainability_shap_global_top30.csv"
SHAP_LOCAL_FILE = RESULTS_TABLE_DIR / "explainability_shap_local_examples.csv"

RANDOM_SEED = 42
PERMUTATION_SAMPLE_SIZE = 3000
SHAP_SAMPLE_SIZE = 1500
LOCAL_EXAMPLE_COUNT = 5


def ensure_directories():
    RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs():
    if not SCORING_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Scoring results not found: {SCORING_RESULTS_FILE}\n"
            "Run scripts/07_trustv2x_scoring.py first."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Scoring model not found: {MODEL_FILE}\n"
            "Run scripts/07_trustv2x_scoring.py first."
        )

    df = pd.read_csv(SCORING_RESULTS_FILE)
    model = joblib.load(MODEL_FILE)

    print("=" * 70)
    print("TrustV2X - Explainability and Decision Interpretation")
    print("=" * 70)

    print(f"[INFO] Scoring results loaded: {SCORING_RESULTS_FILE}")
    print(f"[INFO] Scoring results shape: {df.shape}")
    print(f"[INFO] Model loaded: {MODEL_FILE}")

    return df, model


def get_model_features(df):
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

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    return numeric_features, categorical_features, numeric_features + categorical_features


def plot_barh(df, label_col, value_col, title, xlabel, filename, top_n=20):
    plot_df = df.head(top_n).copy()
    plot_df = plot_df.sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(plot_df[label_col], plot_df[value_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Feature")

    plt.tight_layout()

    figure_path = RESULTS_FIGURE_DIR / filename
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"[OK] Figure saved to: {figure_path}")


def xgboost_global_importance(model):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df.to_csv(GLOBAL_IMPORTANCE_FILE, index=False)

    print(f"\n[OK] XGBoost global importance saved to: {GLOBAL_IMPORTANCE_FILE}")
    print(importance_df.head(30).to_string(index=False))

    plot_barh(
        importance_df,
        "feature",
        "importance",
        "TrustV2X XGBoost Global Feature Importance",
        "Importance",
        "explainability_xgboost_global_importance_top20.png",
        top_n=20,
    )

    return importance_df


def permutation_importance_analysis(df, model, selected_features):
    sample_size = min(PERMUTATION_SAMPLE_SIZE, len(df))

    sample_df = df.sample(n=sample_size, random_state=RANDOM_SEED).copy()

    X_sample = sample_df[selected_features]
    y_sample = sample_df["label"].astype(int)

    print(f"\n[RUNNING] Permutation importance on {sample_size} samples")

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=RANDOM_SEED,
        scoring="f1",
        n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "feature": selected_features,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    perm_df = perm_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    perm_df.to_csv(PERMUTATION_IMPORTANCE_FILE, index=False)

    print(f"[OK] Permutation importance saved to: {PERMUTATION_IMPORTANCE_FILE}")
    print(perm_df.head(30).to_string(index=False))

    plot_barh(
        perm_df,
        "feature",
        "importance_mean",
        "TrustV2X Permutation Importance",
        "Mean F1-score Drop",
        "explainability_permutation_importance_top20.png",
        top_n=20,
    )

    return perm_df


def attack_type_profile(df):
    profile_features = [
        "ml_attack_probability",
        "plausibility_score",
        "plausibility_risk",
        "rule_violation_count",
        "combined_risk_score",
        "trust_score",
        "speed",
        "acceleration",
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

    profile_features = [c for c in profile_features if c in df.columns]

    profile_df = (
        df.groupby("attack_type")[profile_features]
        .mean()
        .round(4)
        .reset_index()
    )

    if "label" in df.columns:
        label_df = df.groupby("attack_type")["label"].first().reset_index()
        profile_df = profile_df.merge(label_df, on="attack_type", how="left")

    profile_df["samples"] = df.groupby("attack_type").size().values

    profile_df = profile_df.sort_values("trust_score", ascending=True)
    profile_df.to_csv(ATTACK_PROFILE_FILE, index=False)

    print(f"\n[OK] Attack-type explanation profile saved to: {ATTACK_PROFILE_FILE}")
    print(profile_df.to_string(index=False))

    return profile_df


def decision_profile(df):
    profile_features = [
        "ml_attack_probability",
        "plausibility_score",
        "plausibility_risk",
        "rule_violation_count",
        "combined_risk_score",
        "trust_score",
        "ml_confidence",
    ]

    profile_features = [c for c in profile_features if c in df.columns]

    decision_df = (
        df.groupby(["label", "trustv2x_decision"])[profile_features]
        .mean()
        .round(4)
        .reset_index()
    )

    decision_df["samples"] = (
        df.groupby(["label", "trustv2x_decision"])
        .size()
        .values
    )

    decision_df.to_csv(DECISION_PROFILE_FILE, index=False)

    print(f"\n[OK] Decision explanation profile saved to: {DECISION_PROFILE_FILE}")
    print(decision_df.to_string(index=False))

    return decision_df


def normal_attack_feature_contrast(df):
    numeric_features, _, _ = get_model_features(df)

    normal_df = df[df["label"] == 0]
    attack_df = df[df["label"] == 1]

    rows = []

    for feature in numeric_features:
        normal_mean = normal_df[feature].mean()
        attack_mean = attack_df[feature].mean()
        normal_std = normal_df[feature].std()

        if normal_std == 0 or pd.isna(normal_std):
            standardised_difference = 0
        else:
            standardised_difference = abs(attack_mean - normal_mean) / normal_std

        rows.append({
            "feature": feature,
            "normal_mean": round(normal_mean, 4),
            "attack_mean": round(attack_mean, 4),
            "absolute_difference": round(abs(attack_mean - normal_mean), 4),
            "standardised_difference": round(standardised_difference, 4),
        })

    contrast_df = pd.DataFrame(rows)
    contrast_df = contrast_df.sort_values("standardised_difference", ascending=False).reset_index(drop=True)

    contrast_df.to_csv(NORMAL_ATTACK_CONTRAST_FILE, index=False)

    print(f"\n[OK] Normal-vs-attack feature contrast saved to: {NORMAL_ATTACK_CONTRAST_FILE}")
    print(contrast_df.head(30).to_string(index=False))

    plot_barh(
        contrast_df,
        "feature",
        "standardised_difference",
        "Normal vs Attack Feature Contrast",
        "Standardised Difference",
        "explainability_normal_attack_feature_contrast_top20.png",
        top_n=20,
    )

    return contrast_df


def local_deviation_explanations(df):
    """
    Fallback local explanation:
    Shows which numeric features deviate most from normal behaviour for selected examples.
    This is not SHAP, but gives a readable local decision interpretation.
    """

    numeric_features, _, _ = get_model_features(df)

    normal_reference = df[df["label"] == 0][numeric_features].copy()
    normal_mean = normal_reference.mean()
    normal_std = normal_reference.std().replace(0, np.nan)

    candidate_groups = {
        "normal_accepted": df[(df["label"] == 0) & (df["trustv2x_decision"] == "ACCEPT")],
        "normal_flagged": df[(df["label"] == 0) & (df["trustv2x_decision"] != "ACCEPT")],
        "attack_rejected": df[(df["label"] == 1) & (df["trustv2x_decision"] == "REJECT_ALERT")],
        "attack_flagged": df[(df["label"] == 1) & (df["trustv2x_decision"] == "FLAG_FOR_VERIFICATION")],
    }

    rows = []

    for group_name, group_df in candidate_groups.items():
        if group_df.empty:
            continue

        sample_df = group_df.sample(
            n=min(LOCAL_EXAMPLE_COUNT, len(group_df)),
            random_state=RANDOM_SEED,
        )

        for idx, row in sample_df.iterrows():
            z_scores = ((row[numeric_features] - normal_mean) / normal_std).abs()
            z_scores = z_scores.replace([np.inf, -np.inf], np.nan).fillna(0)
            top_features = z_scores.sort_values(ascending=False).head(8)

            rows.append({
                "group": group_name,
                "row_index": idx,
                "label": row.get("label"),
                "attack_type": row.get("attack_type"),
                "decision": row.get("trustv2x_decision"),
                "trust_score": round(row.get("trust_score", np.nan), 4),
                "ml_attack_probability": round(row.get("ml_attack_probability", np.nan), 4),
                "top_deviation_features": "; ".join(
                    [f"{feature}={value:.3f}" for feature, value in top_features.items()]
                ),
            })

    local_df = pd.DataFrame(rows)
    local_df.to_csv(LOCAL_EXPLANATION_FILE, index=False)

    print(f"\n[OK] Local deviation explanations saved to: {LOCAL_EXPLANATION_FILE}")
    print(local_df.to_string(index=False))

    return local_df


def try_shap_analysis(df, model, selected_features):
    try:
        import shap
    except Exception as e:
        print("\n[WARNING] SHAP is not installed or could not be imported.")
        print(f"[INFO] Reason: {e}")
        print("[INFO] The fallback local deviation explanations have still been generated.")
        return None

    print("\n[RUNNING] SHAP explainability")

    sample_size = min(SHAP_SAMPLE_SIZE, len(df))
    sample_df = df.sample(n=sample_size, random_state=RANDOM_SEED).copy()

    X_sample = sample_df[selected_features]

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.array(shap_values)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_global_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    })

    shap_global_df = shap_global_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_global_df.head(30).to_csv(SHAP_GLOBAL_FILE, index=False)

    print(f"[OK] SHAP global top 30 saved to: {SHAP_GLOBAL_FILE}")
    print(shap_global_df.head(30).to_string(index=False))

    plot_barh(
        shap_global_df,
        "feature",
        "mean_abs_shap",
        "TrustV2X SHAP Global Importance",
        "Mean Absolute SHAP Value",
        "explainability_shap_global_top20.png",
        top_n=20,
    )

    # Local SHAP examples
    local_rows = []
    local_candidates = sample_df[
        (
            ((sample_df["label"] == 0) & (sample_df["trustv2x_decision"] == "ACCEPT"))
            | ((sample_df["label"] == 1) & (sample_df["trustv2x_decision"] != "ACCEPT"))
        )
    ].head(10)

    if not local_candidates.empty:
        X_local = preprocessor.transform(local_candidates[selected_features])
        local_shap = explainer.shap_values(X_local)

        if isinstance(local_shap, list):
            local_shap = local_shap[1]

        local_shap = np.array(local_shap)

        for local_position, (_, row) in enumerate(local_candidates.iterrows()):
            values = local_shap[local_position]
            top_idx = np.argsort(np.abs(values))[::-1][:8]

            local_rows.append({
                "row_index": row.name,
                "label": row.get("label"),
                "attack_type": row.get("attack_type"),
                "decision": row.get("trustv2x_decision"),
                "trust_score": round(row.get("trust_score", np.nan), 4),
                "ml_attack_probability": round(row.get("ml_attack_probability", np.nan), 4),
                "top_shap_features": "; ".join(
                    [
                        f"{feature_names[i]}={values[i]:.4f}"
                        for i in top_idx
                    ]
                ),
            })

    shap_local_df = pd.DataFrame(local_rows)
    shap_local_df.to_csv(SHAP_LOCAL_FILE, index=False)

    print(f"[OK] SHAP local examples saved to: {SHAP_LOCAL_FILE}")

    return shap_global_df


def main():
    ensure_directories()

    df, model = load_inputs()

    numeric_features, categorical_features, selected_features = get_model_features(df)

    print("\n[INFO] Explainability feature configuration")
    print(f"[INFO] Numeric features: {numeric_features}")
    print(f"[INFO] Categorical features: {categorical_features}")
    print(f"[INFO] Total model input columns before encoding: {len(selected_features)}")

    xgboost_global_importance(model)
    permutation_importance_analysis(df, model, selected_features)
    attack_type_profile(df)
    decision_profile(df)
    normal_attack_feature_contrast(df)
    local_deviation_explanations(df)
    try_shap_analysis(df, model, selected_features)

    print("\n" + "=" * 70)
    print("[OK] TrustV2X explainability complete")
    print("=" * 70)

    print("\nGenerated explainability outputs:")
    print(f"- {GLOBAL_IMPORTANCE_FILE}")
    print(f"- {PERMUTATION_IMPORTANCE_FILE}")
    print(f"- {ATTACK_PROFILE_FILE}")
    print(f"- {DECISION_PROFILE_FILE}")
    print(f"- {NORMAL_ATTACK_CONTRAST_FILE}")
    print(f"- {LOCAL_EXPLANATION_FILE}")
    print(f"- {SHAP_GLOBAL_FILE}  [only if SHAP works]")
    print(f"- {SHAP_LOCAL_FILE}   [only if SHAP works]")

    print("\nFigures generated:")
    print(f"- {RESULTS_FIGURE_DIR / 'explainability_xgboost_global_importance_top20.png'}")
    print(f"- {RESULTS_FIGURE_DIR / 'explainability_permutation_importance_top20.png'}")
    print(f"- {RESULTS_FIGURE_DIR / 'explainability_normal_attack_feature_contrast_top20.png'}")
    print(f"- {RESULTS_FIGURE_DIR / 'explainability_shap_global_top20.png'}  [only if SHAP works]")

    print("\nNext step:")
    print("Interpret the explainability outputs and then prepare the final Results and Discussion tables for the paper.")


if __name__ == "__main__":
    main()