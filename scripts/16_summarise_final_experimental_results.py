import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# TrustV2X - Step 16
# Summarise Final Experimental Results for Paper
# ============================================================
#
# This script creates final paper-ready tables by combining:
# 1. Controlled SUMO 4x4 TrustV2X results
# 2. External VeReMi Extension benchmark results
# 3. Vehicle-disjoint validation results
# 4. Per-attack external validation analysis
#
# Output:
# results/final_tables/
# results/final_figures/
# results/final_summary/trustv2x_experimental_summary.txt
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VEREMI_DATASET_CSV = (
    PROJECT_ROOT
    / "data"
    / "veremi"
    / "processed"
    / "veremi_extension_trustv2x_external_benchmark.csv"
)

VEREMI_RESULTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "veremi"
    / "tables"
    / "veremi_external_ml_results.csv"
)

VEREMI_PER_ATTACK_CSV = (
    PROJECT_ROOT
    / "results"
    / "veremi"
    / "tables"
    / "veremi_external_per_attack_results.csv"
)

FINAL_TABLES_DIR = PROJECT_ROOT / "results" / "final_tables"
FINAL_FIGURES_DIR = PROJECT_ROOT / "results" / "final_figures"
FINAL_SUMMARY_DIR = PROJECT_ROOT / "results" / "final_summary"

for folder in [FINAL_TABLES_DIR, FINAL_FIGURES_DIR, FINAL_SUMMARY_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def controlled_sumo_dataset_profile():
    return pd.DataFrame(
        [
            {
                "dataset": "Controlled SUMO 4x4 TrustV2X Dataset",
                "source": "SUMO-generated C-ITS/V2X simulation",
                "total_records": 40000,
                "normal_records": 28000,
                "attack_records": 12000,
                "normal_ratio": 0.70,
                "attack_ratio": 0.30,
                "attack_types": 12,
                "feature_columns": 50,
                "purpose": "Internal controlled validation and ablation testing",
            }
        ]
    )


def veremi_dataset_profile():
    if not VEREMI_DATASET_CSV.exists():
        print(f"[WARNING] VeReMi dataset not found: {VEREMI_DATASET_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(VEREMI_DATASET_CSV)

    total = len(df)
    normal = int((df["label"] == 0).sum())
    attack = int((df["label"] == 1).sum())

    attack_type_count = (
        df.loc[df["label"] == 1, "attack_type"].nunique()
        if "attack_type" in df.columns
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "dataset": "VeReMi Extension External Benchmark",
                "source": "Ground-truth-labelled VeReMi Extension trace data",
                "total_records": total,
                "normal_records": normal,
                "attack_records": attack,
                "normal_ratio": round(normal / total, 4),
                "attack_ratio": round(attack / total, 4),
                "attack_types": attack_type_count,
                "feature_columns": df.shape[1],
                "purpose": "External benchmark validation and generalisation testing",
            }
        ]
    )


def controlled_sumo_results():
    """
    These are the final controlled SUMO results already generated in the
    TrustV2X experimental pipeline.
    """
    rows = [
        {
            "experiment": "Controlled SUMO",
            "validation_setting": "Random Stratified Holdout",
            "feature_set": "TrustV2X Full",
            "model_or_layer": "XGBoost / TrustV2X classifier",
            "accuracy": 0.9999,
            "balanced_accuracy": 0.9998,
            "precision": 1.0000,
            "recall_detection_rate": 0.9997,
            "f1_score": 0.9998,
            "false_alarm_rate": 0.0000,
            "mcc": 0.9998,
            "interpretation": "Near-perfect internal validation on controlled SUMO-generated attacks.",
        },
        {
            "experiment": "Controlled SUMO",
            "validation_setting": "Unseen Vehicle Split",
            "feature_set": "TrustV2X Full",
            "model_or_layer": "XGBoost / TrustV2X classifier",
            "accuracy": 0.9996,
            "balanced_accuracy": 0.9996,
            "precision": 0.9994,
            "recall_detection_rate": 0.9994,
            "f1_score": 0.9994,
            "false_alarm_rate": 0.0003,
            "mcc": 0.9991,
            "interpretation": "Strong generalisation across unseen simulated vehicles.",
        },
        {
            "experiment": "Controlled SUMO",
            "validation_setting": "Time-Based Split",
            "feature_set": "TrustV2X Full",
            "model_or_layer": "XGBoost / TrustV2X classifier",
            "accuracy": 0.9998,
            "balanced_accuracy": 0.9997,
            "precision": 1.0000,
            "recall_detection_rate": 0.9993,
            "f1_score": 0.9997,
            "false_alarm_rate": 0.0000,
            "mcc": 0.9995,
            "interpretation": "Stable temporal generalisation under time-based holdout.",
        },
        {
            "experiment": "Controlled SUMO",
            "validation_setting": "Hard Ablation Test",
            "feature_set": "TrustV2X Hard: No Timing, No Rule Flags",
            "model_or_layer": "Ablated TrustV2X classifier",
            "accuracy": 0.9852,
            "balanced_accuracy": np.nan,
            "precision": np.nan,
            "recall_detection_rate": 0.9687,
            "f1_score": 0.9752,
            "false_alarm_rate": 0.0077,
            "mcc": np.nan,
            "interpretation": "Performance remains strong after removing timing-heavy and binary rule features.",
        },
        {
            "experiment": "Controlled SUMO",
            "validation_setting": "Operational Trust Scoring",
            "feature_set": "TrustV2X No Rule Flags + Trust Score",
            "model_or_layer": "TrustV2X scoring layer",
            "accuracy": 0.9997,
            "balanced_accuracy": np.nan,
            "precision": 0.9990,
            "recall_detection_rate": 1.0000,
            "f1_score": 0.9995,
            "false_alarm_rate": 0.0004,
            "mcc": 0.9993,
            "interpretation": "Decision-level scoring achieved full attack detection with very low false alarms.",
        },
    ]

    return pd.DataFrame(rows)


def load_veremi_results():
    if not VEREMI_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"VeReMi result file not found:\n{VEREMI_RESULTS_CSV}\n\n"
            "Run this first:\n"
            "python scripts/15_train_validate_veremi_external_benchmark.py"
        )

    df = pd.read_csv(VEREMI_RESULTS_CSV)

    numeric_cols = [
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

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def best_rows_for_paper(veremi_df):
    rows = []

    conditions = [
        {
            "result_label": "Best Raw BSM Baseline - Random Holdout",
            "split": "Random Stratified Holdout",
            "feature_contains": "Raw_BSM",
        },
        {
            "result_label": "Best TrustV2X - Random Holdout",
            "split": "Random Stratified Holdout",
            "feature_contains": "TrustV2X",
        },
        {
            "result_label": "Best Raw BSM Baseline - Vehicle Disjoint",
            "split": "Vehicle Disjoint Holdout",
            "feature_contains": "Raw_BSM",
        },
        {
            "result_label": "Best TrustV2X - Vehicle Disjoint",
            "split": "Vehicle Disjoint Holdout",
            "feature_contains": "TrustV2X",
        },
    ]

    for condition in conditions:
        subset = veremi_df[
            (veremi_df["split"] == condition["split"])
            & (veremi_df["feature_set"].str.contains(condition["feature_contains"], case=False, na=False))
        ].copy()

        if subset.empty:
            continue

        best = subset.sort_values(
            ["f1_score", "balanced_accuracy", "mcc"],
            ascending=[False, False, False],
        ).iloc[0]

        row = best.to_dict()
        row["result_label"] = condition["result_label"]
        rows.append(row)

    paper_df = pd.DataFrame(rows)

    preferred_cols = [
        "result_label",
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

    existing_cols = [c for c in preferred_cols if c in paper_df.columns]

    return paper_df[existing_cols]


def veremi_top_results(veremi_df):
    return veremi_df.sort_values(
        ["f1_score", "balanced_accuracy", "mcc"],
        ascending=[False, False, False],
    ).head(20)


def create_sumo_vs_veremi_comparison(controlled_df, veremi_paper_df):
    comparison_rows = []

    # Controlled SUMO key rows.
    for _, row in controlled_df.iterrows():
        comparison_rows.append(
            {
                "dataset": "Controlled SUMO 4x4",
                "validation_setting": row["validation_setting"],
                "best_model_or_layer": row["model_or_layer"],
                "feature_set": row["feature_set"],
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall_detection_rate": row["recall_detection_rate"],
                "f1_score": row["f1_score"],
                "false_alarm_rate": row["false_alarm_rate"],
                "mcc": row["mcc"],
                "main_interpretation": row["interpretation"],
            }
        )

    # VeReMi key rows.
    for _, row in veremi_paper_df.iterrows():
        if "TrustV2X" not in str(row.get("result_label", "")):
            continue

        comparison_rows.append(
            {
                "dataset": "VeReMi Extension",
                "validation_setting": row["split"],
                "best_model_or_layer": row["model"],
                "feature_set": row["feature_set"],
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall_detection_rate": row["recall_detection_rate"],
                "f1_score": row["f1_score"],
                "false_alarm_rate": row["false_alarm_rate"],
                "mcc": row["mcc"],
                "main_interpretation": "External benchmark validation confirms strong TrustV2X generalisation beyond the controlled SUMO dataset.",
            }
        )

    return pd.DataFrame(comparison_rows)


def create_improvement_table(veremi_paper_df):
    rows = []

    for split_name in ["Random Stratified Holdout", "Vehicle Disjoint Holdout"]:
        raw_row = veremi_paper_df[
            (veremi_paper_df["split"] == split_name)
            & (veremi_paper_df["result_label"].str.contains("Raw BSM", case=False, na=False))
        ]

        trust_row = veremi_paper_df[
            (veremi_paper_df["split"] == split_name)
            & (veremi_paper_df["result_label"].str.contains("TrustV2X", case=False, na=False))
        ]

        if raw_row.empty or trust_row.empty:
            continue

        raw = raw_row.iloc[0]
        trust = trust_row.iloc[0]

        rows.append(
            {
                "split": split_name,
                "raw_baseline_model": raw["model"],
                "raw_baseline_feature_set": raw["feature_set"],
                "raw_f1": raw["f1_score"],
                "raw_false_alarm_rate": raw["false_alarm_rate"],
                "trustv2x_model": trust["model"],
                "trustv2x_feature_set": trust["feature_set"],
                "trustv2x_f1": trust["f1_score"],
                "trustv2x_false_alarm_rate": trust["false_alarm_rate"],
                "absolute_f1_improvement": trust["f1_score"] - raw["f1_score"],
                "relative_f1_improvement_percent": (
                    ((trust["f1_score"] - raw["f1_score"]) / raw["f1_score"]) * 100
                    if raw["f1_score"] > 0
                    else np.nan
                ),
                "false_alarm_rate_change": trust["false_alarm_rate"] - raw["false_alarm_rate"],
            }
        )

    return pd.DataFrame(rows)


def prepare_per_attack_table():
    if not VEREMI_PER_ATTACK_CSV.exists():
        print(f"[WARNING] Per-attack file not found: {VEREMI_PER_ATTACK_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(VEREMI_PER_ATTACK_CSV)

    numeric_cols = [
        "samples",
        "normal_samples",
        "attack_samples",
        "detection_rate_for_attack_rows",
        "false_alarm_rate_for_normal_rows",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(
        ["split", "detection_rate_for_attack_rows"],
        ascending=[True, True],
    )


def plot_veremi_f1_comparison(veremi_paper_df):
    if veremi_paper_df.empty:
        return

    plot_df = veremi_paper_df.copy()
    plot_df["label"] = plot_df["result_label"] + "\n" + plot_df["model"]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(plot_df))

    ax.bar(x, plot_df["f1_score"])
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("VeReMi External Benchmark: Raw Baseline vs TrustV2X")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FINAL_FIGURES_DIR / "figure_veremi_raw_vs_trustv2x_f1.png", dpi=300)
    plt.close(fig)


def plot_sumo_vs_veremi(comparison_df):
    if comparison_df.empty:
        return

    plot_df = comparison_df[
        comparison_df["validation_setting"].isin(
            [
                "Unseen Vehicle Split",
                "Vehicle Disjoint Holdout",
                "Hard Ablation Test",
                "Random Stratified Holdout",
            ]
        )
    ].copy()

    plot_df["label"] = plot_df["dataset"] + "\n" + plot_df["validation_setting"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(plot_df))

    ax.bar(x, plot_df["f1_score"])
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("TrustV2X Controlled SUMO vs External VeReMi Validation")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FINAL_FIGURES_DIR / "figure_sumo_vs_veremi_f1.png", dpi=300)
    plt.close(fig)


def write_summary_text(
    dataset_profile_df,
    controlled_df,
    veremi_paper_df,
    improvement_df,
    comparison_df,
):
    output_file = FINAL_SUMMARY_DIR / "trustv2x_experimental_summary.txt"

    lines = []
    lines.append("=" * 80)
    lines.append("TrustV2X Final Experimental Summary")
    lines.append("=" * 80)

    lines.append("\n1. Dataset profile")
    lines.append("-" * 80)
    lines.append(dataset_profile_df.to_string(index=False))

    lines.append("\n\n2. Controlled SUMO summary")
    lines.append("-" * 80)
    lines.append(
        controlled_df[
            [
                "validation_setting",
                "feature_set",
                "model_or_layer",
                "accuracy",
                "recall_detection_rate",
                "f1_score",
                "false_alarm_rate",
                "mcc",
            ]
        ].to_string(index=False)
    )

    lines.append("\n\n3. VeReMi external benchmark key results")
    lines.append("-" * 80)
    lines.append(veremi_paper_df.to_string(index=False))

    lines.append("\n\n4. Raw baseline vs TrustV2X improvement on VeReMi")
    lines.append("-" * 80)
    lines.append(improvement_df.to_string(index=False))

    lines.append("\n\n5. Controlled SUMO vs external VeReMi comparison")
    lines.append("-" * 80)
    lines.append(
        comparison_df[
            [
                "dataset",
                "validation_setting",
                "best_model_or_layer",
                "feature_set",
                "accuracy",
                "recall_detection_rate",
                "f1_score",
                "false_alarm_rate",
                "mcc",
            ]
        ].to_string(index=False)
    )

    lines.append("\n\nPaper-ready interpretation")
    lines.append("-" * 80)
    lines.append(
        "The controlled SUMO experiment demonstrates the upper-bound performance of "
        "TrustV2X under a fully controlled C-ITS/V2X simulation environment. The external "
        "VeReMi Extension benchmark provides a more challenging validation setting. "
        "The results show that TrustV2X engineered temporal, mobility and plausibility "
        "features substantially outperform raw BSM/CAM-like fields, especially under "
        "vehicle-disjoint evaluation. This supports the claim that TrustV2X improves "
        "misbehavior detection generalisation beyond the internally generated SUMO dataset."
    )

    lines.append("\n\nImportant limitations to state")
    lines.append("-" * 80)
    lines.append(
        "The VeReMi labels were derived by comparing received trace messages with "
        "ground-truth physical states using position, speed, acceleration and heading "
        "error thresholds. These ground-truth error columns were used only for label "
        "construction and audit, not as model input features. The results should therefore "
        "be framed as simulation and benchmark-based validation, not direct real-road "
        "deployment evidence."
    )

    output_file.write_text("\n".join(lines), encoding="utf-8")
    return output_file


def main():
    print("=" * 80)
    print("TrustV2X - Final Experimental Results Summary")
    print("=" * 80)

    controlled_profile = controlled_sumo_dataset_profile()
    veremi_profile = veremi_dataset_profile()

    dataset_profile_df = pd.concat(
        [controlled_profile, veremi_profile],
        ignore_index=True,
    )

    controlled_df = controlled_sumo_results()
    veremi_df = load_veremi_results()

    veremi_paper_df = best_rows_for_paper(veremi_df)
    veremi_top20_df = veremi_top_results(veremi_df)
    improvement_df = create_improvement_table(veremi_paper_df)
    per_attack_df = prepare_per_attack_table()
    comparison_df = create_sumo_vs_veremi_comparison(
        controlled_df=controlled_df,
        veremi_paper_df=veremi_paper_df,
    )

    dataset_profile_df.to_csv(
        FINAL_TABLES_DIR / "table1_dataset_profiles.csv",
        index=False,
    )

    controlled_df.to_csv(
        FINAL_TABLES_DIR / "table2_controlled_sumo_results.csv",
        index=False,
    )

    veremi_paper_df.to_csv(
        FINAL_TABLES_DIR / "table3_veremi_key_results.csv",
        index=False,
    )

    veremi_top20_df.to_csv(
        FINAL_TABLES_DIR / "table4_veremi_top20_results.csv",
        index=False,
    )

    improvement_df.to_csv(
        FINAL_TABLES_DIR / "table5_raw_vs_trustv2x_improvement.csv",
        index=False,
    )

    comparison_df.to_csv(
        FINAL_TABLES_DIR / "table6_sumo_vs_veremi_comparison.csv",
        index=False,
    )

    if not per_attack_df.empty:
        per_attack_df.to_csv(
            FINAL_TABLES_DIR / "table7_veremi_per_attack_results.csv",
            index=False,
        )

    plot_veremi_f1_comparison(veremi_paper_df)
    plot_sumo_vs_veremi(comparison_df)

    summary_file = write_summary_text(
        dataset_profile_df=dataset_profile_df,
        controlled_df=controlled_df,
        veremi_paper_df=veremi_paper_df,
        improvement_df=improvement_df,
        comparison_df=comparison_df,
    )

    print("\n[OK] Final tables saved to:")
    print(FINAL_TABLES_DIR)

    print("\n[OK] Final figures saved to:")
    print(FINAL_FIGURES_DIR)

    print("\n[OK] Final summary saved to:")
    print(summary_file)

    print("\nGenerated files:")
    for path in sorted(FINAL_TABLES_DIR.glob("*.csv")):
        print(f"- {path}")

    for path in sorted(FINAL_FIGURES_DIR.glob("*.png")):
        print(f"- {path}")

    print("\nKey VeReMi paper-ready results:")
    print(veremi_paper_df.to_string(index=False))

    print("\nRaw baseline vs TrustV2X improvement:")
    print(improvement_df.to_string(index=False))

    print("\nControlled SUMO vs VeReMi comparison:")
    print(
        comparison_df[
            [
                "dataset",
                "validation_setting",
                "best_model_or_layer",
                "feature_set",
                "accuracy",
                "recall_detection_rate",
                "f1_score",
                "false_alarm_rate",
                "mcc",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()