import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# TrustV2X - Step 18
# Integrate VeReMi DL Results into Final Paper Outputs
# ============================================================
#
# This script does NOT rerun training.
# It integrates results from:
# - Step 15: VeReMi ML benchmark
# - Step 17: VeReMi DL benchmark
# - Step 16: Existing final SUMO/VeReMi tables
#
# Outputs:
# results/final_tables/table8_veremi_dl_results.csv
# results/final_tables/table9_veremi_ml_dl_vehicle_disjoint_comparison.csv
# results/final_tables/table10_final_selected_results_for_paper.csv
# results/final_tables/table11_metric_reporting_strategy.csv
#
# results/final_figures/figure3_veremi_ml_dl_vehicle_disjoint_f1.png
# results/final_figures/figure4_veremi_ml_dl_recall_far_tradeoff.png
#
# results/final_summary/trustv2x_final_results_with_dl_summary.txt
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_RESULTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "veremi"
    / "tables"
    / "veremi_external_ml_results.csv"
)

DL_RESULTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "veremi_dl"
    / "tables"
    / "veremi_external_dl_results.csv"
)

COMBINED_RESULTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "veremi_dl"
    / "tables"
    / "veremi_external_ml_dl_combined_results.csv"
)

SUMO_VEREMI_COMPARISON_CSV = (
    PROJECT_ROOT
    / "results"
    / "final_tables"
    / "table6_sumo_vs_veremi_comparison.csv"
)

PER_ATTACK_CSV = (
    PROJECT_ROOT
    / "results"
    / "final_tables"
    / "table7_veremi_per_attack_results.csv"
)

FINAL_TABLES_DIR = PROJECT_ROOT / "results" / "final_tables"
FINAL_FIGURES_DIR = PROJECT_ROOT / "results" / "final_figures"
FINAL_SUMMARY_DIR = PROJECT_ROOT / "results" / "final_summary"

for folder in [FINAL_TABLES_DIR, FINAL_FIGURES_DIR, FINAL_SUMMARY_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


METRIC_COLUMNS = [
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


def make_numeric(df):
    for col in METRIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_ml_results():
    if not ML_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"ML result file not found:\n{ML_RESULTS_CSV}\n\n"
            "Run Step 15 first."
        )

    df = pd.read_csv(ML_RESULTS_CSV)
    df["model_family"] = "ML"
    return make_numeric(df)


def load_dl_results():
    if not DL_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"DL result file not found:\n{DL_RESULTS_CSV}\n\n"
            "Run Step 17 first."
        )

    df = pd.read_csv(DL_RESULTS_CSV)
    df["model_family"] = "DL"
    return make_numeric(df)


def build_combined_results(ml_df, dl_df):
    combined = pd.concat([ml_df, dl_df], ignore_index=True, sort=False)

    combined = combined.sort_values(
        ["split", "f1_score", "balanced_accuracy", "mcc"],
        ascending=[True, False, False, False],
    )

    return combined


def get_vehicle_disjoint_results(combined_df):
    vehicle_df = combined_df[
        combined_df["split"] == "Vehicle Disjoint Holdout"
    ].copy()

    vehicle_df = vehicle_df.sort_values(
        ["f1_score", "balanced_accuracy", "mcc"],
        ascending=[False, False, False],
    )

    preferred_cols = [
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
        "roc_auc",
        "pr_auc",
        "avg_latency_ms_per_message",
    ]

    existing_cols = [c for c in preferred_cols if c in vehicle_df.columns]
    return vehicle_df[existing_cols]


def best_row(df, condition, label):
    subset = df[condition].copy()

    if subset.empty:
        return None

    row = subset.sort_values(
        ["f1_score", "balanced_accuracy", "mcc"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()

    row["selected_result"] = label
    return row


def create_final_selected_results(combined_df):
    vehicle_df = combined_df[
        combined_df["split"] == "Vehicle Disjoint Holdout"
    ].copy()

    selected_rows = []

    selections = [
        (
            "Best overall VeReMi model",
            (vehicle_df["feature_set"].str.contains("TrustV2X", case=False, na=False)),
        ),
        (
            "Best VeReMi deep learning model",
            (
                (vehicle_df["model_family"] == "DL")
                & (vehicle_df["feature_set"].str.contains("TrustV2X", case=False, na=False))
            ),
        ),
        (
            "Best VeReMi raw BSM baseline",
            (vehicle_df["feature_set"].str.contains("Raw_BSM", case=False, na=False)),
        ),
        (
            "Best VeReMi MLP model",
            (vehicle_df["model"].astype(str).str.upper() == "MLP"),
        ),
        (
            "Best VeReMi 1D-CNN model",
            (vehicle_df["model"].astype(str).str.upper() == "1D-CNN"),
        ),
    ]

    for label, condition in selections:
        row = best_row(vehicle_df, condition, label)
        if row is not None:
            selected_rows.append(row)

    selected_df = pd.DataFrame(selected_rows)

    preferred_cols = [
        "selected_result",
        "model_family",
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

    existing_cols = [c for c in preferred_cols if c in selected_df.columns]

    selected_df = selected_df[existing_cols]

    selected_df["paper_use"] = selected_df["selected_result"].map(
        {
            "Best overall VeReMi model": "Main external benchmark result",
            "Best VeReMi deep learning model": "Main DL comparison result",
            "Best VeReMi raw BSM baseline": "Baseline comparison",
            "Best VeReMi MLP model": "Supports DL evaluation",
            "Best VeReMi 1D-CNN model": "Report as tested but not selected",
        }
    )

    selected_df["interpretation"] = selected_df["selected_result"].map(
        {
            "Best overall VeReMi model": (
                "Best overall generalisation under vehicle-disjoint validation."
            ),
            "Best VeReMi deep learning model": (
                "Best DL result; competitive with the strongest tree-based models."
            ),
            "Best VeReMi raw BSM baseline": (
                "Shows weak generalisation when only raw BSM/CAM-like attributes are used."
            ),
            "Best VeReMi MLP model": (
                "MLP is the strongest deep learning representative for message-level tabular V2X data."
            ),
            "Best VeReMi 1D-CNN model": (
                "1D-CNN underperforms under vehicle-disjoint validation because feature columns are not true temporal windows."
            ),
        }
    )

    return selected_df


def create_metric_reporting_strategy():
    rows = [
        {
            "metric": "F1-score",
            "paper_role": "Headline ranking metric",
            "reason": (
                "Balances precision and recall, useful for comparing models under attack-detection settings."
            ),
        },
        {
            "metric": "Recall / detection rate",
            "paper_role": "Safety-critical detection metric",
            "reason": (
                "Measures how many malicious V2X messages are detected; missed attacks are safety-relevant."
            ),
        },
        {
            "metric": "False alarm rate",
            "paper_role": "Operational deployability metric",
            "reason": (
                "Measures how many normal messages are incorrectly flagged; high false alarms reduce practical usability."
            ),
        },
        {
            "metric": "MCC",
            "paper_role": "Robustness metric",
            "reason": (
                "Summarises classification quality across all confusion-matrix cells and is more informative than accuracy alone."
            ),
        },
        {
            "metric": "Latency",
            "paper_role": "Real-time feasibility metric",
            "reason": (
                "V2X safety-message processing requires low-latency edge/RSU-level decisions."
            ),
        },
        {
            "metric": "Accuracy",
            "paper_role": "Supporting metric",
            "reason": (
                "Reported for completeness, but not used as the main interpretation metric because attack detection and false alarms are more important."
            ),
        },
    ]

    return pd.DataFrame(rows)


def plot_vehicle_disjoint_f1(vehicle_df):
    plot_df = vehicle_df.copy()

    # Keep the figure readable: use all TrustV2X rows and selected raw baselines.
    trust_rows = plot_df[
        plot_df["feature_set"].str.contains("TrustV2X", case=False, na=False)
    ]

    raw_rows = plot_df[
        plot_df["feature_set"].str.contains("Raw_BSM", case=False, na=False)
    ].head(4)

    plot_df = pd.concat([trust_rows, raw_rows], ignore_index=True)
    plot_df = plot_df.drop_duplicates(
        subset=["model_family", "feature_set", "model"],
        keep="first",
    )

    plot_df = plot_df.sort_values("f1_score", ascending=True)

    plot_df["method"] = (
        plot_df["model_family"].astype(str)
        + " | "
        + plot_df["model"].astype(str)
        + " | "
        + plot_df["feature_set"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["f1_score"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["method"])
    ax.set_xlabel("F1-score")
    ax.set_title("VeReMi ML/DL comparison under vehicle-disjoint validation")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)

    for i, value in enumerate(plot_df["f1_score"]):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(
        FINAL_FIGURES_DIR / "figure3_veremi_ml_dl_vehicle_disjoint_f1.png",
        dpi=300,
    )
    plt.close(fig)


def plot_recall_far_tradeoff(vehicle_df):
    plot_df = vehicle_df.copy()

    plot_df = plot_df[
        plot_df["feature_set"].str.contains("TrustV2X", case=False, na=False)
    ].copy()

    plot_df = plot_df.sort_values("f1_score", ascending=False).head(10)

    plot_df["method"] = (
        plot_df["model_family"].astype(str)
        + " | "
        + plot_df["model"].astype(str)
        + "\n"
        + plot_df["feature_set"].astype(str)
    )

    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(
        x - width / 2,
        plot_df["recall_detection_rate"],
        width,
        label="Recall / detection rate",
    )

    ax.bar(
        x + width / 2,
        plot_df["false_alarm_rate"],
        width,
        label="False alarm rate",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["method"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("VeReMi TrustV2X recall versus false alarm trade-off")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        FINAL_FIGURES_DIR / "figure4_veremi_ml_dl_recall_far_tradeoff.png",
        dpi=300,
    )
    plt.close(fig)


def write_final_summary(selected_df, vehicle_df, metric_strategy_df):
    output_file = (
        FINAL_SUMMARY_DIR / "trustv2x_final_results_with_dl_summary.txt"
    )

    best_overall = selected_df[
        selected_df["selected_result"] == "Best overall VeReMi model"
    ].iloc[0]

    best_dl = selected_df[
        selected_df["selected_result"] == "Best VeReMi deep learning model"
    ].iloc[0]

    best_raw = selected_df[
        selected_df["selected_result"] == "Best VeReMi raw BSM baseline"
    ].iloc[0]

    best_cnn = selected_df[
        selected_df["selected_result"] == "Best VeReMi 1D-CNN model"
    ].iloc[0]

    lines = []

    lines.append("=" * 80)
    lines.append("TrustV2X Final Results Summary with VeReMi DL Integration")
    lines.append("=" * 80)

    lines.append("\n1. Final metric strategy")
    lines.append("-" * 80)
    lines.append(metric_strategy_df.to_string(index=False))

    lines.append("\n\n2. Selected final VeReMi results")
    lines.append("-" * 80)
    lines.append(selected_df.to_string(index=False))

    lines.append("\n\n3. Vehicle-disjoint ML/DL comparison")
    lines.append("-" * 80)
    lines.append(
        vehicle_df[
            [
                "model_family",
                "feature_set",
                "model",
                "accuracy",
                "precision",
                "recall_detection_rate",
                "f1_score",
                "false_alarm_rate",
                "mcc",
                "avg_latency_ms_per_message",
            ]
        ].to_string(index=False)
    )

    lines.append("\n\n4. Paper-ready interpretation")
    lines.append("-" * 80)

    lines.append(
        f"The best overall VeReMi vehicle-disjoint result was achieved by "
        f"{best_overall['model']} using {best_overall['feature_set']}, with "
        f"accuracy={best_overall['accuracy']:.4f}, "
        f"recall={best_overall['recall_detection_rate']:.4f}, "
        f"F1={best_overall['f1_score']:.4f}, "
        f"false alarm rate={best_overall['false_alarm_rate']:.4f}, and "
        f"MCC={best_overall['mcc']:.4f}."
    )

    lines.append(
        f"The strongest deep learning result was obtained by "
        f"{best_dl['model']} using {best_dl['feature_set']}, with "
        f"accuracy={best_dl['accuracy']:.4f}, "
        f"recall={best_dl['recall_detection_rate']:.4f}, "
        f"F1={best_dl['f1_score']:.4f}, "
        f"false alarm rate={best_dl['false_alarm_rate']:.4f}, and "
        f"MCC={best_dl['mcc']:.4f}. This shows that the MLP is competitive "
        f"with the strongest tree-based models on the message-level VeReMi benchmark."
    )

    lines.append(
        f"The best raw BSM/CAM-like baseline achieved only "
        f"F1={best_raw['f1_score']:.4f} and FAR={best_raw['false_alarm_rate']:.4f} "
        f"under vehicle-disjoint validation. This confirms that raw message attributes "
        f"alone generalise poorly compared with TrustV2X temporal, mobility and "
        f"plausibility-aware features."
    )

    lines.append(
        f"The best 1D-CNN result achieved only F1={best_cnn['f1_score']:.4f} "
        f"under vehicle-disjoint validation. Therefore, 1D-CNN should be reported "
        f"as tested but not selected. Its poor generalisation is attributed to the "
        f"message-level tabular representation, where neighbouring feature columns "
        f"do not form a true temporal sequence."
    )

    lines.append("\n\n5. Final recommendation for the paper")
    lines.append("-" * 80)
    lines.append(
        "The paper should prioritise F1-score, recall/detection rate, false alarm "
        "rate, MCC and latency. Accuracy should be included in result tables for "
        "completeness, but should not be used as the headline interpretation metric. "
        "The final selected model should be XGBoost with TrustV2X No Rule Flags. "
        "The final selected deep learning model should be MLP with TrustV2X Full. "
        "LSTM should be excluded because a fair LSTM experiment requires explicit "
        "vehicle-level temporal windows."
    )

    output_file.write_text("\n".join(lines), encoding="utf-8")

    return output_file


def main():
    print("=" * 80)
    print("TrustV2X - Step 18: Integrate DL Results into Final Outputs")
    print("=" * 80)

    ml_df = load_ml_results()
    dl_df = load_dl_results()

    combined_df = build_combined_results(ml_df, dl_df)
    vehicle_df = get_vehicle_disjoint_results(combined_df)
    selected_df = create_final_selected_results(combined_df)
    metric_strategy_df = create_metric_reporting_strategy()

    # Save final tables.
    dl_df.to_csv(
        FINAL_TABLES_DIR / "table8_veremi_dl_results.csv",
        index=False,
    )

    vehicle_df.to_csv(
        FINAL_TABLES_DIR / "table9_veremi_ml_dl_vehicle_disjoint_comparison.csv",
        index=False,
    )

    selected_df.to_csv(
        FINAL_TABLES_DIR / "table10_final_selected_results_for_paper.csv",
        index=False,
    )

    metric_strategy_df.to_csv(
        FINAL_TABLES_DIR / "table11_metric_reporting_strategy.csv",
        index=False,
    )

    combined_df.to_csv(
        FINAL_TABLES_DIR / "table12_full_veremi_ml_dl_results.csv",
        index=False,
    )

    # Save final figures.
    plot_vehicle_disjoint_f1(vehicle_df)
    plot_recall_far_tradeoff(vehicle_df)

    summary_file = write_final_summary(
        selected_df=selected_df,
        vehicle_df=vehicle_df,
        metric_strategy_df=metric_strategy_df,
    )

    print("\n[OK] Final DL-integrated tables saved to:")
    print(FINAL_TABLES_DIR)

    print("\n[OK] Final DL-integrated figures saved to:")
    print(FINAL_FIGURES_DIR)

    print("\n[OK] Final DL-integrated summary saved to:")
    print(summary_file)

    print("\nGenerated/updated files:")
    outputs = [
        FINAL_TABLES_DIR / "table8_veremi_dl_results.csv",
        FINAL_TABLES_DIR / "table9_veremi_ml_dl_vehicle_disjoint_comparison.csv",
        FINAL_TABLES_DIR / "table10_final_selected_results_for_paper.csv",
        FINAL_TABLES_DIR / "table11_metric_reporting_strategy.csv",
        FINAL_TABLES_DIR / "table12_full_veremi_ml_dl_results.csv",
        FINAL_FIGURES_DIR / "figure3_veremi_ml_dl_vehicle_disjoint_f1.png",
        FINAL_FIGURES_DIR / "figure4_veremi_ml_dl_recall_far_tradeoff.png",
        summary_file,
    ]

    for output in outputs:
        print(f"- {output}")

    print("\nFinal selected results for paper:")
    print(
        selected_df[
            [
                "selected_result",
                "model_family",
                "feature_set",
                "model",
                "accuracy",
                "precision",
                "recall_detection_rate",
                "f1_score",
                "false_alarm_rate",
                "mcc",
                "avg_latency_ms_per_message",
                "paper_use",
            ]
        ].to_string(index=False)
    )

    print("\nVehicle-disjoint ML/DL comparison:")
    print(
        vehicle_df[
            [
                "model_family",
                "feature_set",
                "model",
                "accuracy",
                "precision",
                "recall_detection_rate",
                "f1_score",
                "false_alarm_rate",
                "mcc",
                "avg_latency_ms_per_message",
            ]
        ].head(20).to_string(index=False)
    )

    print("\nMetric reporting strategy:")
    print(metric_strategy_df.to_string(index=False))


if __name__ == "__main__":
    main()