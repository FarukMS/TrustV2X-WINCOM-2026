from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "veremi" / "processed" / "veremi_extension_trustv2x_external_benchmark.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 80)
print("VeReMi Label Source Inspection")
print("=" * 80)
print(f"Dataset: {DATA_FILE}")
print(f"Shape: {df.shape}")

print("\nColumns:")
for c in df.columns:
    print("-", c)

candidate_cols = [
    "label",
    "label_by_token",
    "label_by_error",
    "attack_type",
    "attack_token",
    "source_archive",
    "source_file",
    "attackerType",
    "attacker_type",
]

print("\nCandidate label/source columns:")
for c in candidate_cols:
    if c in df.columns:
        print(f"\n{c}:")
        print(df[c].value_counts(dropna=False).head(30))

if "label" in df.columns and "label_by_token" in df.columns:
    print("\nCrosstab: current label vs label_by_token")
    print(pd.crosstab(df["label"], df["label_by_token"], rownames=["label"], colnames=["label_by_token"]))

if "label" in df.columns and "attack_type" in df.columns:
    print("\nCrosstab: current label vs attack_type")
    print(pd.crosstab(df["label"], df["attack_type"], rownames=["label"], colnames=["attack_type"]))

if "label_by_token" in df.columns and "attack_type" in df.columns:
    print("\nCrosstab: label_by_token vs attack_type")
    print(pd.crosstab(df["label_by_token"], df["attack_type"], rownames=["label_by_token"], colnames=["attack_type"]))

print("\nRecommended next step:")
print("- If label_by_token exists and clearly separates normal/attack, use label_by_token as the source-metadata label.")
print("- If label_by_token is not available, use attack_type != 'normal' as the source-metadata label.")
print("- Do not use position_error, speed_error, acceleration_error or heading_error_degrees as input features.")