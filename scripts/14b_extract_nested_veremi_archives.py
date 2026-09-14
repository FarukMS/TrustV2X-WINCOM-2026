import zipfile
from pathlib import Path


# ============================================================
# TrustV2X - Step 14b
# Extract Nested VeReMi ZIP Archives
# ============================================================
#
# Problem:
# - VeReMi Extension attack ZIPs contain inner ZIP files.
# - Script 14 only extracted the outer ZIPs.
# - The real JSON/log/ground-truth files are inside the inner ZIPs.
#
# Purpose:
# - Find all nested ZIP files inside data/veremi/extracted/
# - Extract each nested ZIP into a sibling folder ending with _unzipped
# - Avoid extracting the same file repeatedly
#
# This does NOT modify the original TrustV2X 4x4 experiment.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXTRACTED_DIR = PROJECT_ROOT / "data" / "veremi" / "extracted"
RESULTS_DIR = PROJECT_ROOT / "results" / "veremi"
REPORT_FILE = RESULTS_DIR / "veremi_nested_extraction_report.txt"


def extract_nested_zip(zip_path):
    target_dir = zip_path.parent / f"{zip_path.stem}_unzipped"

    if target_dir.exists() and any(target_dir.rglob("*")):
        return {
            "zip": zip_path,
            "target": target_dir,
            "status": "skipped_already_extracted",
            "files": len([p for p in target_dir.rglob("*") if p.is_file()]),
        }

    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

        return {
            "zip": zip_path,
            "target": target_dir,
            "status": "extracted",
            "files": len([p for p in target_dir.rglob("*") if p.is_file()]),
        }

    except Exception as e:
        return {
            "zip": zip_path,
            "target": target_dir,
            "status": f"failed: {e}",
            "files": 0,
        }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TrustV2X VeReMi Nested ZIP Extraction")
    print("=" * 80)

    if not EXTRACTED_DIR.exists():
        print(f"[ERROR] Extracted directory not found: {EXTRACTED_DIR}")
        return

    nested_zips = sorted(EXTRACTED_DIR.rglob("*.zip"))

    if not nested_zips:
        print("[ERROR] No nested ZIP files found.")
        return

    print(f"[INFO] Nested ZIP files found: {len(nested_zips)}")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TrustV2X VeReMi Nested ZIP Extraction Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Nested ZIP files found: {len(nested_zips)}")
    report_lines.append("")

    extracted_count = 0
    skipped_count = 0
    failed_count = 0

    for zip_path in nested_zips:
        rel = zip_path.relative_to(PROJECT_ROOT)

        # Avoid extracting ZIP files that are already inside _unzipped folders.
        if "_unzipped" in str(zip_path):
            continue

        print(f"\n[ZIP] {rel}")
        result = extract_nested_zip(zip_path)

        target_rel = result["target"].relative_to(PROJECT_ROOT)

        print(f"[{result['status']}] -> {target_rel}")
        print(f"[FILES] {result['files']}")

        report_lines.append(f"ZIP: {rel}")
        report_lines.append(f"TARGET: {target_rel}")
        report_lines.append(f"STATUS: {result['status']}")
        report_lines.append(f"FILES: {result['files']}")
        report_lines.append("")

        if result["status"] == "extracted":
            extracted_count += 1
        elif result["status"] == "skipped_already_extracted":
            skipped_count += 1
        else:
            failed_count += 1

    report_lines.append("=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Extracted: {extracted_count}")
    report_lines.append(f"Skipped: {skipped_count}")
    report_lines.append(f"Failed: {failed_count}")

    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("[OK] Nested extraction complete")
    print("=" * 80)
    print(f"[OK] Extracted: {extracted_count}")
    print(f"[OK] Skipped: {skipped_count}")
    print(f"[OK] Failed: {failed_count}")
    print(f"[OK] Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()