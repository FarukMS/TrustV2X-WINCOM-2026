from pathlib import Path
from collections import Counter
import zipfile
import tarfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "veremi" / "raw"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "veremi" / "extracted"
RESULTS_DIR = PROJECT_ROOT / "results" / "veremi"

REPORT_FILE = RESULTS_DIR / "veremi_file_structure_report.txt"


def is_possible_archive(path):
    name = path.name.lower()

    return (
        name.endswith(".zip")
        or name.endswith(".tar")
        or name.endswith(".tar.gz")
        or name.endswith(".tgz")
        or name.endswith(".tar.xz")
        or name.endswith(".tar.bz2")
        or name.endswith(".gz")
        or name.endswith(".7z")
        or name.endswith(".rar")
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append("=" * 80)
    lines.append("VeReMi File Structure Inspection")
    lines.append("=" * 80)

    lines.append("\nRAW ZIP FILES:")
    raw_files = sorted(RAW_DIR.glob("*"))

    if not raw_files:
        lines.append("No files found in data/veremi/raw/")
    else:
        for path in raw_files:
            lines.append(f"{path.name} | {path.stat().st_size / (1024 * 1024):.2f} MB")

    lines.append("\nEXTRACTED FILE SUMMARY:")

    extracted_files = [p for p in EXTRACTED_DIR.rglob("*") if p.is_file()]

    lines.append(f"Total extracted files: {len(extracted_files)}")

    if not extracted_files:
        lines.append("No files found inside data/veremi/extracted/")
    else:
        extension_counter = Counter()
        suffix_counter = Counter()
        archive_candidates = []

        for path in extracted_files:
            extension_counter[path.suffix.lower()] += 1
            suffix_counter["".join(path.suffixes).lower()] += 1

            if is_possible_archive(path):
                archive_candidates.append(path)

        lines.append("\nExtension counts:")
        for ext, count in extension_counter.most_common(50):
            lines.append(f"{ext or '[no extension]'}: {count}")

        lines.append("\nFull suffix counts:")
        for ext, count in suffix_counter.most_common(50):
            lines.append(f"{ext or '[no suffix]'}: {count}")

        lines.append("\nPossible nested archives:")
        if archive_candidates:
            for path in archive_candidates[:100]:
                rel = path.relative_to(PROJECT_ROOT)
                lines.append(f"{rel} | {path.stat().st_size / (1024 * 1024):.2f} MB")
        else:
            lines.append("No obvious nested archives found.")

        lines.append("\nFirst 120 extracted files:")
        for path in extracted_files[:120]:
            rel = path.relative_to(PROJECT_ROOT)
            lines.append(f"{rel} | {path.stat().st_size / 1024:.2f} KB")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\n" + "=" * 80)
    print(f"[OK] Report saved to: {REPORT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()