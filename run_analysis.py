"""Run the ECG and physical-activity analysis from the command line."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ecg_activity.analysis import cluster_and_classify, feature_statistics
from ecg_activity.features import extract_ecg_features
from ecg_activity.preprocessing import clean_ecg, read_ecg_excel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecg-dir", required=True, type=Path, help="Folder of participant ECG Excel files")
    parser.add_argument("--lifestyle", required=True, type=Path, help="CSV containing subject_id and mvpa_minutes")
    parser.add_argument("--output", default=Path("results"), type=Path, help="Output folder")
    parser.add_argument("--sampling-rate", default=1000.0, type=float, help="ECG sampling rate in Hz")
    parser.add_argument("--skiprows", default=1, type=int, help="Rows to skip when reading ECG files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    paths = sorted([*args.ecg_dir.glob("*.xls"), *args.ecg_dir.glob("*.xlsx")])
    if not paths:
        raise FileNotFoundError(f"No Excel files found in {args.ecg_dir}")

    for path in paths:
        time, raw_signal = read_ecg_excel(path, skiprows=args.skiprows)
        cleaned_signal = clean_ecg(raw_signal, args.sampling_rate)
        record = extract_ecg_features(time, cleaned_signal, args.sampling_rate)
        record["subject_id"] = path.stem.removeprefix("Cleaned_")
        records.append(record)

    features = pd.DataFrame(records)
    lifestyle = pd.read_csv(args.lifestyle)
    required = {"subject_id", "mvpa_minutes"}
    if not required.issubset(lifestyle.columns):
        raise ValueError("Lifestyle CSV must contain subject_id and mvpa_minutes.")

    merged = features.merge(lifestyle, on="subject_id", how="inner")
    if merged.empty:
        raise ValueError("No subject_id values matched between ECG files and the lifestyle CSV.")
    merged["activity_category"] = merged["mvpa_minutes"].apply(
        lambda value: "Active" if value >= 150 else "Sedentary"
    )

    clustered, metrics = cluster_and_classify(merged)
    statistics = feature_statistics(merged)
    args.output.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output / "ecg_features.csv", index=False)
    merged.to_csv(args.output / "merged_analysis_data.csv", index=False)
    clustered[["subject_id", "activity_category", "cluster"]].to_csv(
        args.output / "cluster_assignments.csv", index=False
    )
    statistics.to_csv(args.output / "feature_statistics.csv", index=False)
    (args.output / "summary_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
