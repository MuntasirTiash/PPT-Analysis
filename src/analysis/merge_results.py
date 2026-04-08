"""
merge_results.py
----------------
Utilities for merging CSR analysis results with financial data (CAR).

Functions:
    merge_car       - Load and merge Cumulative Abnormal Returns with presentation metadata.
    merge_CSR       - Aggregate per-slide CSR labels into per-presentation feature ratios.
    analys_results  - Parse the JSONL output from analyze_gemini.py into a flat DataFrame.

Usage (CLI):
    python src/analysis/merge_results.py \\
        --json-path /results/final_csr_analysis_2018.json \\
        --output-csv /results/csr_features_2018.csv

    # Full pipeline (merge with financial data):
    python src/analysis/merge_results.py \\
        --json-path /results/final_csr_analysis_2018.json \\
        --car-path /data/CAR/3factor_post4_13.csv \\
        --index-path /data/index/data(mp3+ppt)_2018.xlsx \\
        --output-csv /results/merged_2018.csv
"""

import os
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def merge_car(
    car_path: str,
    ppt_path: str,
    column_name: str = None,
) -> pd.DataFrame:
    """
    Load and merge CAR (Cumulative Abnormal Returns) data with presentation metadata.

    Args:
        car_path: Path to the CAR CSV file containing columns: ticker, evtdate, car.
        ppt_path: Path to the Excel index file with ppt_id, company_name, date.
        column_name: Column name for the CAR variable. Defaults to the CSV filename stem.

    Returns:
        DataFrame with merged ticker, date, ppt_id, and CAR columns.
    """
    car = pd.read_csv(car_path)
    ticker = pd.read_excel(ppt_path)

    ticker = ticker[ticker["ppt_id"].notnull()]
    ticker["ticker"] = ticker["company_name"].str.extract(r":([A-Z\.]+)", expand=False)
    ticker = ticker[ticker["ticker"].notnull()]

    ticker["date_object"] = pd.to_datetime(ticker["date"], format="%b-%d-%Y %I:%M %p")
    ticker["formatted_date"] = ticker["date_object"].dt.strftime("%Y-%m-%d")

    if column_name is None:
        column_name = Path(car_path).stem  # e.g. "3factor_post4_13"

    merged_df = pd.merge(
        ticker,
        car[["ticker", "evtdate", "car"]],
        left_on=["ticker", "formatted_date"],
        right_on=["ticker", "evtdate"],
        how="left",
    ).rename(columns={"car": column_name})

    return merged_df


def merge_CSR(
    merge_df: pd.DataFrame,
    results_dir: str,
) -> pd.DataFrame:
    """
    Aggregate per-slide CSR labels into per-presentation feature ratios and merge with
    financial data.

    Args:
        merge_df: DataFrame with a 'ppt_id' column (from merge_car output).
        results_dir: Directory containing *_csr_analysis.json files (one per presentation).

    Returns:
        DataFrame merged on ppt_id with CSR feature ratios added.
    """
    category_labels = ["CSR Forward", "CSR Summary", "Others", "Mixed"]
    domain_labels = ["Environmental", "Social", "Governance", "Mixed"]

    summary_list = []

    for file_name in os.listdir(results_dir):
        if not file_name.endswith("_csr_analysis.json"):
            continue

        file_path = os.path.join(results_dir, file_name)
        presentation_id = file_name.split("_")[0]

        try:
            with open(file_path) as fh:
                data = json.load(fh)
                slides = data.get("slides", [])
        except Exception as e:
            print(f"  Error reading {file_name}: {e}")
            continue

        total_slides = len(slides)
        csr_slides = [s for s in slides if s.get("CSR-related") == "Yes"]
        csr_count = len(csr_slides)

        category_counter = Counter(
            s.get("Category", "Others") or "Others" for s in csr_slides
        )
        domain_counter = Counter(
            s.get("Domain", "Others") or "Others" for s in csr_slides
        )

        row = {
            "presentation_id": presentation_id,
            "total_slides": total_slides,
            "csr_count": csr_count,
            "csr_ratio": csr_count / total_slides if total_slides else 0,
        }
        for cat in category_labels:
            key = f"csr_{cat.lower().replace(' ', '_')}_ratio"
            row[key] = category_counter[cat] / total_slides if total_slides else 0
        for dom in domain_labels:
            row[f"csr_{dom.lower()}_ratio"] = (
                domain_counter[dom] / total_slides if total_slides else 0
            )

        summary_list.append(row)

    df_summary = pd.DataFrame(summary_list)

    merge_df = merge_df[merge_df["ppt_id"].notnull()].copy()
    merge_df["ppt_id"] = merge_df["ppt_id"].astype(float).astype(int).astype(str)
    df_summary["presentation_id"] = df_summary["presentation_id"].astype(str)

    merged = pd.merge(
        merge_df, df_summary,
        left_on="ppt_id", right_on="presentation_id",
        how="inner",
    )
    return merged


def analys_results(json_path: str) -> pd.DataFrame:
    """
    Parse the JSONL output from analyze_gemini.py into a flat DataFrame.

    Each line of the JSONL file contains one presentation with per-slide analysis.
    This function aggregates slide-level labels into per-presentation counts and ratios.

    Args:
        json_path: Path to the JSONL output file from analyze_gemini.py.

    Returns:
        DataFrame with one row per presentation and columns for CSR counts and ratios.
    """
    results = []
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            filename = obj.get("filename", "unknown")
            analysis = obj.get("analysis", {})
            slides = analysis.get("slide_analysis", [])
            total_slides = len(slides)

            total_csr = 0
            env_cnt = soc_cnt = gov_cnt = 0
            cat_cnts: dict = defaultdict(int)

            for slide in slides:
                try:
                    if not isinstance(slide, dict):
                        continue
                    if str(slide.get("csr_related", "")).lower() != "yes":
                        continue

                    total_csr += 1
                    breakdown = slide.get("breakdown") or {}
                    if str(breakdown.get("is_environmental", "")).lower() == "yes":
                        env_cnt += 1
                    if str(breakdown.get("is_social", "")).lower() == "yes":
                        soc_cnt += 1
                    if str(breakdown.get("is_governance", "")).lower() == "yes":
                        gov_cnt += 1

                    cat = slide.get("category", "Uncategorized")
                    cat_cnts[cat] += 1
                except Exception as e:
                    print(f"  Skipped slide in {filename}: {e}")

            row = {
                "filename": filename,
                "total_slides": total_slides,
                "csr_slide_count": total_csr,
                "csr_ratio": round(total_csr / total_slides, 3) if total_slides else 0,
                "env_count": env_cnt,
                "soc_count": soc_cnt,
                "gov_count": gov_cnt,
                "env_ratio": round(env_cnt / total_slides, 3) if total_slides else 0,
                "soc_ratio": round(soc_cnt / total_slides, 3) if total_slides else 0,
                "gov_ratio": round(gov_cnt / total_slides, 3) if total_slides else 0,
            }
            for cat, cnt in cat_cnts.items():
                key = cat.lower().replace(" ", "_")
                row[f"{key}_count"] = cnt
                row[f"{key}_ratio"] = round(cnt / total_slides, 3) if total_slides else 0

            results.append(row)

    return pd.DataFrame(results).fillna(0)


def main():
    parser = argparse.ArgumentParser(
        description="Parse Gemini CSR analysis JSONL output into a CSV feature table."
    )
    parser.add_argument(
        "--json-path",
        required=True,
        help="Path to the JSONL output file from analyze_gemini.py.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to write the resulting CSV.",
    )
    parser.add_argument(
        "--car-path",
        default=None,
        help="(Optional) Path to CAR CSV for merging financial data.",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="(Optional) Path to Excel index file for merging financial data.",
    )
    args = parser.parse_args()

    df = analys_results(args.json_path)
    print(f"Parsed {len(df)} presentations from {args.json_path}")

    if args.car_path and args.index_path:
        merged_df = merge_car(args.car_path, args.index_path)
        merged_df["ppt_id_str"] = (
            merged_df["ppt_id"].dropna().astype(float).astype(int).astype(str)
        )
        df["ppt_id_str"] = df["filename"].str.replace(".pdf", "", regex=False)
        df = pd.merge(df, merged_df, left_on="ppt_id_str", right_on="ppt_id_str", how="inner")
        print(f"After merging with financial data: {len(df)} rows")

    df.to_csv(args.output_csv, index=False)
    print(f"Saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
