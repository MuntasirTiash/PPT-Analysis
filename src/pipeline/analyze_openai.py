"""
analyze_openai.py
-----------------
Analyze corporate presentation slides for CSR/ESG content using OpenAI GPT-4.1-mini.

Each presentation folder contains pre-rendered slide images (JPEGs). Each image is
uploaded to the OpenAI Files API, then analyzed for CSR relevance, temporal category,
and ESG domain. Results are saved as per-presentation JSON files.

This script is an alternative to analyze_gemini.py for cases where OpenAI's vision
model is preferred.

Usage:
    python src/pipeline/analyze_openai.py \\
        --images-dir /data/CSR/2010 \\
        --output-dir /results/openai/2010 \\
        --car-path /data/CAR/3factor_post4_13.csv \\
        --index-path /data/index/data(mp3+ppt)_2010.xlsx

    # Process without financial filtering:
    python src/pipeline/analyze_openai.py \\
        --images-dir /data/CSR/2010 \\
        --output-dir /results/openai/2010 \\
        --no-filter
"""

import os
import json
import time
import argparse

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

from src.analysis.merge_results import merge_car

load_dotenv()

ANALYSIS_PROMPT = """
Act as a financial analyst. You are given a corporate conference call presentation
consisting of multiple slide images.

For each slide, perform the following three tasks:

Task 1: Determine whether the slide contains CSR-related imagery.

A CSR-related image is any visual content representing the company's efforts or messaging
related to environmental sustainability, social responsibility, or corporate governance:

- Environmental: visuals of trees, recycling, solar panels, wind turbines, electric vehicles,
  or pollution reduction efforts.
- Social: diversity and inclusion, community service, employee well-being, education, or
  healthcare support.
- Governance: ethics, transparency, board structure, compliance, or anti-corruption.
- Symbolic: ESG badges, UN SDGs, sustainability infographics, or CSR logos.

Classify the slide as:
CSR-related: [Yes / No]
Brief Reasoning: One or two sentences explaining why.

Task 2: If CSR-related, classify into one of:
- CSR Forward: forward-looking content (goals, upcoming initiatives, net-zero targets).
- CSR Summary: backward-looking content (past performance, historical ESG scores, completed milestones).
- Others: symbolic branding or vague visuals not clearly Forward or Summary.

Task 3: If CSR-related, identify the domain:
- Environmental: sustainability, climate, pollution, energy, recycling.
- Social: people, equity, diversity, education, health, communities.
- Governance: ethics, leadership, corporate structure, compliance, accountability.
- Mixed: spans multiple domains.

Return your output as:
{
  "CSR-related": "[Yes / No]",
  "Category": "[CSR Forward / CSR Summary / Others]",
  "Domain": "[Environmental / Social / Governance / Mixed]",
  "Brief Reasoning": "Concise explanation for all decisions"
}
"""


def analyze_presentation(client: OpenAI, presentation_dir: str, output_path: str) -> None:
    """
    Analyze all slide images in a presentation directory and save results.

    Args:
        client: Initialized OpenAI client.
        presentation_dir: Directory containing JPEG slide images.
        output_path: Path to write the JSON result file.
    """
    presentation_id = os.path.basename(presentation_dir)

    image_files = sorted(
        f for f in os.listdir(presentation_dir) if f.lower().endswith(".jpg")
    )

    results = {"presentation": presentation_id, "slides": []}

    for img_file in tqdm(image_files, desc=f"  Slides in {presentation_id}", leave=False):
        img_path = os.path.join(presentation_dir, img_file)

        try:
            with open(img_path, "rb") as fh:
                uploaded = client.files.create(file=fh, purpose="vision")

            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": ANALYSIS_PROMPT},
                            {"type": "input_image", "file_id": uploaded.id},
                        ],
                    }
                ],
            )

            output_text = response.output_text.strip()
            try:
                slide_result = json.loads(output_text)
                for key in ["CSR-related", "Category", "Domain", "Brief Reasoning"]:
                    slide_result.setdefault(key, "Missing")
            except json.JSONDecodeError:
                slide_result = {
                    "CSR-related": "Error",
                    "Category": "Error",
                    "Domain": "Error",
                    "Brief Reasoning": output_text,
                }

        except Exception as e:
            slide_result = {
                "CSR-related": "Error",
                "Category": "Error",
                "Domain": "Error",
                "Brief Reasoning": f"Error: {e}",
            }

        slide_result["filename"] = img_file
        results["slides"].append(slide_result)
        time.sleep(1)  # Respect rate limits

    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=2)

    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze corporate presentation slides for CSR/ESG content using OpenAI GPT-4.1-mini."
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing per-presentation subdirectories with slide JPEGs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write per-presentation JSON result files.",
    )
    parser.add_argument(
        "--car-path",
        default=None,
        help="Path to the CAR CSV file. Defaults to CAR_PATH env var.",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="Path to the Excel index file mapping ppt_id to company/date.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Process all presentation folders without filtering by ppt_id.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )

    client = OpenAI(api_key=api_key)
    os.makedirs(args.output_dir, exist_ok=True)

    # Build filtered list of presentation directories
    if args.no_filter:
        ppt_ids = set(
            d for d in os.listdir(args.images_dir)
            if os.path.isdir(os.path.join(args.images_dir, d))
        )
    else:
        car_path = args.car_path or os.environ.get("CAR_PATH", "")
        index_path = args.index_path or os.environ.get("INDEX_BASE_DIR", "")
        merged_df = merge_car(car_path=car_path, ppt_path=index_path)
        ppt_ids = set(
            merged_df["ppt_id"].dropna().astype(int).astype(str).tolist()
        )

    presentation_dirs = sorted(
        os.path.join(args.images_dir, pid)
        for pid in ppt_ids
        if os.path.isdir(os.path.join(args.images_dir, pid))
    )

    print(f"Presentations to process: {len(presentation_dirs)}")

    for pdir in tqdm(presentation_dirs, desc="Processing presentations"):
        pid = os.path.basename(pdir)
        output_path = os.path.join(args.output_dir, f"{pid}_csr_analysis.json")
        if os.path.exists(output_path):
            continue  # Resume support
        analyze_presentation(client, pdir, output_path)

    print(f"\nDone. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
