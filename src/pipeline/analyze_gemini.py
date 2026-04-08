"""
analyze_gemini.py
-----------------
Analyze corporate presentation slides for CSR/ESG content using Google Gemini 2.5 Flash.

Each PDF is converted to slide images, uploaded to the Gemini API, and classified
per-slide for CSR relevance, temporal category (Forward/Summary), and ESG domain
(Environmental/Social/Governance).

Results are saved to a JSONL file (one JSON object per line) for easy resumption.

Usage:
    python src/pipeline/analyze_gemini.py --year 2018 \\
        --car-path /data/CAR/3factor_post4_13.csv \\
        --output-dir /results/gemini/

    # Run without financial data filtering (e.g. for 2022+ data):
    python src/pipeline/analyze_gemini.py --year 2022 --no-filter \\
        --pdf-dir /data/ppt_2022/ppt --output-dir /results/gemini/
"""

import os
import json
import time
import argparse

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image
import pandas as pd

from src.analysis.merge_results import merge_car

load_dotenv()

MODEL_NAME = "gemini-2.5-flash-preview-05-20"

IMAGE_PROMPT_TEMPLATE = """
# Role
Act as an expert financial analyst specializing in ESG (Environmental, Social, and Governance) reporting.

# Task
You will be given a series of images, each representing a single slide from a corporate conference call
presentation. Analyze the entire presentation for CSR content by holistically examining each slide.
Provide a high-level summary of the presentation's overall CSR focus, followed by a detailed,
slide-by-slide breakdown.

# Analysis Criteria

## Task 1: CSR Identification and Breakdown
Determine if the slide conveys a CSR-related message. If it does, specify which pillar(s)
(Environmental, Social, Governance) it belongs to. A slide can belong to more than one.

- **Environmental**: trees, nature, recycling, solar panels, wind turbines, electric vehicles,
  or pollution reduction efforts.
- **Social**: diversity and inclusion, employee well-being, community engagement, education,
  or healthcare initiatives.
- **Governance**: business ethics, transparency, board structure, compliance, or anti-corruption.
- **Symbolic**: ESG badges, CSR logos, UN SDGs, or sustainability infographics.

## Task 2: Temporal Category (if CSR-related)
- **CSR Forward**: forward-looking statements, future goals, targets, or upcoming initiatives.
- **CSR Summary**: backward-looking information, past performance, historical data, or completed milestones.
- **Others**: CSR-related but doesn't clearly fit Forward or Summary (e.g., symbolic branding).

# Required Output Format
Return a single valid JSON object with two top-level keys: `overall_summary` and `slide_analysis`.

```json
{
  "overall_summary": {
    "primary_csr_focus": "[Environmental / Social / Governance / Mixed / None]",
    "overall_tone": "[Forward-Looking / Backward-Looking / Mixed / N/A]",
    "summary_text": "A one or two-sentence summary of the presentation's key CSR themes."
  },
  "slide_analysis": [
    {
      "slide_number": 1,
      "csr_related": "No",
      "breakdown": null,
      "category": "N/A",
      "reasoning": "Title page with no CSR content."
    },
    {
      "slide_number": 2,
      "csr_related": "Yes",
      "breakdown": {
        "is_environmental": "Yes",
        "is_social": "No",
        "is_governance": "No"
      },
      "category": "CSR Summary",
      "reasoning": "Solar panel imagery reporting on past energy savings achieved last year."
    }
  ]
}
```
"""


def analyze_presentations_from_images(
    filenames_to_process: list,
    api_key: str,
    pdf_directory: str,
    output_file_path: str,
    processed_filenames: set = None,
    limit: int = None,
) -> None:
    """
    Analyze a list of PDF presentation files for CSR/ESG content using Gemini Vision.

    Args:
        filenames_to_process: List of PDF filenames to analyze.
        api_key: Google Gemini API key.
        pdf_directory: Directory containing the PDF files.
        output_file_path: Path to the JSONL output file.
        processed_filenames: Set of already-processed filenames (for resumption).
        limit: If set, process at most this many files (useful for testing).
    """
    if processed_filenames is None:
        processed_filenames = set()

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={"response_mime_type": "application/json"},
    )

    files_to_run = [f for f in filenames_to_process if f not in processed_filenames]
    if limit:
        files_to_run = files_to_run[:limit]

    if not files_to_run:
        print("All files already processed. Nothing to do.")
        return

    print(
        f"Total PDFs in directory : {len(filenames_to_process)}\n"
        f"Already processed       : {len(processed_filenames)}\n"
        f"To process now          : {len(files_to_run)}"
    )

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "a", encoding="utf-8") as out_f:
        for filename in tqdm(files_to_run, desc="Analyzing Presentations"):
            pdf_path = os.path.join(pdf_directory, filename)

            # --- Render PDF to images ---
            slide_images = []
            try:
                doc = fitz.open(pdf_path)
                for page in doc:
                    pix = page.get_pixmap(dpi=150)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    slide_images.append(img)
                doc.close()
            except Exception as e_fitz:
                print(f"\n  PyMuPDF failed on '{filename}': {e_fitz}. Trying pdf2image...")
                try:
                    slide_images = convert_from_path(pdf_path, dpi=150)
                except Exception as e_pdf2:
                    print(f"\n  Both renderers failed for '{filename}': {e_pdf2}")
                    result = {
                        "filename": filename,
                        "analysis": {"error": f"Failed to render PDF: {e_pdf2}"},
                    }
                    json.dump(result, out_f)
                    out_f.write("\n")
                    continue

            if not slide_images:
                result = {
                    "filename": filename,
                    "analysis": {"error": "No images could be generated from PDF."},
                }
                json.dump(result, out_f)
                out_f.write("\n")
                continue

            # --- Call Gemini API with retry ---
            api_input = [IMAGE_PROMPT_TEMPLATE] + slide_images
            max_retries = 3
            result = None

            for attempt in range(1, max_retries + 1):
                try:
                    response = model.generate_content(api_input)
                    analysis_json = json.loads(response.text)
                    result = {"filename": filename, "analysis": analysis_json}
                    break
                except Exception as e:
                    print(f"\n  Error on '{filename}' (attempt {attempt}/{max_retries}): {e}")
                    if attempt == max_retries:
                        result = {
                            "filename": filename,
                            "analysis": {"error": f"Failed after {max_retries} retries: {e}"},
                        }
                    time.sleep(2)

            json.dump(result, out_f)
            out_f.write("\n")


def _load_processed(output_file_path: str) -> set:
    """Return the set of filenames already written to the output JSONL."""
    processed = set()
    if not os.path.exists(output_file_path):
        return processed
    print(f"Resuming from existing output file: {output_file_path}")
    try:
        with open(output_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if "filename" in data and "error" not in data.get("analysis", {}):
                        processed.add(data["filename"])
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not fully parse resume file: {e}")
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Analyze corporate presentation slides for CSR/ESG content using Gemini Vision."
    )
    parser.add_argument(
        "--year",
        required=True,
        help="Year of the dataset to process (e.g. 2018). Used to resolve paths from env vars.",
    )
    parser.add_argument(
        "--car-path",
        default=None,
        help="Path to the CAR CSV file. Defaults to CAR_PATH env var.",
    )
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help=(
            "Override the PDF input directory. "
            "Defaults to {PDF_BASE_DIR}/ppt_{year} from env vars."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Override the output directory. "
            "Defaults to {RESULTS_BASE_DIR}/{year} from env vars."
        ),
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help=(
            "Path to the Excel index file for this year. "
            "Defaults to {INDEX_BASE_DIR}/data(mp3+ppt)_{year}.xlsx from env vars."
        ),
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip ppt_id filtering (process all PDFs in the directory).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to the first N files (useful for testing).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )

    # Resolve paths
    pdf_base = os.environ.get("PDF_BASE_DIR", "")
    results_base = os.environ.get("RESULTS_BASE_DIR", "")
    index_base = os.environ.get("INDEX_BASE_DIR", "")
    car_path = args.car_path or os.environ.get("CAR_PATH", "")

    pdf_directory = args.pdf_dir or os.path.join(pdf_base, f"ppt_{args.year}")
    output_dir = args.output_dir or os.path.join(results_base, args.year)
    output_file_path = os.path.join(output_dir, f"final_csr_analysis_{args.year}.json")
    index_path = args.index_path or os.path.join(
        index_base, f"data(mp3+ppt)_{args.year}.xlsx"
    )

    # Get list of PDFs
    try:
        all_pdfs = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
    except FileNotFoundError:
        print(f"Error: PDF directory not found: '{pdf_directory}'")
        raise SystemExit(1)

    # Optionally filter by valid ppt_ids from merged financial data
    if args.no_filter:
        filtered_pdfs = all_pdfs
        print(f"--no-filter: processing all {len(filtered_pdfs)} PDFs.")
    else:
        merged_df = merge_car(car_path=car_path, ppt_path=index_path)
        ppt_id_list = (
            merged_df["ppt_id"]
            .dropna()
            .astype(str)
            .loc[lambda x: x.str.isdigit()]
            .astype(int)
            .astype(str)
            .tolist()
        )
        filtered_pdfs = [f for f in all_pdfs if f.replace(".pdf", "") in ppt_id_list]
        print(
            f"Total PDFs: {len(all_pdfs)} | "
            f"Matching financial data: {len(filtered_pdfs)}"
        )

    processed = _load_processed(output_file_path)

    analyze_presentations_from_images(
        filenames_to_process=filtered_pdfs,
        api_key=api_key,
        pdf_directory=pdf_directory,
        output_file_path=output_file_path,
        processed_filenames=processed,
        limit=args.limit,
    )

    print(f"\nDone. Results saved to: {output_file_path}")


if __name__ == "__main__":
    main()
