"""
convert_pdf_to_images.py
------------------------
Convert corporate presentation PDFs to per-slide JPEG images.

Each PDF is rendered at high DPI (default: 500) and each page is saved as
image_1.jpg, image_2.jpg, ... inside a sub-folder named after the PDF.

This is a preprocessing step before running analyze_openai.py, which expects
pre-rendered JPEG images rather than raw PDFs.

Usage:
    python src/pipeline/convert_pdf_to_images.py \\
        --input-dir /data/PPT/files/ppt \\
        --output-dir /data/CSR/images \\
        --dpi 300
"""

import os
import argparse

from pdf2image import convert_from_path
from tqdm import tqdm


def convert_directory(input_dir: str, output_dir: str, dpi: int = 500) -> None:
    """
    Convert all PDFs in input_dir to JPEG image folders under output_dir.

    Args:
        input_dir: Directory containing PDF files.
        output_dir: Root directory where per-PDF image folders will be created.
        dpi: Rendering resolution (higher = sharper, larger files).
    """
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in: {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDFs. Rendering at {dpi} DPI...")

    for filename in tqdm(pdf_files, desc="Converting PDFs"):
        pdf_path = os.path.join(input_dir, filename)
        pdf_name = os.path.splitext(filename)[0]
        pdf_output_dir = os.path.join(output_dir, pdf_name)

        # Skip if already converted
        if os.path.isdir(pdf_output_dir) and os.listdir(pdf_output_dir):
            continue

        os.makedirs(pdf_output_dir, exist_ok=True)

        try:
            pages = convert_from_path(pdf_path, dpi=dpi)
            for idx, page in enumerate(
                tqdm(pages, desc=f"  Pages: {filename}", leave=False)
            ):
                image_path = os.path.join(pdf_output_dir, f"image_{idx + 1}.jpg")
                page.save(image_path, "JPEG")
        except Exception as e:
            print(f"  Failed to process '{filename}': {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert corporate presentation PDFs to per-slide JPEG images."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing PDF files to convert.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory where per-PDF image folders will be created.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=500,
        help="Rendering resolution in DPI (default: 500).",
    )
    args = parser.parse_args()
    convert_directory(args.input_dir, args.output_dir, args.dpi)
    print(f"\nDone. Images saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
