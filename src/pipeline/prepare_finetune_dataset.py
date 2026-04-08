"""
prepare_finetune_dataset.py
---------------------------
Convert Gemini CSR analysis outputs into a fine-tuning dataset for Vision Language Models.

Reads the JSONL output from analyze_gemini.py (which contains per-slide CSR labels),
pairs each label with the corresponding slide image, and writes a fine-tuning dataset
in one of two formats:

  - openai     : JSONL compatible with OpenAI vision fine-tuning API (gpt-4o-mini).
                 Images are base64-encoded inline.
  - huggingface: JSONL with image_path, instruction, output fields — compatible with
                 LLaVA-style and Qwen2-VL instruction tuning via trl/SFTTrainer.

The dataset is split into train / val / test splits and a stats summary is saved.

Usage:
    python src/pipeline/prepare_finetune_dataset.py \\
        --results-jsonl /results/final_csr_analysis_2018.json \\
        --images-dir /data/CSR/images/ppt_2018 \\
        --output-dir ./finetune_data \\
        --format openai \\
        --split 0.8 0.1 0.1

    python src/pipeline/prepare_finetune_dataset.py \\
        --results-jsonl /results/final_csr_analysis_2018.json \\
        --images-dir /data/CSR/images/ppt_2018 \\
        --output-dir ./finetune_data \\
        --format huggingface
"""

import os
import json
import base64
import random
import argparse
from pathlib import Path
from collections import Counter

from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Prompt — keep in sync with the prompt used during inference                 #
# --------------------------------------------------------------------------- #
CLASSIFICATION_PROMPT = (
    "Analyze this corporate presentation slide and classify it for ESG/CSR content.\n\n"
    "Task 1: Is this slide CSR/ESG-related? Answer Yes or No.\n"
    "Task 2: If Yes, categorize as one of: CSR Forward (future goals), "
    "CSR Summary (past achievements), or Others (symbolic / unclear).\n"
    "Task 3: If Yes, identify the ESG domain: Environmental, Social, Governance, or Mixed.\n\n"
    "Return a JSON object with exactly these keys:\n"
    '{"CSR-related": "...", "Category": "...", "Domain": "...", "Brief Reasoning": "..."}'
)


def encode_image_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def build_openai_example(image_path: str, label: dict) -> dict:
    """
    Build one OpenAI vision fine-tuning example.

    Format:
        {"messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                {"type": "text", "text": "<prompt>"}
            ]},
            {"role": "assistant", "content": "<json_label>"}
        ]}
    """
    b64 = encode_image_base64(image_path)
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": CLASSIFICATION_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "CSR-related": label.get("csr_related", "No"),
                        "Category": label.get("category", "N/A"),
                        "Domain": _infer_domain(label),
                        "Brief Reasoning": label.get("reasoning", ""),
                    }
                ),
            },
        ]
    }


def build_huggingface_example(image_path: str, label: dict) -> dict:
    """
    Build one HuggingFace / LLaVA-style fine-tuning example.

    Format: {"image_path": "...", "instruction": "...", "output": "..."}
    """
    return {
        "image_path": str(image_path),
        "instruction": CLASSIFICATION_PROMPT,
        "output": json.dumps(
            {
                "CSR-related": label.get("csr_related", "No"),
                "Category": label.get("category", "N/A"),
                "Domain": _infer_domain(label),
                "Brief Reasoning": label.get("reasoning", ""),
            }
        ),
    }


def _infer_domain(label: dict) -> str:
    """Infer a single domain string from the breakdown dict."""
    breakdown = label.get("breakdown") or {}
    domains = []
    if str(breakdown.get("is_environmental", "")).lower() == "yes":
        domains.append("Environmental")
    if str(breakdown.get("is_social", "")).lower() == "yes":
        domains.append("Social")
    if str(breakdown.get("is_governance", "")).lower() == "yes":
        domains.append("Governance")

    if len(domains) == 0:
        return "N/A"
    if len(domains) == 1:
        return domains[0]
    return "Mixed"


def _find_image(images_dir: str, presentation_id: str, slide_number: int) -> str | None:
    """
    Locate the JPEG image for a given presentation and slide number.

    Looks for the pattern: {images_dir}/{presentation_id}/image_{slide_number}.jpg
    """
    candidates = [
        os.path.join(images_dir, presentation_id, f"image_{slide_number}.jpg"),
        os.path.join(images_dir, presentation_id, f"image_{slide_number}.jpeg"),
        os.path.join(images_dir, f"{presentation_id}.pdf", f"image_{slide_number}.jpg"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def build_dataset(
    results_jsonl: str,
    images_dir: str,
    fmt: str,
) -> list:
    """
    Build a list of fine-tuning examples from a Gemini results JSONL file.

    Args:
        results_jsonl: Path to the JSONL output from analyze_gemini.py.
        images_dir: Root directory containing per-presentation image folders.
        fmt: Output format — "openai" or "huggingface".

    Returns:
        List of example dicts ready to write to JSONL.
    """
    examples = []
    missing_images = 0

    with open(results_jsonl, "r", encoding="utf-8") as fh:
        lines = [l.strip() for l in fh if l.strip()]

    for raw in tqdm(lines, desc="Building dataset"):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue

        filename = obj.get("filename", "")
        presentation_id = filename.replace(".pdf", "")
        analysis = obj.get("analysis", {})
        slides = analysis.get("slide_analysis", [])

        for slide in slides:
            if not isinstance(slide, dict):
                continue

            slide_number = slide.get("slide_number")
            if slide_number is None:
                continue

            image_path = _find_image(images_dir, presentation_id, slide_number)
            if image_path is None:
                missing_images += 1
                continue

            if fmt == "openai":
                try:
                    example = build_openai_example(image_path, slide)
                    examples.append(example)
                except Exception as e:
                    print(f"  Skipped {presentation_id} slide {slide_number}: {e}")
            else:
                examples.append(build_huggingface_example(image_path, slide))

    print(f"Built {len(examples)} examples. Missing images: {missing_images}")
    return examples


def split_and_save(
    examples: list,
    output_dir: str,
    split: tuple,
    seed: int = 42,
) -> None:
    """
    Shuffle, split, and save examples as train/val/test JSONL files.

    Args:
        examples: List of example dicts.
        output_dir: Directory to write output files.
        split: Tuple of (train_frac, val_frac, test_frac) summing to 1.0.
        seed: Random seed for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)

    random.seed(seed)
    random.shuffle(examples)

    n = len(examples)
    train_end = int(n * split[0])
    val_end = train_end + int(n * split[1])

    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }

    for name, subset in splits.items():
        out_path = os.path.join(output_dir, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            for ex in subset:
                fh.write(json.dumps(ex) + "\n")
        print(f"  {name}: {len(subset)} examples -> {out_path}")

    # Save class distribution stats
    stats = {}
    for name, subset in splits.items():
        labels = []
        for ex in subset:
            if "messages" in ex:  # openai format
                content = ex["messages"][-1]["content"]
            else:  # huggingface format
                content = ex.get("output", "{}")
            try:
                label_dict = json.loads(content)
                labels.append(label_dict.get("CSR-related", "Unknown"))
            except json.JSONDecodeError:
                pass
        stats[name] = dict(Counter(labels))

    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\nClass distribution saved to: {stats_path}")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Build a VLM fine-tuning dataset from Gemini CSR analysis results."
    )
    parser.add_argument(
        "--results-jsonl",
        required=True,
        help="Path to the JSONL output file from analyze_gemini.py.",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Root directory containing per-presentation image sub-folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="./finetune_data",
        help="Directory to write train/val/test JSONL files (default: ./finetune_data).",
    )
    parser.add_argument(
        "--format",
        choices=["openai", "huggingface"],
        default="huggingface",
        help="Output format: 'openai' (base64 inline) or 'huggingface' (image paths).",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test fractions (must sum to 1.0, default: 0.8 0.1 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    total = sum(args.split)
    if abs(total - 1.0) > 1e-6:
        parser.error(f"Split fractions must sum to 1.0, got {total:.3f}")

    examples = build_dataset(args.results_jsonl, args.images_dir, args.format)

    if not examples:
        print("No examples built — check that image paths match the presentation IDs in the JSONL.")
        return

    split_and_save(examples, args.output_dir, tuple(args.split), args.seed)
    print(f"\nDataset ready in: {args.output_dir}")


if __name__ == "__main__":
    main()
