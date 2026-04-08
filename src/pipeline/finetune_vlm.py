"""
finetune_vlm.py
---------------
Fine-tune a Vision Language Model (VLM) on the labeled CSR slide dataset.

Supports two backends selectable via --backend:

  openai      : Uses OpenAI's hosted fine-tuning API.
                Uploads train.jsonl and starts a fine-tuning job on gpt-4o-mini.
                No GPU required. Billed per token.

  huggingface : Fine-tunes an open-source VLM locally using PEFT/LoRA.
                Default base model: Qwen/Qwen2-VL-2B-Instruct.
                Requires a CUDA GPU. Uses 4-bit quantization to reduce VRAM usage.

After training, the fine-tuned model/adapter can be used to classify new slides in
the same way the original Gemini pipeline does — but at much lower inference cost.

Usage:

  # OpenAI hosted (no GPU needed):
  python src/pipeline/finetune_vlm.py \\
      --backend openai \\
      --train-jsonl finetune_data/train.jsonl \\
      --val-jsonl finetune_data/val.jsonl \\
      --epochs 3

  # HuggingFace open-source (GPU required):
  python src/pipeline/finetune_vlm.py \\
      --backend huggingface \\
      --base-model Qwen/Qwen2-VL-2B-Instruct \\
      --train-jsonl finetune_data/train.jsonl \\
      --val-jsonl finetune_data/val.jsonl \\
      --output-dir ./finetuned_csr_model \\
      --epochs 3 \\
      --batch-size 4 \\
      --lora-rank 16

Requirements (install with pip):
  openai, python-dotenv                        # for --backend openai
  torch, transformers, peft, trl,              # for --backend huggingface
  bitsandbytes, accelerate, datasets, Pillow
"""

import os
import json
import time
import argparse

from dotenv import load_dotenv

load_dotenv()


# =========================================================================== #
# OpenAI fine-tuning backend                                                  #
# =========================================================================== #

def finetune_openai(
    train_jsonl: str,
    val_jsonl: str,
    model: str,
    epochs: int,
) -> str:
    """
    Fine-tune a GPT-4o model using OpenAI's hosted fine-tuning API.

    Steps:
      1. Upload train.jsonl (and optionally val.jsonl) to OpenAI Files API.
      2. Create a fine-tuning job.
      3. Poll until the job completes.
      4. Return the fine-tuned model ID.

    Args:
        train_jsonl: Path to the training JSONL (OpenAI format).
        val_jsonl: Path to the validation JSONL (OpenAI format).
        model: Base model to fine-tune (e.g. "gpt-4o-mini").
        epochs: Number of training epochs.

    Returns:
        Fine-tuned model ID string.
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )
    client = OpenAI(api_key=api_key)

    # Upload training file
    print(f"Uploading training file: {train_jsonl}")
    with open(train_jsonl, "rb") as fh:
        train_file = client.files.create(file=fh, purpose="fine-tune")
    print(f"  Uploaded. File ID: {train_file.id}")

    # Upload validation file (optional but recommended)
    val_file_id = None
    if val_jsonl and os.path.exists(val_jsonl):
        print(f"Uploading validation file: {val_jsonl}")
        with open(val_jsonl, "rb") as fh:
            val_file = client.files.create(file=fh, purpose="fine-tune")
        val_file_id = val_file.id
        print(f"  Uploaded. File ID: {val_file_id}")

    # Create fine-tuning job
    create_kwargs = {
        "training_file": train_file.id,
        "model": model,
        "hyperparameters": {"n_epochs": epochs},
    }
    if val_file_id:
        create_kwargs["validation_file"] = val_file_id

    print(f"\nStarting fine-tuning job on model '{model}' for {epochs} epoch(s)...")
    job = client.fine_tuning.jobs.create(**create_kwargs)
    print(f"  Job ID: {job.id}  |  Status: {job.status}")

    # Poll for completion
    print("\nPolling job status (this may take 10–60 minutes)...")
    poll_interval = 60  # seconds
    while job.status not in ("succeeded", "failed", "cancelled"):
        time.sleep(poll_interval)
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"  Status: {job.status}")

    if job.status != "succeeded":
        raise RuntimeError(f"Fine-tuning job ended with status: {job.status}")

    fine_tuned_model = job.fine_tuned_model
    print(f"\nFine-tuning complete! Model ID: {fine_tuned_model}")
    print("Add this model ID to your .env as FINETUNED_MODEL_ID to use it for inference.")
    return fine_tuned_model


def evaluate_openai(
    model_id: str,
    val_jsonl: str,
    n_samples: int = 50,
) -> dict:
    """
    Run the fine-tuned model on validation samples and compute accuracy.

    Args:
        model_id: Fine-tuned model ID from OpenAI.
        val_jsonl: Path to validation JSONL in OpenAI format.
        n_samples: Number of validation examples to evaluate.

    Returns:
        Dict with per-class accuracy metrics.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    with open(val_jsonl) as fh:
        examples = [json.loads(line) for line in fh if line.strip()]

    examples = examples[:n_samples]
    correct = 0

    for ex in examples:
        messages = ex["messages"]
        ground_truth_content = messages[-1]["content"]
        try:
            ground_truth = json.loads(ground_truth_content)
        except json.JSONDecodeError:
            continue

        user_messages = messages[:-1]
        response = client.chat.completions.create(
            model=model_id,
            messages=user_messages,
        )
        prediction_text = response.choices[0].message.content.strip()
        try:
            prediction = json.loads(prediction_text)
            if prediction.get("CSR-related") == ground_truth.get("CSR-related"):
                correct += 1
        except json.JSONDecodeError:
            pass

    accuracy = correct / len(examples) if examples else 0
    print(f"\nValidation accuracy (CSR-related): {accuracy:.1%} ({correct}/{len(examples)})")
    return {"csr_accuracy": accuracy, "n_samples": len(examples)}


# =========================================================================== #
# HuggingFace / PEFT-LoRA fine-tuning backend                                 #
# =========================================================================== #

def finetune_huggingface(
    base_model: str,
    train_jsonl: str,
    val_jsonl: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lora_rank: int,
    lora_alpha: int,
) -> None:
    """
    Fine-tune an open-source VLM using PEFT/LoRA via the trl SFTTrainer.

    The model is loaded in 4-bit quantization (bitsandbytes) to reduce VRAM usage.
    LoRA adapters are applied to the attention layers and saved to output_dir.

    Args:
        base_model: HuggingFace model ID (e.g. "Qwen/Qwen2-VL-2B-Instruct").
        train_jsonl: Path to HuggingFace-format training JSONL.
        val_jsonl: Path to HuggingFace-format validation JSONL.
        output_dir: Directory to save the LoRA adapter weights.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        lora_rank: LoRA rank (r) — higher = more capacity, more memory.
        lora_alpha: LoRA scaling factor (typically 2x rank).
    """
    import torch
    from datasets import Dataset
    from PIL import Image
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoProcessor,
        AutoModelForVision2Seq,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    hf_token = os.environ.get("HF_AUTH_TOKEN")

    # ---- Load dataset ----
    def load_jsonl(path: str) -> list:
        with open(path) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def make_hf_dataset(records: list) -> Dataset:
        """Convert HuggingFace-format JSONL records to a HuggingFace Dataset."""
        rows = {"image": [], "instruction": [], "output": []}
        for rec in records:
            img_path = rec.get("image_path", "")
            if not os.path.exists(img_path):
                continue
            rows["image"].append(Image.open(img_path).convert("RGB"))
            rows["instruction"].append(rec.get("instruction", ""))
            rows["output"].append(rec.get("output", ""))
        return Dataset.from_dict(rows)

    print(f"Loading training data from: {train_jsonl}")
    train_records = load_jsonl(train_jsonl)
    val_records = load_jsonl(val_jsonl) if val_jsonl and os.path.exists(val_jsonl) else []

    train_dataset = make_hf_dataset(train_records)
    val_dataset = make_hf_dataset(val_records) if val_records else None
    print(f"  Training examples: {len(train_dataset)}")
    if val_dataset:
        print(f"  Validation examples: {len(val_dataset)}")

    # ---- Load model in 4-bit ----
    print(f"\nLoading base model: {base_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=hf_token,
    )

    # ---- Apply LoRA ----
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Collate function ----
    def collate_fn(batch):
        """Format each example into the model's expected chat template."""
        texts, images = [], []
        for ex in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex["instruction"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": ex["output"]}]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False)
            texts.append(text)
            images.append([ex["image"]])

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    # ---- Training ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 8 // batch_size),
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=val_dataset is not None,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        peft_config=lora_config,
    )

    print(f"\nStarting training for {epochs} epoch(s)...")
    trainer.train()

    # ---- Save adapter ----
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\nLoRA adapter weights saved to: {output_dir}")
    print("To load for inference:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{output_dir}')")


# =========================================================================== #
# Entry point                                                                  #
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a VLM on the labeled CSR slide dataset."
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "huggingface"],
        required=True,
        help="Fine-tuning backend: 'openai' (hosted API) or 'huggingface' (local, GPU required).",
    )
    parser.add_argument(
        "--train-jsonl",
        required=True,
        help="Path to training JSONL file (output of prepare_finetune_dataset.py).",
    )
    parser.add_argument(
        "--val-jsonl",
        default=None,
        help="(Optional) Path to validation JSONL file.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="[huggingface] Base model ID on HuggingFace Hub (default: Qwen/Qwen2-VL-2B-Instruct).",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="[openai] Base model to fine-tune (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--output-dir",
        default="./finetuned_csr_model",
        help="[huggingface] Directory to save LoRA adapter weights.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="[huggingface] Per-device training batch size (default: 4).",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="[huggingface] LoRA rank r (default: 16).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="[huggingface] LoRA alpha (default: 32 = 2x rank).",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=50,
        help="[openai] Number of validation samples to evaluate after training (default: 50).",
    )
    args = parser.parse_args()

    if args.backend == "openai":
        fine_tuned_model = finetune_openai(
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            model=args.openai_model,
            epochs=args.epochs,
        )
        if args.val_jsonl and os.path.exists(args.val_jsonl):
            evaluate_openai(
                model_id=fine_tuned_model,
                val_jsonl=args.val_jsonl,
                n_samples=args.eval_samples,
            )

    elif args.backend == "huggingface":
        finetune_huggingface(
            base_model=args.base_model,
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )


if __name__ == "__main__":
    main()
