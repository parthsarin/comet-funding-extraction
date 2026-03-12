"""
Unified SFT training for funding statement extraction.

Two modes:
  - Distillation: Load pre-built SFT JSONL (from build_sft_dataset.py)
  - Ground truth: Build SFT dataset directly from train.jsonl labels
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from prompt import SYSTEM_PROMPT_ON_FUNDING_STATEMENT


def load_sft_data(path: str) -> list[dict]:
    """Load pre-built SFT JSONL (messages format)."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} SFT examples from {path}")
    return records


def build_ground_truth_sft(path: str) -> list[dict]:
    """Build SFT dataset from ground truth train.jsonl."""
    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            funding_statement = entry.get("funding_statement", "")
            funders = entry.get("funders", [])

            if not funding_statement:
                continue

            # Format funders as JSON string for the assistant response
            if isinstance(funders, str):
                funders_json = funders
            else:
                funders_json = json.dumps(funders, indent=2)

            records.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_ON_FUNDING_STATEMENT},
                    {
                        "role": "user",
                        "content": f"Please extract funding information from the following statement:\n\n{funding_statement}",
                    },
                    {"role": "assistant", "content": funders_json},
                ],
            })

    print(f"Built {len(records)} SFT examples from ground truth at {path}")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="SFT training for funding statement extraction"
    )

    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--sft-data-path",
        type=str,
        help="Path to pre-built SFT JSONL (distillation mode)",
    )
    data_group.add_argument(
        "--ground-truth-path",
        type=str,
        help="Path to train.jsonl (ground truth mode)",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name",
    )

    # Training hyperparameters
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--report-to", type=str, default="wandb")

    args = parser.parse_args()

    # Load data
    if args.sft_data_path:
        records = load_sft_data(args.sft_data_path)
        mode = "distill"
    else:
        records = build_ground_truth_sft(args.ground_truth_path)
        mode = "gt"

    if not records:
        print("No training data found. Exiting.")
        return

    # Build HF dataset
    dataset = Dataset.from_list(records)

    # Load tokenizer
    model_short = args.model.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.01,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"sft-{model_short}-{mode}-{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # SFT config
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        max_length=args.max_seq_length,
        logging_steps=10,
        save_strategy="epoch",
        report_to=args.report_to,
        run_name=run_name,
    )

    # Trainer
    trainer = SFTTrainer(
        model=args.model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Train
    print(f"Starting SFT training: {run_name}")
    print(f"  Model: {args.model}")
    print(f"  Mode: {mode}")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Output: {output_dir}")

    trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Training complete. Model saved to {final_dir}")


if __name__ == "__main__":
    main()
