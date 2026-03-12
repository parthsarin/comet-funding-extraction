"""
Evaluate a model on the test set using transformers (no vLLM).

Supports both merged models and base+adapter loading via PEFT.

Usage:
    # Merged model:
    python eval_with_transformers.py \
        --model checkpoints/qwen3.5-9b-sft-distill-merged \
        --data-path data/test.jsonl \
        --output-path results/eval_qwen3.5_distill.json

    # Base model + LoRA adapter:
    python eval_with_transformers.py \
        --model Qwen/Qwen3.5-9B \
        --adapter checkpoints/sft-Qwen3.5-9B-distill-20260302-0917/checkpoint-258 \
        --data-path data/test.jsonl \
        --output-path results/eval_qwen3.5_distill.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import SYSTEM_PROMPT_ON_FUNDING_STATEMENT
from reward import (
    calculate_reward,
    extract_json_from_response,
    format_errors,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model using transformers")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter (if not using a merged model)")
    parser.add_argument("--data-path", type=str, default="data/test.jsonl")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking mode (for thinking-trained models)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}")
    data = []
    with open(args.data_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} entries")

    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter if args.adapter else args.model,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if args.adapter:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {args.adapter}")

        # Fix key mismatch: adapters trained on multimodal Qwen3.5 have
        # "language_model." in keys, but AutoModelForCausalLM doesn't.
        from safetensors.torch import load_file, save_file
        import tempfile, shutil, json as json_mod
        adapter_path = Path(args.adapter)
        weights = load_file(adapter_path / "adapter_model.safetensors")
        if any("language_model" in k for k in weights.keys()):
            print("  Remapping adapter keys (removing language_model prefix)")
            remapped = {
                k.replace(".language_model.", "."): v
                for k, v in weights.items()
            }
            tmp_adapter = Path(tempfile.mkdtemp())
            save_file(remapped, tmp_adapter / "adapter_model.safetensors")
            # Copy config and update base_model paths too
            with open(adapter_path / "adapter_config.json") as f:
                cfg = json_mod.load(f)
            with open(tmp_adapter / "adapter_config.json", "w") as f:
                json_mod.dump(cfg, f, indent=2)
            model = PeftModel.from_pretrained(model, str(tmp_adapter))
            shutil.rmtree(tmp_adapter)
        else:
            model = PeftModel.from_pretrained(model, args.adapter)

        model = model.merge_and_unload()
        print(f"Adapter merged. Model type: {type(model).__name__}")

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build prompts and run inference
    entry_stats = []
    for entry in tqdm(data, desc="Evaluating"):
        doi = entry["doi"]
        funding_statement = entry.get("funding_statement", "")
        expected = entry.get("funders", [])

        if not funding_statement:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_ON_FUNDING_STATEMENT},
            {
                "role": "user",
                "content": f"Please extract funding information from the following statement:\n\n{funding_statement}",
            },
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=1.0,
                do_sample=False,
            )

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Parse and score
        json_str = extract_json_from_response(response)
        parsed = None
        parse_error = None
        reward = 0.0

        if json_str is not None:
            try:
                parsed = json.loads(json_str)
                result = calculate_reward(parsed, expected, verbose=args.verbose)
                reward = result.total_reward
                if args.verbose and result.errors:
                    print(f"\nDOI: {doi}")
                    print(f"Mean reward: {reward:.4f}")
                    print(f"  {format_errors(result.errors)}")
            except json.JSONDecodeError as e:
                parse_error = str(e)
        else:
            parse_error = "Could not extract JSON"

        entry_stats.append({
            "doi": doi,
            "prompt_type": "funding_statement",
            "rewards": [reward],
            "best_reward": reward,
            "mean_reward": reward,
            "parse_success_rate": 1.0 if parsed is not None else 0.0,
            "samples": [{
                "sample_idx": 0,
                "response": response,
                "parsed_json": parsed,
                "parse_error": parse_error,
                "reward": reward,
            }],
        })

    # Aggregate
    all_rewards = [e["mean_reward"] for e in entry_stats]
    aggregate = {
        "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
        "total_entries": len(entry_stats),
    }

    results = {"aggregate_stats": aggregate, "entries": entry_stats}

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Mean reward: {aggregate['mean_reward']:.4f}")
    print(f"Entries: {aggregate['total_entries']}")


if __name__ == "__main__":
    main()
