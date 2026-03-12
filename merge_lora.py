"""
Merge a LoRA adapter into its base model for vLLM inference.
"""

import argparse

from dotenv import load_dotenv
load_dotenv()

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", type=str, required=True, help="HuggingFace base model name")
    parser.add_argument("--adapter-path", type=str, required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save merged model")

    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_path}")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
