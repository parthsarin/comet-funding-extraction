from datasets import load_dataset, Dataset
import json
from random import seed, shuffle
from prompt import SYSTEM_PROMPT_GEPA
from dotenv import load_dotenv
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer
from pathlib import Path

assert load_dotenv(), "Failed to load .env file"

def load(with_synthetic=True):
    # load the real dataset
    print("Loading real data from disk... ", end="", flush=True)
    real_data = []
    with open("data/train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            real_data.append({'doi': d['doi'], 'funding_statement': d['funding_statement'], 'label': d['funders']})
    print("done")

    # make a validation set
    seed(42)
    shuffle(real_data)
    val_data = real_data[:264]
    train_data = real_data[264:]

    if not with_synthetic:
        return Dataset.from_list(train_data), Dataset.from_list(val_data)

    # load the synthetic dataset
    print("Loading synthetic data from HF... ", end="", flush=True)
    synthetic_data = load_dataset("cometadata/synthetic-funding-statements", split="train")
    for d in synthetic_data:
        train_data.append({
            "doi": None,
            "funding_statement": d['funding_statement'],
            "label": d['funders']
        })
    print("done")
    return Dataset.from_list(train_data), Dataset.from_list(val_data)

def make_prompt_dataset(funding_dataset):
    return funding_dataset.map(lambda x: {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_GEPA},
            {"role": "user", "content": x["funding_statement"]},
            {"role": "assistant", "content": f"```json\n{json.dumps(x['label'], indent=2)}\n```"}
        ]
    })

def evaluate_model(model, validation_data):
    # 1. calculate validation loss
    pass

    # 2. calculate precision/recall/F1 on funder_name, funding_scheme, award_ids, award_title
    #    (association-aware, i.e., if the model predicts a funder_name, it should also predict
    #     the corresponding funding_scheme and award_ids)
    pass

    return {
        "val_loss": None,
        "val_funder_name": {
            "precision": None,
            "recall": None,
            "f1": None
        },
        "val_funding_scheme": {
            "precision": None,
            "recall": None,
            "f1": None
        },
        "val_award_ids": {
            "precision": None,
            "recall": None,
            "f1": None
        },
        "val_award_title": {
            "precision": None,
            "recall": None,
            "f1": None
        }
    }

def train_model(model, train_data, is_synthetic=False):
    tokenizer = AutoTokenizer.from_pretrained(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.01,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    train_data = train_data.filter(
        lambda row: len(tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True)["input_ids"]) <= 4096,
        num_proc=4,
        desc="Filtering data for max length",
    )
    run_name = (
        "sft-"
        f"{'synthetic' if is_synthetic else 'non_synthetic'}-"
        "llama_3.1_8b_instruct-"
        "lr2e-5-"
        "ep1-"
        "lora_r16-"
        "lora_a32-"
        "ga4-"
        "maxlen4096-"
        "warmup0.05"
    )
    output_dir = Path("test_synthetic_ablation") / "checkpoints" / run_name
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        max_length=4096,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        run_name=run_name,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_data,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Training complete. Model saved to {final_dir}")


def main():
    train_data_no_synthetic, _ = load(with_synthetic=False)
    train_data_with_synthetic, val_data = load(with_synthetic=True)

    train_data_no_synthetic = make_prompt_dataset(train_data_no_synthetic)
    train_data_with_synthetic = make_prompt_dataset(train_data_with_synthetic)
    val_data = make_prompt_dataset(val_data)

    print(f"Train data without synthetic: {len(train_data_no_synthetic)}")
    print(f"Train data with synthetic: {len(train_data_with_synthetic)}")

    model = "meta-llama/Llama-3.1-8B-Instruct"

    # 1. non-synthetic training
    train_model(model, train_data_no_synthetic, is_synthetic=False)

    # 2. synthetic + non-synthetic training
    train_model(model, train_data_with_synthetic, is_synthetic=True)

    # 3. evaluation
    pass

if __name__ == "__main__":
    main()
