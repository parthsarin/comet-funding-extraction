from datasets import load_dataset, Dataset
import json
from random import seed, shuffle
from prompt import SYSTEM_PROMPT_GEPA
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from rapidfuzz import fuzz
import torch
from tqdm import tqdm

from reward import (
    extract_json_from_response,
    normalize_string,
    normalize_award_id,
)

assert load_dotenv(), "Failed to load .env file"

FUZZY_THRESHOLD = 75.0

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

def fuzzy_match_sets(predicted: list[str], expected: list[str], threshold: float = FUZZY_THRESHOLD) -> tuple[int, int, int]:
    """Match predicted items to expected items using fuzzy matching.
    Returns (TP, FP, FN) counts."""
    if not predicted and not expected:
        return 0, 0, 0

    used_expected = set()
    tp = 0
    for pred in predicted:
        pred_norm = normalize_string(pred)
        best_idx = None
        best_score = 0.0
        for i, exp in enumerate(expected):
            if i in used_expected:
                continue
            score = fuzz.token_set_ratio(pred_norm, normalize_string(exp))
            if score > best_score and score >= threshold:
                best_score = score
                best_idx = i
        if best_idx is not None:
            used_expected.add(best_idx)
            tp += 1

    fp = len(predicted) - tp
    fn = len(expected) - len(used_expected)
    return tp, fp, fn


def exact_match_sets(predicted: list[str], expected: list[str]) -> tuple[int, int, int]:
    """Match predicted items to expected items using exact match after normalization.
    Returns (TP, FP, FN) counts."""
    pred_set = set(normalize_award_id(x) for x in predicted if x)
    exp_set = set(normalize_award_id(x) for x in expected if x)
    tp = len(pred_set & exp_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set - pred_set)
    return tp, fp, fn


def match_funders(predicted: list[dict], expected: list[dict], threshold: float = FUZZY_THRESHOLD) -> list[tuple[dict, dict]]:
    """Match predicted funders to expected funders by funder_name.
    Returns list of (pred_funder, exp_funder) matched pairs, plus unmatched."""
    used_expected = set()
    matched = []
    unmatched_pred = []

    for pred in predicted:
        if not isinstance(pred, dict):
            continue
        pred_name = normalize_string(pred.get("funder_name", "") or "")
        best_idx = None
        best_score = 0.0
        for i, exp in enumerate(expected):
            if i in used_expected:
                continue
            exp_name = normalize_string(exp.get("funder_name", "") or "")
            score = fuzz.token_set_ratio(pred_name, exp_name)
            if score > best_score and score >= threshold:
                best_score = score
                best_idx = i
        if best_idx is not None:
            used_expected.add(best_idx)
            matched.append((pred, expected[best_idx]))
        else:
            unmatched_pred.append(pred)

    unmatched_exp = [exp for i, exp in enumerate(expected) if i not in used_expected]
    return matched, unmatched_pred, unmatched_exp


def extract_field_lists(funders: list[dict], field: str) -> list[str]:
    """Extract all values for a field from a list of funder objects' awards."""
    values = []
    for funder in funders:
        for award in funder.get("awards", []) or []:
            values.extend(award.get(field, []) or [])
    return values


def compute_prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_model(base_model: str, adapter_path: str, val_data, max_new_tokens: int = 4096):
    """Evaluate a trained model on validation data.

    Args:
        base_model: HF model id (e.g. meta-llama/Llama-3.1-8B-Instruct)
        adapter_path: path to the LoRA adapter checkpoint
        val_data: dataset with 'funding_statement' and 'label' columns
        max_new_tokens: max tokens to generate
    """
    print(f"Loading model {base_model} with adapter {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 1. Validation loss ---
    print("Computing validation loss...")
    total_loss = 0.0
    n_loss = 0
    for row in tqdm(val_data, desc="Val loss"):
        inputs = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, return_tensors="pt",
            return_dict=True, add_generation_prompt=False,
        ).to(model.device)
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
        total_loss += out.loss.item()
        n_loss += 1
    val_loss = total_loss / n_loss if n_loss > 0 else float("inf")
    print(f"Validation loss: {val_loss:.4f}")

    # --- 2. Generate predictions and compute P/R/F1 ---
    print("Generating predictions...")
    # Micro-averaged counters
    funder_tp, funder_fp, funder_fn = 0, 0, 0
    aid_tp, aid_fp, aid_fn = 0, 0, 0
    scheme_tp, scheme_fp, scheme_fn = 0, 0, 0
    title_tp, title_fp, title_fn = 0, 0, 0
    parse_failures = 0

    for row in tqdm(val_data, desc="Generating"):
        expected = row["label"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_GEPA},
            {"role": "user", "content": row["funding_statement"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        json_str = extract_json_from_response(response)
        if json_str is None:
            parse_failures += 1
            # Treat as empty prediction
            predicted = []
        else:
            try:
                predicted = json.loads(json_str)
                if not isinstance(predicted, list):
                    predicted = []
            except json.JSONDecodeError:
                parse_failures += 1
                predicted = []

        # funder_name P/R/F1 (association-aware matching)
        matched, unmatched_pred, unmatched_exp = match_funders(predicted, expected)
        funder_tp += len(matched)
        funder_fp += len(unmatched_pred)
        funder_fn += len(unmatched_exp)

        # For matched funders: compare their awards field-by-field
        for pred_f, exp_f in matched:
            pred_awards = pred_f.get("awards", []) or []
            exp_awards = exp_f.get("awards", []) or []

            pred_ids = [aid for a in pred_awards for aid in (a.get("award_ids", []) or [])]
            exp_ids = [aid for a in exp_awards for aid in (a.get("award_ids", []) or [])]
            t, f, m = exact_match_sets(pred_ids, exp_ids)
            aid_tp += t; aid_fp += f; aid_fn += m

            pred_schemes = [s for a in pred_awards for s in (a.get("funding_scheme", []) or [])]
            exp_schemes = [s for a in exp_awards for s in (a.get("funding_scheme", []) or [])]
            t, f, m = fuzzy_match_sets(pred_schemes, exp_schemes)
            scheme_tp += t; scheme_fp += f; scheme_fn += m

            pred_titles = [s for a in pred_awards for s in (a.get("award_title", []) or [])]
            exp_titles = [s for a in exp_awards for s in (a.get("award_title", []) or [])]
            t, f, m = fuzzy_match_sets(pred_titles, exp_titles)
            title_tp += t; title_fp += f; title_fn += m

        # Unmatched predicted funders: their awards are all FP
        for pred_f in unmatched_pred:
            pred_awards = pred_f.get("awards", []) or [] if isinstance(pred_f, dict) else []
            aid_fp += sum(len(a.get("award_ids", []) or []) for a in pred_awards)
            scheme_fp += sum(len(a.get("funding_scheme", []) or []) for a in pred_awards)
            title_fp += sum(len(a.get("award_title", []) or []) for a in pred_awards)

        # Unmatched expected funders: their awards are all FN
        for exp_f in unmatched_exp:
            exp_awards = exp_f.get("awards", []) or []
            aid_fn += sum(len(a.get("award_ids", []) or []) for a in exp_awards)
            scheme_fn += sum(len(a.get("funding_scheme", []) or []) for a in exp_awards)
            title_fn += sum(len(a.get("award_title", []) or []) for a in exp_awards)

    print(f"Parse failures: {parse_failures}/{len(val_data)}")

    return {
        "val_loss": val_loss,
        "parse_failure_rate": parse_failures / len(val_data) if len(val_data) > 0 else 0.0,
        "val_funder_name": compute_prf(funder_tp, funder_fp, funder_fn),
        "val_award_ids": compute_prf(aid_tp, aid_fp, aid_fn),
        "val_funding_scheme": compute_prf(scheme_tp, scheme_fp, scheme_fn),
        "val_award_title": compute_prf(title_tp, title_fp, title_fn),
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
    run_dir = Path("test_synthetic_ablation") / run_name
    checkpoint_dir = run_dir / "checkpoints"

    config = {
        "base_model": model,
        "with_synthetic": is_synthetic,
        "train_size": len(train_data),
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.01,
        "learning_rate": 2e-5,
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "max_length": 4096,
        "bf16": True,
    }

    sft_config = SFTConfig(
        output_dir=str(checkpoint_dir),
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
    final_dir = checkpoint_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save run config
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training complete. Model saved to {final_dir}")
    return run_name, run_dir, str(final_dir)


def main():
    train_data_no_synthetic, _ = load(with_synthetic=False)
    train_data_with_synthetic, val_data = load(with_synthetic=True)

    train_data_no_synthetic = make_prompt_dataset(train_data_no_synthetic)
    train_data_with_synthetic = make_prompt_dataset(train_data_with_synthetic)
    val_data = make_prompt_dataset(val_data)

    print(f"Train data without synthetic: {len(train_data_no_synthetic)}")
    print(f"Train data with synthetic: {len(train_data_with_synthetic)}")

    base_model = "meta-llama/Llama-3.1-8B-Instruct"

    # 1. non-synthetic training
    run_name_ns, run_dir_ns, adapter_ns = train_model(base_model, train_data_no_synthetic, is_synthetic=False)

    # 2. synthetic + non-synthetic training
    run_name_s, run_dir_s, adapter_s = train_model(base_model, train_data_with_synthetic, is_synthetic=True)

    # 3. evaluation
    for run_name, run_dir, adapter_path in [
        (run_name_ns, run_dir_ns, adapter_ns),
        (run_name_s, run_dir_s, adapter_s),
    ]:
        print(f"\n{'='*60}")
        print(f"Evaluating: {run_name}")
        print(f"{'='*60}")
        eval_results = evaluate_model(base_model, adapter_path, val_data)
        eval_path = run_dir / "eval_results.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"Eval results saved to {eval_path}")
        print(json.dumps(eval_results, indent=2))

if __name__ == "__main__":
    main()
