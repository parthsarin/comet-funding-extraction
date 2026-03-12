from datasets import load_dataset, Dataset
import json
from random import seed, shuffle

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

def main():
    pass

if __name__ == "__main__":
    main()
