"""
Build SFT dataset from Haiku-scored responses.

Filters high-reward samples and formats them as chat messages for SFT training.
"""

import argparse
import json
import re
from pathlib import Path

from prompt import SYSTEM_PROMPT_ON_FUNDING_STATEMENT


def extract_thinking(response: str) -> tuple[str | None, str]:
    """Extract thinking trace and answer from a response.

    Returns (thinking, answer) where thinking may be None.
    """
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = response[match.end() :].strip()
        return thinking, answer
    return None, response.strip()


def format_with_thinking(thinking: str | None, json_answer: str) -> str:
    """Format assistant content with optional thinking trace."""
    if thinking:
        return f"<think>\n{thinking}\n</think>\n\n{json_answer}"
    return json_answer


def build_sft_dataset(
    scored_path: str,
    data_path: str,
    output_path: str,
    score_threshold: float = 0.7,
    max_samples_per_entry: int = 1,
    include_thinking: bool = False,
):
    """Filter scored samples and build SFT JSONL."""
    # Load scored results
    with open(scored_path) as f:
        scored = json.load(f)

    # Load original data to get funding statements
    funding_statements = {}
    with open(data_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                funding_statements[entry["doi"]] = entry.get("funding_statement", "")

    total_samples = 0
    kept_samples = 0
    skipped_no_statement = 0
    skipped_no_thinking = 0
    output_records = []

    for entry in scored["entries"]:
        doi = entry["doi"]
        funding_statement = funding_statements.get(doi, "")

        if not funding_statement:
            skipped_no_statement += 1
            continue

        # Collect qualifying samples sorted by score (descending)
        qualifying = [
            s for s in entry["samples"]
            if s["haiku_score"] >= score_threshold
            and s["extracted_json"] is not None
            and s.get("error") is None
        ]
        qualifying.sort(key=lambda s: s["haiku_score"], reverse=True)

        # Keep top N samples per entry
        for sample in qualifying[:max_samples_per_entry]:
            total_samples += 1

            if include_thinking:
                thinking, _ = extract_thinking(sample.get("response", ""))
                if thinking is None:
                    skipped_no_thinking += 1
                    continue
                assistant_content = format_with_thinking(
                    thinking, sample["extracted_json"]
                )
            else:
                assistant_content = sample["extracted_json"]

            kept_samples += 1
            output_records.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_ON_FUNDING_STATEMENT},
                    {
                        "role": "user",
                        "content": f"Please extract funding information from the following statement:\n\n{funding_statement}",
                    },
                    {"role": "assistant", "content": assistant_content},
                ],
                "doi": doi,
                "haiku_score": sample["haiku_score"],
            })

        total_samples += len(entry["samples"])

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in output_records:
            f.write(json.dumps(record) + "\n")

    print(f"Built SFT dataset: {kept_samples} samples from {len(scored['entries'])} entries")
    print(f"  Score threshold: {score_threshold}")
    print(f"  Max samples per entry: {max_samples_per_entry}")
    print(f"  Include thinking: {include_thinking}")
    print(f"  Skipped (no funding statement): {skipped_no_statement}")
    if include_thinking:
        print(f"  Skipped (no thinking trace): {skipped_no_thinking}")
    print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build SFT dataset from Haiku-scored responses"
    )
    parser.add_argument(
        "--scored-path",
        type=str,
        default="results/glm_scored.json",
        help="Path to Haiku-scored results JSON",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/train.jsonl",
        help="Path to original training JSONL (for funding statements)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/sft_glm_distill.jsonl",
        help="Output path for SFT JSONL",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.7,
        help="Minimum Haiku score to include a sample",
    )
    parser.add_argument(
        "--max-samples-per-entry",
        type=int,
        default=1,
        help="Maximum samples to keep per DOI",
    )
    parser.add_argument(
        "--include-thinking",
        action="store_true",
        help="Include <think> traces in assistant responses",
    )

    args = parser.parse_args()
    build_sft_dataset(
        scored_path=args.scored_path,
        data_path=args.data_path,
        output_path=args.output_path,
        score_threshold=args.score_threshold,
        max_samples_per_entry=args.max_samples_per_entry,
        include_thinking=args.include_thinking,
    )


if __name__ == "__main__":
    main()
