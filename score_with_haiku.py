"""
Score GLM-4.5-Air sampled responses using Claude Haiku as a reward model.

Reads raw responses from sample.py output, sends (predicted, ground truth) pairs
to claude-haiku-4-5, and saves scored results.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic

from reward import extract_json_from_response, strip_think_tags

SCORING_PROMPT = """\
You are evaluating the quality of a predicted JSON extraction against the expected ground truth.

Both the predicted and expected values follow this schema: a JSON array of objects, each with:
- "funder_name": string or null
- "awards": array of objects with "funding_scheme" (array), "award_ids" (array), "award_title" (array)

## Predicted JSON
```json
{predicted}
```

## Expected JSON
```json
{expected}
```

## Instructions
Compare the predicted JSON against the expected JSON. Consider:
1. **Funder coverage**: Are all expected funders present? Are there spurious funders?
2. **Funder name accuracy**: Are names correct (minor variations are OK)?
3. **Award ID accuracy**: Are award IDs correct and complete?
4. **Funding scheme accuracy**: Are schemes correctly identified?
5. **Award title accuracy**: Are titles correctly identified?

Score from 0.0 to 1.0 where:
- 1.0 = perfect or near-perfect match
- 0.7+ = mostly correct with minor issues
- 0.4-0.7 = partially correct, some funders/awards missing or wrong
- 0.0-0.4 = mostly wrong or empty

Respond with ONLY a JSON object (no markdown, no explanation outside the JSON):
{{"score": <float>, "reasoning": "<brief explanation>"}}"""


async def score_one(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    predicted_json: str,
    expected_json: str,
) -> dict:
    """Score a single predicted vs expected pair using Haiku."""
    async with semaphore:
        prompt = SCORING_PROMPT.format(
            predicted=predicted_json,
            expected=expected_json,
        )
        try:
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            result = json.loads(text)
            return {
                "score": float(result["score"]),
                "reasoning": result.get("reasoning", ""),
                "error": None,
            }
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            return {"score": 0.0, "reasoning": "", "error": f"Parse error: {e}"}
        except anthropic.APIError as e:
            return {"score": 0.0, "reasoning": "", "error": f"API error: {e}"}


async def score_all(
    input_data: dict,
    concurrency: int = 20,
) -> list[dict]:
    """Score all responses from raw sample output."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    metadata = input_data["metadata"]
    responses = input_data["responses"]

    tasks = []
    task_indices = []  # (entry_idx, sample_idx)

    for entry_idx, (meta, samples) in enumerate(zip(metadata, responses)):
        expected = meta["expected"]
        expected_json = json.dumps(expected, indent=2)

        for sample_idx, raw_response in enumerate(samples):
            cleaned = strip_think_tags(raw_response)
            json_str = extract_json_from_response(cleaned)

            if json_str is None:
                # Can't score if we can't parse - record as zero
                tasks.append(None)
                task_indices.append((entry_idx, sample_idx))
                continue

            tasks.append(
                score_one(client, semaphore, json_str, expected_json)
            )
            task_indices.append((entry_idx, sample_idx))

    # Run all scoring tasks concurrently
    print(f"Scoring {sum(1 for t in tasks if t is not None)} parseable responses "
          f"({sum(1 for t in tasks if t is None)} unparseable)...")

    results_flat = []
    for task in tasks:
        if task is None:
            results_flat.append({
                "score": 0.0,
                "reasoning": "Could not extract JSON from response",
                "error": "parse_failure",
            })
        else:
            results_flat.append(await task)

    # Actually let's gather them properly for concurrency
    # Redo: collect real coroutines and gather them
    return results_flat


async def score_all_concurrent(
    input_data: dict,
    concurrency: int = 20,
) -> dict:
    """Score all responses and build output structure."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    metadata = input_data["metadata"]
    responses = input_data["responses"]

    # Build flat list of scoring coroutines
    coro_list = []
    coro_map = []  # (entry_idx, sample_idx, raw_response, json_str)

    for entry_idx, (meta, samples) in enumerate(zip(metadata, responses)):
        expected = meta["expected"]
        expected_json = json.dumps(expected, indent=2)

        for sample_idx, raw_response in enumerate(samples):
            json_str = extract_json_from_response(raw_response)

            if json_str is None:
                coro_list.append(None)
            else:
                coro_list.append(
                    score_one(client, semaphore, json_str, expected_json)
                )
            coro_map.append((entry_idx, sample_idx, raw_response, json_str))

    parseable = sum(1 for c in coro_list if c is not None)
    unparseable = sum(1 for c in coro_list if c is None)
    print(f"Scoring {parseable} parseable responses ({unparseable} unparseable)...")

    # Gather all real coroutines
    real_coros = [c for c in coro_list if c is not None]
    real_results = await asyncio.gather(*real_coros, return_exceptions=True)

    # Map results back
    real_idx = 0
    scored_entries = []

    # Initialize output structure
    for entry_idx, meta in enumerate(metadata):
        scored_entries.append({
            "doi": meta["doi"],
            "prompt_type": meta["prompt_type"],
            "expected": meta["expected"],
            "samples": [],
        })

    for i, (entry_idx, sample_idx, raw_response, json_str) in enumerate(coro_map):
        if coro_list[i] is None:
            score_result = {
                "score": 0.0,
                "reasoning": "Could not extract JSON from response",
                "error": "parse_failure",
            }
        else:
            r = real_results[real_idx]
            real_idx += 1
            if isinstance(r, Exception):
                score_result = {
                    "score": 0.0,
                    "reasoning": "",
                    "error": f"Exception: {r}",
                }
            else:
                score_result = r

        scored_entries[entry_idx]["samples"].append({
            "sample_idx": sample_idx,
            "response": raw_response,
            "extracted_json": json_str,
            "haiku_score": score_result["score"],
            "haiku_reasoning": score_result["reasoning"],
            "error": score_result["error"],
        })

    # Compute summary stats
    all_scores = [
        s["haiku_score"]
        for entry in scored_entries
        for s in entry["samples"]
    ]
    summary = {
        "total_samples": len(all_scores),
        "mean_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "parse_failure_rate": unparseable / len(all_scores) if all_scores else 0.0,
    }

    return {"summary": summary, "entries": scored_entries}


def main():
    parser = argparse.ArgumentParser(
        description="Score sampled responses using Claude Haiku"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to raw responses JSON from sample.py (the _raw.json file)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/glm_scored.json",
        help="Path to save scored results",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Maximum concurrent API calls",
    )

    args = parser.parse_args()

    # Load raw responses
    print(f"Loading raw responses from: {args.input_path}")
    with open(args.input_path) as f:
        input_data = json.load(f)

    print(f"Loaded {len(input_data['metadata'])} entries, "
          f"{sum(len(r) for r in input_data['responses'])} total samples")

    # Run scoring
    results = asyncio.run(
        score_all_concurrent(input_data, concurrency=args.concurrency)
    )

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Summary: {json.dumps(results['summary'], indent=2)}")


if __name__ == "__main__":
    main()
