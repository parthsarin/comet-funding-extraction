"""
Score sampled responses using a local LLM as a reward model via vLLM.

Reads raw responses from sample.py output, sends (predicted, ground truth) pairs
to a local LLM, and saves scored results.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from reward import extract_json_from_response

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


def build_scoring_prompts(input_data: dict) -> tuple[list[list[dict]], list[tuple[int, int, str, str | None]]]:
    """Build scoring prompts from raw sample output.

    Returns:
        prompts: list of chat message lists for vLLM
        prompt_map: list of (entry_idx, sample_idx, raw_response, extracted_json) tuples
    """
    metadata = input_data["metadata"]
    responses = input_data["responses"]

    prompts = []
    prompt_map = []

    for entry_idx, (meta, samples) in enumerate(zip(metadata, responses)):
        expected = meta["expected"]
        expected_json = json.dumps(expected, indent=2)

        for sample_idx, raw_response in enumerate(samples):
            json_str = extract_json_from_response(raw_response)

            if json_str is None:
                # Can't score if we can't parse — will record as zero later
                prompt_map.append((entry_idx, sample_idx, raw_response, None))
                continue

            user_content = SCORING_PROMPT.format(
                predicted=json_str,
                expected=expected_json,
            )
            prompts.append([{"role": "user", "content": user_content}])
            prompt_map.append((entry_idx, sample_idx, raw_response, json_str))

    return prompts, prompt_map


def parse_score_response(text: str) -> dict:
    """Parse a scoring response into score + reasoning."""
    from reward import strip_think_tags
    text = strip_think_tags(text)
    text = extract_json_from_response(text) or text.strip()
    # Try to find a JSON object
    try:
        # Handle case where model wraps in ```json
        import re
        obj_match = re.search(r'\{[\s\S]*\}', text)
        if obj_match:
            result = json.loads(obj_match.group(0))
            return {
                "score": float(result["score"]),
                "reasoning": result.get("reasoning", ""),
                "error": None,
            }
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return {"score": 0.0, "reasoning": "", "error": f"Could not parse score from: {text[:200]}"}


def main():
    parser = argparse.ArgumentParser(
        description="Score sampled responses using a local LLM via vLLM"
    )
    parser.add_argument(
        "--input-path", type=str, required=True,
        help="Path to raw responses JSON from sample.py (the _raw.json file)",
    )
    parser.add_argument(
        "--output-path", type=str, default="results/glm_scored.json",
        help="Path to save scored results",
    )
    parser.add_argument(
        "--model", type=str, default="zai-org/GLM-4.5-Air",
        help="HuggingFace model name for scoring",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=8,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=16384,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="GPU memory utilization fraction",
    )

    args = parser.parse_args()

    # Load raw responses
    print(f"Loading raw responses from: {args.input_path}")
    with open(args.input_path) as f:
        input_data = json.load(f)

    total_samples = sum(len(r) for r in input_data["responses"])
    print(f"Loaded {len(input_data['metadata'])} entries, {total_samples} total samples")

    # Build scoring prompts
    prompts, prompt_map = build_scoring_prompts(input_data)
    parseable = len(prompts)
    unparseable = len(prompt_map) - parseable
    print(f"Scoring {parseable} parseable responses ({unparseable} unparseable)")

    # Run vLLM inference for scoring
    if prompts:
        print(f"Loading model: {args.model}")
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            n=1,
        )

        print("Running scoring inference...")
        outputs = llm.chat(messages=prompts, sampling_params=sampling_params, use_tqdm=True)
        score_responses = [o.outputs[0].text for o in outputs]
    else:
        score_responses = []

    # Build output structure
    metadata = input_data["metadata"]
    scored_entries = []
    for entry_idx, meta in enumerate(metadata):
        scored_entries.append({
            "doi": meta["doi"],
            "prompt_type": meta["prompt_type"],
            "expected": meta["expected"],
            "samples": [],
        })

    # Map results back
    score_idx = 0
    for entry_idx, sample_idx, raw_response, json_str in prompt_map:
        if json_str is None:
            score_result = {
                "score": 0.0,
                "reasoning": "Could not extract JSON from response",
                "error": "parse_failure",
            }
        else:
            score_result = parse_score_response(score_responses[score_idx])
            score_idx += 1

        scored_entries[entry_idx]["samples"].append({
            "sample_idx": sample_idx,
            "response": raw_response,
            "extracted_json": json_str,
            "haiku_score": score_result["score"],  # keep key name for compatibility
            "haiku_reasoning": score_result["reasoning"],
            "error": score_result["error"],
        })

    # Compute summary stats
    all_scores = [s["haiku_score"] for entry in scored_entries for s in entry["samples"]]
    summary = {
        "total_samples": len(all_scores),
        "mean_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "parse_failure_rate": unparseable / len(all_scores) if all_scores else 0.0,
        "scorer_model": args.model,
    }

    results = {"summary": summary, "entries": scored_entries}

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
