"""
Sampling script for funding statement extraction using vLLM.

Runs inference with Qwen3-8B on funding statements and full article markdown,
calculates rewards, and reports statistics.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from prompt import SYSTEM_PROMPT_ON_FUNDING_STATEMENT, SYSTEM_PROMPT_ON_ENTIRE_ARTICLE
from reward import (
    RewardResult,
    calculate_reward,
    extract_json_from_response,
    format_errors,
)


@dataclass
class SampleResult:
    """Result for a single sample."""
    doi: str
    prompt_type: str  # "funding_statement" or "full_markdown"
    sample_idx: int
    response: str
    parsed_json: list | None
    reward_result: RewardResult | None
    parse_error: str | None = None


@dataclass
class EntryStats:
    """Statistics for a single dataset entry."""
    doi: str
    prompt_type: str
    rewards: list[float]
    best_reward: float
    mean_reward: float
    parse_success_rate: float
    samples: list[SampleResult]


def load_dataset(path: str) -> list[dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_prompts(
    data: list[dict],
    tokenizer: AutoTokenizer,
    max_context_length: int = 32768,
    include_funding_statement: bool = True,
    include_full_markdown: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Create prompts for all entries in the dataset.

    Returns:
        Tuple of (prompt_metadata_list, conversation_list)
        Each conversation is a list of messages for vLLM chat template.
    """
    prompts = []
    metadata = []
    skipped_funding = 0
    skipped_markdown = 0
    truncated_markdown = 0

    for entry in data:
        doi = entry.get("doi", "unknown")
        funding_statement = entry.get("funding_statement", "")
        markdown = entry.get("markdown", "")
        funders = entry.get("funders", [])

        # Parse funders if it's a string
        if isinstance(funders, str):
            try:
                funders = json.loads(funders)
            except json.JSONDecodeError:
                funders = []

        # Prompt type 1: Funding statement only
        if include_funding_statement and funding_statement:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_ON_FUNDING_STATEMENT},
                {"role": "user", "content": f"Please extract funding information from the following statement:\n\n{funding_statement}"}
            ]
            token_count = len(tokenizer.apply_chat_template(messages, tokenize=True))

            if token_count < max_context_length - 2048:  # Leave room for generation
                prompts.append(messages)
                metadata.append({
                    "doi": doi,
                    "prompt_type": "funding_statement",
                    "expected": funders
                })
            else:
                skipped_funding += 1

        # Prompt type 2: Full markdown
        if include_full_markdown and markdown:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_ON_ENTIRE_ARTICLE},
                {"role": "user", "content": f"Please extract funding information from the following article:\n\n{markdown}"}
            ]
            token_count = len(tokenizer.apply_chat_template(messages, tokenize=True))

            if token_count < max_context_length - 2048:  # Leave room for generation
                prompts.append(messages)
                metadata.append({
                    "doi": doi,
                    "prompt_type": "full_markdown",
                    "expected": funders
                })
            else:
                # Try truncating markdown to fit
                # Binary search for the right truncation length
                system_tokens = len(tokenizer.encode(SYSTEM_PROMPT_ON_ENTIRE_ARTICLE))
                available_tokens = max_context_length - system_tokens - 2048 - 100  # buffer for chat template overhead

                if available_tokens > 2000:
                    # Estimate chars to keep (start conservative, ~2 chars per token)
                    char_limit = available_tokens * 2
                    truncated_text = markdown[:char_limit]

                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT_ON_ENTIRE_ARTICLE},
                        {"role": "user", "content": f"Please extract funding information from the following article (truncated due to length):\n\n{truncated_text}"}
                    ]
                    # Verify it fits
                    actual_tokens = len(tokenizer.apply_chat_template(messages, tokenize=True))

                    if actual_tokens < max_context_length - 2048:
                        truncated_markdown += 1
                        prompts.append(messages)
                        metadata.append({
                            "doi": doi,
                            "prompt_type": "full_markdown_truncated",
                            "expected": funders
                        })
                    else:
                        skipped_markdown += 1
                else:
                    skipped_markdown += 1

    print(f"Prompt stats: {len(prompts)} created, {skipped_funding} funding skipped, "
          f"{skipped_markdown} markdown skipped, {truncated_markdown} markdown truncated")

    return metadata, prompts


def run_inference(
    model_name: str,
    prompts: list[list[dict]],
    k: int = 5,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    max_model_len: int = 32768,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> list[list[str]]:
    """
    Run inference with vLLM.

    Args:
        model_name: HuggingFace model name
        prompts: List of conversation prompts
        k: Number of samples per prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        max_model_len: Maximum model context length
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory fraction to use

    Returns:
        List of lists, where each inner list contains k responses for a prompt
    """
    print(f"Loading model: {model_name}")
    print(f"Max model length: {max_model_len}")
    print(f"Tensor parallel size: {tensor_parallel_size}")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=k,  # Generate k samples per prompt
    )

    print(f"Running inference on {len(prompts)} prompts with k={k} samples each...")

    # vLLM handles batching internally
    outputs = llm.chat(
        messages=prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    # Extract responses
    results = []
    for output in outputs:
        prompt_responses = [o.text for o in output.outputs]
        results.append(prompt_responses)

    return results


def process_results(
    metadata: list[dict],
    responses: list[list[str]],
    verbose: bool = False,
) -> tuple[list[EntryStats], dict]:
    """
    Process inference results and calculate rewards.

    Returns:
        Tuple of (entry_stats_list, aggregate_statistics)
    """
    entry_stats = []
    all_rewards = {
        "funding_statement": [],
        "full_markdown": [],
        "full_markdown_truncated": [],
    }
    parse_failures = {
        "funding_statement": 0,
        "full_markdown": 0,
        "full_markdown_truncated": 0,
    }
    total_by_type = {
        "funding_statement": 0,
        "full_markdown": 0,
        "full_markdown_truncated": 0,
    }

    for meta, prompt_responses in tqdm(
        zip(metadata, responses),
        total=len(metadata),
        desc="Processing results"
    ):
        doi = meta["doi"]
        prompt_type = meta["prompt_type"]
        expected = meta["expected"]

        samples = []
        rewards = []
        parse_successes = 0

        for idx, response in enumerate(prompt_responses):
            # Try to extract JSON from response
            json_str = extract_json_from_response(response)
            parsed_json = None
            reward_result = None
            parse_error = None

            if json_str:
                try:
                    parsed_json = json.loads(json_str)
                    reward_result = calculate_reward(parsed_json, expected, verbose=verbose)
                    rewards.append(reward_result.total_reward)
                    parse_successes += 1
                except json.JSONDecodeError as e:
                    parse_error = f"JSON decode error: {e}"
                    rewards.append(0.0)
            else:
                parse_error = "No JSON array found in response"
                rewards.append(0.0)

            samples.append(SampleResult(
                doi=doi,
                prompt_type=prompt_type,
                sample_idx=idx,
                response=response,
                parsed_json=parsed_json,
                reward_result=reward_result,
                parse_error=parse_error,
            ))

        # Calculate stats for this entry
        best_reward = max(rewards) if rewards else 0.0
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        parse_success_rate = parse_successes / len(prompt_responses) if prompt_responses else 0.0

        entry_stats.append(EntryStats(
            doi=doi,
            prompt_type=prompt_type,
            rewards=rewards,
            best_reward=best_reward,
            mean_reward=mean_reward,
            parse_success_rate=parse_success_rate,
            samples=samples,
        ))

        # Aggregate stats
        all_rewards[prompt_type].extend(rewards)
        parse_failures[prompt_type] += len(prompt_responses) - parse_successes
        total_by_type[prompt_type] += len(prompt_responses)

    # Calculate aggregate statistics
    aggregate_stats = {}
    for ptype in ["funding_statement", "full_markdown", "full_markdown_truncated"]:
        if all_rewards[ptype]:
            aggregate_stats[ptype] = {
                "mean_reward": sum(all_rewards[ptype]) / len(all_rewards[ptype]),
                "max_reward": max(all_rewards[ptype]),
                "min_reward": min(all_rewards[ptype]),
                "total_samples": len(all_rewards[ptype]),
                "parse_failure_rate": parse_failures[ptype] / total_by_type[ptype] if total_by_type[ptype] > 0 else 0,
            }

    return entry_stats, aggregate_stats


def print_statistics(entry_stats: list[EntryStats], aggregate_stats: dict, verbose: bool = False):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS SUMMARY")
    print("=" * 80)

    for ptype, stats in aggregate_stats.items():
        print(f"\n{ptype.upper().replace('_', ' ')}:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Mean reward: {stats['mean_reward']:.4f}")
        print(f"  Max reward: {stats['max_reward']:.4f}")
        print(f"  Min reward: {stats['min_reward']:.4f}")
        print(f"  Parse failure rate: {stats['parse_failure_rate']:.2%}")

    # Overall stats
    all_mean_rewards = [e.mean_reward for e in entry_stats]
    all_best_rewards = [e.best_reward for e in entry_stats]

    print(f"\nOVERALL (across all prompt types):")
    print(f"  Total entries: {len(entry_stats)}")
    print(f"  Mean of mean rewards: {sum(all_mean_rewards) / len(all_mean_rewards):.4f}")
    print(f"  Mean of best rewards: {sum(all_best_rewards) / len(all_best_rewards):.4f}")

    if verbose:
        print("\n" + "-" * 80)
        print("DETAILED ERRORS (first 10 entries with errors):")
        print("-" * 80)

        error_count = 0
        for entry in entry_stats:
            if error_count >= 10:
                break

            has_errors = False
            for sample in entry.samples:
                if sample.parse_error or (sample.reward_result and sample.reward_result.errors):
                    has_errors = True
                    break

            if has_errors:
                error_count += 1
                print(f"\nDOI: {entry.doi}")
                print(f"Prompt type: {entry.prompt_type}")
                print(f"Mean reward: {entry.mean_reward:.4f}")

                for sample in entry.samples:
                    if sample.parse_error:
                        print(f"  Sample {sample.sample_idx}: PARSE ERROR - {sample.parse_error}")
                    elif sample.reward_result and sample.reward_result.errors:
                        print(f"  Sample {sample.sample_idx}: Reward {sample.reward_result.total_reward:.4f}")
                        print(format_errors(sample.reward_result.errors, indent=4))


def save_results(
    entry_stats: list[EntryStats],
    aggregate_stats: dict,
    output_path: str,
):
    """Save results to JSON file."""
    results = {
        "aggregate_stats": aggregate_stats,
        "entries": []
    }

    for entry in entry_stats:
        entry_data = {
            "doi": entry.doi,
            "prompt_type": entry.prompt_type,
            "rewards": entry.rewards,
            "best_reward": entry.best_reward,
            "mean_reward": entry.mean_reward,
            "parse_success_rate": entry.parse_success_rate,
            "samples": []
        }

        for sample in entry.samples:
            sample_data = {
                "sample_idx": sample.sample_idx,
                "response": sample.response,
                "parsed_json": sample.parsed_json,
                "parse_error": sample.parse_error,
            }
            if sample.reward_result:
                sample_data["reward"] = {
                    "total": sample.reward_result.total_reward,
                    "funder": sample.reward_result.funder_reward,
                    "award_id": sample.reward_result.award_id_reward,
                    "scheme": sample.reward_result.scheme_reward,
                    "title": sample.reward_result.title_reward,
                    "matched_funders": sample.reward_result.matched_funders,
                    "total_predicted": sample.reward_result.total_predicted_funders,
                    "total_expected": sample.reward_result.total_expected_funders,
                    "errors": [
                        {
                            "type": e.error_type,
                            "message": e.message,
                            "predicted": e.predicted,
                            "expected": e.expected,
                        }
                        for e in sample.reward_result.errors
                    ]
                }
            entry_data["samples"].append(sample_data)

        results["entries"].append(entry_data)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run funding extraction inference with vLLM"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/train.jsonl",
        help="Path to input JSONL dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/inference_results.json",
        help="Path to save results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of samples per prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=32768,
        help="Maximum context length for filtering prompts",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction",
    )
    parser.add_argument(
        "--funding-statement-only",
        action="store_true",
        help="Only run on funding statements (not full markdown)",
    )
    parser.add_argument(
        "--full-markdown-only",
        action="store_true",
        help="Only run on full markdown (not funding statements)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed error information",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of dataset entries to process",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    data = load_dataset(args.data_path)
    print(f"Loaded {len(data)} entries")

    if args.limit:
        data = data[:args.limit]
        print(f"Limited to {len(data)} entries")

    # Load tokenizer
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Create prompts
    include_funding = not args.full_markdown_only
    include_markdown = not args.funding_statement_only

    print(f"Creating prompts (funding_statement={include_funding}, full_markdown={include_markdown})...")
    metadata, prompts = create_prompts(
        data,
        tokenizer=tokenizer,
        max_context_length=args.max_context_length,
        include_funding_statement=include_funding,
        include_full_markdown=include_markdown,
    )
    print(f"Created {len(prompts)} prompts")

    if not prompts:
        print("No prompts to process. Check your data file.")
        sys.exit(1)

    # Run inference
    responses = run_inference(
        model_name=args.model,
        prompts=prompts,
        k=args.k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Process results
    print("\nCalculating rewards...")
    entry_stats, aggregate_stats = process_results(metadata, responses, verbose=args.verbose)

    # Print statistics
    print_statistics(entry_stats, aggregate_stats, verbose=args.verbose)

    # Save results
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(entry_stats, aggregate_stats, args.output_path)


if __name__ == "__main__":
    main()
