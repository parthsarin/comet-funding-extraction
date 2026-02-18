"""
Sampling script for funding statement extraction using Claude API.

Runs inference with Claude claude-sonnet-4-6 on funding statements and full article markdown,
calculates rewards, and reports statistics.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from tqdm import tqdm

from prompt import SYSTEM_PROMPT_ON_FUNDING_STATEMENT, SYSTEM_PROMPT_ON_ENTIRE_ARTICLE
from reward import (
    calculate_reward,
    extract_json_from_response,
)


@dataclass
class RunningStats:
    """Track running statistics for tqdm display."""
    total_requests: int = 0
    completed: int = 0
    parse_successes: int = 0
    total_reward: float = 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.completed if self.completed > 0 else 0.0

    @property
    def parse_rate(self) -> float:
        return self.parse_successes / self.completed if self.completed > 0 else 0.0


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
    max_chars: int = 100000,
    include_funding_statement: bool = True,
    include_full_markdown: bool = True,
) -> list[dict]:
    """
    Create prompts for all entries in the dataset.

    Returns:
        List of prompt metadata dicts with 'messages', 'system', 'doi', 'prompt_type', 'expected'
    """
    prompts = []
    skipped_funding = 0
    skipped_markdown = 0

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
            user_msg = f"Please extract funding information from the following statement:\n\n{funding_statement}"
            if len(user_msg) < max_chars:
                prompts.append({
                    "doi": doi,
                    "prompt_type": "funding_statement",
                    "expected": funders,
                    "system": SYSTEM_PROMPT_ON_FUNDING_STATEMENT,
                    "user_message": user_msg,
                })
            else:
                skipped_funding += 1

        # Prompt type 2: Full markdown
        if include_full_markdown and markdown:
            user_msg = f"Please extract funding information from the following article:\n\n{markdown}"
            if len(user_msg) < max_chars:
                prompts.append({
                    "doi": doi,
                    "prompt_type": "full_markdown",
                    "expected": funders,
                    "system": SYSTEM_PROMPT_ON_ENTIRE_ARTICLE,
                    "user_message": user_msg,
                })
            else:
                # Try truncating
                truncated = markdown[:max_chars - 1000]
                user_msg = f"Please extract funding information from the following article (truncated due to length):\n\n{truncated}"
                prompts.append({
                    "doi": doi,
                    "prompt_type": "full_markdown_truncated",
                    "expected": funders,
                    "system": SYSTEM_PROMPT_ON_ENTIRE_ARTICLE,
                    "user_message": user_msg,
                })

    print(f"Prompt stats: {len(prompts)} created, {skipped_funding} funding skipped, {skipped_markdown} markdown skipped")
    return prompts


async def call_claude(
    client: anthropic.AsyncAnthropic,
    prompt: dict,
    model: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    stats: RunningStats,
    pbar: tqdm,
    results: list,
    results_lock: asyncio.Lock,
) -> None:
    """Make a single Claude API call and update stats."""
    async with semaphore:
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=prompt["system"],
                messages=[{"role": "user", "content": prompt["user_message"]}],
            )

            # Handle empty response content
            if not response.content:
                raise ValueError(f"Empty response content, stop_reason={response.stop_reason}")

            response_text = response.content[0].text

            # Parse and calculate reward
            json_str = extract_json_from_response(response_text)
            parsed_json = None
            reward_result = None
            parse_error = None

            if json_str:
                try:
                    parsed_json = json.loads(json_str)
                    reward_result = calculate_reward(parsed_json, prompt["expected"], verbose=False)
                    stats.parse_successes += 1
                except json.JSONDecodeError as e:
                    parse_error = f"JSON decode error: {e}"
                except Exception as e:
                    parse_error = f"Reward calculation error: {e}"
            else:
                parse_error = "No JSON array found in response"

            # Build result
            result = {
                "doi": prompt["doi"],
                "prompt_type": prompt["prompt_type"],
                "response": response_text,
                "parsed_json": parsed_json,
                "parse_error": parse_error,
                "expected": prompt["expected"],
            }

            if reward_result:
                result["reward"] = {
                    "total": reward_result.total_reward,
                    "funder": reward_result.funder_reward,
                    "award_id": reward_result.award_id_reward,
                    "scheme": reward_result.scheme_reward,
                    "title": reward_result.title_reward,
                }
                stats.total_reward += reward_result.total_reward
            else:
                result["reward"] = {"total": 0.0}

            async with results_lock:
                results.append(result)

        except (anthropic.APIError, ValueError) as e:
            async with results_lock:
                results.append({
                    "doi": prompt["doi"],
                    "prompt_type": prompt["prompt_type"],
                    "response": None,
                    "parse_error": f"API error: {e}",
                    "expected": prompt["expected"],
                    "reward": {"total": 0.0},
                })

        stats.completed += 1
        pbar.set_postfix({
            "done": stats.completed,
            "avg_reward": f"{stats.avg_reward:.3f}",
            "parse_rate": f"{stats.parse_rate:.1%}",
        })
        pbar.update(1)


async def run_inference_async(
    prompts: list[dict],
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    max_concurrent: int = 20,
    output_path: str = "results/claude_results.json",
    desc: str = "Claude API calls",
) -> list[dict]:
    """Run inference with Claude API in parallel."""

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)
    stats = RunningStats(total_requests=len(prompts))
    results = []
    results_lock = asyncio.Lock()

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with tqdm(total=len(prompts), desc=desc) as pbar:
        tasks = [
            call_claude(
                client, prompt, model, max_tokens, temperature,
                semaphore, stats, pbar, results, results_lock
            )
            for prompt in prompts
        ]
        await asyncio.gather(*tasks)

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


async def run_all_inference(
    prompts: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    max_concurrent: int,
    output_dir: str,
) -> dict[str, list[dict]]:
    """Run inference separately for each prompt type."""

    # Split prompts by type
    by_type = {}
    for p in prompts:
        ptype = p["prompt_type"]
        if ptype not in by_type:
            by_type[ptype] = []
        by_type[ptype].append(p)

    all_results = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ptype, type_prompts in by_type.items():
        print(f"\n{'='*60}")
        print(f"Running {ptype.upper().replace('_', ' ')} ({len(type_prompts)} prompts)")
        print(f"{'='*60}")

        output_path = output_dir / f"claude_{ptype}.json"

        results = await run_inference_async(
            prompts=type_prompts,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            max_concurrent=max_concurrent,
            output_path=str(output_path),
            desc=ptype.replace("_", " "),
        )

        all_results[ptype] = results

        # Print stats for this type
        rewards = [r.get("reward", {}).get("total", 0.0) for r in results]
        parse_success = sum(1 for r in results if r.get("parsed_json") is not None)
        print(f"  Mean reward: {sum(rewards)/len(rewards):.4f}")
        print(f"  Parse success: {parse_success}/{len(results)} ({parse_success/len(results):.1%})")
        print(f"  Saved to: {output_path}")

    return all_results


def print_statistics(results: list[dict]):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("INFERENCE RESULTS SUMMARY")
    print("=" * 70)

    by_type = {}
    for r in results:
        ptype = r["prompt_type"]
        if ptype not in by_type:
            by_type[ptype] = {"rewards": [], "parse_success": 0, "total": 0}

        by_type[ptype]["total"] += 1
        reward = r.get("reward", {}).get("total", 0.0)
        by_type[ptype]["rewards"].append(reward)

        if r.get("parsed_json") is not None:
            by_type[ptype]["parse_success"] += 1

    for ptype, stats in by_type.items():
        rewards = stats["rewards"]
        print(f"\n{ptype.upper().replace('_', ' ')}:")
        print(f"  Total samples: {stats['total']}")
        print(f"  Mean reward: {sum(rewards)/len(rewards):.4f}")
        print(f"  Max reward: {max(rewards):.4f}")
        print(f"  Parse success rate: {stats['parse_success']/stats['total']:.2%}")

    # Overall
    all_rewards = [r.get("reward", {}).get("total", 0.0) for r in results]
    print(f"\nOVERALL:")
    print(f"  Total samples: {len(results)}")
    print(f"  Mean reward: {sum(all_rewards)/len(all_rewards):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run funding extraction inference with Claude API"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/train.jsonl",
        help="Path to input JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/claude",
        help="Directory to save results (separate files per prompt type)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-6",
        help="Claude model name",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Maximum concurrent API requests",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=100000,
        help="Maximum characters per prompt",
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

    # Create prompts
    include_funding = not args.full_markdown_only
    include_markdown = not args.funding_statement_only

    print(f"Creating prompts (funding_statement={include_funding}, full_markdown={include_markdown})...")
    prompts = create_prompts(
        data,
        max_chars=args.max_chars,
        include_funding_statement=include_funding,
        include_full_markdown=include_markdown,
    )
    print(f"Created {len(prompts)} prompts")

    if not prompts:
        print("No prompts to process. Check your data file.")
        sys.exit(1)

    # Run inference
    print(f"\nRunning inference with {args.model}...")
    print(f"Max concurrent requests: {args.max_concurrent}")
    print(f"Output directory: {args.output_dir}")

    all_results = asyncio.run(run_all_inference(
        prompts=prompts,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
        output_dir=args.output_dir,
    ))

    # Print overall statistics
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    for ptype, results in all_results.items():
        print_statistics(results)
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
