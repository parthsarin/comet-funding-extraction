"""
Retry failed requests from a Claude results file.

Usage:
    python retry_failed.py results/claude/claude_full_markdown.json
"""

import argparse
import asyncio
import json
from pathlib import Path

import anthropic
from tqdm import tqdm

from prompt import SYSTEM_PROMPT_ON_FUNDING_STATEMENT, SYSTEM_PROMPT_ON_ENTIRE_ARTICLE
from reward import calculate_reward, extract_json_from_response


def load_dataset(path: str) -> list[dict]:
    """Load dataset from JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


async def retry_request(
    client: anthropic.AsyncAnthropic,
    result: dict,
    original_data: dict,
    model: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    stats: dict,
) -> dict:
    """Retry a single failed request."""
    async with semaphore:
        try:
            # Reconstruct the prompt
            prompt_type = result["prompt_type"]
            if prompt_type == "funding_statement":
                system = SYSTEM_PROMPT_ON_FUNDING_STATEMENT
                user_msg = f"Please extract funding information from the following statement:\n\n{original_data['funding_statement']}"
            else:
                system = SYSTEM_PROMPT_ON_ENTIRE_ARTICLE
                markdown = original_data.get("markdown", "")
                if prompt_type == "full_markdown_truncated":
                    markdown = markdown[:99000]
                    user_msg = f"Please extract funding information from the following article (truncated due to length):\n\n{markdown}"
                else:
                    user_msg = f"Please extract funding information from the following article:\n\n{markdown}"

            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )

            if not response.content:
                raise ValueError(f"Empty response content, stop_reason={response.stop_reason}")

            response_text = response.content[0].text

            # Parse and calculate reward
            json_str = extract_json_from_response(response_text)
            parsed_json = None
            parse_error = None
            reward_result = None

            if json_str:
                try:
                    parsed_json = json.loads(json_str)
                    reward_result = calculate_reward(parsed_json, result["expected"], verbose=False)
                    stats["success"] += 1
                    stats["total_reward"] += reward_result.total_reward
                except json.JSONDecodeError as e:
                    parse_error = f"JSON decode error: {e}"
                except Exception as e:
                    parse_error = f"Reward calculation error: {e}"
            else:
                parse_error = "No JSON array found in response"

            new_result = {
                "doi": result["doi"],
                "prompt_type": result["prompt_type"],
                "response": response_text,
                "parsed_json": parsed_json,
                "parse_error": parse_error,
                "expected": result["expected"],
            }

            if reward_result:
                new_result["reward"] = {
                    "total": reward_result.total_reward,
                    "funder": reward_result.funder_reward,
                    "award_id": reward_result.award_id_reward,
                    "scheme": reward_result.scheme_reward,
                    "title": reward_result.title_reward,
                }
            else:
                new_result["reward"] = {"total": 0.0}

            pbar.update(1)
            avg_reward = stats["total_reward"] / stats["success"] if stats["success"] > 0 else 0
            pbar.set_postfix({"success": stats["success"], "avg_reward": f"{avg_reward:.3f}"})
            return new_result

        except Exception as e:
            stats["failed"] += 1
            pbar.update(1)
            pbar.set_postfix({"success": stats["success"], "failed": stats["failed"]})
            # Return original result with updated error
            result["parse_error"] = f"Retry failed: {e}"
            return result


async def main_async(args):
    # Load results file
    print(f"Loading results from: {args.results_file}")
    with open(args.results_file) as f:
        results = json.load(f)

    # Load original dataset to reconstruct prompts
    print(f"Loading dataset from: {args.data_path}")
    dataset = load_dataset(args.data_path)
    doi_to_data = {d["doi"]: d for d in dataset}

    # Find failed requests
    failed = []
    successful = []
    for r in results:
        if r.get("parse_error") and "API error" in r.get("parse_error", ""):
            if r["doi"] in doi_to_data:
                failed.append((r, doi_to_data[r["doi"]]))
        else:
            successful.append(r)

    print(f"Found {len(failed)} failed requests to retry")
    print(f"Keeping {len(successful)} successful requests")

    if not failed:
        print("No failed requests to retry!")
        return

    # Retry failed requests
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(args.max_concurrent)
    stats = {"success": 0, "failed": 0, "total_reward": 0.0}

    retried_results = []
    with tqdm(total=len(failed), desc="Retrying") as pbar:
        tasks = [
            retry_request(client, r, data, args.model, args.max_tokens, semaphore, pbar, stats)
            for r, data in failed
        ]
        retried_results = await asyncio.gather(*tasks)

    # Combine results
    all_results = successful + retried_results

    # Save
    output_path = args.results_file.replace(".json", "_retried.json")
    if args.output:
        output_path = args.output

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print stats
    new_success = sum(1 for r in all_results if r.get("parsed_json") is not None)
    rewards = [r.get("reward", {}).get("total", 0) for r in all_results]
    print(f"\nResults:")
    print(f"  Total: {len(all_results)}")
    print(f"  Parse success: {new_success} ({new_success/len(all_results):.1%})")
    print(f"  Mean reward: {sum(rewards)/len(rewards):.4f}")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Retry failed Claude API requests")
    parser.add_argument("results_file", help="Path to results JSON file with failures")
    parser.add_argument("--data-path", default="data/train.jsonl", help="Path to original dataset")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Claude model")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent requests")
    parser.add_argument("--output", "-o", help="Output path (default: adds _retried suffix)")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
