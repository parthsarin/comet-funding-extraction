"""
Analyze inference results broken down by prompt type.

Usage:
    python analyze_results.py [results_file]

If no file is specified, defaults to results/inference_results_fixed.json
"""

import argparse
import json
from pathlib import Path


def avg(lst):
    return sum(lst) / len(lst) if lst else 0


def analyze_results(results_path: str):
    with open(results_path) as f:
        data = json.load(f)

    # Separate by prompt type
    by_type = {
        "funding_statement": {"funder": [], "award_id": [], "scheme": [], "title": [], "total": []},
        "full_markdown": {"funder": [], "award_id": [], "scheme": [], "title": [], "total": []},
        "full_markdown_truncated": {"funder": [], "award_id": [], "scheme": [], "title": [], "total": []},
    }

    for e in data['entries']:
        ptype = e['prompt_type']
        for s in e['samples']:
            if 'reward' in s:
                by_type[ptype]["funder"].append(s['reward']['funder'])
                by_type[ptype]["award_id"].append(s['reward']['award_id'])
                by_type[ptype]["scheme"].append(s['reward']['scheme'])
                by_type[ptype]["title"].append(s['reward']['title'])
                by_type[ptype]["total"].append(s['reward']['total'])

    print("=" * 75)
    print("COMPONENT SCORES BY PROMPT TYPE")
    print("=" * 75)
    print(f"{'Component':<15} {'Funding Stmt':>15} {'Full Markdown':>15} {'Truncated':>15}")
    print("-" * 75)

    for comp in ["total", "funder", "award_id", "scheme", "title"]:
        fs = avg(by_type["funding_statement"][comp])
        fm = avg(by_type["full_markdown"][comp])
        tr = avg(by_type["full_markdown_truncated"][comp])
        label = comp.replace("_", " ").title()
        print(f"{label:<15} {fs:>15.4f} {fm:>15.4f} {tr:>15.4f}")

    print("-" * 75)
    print(f"{'Sample count':<15} {len(by_type['funding_statement']['total']):>15} {len(by_type['full_markdown']['total']):>15} {len(by_type['full_markdown_truncated']['total']):>15}")

    # Reward distribution by type
    print("\n" + "=" * 75)
    print("REWARD DISTRIBUTION BY PROMPT TYPE")
    print("=" * 75)

    buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]

    for ptype in ["funding_statement", "full_markdown", "full_markdown_truncated"]:
        rewards = by_type[ptype]["total"]
        if not rewards:
            continue
        print(f"\n{ptype.upper().replace('_', ' ')} (n={len(rewards)}):")
        for low, high in buckets:
            count = sum(1 for r in rewards if low <= r < high)
            pct = count / len(rewards) * 100
            bar = "█" * int(pct / 2)
            print(f"  {low:.1f}-{high:.1f}: {count:5d} ({pct:5.1f}%) {bar}")

    # High performers
    print("\n" + "=" * 75)
    print("HIGH PERFORMERS (reward >= 0.8)")
    print("=" * 75)
    for ptype in ["funding_statement", "full_markdown", "full_markdown_truncated"]:
        rewards = by_type[ptype]["total"]
        if not rewards:
            continue
        high = sum(1 for r in rewards if r >= 0.8)
        print(f"{ptype}: {high}/{len(rewards)} ({high/len(rewards)*100:.1f}%)")

    # Overall summary
    print("\n" + "=" * 75)
    print("OVERALL SUMMARY")
    print("=" * 75)

    all_rewards = []
    for ptype in by_type:
        all_rewards.extend(by_type[ptype]["total"])

    print(f"Total samples: {len(all_rewards)}")
    print(f"Mean reward: {avg(all_rewards):.4f}")
    print(f"Max reward: {max(all_rewards):.4f}")
    print(f"Min reward: {min(all_rewards):.4f}")

    perfect = sum(1 for r in all_rewards if r == 1.0)
    print(f"Perfect scores: {perfect} ({perfect/len(all_rewards)*100:.1f}%)")

    # Parse failure info from aggregate stats
    if 'aggregate_stats' in data:
        print("\n" + "=" * 75)
        print("PARSE FAILURE RATES")
        print("=" * 75)
        for ptype, stats in data['aggregate_stats'].items():
            if 'parse_failure_rate' in stats:
                print(f"{ptype}: {stats['parse_failure_rate']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Analyze inference results by prompt type")
    parser.add_argument(
        "results_file",
        nargs="?",
        default="results/inference_results_fixed.json",
        help="Path to results JSON file",
    )
    args = parser.parse_args()

    if not Path(args.results_file).exists():
        print(f"Error: File not found: {args.results_file}")
        return 1

    analyze_results(args.results_file)
    return 0


if __name__ == "__main__":
    exit(main())
