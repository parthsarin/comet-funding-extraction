"""
Reward calculation for funding statement extraction.

Compares predicted JSON output against ground truth labels using:
- Fuzzy matching for funder names
- IoU (Intersection over Union) for award IDs
- Fuzzy matching for funding schemes and award titles
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz


@dataclass
class MatchError:
    """Represents a specific matching error for verbose output."""
    error_type: str
    message: str
    predicted: Any = None
    expected: Any = None


@dataclass
class RewardResult:
    """Result of reward calculation for a single prediction."""
    total_reward: float
    funder_reward: float
    award_id_reward: float
    scheme_reward: float
    title_reward: float
    errors: list[MatchError] = field(default_factory=list)
    matched_funders: int = 0
    total_predicted_funders: int = 0
    total_expected_funders: int = 0


def normalize_string(s: str) -> str:
    """Normalize a string for comparison."""
    if not s:
        return ""
    # Lowercase, remove extra whitespace, strip
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def normalize_award_id(award_id: str) -> str:
    """Normalize award ID for comparison (remove common formatting differences)."""
    if not award_id:
        return ""
    # Remove spaces, dashes variations, lowercase
    s = award_id.upper().strip()
    # Keep alphanumeric and basic punctuation
    s = re.sub(r'\s+', '', s)
    return s


def fuzzy_match_score(s1: str, s2: str, threshold: float = 70.0) -> float:
    """
    Calculate fuzzy match score between two strings.
    Returns a score between 0 and 1 if above threshold, else 0.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    n1 = normalize_string(s1)
    n2 = normalize_string(s2)

    # Use token_set_ratio for better handling of word order and abbreviations
    score = fuzz.token_set_ratio(n1, n2)

    if score >= threshold:
        return score / 100.0
    return 0.0


def calculate_set_iou(predicted: list[str], expected: list[str], normalize_fn=None) -> float:
    """
    Calculate Intersection over Union for two lists of strings.

    Args:
        predicted: List of predicted values
        expected: List of expected values
        normalize_fn: Optional function to normalize strings before comparison

    Returns:
        IoU score between 0 and 1
    """
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0

    if normalize_fn:
        pred_set = set(normalize_fn(x) for x in predicted if x)
        exp_set = set(normalize_fn(x) for x in expected if x)
    else:
        pred_set = set(x for x in predicted if x)
        exp_set = set(x for x in expected if x)

    if not pred_set and not exp_set:
        return 1.0
    if not pred_set or not exp_set:
        return 0.0

    intersection = len(pred_set & exp_set)
    union = len(pred_set | exp_set)

    return intersection / union if union > 0 else 0.0


def fuzzy_set_match(predicted: list[str], expected: list[str], threshold: float = 80.0) -> tuple[float, list[tuple[str, str]]]:
    """
    Match predicted items to expected items using fuzzy matching.
    Returns average match score and list of matched pairs.
    """
    if not predicted and not expected:
        return 1.0, []
    if not predicted or not expected:
        return 0.0, []

    matches = []
    used_expected = set()
    total_score = 0.0

    for pred in predicted:
        best_match = None
        best_score = 0.0

        for i, exp in enumerate(expected):
            if i in used_expected:
                continue
            score = fuzz.token_set_ratio(normalize_string(pred), normalize_string(exp))
            if score > best_score and score >= threshold:
                best_score = score
                best_match = (i, exp)

        if best_match:
            used_expected.add(best_match[0])
            matches.append((pred, best_match[1]))
            total_score += best_score / 100.0

    # Penalize for unmatched items
    total_items = max(len(predicted), len(expected))
    return total_score / total_items, matches


def match_funder_to_expected(
    pred_funder: dict,
    expected_funders: list[dict],
    used_indices: set,
    threshold: float = 70.0
) -> tuple[int | None, float]:
    """
    Find the best matching expected funder for a predicted funder.

    Returns:
        Tuple of (matched index or None, match score)
    """
    pred_name = pred_funder.get("funder_name", "") or ""
    best_idx = None
    best_score = 0.0

    for i, exp_funder in enumerate(expected_funders):
        if i in used_indices:
            continue

        exp_name = exp_funder.get("funder_name", "") or ""
        score = fuzzy_match_score(pred_name, exp_name, threshold)

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, best_score


def calculate_award_reward(
    pred_awards: list[dict],
    exp_awards: list[dict],
    verbose: bool = False
) -> tuple[float, float, float, list[MatchError]]:
    """
    Calculate reward for awards matching.

    Returns:
        Tuple of (award_id_reward, scheme_reward, title_reward, errors)
    """
    errors = []

    # Flatten all award IDs, schemes, and titles from both sides
    pred_ids = []
    pred_schemes = []
    pred_titles = []

    exp_ids = []
    exp_schemes = []
    exp_titles = []

    for award in (pred_awards or []):
        pred_ids.extend(award.get("award_ids", []) or [])
        pred_schemes.extend(award.get("funding_scheme", []) or [])
        pred_titles.extend(award.get("award_title", []) or [])

    for award in (exp_awards or []):
        exp_ids.extend(award.get("award_ids", []) or [])
        exp_schemes.extend(award.get("funding_scheme", []) or [])
        exp_titles.extend(award.get("award_title", []) or [])

    # Calculate IoU for award IDs (exact match after normalization)
    award_id_iou = calculate_set_iou(pred_ids, exp_ids, normalize_award_id)

    if verbose and award_id_iou < 1.0:
        pred_normalized = set(normalize_award_id(x) for x in pred_ids if x)
        exp_normalized = set(normalize_award_id(x) for x in exp_ids if x)
        missing = exp_normalized - pred_normalized
        extra = pred_normalized - exp_normalized

        if missing:
            errors.append(MatchError(
                error_type="missing_award_ids",
                message=f"Missing award IDs: {missing}",
                expected=list(missing)
            ))
        if extra:
            errors.append(MatchError(
                error_type="extra_award_ids",
                message=f"Extra award IDs: {extra}",
                predicted=list(extra)
            ))

    # Fuzzy match for schemes
    scheme_score, _ = fuzzy_set_match(pred_schemes, exp_schemes)

    if verbose and scheme_score < 1.0:
        errors.append(MatchError(
            error_type="scheme_mismatch",
            message=f"Scheme mismatch",
            predicted=pred_schemes,
            expected=exp_schemes
        ))

    # Fuzzy match for titles
    title_score, _ = fuzzy_set_match(pred_titles, exp_titles)

    if verbose and title_score < 1.0 and (pred_titles or exp_titles):
        errors.append(MatchError(
            error_type="title_mismatch",
            message=f"Title mismatch",
            predicted=pred_titles,
            expected=exp_titles
        ))

    return award_id_iou, scheme_score, title_score, errors


def is_flat_schema(data: list[dict]) -> bool:
    """
    Detect if the data uses flat schema (award objects directly) vs nested schema (funder objects with awards).

    Flat schema: [{"funding_scheme": [...], "award_ids": [...], "award_title": [...]}]
    Nested schema: [{"funder_name": "...", "awards": [...]}]
    """
    if not data:
        return False

    first = data[0]
    if not isinstance(first, dict):
        return False

    # Flat schema has award fields directly, no "awards" key
    has_award_fields = any(k in first for k in ["funding_scheme", "award_ids", "award_title"])
    has_awards_key = "awards" in first

    return has_award_fields and not has_awards_key


def normalize_to_nested_schema(data: list[dict]) -> list[dict]:
    """
    Convert flat schema to nested schema.

    Flat schema items become awards under a funder with null name,
    unless they have a "funder_name" field.
    """
    if not data or not is_flat_schema(data):
        return data

    result = []
    for item in data:
        # Check if this flat item has a funder_name field
        funder_name = item.get("funder_name")

        # Build the award object from flat fields
        award = {
            "funding_scheme": item.get("funding_scheme", []),
            "award_ids": item.get("award_ids", []),
            "award_title": item.get("award_title", []),
        }

        # Wrap in nested structure
        result.append({
            "funder_name": funder_name,
            "awards": [award]
        })

    return result


def extract_all_awards(funders: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """Extract all award IDs, schemes, and titles from a list of funder objects."""
    all_ids = []
    all_schemes = []
    all_titles = []

    for funder in funders:
        for award in funder.get("awards", []) or []:
            all_ids.extend(award.get("award_ids", []) or [])
            all_schemes.extend(award.get("funding_scheme", []) or [])
            all_titles.extend(award.get("award_title", []) or [])

    return all_ids, all_schemes, all_titles


def calculate_reward(
    predicted: list[dict] | str,
    expected: list[dict] | str,
    verbose: bool = False,
    funder_weight: float = 0.3,
    award_id_weight: float = 0.4,
    scheme_weight: float = 0.2,
    title_weight: float = 0.1,
    funder_threshold: float = 70.0,
    use_global_matching: bool = True,
) -> RewardResult:
    """
    Calculate reward for a prediction against ground truth.

    Args:
        predicted: Predicted JSON (list of funder objects) or JSON string
        expected: Expected JSON (list of funder objects) or JSON string
        verbose: If True, collect detailed error information
        funder_weight: Weight for funder name matching (default 0.3)
        award_id_weight: Weight for award ID IoU (default 0.4)
        scheme_weight: Weight for funding scheme matching (default 0.2)
        title_weight: Weight for award title matching (default 0.1)
        funder_threshold: Fuzzy match threshold for funder names (default 70.0)

    Returns:
        RewardResult with scores and optional error details
    """
    errors = []

    # Parse JSON strings if needed
    if isinstance(predicted, str):
        try:
            predicted = json.loads(predicted)
        except json.JSONDecodeError as e:
            errors.append(MatchError(
                error_type="json_parse_error",
                message=f"Failed to parse predicted JSON: {e}",
                predicted=predicted[:200] if len(predicted) > 200 else predicted
            ))
            return RewardResult(
                total_reward=0.0,
                funder_reward=0.0,
                award_id_reward=0.0,
                scheme_reward=0.0,
                title_reward=0.0,
                errors=errors,
                total_predicted_funders=0,
                total_expected_funders=len(expected) if isinstance(expected, list) else 0
            )

    if isinstance(expected, str):
        expected = json.loads(expected)

    # Handle None or empty cases
    if not predicted:
        predicted = []
    if not expected:
        expected = []

    # Ensure we have lists
    if not isinstance(predicted, list):
        errors.append(MatchError(
            error_type="invalid_format",
            message="Predicted value is not a list",
            predicted=type(predicted).__name__
        ))
        return RewardResult(
            total_reward=0.0,
            funder_reward=0.0,
            award_id_reward=0.0,
            scheme_reward=0.0,
            title_reward=0.0,
            errors=errors,
            total_predicted_funders=0,
            total_expected_funders=len(expected)
        )

    # Normalize flat schema to nested schema if needed
    predicted = normalize_to_nested_schema(predicted)

    total_predicted = len(predicted)
    total_expected = len(expected)

    # Special case: both empty
    if not predicted and not expected:
        return RewardResult(
            total_reward=1.0,
            funder_reward=1.0,
            award_id_reward=1.0,
            scheme_reward=1.0,
            title_reward=1.0,
            errors=[],
            matched_funders=0,
            total_predicted_funders=0,
            total_expected_funders=0
        )

    # Special case: one empty, one not
    if not predicted or not expected:
        if verbose:
            if not predicted:
                errors.append(MatchError(
                    error_type="no_predictions",
                    message=f"No funders predicted, expected {total_expected}",
                    expected=[f.get("funder_name") for f in expected]
                ))
            else:
                errors.append(MatchError(
                    error_type="no_expected",
                    message=f"Predicted {total_predicted} funders but expected none",
                    predicted=[f.get("funder_name") for f in predicted]
                ))
        return RewardResult(
            total_reward=0.0,
            funder_reward=0.0,
            award_id_reward=0.0,
            scheme_reward=0.0,
            title_reward=0.0,
            errors=errors,
            matched_funders=0,
            total_predicted_funders=total_predicted,
            total_expected_funders=total_expected
        )

    # Match predicted funders to expected funders
    used_expected = set()
    funder_scores = []
    award_id_scores = []
    scheme_scores = []
    title_scores = []
    matched_count = 0

    for pred_funder in predicted:
        # Handle malformed predictions (e.g., strings instead of dicts)
        if not isinstance(pred_funder, dict):
            funder_scores.append(0.0)
            award_id_scores.append(0.0)
            scheme_scores.append(0.0)
            title_scores.append(0.0)
            if verbose:
                errors.append(MatchError(
                    error_type="invalid_funder_format",
                    message=f"Funder is not a dict",
                    predicted=str(pred_funder)[:100]
                ))
            continue

        match_idx, funder_score = match_funder_to_expected(
            pred_funder, expected, used_expected, funder_threshold
        )

        if match_idx is not None:
            used_expected.add(match_idx)
            matched_count += 1
            funder_scores.append(funder_score)

            # Calculate award-level scores
            exp_funder = expected[match_idx]
            aid_score, sch_score, tit_score, award_errors = calculate_award_reward(
                pred_funder.get("awards", []),
                exp_funder.get("awards", []),
                verbose
            )

            award_id_scores.append(aid_score)
            scheme_scores.append(sch_score)
            title_scores.append(tit_score)
            errors.extend(award_errors)
        else:
            # Unmatched prediction
            funder_scores.append(0.0)
            award_id_scores.append(0.0)
            scheme_scores.append(0.0)
            title_scores.append(0.0)

            if verbose:
                errors.append(MatchError(
                    error_type="unmatched_prediction",
                    message=f"Predicted funder not found in expected",
                    predicted=pred_funder.get("funder_name")
                ))

    # Account for unmatched expected funders
    unmatched_expected = set(range(len(expected))) - used_expected
    for idx in unmatched_expected:
        funder_scores.append(0.0)
        award_id_scores.append(0.0)
        scheme_scores.append(0.0)
        title_scores.append(0.0)

        if verbose:
            errors.append(MatchError(
                error_type="missed_funder",
                message=f"Expected funder not predicted",
                expected=expected[idx].get("funder_name")
            ))

    # Calculate average scores from per-funder matching
    n_items = max(total_predicted, total_expected)
    avg_funder = sum(funder_scores) / n_items if funder_scores else 0.0
    avg_award_id = sum(award_id_scores) / n_items if award_id_scores else 0.0
    avg_scheme = sum(scheme_scores) / n_items if scheme_scores else 0.0
    avg_title = sum(title_scores) / n_items if title_scores else 0.0

    # Global matching: compare all awards regardless of funder grouping
    # This helps when model splits one funder's awards into multiple entries
    if use_global_matching:
        pred_ids, pred_schemes, pred_titles = extract_all_awards(predicted)
        exp_ids, exp_schemes, exp_titles = extract_all_awards(expected)

        global_award_id = calculate_set_iou(pred_ids, exp_ids, normalize_award_id)
        global_scheme, _ = fuzzy_set_match(pred_schemes, exp_schemes)
        global_title, _ = fuzzy_set_match(pred_titles, exp_titles)

        # Use the better of per-funder or global matching for award components
        avg_award_id = max(avg_award_id, global_award_id)
        avg_scheme = max(avg_scheme, global_scheme)
        avg_title = max(avg_title, global_title)

    # Calculate weighted total
    total_reward = (
        funder_weight * avg_funder +
        award_id_weight * avg_award_id +
        scheme_weight * avg_scheme +
        title_weight * avg_title
    )

    return RewardResult(
        total_reward=total_reward,
        funder_reward=avg_funder,
        award_id_reward=avg_award_id,
        scheme_reward=avg_scheme,
        title_reward=avg_title,
        errors=errors,
        matched_funders=matched_count,
        total_predicted_funders=total_predicted,
        total_expected_funders=total_expected
    )


def extract_json_from_response(response: str) -> str | None:
    """
    Extract JSON array from a model response that may contain other text.

    Args:
        response: Raw model response string

    Returns:
        Extracted JSON string or None if not found
    """
    # Strip out <think>...</think> blocks first (used by some models for chain-of-thought)
    cleaned_response = re.sub(r'<think>[\s\S]*?</think>', '', response)

    # Try to find JSON array in the response
    # Look for ```json blocks first
    json_block_match = re.search(r'```(?:json)?\s*(\[[\s\S]*\])\s*```', cleaned_response)
    if json_block_match:
        try:
            json.loads(json_block_match.group(1))
            return json_block_match.group(1)
        except json.JSONDecodeError:
            pass

    # Find all opening bracket positions and try to parse from each
    # Start from the last one (output is typically at the end)
    bracket_positions = [i for i, c in enumerate(cleaned_response) if c == '[']

    best_empty_array = None  # Track empty arrays as fallback

    for start_pos in reversed(bracket_positions):
        # Find the matching closing bracket by counting
        depth = 0
        in_string = False
        escape_next = False
        end_pos = None

        for i in range(start_pos, len(cleaned_response)):
            c = cleaned_response[i]

            if escape_next:
                escape_next = False
                continue

            if c == '\\' and in_string:
                escape_next = True
                continue

            if c == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    end_pos = i + 1
                    break

        if end_pos:
            candidate = cleaned_response[start_pos:end_pos]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    # Prefer non-empty arrays with dict elements (actual funder objects)
                    if len(parsed) > 0 and isinstance(parsed[0], dict):
                        return candidate
                    # Track empty arrays as fallback (for legitimate "no funders" case)
                    elif len(parsed) == 0 and best_empty_array is None:
                        best_empty_array = candidate
            except json.JSONDecodeError:
                continue

    # If we only found empty arrays, return one (legitimate "no funders" response)
    if best_empty_array is not None:
        return best_empty_array

    # Fallback: try greedy match on cleaned response
    json_match = re.search(r'\[[\s\S]*\]', cleaned_response)
    if json_match:
        try:
            json.loads(json_match.group(0))
            return json_match.group(0)
        except json.JSONDecodeError:
            pass

    return None


def format_errors(errors: list[MatchError], indent: int = 2) -> str:
    """Format error list for human-readable output."""
    if not errors:
        return "No errors"

    lines = []
    prefix = " " * indent
    for err in errors:
        lines.append(f"{prefix}- [{err.error_type}] {err.message}")
        if err.predicted is not None:
            lines.append(f"{prefix}  Predicted: {err.predicted}")
        if err.expected is not None:
            lines.append(f"{prefix}  Expected: {err.expected}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test with example data
    predicted = [
        {
            "funder_name": "National Science Foundation",
            "awards": [
                {
                    "funding_scheme": [],
                    "award_ids": ["DMS-1613091", "CCF-1714305"],
                    "award_title": []
                }
            ]
        }
    ]

    expected = [
        {
            "funder_name": "NSF",
            "awards": [
                {
                    "funding_scheme": [],
                    "award_ids": ["DMS-1613091", "CCF-1714305", "IIS-1741162"],
                    "award_title": []
                }
            ]
        },
        {
            "funder_name": "ONR",
            "awards": [
                {
                    "funding_scheme": [],
                    "award_ids": ["N00014-18-1-2729"],
                    "award_title": []
                }
            ]
        }
    ]

    result = calculate_reward(predicted, expected, verbose=True)
    print(f"Total Reward: {result.total_reward:.3f}")
    print(f"Funder Reward: {result.funder_reward:.3f}")
    print(f"Award ID Reward: {result.award_id_reward:.3f}")
    print(f"Scheme Reward: {result.scheme_reward:.3f}")
    print(f"Title Reward: {result.title_reward:.3f}")
    print(f"Matched: {result.matched_funders}/{result.total_expected_funders}")
    print("\nErrors:")
    print(format_errors(result.errors))
