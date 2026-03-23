"""
bias_correction.py
------------------
Learns per-player, per-stat prediction bias from resolved picks and applies
corrections to future predictions.

Persists to data/cache/bias_corrections.json.

Schema of the JSON store:
{
    "Jaylen Brown": {
        "pts": [
            {"error": 2.3, "date": "2026-03-10", "actual": 27.3, "predicted": 25.0},
            ...  (up to 20 entries, newest at end)
        ],
        ...
    },
    ...
}
"""

import json
import os
import math
from datetime import date
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DIR = os.path.dirname(__file__)
_CACHE_PATH = os.path.join(_DIR, "cache", "bias_corrections.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_ERRORS_KEPT = 20        # rolling window per (player, stat)
_MIN_DATA_POINTS = 3         # minimum picks before applying correction
_DECAY = 0.85                # exponential decay per game (older = less weight)
_MAX_CORRECTION = 8.0        # cap correction magnitude

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _load() -> dict:
    """Load bias data from JSON; return empty dict on missing/corrupt file."""
    if not os.path.exists(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    """Persist bias data to JSON."""
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def update_bias_from_picks(picks: Optional[list] = None) -> int:
    """
    Update the bias correction store from resolved picks.

    Parameters
    ----------
    picks : list of pick dicts (optional).
            If None, loads from underdog_tracker.get_picks(resolved_only=True).

    Returns
    -------
    int : number of pick errors processed (including previously processed ones
          that refresh the rolling window)
    """
    if picks is None:
        # Lazy import to avoid circular dependency
        from data.underdog_tracker import get_picks
        picks = get_picks(resolved_only=True)

    bias_data = _load()
    processed = 0

    for pick in picks:
        predicted = pick.get("predicted")
        actual = pick.get("actual")
        player = pick.get("player", "")
        stat = pick.get("stat", "")
        pick_date = pick.get("date", date.today().isoformat())

        # Only process picks where we have both values
        if predicted is None or actual is None or not player or not stat:
            continue

        error = float(actual) - float(predicted)

        entry = {
            "error": error,
            "date": pick_date,
            "actual": float(actual),
            "predicted": float(predicted),
        }

        # Initialise nested structure if needed
        if player not in bias_data:
            bias_data[player] = {}
        if stat not in bias_data[player]:
            bias_data[player][stat] = []

        error_list = bias_data[player][stat]

        # Avoid duplicate entries for the same pick (match by date + actual)
        already_stored = any(
            e.get("date") == pick_date
            and abs(e.get("actual", float("nan")) - float(actual)) < 1e-6
            and abs(e.get("predicted", float("nan")) - float(predicted)) < 1e-6
            for e in error_list
        )
        if already_stored:
            continue

        error_list.append(entry)

        # Keep only the most recent _MAX_ERRORS_KEPT entries (sorted by date)
        error_list.sort(key=lambda e: e.get("date", ""))
        if len(error_list) > _MAX_ERRORS_KEPT:
            bias_data[player][stat] = error_list[-_MAX_ERRORS_KEPT:]

        processed += 1

    _save(bias_data)
    return processed


def get_correction(player: str, stat: str) -> float:
    """
    Return the bias correction offset for a (player, stat) pair.

    The correction is an exponentially weighted average of recent errors.
    More recent errors receive higher weight (decay = 0.85 per game).

    Returns 0.0 if fewer than MIN_DATA_POINTS errors are stored.
    Caps the result at ±MAX_CORRECTION.
    """
    bias_data = _load()
    errors = bias_data.get(player, {}).get(stat, [])

    if len(errors) < _MIN_DATA_POINTS:
        return 0.0

    # Sort oldest-to-newest so we assign higher weights to later entries
    sorted_errors = sorted(errors, key=lambda e: e.get("date", ""))
    n = len(sorted_errors)

    # Weight[i] = decay^(n-1-i)  → most recent entry (i=n-1) gets weight 1.0
    weighted_sum = 0.0
    weight_total = 0.0
    for i, entry in enumerate(sorted_errors):
        w = _DECAY ** (n - 1 - i)
        weighted_sum += w * entry["error"]
        weight_total += w

    if weight_total == 0:
        return 0.0

    correction = weighted_sum / weight_total

    # Cap the correction
    correction = max(-_MAX_CORRECTION, min(_MAX_CORRECTION, correction))
    return round(correction, 4)


def get_all_corrections() -> dict:
    """
    Return corrections for all players with enough data.

    Returns
    -------
    dict : {player: {stat: correction_float}}
           Only includes (player, stat) pairs with >= MIN_DATA_POINTS errors.
    """
    bias_data = _load()
    result: dict = {}

    for player, stats in bias_data.items():
        player_corrections: dict = {}
        for stat, errors in stats.items():
            if len(errors) >= _MIN_DATA_POINTS:
                correction = get_correction(player, stat)
                player_corrections[stat] = correction
        if player_corrections:
            result[player] = player_corrections

    return result


def get_player_bias_summary(player: str) -> dict:
    """
    Return a detailed bias summary for a single player.

    Returns
    -------
    dict:
        {
            "player": str,
            "stats": {
                stat: {
                    "correction": float,
                    "n_picks": int,
                    "avg_error": float,
                    "trend": "up" | "down" | "flat" | "insufficient"
                },
                ...
            }
        }

    Trend is determined by comparing the weighted average of the 3 most recent
    errors to the weighted average of all errors:
      - "up"   → recent errors are more positive (model is under-predicting more)
      - "down" → recent errors are more negative (model is over-predicting more)
      - "flat" → within ±0.5 of overall correction
    """
    bias_data = _load()
    stats_data = bias_data.get(player, {})

    result_stats: dict = {}
    for stat, errors in stats_data.items():
        n = len(errors)
        if n == 0:
            continue

        sorted_errors = sorted(errors, key=lambda e: e.get("date", ""))
        avg_error = sum(e["error"] for e in sorted_errors) / n
        correction = get_correction(player, stat)

        # Trend: compare most recent 3 errors vs overall
        if n < _MIN_DATA_POINTS:
            trend = "insufficient"
        else:
            recent = sorted_errors[-3:]
            recent_avg = sum(e["error"] for e in recent) / len(recent)
            delta = recent_avg - correction
            if delta > 0.5:
                trend = "up"
            elif delta < -0.5:
                trend = "down"
            else:
                trend = "flat"

        result_stats[stat] = {
            "correction": correction,
            "n_picks": n,
            "avg_error": round(avg_error, 4),
            "trend": trend,
        }

    return {
        "player": player,
        "stats": result_stats,
    }


def clear_player_corrections(player: str) -> bool:
    """
    Remove all correction data for a player (e.g. after model retraining).

    Returns True if the player was found and removed, False otherwise.
    """
    bias_data = _load()
    if player in bias_data:
        del bias_data[player]
        _save(bias_data)
        return True
    return False
