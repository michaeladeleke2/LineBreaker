"""
linebreaker/data/accuracy_tracker.py

Prediction tracking and accuracy analytics system.
Stores predictions in data/cache/predictions.json.

Each prediction record:
{
    "id": str,              # uuid4
    "player_name": str,
    "player_id": int,
    "target": str,          # "pts", "reb", etc.
    "predicted": float,
    "custom_line": float,   # the line user set or sportsbook line
    "pick": str,            # "OVER" or "UNDER"
    "over_prob": float,     # probability as decimal (0-1)
    "opponent": str,
    "is_home": bool,
    "confidence": str,      # "High" / "Medium" / "Low"
    "bet_size": str,        # "1u" / "2u" / "3u"
    "game_date": str,       # ISO date string
    "resolved": bool,
    "correct": bool|None,
    "actual": float|None,
    "notes": str,
    "units_wagered": float,
    "units_won": float,
}

Units model (standard -110 odds):
  Wagered: 1u / 2u / 3u depending on bet_size
  Win:     wagered * 0.91  (e.g. 1u wagered → +0.91u)
  Loss:    -wagered         (e.g. 1u wagered → -1.0u)
"""

import uuid
import json
import math
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

PREDICTIONS_FILE = CACHE_DIR / "predictions.json"

# Standard -110 payout multiplier
WIN_MULTIPLIER = 0.91   # net units won per unit wagered

BET_SIZE_UNITS = {
    "1u": 1.0,
    "2u": 2.0,
    "3u": 3.0,
}


# ── File I/O ──────────────────────────────────────────────────────────────────

def _load_predictions() -> list:
    """Load all predictions from disk. Returns [] if file missing or corrupt."""
    if not PREDICTIONS_FILE.exists():
        return []
    try:
        with open(PREDICTIONS_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_predictions(predictions: list) -> None:
    """Persist predictions list to disk."""
    try:
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(predictions, f, indent=2, default=str)
    except OSError as e:
        print(f"  accuracy_tracker: failed to save predictions: {e}")


def _find_by_id(predictions: list, prediction_id: str) -> Optional[dict]:
    for p in predictions:
        if p.get("id") == prediction_id:
            return p
    return None


# ── Units helpers ─────────────────────────────────────────────────────────────

def _units_wagered(bet_size: str) -> float:
    return BET_SIZE_UNITS.get(bet_size, 1.0)


def _units_won(wagered: float, correct: Optional[bool]) -> float:
    """Return net units won/lost. None if not resolved."""
    if correct is None:
        return 0.0
    return round(wagered * WIN_MULTIPLIER, 3) if correct else -wagered


# ── Public API ────────────────────────────────────────────────────────────────

def log_prediction(
    player_name:  str,
    player_id:    int,
    target:       str,
    predicted:    float,
    custom_line:  float,
    over_prob:    float,
    opponent:     str,
    is_home:      bool,
    confidence:   str   = "Medium",
    bet_size:     str   = "1u",
    notes:        str   = "",
    game_date:    str   = None,
) -> dict:
    """
    Save a new prediction to the tracker.

    Parameters
    ----------
    player_name  : Display name of the player.
    player_id    : NBA API player ID.
    target       : Stat target key ("pts", "reb", etc.).
    predicted    : Model's predicted value.
    custom_line  : Sportsbook line (or user-set line) for OVER/UNDER.
    over_prob    : Model's over probability (0.0 – 1.0).
    opponent     : Opponent team abbreviation.
    is_home      : Whether the player's team is playing at home.
    confidence   : "High", "Medium", or "Low".
    bet_size     : "1u", "2u", or "3u".
    notes        : Optional handicapper notes.
    game_date    : ISO date string (defaults to today).

    Returns
    -------
    The saved prediction dict.
    """
    if game_date is None:
        game_date = date.today().isoformat()

    pick = "OVER" if over_prob >= 0.5 else "UNDER"
    wagered = _units_wagered(bet_size)

    record = {
        "id":           str(uuid.uuid4()),
        "player_name":  player_name,
        "player_id":    int(player_id) if player_id else 0,
        "target":       target,
        "predicted":    round(float(predicted), 2),
        "custom_line":  round(float(custom_line), 2),
        "pick":         pick,
        "over_prob":    round(float(over_prob), 4),
        "opponent":     opponent,
        "is_home":      bool(is_home),
        "confidence":   confidence,
        "bet_size":     bet_size,
        "game_date":    game_date,
        "resolved":     False,
        "correct":      None,
        "actual":       None,
        "notes":        notes,
        "units_wagered": wagered,
        "units_won":    0.0,
    }

    predictions = _load_predictions()
    predictions.append(record)
    _save_predictions(predictions)
    return record


def manual_resolve(prediction_id: str, actual_value: float) -> bool:
    """
    Manually mark a prediction as resolved with the actual stat result.

    Parameters
    ----------
    prediction_id : UUID string of the prediction to resolve.
    actual_value  : The player's actual stat value in that game.

    Returns
    -------
    True if the prediction was found and updated, False otherwise.
    """
    predictions = _load_predictions()
    record = _find_by_id(predictions, prediction_id)
    if record is None:
        return False

    actual = round(float(actual_value), 2)
    line   = record["custom_line"]
    pick   = record["pick"]

    went_over = actual > line
    correct   = (pick == "OVER" and went_over) or (pick == "UNDER" and not went_over)

    wagered = record.get("units_wagered", _units_wagered(record.get("bet_size", "1u")))
    won     = _units_won(wagered, correct)

    record.update({
        "resolved":    True,
        "correct":     correct,
        "actual":      actual,
        "units_won":   won,
    })

    _save_predictions(predictions)
    return True


def resolve_predictions() -> int:
    """
    Auto-resolve pending predictions where game_date < today by attempting
    to fetch actual results from the NBA API.

    Currently marks past-date predictions as 'resolvable' (sets a flag) and
    attempts NBA API lookups. Returns count of newly resolved predictions.

    If the NBA boxscore API is unavailable, predictions remain pending and
    must be resolved manually via manual_resolve().
    """
    predictions = _load_predictions()
    today = date.today()
    newly_resolved = 0

    pending = [p for p in predictions if not p.get("resolved", False)]
    if not pending:
        return 0

    # Try to import NBA API for actual results
    try:
        from nba_api.stats.endpoints import playergamelog
        from nba_api.stats.library.parameters import SeasonAll
        nba_available = True
    except ImportError:
        nba_available = False

    for record in pending:
        game_date_str = record.get("game_date", "")
        try:
            game_date = date.fromisoformat(game_date_str)
        except (ValueError, TypeError):
            continue

        # Only attempt resolution for past games
        if game_date >= today:
            continue

        resolved = False

        if nba_available:
            resolved = _try_nba_resolve(record)

        if resolved:
            newly_resolved += 1

    if newly_resolved:
        _save_predictions(predictions)

    return newly_resolved


def _try_nba_resolve(record: dict) -> bool:
    """
    Attempt to resolve a single prediction using the NBA API.
    Returns True if successfully resolved, False otherwise.
    Modifies record in-place.
    """
    try:
        from nba_api.stats.endpoints import playergamelog
        import time as _time

        player_id   = record.get("player_id", 0)
        target      = record.get("target", "")
        game_date   = record.get("game_date", "")
        line        = record.get("custom_line", 0.0)
        pick        = record.get("pick", "OVER")

        if not player_id or not game_date:
            return False

        _time.sleep(0.6)  # NBA API rate limit
        logs = playergamelog.PlayerGameLog(
            player_id=player_id,
            season="2024-25",
        ).get_data_frames()[0]

        if logs.empty:
            return False

        # Normalize date column
        date_col = "GAME_DATE" if "GAME_DATE" in logs.columns else logs.columns[0]
        logs[date_col] = pd.to_datetime(logs[date_col], errors="coerce")

        target_date = pd.to_datetime(game_date)
        game_rows = logs[logs[date_col].dt.date == target_date.date()]

        if game_rows.empty:
            return False

        row = game_rows.iloc[0]

        # Map target to NBA API column
        nba_col_map = {
            "pts": "PTS", "reb": "REB", "ast": "AST",
            "stl": "STL", "blk": "BLK", "tov": "TOV",
            "fg3m": "FG3M", "fg3a": "FG3A", "fga": "FGA",
        }

        from features.engineer import COMBO_TARGETS
        import pandas as pd

        if target in COMBO_TARGETS:
            parts = COMBO_TARGETS[target]
            vals = []
            for p in parts:
                col = nba_col_map.get(p)
                if col and col in row.index:
                    vals.append(float(row[col]))
            actual = sum(vals) if vals else None
        else:
            col = nba_col_map.get(target)
            actual = float(row[col]) if col and col in row.index else None

        if actual is None:
            return False

        went_over = actual > line
        correct   = (pick == "OVER" and went_over) or (pick == "UNDER" and not went_over)
        wagered   = record.get("units_wagered", _units_wagered(record.get("bet_size", "1u")))
        won       = _units_won(wagered, correct)

        record.update({
            "resolved":  True,
            "correct":   correct,
            "actual":    round(actual, 2),
            "units_won": won,
        })
        return True

    except Exception as e:
        print(f"  _try_nba_resolve failed: {e}")
        return False


def add_note(prediction_id: str, note: str) -> bool:
    """
    Add or update the handicapper note on a prediction.

    Returns True if found and updated, False otherwise.
    """
    predictions = _load_predictions()
    record = _find_by_id(predictions, prediction_id)
    if record is None:
        return False

    record["notes"] = note
    _save_predictions(predictions)
    return True


def get_recent_predictions(n: int = 20) -> list:
    """
    Return the last n predictions sorted by game_date descending.
    """
    predictions = _load_predictions()
    sorted_preds = sorted(
        predictions,
        key=lambda p: p.get("game_date", ""),
        reverse=True,
    )
    return sorted_preds[:n]


def get_weekly_best(days: int = 7) -> list:
    """
    Return top 5 predictions by highest over_prob (or by confidence)
    from the past `days` days.
    """
    predictions = _load_predictions()
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    recent = [
        p for p in predictions
        if p.get("game_date", "") >= cutoff
    ]

    # Sort by max(over_prob, 1-over_prob) descending
    def edge_score(p):
        op = p.get("over_prob", 0.5)
        return max(op, 1 - op)

    recent.sort(key=edge_score, reverse=True)
    return recent[:5]


def get_accuracy_stats() -> dict:
    """
    Compute comprehensive accuracy and ROI statistics.

    Returns
    -------
    {
        "total": int,
        "correct": int,
        "accuracy": float,          # percentage (0-100)
        "streak": int,              # current win streak (positive) / loss streak (negative)
        "units_wagered": float,
        "units_profit": float,
        "roi": float,               # units_profit / units_wagered * 100
        "by_target": {target: {"total": int, "correct": int, "accuracy": float}},
        "by_confidence": {"High": {...}, "Medium": {...}, "Low": {...}},
        "recent_30d": {"total": int, "correct": int, "accuracy": float},
    }
    """
    predictions = _load_predictions()
    resolved = [p for p in predictions if p.get("resolved", False) and p.get("correct") is not None]

    total   = len(resolved)
    correct = sum(1 for p in resolved if p.get("correct", False))

    accuracy = round(correct / total * 100, 1) if total > 0 else 0.0

    # Current streak — walk backwards through resolved predictions sorted by date
    sorted_resolved = sorted(resolved, key=lambda p: p.get("game_date", ""))
    streak = 0
    if sorted_resolved:
        last_correct = sorted_resolved[-1].get("correct", False)
        for p in reversed(sorted_resolved):
            if p.get("correct", False) == last_correct:
                streak += 1
            else:
                break
        if not last_correct:
            streak = -streak  # negative = loss streak

    # Units
    total_wagered = sum(p.get("units_wagered", 1.0) for p in resolved)
    total_won     = sum(p.get("units_won", 0.0) for p in resolved)
    roi = round(total_won / total_wagered * 100, 1) if total_wagered > 0 else 0.0

    # By target
    by_target: dict = {}
    for p in resolved:
        tgt = p.get("target", "unknown")
        if tgt not in by_target:
            by_target[tgt] = {"total": 0, "correct": 0, "accuracy": 0.0}
        by_target[tgt]["total"] += 1
        if p.get("correct", False):
            by_target[tgt]["correct"] += 1
    for tgt in by_target:
        t = by_target[tgt]
        t["accuracy"] = round(t["correct"] / t["total"] * 100, 1) if t["total"] > 0 else 0.0

    # By confidence
    by_confidence: dict = {lvl: {"total": 0, "correct": 0, "accuracy": 0.0}
                           for lvl in ("High", "Medium", "Low")}
    for p in resolved:
        conf = p.get("confidence", "Medium")
        if conf not in by_confidence:
            by_confidence[conf] = {"total": 0, "correct": 0, "accuracy": 0.0}
        by_confidence[conf]["total"] += 1
        if p.get("correct", False):
            by_confidence[conf]["correct"] += 1
    for conf in by_confidence:
        c = by_confidence[conf]
        c["accuracy"] = round(c["correct"] / c["total"] * 100, 1) if c["total"] > 0 else 0.0

    # Recent 30 days
    cutoff_30 = (date.today() - timedelta(days=30)).isoformat()
    recent_30 = [p for p in resolved if p.get("game_date", "") >= cutoff_30]
    r30_total   = len(recent_30)
    r30_correct = sum(1 for p in recent_30 if p.get("correct", False))
    recent_30d = {
        "total":    r30_total,
        "correct":  r30_correct,
        "accuracy": round(r30_correct / r30_total * 100, 1) if r30_total > 0 else 0.0,
    }

    return {
        "total":          total,
        "correct":        correct,
        "accuracy":       accuracy,
        "streak":         streak,
        "units_wagered":  round(total_wagered, 2),
        "units_profit":   round(total_won, 2),
        "roi":            roi,
        "by_target":      by_target,
        "by_confidence":  by_confidence,
        "recent_30d":     recent_30d,
    }


def auto_resolve_from_espn() -> int:
    """
    Auto-resolve pending predictions by looking up actual stats from
    ESPN box scores for recently completed games.
    Returns the count of predictions resolved.
    """
    try:
        from data.fetch_boxscores import lookup_player_actual
    except ImportError:
        return 0

    predictions = _load_predictions()
    resolved_count = 0
    today     = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    for p in predictions:
        if p.get("resolved"):
            continue
        game_date = p.get("game_date", "")[:10]
        if game_date not in (today, yesterday):
            continue
        actual = lookup_player_actual(p["player_name"], p["target"], days_back=2)
        if actual is None:
            continue
        pick    = p.get("pick", "OVER")
        line    = float(p.get("custom_line", 0))
        correct = (actual > line) if pick == "OVER" else (actual <= line)
        wagered = p.get("units_wagered", _units_wagered(p.get("bet_size", "1u")))
        p.update({
            "resolved":    True,
            "correct":     correct,
            "actual":      actual,
            "units_won":   _units_won(wagered, correct),
        })
        resolved_count += 1

    if resolved_count:
        _save_predictions(predictions)
    return resolved_count


def get_accuracy_trend(days: int = 30) -> list:
    """
    Return daily accuracy trend for the past `days` days.
    Returns list of {date, accuracy, count} dicts sorted oldest→newest.
    Only includes dates that have at least one resolved prediction.
    """
    predictions = _load_predictions()
    resolved    = [p for p in predictions if p.get("resolved") and p.get("correct") is not None]
    cutoff      = (date.today() - timedelta(days=days)).isoformat()
    recent      = [p for p in resolved if p.get("game_date", "")[:10] >= cutoff]

    by_date: dict = {}
    for p in recent:
        d = p.get("game_date", "")[:10]
        if d not in by_date:
            by_date[d] = {"total": 0, "correct": 0}
        by_date[d]["total"] += 1
        if p.get("correct"):
            by_date[d]["correct"] += 1

    trend = []
    running_correct = 0
    running_total   = 0
    for d in sorted(by_date.keys()):
        entry = by_date[d]
        running_correct += entry["correct"]
        running_total   += entry["total"]
        trend.append({
            "date":     d,
            "accuracy": round(entry["correct"] / entry["total"] * 100, 1),
            "rolling":  round(running_correct / running_total * 100, 1),
            "count":    entry["total"],
        })
    return trend


def get_all_predictions() -> list:
    """Return all predictions (resolved and pending), most recent first."""
    predictions = _load_predictions()
    return sorted(predictions, key=lambda p: p.get("game_date", ""), reverse=True)


def delete_prediction(prediction_id: str) -> bool:
    """
    Remove a prediction by ID. Returns True if found and deleted.
    Use with caution — this is permanent.
    """
    predictions = _load_predictions()
    before = len(predictions)
    predictions = [p for p in predictions if p.get("id") != prediction_id]
    if len(predictions) == before:
        return False
    _save_predictions(predictions)
    return True


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Accuracy Tracker Test ===")

    # Log a sample prediction
    pred = log_prediction(
        player_name="LeBron James",
        player_id=2544,
        target="pts",
        predicted=27.3,
        custom_line=25.5,
        over_prob=0.68,
        opponent="GSW",
        is_home=True,
        confidence="Medium",
        bet_size="1u",
        notes="Strong home game vs weak GSW defense",
    )
    print(f"Logged prediction: {pred['id'][:8]}... ({pred['player_name']} {pred['target']} {pred['pick']} {pred['custom_line']})")

    # Manually resolve it
    resolved = manual_resolve(pred["id"], actual_value=29.0)
    print(f"Resolved: {resolved}")

    # Stats
    stats = get_accuracy_stats()
    print(f"\nAccuracy stats:")
    print(f"  Total: {stats['total']}  Correct: {stats['correct']}  Accuracy: {stats['accuracy']}%")
    print(f"  ROI: {stats['roi']}%  Units profit: {stats['units_profit']}")
    print(f"  Streak: {stats['streak']}")

    # Recent
    recent = get_recent_predictions(5)
    print(f"\nRecent predictions ({len(recent)}):")
    for p in recent:
        status = "✓" if p.get("correct") else ("✗" if p.get("resolved") else "?")
        print(f"  [{status}] {p['player_name']} {p['target']} {p['pick']} {p['custom_line']} "
              f"(actual: {p.get('actual')})")
