"""
underdog_tracker.py
-------------------
Pick tracking system for Underdog Fantasy NBA prop picks.
Persists to data/cache/underdog_picks.json.
"""

import json
import uuid
import os
from datetime import datetime, date, timedelta
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DIR = os.path.dirname(__file__)
_CACHE_PATH = os.path.join(_DIR, "cache", "underdog_picks.json")

# ---------------------------------------------------------------------------
# ESPN stat name -> internal key mapping
# ---------------------------------------------------------------------------

_ESPN_STAT_MAP = {
    "PTS": "pts",
    "REB": "reb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "TO":  "tov",
    "3PM": "fg3m",
    "3PA": "fg3a",
    "FGA": "fga",
}

# Combo stats are sums of component keys
_COMBO_STATS = {
    "pts_reb_ast": ["pts", "reb", "ast"],
    "pts_reb":     ["pts", "reb"],
    "pts_ast":     ["pts", "ast"],
    "reb_ast":     ["reb", "ast"],
    "blk_stl":     ["blk", "stl"],
}

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _load() -> list:
    """Load picks from JSON file; return empty list if file missing."""
    if not os.path.exists(_CACHE_PATH):
        return []
    try:
        with open(_CACHE_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError):
        return []


def _save(picks: list) -> None:
    """Persist picks list to JSON file."""
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(picks, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_pick(
    player: str,
    team: str,
    opponent: str,
    stat: str,
    stat_label: str,
    line: float,
    direction: str,
    predicted: Optional[float] = None,
    game_id: Optional[str] = None,
    notes: str = "",
) -> str:
    """
    Create a new pick entry and append it to the store.

    Parameters
    ----------
    player      : Player display name (e.g. "Jaylen Brown")
    team        : Team abbreviation (e.g. "BOS")
    opponent    : Opponent abbreviation (e.g. "MIL")
    stat        : Internal target key (e.g. "pts")
    stat_label  : Display label (e.g. "Points")
    line        : Underdog line value
    direction   : "OVER" or "UNDER"
    predicted   : Model prediction (optional)
    game_id     : ESPN game_id for auto-resolve (optional)
    notes       : Free-text notes

    Returns
    -------
    str : The pick UUID
    """
    direction = direction.upper()
    if direction not in ("OVER", "UNDER"):
        raise ValueError(f"direction must be 'OVER' or 'UNDER', got {direction!r}")

    pick_id = str(uuid.uuid4())
    pick = {
        "id": pick_id,
        "date": date.today().isoformat(),
        "player": player,
        "team": team,
        "opponent": opponent,
        "stat": stat,
        "stat_label": stat_label,
        "line": float(line),
        "direction": direction,
        "predicted": float(predicted) if predicted is not None else None,
        "actual": None,
        "outcome": None,
        "game_id": game_id,
        "source": "manual",
        "notes": notes,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    picks = _load()
    picks.append(pick)
    _save(picks)
    return pick_id


def resolve_pick(pick_id: str, actual_value: float) -> dict:
    """
    Set the actual result for a pick and compute outcome.

    Outcome:
      "W"  — direction was correct
      "L"  — direction was wrong
      "P"  — within 0.25 of the line (push)

    Returns the updated pick dict.
    """
    picks = _load()
    for pick in picks:
        if pick["id"] == pick_id:
            actual = float(actual_value)
            pick["actual"] = actual
            line = pick["line"]
            diff = actual - line
            if abs(diff) <= 0.25:
                pick["outcome"] = "P"
            elif pick["direction"] == "OVER":
                pick["outcome"] = "W" if diff > 0 else "L"
            else:  # UNDER
                pick["outcome"] = "W" if diff < 0 else "L"
            _save(picks)
            return pick
    raise KeyError(f"Pick not found: {pick_id}")


def _get_espn_scoreboard_game_ids(target_date: date) -> list:
    """Return list of ESPN game IDs for a given date."""
    date_str = target_date.strftime("%Y%m%d")
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        f"?dates={date_str}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [ev["id"] for ev in data.get("events", [])]
    except Exception:
        return []


def _get_player_stats_from_boxscore(game_id: str) -> dict:
    """
    Fetch ESPN boxscore for a game and return:
      {player_name_lower: {stat_key: value, ...}}
    """
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
        f"?event={game_id}"
    )
    result = {}
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        boxscore = data.get("boxscore", {})
        teams = boxscore.get("players", [])

        for team_data in teams:
            stats_groups = team_data.get("statistics", [])
            if not stats_groups:
                continue
            group = stats_groups[0]
            col_names = group.get("names", [])
            athletes = group.get("athletes", [])

            for athlete_entry in athletes:
                athlete = athlete_entry.get("athlete", {})
                display_name = athlete.get("displayName", "")
                raw_stats = athlete_entry.get("stats", [])

                if not display_name or not raw_stats:
                    continue

                # Build a dict of ESPN stat name -> float value
                espn_vals = {}
                for col, raw in zip(col_names, raw_stats):
                    try:
                        espn_vals[col] = float(raw)
                    except (ValueError, TypeError):
                        espn_vals[col] = 0.0

                # Map to internal keys
                player_stats = {}
                for espn_name, internal_key in _ESPN_STAT_MAP.items():
                    if espn_name in espn_vals:
                        player_stats[internal_key] = espn_vals[espn_name]

                # Compute combo stats
                for combo_key, components in _COMBO_STATS.items():
                    if all(c in player_stats for c in components):
                        player_stats[combo_key] = sum(
                            player_stats[c] for c in components
                        )

                result[display_name.lower()] = player_stats

    except Exception:
        pass

    return result


def auto_resolve_all() -> int:
    """
    Attempt to auto-resolve all unresolved picks from the last 7 days.

    Pulls ESPN boxscores for today and yesterday, matches picks by game_id
    or by scanning all games for the pick date.

    Returns count of newly resolved picks.
    """
    picks = _load()
    today = date.today()
    cutoff = today - timedelta(days=7)

    # Collect candidate picks (unresolved, recent)
    candidates = [
        p for p in picks
        if p.get("outcome") is None
        and p.get("date", "") >= cutoff.isoformat()
    ]

    if not candidates:
        return 0

    # Determine dates to fetch boxscores for
    pick_dates = set()
    for p in candidates:
        pick_dates.add(p.get("date", ""))
    # Always include today and yesterday
    pick_dates.add(today.isoformat())
    pick_dates.add((today - timedelta(days=1)).isoformat())

    # Build cache: game_id -> {player_lower: {stat: value}}
    game_cache: dict = {}

    # Build per-date game list (for picks without game_id)
    date_games: dict = {}  # date_str -> [game_id, ...]

    for date_str in pick_dates:
        try:
            d = date.fromisoformat(date_str)
        except ValueError:
            continue
        gids = _get_espn_scoreboard_game_ids(d)
        date_games[date_str] = gids

    resolved_count = 0

    for pick in candidates:
        player_lower = pick.get("player", "").lower()
        stat = pick.get("stat", "")
        pick_date = pick.get("date", "")
        explicit_game_id = pick.get("game_id")

        # Gather candidate game IDs
        if explicit_game_id:
            candidate_game_ids = [explicit_game_id]
        else:
            candidate_game_ids = date_games.get(pick_date, [])
            # Also try yesterday/today in case the game crossed midnight
            for fallback_date in [
                pick_date,
                (date.fromisoformat(pick_date) + timedelta(days=1)).isoformat()
                if pick_date else None,
            ]:
                if fallback_date and fallback_date != pick_date:
                    candidate_game_ids += date_games.get(fallback_date, [])

        actual_value = None
        for gid in candidate_game_ids:
            if gid not in game_cache:
                game_cache[gid] = _get_player_stats_from_boxscore(gid)

            player_data = game_cache[gid].get(player_lower)
            if player_data and stat in player_data:
                actual_value = player_data[stat]
                break

        if actual_value is not None:
            try:
                resolve_pick(pick["id"], actual_value)
                resolved_count += 1
                # Reload to get updated state for next iteration
                picks = _load()
            except (KeyError, Exception):
                pass

    return resolved_count


def get_picks(
    days: int = 30,
    resolved_only: bool = False,
    unresolved_only: bool = False,
) -> list:
    """
    Return picks filtered by recency and resolution status.

    Parameters
    ----------
    days            : Only return picks from the last N days
    resolved_only   : If True, only return picks with an outcome
    unresolved_only : If True, only return picks without an outcome
    """
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    picks = _load()

    result = []
    for p in picks:
        if p.get("date", "") < cutoff:
            continue
        if resolved_only and p.get("outcome") is None:
            continue
        if unresolved_only and p.get("outcome") is not None:
            continue
        result.append(p)

    return result


def get_stats() -> dict:
    """
    Return aggregated performance stats across all picks.

    Returns
    -------
    dict with keys: total, resolved, wins, losses, pushes,
                    win_rate, by_stat, by_direction, recent_form
    """
    picks = _load()

    total = len(picks)
    resolved = [p for p in picks if p.get("outcome") is not None]
    wins = [p for p in resolved if p.get("outcome") == "W"]
    losses = [p for p in resolved if p.get("outcome") == "L"]
    pushes = [p for p in resolved if p.get("outcome") == "P"]

    n_resolved = len(resolved)
    n_wins = len(wins)
    win_rate = round(n_wins / n_resolved * 100, 1) if n_resolved else 0.0

    # Per-stat breakdown
    by_stat: dict = {}
    for p in resolved:
        stat = p.get("stat", "unknown")
        if stat not in by_stat:
            by_stat[stat] = {"total": 0, "wins": 0, "win_rate": 0.0}
        by_stat[stat]["total"] += 1
        if p.get("outcome") == "W":
            by_stat[stat]["wins"] += 1

    for stat, s in by_stat.items():
        s["win_rate"] = (
            round(s["wins"] / s["total"] * 100, 1) if s["total"] else 0.0
        )

    # Per-direction breakdown
    by_direction: dict = {}
    for p in resolved:
        direction = p.get("direction", "OVER")
        if direction not in by_direction:
            by_direction[direction] = {"total": 0, "wins": 0, "win_rate": 0.0}
        by_direction[direction]["total"] += 1
        if p.get("outcome") == "W":
            by_direction[direction]["wins"] += 1

    for direction, s in by_direction.items():
        s["win_rate"] = (
            round(s["wins"] / s["total"] * 100, 1) if s["total"] else 0.0
        )

    # Recent form: last 10 resolved picks, newest first
    resolved_sorted = sorted(
        resolved,
        key=lambda p: p.get("created_at", p.get("date", "")),
        reverse=True,
    )
    recent_form = [
        {
            "player": p.get("player"),
            "stat": p.get("stat"),
            "stat_label": p.get("stat_label"),
            "outcome": p.get("outcome"),
            "date": p.get("date"),
            "line": p.get("line"),
            "actual": p.get("actual"),
            "direction": p.get("direction"),
        }
        for p in resolved_sorted[:10]
    ]

    return {
        "total": total,
        "resolved": n_resolved,
        "wins": n_wins,
        "losses": len(losses),
        "pushes": len(pushes),
        "win_rate": win_rate,
        "by_stat": by_stat,
        "by_direction": by_direction,
        "recent_form": recent_form,
    }


def delete_pick(pick_id: str) -> bool:
    """
    Remove a pick by id.

    Returns True if the pick was found and removed, False otherwise.
    """
    picks = _load()
    original_len = len(picks)
    picks = [p for p in picks if p.get("id") != pick_id]
    if len(picks) < original_len:
        _save(picks)
        return True
    return False


def get_all_picks() -> list:
    """Return all picks, newest first (by created_at)."""
    picks = _load()
    return sorted(
        picks,
        key=lambda p: p.get("created_at", p.get("date", "")),
        reverse=True,
    )
