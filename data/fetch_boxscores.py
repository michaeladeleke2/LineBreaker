"""
linebreaker/data/fetch_boxscores.py

Fetches completed NBA game box scores from ESPN's free public API.
Used to auto-resolve pending predictions in the accuracy tracker.

ESPN endpoints:
  Scoreboard: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
  Summary:    https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=<id>
"""

import time
import requests
from datetime import date, timedelta

HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 12

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY    = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

# Combo targets — value is sum of component targets
COMBO_TARGETS = {
    "pts_reb_ast": ["pts", "reb", "ast"],
    "pts_reb":     ["pts", "reb"],
    "pts_ast":     ["pts", "ast"],
    "reb_ast":     ["reb", "ast"],
    "blk_stl":     ["blk", "stl"],
}

# ESPN short-column → our target key
ESPN_COL_MAP = {
    "pts": "pts", "p": "pts",
    "reb": "reb", "r": "reb",
    "ast": "ast", "a": "ast",
    "stl": "stl",
    "blk": "blk",
    "to":  "tov",
    "3pm": "fg3m",
    "3pa": "fg3a",
    "fga": "fga",
}

_mem: dict = {}


def _get(url, params=None):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _cache(key, ttl=3600):
    entry = _mem.get(key)
    if entry and (time.time() - entry[0]) < ttl:
        return entry[1]
    return None


def _store(key, val):
    _mem[key] = (time.time(), val)


def get_completed_game_ids(days_back: int = 2) -> list:
    """
    Return ESPN event IDs for games completed within the last `days_back` days.
    """
    ids = []
    for d in range(days_back + 1):
        check = (date.today() - timedelta(days=d)).strftime("%Y%m%d")
        board = _get(ESPN_SCOREBOARD, params={"dates": check})
        for event in board.get("events", []):
            comps = event.get("competitions", [])
            if not comps:
                continue
            status_type = comps[0].get("status", {}).get("type", {})
            if status_type.get("completed", False) or status_type.get("name", "") == "STATUS_FINAL":
                eid = event.get("id", "")
                if eid:
                    ids.append(eid)
    return ids


def fetch_player_stats_from_game(event_id: str) -> dict:
    """
    Parse an ESPN game summary and return:
    {player_name_lower: {target_key: float_value, ...}}

    Includes combo targets (pts_reb_ast, etc.) and double_double/triple_double.
    Cached for 1 hour.
    """
    cache_key = f"bs_{event_id}"
    cached = _cache(cache_key)
    if cached is not None:
        return cached

    data = _get(ESPN_SUMMARY, params={"event": event_id})
    result: dict = {}

    boxscore = data.get("boxscore", {})
    for team_section in boxscore.get("players", []):
        for stat_section in team_section.get("statistics", []):
            col_names = [c.lower() for c in stat_section.get("names", [])]
            for athlete_entry in stat_section.get("athletes", []):
                athlete    = athlete_entry.get("athlete", {})
                name_full  = athlete.get("displayName", "").lower().strip()
                if not name_full:
                    continue
                raw_stats  = athlete_entry.get("stats", [])

                # Build raw stat dict
                raw: dict = {}
                for i, col in enumerate(col_names):
                    if i < len(raw_stats):
                        try:
                            raw[col] = float(raw_stats[i])
                        except (ValueError, TypeError):
                            pass

                # Map to our target keys
                player_stats: dict = {}
                for espn_col, our_key in ESPN_COL_MAP.items():
                    if espn_col in raw:
                        player_stats[our_key] = raw[espn_col]

                if not player_stats:
                    continue

                # Combo targets
                for combo, parts in COMBO_TARGETS.items():
                    if all(p in player_stats for p in parts):
                        player_stats[combo] = round(sum(player_stats[p] for p in parts), 1)

                # Double-double / triple-double
                counting = [player_stats.get(s, 0) for s in ("pts", "reb", "ast", "stl", "blk")]
                tens = sum(1 for v in counting if v >= 10)
                player_stats["double_double"] = 1.0 if tens >= 2 else 0.0
                player_stats["triple_double"] = 1.0 if tens >= 3 else 0.0

                result[name_full] = player_stats

    _store(cache_key, result)
    return result


def lookup_player_actual(player_name: str, target: str, days_back: int = 2):
    """
    Scan recent completed games and return the player's actual stat value.
    Returns float or None if not found.
    """
    name_lower = player_name.lower().strip()
    name_last  = name_lower.split()[-1] if name_lower.split() else name_lower

    for event_id in get_completed_game_ids(days_back):
        stats = fetch_player_stats_from_game(event_id)
        for pname, pstats in stats.items():
            if name_lower == pname or (len(name_last) > 3 and name_last in pname):
                val = pstats.get(target)
                if val is not None:
                    return float(val)
    return None


def get_team_defensive_rankings(feature_matrix_path) -> dict:
    """
    Compute per-team defensive rankings from the feature matrix.
    Returns {team_abbr: {"rank": int, "pts_allowed": float, "percentile": int}}
    for each team, ranked 1 (best defense) to 30 (worst).
    """
    import pandas as pd
    try:
        fm = pd.read_csv(feature_matrix_path)
        # opp_avg_pts_allowed is the column tracking opponent scoring allowed
        opp_cols = [c for c in fm.columns if "opp" in c and "pts" in c and "allowed" in c]
        if not opp_cols:
            # Fallback: look for any defensive indicator
            opp_cols = [c for c in fm.columns if c.startswith("opp_") and "pts" in c]
        if not opp_cols or "opponent_team" not in fm.columns:
            return {}

        col = opp_cols[0]
        agg = (fm.groupby("opponent_team")[col]
                 .mean()
                 .reset_index()
                 .rename(columns={col: "pts_allowed", "opponent_team": "team"}))
        agg = agg.sort_values("pts_allowed")  # ascending → lower = better defense
        agg["rank"] = range(1, len(agg) + 1)
        agg["percentile"] = (agg["rank"] / len(agg) * 100).astype(int)

        return {
            row["team"]: {
                "rank":        int(row["rank"]),
                "pts_allowed": round(float(row["pts_allowed"]), 1),
                "percentile":  int(row["percentile"]),
            }
            for _, row in agg.iterrows()
        }
    except Exception:
        return {}


if __name__ == "__main__":
    print("Fetching recent completed game IDs...")
    ids = get_completed_game_ids(days_back=1)
    print(f"Found {len(ids)} completed games: {ids[:3]}")
    if ids:
        print(f"\nFetching box score for game {ids[0]}...")
        stats = fetch_player_stats_from_game(ids[0])
        print(f"Players found: {len(stats)}")
        sample = list(stats.items())[:3]
        for name, s in sample:
            print(f"  {name}: {s}")
