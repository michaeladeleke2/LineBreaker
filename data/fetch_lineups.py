"""
linebreaker/data/fetch_lineups.py

Fetches NBA starting lineup data from ESPN's free public API.
Note: The depth chart endpoint is no longer available (404).
Lineups are only available from actual boxscores once games start.
"""

import json
import requests
import pandas as pd
from pathlib import Path
from datetime import date, datetime

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY    = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
HEADERS         = {"User-Agent": "Mozilla/5.0"}
TIMEOUT         = 10

ROLE_MULTIPLIERS = {
    "starter":   1.0,
    "bench":     0.75,
    "inactive":  0.0,
    "unknown":   1.0,
}


def _get(url, params=None):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def fetch_all_lineups(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch today's lineup data from ESPN boxscores only.
    Only works for live or completed games (not pre-game).
    Returns DataFrame with player lineup status.
    """
    today      = date.today().isoformat()
    cache_path = CACHE_DIR / f"lineups_{today}.csv"

    # Use cache if fresh (within 30 min)
    if cache_path.exists() and not force_refresh:
        age = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).seconds
        if age < 1800:
            return pd.read_csv(cache_path)

    # Clean old lineup caches
    for old in CACHE_DIR.glob("lineups_*.csv"):
        if old.name != cache_path.name:
            try: old.unlink()
            except: pass

    rows = []

    # Only fetch boxscores for live/final games — no depth chart calls
    board  = _get(ESPN_SCOREBOARD)
    events = board.get("events", [])

    for event in events:
        game_id    = event.get("id", "")
        status_num = event.get("status", {}).get("type", {}).get("id", "1")

        # Only process live (2) or final (3) games
        if str(status_num) not in ("2", "3"):
            continue

        summary = _get(ESPN_SUMMARY, params={"event": game_id})
        box     = summary.get("boxscore", {})

        for team_box in box.get("players", []):
            team_abbr = team_box.get("team", {}).get("abbreviation", "")
            stats     = team_box.get("statistics", [{}])
            athletes  = stats[0].get("athletes", []) if stats else []

            for ath in athletes:
                name         = ath.get("athlete", {}).get("displayName", "")
                starter      = ath.get("starter", False)
                active       = ath.get("active", True)
                did_not_play = ath.get("didNotPlay", False)

                if did_not_play or not active:
                    role = "inactive"
                elif starter:
                    role = "starter"
                else:
                    role = "bench"

                rows.append({
                    "player_name":       name,
                    "team_abbr":         team_abbr,
                    "is_starter":        starter and active and not did_not_play,
                    "role":              role,
                    "starter_confirmed": True,
                    "game_id":           game_id,
                })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["player_name", "team_abbr", "is_starter", "role",
                 "starter_confirmed", "game_id"])

    if not df.empty:
        df.to_csv(cache_path, index=False)

    return df


def get_player_lineup_status(player_name: str,
                              lineup_df: pd.DataFrame = None) -> dict:
    """Get lineup status for a player. Returns unknown if no data."""
    if lineup_df is None:
        lineup_df = fetch_all_lineups()

    default = {"is_starter": None, "role": "unknown",
               "multiplier": 1.0, "confirmed": False}

    if lineup_df is None or lineup_df.empty:
        return default

    last    = player_name.split()[-1]
    matches = lineup_df[lineup_df["player_name"].str.contains(last, case=False, na=False)]

    if matches.empty:
        return default

    confirmed = matches[matches["starter_confirmed"]]
    row       = confirmed.iloc[0] if not confirmed.empty else matches.iloc[0]
    role      = str(row.get("role", "unknown"))

    return {
        "is_starter": bool(row.get("is_starter", False)),
        "role":        role,
        "multiplier":  ROLE_MULTIPLIERS.get(role, 1.0),
        "confirmed":   bool(row.get("starter_confirmed", False)),
    }


def get_team_starters(team_abbr: str, lineup_df: pd.DataFrame = None) -> list:
    """Get confirmed starters for a team."""
    if lineup_df is None:
        lineup_df = fetch_all_lineups()
    if lineup_df is None or lineup_df.empty:
        return []
    mask = ((lineup_df["team_abbr"].str.upper() == team_abbr.upper()) &
            (lineup_df["is_starter"] == True))
    return lineup_df[mask]["player_name"].tolist()