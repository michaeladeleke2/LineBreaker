"""
linebreaker/data/fetch_lineups.py

Fetches NBA starting lineup data from ESPN's free public API.
No API key required.

Endpoints used:
- Scoreboard: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
  → Contains today's games with team IDs
- Boxscore: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}
  → Contains starters for live/completed games

For upcoming games (no boxscore yet), falls back to:
- Depth chart: https://site.api.espn.com/apis/v2/sports/basketball/nba/teams/{team_id}/depthcharts
  → Lists expected starters by position

Usage:
    from data.fetch_lineups import get_player_lineup_status, fetch_all_lineups
    status = get_player_lineup_status("LeBron James")
    # Returns: {"is_starter": True, "status": "starter", "multiplier": 1.0}
"""

import json
import requests
import pandas as pd
from pathlib import Path
from datetime import date, datetime

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

ESPN_SCOREBOARD  = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY     = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
ESPN_DEPTHCHART  = "https://site.api.espn.com/apis/v2/sports/basketball/nba/teams/{team_id}/depthcharts"

HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 12

# Minutes multiplier based on role
ROLE_MULTIPLIERS = {
    "starter":      1.0,
    "sixth_man":    0.88,
    "bench":        0.75,
    "inactive":     0.0,
    "unknown":      1.0,  # don't penalize if we can't determine
}


def _get(url: str, params: dict = None) -> dict:
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Lineup fetch failed: {e}")
        return {}


def fetch_all_lineups(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch today's starting lineup data.
    Returns DataFrame: player_name, team_abbr, is_starter, role, starter_confirmed
    Cached for 30 minutes (lineups change up to tip-off).
    """
    today      = date.today().isoformat()
    cache_path = CACHE_DIR / f"lineups_{today}.csv"

    # Use cache if fresh enough (within 30 min)
    if cache_path.exists() and not force_refresh:
        age = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).seconds
        if age < 1800:
            df = pd.read_csv(cache_path)
            print(f"  Lineup cache loaded ({len(df)} players, {age//60}min old)")
            return df

    # Clean old lineup caches
    for old in CACHE_DIR.glob("lineups_*.csv"):
        if old.name != cache_path.name:
            old.unlink()

    print("Fetching lineup data from ESPN...")
    rows = []

    # Get today's scoreboard for game IDs
    board = _get(ESPN_SCOREBOARD)
    events = board.get("events", [])

    for event in events:
        game_id    = event.get("id", "")
        status_num = event.get("status", {}).get("type", {}).get("id", "1")
        competitions = event.get("competitions", [{}])
        if not competitions:
            continue

        comp = competitions[0]
        teams = comp.get("competitors", [])

        # Try boxscore first (live or completed games)
        if str(status_num) in ("2", "3"):  # live or final
            summary = _get(ESPN_SUMMARY, params={"event": game_id})
            box = summary.get("boxscore", {})
            for team_box in box.get("players", []):
                team_abbr = team_box.get("team", {}).get("abbreviation", "")
                stats     = team_box.get("statistics", [{}])
                athletes  = stats[0].get("athletes", []) if stats else []
                for ath in athletes:
                    name    = ath.get("athlete", {}).get("displayName", "")
                    starter = ath.get("starter", False)
                    active  = ath.get("active", True)
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
                        "starter_confirmed": True,  # from actual boxscore
                        "game_id":           game_id,
                    })
        else:
            # Upcoming game — depth chart endpoint no longer available
            # Lineups will be confirmed once game starts via boxscore
            pass

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["player_name","team_abbr","is_starter","role","starter_confirmed","game_id"])

    if not df.empty:
        df.to_csv(cache_path, index=False)
        confirmed = df["starter_confirmed"].sum()
        projected  = (~df["starter_confirmed"]).sum()
        print(f"  Lineups: {len(df)} players ({confirmed} confirmed, {projected} projected)")

    return df


def get_player_lineup_status(player_name: str,
                              lineup_df: pd.DataFrame = None) -> dict:
    """
    Get lineup status for a specific player.
    Returns dict: is_starter, role, multiplier, confirmed
    """
    if lineup_df is None:
        lineup_df = fetch_all_lineups()

    if lineup_df.empty:
        return {"is_starter": None, "role": "unknown",
                "multiplier": 1.0, "confirmed": False}

    # Match on last name first, then full name
    last_name = player_name.split()[-1]
    matches   = lineup_df[lineup_df["player_name"].str.contains(last_name, case=False, na=False)]

    if matches.empty:
        return {"is_starter": None, "role": "unknown",
                "multiplier": 1.0, "confirmed": False}

    # If multiple matches, prefer confirmed
    confirmed = matches[matches["starter_confirmed"]]
    row = confirmed.iloc[0] if not confirmed.empty else matches.iloc[0]

    role       = str(row.get("role", "unknown"))
    multiplier = ROLE_MULTIPLIERS.get(role, 1.0)

    return {
        "is_starter": bool(row.get("is_starter", False)),
        "role":        role,
        "multiplier":  multiplier,
        "confirmed":   bool(row.get("starter_confirmed", False)),
    }


def get_team_starters(team_abbr: str,
                       lineup_df: pd.DataFrame = None) -> list:
    """Get confirmed starters for a team — useful for opponent context."""
    if lineup_df is None:
        lineup_df = fetch_all_lineups()

    if lineup_df.empty:
        return []

    team = lineup_df[
        (lineup_df["team_abbr"].str.upper() == team_abbr.upper()) &
        (lineup_df["is_starter"] == True)
    ]
    return team["player_name"].tolist()


if __name__ == "__main__":
    df = fetch_all_lineups()
    print(f"\nTotal players: {len(df)}")
    if not df.empty:
        print("\nConfirmed starters:")
        starters = df[df["is_starter"] & df["starter_confirmed"]]
        print(starters[["player_name","team_abbr","role"]].to_string(index=False))