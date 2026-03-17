"""
linebreaker/data/fetch_injuries.py

Fetches current NBA injury report from ESPN's free public API.
No API key required.

Endpoint: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries

Returns player injury status (Out / Questionable / Day-To-Day / Probable)
which is used as a feature multiplier in predictions.
"""

import time
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import date, datetime

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

ESPN_INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

# Status multipliers — used to scale predicted value down for injured players
# e.g. Questionable player predicted 25pts -> 25 * 0.85 = 21.25 adjusted
STATUS_MULTIPLIERS = {
    "out":          0.0,
    "doubtful":     0.15,
    "questionable": 0.80,
    "day-to-day":   0.88,
    "probable":     0.95,
    "active":       1.0,
    "healthy":      1.0,
}


def fetch_injury_report(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch today's NBA injury report from ESPN.
    Cached for the day — refreshes automatically next calendar day.

    Returns DataFrame with columns:
        player_name, team_abbr, status, description, last_updated
    """
    today      = date.today().isoformat()
    cache_path = CACHE_DIR / f"injuries_{today}.csv"

    # Use today's cache if it exists
    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path)
        print(f"Loaded injury cache ({len(df)} players)")
        return df

    # Clean up old injury caches
    for old in CACHE_DIR.glob("injuries_*.csv"):
        if old.name != cache_path.name:
            old.unlink()

    print("Fetching injury report from ESPN...")
    try:
        resp = requests.get(
            ESPN_INJURY_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ESPN injury fetch failed: {e}")
        return pd.DataFrame(columns=["player_name","team_abbr","status",
                                      "description","last_updated","multiplier"])

    rows = []
    teams = data.get("injuries", [])
    for team_entry in teams:
        team_abbr = team_entry.get("team", {}).get("abbreviation", "")
        injuries  = team_entry.get("injuries", [])
        for inj in injuries:
            athlete = inj.get("athlete", {})
            name    = athlete.get("displayName", "")
            status  = inj.get("status", "").lower().strip()
            desc    = inj.get("shortComment", inj.get("longComment", ""))
            updated = inj.get("date", "")

            multiplier = STATUS_MULTIPLIERS.get(status, 1.0)
            # Map common variants
            for key in STATUS_MULTIPLIERS:
                if key in status:
                    multiplier = STATUS_MULTIPLIERS[key]
                    break

            rows.append({
                "player_name":  name,
                "team_abbr":    team_abbr,
                "status":       status,
                "description":  desc,
                "last_updated": updated,
                "multiplier":   multiplier,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(cache_path, index=False)
        print(f"  Injury report saved — {len(df)} players")
    else:
        print("  No injury data returned")

    return df


def get_player_injury(player_name: str,
                       injury_df: pd.DataFrame = None) -> dict:
    """
    Look up a specific player's injury status.
    Returns dict with status, multiplier, description.
    """
    if injury_df is None:
        injury_df = fetch_injury_report()

    if injury_df.empty:
        return {"status": "unknown", "multiplier": 1.0, "description": ""}

    # Fuzzy name match
    mask = injury_df["player_name"].str.contains(
        player_name.split()[-1],  # match on last name
        case=False, na=False
    )
    matches = injury_df[mask]

    if matches.empty:
        return {"status": "active", "multiplier": 1.0, "description": ""}

    row = matches.iloc[0]
    return {
        "status":      row["status"],
        "multiplier":  float(row["multiplier"]),
        "description": row["description"],
    }


def get_team_injury_summary(team_abbr: str,
                             injury_df: pd.DataFrame = None) -> list:
    """
    Get all injured players for a team — useful for opponent defense context.
    """
    if injury_df is None:
        injury_df = fetch_injury_report()

    if injury_df.empty:
        return []

    team_injuries = injury_df[
        injury_df["team_abbr"].str.upper() == team_abbr.upper()
    ]
    return team_injuries.to_dict("records")


if __name__ == "__main__":
    df = fetch_injury_report()
    print(f"\nTotal injured players: {len(df)}")
    if not df.empty:
        print("\nOut players:")
        out = df[df["status"] == "out"][["player_name","team_abbr","description"]]
        print(out.to_string(index=False))
        print("\nQuestionable players:")
        q = df[df["status"].str.contains("question", na=False)][["player_name","team_abbr","description"]]
        print(q.to_string(index=False))