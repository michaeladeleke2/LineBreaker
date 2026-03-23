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


def _name_similarity(a: str, b: str) -> float:
    """
    Return 0-1 similarity between two player name strings.
    Prefers exact match, then first+last match, then last-name-only as last resort.
    """
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    a_parts, b_parts = a.split(), b.split()
    a_last = a_parts[-1] if a_parts else ""
    b_last = b_parts[-1] if b_parts else ""
    a_first = a_parts[0] if a_parts else ""
    b_first = b_parts[0] if b_parts else ""
    # Both first AND last name must match for a confident hit
    if a_last == b_last and len(a_last) > 2:
        if a_first == b_first:
            return 0.95   # first + last match
        if a_first and b_first and (a_first[0] == b_first[0]):
            return 0.7    # first initial + last match
        # Last name only — only use if last name is distinctive (len > 5)
        return 0.4 if len(a_last) > 5 else 0.0
    return 0.0


def get_player_injury(player_name: str,
                       injury_df: pd.DataFrame = None) -> dict:
    """
    Look up a specific player's injury status.
    Uses full-name matching to avoid false positives from shared last names
    (e.g. Jaylen Brown ≠ Moses Brown).
    Returns dict with status, multiplier, description.
    """
    if injury_df is None:
        injury_df = fetch_injury_report()

    if injury_df.empty:
        return {"status": "active", "multiplier": 1.0, "description": ""}

    # Score every row and take the best match above threshold
    best_score, best_row = 0.0, None
    for _, row in injury_df.iterrows():
        score = _name_similarity(player_name, str(row["player_name"]))
        if score > best_score:
            best_score, best_row = score, row

    # Require confident match — last-name-only hits (score ~0.4) are rejected
    # to avoid false positives like Jaylen Brown → Moses Brown
    if best_score < 0.6 or best_row is None:
        return {"status": "active", "multiplier": 1.0, "description": ""}

    return {
        "status":      best_row["status"],
        "multiplier":  float(best_row["multiplier"]),
        "description": best_row["description"],
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