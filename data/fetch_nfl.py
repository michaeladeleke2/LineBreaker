"""
linebreaker/data/fetch_nfl.py

Fetches NFL player/team data from ESPN's free public API.
No API key required — same ESPN API approach as fetch_injuries.py.

Data available: teams, rosters, per-game stats (gamelog), season averages.
"""

import math
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import date

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

ESPN_NFL     = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_NFL_WEB = "https://site.web.api.espn.com/apis/common/v3/sports/football/nfl"

# Current NFL season — update each September
NFL_SEASON = 2025  # 2025 NFL season (Sept 2025 – Feb 2026)

# ── Stat label mappings (ESPN name → our internal column name) ─────────────────
# The web API uses camelCase names matching the 'names' array in the response.
STAT_LABEL_MAP = {
    # Passing (QB)
    "completions":           "completions",
    "passingAttempts":       "pass_att",
    "passingYards":          "pass_yds",
    "passingTouchdowns":     "pass_tds",
    "touchdowns":            "pass_tds",       # some positions use generic name
    "interceptionsThrown":   "interceptions",
    "interceptions":         "interceptions",
    "QBRating":              "qb_rating",
    # Rushing (QB/RB/WR)
    "rushingAttempts":       "rush_att",
    "rushingYards":          "rush_yds",
    "rushingTouchdowns":     "rush_tds",
    # Receiving (WR/TE/RB)
    "receivingReceptions":   "receptions",
    "receptions":            "receptions",
    "receivingTargets":      "targets",
    "targets":               "targets",
    "receivingYards":        "rec_yds",
    "receivingTouchdowns":   "rec_tds",
}

# ── Target display config ──────────────────────────────────────────────────────
NFL_TARGET_DISPLAY = {
    "pass_yds":      {"label": "Pass Yards",    "short": "PASS YDS", "group": "Passing",   "pos": ["QB"]},
    "pass_tds":      {"label": "Pass TDs",      "short": "PASS TD",  "group": "Passing",   "pos": ["QB"]},
    "completions":   {"label": "Completions",   "short": "CMP",      "group": "Passing",   "pos": ["QB"]},
    "pass_att":      {"label": "Pass Attempts", "short": "ATT",      "group": "Passing",   "pos": ["QB"]},
    "interceptions": {"label": "Interceptions", "short": "INT",      "group": "Passing",   "pos": ["QB"]},
    "rush_yds":      {"label": "Rush Yards",    "short": "RUSH YDS", "group": "Rushing",   "pos": ["QB", "RB", "WR"]},
    "rush_tds":      {"label": "Rush TDs",      "short": "RUSH TD",  "group": "Rushing",   "pos": ["QB", "RB"]},
    "rush_att":      {"label": "Carries",       "short": "CAR",      "group": "Rushing",   "pos": ["QB", "RB"]},
    "rec_yds":       {"label": "Rec Yards",     "short": "REC YDS",  "group": "Receiving", "pos": ["RB", "WR", "TE"]},
    "receptions":    {"label": "Receptions",    "short": "REC",      "group": "Receiving", "pos": ["RB", "WR", "TE"]},
    "rec_tds":       {"label": "Rec TDs",       "short": "REC TD",   "group": "Receiving", "pos": ["RB", "WR", "TE"]},
    "targets":       {"label": "Targets",       "short": "TGT",      "group": "Receiving", "pos": ["RB", "WR", "TE"]},
}

NFL_TARGET_GROUPS = ["Passing", "Rushing", "Receiving"]

NFL_THRESHOLDS = {
    "pass_yds": 250, "pass_tds": 1.5, "completions": 22, "pass_att": 35,
    "interceptions": 0.5, "rush_yds": 65, "rush_tds": 0.5, "rush_att": 14,
    "rec_yds": 55, "receptions": 5, "rec_tds": 0.5, "targets": 6,
}

NFL_TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#C83803", "CIN": "#FB4F14", "CLE": "#311D00",
    "DAL": "#003594", "DEN": "#FA4616", "DET": "#0076B6", "GB": "#203731",
    "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837",
    "LV": "#A5ACAF", "LAC": "#0080C6", "LAR": "#003594", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SF": "#AA0000",
    "SEA": "#002244", "TB": "#D50A0A", "TEN": "#0C2340", "WAS": "#5A1414",
}


# ── HTTP helper ────────────────────────────────────────────────────────────────

def _req(url: str, params: dict = None) -> dict:
    time.sleep(0.4)
    r = requests.get(
        url, params=params,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


# ── Teams ──────────────────────────────────────────────────────────────────────

def fetch_nfl_teams(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch all 32 NFL teams from ESPN."""
    cache = CACHE_DIR / "nfl_teams.csv"
    if cache.exists() and not force_refresh:
        return pd.read_csv(cache, dtype={"team_id": str})

    try:
        data = _req(f"{ESPN_NFL}/teams", {"limit": 32})
        rows = []
        for grp in data.get("sports", [{}]):
            for league in grp.get("leagues", [{}]):
                for t in league.get("teams", []):
                    team = t.get("team", {})
                    rows.append({
                        "team_id":           str(team.get("id", "")),
                        "team_name":         team.get("displayName", ""),
                        "team_abbreviation": team.get("abbreviation", ""),
                    })
        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_csv(cache, index=False)
        return df
    except Exception as e:
        print(f"fetch_nfl_teams failed: {e}")
        return pd.DataFrame(columns=["team_id", "team_name", "team_abbreviation"])


# ── Rosters ────────────────────────────────────────────────────────────────────

def _fetch_roster(team_id: str, force: bool = False) -> pd.DataFrame:
    cache = CACHE_DIR / f"nfl_roster_{team_id}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache, dtype={"player_id": str, "team_id": str})

    try:
        data = _req(f"{ESPN_NFL}/teams/{team_id}/roster")
        rows = []
        for group in data.get("athletes", []):
            for ath in group.get("items", []):
                hs = ath.get("headshot", {})
                rows.append({
                    "player_id":    str(ath.get("id", "")),
                    "full_name":    ath.get("fullName", ""),
                    "position":     ath.get("position", {}).get("abbreviation", ""),
                    "jersey":       ath.get("jersey", ""),
                    "team_id":      str(team_id),
                    "headshot_url": hs.get("href", "") if isinstance(hs, dict) else "",
                })
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty:
            df.to_csv(cache, index=False)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_all_nfl_players(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch rosters for all 32 teams and combine into one players dataframe."""
    cache = CACHE_DIR / "nfl_all_players.csv"
    if cache.exists() and not force_refresh:
        return pd.read_csv(cache, dtype={"player_id": str, "team_id": str})

    teams = fetch_nfl_teams(force_refresh)
    if teams.empty:
        return pd.DataFrame()

    frames = []
    for _, row in teams.iterrows():
        roster = _fetch_roster(str(row["team_id"]), force=force_refresh)
        if not roster.empty:
            roster["team_abbreviation"] = row["team_abbreviation"]
            frames.append(roster)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).drop_duplicates("player_id")
    # Keep skill positions only
    skill_pos = ["QB", "RB", "WR", "TE", "K"]
    df = df[df["position"].isin(skill_pos)].reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


# ── Player gamelogs ────────────────────────────────────────────────────────────

def fetch_nfl_gamelog(player_id: str, season: int = NFL_SEASON, force: bool = False) -> pd.DataFrame:
    """
    Fetch game-by-game stats for one NFL player from ESPN.

    Uses ESPN web API:
      GET site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{id}/gamelog

    Response structure:
      names: [col_name, ...]          ← stat column names
      seasonTypes[0].categories[0].events: [{eventId, stats: [val, ...]}, ...]
      events: {eventId: {gameDate, opponent, gameResult, ...}}  ← metadata
    """
    cache = CACHE_DIR / f"nfl_gl_{player_id}_{season}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)

    try:
        data = _req(f"{ESPN_NFL_WEB}/athletes/{player_id}/gamelog")

        names     = data.get("names", [])          # stat column names in order
        ev_meta   = data.get("events", {})          # {eventId: {gameDate, opponent, ...}}
        season_types = data.get("seasonTypes", [])

        if not names or not season_types:
            return pd.DataFrame()

        # Collect rows from all categories (passing, rushing, receiving all in one event list)
        # Use the first category's events as the primary stat source — it contains all stats
        rows = []
        for st in season_types:
            for cat in st.get("categories", []):
                for ev in cat.get("events", []):
                    ev_id = str(ev.get("eventId", ""))
                    stats = ev.get("stats", [])
                    if not stats or not ev_id:
                        continue

                    # Skip duplicate event_ids already added
                    if any(r["game_id"] == ev_id for r in rows):
                        continue

                    row = {"game_id": ev_id}

                    # Attach game metadata from top-level events dict
                    meta = ev_meta.get(ev_id, {})
                    opp  = meta.get("opponent", {})
                    row["game_date"]   = meta.get("gameDate", "")
                    row["opponent"]    = opp.get("abbreviation", "") if isinstance(opp, dict) else ""
                    row["game_result"] = meta.get("gameResult", "")

                    # Map stats by name
                    for name, val in zip(names, stats):
                        row[name] = val
                    rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Rename ESPN stat names → our internal column names
        rename = {
            espn: ours
            for espn, ours in STAT_LABEL_MAP.items()
            if espn in df.columns and espn != ours
        }
        df = df.rename(columns=rename)

        # Convert stat columns to numeric (leave game_id, game_date, opponent, game_result as str)
        skip = {"game_id", "game_date", "opponent", "game_result"}
        for col in df.columns:
            if col not in skip:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.reset_index(drop=True)
        df.to_csv(cache, index=False)
        return df
    except Exception:
        return pd.DataFrame()


# ── Today's NFL slate ──────────────────────────────────────────────────────────

def fetch_nfl_today_slate() -> list:
    """Get today's NFL games from ESPN scoreboard. Returns [] in offseason."""
    try:
        data = _req(f"{ESPN_NFL}/scoreboard")
        games = []
        for event in data.get("events", []):
            comps = event.get("competitions", [{}])[0]
            teams = comps.get("competitors", [])
            if len(teams) < 2:
                continue

            home = next((t for t in teams if t.get("homeAway") == "home"), teams[0])
            away = next((t for t in teams if t.get("homeAway") == "away"), teams[1])

            status = event.get("status", {}).get("type", {})
            state  = status.get("state", "pre")

            games.append({
                "home_abbr": home.get("team", {}).get("abbreviation", "?"),
                "away_abbr": away.get("team", {}).get("abbreviation", "?"),
                "home_pts":  home.get("score", ""),
                "away_pts":  away.get("score", ""),
                "status":    "live" if state == "in" else ("final" if state == "post" else "pre"),
                "game_time": event.get("date", "TBD"),
            })
        return games
    except Exception:
        return []


# ── Core prediction ────────────────────────────────────────────────────────────

def predict_nfl_prop(
    player_id:   str,
    target:      str,
    custom_line: float,
    is_home:     bool = True,
    season:      int  = NFL_SEASON,
) -> dict:
    """
    Predict an NFL prop using rolling game averages + normal distribution.

    Returns dict with predicted_value, over_prob, under_prob, confidence,
    recent_avg_5, recent_avg_10, std_dev, n_games, recent_games.
    """
    fallback = {
        "predicted_value": custom_line,
        "over_prob":       0.50,
        "under_prob":      0.50,
        "confidence":      "Low",
        "recent_avg_5":    custom_line,
        "recent_avg_10":   custom_line,
        "std_dev":         0.0,
        "n_games":         0,
        "recent_games":    [],
    }

    df = fetch_nfl_gamelog(player_id, season)

    if df.empty or target not in df.columns:
        return fallback

    vals = pd.to_numeric(df[target], errors="coerce").dropna()
    if len(vals) == 0:
        return fallback

    avg_5  = round(float(vals.tail(5).mean()),  1)
    avg_10 = round(float(vals.tail(10).mean()), 1)
    std    = float(vals.std()) if len(vals) >= 3 else max(float(vals.mean()) * 0.28, 0.5)

    # Weighted blend: 65% last-5, 35% last-10 (more weight on recent form)
    predicted = round(avg_5 * 0.65 + avg_10 * 0.35, 1)
    if is_home:
        predicted = round(predicted * 1.02, 1)

    # Over probability via normal CDF (math.erf, no scipy needed)
    if std > 0:
        z = (custom_line - predicted) / std
        over_prob = round(0.5 * (1 - math.erf(z / math.sqrt(2))), 3)
    else:
        over_prob = 0.65 if predicted > custom_line else 0.35

    over_prob = max(0.05, min(0.95, over_prob))

    p_max = max(over_prob, 1 - over_prob)
    confidence = (
        "High"   if p_max >= 0.70 and len(vals) >= 5 else
        "Medium" if p_max >= 0.58 else
        "Low"
    )

    return {
        "predicted_value": predicted,
        "over_prob":       round(over_prob, 3),
        "under_prob":      round(1 - over_prob, 3),
        "confidence":      confidence,
        "recent_avg_5":    avg_5,
        "recent_avg_10":   avg_10,
        "std_dev":         round(std, 1),
        "n_games":         len(vals),
        "recent_games":    vals.tail(10).tolist(),
    }


# ── Utility ────────────────────────────────────────────────────────────────────

def get_nfl_player_headshot(player_id: str, players_df: pd.DataFrame) -> str:
    """Return headshot URL for an NFL player from the players dataframe."""
    if players_df is None or players_df.empty:
        return ""
    match = players_df[players_df["player_id"].astype(str) == str(player_id)]
    if match.empty:
        return ""
    url = match.iloc[0].get("headshot_url", "")
    return str(url) if url and str(url) not in ("nan", "None", "") else ""


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching NFL teams...")
    teams = fetch_nfl_teams(force_refresh=True)
    print(f"  {len(teams)} teams: {', '.join(teams['team_abbreviation'].tolist()[:8])}...")

    print("\nFetching all NFL skill-position players (32 rosters)...")
    players = fetch_all_nfl_players(force_refresh=True)
    print(f"  {len(players)} players")
    if not players.empty:
        print(f"  Positions: {players['position'].value_counts().to_dict()}")

    # Test gamelog with Patrick Mahomes (ESPN ID: 3139477)
    test_id = "3139477"
    print(f"\nFetching 2025 gamelog for Mahomes ({test_id})...")
    gl = fetch_nfl_gamelog(test_id, season=2025, force=True)
    if not gl.empty:
        print(f"  {len(gl)} games, columns: {list(gl.columns)}")
        pred = predict_nfl_prop(test_id, "pass_yds", 250.5)
        print(f"  Pass Yds prediction: {pred['predicted_value']} | Over 250.5: {pred['over_prob']*100:.1f}% | {pred['confidence']}")
    else:
        print("  No gamelog data returned")
