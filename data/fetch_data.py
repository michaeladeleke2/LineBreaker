"""
linebreaker/data/fetch_data.py

Fetches NBA player game logs and team stats from nba_api.
Results are cached locally as CSVs to avoid repeated API calls.
"""

import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from nba_api.stats.endpoints import (
    playergamelogs,
    leaguedashteamstats,
    commonallplayers,
    scoreboardv2,
    leaguestandings,
)
from datetime import date
from nba_api.stats.static import players

# ── Cache directory ───────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Auto-detect current NBA season ───────────────────────────────────────────
def _current_nba_season() -> str:
    from datetime import date
    today = date.today()
    y, m  = today.year, today.month
    if m >= 10:
        return f"{y}-{str(y+1)[-2:]}"
    else:
        return f"{y-1}-{str(y)[-2:]}"

CURRENT_SEASON   = _current_nba_season()
TRAINING_SEASONS = [
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]
# If current season isn't already in training, add it for completeness
# but keep training separate so we can refresh current independently
ALL_SEASONS = TRAINING_SEASONS + ([CURRENT_SEASON] if CURRENT_SEASON not in TRAINING_SEASONS else [])

REQUEST_DELAY = 0.7
MAX_RETRIES   = 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.csv"


def _load_or_fetch(name: str, fetch_fn, force_refresh: bool = False) -> pd.DataFrame:
    path = _cache_path(name)
    if path.exists() and not force_refresh:
        return pd.read_csv(path)
    df = fetch_fn()
    df.to_csv(path, index=False)
    return df


def _retry(fn, retries: int = MAX_RETRIES, delay: float = REQUEST_DELAY):
    for attempt in range(retries):
        try:
            time.sleep(delay)
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = delay * (2 ** attempt)
            print(f"  Retry {attempt + 1}/{retries} — {e}. Waiting {wait:.1f}s...")
            time.sleep(wait)


# ── Player list ───────────────────────────────────────────────────────────────

def get_all_players(force_refresh: bool = False) -> pd.DataFrame:
    def fetch():
        raw = players.get_players()
        return pd.DataFrame(raw)[["id", "full_name", "is_active"]]
    return _load_or_fetch("all_players", fetch, force_refresh)


def get_active_players() -> pd.DataFrame:
    df = get_all_players()
    return df[df["is_active"] == True].reset_index(drop=True)


def search_players(query: str) -> pd.DataFrame:
    df = get_all_players()
    return df[df["full_name"].str.contains(query, case=False, na=False)].reset_index(drop=True)


# ── Full-game logs ────────────────────────────────────────────────────────────

def fetch_season_gamelogs(season: str, force_refresh: bool = False) -> pd.DataFrame:
    cache_name = f"gamelogs_{season.replace('-', '_')}"

    def fetch():
        print(f"  Fetching game logs for {season}...")
        data = _retry(
            lambda: playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable="Regular Season",
            ).get_data_frames()[0]
        )
        return data

    df = _load_or_fetch(cache_name, fetch, force_refresh)
    df.columns = [c.lower() for c in df.columns]
    return df


def fetch_all_gamelogs(force_refresh: bool = False) -> pd.DataFrame:
    frames = []
    for season in tqdm(ALL_SEASONS, desc="Loading seasons"):
        df = fetch_season_gamelogs(season, force_refresh)
        df["season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def get_player_gamelogs(player_id: int, force_refresh: bool = False) -> pd.DataFrame:
    all_logs = fetch_all_gamelogs(force_refresh)
    df = all_logs[all_logs["player_id"] == player_id].copy()
    return df.sort_values("game_date").reset_index(drop=True)


# ── Team defensive stats ──────────────────────────────────────────────────────

def fetch_team_defense(season: str, force_refresh: bool = False) -> pd.DataFrame:
    cache_name = f"team_defense_{season.replace('-', '_')}"

    def fetch():
        print(f"  Fetching team defense for {season}...")
        data = _retry(
            lambda: leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Defense",
                per_mode_detailed="PerGame",
            ).get_data_frames()[0]
        )
        return data

    df = _load_or_fetch(cache_name, fetch, force_refresh)
    df.columns = [c.lower() for c in df.columns]
    keep = ["team_id", "team_name", "def_rating", "w_pct"]
    return df[[c for c in keep if c in df.columns]]


def fetch_all_team_defense(force_refresh: bool = False) -> pd.DataFrame:
    frames = []
    for season in tqdm(ALL_SEASONS, desc="Loading team defense"):
        df = fetch_team_defense(season, force_refresh)
        df["season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── Daily slate ──────────────────────────────────────────────────────────────

def fetch_today_slate() -> list:
    """
    Fetch today's NBA games. Returns one entry per unique game_id.
    """
    try:
        today = date.today().strftime("%Y-%m-%d")
        board = _retry(
            lambda: scoreboardv2.ScoreboardV2(
                game_date=today,
                league_id="00",
                day_offset=0,
            )
        )
        frames = board.get_data_frames()

        game_header = frames[0].copy()
        line_score  = frames[1].copy() if len(frames) > 1 else pd.DataFrame()

        game_header.columns = [c.lower() for c in game_header.columns]
        if not line_score.empty:
            line_score.columns = [c.lower() for c in line_score.columns]

        seen_ids = set()
        games    = []

        for _, row in game_header.iterrows():
            game_id = str(row.get("game_id", ""))

            # Deduplicate — GameHeader has one row per game
            if game_id in seen_ids:
                continue
            seen_ids.add(game_id)

            home_abbr = away_abbr = ""
            home_record = away_record = ""
            home_pts = away_pts = None

            if not line_score.empty and "game_id" in line_score.columns:
                teams = line_score[line_score["game_id"].astype(str) == game_id]
                home_id = str(row.get("home_team_id", ""))
                for _, t in teams.iterrows():
                    abbr = str(t.get("team_abbreviation", ""))
                    wins = t.get("team_wins_losses", "")
                    pts  = t.get("pts", None)
                    if str(t.get("team_id", "")) == home_id:
                        home_abbr   = abbr
                        home_record = str(wins) if wins else ""
                        home_pts    = pts
                    else:
                        away_abbr   = abbr
                        away_record = str(wins) if wins else ""
                        away_pts    = pts

            status_num = int(row.get("game_status_id", 1))
            status_txt = str(row.get("game_status_text", "")).strip()
            status     = {1: "scheduled", 2: "live", 3: "final"}.get(status_num, "scheduled")

            games.append({
                "game_id":     game_id,
                "home_abbr":   home_abbr,
                "away_abbr":   away_abbr,
                "home_record": home_record,
                "away_record": away_record,
                "home_pts":    home_pts,
                "away_pts":    away_pts,
                "status":      status,
                "game_time":   status_txt,
            })

        return games

    except Exception as e:
        print(f"  Slate fetch failed: {e}")
        return []


# ── Team standings (for records) ──────────────────────────────────────────────

def fetch_team_records(season: str = "2024-25") -> dict:
    """Returns dict of team_abbr -> 'W-L' string."""
    try:
        standings = _retry(
            lambda: leaguestandings.LeagueStandings(
                season=season,
                season_type="Regular Season",
            ).get_data_frames()[0]
        )
        standings.columns = [c.lower() for c in standings.columns]
        result = {}
        for _, r in standings.iterrows():
            abbr = r.get("teamslug","") or r.get("team","")
            w    = r.get("wins", r.get("w",""))
            l    = r.get("losses", r.get("l",""))
            if abbr:
                result[str(abbr).upper()] = f"{w}-{l}"
        return result
    except Exception as e:
        print(f"  Records fetch failed: {e}")
        return {}


# ── Enriched player list (with team + position) ──────────────────────────────

def get_enriched_players(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns active players with team abbreviation and position.
    Columns: id, full_name, is_active, team_abbreviation, position
    Cached to all_players_enriched.csv.
    """
    def fetch():
        print("  Fetching enriched player list...")
        data = _retry(
            lambda: commonallplayers.CommonAllPlayers(
                is_only_current_season=1,
                league_id="00",
                season="2024-25",
            ).get_data_frames()[0]
        )
        data.columns = [c.lower() for c in data.columns]

        # Map to clean columns
        col_map = {
            "person_id":        "id",
            "display_first_last": "full_name",
            "team_abbreviation": "team_abbreviation",
            "rosterstatus":      "is_active",
        }
        data = data.rename(columns={k: v for k, v in col_map.items() if k in data.columns})

        # Position lives in a different endpoint — derive from game logs as fallback
        # CommonAllPlayers doesn't reliably have position, so we'll add it as Unknown
        # and let users filter by team only from this source
        if "position" not in data.columns:
            data["position"] = "Unknown"

        keep = ["id", "full_name", "team_abbreviation", "position"]
        available = [c for c in keep if c in data.columns]
        return data[available]

    return _load_or_fetch("all_players_enriched", fetch, force_refresh)


# ── Team ID lookup ────────────────────────────────────────────────────────────

def get_team_list() -> pd.DataFrame:
    all_logs = fetch_all_gamelogs()
    if "team_id" in all_logs.columns and "team_abbreviation" in all_logs.columns:
        return (
            all_logs[["team_id", "team_abbreviation"]]
            .drop_duplicates()
            .sort_values("team_abbreviation")
            .reset_index(drop=True)
        )
    return pd.DataFrame(columns=["team_id", "team_abbreviation"])


# ── Current season refresh ───────────────────────────────────────────────────

def refresh_current_season() -> bool:
    """
    Re-fetches current season game logs and clears all derived caches.
    Returns True if successful.
    """
    try:
        print(f"Refreshing {CURRENT_SEASON} game logs...")
        fetch_season_gamelogs(CURRENT_SEASON, force_refresh=True)

        root = Path(__file__).resolve().parents[1]
        # Clear feature matrix so it rebuilds with fresh data
        for fname in ["feature_matrix.csv"]:
            p = root / "data" / "cache" / fname
            if p.exists():
                p.unlink()
                print(f"Cleared {fname}")

        print("Refresh complete.")
        return True
    except Exception as e:
        print(f"Refresh failed: {e}")
        return False


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Full-game logs ===")
    logs = fetch_all_gamelogs()
    print(f"Rows: {len(logs):,}")

    print("\n=== Team defense ===")
    defense = fetch_all_team_defense()
    print(defense.head(3))