"""
linebreaker/data/fetch_lines.py

Fetches sportsbook betting lines (spreads, totals, player props) from ESPN's
free public API and optionally The Odds API (if ODDS_API_KEY env var is set).
No API key required for ESPN endpoints.

Endpoints used:
- NBA scoreboard: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
  → Contains today's games with embedded odds (spread, overUnder)
- ESPN odds items: https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{event_id}/competitions/{event_id}/odds/{provider_id}/items
  → Contains player prop lines per provider

Usage:
    from data.fetch_lines import fetch_today_odds, fetch_player_props, compute_edge
    odds = fetch_today_odds()
    props = fetch_player_props("LeBron James")
    edge = compute_edge(predicted=28.3, line=26.5, target="pts")
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import date, datetime

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_ODDS_ITEMS = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
    "/events/{event_id}/competitions/{event_id}/odds/{provider_id}/items"
)
THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"

HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 10

# Cache TTL in seconds
ODDS_CACHE_TTL   = 300   # 5 minutes for odds
PROPS_CACHE_TTL  = 300   # 5 minutes for props

# In-memory cache: {key: (timestamp, data)}
_mem_cache: dict = {}


# ── Default lines used as fallbacks ───────────────────────────────────────────

DEFAULT_LINES = {
    "pts":          22.5,
    "reb":           6.5,
    "ast":           5.5,
    "stl":           1.5,
    "blk":           1.5,
    "tov":           2.5,
    "fg3m":          2.5,
    "fg3a":          6.5,
    "fga":          14.5,
    "pts_reb_ast":  34.5,
    "pts_reb":      29.5,
    "pts_ast":      29.5,
    "reb_ast":      11.5,
    "blk_stl":       2.5,
    "double_double": 0.5,
    "triple_double": 0.5,
}

# Map ESPN prop type labels to our target keys (best-effort)
ESPN_PROP_MAP = {
    "points":              "pts",
    "total points":        "pts",
    "rebounds":            "reb",
    "total rebounds":      "reb",
    "assists":             "ast",
    "total assists":       "ast",
    "steals":              "stl",
    "blocks":              "blk",
    "turnovers":           "tov",
    "three point field goals made": "fg3m",
    "threes":              "fg3m",
    "three-pointers made": "fg3m",
    "field goals attempted": "fga",
    "pts + reb + ast":     "pts_reb_ast",
    "points + rebounds + assists": "pts_reb_ast",
    "pts + reb":           "pts_reb",
    "points + rebounds":   "pts_reb",
    "pts + ast":           "pts_ast",
    "points + assists":    "pts_ast",
    "reb + ast":           "reb_ast",
    "rebounds + assists":  "reb_ast",
    "blk + stl":           "blk_stl",
    "blocks + steals":     "blk_stl",
}

# Map The Odds API market keys → our target keys
ODDS_API_MARKET_MAP = {
    "player_points":       "pts",
    "player_rebounds":     "reb",
    "player_assists":      "ast",
    "player_steals":       "stl",
    "player_blocks":       "blk",
    "player_turnovers":    "tov",
    "player_threes":       "fg3m",
    "player_field_goals":  "fga",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get(url: str, params: dict = None) -> dict:
    """HTTP GET with timeout and graceful error handling."""
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def _cache_get(key: str, ttl: int):
    """Return cached value if still fresh, else None."""
    entry = _mem_cache.get(key)
    if entry is None:
        return None
    ts, data = entry
    if time.time() - ts < ttl:
        return data
    return None


def _cache_set(key: str, data):
    _mem_cache[key] = (time.time(), data)


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents-agnostic normalization for fuzzy matching."""
    return name.lower().strip()


def _name_match(a: str, b: str) -> bool:
    """
    True if player names are likely the same person.
    Checks exact match, last-name match, and substring match.
    """
    a, b = _normalize_name(a), _normalize_name(b)
    if a == b:
        return True
    a_last = a.split()[-1] if a.split() else a
    b_last = b.split()[-1] if b.split() else b
    if a_last == b_last and len(a_last) > 3:
        return True
    return a in b or b in a


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_today_odds() -> dict:
    """
    Fetch today's NBA game odds from ESPN scoreboard.

    Returns dict keyed by team abbreviation:
        {
            "LAL": {"spread": -4.5, "total": 228.5, "game_id": "401773456"},
            "GSW": {"spread":  4.5, "total": 228.5, "game_id": "401773456"},
            ...
        }
    Falls back to {} if unavailable.
    Cached for 5 minutes.
    """
    cache_key = f"today_odds_{date.today().isoformat()}"
    cached = _cache_get(cache_key, ODDS_CACHE_TTL)
    if cached is not None:
        return cached

    today_str = date.today().strftime("%Y%m%d")
    board = _get(ESPN_SCOREBOARD, params={"dates": today_str, "lang": "en", "region": "us"})
    if not board:
        board = _get(ESPN_SCOREBOARD)

    result: dict = {}
    events = board.get("events", [])

    for event in events:
        event_id = event.get("id", "")
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        comp = competitions[0]

        # Pull teams
        competitors = comp.get("competitors", [])
        team_abbrs: dict = {}   # home_away -> abbr
        for c in competitors:
            abbr = (c.get("team", {}) or {}).get("abbreviation", "")
            home_away = c.get("homeAway", "")
            if abbr:
                team_abbrs[home_away] = abbr

        # Pull odds — ESPN embeds a list of providers in comp["odds"]
        odds_list = comp.get("odds", [])
        if not odds_list:
            continue

        # Use first available provider
        for odds_entry in odds_list:
            spread_raw    = odds_entry.get("spread")
            over_under    = odds_entry.get("overUnder")
            details       = odds_entry.get("details", "")   # e.g. "LAL -4.5"
            home_team_odd = odds_entry.get("homeTeamOdds", {}) or {}
            away_team_odd = odds_entry.get("awayTeamOdds", {}) or {}

            # Parse spread per team from details string or homeTeamOdds
            home_spread = home_team_odd.get("spreadOdds") or home_team_odd.get("pointSpread", {}).get("current")
            away_spread = away_team_odd.get("spreadOdds") or away_team_odd.get("pointSpread", {}).get("current")

            # Fallback: try numeric spread field
            if home_spread is None and spread_raw is not None:
                try:
                    home_spread = float(spread_raw)
                    away_spread = -float(spread_raw)
                except (TypeError, ValueError):
                    pass

            total = None
            if over_under is not None:
                try:
                    total = float(over_under)
                except (TypeError, ValueError):
                    pass

            home_abbr = team_abbrs.get("home", "")
            away_abbr = team_abbrs.get("away", "")

            if home_abbr:
                result[home_abbr] = {
                    "spread":   float(home_spread) if home_spread is not None else None,
                    "total":    total,
                    "game_id":  event_id,
                    "is_home":  True,
                }
            if away_abbr:
                result[away_abbr] = {
                    "spread":   float(away_spread) if away_spread is not None else None,
                    "total":    total,
                    "game_id":  event_id,
                    "is_home":  False,
                }
            break  # use first provider only

    _cache_set(cache_key, result)
    return result


def _fetch_espn_props(event_id: str, player_name: str) -> dict:
    """
    ESPN's /odds/{provider_id}/items endpoint is not publicly accessible (returns 404).
    Player props require The Odds API or another source — return {} immediately.
    """
    return {}


def _fetch_odds_api_props(player_name: str) -> dict:
    """
    Fetch player props from The Odds API (https://the-odds-api.com).
    Only used if ODDS_API_KEY environment variable is set.
    Returns {target_key: line_value} or {}.
    """
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        return {}

    props: dict = {}
    markets = ",".join(ODDS_API_MARKET_MAP.keys())

    # First get today's events
    try:
        events_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
        resp = requests.get(
            events_url,
            params={
                "apiKey":     api_key,
                "regions":    "us",
                "markets":    "h2h",
                "dateFormat": "iso",
            },
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        print(f"  The Odds API events failed: {e}")
        return {}

    today_str = date.today().isoformat()
    today_events = [e for e in events if e.get("commence_time", "").startswith(today_str)]

    for event in today_events:
        event_id = event.get("id", "")
        try:
            props_resp = requests.get(
                f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds",
                params={
                    "apiKey":     api_key,
                    "regions":    "us",
                    "markets":    markets,
                    "dateFormat": "iso",
                    "oddsFormat": "american",
                },
                headers=HEADERS,
                timeout=TIMEOUT,
            )
            props_resp.raise_for_status()
            event_data = props_resp.json()
        except Exception:
            continue

        for book in event_data.get("bookmakers", []):
            for market in book.get("markets", []):
                mkey = market.get("key", "")
                target = ODDS_API_MARKET_MAP.get(mkey)
                if not target:
                    continue
                for outcome in market.get("outcomes", []):
                    oname = outcome.get("description", outcome.get("name", ""))
                    if not _name_match(player_name, oname):
                        continue
                    point = outcome.get("point")
                    if point is not None and target not in props:
                        try:
                            props[target] = float(point)
                        except (TypeError, ValueError):
                            pass
            if props:
                break  # stop after first bookmaker with data

        if props:
            break

    return props


def fetch_player_props(player_name: str, player_id: int = None) -> dict:
    """
    Try to fetch prop lines for a specific player.

    Strategy:
    1. Check in-memory cache (5-minute TTL).
    2. Get today's scoreboard to find which game this player is in.
    3. Try ESPN odds/items endpoint for each game.
    4. If ODDS_API_KEY is set, also try The Odds API.
    5. Fall back to {} if nothing found.

    Returns {target_key: line_value}, e.g. {"pts": 24.5, "reb": 7.5}
    """
    cache_key = f"props_{_normalize_name(player_name)}_{date.today().isoformat()}"
    cached = _cache_get(cache_key, PROPS_CACHE_TTL)
    if cached is not None:
        return cached

    props: dict = {}

    # Try ESPN props via today's scoreboard event IDs
    today_str = date.today().strftime("%Y%m%d")
    board = _get(ESPN_SCOREBOARD, params={"dates": today_str, "lang": "en", "region": "us"})
    if not board:
        board = _get(ESPN_SCOREBOARD)

    events = board.get("events", [])
    for event in events:
        event_id = event.get("id", "")
        if not event_id:
            continue
        espn_props = _fetch_espn_props(event_id, player_name)
        if espn_props:
            props.update(espn_props)
            break

    # If ESPN gave nothing, try The Odds API
    if not props:
        odds_api_props = _fetch_odds_api_props(player_name)
        if odds_api_props:
            props.update(odds_api_props)

    _cache_set(cache_key, props)
    return props


def compute_edge(predicted: float, line: float, target: str) -> dict:
    """
    Given a model prediction and sportsbook line, compute edge metrics.

    Parameters
    ----------
    predicted : float
        Model's predicted stat value.
    line : float
        Sportsbook line (over/under point).
    target : str
        Target key (e.g. "pts", "reb").

    Returns
    -------
    dict with keys:
        line        : float — the line used
        edge_abs    : float — predicted - line
        edge_pct    : float — edge_abs / line * 100
        direction   : str   — "OVER" or "UNDER"
        has_live_line: bool — True if line came from ESPN/API, False if default
    """
    default_line = DEFAULT_LINES.get(target, line)
    has_live_line = abs(line - default_line) > 0.01  # non-default → live

    edge_abs = round(predicted - line, 2)
    edge_pct = round(edge_abs / line * 100, 1) if line != 0 else 0.0
    direction = "OVER" if edge_abs >= 0 else "UNDER"

    return {
        "line":          line,
        "edge_abs":      edge_abs,
        "edge_pct":      edge_pct,
        "direction":     direction,
        "has_live_line": has_live_line,
    }


def get_line_for_target(player_name: str, target: str,
                         player_id: int = None) -> tuple:
    """
    Convenience: return (line_value, has_live_line) for a player/target combo.
    Only attempts live fetch when ODDS_API_KEY is set; otherwise returns (None, False)
    so the caller falls back to the default threshold.
    """
    # Without an API key ESPN props aren't available — skip the network call
    if not os.environ.get("ODDS_API_KEY", ""):
        return None, False
    props = fetch_player_props(player_name, player_id)
    if target in props:
        return props[target], True
    return None, False


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Today's NBA Odds ===")
    odds = fetch_today_odds()
    if odds:
        for team, info in sorted(odds.items()):
            spread = info.get("spread")
            total  = info.get("total")
            print(f"  {team}: spread={spread}, total={total}")
    else:
        print("  No odds available today.")

    print("\n=== Player Props (LeBron James) ===")
    props = fetch_player_props("LeBron James")
    if props:
        for tgt, val in props.items():
            print(f"  {tgt}: {val}")
    else:
        print("  No props found (ESPN may not have today's lines yet).")

    print("\n=== Edge Calculation ===")
    edge = compute_edge(predicted=28.3, line=26.5, target="pts")
    for k, v in edge.items():
        print(f"  {k}: {v}")
