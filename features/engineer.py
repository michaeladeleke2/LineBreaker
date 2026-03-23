"""
linebreaker/features/engineer.py

Builds the full feature matrix for all stat targets.

Stat targets (16 total):
  Regression:    pts, reb, ast, stl, blk, tov, fg3m, fg3a, fga,
                 pts_reb_ast, pts_reb, pts_ast, reb_ast, blk_stl
  Classification: over/under per target, double_double, triple_double
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.fetch_data import (
    fetch_all_gamelogs,
    fetch_all_team_defense,
)

# ── Stat definitions ──────────────────────────────────────────────────────────

BASE_STATS = [
    "pts", "reb", "ast", "stl", "blk", "tov",
    "fg3m", "fg3a", "fga", "min",
    "fg_pct", "fg3_pct", "fta", "ft_pct", "usg_pct",
]

COMBO_TARGETS = {
    "pts_reb_ast": ["pts", "reb", "ast"],
    "pts_reb":     ["pts", "reb"],
    "pts_ast":     ["pts", "ast"],
    "reb_ast":     ["reb", "ast"],
    "blk_stl":     ["blk", "stl"],
}

ALL_TARGETS = (
    ["pts", "reb", "ast", "stl", "blk", "tov", "fg3m", "fg3a", "fga"]
    + list(COMBO_TARGETS.keys())
)

ROLLING_STAT_COLS = [
    "pts", "reb", "ast", "stl", "blk", "tov",
    "fg3m", "fg3a", "fga", "min",
    "fg_pct", "fg3_pct", "fta", "ft_pct", "usg_pct",
    "pts_reb_ast", "pts_reb", "pts_ast", "reb_ast", "blk_stl",
    "plus_minus",
]

ROLLING_WINDOWS = [5, 10]

DEFAULT_THRESHOLDS = {
    "pts":          20,
    "reb":           5,
    "ast":           5,
    "stl":           1,
    "blk":           1,
    "tov":           3,
    "fg3m":          2,
    "fg3a":          5,
    "fga":          15,
    "pts_reb_ast":  30,
    "pts_reb":      25,
    "pts_ast":      25,
    "reb_ast":      10,
    "blk_stl":       2,
    "double_double": 1,
    "triple_double": 1,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _parse_minutes(min_series: pd.Series) -> pd.Series:
    def convert(val):
        try:
            if isinstance(val, str) and ":" in val:
                m, s = val.split(":")
                return float(m) + float(s) / 60
            return float(val)
        except (ValueError, TypeError):
            return 0.0
    return min_series.apply(convert)


# ── Feature builders ──────────────────────────────────────────────────────────

def _add_combo_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, parts in COMBO_TARGETS.items():
        available = [p for p in parts if p in df.columns]
        if available:
            df[name] = df[available].sum(axis=1)
    return df


def _add_dd_td_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dd_cols = [c for c in ["pts", "reb", "ast", "stl", "blk"] if c in df.columns]
    if dd_cols:
        tens = (df[dd_cols] >= 10).sum(axis=1)
        df["double_double"] = (tens >= 2).astype(int)
        df["triple_double"] = (tens >= 3).astype(int)
    else:
        df["double_double"] = 0
        df["triple_double"] = 0
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby("player_id")
    for col in ROLLING_STAT_COLS:
        if col not in df.columns:
            continue
        for w in ROLLING_WINDOWS:
            feat = f"{col}_roll{w}"
            df[feat] = (
                grp[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
    return df


def _add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"])

    df["rest_days"] = (
        df.groupby("player_id")["game_date"]
        .diff().dt.days.fillna(3).clip(upper=14)
    )
    df["is_back_to_back"] = (df["rest_days"] <= 1).astype(int)

    if "matchup" in df.columns:
        df["is_home"] = df["matchup"].str.contains(r" vs\. ", na=False).astype(int)
    else:
        df["is_home"] = 0

    df["game_number"] = df.groupby(["player_id", "season"]).cumcount() + 1
    return df


def _add_pace_and_script_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby("player_id")

    # 1. Rolling minutes std — consistency of playing time (high std = unreliable minutes)
    df["min_std_l10"] = (
        grp["min"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).std().fillna(5.0))
    )

    # 2. Plus/minus rolling — game script proxy (negative = team losing = fewer late-game mins)
    if "plus_minus" in df.columns:
        df["plus_minus_roll5"] = (
            grp["plus_minus"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(0.0))
        )
        # Blowout rate — pct of last 10 games where player's team was losing/winning big
        df["blowout_rate_l10"] = (
            grp["plus_minus"].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3)
                .apply(lambda v: (abs(v) > 15).mean(), raw=True).fillna(0.3)
            )
        )
    else:
        df["plus_minus_roll5"] = 0.0
        df["blowout_rate_l10"] = 0.3

    # 3. Game pace proxy — FGA is the best available pace signal in player gamelogs
    #    High FGA rolling = fast-paced games = more counting stat opportunities
    if "fga" in df.columns:
        df["pace_proxy_l5"] = (
            grp["fga"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(12.0))
        )
    else:
        df["pace_proxy_l5"] = 12.0

    # 4. Usage consistency — fg3a / fga ratio rolling (3PT-heavy player vs slasher)
    if "fg3a" in df.columns and "fga" in df.columns:
        df["three_pt_rate_l5"] = (
            grp["fg3a"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(3.0)) /
            df["pace_proxy_l5"].clip(lower=1.0)
        )
    else:
        df["three_pt_rate_l5"] = 0.25

    return df


def _add_opponent_defense(df: pd.DataFrame, defense: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "matchup" in df.columns:
        df["opp_abbrev"] = (
            df["matchup"]
            .str.extract(r"(?:vs\.|@)\s+(\w+)", expand=False)
            .str.upper()
        )
    else:
        df["opp_abbrev"] = None

    if "team_abbreviation" in df.columns and "team_id" in df.columns:
        abbrev_map = (
            df[["team_abbreviation", "team_id"]]
            .drop_duplicates()
            .set_index("team_abbreviation")["team_id"]
            .to_dict()
        )
        df["opp_team_id"] = df["opp_abbrev"].map(abbrev_map)
    else:
        df["opp_team_id"] = np.nan

    if "def_rating" in defense.columns:
        def_lookup = defense.set_index(["team_id", "season"])["def_rating"]
        df["def_rating"] = df.apply(
            lambda r: def_lookup.get((r["opp_team_id"], r["season"]), np.nan)
            if pd.notna(r.get("opp_team_id")) else np.nan,
            axis=1,
        )
    else:
        df["def_rating"] = np.nan

    league_avg = df["def_rating"].median()
    df["def_rating"] = df["def_rating"].fillna(league_avg if pd.notna(league_avg) else 112.0)
    return df


# ── Feature column accessor ───────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    rolling = [
        f"{col}_roll{w}"
        for col in ROLLING_STAT_COLS
        for w in ROLLING_WINDOWS
        if f"{col}_roll{w}" in df.columns
    ]
    context = [
        c for c in ["rest_days", "is_back_to_back", "is_home", "game_number", "def_rating"]
        if c in df.columns
    ]
    # plus_minus_roll5 is already captured via ROLLING_STAT_COLS; omit it here
    pace_cols = [
        c for c in ["min_std_l10", "blowout_rate_l10",
                     "pace_proxy_l5", "three_pt_rate_l5"]
        if c in df.columns
    ]
    # Deduplicate while preserving order — duplicate names cause XGBoost to
    # receive a DataFrame instead of a Series for that column
    seen = set()
    out  = []
    for c in rolling + context + pace_cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────

# Module-level in-memory cache — survives across predict() calls in same process
_FM_CACHE = None

def _get_cached_fm(force_refresh: bool = False):
    """Return cached feature matrix. Loads from disk once, then serves from memory."""
    global _FM_CACHE
    if not force_refresh and _FM_CACHE is not None and len(_FM_CACHE) > 0:
        return _FM_CACHE
    _FM_CACHE = build_feature_matrix(force_refresh=force_refresh)
    return _FM_CACHE

def build_feature_matrix(force_refresh: bool = False) -> pd.DataFrame:
    cache_path = ROOT / "data" / "cache" / "feature_matrix.csv"

    if cache_path.exists() and not force_refresh:
        print("Loading cached feature matrix...")
        return pd.read_csv(cache_path)

    print("Building feature matrix from raw game logs...")

    logs    = fetch_all_gamelogs()
    defense = fetch_all_team_defense()

    logs["min"] = _parse_minutes(logs["min"])
    for col in BASE_STATS:
        if col in logs.columns:
            logs[col] = _safe_float(logs[col])

    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs = logs.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    logs = _add_combo_columns(logs)
    logs = _add_dd_td_labels(logs)
    logs = _add_rolling_features(logs)
    logs = _add_context_features(logs)
    logs = _add_pace_and_script_features(logs)
    logs = _add_opponent_defense(logs, defense)

    logs = logs.dropna(subset=["pts"]).reset_index(drop=True)

    logs.to_csv(cache_path, index=False)
    print(f"Feature matrix saved — {len(logs):,} rows, {len(logs.columns)} columns")
    return logs


# ── Inference row builder ─────────────────────────────────────────────────────

def build_player_features(
    player_id: int,
    opponent_team_id: int,
    is_home: bool,
    rest_days: int,
    season: str = None,
    n_recent: int = 10,
    fm: pd.DataFrame = None,
) -> pd.DataFrame:
    from data.fetch_data import CURRENT_SEASON as _CUR
    if season is None:
        season = _CUR
    if fm is None:
        fm = _get_cached_fm()
    player_games = fm[fm["player_id"] == player_id].sort_values("game_date")

    if player_games.empty:
        raise ValueError(f"No game data found for player_id={player_id}")

    recent = player_games.tail(n_recent)
    row    = {}

    for col in ROLLING_STAT_COLS:
        if col not in recent.columns:
            continue
        for w in ROLLING_WINDOWS:
            feat = f"{col}_roll{w}"
            tail = recent[col].dropna().tail(w)
            row[feat] = float(tail.mean()) if len(tail) > 0 else 0.0

    row["rest_days"]       = float(min(rest_days, 14))
    row["is_back_to_back"] = int(rest_days <= 1)
    row["is_home"]         = int(is_home)
    row["game_number"]     = float(
        recent["game_number"].iloc[-1] + 1
        if "game_number" in recent.columns else 50.0
    )

    defense = fetch_all_team_defense()
    opp_def = defense[
        (defense["team_id"] == opponent_team_id) &
        (defense["season"] == season)
    ]
    if opp_def.empty:
        opp_def = defense[defense["team_id"] == opponent_team_id]
    row["def_rating"] = float(opp_def["def_rating"].values[0]) if not opp_def.empty else 112.0

    feat_df   = pd.DataFrame([row])
    feat_cols = get_feature_cols(fm)
    for c in feat_cols:
        if c not in feat_df.columns:
            feat_df[c] = 0.0
    feat_df = feat_df[feat_cols]

    return feat_df


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fm        = build_feature_matrix(force_refresh=True)
    feat_cols = get_feature_cols(fm)

    print(f"\nShape: {fm.shape}")
    print(f"Feature columns ({len(feat_cols)}): {feat_cols}")
    print(f"\nTargets present:")
    for t in ALL_TARGETS + ["double_double", "triple_double"]:
        if t in fm.columns:
            non_null = fm[t].notna().sum()
            print(f"  {t:20s}  {non_null:>7,} non-null rows")
        else:
            print(f"  {t:20s}  MISSING")