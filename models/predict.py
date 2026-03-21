"""
linebreaker/models/predict.py

Inference layer — loads saved models and generates predictions
for all stat targets for a given player vs. opponent matchup.

Called by app.py. Can also be run directly for CLI testing:
  python models/predict.py --player "LeBron James" --opponent GSW
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from features.engineer import (
    build_feature_matrix,
    build_player_features,
    get_feature_cols,
    ALL_TARGETS,
    DEFAULT_THRESHOLDS,
    _get_cached_fm,
)
from data.fetch_injuries import fetch_injury_report, get_player_injury
from data.fetch_lineups import fetch_all_lineups, get_player_lineup_status
from data.fetch_data import (
    get_all_players,
    get_enriched_players,
    search_players,
    get_player_gamelogs,
    fetch_all_gamelogs,
    fetch_all_team_defense,
    get_team_list,
)

SAVE_DIR = ROOT / "models" / "saved"


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class TargetResult:
    target:           str
    predicted_value:  float
    over_prob:        float
    under_prob:       float
    threshold:        int
    confidence_label: str    # High / Medium / Low
    recent_avg_5:     float
    recent_avg_10:    float
    model_mae:        float
    model_auc:        float


@dataclass
class PredictionResult:
    player_name:  str
    player_id:    int
    opponent:     str
    is_home:      bool
    rest_days:    int
    targets:      dict         # target_name -> TargetResult
    recent_games: pd.DataFrame # last 10 games for form chart
    feature_row:  pd.DataFrame # single-row feature vector
    injury_info:  dict = None  # ESPN injury status if available
    lineup_info:  dict = None  # ESPN lineup/starter status


# ── Model cache (loaded once per session) ─────────────────────────────────────

_models   = {}   # target -> (regressor, classifier)
_metadata = None


def load_models(force: bool = False):
    global _models, _metadata
    if _metadata is not None and not force:
        return _models, _metadata

    meta_path = SAVE_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata.json found at {meta_path}. "
            "Run python models/train.py first."
        )

    with open(meta_path) as f:
        _metadata = json.load(f)

    _models = {}
    for target in _metadata.get("targets", {}):
        reg_path = SAVE_DIR / f"{target}_regressor.pkl"
        cls_path = SAVE_DIR / f"{target}_classifier.pkl"
        if reg_path.exists() and cls_path.exists():
            _models[target] = (
                joblib.load(reg_path),
                joblib.load(cls_path),
            )

    print(f"Loaded {len(_models)} target models.")
    return _models, _metadata


def get_available_targets() -> list:
    """Return list of targets that have trained models saved."""
    _, meta = load_models()
    return list(meta.get("targets", {}).keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence_label(over_prob: float) -> str:
    p = max(over_prob, 1 - over_prob)
    if p >= 0.75: return "High"
    if p >= 0.60: return "Medium"
    return "Low"


def get_recent_form(player_id: int, n: int = 10, preloaded_gamelogs: pd.DataFrame = None) -> pd.DataFrame:
    """Return last n games with key stats for the form chart."""
    if preloaded_gamelogs is not None:
        logs = preloaded_gamelogs[preloaded_gamelogs["player_id"] == player_id].copy()
    else:
        logs = get_player_gamelogs(player_id)
    keep = [c for c in ["game_date", "pts", "reb", "ast", "stl", "blk",
                         "fg3m", "min", "matchup", "wl"] if c in logs.columns]
    recent = logs[keep].tail(n).copy()
    recent["game_date"] = pd.to_datetime(recent["game_date"])
    return recent.sort_values("game_date").reset_index(drop=True)


def _get_recent_avg(recent: pd.DataFrame, col: str, n: int) -> float:
    """Get rolling average of a column over last n games."""
    if col not in recent.columns:
        # Try mapping combo targets to their component display stat
        col_map = {
            "pts_reb_ast": None, "pts_reb": None,
            "pts_ast": None, "reb_ast": None, "blk_stl": None,
        }
        return 0.0
    vals = pd.to_numeric(recent[col], errors="coerce").dropna().tail(n)
    return round(float(vals.mean()), 1) if len(vals) > 0 else 0.0


def _compute_recent_combo(recent: pd.DataFrame, target: str, n: int) -> float:
    """Compute recent average for combo targets from component stats."""
    from features.engineer import COMBO_TARGETS
    if target in COMBO_TARGETS:
        parts = COMBO_TARGETS[target]
        available = [p for p in parts if p in recent.columns]
        if not available:
            return 0.0
        vals = recent[available].apply(pd.to_numeric, errors="coerce").sum(axis=1).tail(n)
        return round(float(vals.mean()), 1)
    return _get_recent_avg(recent, target, n)


# ── Core prediction ───────────────────────────────────────────────────────────

def predict(
    player_id:            int,
    opponent_team_id:     int,
    opponent_name:        str,
    is_home:              bool  = True,
    rest_days:            int   = 2,
    season:               str   = None,
    targets:              list  = None,
    preloaded_injury_df         = None,
    preloaded_lineup_df         = None,
    preloaded_fm:               pd.DataFrame = None,
    preloaded_players_df:       pd.DataFrame = None,
    preloaded_gamelogs:         pd.DataFrame = None,
) -> PredictionResult:
    """
    Generate predictions across all (or specified) stat targets
    for one player vs. one opponent.
    """
    models, metadata = load_models()

    if targets is None:
        targets = list(models.keys())

    # Build feature vector once — shared across all targets
    from data.fetch_data import CURRENT_SEASON as _CUR
    if season is None:
        season = _CUR
    feat_row = build_player_features(
        player_id        = player_id,
        opponent_team_id = opponent_team_id,
        is_home          = is_home,
        rest_days        = rest_days,
        season           = season,
        fm               = preloaded_fm,
    )

    feature_cols = metadata["feature_cols"]
    for c in feature_cols:
        if c not in feat_row.columns:
            feat_row[c] = 0.0
    feat_row = feat_row[feature_cols]

    # Recent form
    recent_games = get_recent_form(player_id, n=10, preloaded_gamelogs=preloaded_gamelogs)

    # Player name
    _players    = preloaded_players_df if preloaded_players_df is not None else get_all_players()
    player_row  = _players[_players["id"] == player_id]
    player_name = player_row["full_name"].values[0] if not player_row.empty else f"Player {player_id}"

    # Predict each target
    target_results = {}
    for target in targets:
        if target not in models:
            continue

        reg, cls = models[target]
        target_meta = metadata["targets"].get(target, {})
        threshold   = target_meta.get("threshold", DEFAULT_THRESHOLDS.get(target, 10))

        predicted_value = float(max(reg.predict(feat_row)[0], 0.0))
        over_prob       = float(cls.predict_proba(feat_row)[0][1])
        under_prob      = 1.0 - over_prob

        recent_avg_5  = _compute_recent_combo(recent_games, target, 5)
        recent_avg_10 = _compute_recent_combo(recent_games, target, 10)

        target_results[target] = TargetResult(
            target           = target,
            predicted_value  = round(predicted_value, 1),
            over_prob        = round(over_prob,  3),
            under_prob       = round(under_prob, 3),
            threshold        = threshold,
            confidence_label = _confidence_label(over_prob),
            recent_avg_5     = recent_avg_5,
            recent_avg_10    = recent_avg_10,
            model_mae        = target_meta.get("reg_cv_mae", 0.0),
            model_auc        = target_meta.get("cls_cv_auc", 0.0),
        )

    # ── Injury + lineup adjustment ───────────────────────────────────────────
    try:
        _inj_df    = preloaded_injury_df if preloaded_injury_df is not None else fetch_injury_report()
        inj_status = get_player_injury(player_name, _inj_df)
        inj_mult   = inj_status["multiplier"]
        inj_info   = inj_status
    except Exception:
        inj_mult = 1.0
        inj_info = {"status": "unknown", "multiplier": 1.0, "description": ""}

    try:
        _lu_df      = preloaded_lineup_df if preloaded_lineup_df is not None else fetch_all_lineups()
        lu_status   = get_player_lineup_status(player_name, _lu_df)
        lu_mult     = lu_status["multiplier"]
        lineup_info = lu_status
    except Exception:
        lu_mult     = 1.0
        lineup_info = {"role": "unknown", "multiplier": 1.0, "confirmed": False}

    # Combined multiplier — injury takes priority, lineup modulates bench players
    multiplier = inj_mult * (lu_mult if inj_mult > 0 else 1.0)

    # Apply multiplier to all regression predictions if player is injured
    if multiplier < 1.0:
        for target in target_results:
            tr = target_results[target]
            adjusted = round(tr.predicted_value * multiplier, 1)
            # Recalculate over probability with adjusted value
            import math
            mae  = tr.model_mae if tr.model_mae > 0 else 1.0
            diff = (adjusted - tr.threshold) / mae
            adj_over = round(1 / (1 + math.exp(-1.2 * diff)), 3)
            target_results[target] = TargetResult(
                target           = tr.target,
                predicted_value  = adjusted,
                over_prob        = adj_over,
                under_prob       = round(1 - adj_over, 3),
                threshold        = tr.threshold,
                confidence_label = _confidence_label(adj_over),
                recent_avg_5     = tr.recent_avg_5,
                recent_avg_10    = tr.recent_avg_10,
                model_mae        = tr.model_mae,
                model_auc        = tr.model_auc,
            )

    return PredictionResult(
        player_name  = player_name,
        player_id    = player_id,
        opponent     = opponent_name,
        is_home      = is_home,
        rest_days    = rest_days,
        targets      = target_results,
        recent_games = recent_games,
        feature_row  = feat_row,
        injury_info  = inj_info,
        lineup_info  = lineup_info,
    )


# ── UI helpers ────────────────────────────────────────────────────────────────

def get_teams_for_ui() -> pd.DataFrame:
    return get_team_list()


def get_players_for_ui(active_only: bool = True) -> pd.DataFrame:
    fm        = _get_cached_fm()
    valid_ids = set(fm["player_id"].unique())

    base = get_all_players()
    if active_only:
        base = base[base["is_active"] == True]
    base = base[base["id"].isin(valid_ids)].copy()

    # Try enriched data first (team + position from CommonAllPlayers)
    try:
        enriched = get_enriched_players()
        enriched["id"] = pd.to_numeric(enriched["id"], errors="coerce")
        base = base.merge(
            enriched[["id", "team_abbreviation", "position"]],
            on="id", how="left"
        )
    except Exception:
        pass

    # Fall back to most recent team from game logs
    if "team_abbreviation" not in base.columns or base["team_abbreviation"].isna().all():
        try:
            logs   = fetch_all_gamelogs()
            latest = (
                logs.sort_values("game_date")
                .groupby("player_id")["team_abbreviation"]
                .last().reset_index()
                .rename(columns={"player_id": "id"})
            )
            base = base.merge(latest, on="id", how="left", suffixes=("", "_log"))
            if "team_abbreviation_log" in base.columns:
                base["team_abbreviation"] = base["team_abbreviation"].fillna(base["team_abbreviation_log"])
                base = base.drop(columns=["team_abbreviation_log"])
        except Exception:
            base["team_abbreviation"] = "UNK"

    if "position" not in base.columns:
        base["position"] = "Unknown"

    base["team_abbreviation"] = base["team_abbreviation"].fillna("UNK")
    base["position"]          = base["position"].fillna("Unknown")

    return base.sort_values("full_name").reset_index(drop=True)


# ── Target display config (used by app.py) ────────────────────────────────────

TARGET_DISPLAY = {
    "pts":          {"label": "Points",              "short": "PTS",     "group": "Core"},
    "reb":          {"label": "Rebounds",            "short": "REB",     "group": "Core"},
    "ast":          {"label": "Assists",             "short": "AST",     "group": "Core"},
    "stl":          {"label": "Steals",              "short": "STL",     "group": "Core"},
    "blk":          {"label": "Blocks",              "short": "BLK",     "group": "Core"},
    "tov":          {"label": "Turnovers",           "short": "TOV",     "group": "Core"},
    "fg3m":         {"label": "3-Pointers Made",     "short": "3PM",     "group": "Shooting"},
    "fg3a":         {"label": "3-Pointers Attempted","short": "3PA",     "group": "Shooting"},
    "fga":          {"label": "FG Attempted",        "short": "FGA",     "group": "Shooting"},
    "pts_reb_ast":  {"label": "Pts + Reb + Ast",     "short": "PRA",     "group": "Combos"},
    "pts_reb":      {"label": "Pts + Rebounds",      "short": "PR",      "group": "Combos"},
    "pts_ast":      {"label": "Pts + Assists",       "short": "PA",      "group": "Combos"},
    "reb_ast":      {"label": "Reb + Assists",       "short": "RA",      "group": "Combos"},
    "blk_stl":      {"label": "Blk + Steals",        "short": "BS",      "group": "Combos"},
    "double_double":{"label": "Double Double",       "short": "DD",      "group": "Milestones"},
    "triple_double":{"label": "Triple Double",       "short": "TD",      "group": "Milestones"},
    "q1_pts":       {"label": "Q1 Points",           "short": "Q1P",     "group": "Q1"},
    "q1_reb":       {"label": "Q1 Rebounds",         "short": "Q1R",     "group": "Q1"},
    "q1_ast":       {"label": "Q1 Assists",          "short": "Q1A",     "group": "Q1"},
}

TARGET_GROUPS = ["Core", "Shooting", "Combos", "Milestones", "Q1"]


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LineBreaker prediction test")
    parser.add_argument("--player",   default="LeBron James")
    parser.add_argument("--opponent", default="GSW")
    parser.add_argument("--home",     action="store_true", default=True)
    parser.add_argument("--rest",     type=int, default=2)
    parser.add_argument("--targets",  nargs="+", default=None,
                        help="Specific targets (default: all)")
    args = parser.parse_args()

    results = search_players(args.player)
    if results.empty:
        print(f"No player found: {args.player}")
        sys.exit(1)
    player_row = results.iloc[0]

    teams  = get_teams_for_ui()
    opp    = teams[teams["team_abbreviation"].str.upper() == args.opponent.upper()]
    if opp.empty:
        print(f"No team found: {args.opponent}")
        sys.exit(1)
    opp_row = opp.iloc[0]

    result = predict(
        player_id        = int(player_row["id"]),
        opponent_team_id = int(opp_row["team_id"]),
        opponent_name    = opp_row["team_abbreviation"],
        is_home          = args.home,
        rest_days        = args.rest,
        targets          = args.targets,
    )

    print(f"\n{'─'*55}")
    print(f"  {result.player_name} vs. {result.opponent}")
    print(f"{'─'*55}")
    fmt = "  {:<18} {:>8}  over {:<4} {:>7}  ({})  "
    print(f"  {'Target':<18} {'Pred':>8}  {'Line':<8} {'Over%':>7}  Confidence")
    print("  " + "-" * 55)
    for target, tr in result.targets.items():
        disp = TARGET_DISPLAY.get(target, {}).get("label", target)
        print(fmt.format(
            disp, tr.predicted_value,
            tr.threshold, f"{tr.over_prob*100:.1f}%",
            tr.confidence_label,
        ))
    print(f"{'─'*55}\n")