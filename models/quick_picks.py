"""
linebreaker/models/quick_picks.py — Today's best picks by model edge.
"""
import sys, math
import pandas as pd
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

QUICK_PICK_TARGETS = ["pts", "reb", "ast", "fg3m", "pts_reb_ast", "pts_ast", "pts_reb"]


def generate_quick_picks(top_n=10, min_confidence=None, targets=None,
                          injury_df=None, lineup_df=None) -> pd.DataFrame:
    try:
        from models.predict import predict, get_players_for_ui, get_teams_for_ui, TARGET_DISPLAY
        from features.engineer import DEFAULT_THRESHOLDS
        from data.fetch_data import fetch_today_slate
    except Exception as e:
        print(f"Import error in quick_picks: {e}")
        return pd.DataFrame()

    try:
        players_df = get_players_for_ui(active_only=True)
        teams_df   = get_teams_for_ui()
        slate      = fetch_today_slate()
    except Exception as e:
        print(f"Data load error: {e}")
        return pd.DataFrame()

    if not slate:
        return pd.DataFrame()

    if targets is None:
        targets = QUICK_PICK_TARGETS

    # Pre-load feature matrix and injury data ONCE before the loop
    # This prevents each predict() call from reloading from disk
    try:
        from features.engineer import build_feature_matrix, build_player_features
        from data.fetch_injuries import fetch_injury_report
        _fm   = build_feature_matrix()   # loads cached matrix once
        _inj  = fetch_injury_report()    # loads cached injuries once
    except Exception as e:
        print(f"Pre-load failed: {e}")
        _fm, _inj = None, None

    picks = []

    # Deduplicate slate by game matchup to avoid processing same game twice
    seen_games: set = set()
    deduped_slate = []
    for g in slate:
        key = tuple(sorted([g.get("home_abbr",""), g.get("away_abbr","")]))
        if key not in seen_games:
            seen_games.add(key)
            deduped_slate.append(g)
    slate = deduped_slate

    for game in slate:
        ha, aa = game.get("home_abbr",""), game.get("away_abbr","")

        for player_team, opp in [(ha, aa), (aa, ha)]:
            if not player_team or not opp:
                continue

            opp_rows = teams_df[teams_df["team_abbreviation"] == opp]
            if opp_rows.empty:
                continue
            opp_team_id = int(opp_rows.iloc[0]["team_id"])

            if "team_abbreviation" in players_df.columns:
                team_players = players_df[players_df["team_abbreviation"] == player_team]
            else:
                continue

            if team_players.empty:
                continue

            # Limit to 6 players per team — starters most likely
            for _, p_row in team_players.head(6).iterrows():
                for target in targets:
                    try:
                        result = predict(
                            player_id            = int(p_row["id"]),
                            opponent_team_id     = opp_team_id,
                            opponent_name        = opp,
                            is_home              = (player_team == ha),
                            rest_days            = 2,
                            preloaded_injury_df  = _inj,
                        )
                        tr = result.targets.get(target)
                        if tr is None:
                            continue

                        # Skip injured
                        if (result.injury_info or {}).get("multiplier", 1.0) == 0.0:
                            continue

                        # Filter confidence
                        if min_confidence and tr.confidence_label != min_confidence:
                            continue

                        threshold = DEFAULT_THRESHOLDS.get(target, tr.threshold)
                        edge_abs  = abs(float(tr.predicted_value) - threshold)
                        edge_norm = round(edge_abs / max(float(tr.model_mae), 0.1), 2)
                        direction = "OVER" if float(tr.predicted_value) >= threshold else "UNDER"

                        mae  = float(tr.model_mae) if float(tr.model_mae) > 0 else 1.0
                        diff = (float(tr.predicted_value) - threshold) / mae
                        op   = round(1 / (1 + math.exp(-1.2 * diff)) * 100, 1)

                        info = TARGET_DISPLAY.get(target, {"label": target, "short": target})
                        picks.append({
                            "player":     p_row.get("full_name", "Unknown"),
                            "team":       player_team,
                            "opponent":   opp,
                            "stat":       info.get("label", target),
                            "short":      info.get("short", target),
                            "target":     target,
                            "predicted":  float(tr.predicted_value),
                            "line":       threshold,
                            "direction":  direction,
                            "edge":       edge_norm,
                            "edge_abs":   round(edge_abs, 2),
                            "over_prob":  op,
                            "confidence": tr.confidence_label,
                            "l5_avg":     tr.recent_avg_5,
                            "mae":        float(tr.model_mae),
                        })
                    except Exception:
                        continue  # skip silently per player/target

    if not picks:
        return pd.DataFrame()

    df = (pd.DataFrame(picks)
            .sort_values("edge", ascending=False)
            .drop_duplicates(subset=["player", "target"], keep="first")
            .sort_values("edge", ascending=False)
            .head(top_n)
            .reset_index(drop=True))
    return df


if __name__ == "__main__":
    picks = generate_quick_picks(top_n=10)
    if picks.empty:
        print("No picks.")
    else:
        for _, r in picks.iterrows():
            arrow = "⬆" if r["direction"] == "OVER" else "⬇"
            print(f"{arrow} {r['player']} {r['short']} {r['direction']} {r['line']} | Proj {r['predicted']} | Edge {r['edge']}x | {r['confidence']}")