"""
linebreaker/models/quick_picks.py — Today's best picks by model edge.
Preloads all data once before the player loop to avoid repeated I/O.
"""
import sys, math
import pandas as pd
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

QUICK_PICK_TARGETS = ["pts", "reb", "ast", "fg3m", "pts_reb_ast"]


def generate_quick_picks(top_n=10, min_confidence=None, targets=None) -> pd.DataFrame:
    try:
        from models.predict import predict, get_players_for_ui, get_teams_for_ui, TARGET_DISPLAY
        from features.engineer import DEFAULT_THRESHOLDS, _get_cached_fm
        from data.fetch_data import fetch_today_slate, fetch_all_gamelogs, get_all_players
    except Exception as e:
        print(f"Import error in quick_picks: {e}")
        return pd.DataFrame()

    try:
        slate = fetch_today_slate()
    except Exception as e:
        print(f"Slate fetch error: {e}")
        return pd.DataFrame()

    if not slate:
        print("No games on today's slate.")
        return pd.DataFrame()

    # ── Deduplicate slate ────────────────────────────────────────────────────
    seen, deduped = set(), []
    for g in slate:
        key = tuple(sorted([g.get("home_abbr", ""), g.get("away_abbr", "")]))
        if key not in seen:
            seen.add(key)
            deduped.append(g)
    slate = deduped

    if targets is None:
        targets = QUICK_PICK_TARGETS

    # ── Preload ALL heavy data once ──────────────────────────────────────────
    print("Quick picks: preloading data...")
    try:
        _fm        = _get_cached_fm()
        _players   = get_all_players()
        _teams_df  = get_teams_for_ui()
        _players_df = get_players_for_ui(active_only=True)
    except Exception as e:
        print(f"Preload error (core): {e}")
        return pd.DataFrame()

    try:
        from data.fetch_injuries import fetch_injury_report
        _inj = fetch_injury_report()
    except Exception:
        _inj = None

    try:
        _gamelogs = fetch_all_gamelogs()
    except Exception:
        _gamelogs = None

    picks = []

    for game in slate:
        ha, aa = game.get("home_abbr", ""), game.get("away_abbr", "")

        for player_team, opp in [(ha, aa), (aa, ha)]:
            if not player_team or not opp:
                continue

            opp_rows = _teams_df[_teams_df["team_abbreviation"] == opp]
            if opp_rows.empty:
                continue
            opp_team_id = int(opp_rows.iloc[0]["team_id"])

            if "team_abbreviation" not in _players_df.columns:
                continue
            team_players = _players_df[_players_df["team_abbreviation"] == player_team]
            if team_players.empty:
                continue

            for _, p_row in team_players.head(8).iterrows():
                for target in targets:
                    try:
                        result = predict(
                            player_id            = int(p_row["id"]),
                            opponent_team_id     = opp_team_id,
                            opponent_name        = opp,
                            is_home              = (player_team == ha),
                            rest_days            = 2,
                            targets              = [target],
                            preloaded_injury_df  = _inj,
                            preloaded_fm         = _fm,
                            preloaded_players_df = _players,
                            preloaded_gamelogs   = _gamelogs,
                        )
                        tr = result.targets.get(target)
                        if tr is None:
                            continue

                        # Skip injured (multiplier == 0 means OUT)
                        if (result.injury_info or {}).get("multiplier", 1.0) == 0.0:
                            continue

                        # Filter confidence
                        if min_confidence and tr.confidence_label != min_confidence:
                            continue

                        threshold = DEFAULT_THRESHOLDS.get(target, tr.threshold)
                        edge_abs  = abs(float(tr.predicted_value) - threshold)
                        mae       = float(tr.model_mae) if float(tr.model_mae) > 0 else 1.0
                        edge_norm = round(edge_abs / max(mae, 0.1), 2)
                        direction = "OVER" if float(tr.predicted_value) >= threshold else "UNDER"

                        diff = (float(tr.predicted_value) - threshold) / mae
                        op   = round(1 / (1 + math.exp(-1.2 * diff)) * 100, 1)

                        # Quality gate: require meaningful edge AND sufficient win probability
                        win_prob = op if direction == "OVER" else (100 - op)
                        if edge_norm < 1.2 or win_prob < 57:
                            continue

                        # HOT/COLD trend
                        l5  = float(tr.recent_avg_5)
                        l10 = float(tr.recent_avg_10)
                        if l10 > 0:
                            ratio = l5 / l10
                            trend = "🔥 HOT" if ratio >= 1.10 else "🥶 COLD" if ratio <= 0.90 else "→"
                        else:
                            trend = "→"

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
                            "l5_avg":     l5,
                            "l10_avg":    l10,
                            "trend":      trend,
                            "mae":        float(tr.model_mae),
                            "ovr":        tr.ovr_score,
                            "badges":     tr.skill_badges,
                            "insight":    tr.scout_insight,
                        })
                    except Exception as e:
                        continue  # skip silently per player/target

    if not picks:
        print("Quick picks: no picks generated. Check slate and player data.")
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
            print(f"{arrow} {r['player']} {r['short']} {r['direction']} {r['line']} | Proj {r['predicted']} | Edge {r['edge']}x | {r['confidence']} | {r['trend']}")
