"""
models/backtest.py
------------------
Backtest LineBreaker predictions against actual ESPN box score results.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import date, timedelta
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

_COL_MAP = {
    "pts":"pts","reb":"reb","ast":"ast","stl":"stl","blk":"blk",
    "tov":"tov","fg3m":"fg3m","fg3a":"fg3a","fga":"fga",
}
_COMBO = {
    "pts_reb_ast":["pts","reb","ast"],"pts_reb":["pts","reb"],
    "pts_ast":["pts","ast"],"reb_ast":["reb","ast"],"blk_stl":["blk","stl"],
}
_DEFAULT_LINE = {
    "pts":20,"reb":5,"ast":5,"stl":1,"blk":1,"tov":3,
    "fg3m":2,"fg3a":5,"fga":15,"pts_reb_ast":30,"pts_reb":25,
    "pts_ast":25,"reb_ast":10,"blk_stl":2,
}

def _load_recent_gamelogs(days_back=30):
    cache = ROOT / "data" / "cache" / "gamelogs_2025_26.csv"
    if not cache.exists():
        return pd.DataFrame()
    df = pd.read_csv(cache, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    cutoff = pd.Timestamp(date.today() - timedelta(days=days_back))
    return df[df["game_date"] >= cutoff].copy()

def _actual_stat(row, target):
    if target in _COL_MAP:
        v = pd.to_numeric(row.get(_COL_MAP[target], None), errors="coerce")
        return float(v) if pd.notna(v) else None
    if target in _COMBO:
        vals = [pd.to_numeric(row.get(p, None), errors="coerce") for p in _COMBO[target]]
        if all(pd.notna(v) for v in vals):
            return float(sum(vals))
    return None

def backtest_vs_actuals(target="pts", days_back=30, min_games=5, verbose=True):
    """
    Compare model predictions vs actual results for recent games.
    Returns DataFrame: player, game_date, predicted, actual, line,
                       direction, correct, confidence, mae_error
    """
    from models.predict import predict, get_players_for_ui
    logs = _load_recent_gamelogs(days_back)
    if logs.empty:
        return pd.DataFrame()
    players_df = get_players_for_ui()
    if players_df.empty:
        return pd.DataFrame()
    id_to_name = dict(zip(players_df["id"].astype(str), players_df["full_name"]))
    line = _DEFAULT_LINE.get(target, 10)
    records = []
    game_counts = logs.groupby("player_id").size()
    active_ids  = game_counts[game_counts >= min_games].index.tolist()
    seen = set()
    for pid in active_ids[:60]:
        pname = id_to_name.get(str(pid), "")
        if not pname or pname in seen:
            continue
        seen.add(pname)
        pgames = logs[logs["player_id"] == pid].sort_values("game_date")
        for i, (_, row) in enumerate(pgames.iterrows()):
            if i < 5:
                continue
            actual = _actual_stat(row, target)
            if actual is None:
                continue
            matchup = str(row.get("matchup", ""))
            opp = matchup.split("@")[-1].strip() if "@" in matchup else matchup.split("vs.")[-1].strip()
            try:
                result = predict(player_name=pname, opponent=opp,
                                 is_home=("vs." in matchup), rest_days=2,
                                 compute_matchup=False)
            except Exception:
                continue
            tr = result.targets.get(target)
            if tr is None:
                continue
            predicted = float(tr.predicted_value)
            direction = "OVER" if predicted >= line else "UNDER"
            correct   = (direction=="OVER" and actual>line) or (direction=="UNDER" and actual<=line)
            records.append({
                "player":pname,"game_date":row["game_date"].date(),
                "predicted":round(predicted,1),"actual":round(actual,1),
                "line":line,"direction":direction,"correct":correct,
                "confidence":tr.confidence_label,"mae_error":round(abs(predicted-actual),1),
            })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    if verbose:
        acc = df["correct"].mean()*100
        hi  = df[df["confidence"]=="High"]["correct"].mean()*100 if (df["confidence"]=="High").any() else 0
        print(f"Backtest {target} ({days_back}d): {acc:.1f}% acc | High-conf: {hi:.1f}% | MAE: {df['mae_error'].mean():.2f}")
    return df

def quick_mae_check(days_back=14):
    """Fast rolling-avg MAE check across all targets. Returns {target: mae}."""
    logs = _load_recent_gamelogs(days_back)
    if logs.empty:
        return {}
    results = {}
    all_targets = {**{k:[k] for k in _COL_MAP}, **_COMBO}
    for target, cols in all_targets.items():
        avail = [c for c in cols if c in logs.columns]
        if not avail:
            continue
        logs[f"__t_{target}"] = logs[avail].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        roll = logs.groupby("player_id")[f"__t_{target}"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=3).mean())
        valid = logs.dropna(subset=[f"__t_{target}"]).copy()
        valid["__roll"] = roll[valid.index]
        valid = valid.dropna(subset=["__roll"])
        if len(valid) < 10:
            continue
        results[target] = round((valid["__roll"] - valid[f"__t_{target}"]).abs().mean(), 3)
    return results
