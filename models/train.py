"""
linebreaker/models/train.py

Trains one XGBoost regressor + one XGBoost classifier per stat target.
Models and metadata saved to models/saved/.

Run: python models/train.py
     python models/train.py --targets pts reb ast   # specific targets only
     python models/train.py --refresh               # force rebuild feature matrix
"""

import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
)
from xgboost import XGBRegressor, XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from features.engineer import (
    build_feature_matrix,
    get_feature_cols,
    ALL_TARGETS,
    DEFAULT_THRESHOLDS,
)

SAVE_DIR = ROOT / "models" / "saved"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5

REGRESSOR_PARAMS = dict(
    n_estimators=600, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    n_jobs=-1, verbosity=0,
)

CLASSIFIER_PARAMS = dict(
    n_estimators=600, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    n_jobs=-1, verbosity=0, eval_metric="logloss",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _divider(title: str = ""):
    print(f"\n{'─' * 55}")
    if title:
        print(f"  {title}")
        print(f"{'─' * 55}")


def _time_series_cv(X, y, model, scorer):
    tscv   = TimeSeriesSplit(n_splits=N_SPLITS)
    scores = []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        scores.append(scorer(model, X.iloc[te], y.iloc[te]))
        print(f"    Fold {fold}: {scores[-1]:.4f}")
    return scores


# ── Per-target training ───────────────────────────────────────────────────────

def train_target(
    target: str,
    X: pd.DataFrame,
    y_full: pd.DataFrame,
    threshold: int,
) -> dict:
    """
    Train regressor + classifier for one stat target.
    Returns dict with both models and their CV metrics.
    """
    if target not in y_full.columns:
        print(f"  Skipping {target} — column not in feature matrix.")
        return None

    y = y_full[target].copy()
    valid = y.notna()
    if valid.sum() < 500:
        print(f"  Skipping {target} — only {valid.sum()} non-null rows (need 500+).")
        return None

    Xv = X[valid].reset_index(drop=True)
    yv = y[valid].reset_index(drop=True)

    result = {"target": target, "threshold": threshold, "n_rows": int(valid.sum())}

    # ── Regressor ──
    print(f"\n  [{target}] Regressor — {valid.sum():,} rows")

    def mae_scorer(m, Xt, yt): return mean_absolute_error(yt, m.predict(Xt))
    cv = _time_series_cv(Xv, yv, XGBRegressor(**REGRESSOR_PARAMS), mae_scorer)
    result["reg_cv_mae"]  = round(float(np.mean(cv)), 3)
    result["reg_cv_std"]  = round(float(np.std(cv)),  3)

    reg = XGBRegressor(**REGRESSOR_PARAMS)
    reg.fit(Xv, yv)
    preds = reg.predict(Xv)
    result["reg_train_mae"]  = round(float(mean_absolute_error(yv, preds)),  3)
    result["reg_train_rmse"] = round(float(root_mean_squared_error(yv, preds)), 3)
    result["regressor"] = reg

    # Top features
    result["top_features_reg"] = (
        pd.Series(reg.feature_importances_, index=Xv.columns)
        .sort_values(ascending=False).head(10).to_dict()
    )

    # ── Classifier ──
    # Special case: double_double and triple_double are already binary
    if target in ("double_double", "triple_double"):
        y_cls = yv.astype(int)
    else:
        y_cls = (yv >= threshold).astype(int)

    pos_rate = float(y_cls.mean())
    print(f"\n  [{target}] Classifier — over {threshold}: {pos_rate:.1%}")

    cls_params = CLASSIFIER_PARAMS.copy()
    if pos_rate < 0.4:
        cls_params["scale_pos_weight"] = round((1 - pos_rate) / pos_rate, 2)

    def auc_scorer(m, Xt, yt):
        p = m.predict_proba(Xt)[:, 1]
        return roc_auc_score(yt, p) if len(np.unique(yt)) > 1 else 0.5

    cv_cls = _time_series_cv(Xv, y_cls, XGBClassifier(**cls_params), auc_scorer)
    result["cls_cv_auc"] = round(float(np.mean(cv_cls)), 4)
    result["cls_cv_std"] = round(float(np.std(cv_cls)),  4)
    result["pos_rate"]   = round(pos_rate, 3)

    cls = XGBClassifier(**cls_params)
    cls.fit(Xv, y_cls)
    proba = cls.predict_proba(Xv)[:, 1]
    result["cls_train_acc"]   = round(float(accuracy_score(y_cls, (proba >= 0.5).astype(int))), 3)
    result["cls_brier"]       = round(float(brier_score_loss(y_cls, proba)), 4)
    result["classifier"] = cls

    result["top_features_cls"] = (
        pd.Series(cls.feature_importances_, index=Xv.columns)
        .sort_values(ascending=False).head(10).to_dict()
    )

    print(f"\n  [{target}] CV MAE: {result['reg_cv_mae']:.3f} ± {result['reg_cv_std']:.3f}"
          f"  |  CV AUC: {result['cls_cv_auc']:.4f} ± {result['cls_cv_std']:.4f}")

    return result


# ── Save ──────────────────────────────────────────────────────────────────────

def save_all(results: dict, feature_cols: list):
    _divider("Saving models")

    metadata = {
        "trained_at":   datetime.now().isoformat(),
        "feature_cols": feature_cols,
        "targets":      {},
    }

    for target, res in results.items():
        if res is None:
            continue
        joblib.dump(res["regressor"],  SAVE_DIR / f"{target}_regressor.pkl")
        joblib.dump(res["classifier"], SAVE_DIR / f"{target}_classifier.pkl")
        print(f"  Saved {target}_regressor.pkl + {target}_classifier.pkl")

        # Store metadata without model objects
        metadata["targets"][target] = {
            k: v for k, v in res.items()
            if k not in ("regressor", "classifier")
        }

    with open(SAVE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  metadata.json → {SAVE_DIR / 'metadata.json'}")


def load_models(target: str):
    """Load regressor + classifier for a specific target."""
    reg = joblib.load(SAVE_DIR / f"{target}_regressor.pkl")
    cls = joblib.load(SAVE_DIR / f"{target}_classifier.pkl")
    with open(SAVE_DIR / "metadata.json") as f:
        meta = json.load(f)
    return reg, cls, meta


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results: dict):
    _divider("Training summary")
    fmt = "  {:<18} {:>10} {:>8} {:>10} {:>8}"
    print(fmt.format("Target", "CV MAE", "±", "CV AUC", "±"))
    print("  " + "-" * 56)
    for target, res in results.items():
        if res is None:
            print(f"  {target:<18} skipped")
            continue
        print(fmt.format(
            target,
            f"{res['reg_cv_mae']:.3f}",
            f"{res['reg_cv_std']:.3f}",
            f"{res['cls_cv_auc']:.4f}",
            f"{res['cls_cv_std']:.4f}",
        ))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(targets: list = None, force_refresh: bool = False):
    _divider("LineBreaker — Multi-Target Training")

    print("\n  Loading feature matrix...")
    fm           = build_feature_matrix(force_refresh=force_refresh)
    feature_cols = get_feature_cols(fm)

    # Include DD/TD in target list
    all_targets = ALL_TARGETS + ["double_double", "triple_double"]
    if targets:
        all_targets = [t for t in all_targets if t in targets]

    df = fm.sort_values("game_date").reset_index(drop=True)
    X  = df[feature_cols]

    print(f"  Rows: {len(df):,}  |  Features: {len(feature_cols)}  |  Targets: {len(all_targets)}")

    results = {}
    for target in all_targets:
        _divider(f"Target: {target}")
        threshold = DEFAULT_THRESHOLDS.get(target, 10)
        results[target] = train_target(target, X, df, threshold)

    save_all(results, feature_cols)
    print_summary(results)
    _divider("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LineBreaker multi-target trainer")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Specific targets to train (default: all)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force rebuild feature matrix before training")
    args = parser.parse_args()
    main(targets=args.targets, force_refresh=args.refresh)