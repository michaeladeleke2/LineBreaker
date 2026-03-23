"""
linebreaker/retrain.py

Monthly retraining script. Run manually or schedule via cron.

Usage:
    python retrain.py              # Full retrain
    python retrain.py --check      # Check if retrain is needed
    python retrain.py --data-only  # Refresh data without retraining

Cron example (1st of every month at 3am):
    0 3 1 * * cd ~/LineBreaker && python retrain.py >> logs/retrain.log 2>&1
"""

import sys, argparse, json
from pathlib import Path
from datetime import date, datetime

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

METADATA_PATH = ROOT / "models" / "saved" / "metadata.json"
LOG_PATH      = ROOT / "logs" / "retrain.log"
LOG_PATH.parent.mkdir(exist_ok=True)


def _log(msg: str):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def needs_retrain(days_threshold: int = 30) -> bool:
    """Check if models are older than the threshold."""
    if not METADATA_PATH.exists():
        return True
    try:
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        trained_at = datetime.fromisoformat(meta.get("trained_at", "2020-01-01"))
        age_days   = (datetime.now() - trained_at).days
        _log(f"Models trained {age_days} days ago")
        return age_days >= days_threshold
    except Exception:
        return True


def run_retrain(force: bool = False, data_only: bool = False):
    _log("="*50)
    _log("LineBreaker Retraining Pipeline")
    _log("="*50)

    if not force and not needs_retrain():
        _log("Models are recent — skipping retrain. Use --force to override.")
        return

    # Step 1: Refresh data
    _log("Step 1/3: Refreshing current season data...")
    from data.fetch_data import refresh_current_season
    ok = refresh_current_season()
    if not ok:
        _log("WARNING: Data refresh failed — continuing with cached data")
    else:
        _log("Data refresh complete")

    if data_only:
        _log("--data-only flag set. Stopping after data refresh.")
        return

    # Step 2: Rebuild feature matrix
    _log("Step 2/3: Rebuilding feature matrix...")
    from features.engineer import build_feature_matrix
    fm = build_feature_matrix(force_refresh=True)
    _log(f"Feature matrix: {len(fm):,} rows, {len(fm.columns)} columns")

    # Step 3: Retrain models
    _log("Step 3/3: Retraining models...")
    from models.train import train_all
    results = train_all()

    _log("Retraining complete!")
    _log(f"Targets trained: {len(results)}")
    for target, metrics in results.items():
        _log(f"  {target:<20} MAE {metrics['cv_mae']:.3f} | AUC {metrics['cv_auc']:.4f}")

    # Run a quick backtest after retraining
    _log("\nRunning post-retrain backtest (7 days)...")
    try:
        from models.backtest import backtest_vs_actuals
        df = backtest_vs_actuals(target="pts", days_back=7, verbose=False)
        if not df.empty:
            acc = df["correct"].mean() * 100
            _log(f"Post-retrain pts accuracy (7d): {acc:.1f}%")
    except Exception as e:
        _log(f"Backtest failed: {e}")

    _log("Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LineBreaker retraining pipeline")
    parser.add_argument("--force",     action="store_true", help="Force retrain even if recent")
    parser.add_argument("--check",     action="store_true", help="Only check if retrain needed")
    parser.add_argument("--data-only", action="store_true", help="Refresh data without retraining")
    parser.add_argument("--days",      default=30, type=int, help="Retrain threshold in days")
    args = parser.parse_args()

    if args.check:
        needed = needs_retrain(days_threshold=args.days)
        print(f"Retrain needed: {needed}")
    else:
        run_retrain(force=args.force, data_only=args.data_only)