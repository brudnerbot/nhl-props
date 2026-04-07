"""
build_goalie_model.py

Builds XGBoost models to predict goalie performance:
  - saves:         total saves in game (for save prop)
  - goals_against: goals allowed (feeds player goal formula + team model)
  - save_pct:      save percentage (context/validation)

Architecture: fully independent — does NOT use player predictions.
Inputs: goalie rolling stats + opponent offense context + team defense + B2B + home/away

Output:
  models/goalie/saves.json
  models/goalie/goals_against.json
  models/goalie/feature_lists.pkl
  models/goalie/residual_stds.pkl

Usage:
  python scripts/build_goalie_model.py
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models/goalie"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

GOALIE_LOGS  = DATA_DIR / "raw/goalie_game_logs/goalie_game_logs.csv"
TEAM_LOGS    = DATA_DIR / "raw/team_game_logs/team_game_logs.csv"
MIN_SEASON   = 20152016
MIN_TOI      = 30  # only games where goalie played 30+ min


def load_and_merge():
    print("Loading goalie game logs...")
    gl = pd.read_csv(GOALIE_LOGS, low_memory=False)
    gl["date"] = pd.to_datetime(gl["date"])
    gl = gl[gl["season"] >= MIN_SEASON].copy()
    gl = gl[gl["toi"] >= MIN_TOI].copy()
    print(f"  Goalie games: {len(gl):,}")

    # ── B2B flag ──────────────────────────────────────────────────────────────
    gl = gl.sort_values(["player_id","date"]).reset_index(drop=True)
    gl["prev_date"] = gl.groupby("player_id")["date"].shift(1)
    gl["days_rest"] = (gl["date"] - gl["prev_date"]).dt.days
    gl["is_b2b"]    = (gl["days_rest"] == 1).astype(int)
    gl = gl.drop(columns=["prev_date","days_rest"])

    # ── Team context: opponent offense + own team defense ─────────────────────
    print("Loading team game logs for opponent context...")
    tgl = pd.read_csv(TEAM_LOGS, low_memory=False)
    tgl["date"] = pd.to_datetime(tgl["date"])
    tgl = tgl.sort_values(["team","date"])

    # Opponent offense rolling (shots they generate, xG they produce)
    opp_cols = ["ev_shots_on_goal_for","ev_shot_attempts_for",
                "pp_toi","goals_for","ev_goals_for"]
    for col in opp_cols:
        if col not in tgl.columns: continue
        tgl[f"opp_{col}_last10"] = (
            tgl.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        )
        tgl[f"opp_{col}_last30"] = (
            tgl.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(30, min_periods=8).mean())
        )
        tgl[f"opp_{col}_season_avg"] = (
            tgl.groupby(["team","season"])[col]
            .transform(lambda x: x.shift(1).expanding().mean())
        )

    # Own team defense rolling (shots they allow)
    def_cols = ["ev_shots_on_goal_against","ev_shot_attempts_against","goals_against"]
    for col in def_cols:
        if col not in tgl.columns: continue
        tgl[f"team_{col}_last10"] = (
            tgl.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        )
        tgl[f"team_{col}_last30"] = (
            tgl.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(30, min_periods=8).mean())
        )

    # Join opponent context (opponent = team shooting against this goalie)
    opp_feature_cols = (
        [f"opp_{c}_last10"     for c in opp_cols if f"opp_{c}_last10" in tgl.columns] +
        [f"opp_{c}_last30"     for c in opp_cols if f"opp_{c}_last30" in tgl.columns] +
        [f"opp_{c}_season_avg" for c in opp_cols if f"opp_{c}_season_avg" in tgl.columns]
    )
    def_feature_cols = (
        [f"team_{c}_last10" for c in def_cols if f"team_{c}_last10" in tgl.columns] +
        [f"team_{c}_last30" for c in def_cols if f"team_{c}_last30" in tgl.columns]
    )

    opp_ctx = tgl[["game_id","team"] + opp_feature_cols + def_feature_cols].copy()

    # Join as opponent (goalie's team faces this opponent)
    opp_ctx_renamed = opp_ctx.rename(columns={"team":"opponent"})
    gl = gl.merge(opp_ctx_renamed, on=["game_id","opponent"], how="left")

    # Join own team defense
    own_ctx = tgl[["game_id","team"] + def_feature_cols].copy()
    gl = gl.merge(own_ctx, on=["game_id","team"], how="left",
                  suffixes=("","_own"))

    covered = gl["opp_ev_shots_on_goal_for_last10"].notna().sum()
    print(f"  Opponent context joined: {covered:,}/{len(gl):,} ({covered/len(gl):.1%})")
    return gl


def build_features(df):
    print("Building feature matrix...")

    feature_cols = [
        # Goalie quality
        "gsax_last10", "gsax_last20", "gsax_last30", "gsax_season_avg",
        "regressed_gsax_per_game", "career_gsax_per_game",
        # Save rate
        "save_pct_last10", "save_pct_last20", "save_pct_last30", "save_pct_season_avg",
        # Shot volume faced
        "shots_against_last10", "shots_against_last20",
        # Opponent offense context
        "opp_ev_shots_on_goal_for_last10", "opp_ev_shots_on_goal_for_last30",
        "opp_ev_shots_on_goal_for_season_avg",
        "opp_ev_shot_attempts_for_last10",  "opp_ev_shot_attempts_for_last30",
        "opp_pp_toi_last10",                "opp_pp_toi_last30",
        "opp_goals_for_last10",             "opp_goals_for_last30",
        # Own team defense
        "team_ev_shots_on_goal_against_last10", "team_ev_shots_on_goal_against_last30",
        # Game context
        "is_home", "is_b2b", "season",
    ]

    # Only keep cols that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"  Features: {len(feature_cols)}")
    return feature_cols


def train_model(df, target, feature_cols, min_season_train=20172018):
    print(f"\nTraining model: {target}")
    train = df[(df["season"] >= min_season_train) &
               (df["season"] <  20242025)].copy()
    val   = df[df["season"] == 20242025].copy()

    train = train.dropna(subset=[target] + feature_cols[:5])
    val   = val.dropna(subset=[target] + feature_cols[:5])

    X_train = train[feature_cols].fillna(0)
    y_train = train[target]
    X_val   = val[feature_cols].fillna(0)
    y_val   = val[target]

    print(f"  Train: {len(train):,}  Val: {len(val):,}")

    params = {
        "n_estimators":     400,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "random_state":     42,
        "n_jobs":           -1,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds = model.predict(X_val)
    mae   = float(np.abs(y_val - preds).mean())
    resid = float(np.std(y_val - preds))
    print(f"  Val MAE: {mae:.3f}  ResidStd: {resid:.3f}")
    print(f"  Val mean actual: {y_val.mean():.2f}  mean pred: {preds.mean():.2f}")

    # Feature importance top 10
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    print(f"  Top features:")
    for feat, score in imp.nlargest(10).items():
        print(f"    {feat:<45}: {score:.4f}")

    return model, mae, resid


def main():
    print("="*60)
    print("BUILDING GOALIE MODEL")
    print("="*60)

    df = load_and_merge()
    feature_cols = build_features(df)

    models       = {}
    feature_lists = {}
    residual_stds = {}
    maes          = {}

    for target in ["saves", "goals_against"]:
        model, mae, resid = train_model(df, target, feature_cols)
        models[target]        = model
        feature_lists[target] = feature_cols
        residual_stds[target] = resid
        maes[target]          = mae

        model.save_model(str(MODEL_DIR / f"{target}.json"))
        print(f"  Saved {target}.json")

    with open(MODEL_DIR / "feature_lists.pkl", "wb") as f:
        pickle.dump(feature_lists, f)
    with open(MODEL_DIR / "residual_stds.pkl", "wb") as f:
        pickle.dump(residual_stds, f)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for target, mae in maes.items():
        unit = "saves" if target == "saves" else "goals"
        print(f"  {target:<20} MAE: {mae:.3f} {unit}")

    # Baseline comparison
    print(f"\nBaseline (predict mean):")
    test = df[df["season"]==20242025].dropna(subset=["saves","goals_against"])
    for target in ["saves","goals_against"]:
        baseline_mae = (test[target] - test[target].mean()).abs().mean()
        print(f"  {target:<20} baseline MAE: {baseline_mae:.3f}")


if __name__ == "__main__":
    main()
