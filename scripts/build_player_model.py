"""
build_player_model.py

Trains XGBoost models for player props:
  1. toi_ev          — EV ice time (regression)
  2. toi_pp          — PP ice time (regression)
  3. ev_shots        — EV shots on goal (regression)
  4. pp_shots        — PP shots on goal (regression)
  5. ev_assists      — EV assists (regression)
  6. pp_assists      — PP assists (regression)
  7. scored_ev_goal  — P(scored EV goal this game) (classifier)
  8. scored_pp_goal  — P(scored PP goal this game) (classifier)

Train: 20152016-20242025
Test:  20252026
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/processed/player_features.csv"
MODEL_DIR = ROOT / "models/player"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEST_SEASON = 20252026

EXCLUDE = {
    # Identity / metadata
    "game_id","season","date","player_id","first_name","last_name","full_name",
    "team","opponent","shoots","height_in","weight_lbs","birth_date","birth_country",
    # Raw same-game targets (leakage)
    "toi","toi_ev","toi_pp","toi_sh","toi_en",
    "goals","assists","points","shots","plus_minus","pim","shifts",
    "gw_goals","ot_goals",
    "ev_goals","pp_goals","sh_goals",
    "ev_assists","pp_assists","sh_assists",
    "ev_shots","pp_shots","sh_shots",
    "ixg","ev_ixg","pp_ixg","sh_ixg",
    "ev_goals_per60","pp_goals_per60",
    "scored_ev_goal","scored_pp_goal",
    # Same-game point totals (leakage)
    "pp_points","sh_points",
    # Same-game derived rates (leakage)
    "ev_shooting_pct","pp_shooting_pct","ev_goals_per_ixg",
    "ev_onice_sf","ev_onice_sa","pp_onice_sf","pp_onice_sa",
    "sh_onice_sf","sh_onice_sa",
    "ev_onice_gf","ev_onice_ga","pp_onice_gf","pp_onice_ga",
    "sh_onice_gf","sh_onice_ga",
    "ev_onice_sf_per60","ev_onice_sa_per60",
    "ev_onice_gf_per60","ev_onice_ga_per60",
    "ev_shots_per60","pp_shots_per60",
    "ev_ixg_per60","pp_ixg_per60",
    "ev_toi_share","pp_toi_share","ev_cf_pct","ipp",
    "hits","blocks","faceoffs_won","faceoffs_taken","giveaways","takeaways",
    "team_ev_toi","team_pp_toi","team_sh_toi","team_ev_sog","team_pp_sog",
    "pp_shots_api", "pp_goals_api", "pp_toi_sec_api",
}

TARGETS = [
    ("toi_ev",         "regression", "min"),
    ("toi_pp",         "regression", "min"),
    ("ev_shots",       "regression", "shots"),
    ("pp_shots",       "regression", "shots"),
    ("ev_assists",     "regression", "assists"),
    ("pp_assists",     "regression", "assists"),
    ("scored_ev_goal", "classifier", "prob"),
]


def get_features(df, exclude_set):
    return [c for c in df.columns
            if c not in exclude_set
            and df[c].dtype in [np.float64, np.int64, float, int]
            and not c.startswith("Unnamed")
            and not c.endswith("_x")
            and not c.endswith("_y")]


def prepare(df, feature_cols, target):
    valid = df[feature_cols + [target]].dropna(subset=[target])
    X = valid[feature_cols].fillna(0)
    y = valid[target]
    return X, y


def select_features(X, y, model_type, n=60):
    if model_type == "classifier":
        m = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric="logloss"
        )
    else:
        m = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric="rmse"
        )
    m.fit(X, y)
    imp = pd.Series(m.feature_importances_, index=X.columns)
    return imp.nlargest(n).index.tolist()


def train_model(X_tr, y_tr, X_val, y_val, model_type):
    if model_type == "classifier":
        pos = (y_tr == 1).sum()
        neg = (y_tr == 0).sum()
        m = xgb.XGBClassifier(
            n_estimators=1000, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, gamma=1,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=neg / max(pos, 1),
            eval_metric="logloss", early_stopping_rounds=30,
            random_state=42, n_jobs=-1,
        )
    else:
        m = xgb.XGBRegressor(
            n_estimators=1000, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, gamma=1,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="rmse", early_stopping_rounds=30,
            random_state=42, n_jobs=-1,
        )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m


def evaluate(y_true, y_pred, unit, model_type):
    if model_type == "classifier":
        # Guard against all-zero test set
        if len(np.unique(y_true)) < 2:
            print(f"  WARNING: test set has only one class, skipping metrics")
            return 0.0
        auc  = roc_auc_score(y_true, y_pred)
        ll   = log_loss(y_true, y_pred, labels=[0, 1])
        base = log_loss(y_true, [y_true.mean()] * len(y_true), labels=[0, 1])
        print(f"  AUC:       {auc:.4f}")
        print(f"  LogLoss:   {ll:.4f}  (baseline: {base:.4f})")
        print(f"  Goal rate: {y_true.mean():.3f}  Mean pred: {y_pred.mean():.3f}")
        return auc
    else:
        mae  = mean_absolute_error(y_true, y_pred)
        bias = float(np.mean(y_pred - y_true))
        base = mean_absolute_error(y_true, [y_true.mean()] * len(y_true))
        try:
            corr, _ = pearsonr(y_true, y_pred)
        except Exception:
            corr = float("nan")
        print(f"  MAE:      {mae:.3f} {unit}  (baseline: {base:.3f})")
        print(f"  Bias:     {bias:+.3f}")
        print(f"  Corr:     {corr:.3f}")
        print(f"  Pred std: {np.std(y_pred):.3f}  Actual std: {np.std(y_true):.3f}")
        return mae


def main():
    print("Loading player features...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    df["is_defense"] = (df["position"] == "D").astype(int)
    df["is_forward"] = (df["position"] != "D").astype(int)
    df["is_center"]  = (df["position"] == "C").astype(int)
    df["is_home"]    = df["is_home"].astype(int)

    train_df = df[(df["season"] != TEST_SEASON) & (df["position"] != "G")].copy()
    test_df  = df[(df["season"] == TEST_SEASON) & (df["position"] != "G")].copy()
    print(f"Train skaters: {len(train_df):,} | Test skaters: {len(test_df):,}")

    feature_cols = get_features(df, EXCLUDE)
    print(f"Candidate features: {len(feature_cols)}")

    all_models      = {}
    all_features    = {}
    all_stds        = {}
    all_calibrators = {}
    results         = {}

    print("\n" + "="*60)
    print("TRAINING PLAYER MODELS")
    print("="*60)

    for target, model_type, unit in TARGETS:
        print(f"\n--- {target} ({model_type}) ---")

        if target not in df.columns:
            print(f"  WARNING: {target} not found, skipping")
            continue

        X_train_full, y_train = prepare(train_df, feature_cols, target)
        X_test_full,  y_test  = prepare(test_df,  feature_cols, target)

        # Filter by minimum TOI
        if target in ("ev_shots", "ev_assists", "scored_ev_goal"):
            mask_tr = train_df.loc[X_train_full.index, "toi_ev"] > 2
            mask_te = test_df.loc[X_test_full.index,  "toi_ev"] > 2
            X_train_full = X_train_full[mask_tr]
            y_train      = y_train[mask_tr]
            X_test_full  = X_test_full[mask_te]
            y_test       = y_test[mask_te]
        elif target in ("pp_shots", "pp_assists"):
            mask_tr = train_df.loc[X_train_full.index, "toi_pp_season_avg"].fillna(0) > 0.1
            mask_te = test_df.loc[X_test_full.index,  "toi_pp_season_avg"].fillna(0) > 0.1
            X_train_full = X_train_full[mask_tr]
            y_train      = y_train[mask_tr]
            X_test_full  = X_test_full[mask_te]
            y_test       = y_test[mask_te]

        print(f"  Train: {len(X_train_full):,} | Test: {len(X_test_full):,}")
        if model_type == "classifier":
            print(f"  Positive rate: {y_train.mean():.3f} train / {y_test.mean():.3f} test")

        print("  Selecting features...")
        top_feats = select_features(X_train_full, y_train, model_type, n=60)

        X_tr = X_train_full[top_feats]
        X_te = X_test_full[top_feats]

        val_split = int(len(X_tr) * 0.8)
        X_t, X_v = X_tr.iloc[:val_split], X_tr.iloc[val_split:]
        y_t, y_v = y_train.iloc[:val_split], y_train.iloc[val_split:]

        print("  Training...")
        model = train_model(X_t, y_t, X_v, y_v, model_type)

        # Calibrate classifier probabilities using validation set
        if model_type == "classifier":
            from sklearn.isotonic import IsotonicRegression
            raw_val_probs = model.predict_proba(X_v)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_val_probs, y_v)
            all_calibrators[target] = iso
            y_pred = iso.transform(model.predict_proba(X_te)[:, 1])
        else:
            y_pred = model.predict(X_te)

        result = evaluate(y_test.values, y_pred, unit, model_type)

        residual_std = float(np.std(y_test.values - y_pred))
        all_stds[target] = residual_std

        print(f"  Top 10 features:")
        fi = pd.Series(model.feature_importances_, index=top_feats).sort_values(ascending=False)
        for f, v in fi.head(10).items():
            print(f"    {f:<45} {v:.4f}")

        all_models[target]   = model
        all_features[target] = top_feats
        results[target]      = {"result": result, "unit": unit, "type": model_type}

    # Save all models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)

    for target, model in all_models.items():
        path = MODEL_DIR / f"{target}.json"
        model.save_model(str(path))
        print(f"  Saved: {path.name}")

    with open(MODEL_DIR / "feature_lists.pkl", "wb") as f:
        pickle.dump(all_features, f)
    with open(MODEL_DIR / "residual_stds.pkl", "wb") as f:
        pickle.dump(all_stds, f)
    with open(MODEL_DIR / "model_types.pkl", "wb") as f:
        pickle.dump({t: mt for t, mt, _ in TARGETS}, f)
    print("  Saved: feature_lists.pkl, residual_stds.pkl, model_types.pkl")

    if all_calibrators:
        with open(MODEL_DIR / "calibrators.pkl", "wb") as f:
            pickle.dump(all_calibrators, f)
        print("  Saved: calibrators.pkl")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for target, res in results.items():
        if res["type"] == "classifier":
            print(f"  {target:<22} AUC: {res['result']:.4f}")
        else:
            print(f"  {target:<22} MAE: {res['result']:.3f} {res['unit']}")


if __name__ == "__main__":
    main()