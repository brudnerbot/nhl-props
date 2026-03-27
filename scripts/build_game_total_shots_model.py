"""
build_game_total_shots_model.py

Trains a dedicated XGBoost model for combined game total shots on goal.
Target: home_total_shots + away_total_shots (one row per game)
Baseline (naive sum of empirical formula): MAE 6.671
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

import os
DATA_PATH  = os.path.expanduser("~/nhl-props/data/processed/team_features.csv")
MODEL_DIR  = os.path.expanduser("~/nhl-props/models/team")

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"  {len(df):,} game rows | seasons: {sorted(df['season'].unique())}")

# Target: combined game shots
df["game_total_shots"] = df["home_total_shots_on_goal_for"] + df["away_total_shots_on_goal_for"]

train = df[df["season"] != 20252026].copy()
test  = df[df["season"] == 20252026].copy()
print(f"  Train: {len(train):,}  |  Test: {len(test):,}")
print(f"  Train target mean: {train['game_total_shots'].mean():.2f}  std: {train['game_total_shots'].std():.2f}")
print(f"  Test  target mean: {test['game_total_shots'].mean():.2f}  std: {test['game_total_shots'].std():.2f}")

# ── Features ───────────────────────────────────────────────────────────────────
# Use symmetric game-level features — both team perspectives on shots + pace
EXCLUDE = {
    "game_id", "season", "date", "home_team", "away_team",
    "home_won", "went_to_ot", "game_total_shots",
    # raw same-game actuals
    "home_total_shots_on_goal_for", "home_total_shots_on_goal_against",
    "away_total_shots_on_goal_for", "away_total_shots_on_goal_against",
    "home_total_shot_attempts_for", "home_total_shot_attempts_against",
    "away_total_shot_attempts_for", "away_total_shot_attempts_against",
    "home_total_fenwick_for", "home_total_fenwick_against",
    "away_total_fenwick_for", "away_total_fenwick_against",
    "home_total_penalties_taken", "home_total_penalties_drawn",
    "away_total_penalties_taken", "away_total_penalties_drawn",
    "home_total_minor_penalties_taken", "home_total_minor_penalties_drawn",
    "away_total_minor_penalties_taken", "away_total_minor_penalties_drawn",
}

# Also exclude raw per-game stats that would leak
raw_leak_patterns = [
    "home_ev_shots_on_goal_for", "home_ev_shots_on_goal_against",
    "away_ev_shots_on_goal_for", "away_ev_shots_on_goal_against",
    "home_pp_shots_on_goal_for", "home_pp_shots_on_goal_against",
    "away_pp_shots_on_goal_for", "away_pp_shots_on_goal_against",
    "home_sh_shots_on_goal_for", "home_sh_shots_on_goal_against",
    "away_sh_shots_on_goal_for", "away_sh_shots_on_goal_against",
    "home_goals_for", "home_goals_against",
    "away_goals_for", "away_goals_against",
    "home_ev_toi", "home_pp_toi", "home_sh_toi", "home_en_toi",
    "away_ev_toi", "away_pp_toi", "away_sh_toi", "away_en_toi",
    "home_xgf_total", "home_xga_total", "away_xgf_total", "away_xga_total",
    "home_xgf_sog_total", "home_xga_sog_total",
    "away_xgf_sog_total", "away_xga_sog_total",
    # xG by strength — same-game actuals
    "home_xgf_ev", "home_xga_ev", "away_xgf_ev", "away_xga_ev",
    "home_xgf_pp", "home_xga_pp", "away_xgf_pp", "away_xga_pp",
    "home_xgf_sh", "home_xga_sh", "away_xgf_sh", "away_xga_sh",
    "home_xgf_sog_ev", "home_xga_sog_ev", "away_xgf_sog_ev", "away_xga_sog_ev",
    "home_xgf_sog_pp", "home_xga_sog_pp", "away_xgf_sog_pp", "away_xga_sog_pp",
    "home_xgf_sog_sh", "home_xga_sog_sh", "away_xgf_sog_sh", "away_xga_sog_sh",
    # PP%/PK% single game
    "home_pp_pct", "away_pp_pct", "home_pk_pct", "away_pk_pct",
]
for p in raw_leak_patterns:
    EXCLUDE.add(p)

# Exclude any remaining raw single-game per-60 or rate cols
for col in df.columns:
    if any(col.startswith(p) for p in
           ["home_ev_","home_pp_","home_sh_","away_ev_","away_pp_","away_sh_"]):
        if not any(s in col for s in
                   ["last10","last20","last30","season_avg","cumulative",
                    "weighted","opp_adj","per60_last","per60_season","per60_cumul"]):
            EXCLUDE.add(col)

feature_cols = [c for c in df.columns
                if c not in EXCLUDE
                and df[c].dtype in [np.float64, np.int64, float, int]
                and not c.startswith("Unnamed")]

print(f"\n  Candidate features: {len(feature_cols)}")

# ── Prepare ────────────────────────────────────────────────────────────────────
def prep(data, cols):
    X = data[cols].copy()
    for c in X.columns:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
    return X

X_train_full = prep(train, feature_cols)
X_test_full  = prep(test,  feature_cols)
y_train = train["game_total_shots"]
y_test  = test["game_total_shots"]

# ── Feature selection ──────────────────────────────────────────────────────────
print("\nSelecting top features...")
selector = xgb.XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    random_state=42, n_jobs=-1, eval_metric="rmse"
)
selector.fit(X_train_full, y_train)
importance = pd.Series(selector.feature_importances_, index=feature_cols)
top_features = importance.nlargest(80).index.tolist()
print(f"  Top 20 features:")
for f, v in importance.nlargest(20).items():
    print(f"    {f:<60} {v:.4f}")

X_train = X_train_full[top_features]
X_test  = X_test_full[top_features]

# ── Train ──────────────────────────────────────────────────────────────────────
val_split = int(len(X_train) * 0.8)
X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

print(f"\nTraining on {len(X_tr):,} games, validating on {len(X_val):,}...")
model = xgb.XGBRegressor(
    n_estimators=1000, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, gamma=1,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="rmse", early_stopping_rounds=30,
    random_state=42, n_jobs=-1,
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
bias   = float(np.mean(y_pred - y_test))
rmse   = float(np.sqrt(np.mean((y_pred - y_test)**2)))
baseline_mae = mean_absolute_error(y_test, [y_test.mean()] * len(y_test))
residual_std = float(np.std(y_test.values - y_pred))

print(f"\n{'='*50}")
print(f"GAME TOTAL SHOTS MODEL — TEST SET")
print(f"{'='*50}")
print(f"  MAE:          {mae:.3f}  (baseline mean: {baseline_mae:.3f})")
print(f"  RMSE:         {rmse:.3f}")
print(f"  Bias:         {bias:+.3f}")
print(f"  Residual std: {residual_std:.3f}")
print(f"  Naive sum MAE:{6.671:.3f}  (empirical formula)")
print(f"  Pred mean:    {np.mean(y_pred):.2f}  (actual: {y_test.mean():.2f})")
print(f"  Pred std:     {np.std(y_pred):.2f}  (actual: {y_test.std():.2f})")

# Error distribution
print(f"\n  50th pct abs error: {np.percentile(np.abs(y_pred-y_test), 50):.2f}")
print(f"  75th pct abs error: {np.percentile(np.abs(y_pred-y_test), 75):.2f}")
print(f"  90th pct abs error: {np.percentile(np.abs(y_pred-y_test), 90):.2f}")

# ── Save ───────────────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "game_total_shots.json")
model.save_model(model_path)
print(f"\n  Saved model → {model_path}")

# Save feature list and residual std alongside team models
with open(os.path.join(MODEL_DIR, "feature_lists.pkl"), "rb") as f:
    feature_lists = pickle.load(f)
feature_lists["game_total_shots"] = top_features
with open(os.path.join(MODEL_DIR, "feature_lists.pkl"), "wb") as f:
    pickle.dump(feature_lists, f)

with open(os.path.join(MODEL_DIR, "residual_stds.pkl"), "rb") as f:
    residual_stds = pickle.load(f)
residual_stds["game_total_shots"] = residual_std
with open(os.path.join(MODEL_DIR, "residual_stds.pkl"), "wb") as f:
    pickle.dump(residual_stds, f)

print(f"  Saved feature list + residual std to pkl files")
print(f"\nDone.")