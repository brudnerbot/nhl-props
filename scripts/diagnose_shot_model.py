"""
diagnose_shot_model.py

Evaluates EV shot prediction accuracy on the 2025-2026 test set.
Each row in team_features.csv is a full game (home + away columns).
We predict home EV shots and away EV shots separately, then combine.
"""

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
FEATURES_F = ROOT / "data/processed/team_features.csv"
FEAT_LISTS = ROOT / "models/team/feature_lists.pkl"
MODEL_DIR  = ROOT / "models/team"
OUT_DIR    = ROOT / "notebooks"
OUT_DIR.mkdir(exist_ok=True)

TEST_SEASON = 20252026

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading team_features.csv ...")
df = pd.read_csv(FEATURES_F, low_memory=False)
print(f"  Total rows: {len(df):,}  |  Columns: {df.shape[1]}")

test = df[df["season"] == TEST_SEASON].copy()
print(f"  Test rows (season {TEST_SEASON}): {len(test):,}")

with open(FEAT_LISTS, "rb") as f:
    feature_lists = pickle.load(f)

# ── Load model helper ──────────────────────────────────────────────────────────
def load_xgb(name):
    m = xgb.XGBRegressor()
    m.load_model(str(MODEL_DIR / f"{name}.json"))
    return m

home_sog_model = load_xgb("home_ev_shots_on_goal_for_per60")
away_sog_model = load_xgb("away_ev_shots_on_goal_for_per60")
home_toi_model = load_xgb("home_ev_toi")
away_toi_model = load_xgb("away_ev_toi")

home_sog_feats = feature_lists["home_ev_shots_on_goal_for_per60"]
away_sog_feats = feature_lists["away_ev_shots_on_goal_for_per60"]
home_toi_feats = feature_lists["home_ev_toi"]
away_toi_feats = feature_lists["away_ev_toi"]

# ── Predict EV shots ───────────────────────────────────────────────────────────
def predict_ev_shots(df, sog_model, sog_feats, toi_model, toi_feats):
    X_sog = df[sog_feats].fillna(0)
    X_toi = df[toi_feats].fillna(0)
    sog60 = sog_model.predict(X_sog)
    toi   = toi_model.predict(X_toi)
    return sog60 * (toi / 60.0), sog60, toi

test["home_pred_ev_shots"], test["home_pred_sog60"], test["home_pred_toi"] = predict_ev_shots(
    test, home_sog_model, home_sog_feats, home_toi_model, home_toi_feats)

test["away_pred_ev_shots"], test["away_pred_sog60"], test["away_pred_toi"] = predict_ev_shots(
    test, away_sog_model, away_sog_feats, away_toi_model, away_toi_feats)

# ── Actual EV shots columns ────────────────────────────────────────────────────
# Confirm which columns hold actuals
actual_home = "home_ev_shots_on_goal_for"
actual_away = "away_ev_shots_on_goal_for"
for c in [actual_home, actual_away]:
    assert c in test.columns, f"Missing column: {c}"
print(f"Actual columns: '{actual_home}', '{actual_away}'")

# ── Build long-form results: one row per team per game ─────────────────────────
home_rows = test[["home_team", actual_home, "home_pred_ev_shots",
                   "home_pred_sog60", "home_pred_toi"]].copy()
home_rows.columns = ["team", "actual", "pred", "pred_sog60", "pred_toi"]
home_rows["split"] = "home"

away_rows = test[["away_team", actual_away, "away_pred_ev_shots",
                   "away_pred_sog60", "away_pred_toi"]].copy()
away_rows.columns = ["team", "actual", "pred", "pred_sog60", "pred_toi"]
away_rows["split"] = "away"

results = pd.concat([home_rows, away_rows], ignore_index=True)
results["residual"]  = results["pred"] - results["actual"]
results["abs_error"] = results["residual"].abs()

# ── Overall metrics ────────────────────────────────────────────────────────────
mae  = results["abs_error"].mean()
rmse = np.sqrt((results["residual"] ** 2).mean())
bias = results["residual"].mean()

print(f"\n{'='*55}")
print(f"EV SHOTS — TEST SET METRICS  (season {TEST_SEASON})")
print(f"{'='*55}")
print(f"  N     : {len(results):,} team-game observations")
print(f"  MAE   : {mae:.3f}")
print(f"  RMSE  : {rmse:.3f}")
print(f"  Bias  : {bias:+.3f}  ({'over' if bias > 0 else 'under'}-predicting)")
print(f"  Actual mean  : {results['actual'].mean():.2f}")
print(f"  Pred mean    : {results['pred'].mean():.2f}")

for split in ["home", "away"]:
    s = results[results["split"] == split]
    print(f"  {split.capitalize():5s} | MAE: {s['abs_error'].mean():.3f}  "
          f"Bias: {s['residual'].mean():+.3f}  "
          f"Actual avg: {s['actual'].mean():.2f}  Pred avg: {s['pred'].mean():.2f}")

# ── Per-team residuals ─────────────────────────────────────────────────────────
team_errors = results.groupby("team").agg(
    games      =("abs_error", "count"),
    mae        =("abs_error", "mean"),
    bias       =("residual", "mean"),
    actual_avg =("actual", "mean"),
    pred_avg   =("pred", "mean"),
).sort_values("bias")

print(f"\n{'='*55}")
print("PER-TEAM RESIDUALS  (sorted worst under → worst over)")
print(f"{'='*55}")
print(team_errors.to_string())
team_errors.to_csv(OUT_DIR / "team_ev_shot_residuals.csv")
print(f"\n  Saved → notebooks/team_ev_shot_residuals.csv")

# ── Error percentile breakdown ────────────────────────────────────────────────
print(f"\n{'='*55}")
print("ERROR DISTRIBUTION")
print(f"{'='*55}")
for p in [50, 75, 90, 95]:
    print(f"  {p}th percentile abs error: {np.percentile(results['abs_error'], p):.2f}")
print(f"  Games with abs_error > 10 : {(results['abs_error'] > 10).sum()}")
print(f"  Games with abs_error > 15 : {(results['abs_error'] > 15).sum()}")

# ── Residual correlations ──────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("RESIDUAL CORRELATIONS (home side, numeric features)")
print(f"{'='*55}")
corr_cols = [c for c in test.columns if any(x in c for x in
    ["days_rest", "back_to_back", "game_number", "toi_season",
     "opp_adj", "cumulative", "weighted"]
) and test[c].dtype in [np.float64, np.int64]][:30]

if corr_cols:
    tmp = test[corr_cols].copy()
    tmp["residual"] = test["home_pred_ev_shots"].values - test[actual_home].values
    corr = tmp.corr()["residual"].drop("residual").sort_values()
    print("  Most negative (model over-predicts when high):")
    print(corr.head(5).to_string())
    print("  Most positive (model under-predicts when high):")
    print(corr.tail(5).to_string())

# ── Feature importances ────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("TOP 20 FEATURES — home_ev_shots_on_goal_for_per60 model")
print(f"{'='*55}")
fi = pd.Series(home_sog_model.feature_importances_,
               index=home_sog_feats).sort_values(ascending=False)
print(fi.head(20).to_string())
fi.to_csv(OUT_DIR / "home_ev_sog_feature_importance.csv", header=["importance"])

print(f"\n{'='*55}")
print("TOP 20 FEATURES — home_ev_toi model")
print(f"{'='*55}")
fi_toi = pd.Series(home_toi_model.feature_importances_,
                   index=home_toi_feats).sort_values(ascending=False)
print(fi_toi.head(20).to_string())

# ── Plots ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig)
fig.suptitle("EV Shot Model Diagnostics — Test Set 2025-26", fontsize=14, fontweight="bold")

ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(results["actual"], results["pred"], alpha=0.3, s=10, color="steelblue")
lims = [results[["actual","pred"]].min().min(), results[["actual","pred"]].max().max()]
ax1.plot(lims, lims, "r--", lw=1)
ax1.set_xlabel("Actual EV Shots"); ax1.set_ylabel("Predicted EV Shots")
ax1.set_title(f"Predicted vs Actual  (MAE={mae:.2f})")

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(results["residual"], bins=40, color="steelblue", edgecolor="white")
ax2.axvline(0, color="red", lw=1, linestyle="--")
ax2.set_xlabel("Residual (pred − actual)"); ax2.set_ylabel("Count")
ax2.set_title(f"Residual Distribution  (bias={bias:+.2f})")

ax3 = fig.add_subplot(gs[1, :])
tc = team_errors.sort_values("bias")
colors = ["#d73027" if b > 0 else "#4575b4" for b in tc["bias"]]
ax3.bar(tc.index, tc["bias"], color=colors)
ax3.axhline(0, color="black", lw=0.8)
ax3.set_xlabel("Team"); ax3.set_ylabel("Bias (pred − actual shots)")
ax3.set_title("Per-Team Prediction Bias  (blue=under-predict, red=over-predict)")
ax3.tick_params(axis="x", rotation=45)

plt.tight_layout()
plot_path = OUT_DIR / "shot_model_diagnostics.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\n  Saved → notebooks/shot_model_diagnostics.png")
plt.show()
print("\nDone.")