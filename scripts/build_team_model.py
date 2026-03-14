import pandas as pd
import numpy as np
import os
import pickle
from scipy import stats
from sklearn.metrics import (roc_auc_score, log_loss, mean_absolute_error,
                             mean_squared_error, brier_score_loss)
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
DATA_PATH = os.path.expanduser("~/nhl-props/data/processed/team_features.csv")
MODEL_DIR = os.path.expanduser("~/nhl-props/models/team")
os.makedirs(MODEL_DIR, exist_ok=True)

# Target stats to predict (one model per target)
TARGETS = [
    # Total goals (Poisson - small counts)
    ("home_goals_for",                      "regression"),
    ("away_goals_for",                      "regression"),
    # Per-60 shot rates at each strength (rate stats, not circular with TOI)
    ("home_ev_shots_on_goal_for_per60",     "regression"),
    ("away_ev_shots_on_goal_for_per60",     "regression"),
    ("home_pp_shots_on_goal_for_per60",     "regression"),
    ("away_pp_shots_on_goal_for_per60",     "regression"),
    ("home_sh_shots_on_goal_for_per60",     "regression"),
    ("away_sh_shots_on_goal_for_per60",     "regression"),
    # Per-60 Fenwick at each strength
    ("home_ev_fenwick_for_per60",           "regression"),
    ("away_ev_fenwick_for_per60",           "regression"),
    ("home_pp_fenwick_for_per60",           "regression"),
    ("away_pp_fenwick_for_per60",           "regression"),
    # Per-60 shot attempts (Corsi)
    ("home_ev_shot_attempts_for_per60",     "regression"),
    ("away_ev_shot_attempts_for_per60",     "regression"),
    # xG total
    ("home_xgf_total",                      "regression"),
    ("away_xgf_total",                      "regression"),
    # Strength TOI
    ("home_ev_toi",                         "regression"),
    ("away_ev_toi",                         "regression"),
    ("home_pp_toi",                         "regression"),
    ("away_pp_toi",                         "regression"),
    # Win probability
    ("home_won",                            "classifier"),
]

# Features to exclude from training (metadata and raw game outcomes)
EXCLUDE_COLS = [
    "game_id", "season", "date", "home_team", "away_team",
    "home_won", "went_to_ot",
    # Raw game outcomes - data leakage
    "home_goals_for", "home_goals_against",
    "away_goals_for", "away_goals_against",
    "home_ev_goals_for", "home_ev_goals_against",
    "away_ev_goals_for", "away_ev_goals_against",
    "home_pp_goals_for", "home_pp_goals_against",
    "away_pp_goals_for", "away_pp_goals_against",
    "home_sh_goals_for", "home_sh_goals_against",
    "away_sh_goals_for", "away_sh_goals_against",
    "home_ev_shots_on_goal_for", "home_ev_shots_on_goal_against",
    "away_ev_shots_on_goal_for", "away_ev_shots_on_goal_against",
    "home_pp_shots_on_goal_for", "home_pp_shots_on_goal_against",
    "away_pp_shots_on_goal_for", "away_pp_shots_on_goal_against",
    "home_sh_shots_on_goal_for", "home_sh_shots_on_goal_against",
    "away_sh_shots_on_goal_for", "away_sh_shots_on_goal_against",
    "home_total_shots_on_goal_for", "home_total_shots_on_goal_against",
    "away_total_shots_on_goal_for", "away_total_shots_on_goal_against",
    "home_ev_shot_attempts_for", "home_ev_shot_attempts_against",
    "away_ev_shot_attempts_for", "away_ev_shot_attempts_against",
    "home_pp_shot_attempts_for", "home_pp_shot_attempts_against",
    "away_pp_shot_attempts_for", "away_pp_shot_attempts_against",
    "home_sh_shot_attempts_for", "home_sh_shot_attempts_against",
    "away_sh_shot_attempts_for", "away_sh_shot_attempts_against",
    "home_total_shot_attempts_for", "home_total_shot_attempts_against",
    "away_total_shot_attempts_for", "away_total_shot_attempts_against",
    "home_ev_fenwick_for", "home_ev_fenwick_against",
    "away_ev_fenwick_for", "away_ev_fenwick_against",
    "home_pp_fenwick_for", "home_pp_fenwick_against",
    "away_pp_fenwick_for", "away_pp_fenwick_against",
    "home_sh_fenwick_for", "home_sh_fenwick_against",
    "away_sh_fenwick_for", "away_sh_fenwick_against",
    "home_total_fenwick_for", "home_total_fenwick_against",
    "away_total_fenwick_for", "away_total_fenwick_against",
    "home_xgf_total", "home_xga_total",
    "away_xgf_total", "away_xga_total",
    "home_xgf_ev", "home_xga_ev",
    "away_xgf_ev", "away_xga_ev",
    "home_xgf_pp", "home_xga_pp",
    "away_xgf_pp", "away_xga_pp",
    "home_xgf_sh", "home_xga_sh",
    "away_xgf_sh", "away_xga_sh",
    "home_ev_toi", "home_pp_toi", "home_sh_toi", "home_en_toi",
    "away_ev_toi", "away_pp_toi", "away_sh_toi", "away_en_toi",
    "home_ev_hits_for", "home_ev_hits_against",
    "away_ev_hits_for", "away_ev_hits_against",
    "home_ev_giveaways", "home_ev_takeaways",
    "away_ev_giveaways", "away_ev_takeaways",
    "home_ev_faceoffs_won", "home_ev_faceoffs_taken",
    "away_ev_faceoffs_won", "away_ev_faceoffs_taken",
    "home_ev_missed_shots_for", "home_ev_missed_shots_against",
    "away_ev_missed_shots_for", "away_ev_missed_shots_against",
    "home_ev_blocked_shots_for", "home_ev_blocked_shots_against",
    "away_ev_blocked_shots_for", "away_ev_blocked_shots_against",
    "home_pp_missed_shots_for", "home_pp_missed_shots_against",
    "away_pp_missed_shots_for", "away_pp_missed_shots_against",
    "home_pp_blocked_shots_for", "home_pp_blocked_shots_against",
    "away_pp_blocked_shots_for", "away_pp_blocked_shots_against",
    "home_pp_hits_for", "home_pp_hits_against",
    "away_pp_hits_for", "away_pp_hits_against",
    "home_pp_giveaways", "home_pp_takeaways",
    "away_pp_giveaways", "away_pp_takeaways",
    "home_pp_faceoffs_won", "home_pp_faceoffs_taken",
    "away_pp_faceoffs_won", "away_pp_faceoffs_taken",
    "home_sh_missed_shots_for", "home_sh_missed_shots_against",
    "away_sh_missed_shots_for", "away_sh_missed_shots_against",
    "home_sh_blocked_shots_for", "home_sh_blocked_shots_against",
    "away_sh_blocked_shots_for", "away_sh_blocked_shots_against",
    "home_sh_hits_for", "home_sh_hits_against",
    "away_sh_hits_for", "away_sh_hits_against",
    "home_sh_giveaways", "home_sh_takeaways",
    "away_sh_giveaways", "away_sh_takeaways",
    "home_sh_faceoffs_won", "home_sh_faceoffs_taken",
    "away_sh_faceoffs_won", "away_sh_faceoffs_taken",
    "home_ev_penalties_taken", "away_ev_penalties_taken",
    "home_ev_penalties_drawn", "away_ev_penalties_drawn",
    "home_ev_penalty_minutes", "away_ev_penalty_minutes",
    "home_pp_penalties_taken", "away_pp_penalties_taken",
    "home_pp_penalties_drawn", "away_pp_penalties_drawn",
    "home_pp_penalty_minutes", "away_pp_penalty_minutes",
    "home_sh_penalties_taken", "away_sh_penalties_taken",
    "home_sh_penalties_drawn", "away_sh_penalties_drawn",
    "home_sh_penalty_minutes", "away_sh_penalty_minutes",
    "home_pp_pct", "away_pp_pct",
    "home_pk_pct", "away_pk_pct",
    # Per-60 raw game outcomes - data leakage
    "home_ev_shots_on_goal_for_per60", "home_ev_shots_on_goal_against_per60",
    "away_ev_shots_on_goal_for_per60", "away_ev_shots_on_goal_against_per60",
    "home_pp_shots_on_goal_for_per60", "home_pp_shots_on_goal_against_per60",
    "away_pp_shots_on_goal_for_per60", "away_pp_shots_on_goal_against_per60",
    "home_sh_shots_on_goal_for_per60", "home_sh_shots_on_goal_against_per60",
    "away_sh_shots_on_goal_for_per60", "away_sh_shots_on_goal_against_per60",
    "home_ev_fenwick_for_per60", "home_ev_fenwick_against_per60",
    "away_ev_fenwick_for_per60", "away_ev_fenwick_against_per60",
    "home_pp_fenwick_for_per60", "home_pp_fenwick_against_per60",
    "away_pp_fenwick_for_per60", "away_pp_fenwick_against_per60",
    "home_sh_fenwick_for_per60", "home_sh_fenwick_against_per60",
    "away_sh_fenwick_for_per60", "away_sh_fenwick_against_per60",
    "home_ev_shot_attempts_for_per60", "home_ev_shot_attempts_against_per60",
    "away_ev_shot_attempts_for_per60", "away_ev_shot_attempts_against_per60",
    "home_ev_goals_for_per60", "home_ev_goals_against_per60",
    "away_ev_goals_for_per60", "away_ev_goals_against_per60",
    "home_pp_goals_for_per60", "home_pp_goals_against_per60",
    "away_pp_goals_for_per60", "away_pp_goals_against_per60",
    "home_sh_goals_for_per60", "home_sh_goals_against_per60",
    "away_sh_goals_for_per60", "away_sh_goals_against_per60",
    "home_ev_hits_for_per60", "home_ev_hits_against_per60",
    "away_ev_hits_for_per60", "away_ev_hits_against_per60",
    "home_ev_giveaways_per60", "home_ev_takeaways_per60",
    "away_ev_giveaways_per60", "away_ev_takeaways_per60",
]

# Store residual stds from training for Normal distributions
RESIDUAL_STDS = {}


def load_data(path):
    """Load and prepare the team features dataset."""
    print("Loading team features...")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  {len(df)} games loaded")
    print(f"  Seasons: {df['season'].value_counts().sort_index().to_dict()}")
    return df


def get_feature_cols(df):
    """Get all valid feature columns (exclude metadata and targets)."""
    exclude = set(EXCLUDE_COLS)
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


def prepare_features(df, feature_cols):
    """Prepare feature matrix, filling nulls with column medians."""
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    return X


def select_features(X, y, model_type, n_features=80):
    """Use XGBoost feature importance to select top N features."""
    if model_type == "classifier":
        selector = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric="logloss"
        )
    else:
        selector = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric="rmse"
        )
    selector.fit(X, y)
    importance = pd.Series(selector.feature_importances_, index=X.columns)
    top_features = importance.nlargest(n_features).index.tolist()
    return top_features


def train_model(X_train, y_train, X_val, y_val, model_type, target_name):
    """Train an XGBoost model with early stopping."""
    if model_type == "classifier":
        model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            eval_metric="logloss",
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
    else:
        model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="rmse",
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        # Compute residual std on validation set
        val_pred = model.predict(X_val)
        residuals = y_val.values - val_pred
        RESIDUAL_STDS[target_name] = float(np.std(residuals))

    return model


def evaluate_regression(y_true, y_pred, target_name):
    """Print regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    baseline_mae = mean_absolute_error(y_true, [y_true.mean()] * len(y_true))
    print(f"  MAE:  {mae:.3f}  (baseline: {baseline_mae:.3f})")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  Mean actual: {y_true.mean():.2f}  Mean predicted: {y_pred.mean():.2f}")


def evaluate_classifier(y_true, y_pred_proba, target_name):
    """Print classifier evaluation metrics."""
    auc = roc_auc_score(y_true, y_pred_proba)
    ll = log_loss(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    baseline_ll = log_loss(y_true, [y_true.mean()] * len(y_true))
    print(f"  AUC:    {auc:.4f}")
    print(f"  LogLoss:{ll:.4f}  (baseline: {baseline_ll:.4f})")
    print(f"  Brier:  {brier:.4f}")


def predict_with_distribution(model, X, model_type, target_name):
    """
    Generate point prediction and probability distribution.
    Goals: Poisson distribution.
    Everything else: Normal distribution with sigma from training residuals.
    Classifier: raw probability.
    """
    poisson_targets = {"home_goals_for", "away_goals_for"}

    if model_type == "classifier":
        prob = model.predict_proba(X)[:, 1]
        return {"prediction": prob, "distribution": None}

    point_pred = model.predict(X)

    if target_name in poisson_targets:
        distributions = []
        for pred in point_pred:
            lam = max(pred, 0.01)
            probs = [stats.poisson.pmf(k, lam) for k in range(16)]
            distributions.append({
                "type": "poisson",
                "lambda": lam,
                "probs": probs,
                "over_0.5": 1 - stats.poisson.cdf(0, lam),
                "over_1.5": 1 - stats.poisson.cdf(1, lam),
                "over_2.5": 1 - stats.poisson.cdf(2, lam),
                "over_3.5": 1 - stats.poisson.cdf(3, lam),
                "over_4.5": 1 - stats.poisson.cdf(4, lam),
            })
    else:
        sigma = RESIDUAL_STDS.get(target_name, point_pred.mean() * 0.22)
        distributions = []
        for pred in point_pred:
            mu = max(pred, 0.01)
            distributions.append({
                "type": "normal",
                "mu": mu,
                "sigma": sigma,
            })

    return {"prediction": point_pred, "distribution": distributions}


def print_sample_prediction(model, X_sample, model_type, target_name):
    """Print a sample prediction with distribution for one game."""
    result = predict_with_distribution(model, X_sample, model_type, target_name)
    pred = result["prediction"][0]

    if model_type == "classifier":
        print(f"  Probability: {pred:.1%}")
    else:
        dist = result["distribution"][0]
        if dist["type"] == "poisson":
            lam = dist["lambda"]
            print(f"  Projected: {pred:.2f}")
            print(f"  Distribution: ", end="")
            for k in range(8):
                print(f"P({k})={dist['probs'][k]:.1%}", end="  ")
            print()
            print(f"  Over 0.5: {dist['over_0.5']:.1%}  "
                  f"Over 1.5: {dist['over_1.5']:.1%}  "
                  f"Over 2.5: {dist['over_2.5']:.1%}  "
                  f"Over 3.5: {dist['over_3.5']:.1%}")
        else:
            mu = dist["mu"]
            sigma = dist["sigma"]
            print(f"  Projected: {pred:.2f} (±{sigma:.2f})")
            print(f"  68% range: {mu-sigma:.2f} to {mu+sigma:.2f}")
            print(f"  90% range: {mu-1.645*sigma:.2f} to {mu+1.645*sigma:.2f}")
            if "per60" in target_name:
                for threshold in [20, 25, 30, 35, 40]:
                    prob = float(1 - stats.norm.cdf(threshold, mu, sigma))
                    print(f"  Over {threshold}/60: {prob:.1%}", end="  ")
                print()
            elif "toi" in target_name:
                for threshold in [3, 5, 7, 9]:
                    prob = float(1 - stats.norm.cdf(threshold, mu, sigma))
                    print(f"  Over {threshold}min: {prob:.1%}", end="  ")
                print()


def main():
    # Load data
    df = load_data(DATA_PATH)

    # Train/test split — use 20252026 as test
    train_df = df[df["season"] != 20252026].copy()
    test_df = df[df["season"] == 20252026].copy()
    print(f"\nTrain: {len(train_df)} games | Test: {len(test_df)} games")

    # Get feature columns
    feature_cols = get_feature_cols(df)
    print(f"Total features available: {len(feature_cols)}")

    # Prepare full feature matrices
    X_train_full = prepare_features(train_df, feature_cols)
    X_test_full = prepare_features(test_df, feature_cols)

    # Store all trained models and results
    all_models = {}
    all_features = {}
    results = {}

    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)

    for target_name, model_type in TARGETS:
        print(f"\n--- {target_name} ({model_type}) ---")

        if target_name not in df.columns:
            print(f"  WARNING: {target_name} not found in data, skipping")
            continue

        y_train = train_df[target_name].fillna(0)
        y_test = test_df[target_name].fillna(0)

        # Feature selection on training data only
        print(f"  Selecting top features...")
        top_features = select_features(X_train_full, y_train, model_type, n_features=80)
        X_train = X_train_full[top_features]
        X_test = X_test_full[top_features]

        # Validation split for early stopping
        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
        y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

        # Train
        print(f"  Training on {len(X_tr)} games, validating on {len(X_val)}...")
        model = train_model(X_tr, y_tr, X_val, y_val, model_type, target_name)

        # Evaluate on test set
        if model_type == "classifier":
            y_pred = model.predict_proba(X_test)[:, 1]
            evaluate_classifier(y_test, y_pred, target_name)
        else:
            y_pred = model.predict(X_test)
            evaluate_regression(y_test, y_pred, target_name)

        # Sample prediction
        print(f"  Sample prediction (first test game):")
        print_sample_prediction(model, X_test.iloc[[0]], model_type, target_name)

        # Store
        all_models[target_name] = model
        all_features[target_name] = top_features
        results[target_name] = {
            "model_type": model_type,
            "n_features": len(top_features),
            "test_mae": mean_absolute_error(y_test, y_pred) if model_type == "regression" else None,
            "test_auc": roc_auc_score(y_test, y_pred) if model_type == "classifier" else None,
        }

    # Save models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    for target_name, model in all_models.items():
        model_path = os.path.join(MODEL_DIR, f"{target_name}.json")
        model.save_model(model_path)
        print(f"  Saved: {target_name}.json")

    features_path = os.path.join(MODEL_DIR, "feature_lists.pkl")
    with open(features_path, "wb") as f:
        pickle.dump(all_features, f)

    residuals_path = os.path.join(MODEL_DIR, "residual_stds.pkl")
    with open(residuals_path, "wb") as f:
        pickle.dump(RESIDUAL_STDS, f)

    print(f"  Saved: feature_lists.pkl")
    print(f"  Saved: residual_stds.pkl")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for target_name, res in results.items():
        if res["model_type"] == "regression":
            print(f"  {target_name:<45} MAE: {res['test_mae']:.3f}")
        else:
            print(f"  {target_name:<45} AUC: {res['test_auc']:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()