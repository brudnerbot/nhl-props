import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import xgboost as xgb

# --- CONFIG ---
DATA_PATH = os.path.expanduser("~/nhl-props/data/raw/shot_data/shot_data.csv")
MODEL_DIR = os.path.expanduser("~/nhl-props/models")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- FEATURES ---
# Only use shots on goal and goals for xG model
# (blocked and missed shots never had a chance to be goals)
SHOT_TYPES = ["wrist", "slap", "snap", "tip-in", "backhand", "deflected", "wrap-around"]


def load_and_prepare(path):
    print("Loading shot data...")
    df = pd.read_csv(path)
    print(f"  Total shots: {len(df)}")

   # Train only on shots on goal and goals (clean save/goal outcome)
    # We'll apply xG to missed shots separately after training
    df_train = df[df["event_type"].isin(["shot-on-goal", "goal"])].copy()
    df_missed = df[df["event_type"] == "missed-shot"].copy()
    print(f"  Shots on goal + goals (training data): {len(df_train)}")
    print(f"  Missed shots (xG applied after training): {len(df_missed)}")
    df = df_train  # train on clean outcomes only

    # Drop empty net shots - goalie not present, not useful for xG
    df = df[df["is_empty_net"] == 0].copy()
    print(f"  After removing empty net shots: {len(df)}")

    # Drop rows missing key features
    df = df.dropna(subset=["distance", "angle", "strength"]).copy()
    print(f"  After dropping missing key features: {len(df)}")

    # Fill missing prev event features with neutral values
    df["speed_from_prev"] = df["speed_from_prev"].fillna(0)
    df["prev_distance"] = df["prev_distance"].fillna(0)
    df["rebound_angle_change"] = df["rebound_angle_change"].fillna(0)
    df["is_rebound"] = df["is_rebound"].fillna(0)

    # Encode shot type
    df["shot_type"] = df["shot_type"].fillna("unknown")
    shot_type_encoder = LabelEncoder()
    df["shot_type_enc"] = shot_type_encoder.fit_transform(df["shot_type"])

    # Encode prev event type
    df["prev_event_type"] = df["prev_event_type"].fillna("unknown")
    prev_event_encoder = LabelEncoder()
    df["prev_event_type_enc"] = prev_event_encoder.fit_transform(df["prev_event_type"])

    # Encode strength
    strength_map = {"ev": 0, "pp": 1, "sh": 2}
    df["strength_enc"] = df["strength"].map(strength_map).fillna(0)

    # Period - cap at 4 (treat all OT periods the same)
    df["period_adj"] = df["period"].clip(upper=4)

    print(f"\nGoal rate in filtered dataset: {df['is_goal'].mean()*100:.1f}%")
    print(f"Total goals: {df['is_goal'].sum()}")

    return df, df_missed, shot_type_encoder, prev_event_encoder


def build_features(df):
    """Select and return feature matrix."""
    features = [
        "distance",           # distance from net
        "angle",              # shot angle
        "shot_type_enc",      # shot type
        "is_rebound",         # rebound shot flag
        "rebound_angle_change", # change in angle for rebounds
        "speed_from_prev",    # speed from previous event (rush shots)
        "prev_distance",      # distance from previous event
        "prev_event_type_enc", # what happened before the shot
        "strength_enc",       # EV/PP/SH
        "period_adj",         # period
    ]
    return df[features], df["is_goal"]


def evaluate_model(model, X_test, y_test):
    """Print model evaluation metrics."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    ll = log_loss(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    print(f"\nModel Evaluation:")
    print(f"  AUC-ROC:     {auc:.4f}  (higher is better, 0.5 = random)")
    print(f"  Log Loss:    {ll:.4f}  (lower is better)")
    print(f"  Brier Score: {brier:.4f}  (lower is better, 0 = perfect)")
    print(f"  Baseline log loss (always predict mean): {log_loss(y_test, [y_test.mean()]*len(y_test)):.4f}")

    return y_pred_proba


def print_feature_importance(model, feature_names):
    """Print feature importances."""
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)
    print("\nFeature Importances:")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"  {row['feature']:<25} {row['importance']:.4f} {bar}")


def main():
    # Load and prepare data
    df, df_missed, shot_type_encoder, prev_event_encoder = load_and_prepare(DATA_PATH)

    # Build features
    print("\nBuilding features...")
    X, y = build_features(df)
    feature_names = X.columns.tolist()

    # Train/test split - use 20252026 season as test set
    # This mirrors real-world usage: train on past, predict current season
    train_mask = df["season"] != 20252026
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    print(f"\nTrain set: {len(X_train)} shots ({y_train.sum()} goals)")
    print(f"Test set:  {len(X_test)} shots ({y_test.sum()} goals)")

    # Train XGBoost model
    print("\nTraining XGBoost xG model...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    y_pred_proba = evaluate_model(model, X_test, y_test)

    # Feature importance
    print_feature_importance(model, feature_names)

   # Add xG predictions back to full dataset including missed shots
    print("\nGenerating xG values for full dataset...")
    df["xg"] = model.predict_proba(X)[:, 1]

    # Apply xG to missed shots too
    df_missed = df_missed[df_missed["distance"].notna() & df_missed["angle"].notna()].copy()
    df_missed["shot_type"] = df_missed["shot_type"].fillna("unknown")
    df_missed["shot_type_enc"] = df_missed["shot_type"].map(
        dict(zip(shot_type_encoder.classes_, shot_type_encoder.transform(shot_type_encoder.classes_)))
    ).fillna(0)
    df_missed["prev_event_type"] = df_missed["prev_event_type"].fillna("unknown")
    df_missed["prev_event_type_enc"] = df_missed["prev_event_type"].map(
        dict(zip(prev_event_encoder.classes_, prev_event_encoder.transform(prev_event_encoder.classes_)))
    ).fillna(0)
    df_missed["strength_enc"] = df_missed["strength"].map({"ev": 0, "pp": 1, "sh": 2}).fillna(0)
    df_missed["period_adj"] = df_missed["period"].clip(upper=4)
    df_missed["speed_from_prev"] = df_missed["speed_from_prev"].fillna(0)
    df_missed["prev_distance"] = df_missed["prev_distance"].fillna(0)
    df_missed["rebound_angle_change"] = df_missed["rebound_angle_change"].fillna(0)
    df_missed["is_rebound"] = df_missed["is_rebound"].fillna(0)
    df_missed["is_empty_net"] = df_missed["is_empty_net"].fillna(0)
    X_missed, _ = build_features(df_missed)
    df_missed["xg"] = model.predict_proba(X_missed)[:, 1]

    # Combine all shots with xG values
    df_all = pd.concat([df, df_missed], ignore_index=True)
    df_all = df_all.sort_values(["game_id", "period", "time_seconds"]).reset_index(drop=True)

    # Quick sanity check - top xG shots should be close rebounds
    print("\nTop 10 highest xG shots:")
    top_xg = df.nlargest(10, "xg")[["date", "shooting_team", "shot_type",
                                     "distance", "angle", "is_rebound",
                                     "is_goal", "xg"]]
    print(top_xg.to_string())

    # Save model and encoders
    print("\nSaving model...")
    model.save_model(os.path.join(MODEL_DIR, "xg_model.json"))
    with open(os.path.join(MODEL_DIR, "xg_encoders.pkl"), "wb") as f:
        pickle.dump({
            "shot_type_encoder": shot_type_encoder,
            "prev_event_encoder": prev_event_encoder,
        }, f)

    # Save shot data with xG values
    xg_output_path = os.path.expanduser("~/nhl-props/data/processed/shot_data_with_xg.csv")
    os.makedirs(os.path.dirname(xg_output_path), exist_ok=True)
    df_all.to_csv(xg_output_path, index=False)
    print(f"Shot data with xG saved to {xg_output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()