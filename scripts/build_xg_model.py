import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

# --- CONFIG ---
DATA_PATH = os.path.expanduser("~/nhl-props/data/raw/shot_data/shot_data.csv")
MODEL_DIR = os.path.expanduser("~/nhl-props/models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_prepare(path):
    print("Loading shot data...")
    df = pd.read_csv(path)
    print(f"  Total shots: {len(df)}")

    # Train only on shots on goal and goals (clean save/goal outcome)
    df_train = df[df["event_type"].isin(["shot-on-goal", "goal"])].copy()
    df_missed = df[df["event_type"] == "missed-shot"].copy()
    print(f"  Shots on goal + goals (training data): {len(df_train)}")
    print(f"  Missed shots (xG applied after training): {len(df_missed)}")
    df = df_train

    # Separate empty net shots - handled by separate EN xG model
    df_empty_net = df[df["is_empty_net"] == 1].copy()
    df_empty_net_missed = df_missed[df_missed["is_empty_net"] == 1].copy()
    df = df[df["is_empty_net"] == 0].copy()
    df_missed = df_missed[df_missed["is_empty_net"] == 0].copy()
    print(f"  After removing empty net shots from training: {len(df)}")
    print(f"  Empty net SOG/goals: {len(df_empty_net)}, missed: {len(df_empty_net_missed)}")

    # Drop rows missing key features
    df = df.dropna(subset=["distance", "angle"]).copy()
    print(f"  After dropping missing key features: {len(df)}")

    # Fill missing features
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

    # Period - cap at 4
    df["period_adj"] = df["period"].clip(upper=4)

    print(f"\nGoal rate in filtered dataset: {df['is_goal'].mean()*100:.1f}%")
    print(f"Total goals: {df['is_goal'].sum()}")

    return df, df_missed, df_empty_net, df_empty_net_missed, shot_type_encoder, prev_event_encoder


def build_features(df):
    """Select and return feature matrix."""
    features = [
        "distance",
        "angle",
        "shot_type_enc",
        "is_rebound",
        "rebound_angle_change",
        "speed_from_prev",
        "prev_distance",
        "prev_event_type_enc",
        "period_adj",
        # Deliberately excluding strength_enc - xG should be location/context based
    ]
    return df[features], df["is_goal"]


def evaluate_model(model, X_test, y_test):
    """Print model evaluation metrics."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    ll = log_loss(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    print(f"\nModel Evaluation:")
    print(f"  AUC-ROC:     {auc:.4f}  (higher is better, 0.5 = random)")
    print(f"  Log Loss:    {ll:.4f}  (lower is better)")
    print(f"  Brier Score: {brier:.4f}  (lower is better, 0 = perfect)")
    print(f"  Baseline log loss: {log_loss(y_test, [y_test.mean()]*len(y_test)):.4f}")
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


def build_empty_net_xg_model(df_empty_net_all, shot_type_encoder):
    """
    Train a separate xG model for empty net shots using logistic regression.
    Trained on unblocked EN shots (SOG + goals + missed shots).
    """
    print("\nBuilding empty net xG model...")

    en = df_empty_net_all[df_empty_net_all["event_type"].isin(
        ["shot-on-goal", "goal", "missed-shot"])].copy()

    print(f"  Empty net unblocked attempts: {len(en)}")
    print(f"  Empty net goal rate: {en['is_goal'].mean():.3f}")

    # Encode shot type
    en["shot_type"] = en["shot_type"].fillna("unknown")
    known_types = set(shot_type_encoder.classes_)
    en["shot_type_enc"] = en["shot_type"].apply(
        lambda x: shot_type_encoder.transform([x])[0] if x in known_types else 0
    )

    # Fill missing features
    en["speed_from_prev"] = en["speed_from_prev"].fillna(0)
    en["is_rebound"] = en["is_rebound"].fillna(0)
    en["rebound_angle_change"] = en["rebound_angle_change"].fillna(0)
    en["period_adj"] = en["period"].clip(upper=4)
    en = en.dropna(subset=["distance", "angle"]).copy()

    features = [
        "distance",
        "angle",
        "shot_type_enc",
        "is_rebound",
        "speed_from_prev",
        "rebound_angle_change",
        "period_adj",
    ]

    X = en[features].fillna(0)
    y = en["is_goal"]

    # Train/test split by season
    test_mask = en["season"] == 20252026
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    print(f"  Train: {len(X_train)} shots ({y_train.sum()} goals)")
    print(f"  Test:  {len(X_test)} shots ({y_test.sum()} goals)")

    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    ll = log_loss(y_test, y_pred)
    baseline_ll = log_loss(y_test, [y_test.mean()] * len(y_test))
    print(f"  AUC: {auc:.4f} | LogLoss: {ll:.4f} (baseline: {baseline_ll:.4f})")

    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", ascending=False)
    print("\n  Feature coefficients:")
    for _, row in coef_df.iterrows():
        print(f"    {row['feature']:<25} {row['coefficient']:+.3f}")

    return model, scaler, features


def apply_empty_net_xg(df_empty_net_all, en_model, en_scaler, en_features, shot_type_encoder):
    """Apply empty net xG model to all empty net shots."""
    df = df_empty_net_all.copy()

    df["shot_type"] = df["shot_type"].fillna("unknown")
    known_types = set(shot_type_encoder.classes_)
    df["shot_type_enc"] = df["shot_type"].apply(
        lambda x: shot_type_encoder.transform([x])[0] if x in known_types else 0
    )

    df["speed_from_prev"] = df["speed_from_prev"].fillna(0)
    df["is_rebound"] = df["is_rebound"].fillna(0)
    df["rebound_angle_change"] = df["rebound_angle_change"].fillna(0)
    df["period_adj"] = df["period"].clip(upper=4)
    df["distance"] = df["distance"].fillna(30)
    df["angle"] = df["angle"].fillna(15)

    X = df[en_features].fillna(0)
    X_scaled = en_scaler.transform(X)
    df["xg"] = en_model.predict_proba(X_scaled)[:, 1]

    print(f"  EN xG mean: {df['xg'].mean():.3f}  "
          f"Actual goal rate: {df['is_goal'].mean():.3f}")

    return df


def main():
    # Load and prepare data
    df, df_missed, df_empty_net, df_empty_net_missed, \
        shot_type_encoder, prev_event_encoder = load_and_prepare(DATA_PATH)

    # Build features
    print("\nBuilding features...")
    X, y = build_features(df)
    feature_names = X.columns.tolist()

    # Season window comparison
    print("\nComparing training windows...")
    test_mask = df["season"] == 20252026
    X_test = X[test_mask]
    y_test = y[test_mask]

    windows = {
        "20222023+20232024+20242025 (3 seasons)": df["season"].isin([20222023, 20232024, 20242025]),
        "20232024+20242025 (2 seasons)":          df["season"].isin([20232024, 20242025]),
        "20242025 only (1 season)":               df["season"] == 20242025,
        "20222023 only (oldest season)":          df["season"] == 20222023,
    }

    best_auc = 0
    best_model = None
    best_label = None

    for label, train_mask in windows.items():
        train_mask = train_mask & ~test_mask
        X_train = X[train_mask]
        y_train = y[train_mask]

        print(f"\n  [{label}]")
        print(f"  Train: {len(X_train)} shots ({y_train.sum()} goals)")

        m = xgb.XGBClassifier(
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
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = m.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        ll = log_loss(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        print(f"  AUC: {auc:.4f} | Log Loss: {ll:.4f} | Brier: {brier:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = m
            best_label = label

    print(f"\nBest training window: {best_label} (AUC: {best_auc:.4f})")
    model = best_model

    # Final evaluation
    print("\n--- Final Model Evaluation ---")
    y_pred_proba = evaluate_model(model, X_test, y_test)
    print_feature_importance(model, feature_names)

    # Retrain on all non-test data
    print(f"\nRetraining best model on all data excluding 20252026 test set...")
    final_train_mask = ~test_mask
    X_final = X[final_train_mask]
    y_final = y[final_train_mask]
    model.set_params(early_stopping_rounds=None)
    model.fit(X_final, y_final, verbose=False)
    print(f"  Final train set: {len(X_final)} shots ({y_final.sum()} goals)")

    # Calibrate probabilities
    print("\nCalibrating xG probabilities...")
    cal_split = int(len(X_final) * 0.8)
    X_cal = X_final.iloc[cal_split:]
    y_cal = y_final.iloc[cal_split:]
    raw_probs = model.predict_proba(X_cal)[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_cal)

    print(f"  Raw prob mean: {raw_probs.mean():.4f}  Actual goal rate: {y_cal.mean():.4f}")
    cal_probs = calibrator.predict(raw_probs)
    print(f"  Calibrated prob mean: {cal_probs.mean():.4f}")

    cal_path = os.path.join(MODEL_DIR, "xg_calibrator.pkl")
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"  Calibrator saved to {cal_path}")

    # Apply calibrated xG to training shots (SOG + goals, non-empty-net)
    print("\nGenerating calibrated xG values for full dataset...")
    raw_xg = model.predict_proba(X)[:, 1]
    df["xg"] = np.clip(calibrator.predict(raw_xg), 0.001, 0.95)

    # Apply xG to missed shots (non-empty-net)
    df_missed = df_missed[df_missed["distance"].notna() & df_missed["angle"].notna()].copy()
    df_missed["shot_type"] = df_missed["shot_type"].fillna("unknown")
    df_missed["shot_type_enc"] = df_missed["shot_type"].map(
        dict(zip(shot_type_encoder.classes_,
                 shot_type_encoder.transform(shot_type_encoder.classes_)))
    ).fillna(0)
    df_missed["prev_event_type"] = df_missed["prev_event_type"].fillna("unknown")
    df_missed["prev_event_type_enc"] = df_missed["prev_event_type"].map(
        dict(zip(prev_event_encoder.classes_,
                 prev_event_encoder.transform(prev_event_encoder.classes_)))
    ).fillna(0)
    df_missed["period_adj"] = df_missed["period"].clip(upper=4)
    df_missed["speed_from_prev"] = df_missed["speed_from_prev"].fillna(0)
    df_missed["prev_distance"] = df_missed["prev_distance"].fillna(0)
    df_missed["rebound_angle_change"] = df_missed["rebound_angle_change"].fillna(0)
    df_missed["is_rebound"] = df_missed["is_rebound"].fillna(0)
    X_missed, _ = build_features(df_missed)
    raw_missed = model.predict_proba(X_missed)[:, 1]
    df_missed["xg"] = np.clip(calibrator.predict(raw_missed), 0.001, 0.95)

    # Build and apply empty net xG model
    df_empty_net_all = pd.concat([df_empty_net, df_empty_net_missed], ignore_index=True)
    en_model, en_scaler, en_features = build_empty_net_xg_model(
        df_empty_net_all, shot_type_encoder
    )
    df_empty_net_with_xg = apply_empty_net_xg(
        df_empty_net_all, en_model, en_scaler, en_features, shot_type_encoder
    )

    # Save EN model
    with open(os.path.join(MODEL_DIR, "xg_en_model.pkl"), "wb") as f:
        pickle.dump({
            "model": en_model,
            "scaler": en_scaler,
            "features": en_features
        }, f)
    print("  Empty net model saved to xg_en_model.pkl")

    # Combine all shots
    df_all = pd.concat([df, df_missed, df_empty_net_with_xg], ignore_index=True)
    df_all = df_all.sort_values(["game_id", "period", "time_seconds"]).reset_index(drop=True)

    # Sanity check
    print("\nTop 10 highest xG shots (non-empty-net):")
    top_xg = df.nlargest(10, "xg")[["date", "shooting_team", "shot_type",
                                     "distance", "angle", "is_rebound",
                                     "is_goal", "xg"]]
    print(top_xg.to_string())

    print("\nxG summary by shot type:")
    sog = df_all[df_all["is_on_goal"] == 1]
    print(f"  Mean xG per SOG: {sog['xg'].mean():.4f}")
    print(f"  Expected goals per game per team (30 SOG): {30 * sog['xg'].mean():.2f}")

    # Save model and encoders
    print("\nSaving model...")
    model.save_model(os.path.join(MODEL_DIR, "xg_model.json"))
    with open(os.path.join(MODEL_DIR, "xg_encoders.pkl"), "wb") as f:
        pickle.dump({
            "shot_type_encoder": shot_type_encoder,
            "prev_event_encoder": prev_event_encoder,
        }, f)

    xg_output_path = os.path.expanduser("~/nhl-props/data/processed/shot_data_with_xg.csv")
    os.makedirs(os.path.dirname(xg_output_path), exist_ok=True)
    df_all.to_csv(xg_output_path, index=False)
    print(f"Shot data with xG saved to {xg_output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()