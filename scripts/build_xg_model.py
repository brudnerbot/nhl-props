import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# --- CONFIG ---
DATA_PATH   = os.path.expanduser("~/nhl-props/data/raw/shot_data/shot_data.csv")
PLAYER_PATH = os.path.expanduser("~/nhl-props/data/raw/game_logs/player_game_logs.csv")
MODEL_DIR   = os.path.expanduser("~/nhl-props/models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Rush definition: fast shot from distance (neutral zone entry or odd-man rush)
RUSH_SPEED_THRESHOLD    = 20.0   # ft/s
RUSH_PREV_DIST_THRESHOLD = 75.0  # ft from previous event


def build_position_map(player_path):
    """Build shooter_id -> is_forward mapping from player game logs."""
    print("Building position map from player game logs...")
    pl = pd.read_csv(player_path, low_memory=False)
    pl = pl[["player_id", "position"]].drop_duplicates("player_id")
    # F = forwards (C, L, R, W, LW, RW), D = defenseman
    pl["is_forward"] = pl["position"].str.upper().apply(
        lambda p: 1 if any(f in str(p) for f in ["C","L","R","W","F"]) else 0
    )
    pos_map = dict(zip(pl["player_id"], pl["is_forward"]))
    forwards = sum(v == 1 for v in pos_map.values())
    print(f"  {len(pos_map):,} players mapped: {forwards} forwards, "
          f"{len(pos_map)-forwards} defensemen")
    return pos_map


def load_and_prepare(path, pos_map):
    print("\nLoading shot data...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Total shots: {len(df):,}")

    # Map shooter position
    df["is_forward"] = df["shooter_id"].map(pos_map).fillna(1)  # default to forward if unknown
    print(f"  Position mapped: {df['is_forward'].notna().sum():,} "
          f"(unknown defaulted to forward: {df['shooter_id'].map(pos_map).isna().sum():,})")

    # Rush flag: high speed from distant previous event
    df["is_rush"] = (
        (df["speed_from_prev"].fillna(0) > RUSH_SPEED_THRESHOLD) &
        (df["prev_distance"].fillna(0) > RUSH_PREV_DIST_THRESHOLD)
    ).astype(int)
    print(f"  Rush shots: {df['is_rush'].sum():,} ({df['is_rush'].mean():.1%})")

    # Train only on shots on goal and goals
    df_train = df[df["event_type"].isin(["shot-on-goal", "goal"])].copy()
    df_missed = df[df["event_type"] == "missed-shot"].copy()
    print(f"  SOG + goals: {len(df_train):,} | Missed: {len(df_missed):,}")

    # Separate empty net
    df_empty_net        = df_train[df_train["is_empty_net"] == 1].copy()
    df_empty_net_missed = df_missed[df_missed["is_empty_net"] == 1].copy()
    df_train = df_train[df_train["is_empty_net"] == 0].copy()
    df_missed = df_missed[df_missed["is_empty_net"] == 0].copy()
    print(f"  After removing EN — SOG+goals: {len(df_train):,} | Missed: {len(df_missed):,}")

    # Drop rows missing key features
    df_train = df_train.dropna(subset=["distance", "angle"]).copy()

    # Fill missing features
    for col in ["speed_from_prev", "prev_distance", "rebound_angle_change", "is_rebound"]:
        df_train[col] = df_train[col].fillna(0)

    # Encode shot type
    df_train["shot_type"] = df_train["shot_type"].fillna("unknown")
    shot_type_encoder = LabelEncoder()
    df_train["shot_type_enc"] = shot_type_encoder.fit_transform(df_train["shot_type"])

    # Encode prev event type
    df_train["prev_event_type"] = df_train["prev_event_type"].fillna("unknown")
    prev_event_encoder = LabelEncoder()
    df_train["prev_event_type_enc"] = prev_event_encoder.fit_transform(df_train["prev_event_type"])

    # Period — cap at 4
    df_train["period_adj"] = df_train["period"].clip(upper=4)

    print(f"\n  Goal rate: {df_train['is_goal'].mean()*100:.2f}%")
    print(f"  Rush shot goal rate: {df_train[df_train['is_rush']==1]['is_goal'].mean()*100:.2f}%")
    print(f"  Non-rush goal rate:  {df_train[df_train['is_rush']==0]['is_goal'].mean()*100:.2f}%")
    print(f"  Forward goal rate:   {df_train[df_train['is_forward']==1]['is_goal'].mean()*100:.2f}%")
    print(f"  Defense goal rate:   {df_train[df_train['is_forward']==0]['is_goal'].mean()*100:.2f}%")

    return (df_train, df_missed, df_empty_net, df_empty_net_missed,
            shot_type_encoder, prev_event_encoder)


def build_features(df):
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
        "is_forward",   # NEW: forward vs defenseman
        "is_rush",      # NEW: rush shot flag
    ]
    available = [f for f in features if f in df.columns]
    return df[available], df["is_goal"]


def platt_calibrate(model, X_cal, y_cal):
    """Platt scaling — logistic regression on top of XGBoost raw log-odds.
    Better than isotonic regression for tail calibration."""
    raw_probs = model.predict_proba(X_cal)[:, 1]
    # Use log-odds as input to logistic regression
    eps = 1e-6
    log_odds = np.log(raw_probs / (1 - raw_probs + eps) + eps)
    lr = LogisticRegression(random_state=42)
    lr.fit(log_odds.reshape(-1, 1), y_cal)
    print(f"  Platt scaling coefficients: a={lr.coef_[0][0]:.4f}, b={lr.intercept_[0]:.4f}")
    return lr


def apply_platt(calibrator, raw_probs):
    eps = 1e-6
    log_odds = np.log(raw_probs / (1 - raw_probs + eps) + eps)
    return calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]


def evaluate_calibration(y_true, y_pred, label=""):
    """Print calibration by bucket."""
    df_eval = pd.DataFrame({"y": y_true, "p": y_pred})
    df_eval["bucket"] = pd.cut(df_eval["p"],
        bins=[0,.05,.10,.15,.20,.30,.40,.60,1.0])
    cal = df_eval.groupby("bucket").agg(
        shots=("y","count"),
        actual=("y","mean"),
        predicted=("p","mean"),
    ).dropna()
    cal["error"] = cal["predicted"] - cal["actual"]
    print(f"\nCalibration {label}:")
    print(cal.to_string())
    return cal


def build_empty_net_model(df_empty_net_all, shot_type_encoder, pos_map):
    print("\nBuilding empty net xG model...")
    en = df_empty_net_all[df_empty_net_all["event_type"].isin(
        ["shot-on-goal", "goal", "missed-shot"])].copy()

    en["is_forward"] = en["shooter_id"].map(pos_map).fillna(1)
    en["shot_type"] = en["shot_type"].fillna("unknown")
    known = set(shot_type_encoder.classes_)
    en["shot_type_enc"] = en["shot_type"].apply(
        lambda x: shot_type_encoder.transform([x])[0] if x in known else 0)
    for col in ["speed_from_prev","is_rebound","rebound_angle_change"]:
        en[col] = en[col].fillna(0)
    en["period_adj"] = en["period"].clip(upper=4)
    en = en.dropna(subset=["distance","angle"]).copy()

    features = ["distance","angle","shot_type_enc","is_rebound",
                "speed_from_prev","rebound_angle_change","period_adj","is_forward"]
    X = en[features].fillna(0)
    y = en["is_goal"]

    test_mask = en["season"] == 20252026
    X_tr, X_te = X[~test_mask], X[test_mask]
    y_tr, y_te = y[~test_mask], y[test_mask]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr_s, y_tr)

    y_pred = model.predict_proba(X_te_s)[:, 1]
    auc = roc_auc_score(y_te, y_pred)
    ll  = log_loss(y_te, y_pred)
    baseline_ll = log_loss(y_te, [y_te.mean()]*len(y_te))
    print(f"  EN model — AUC: {auc:.4f} | LogLoss: {ll:.4f} (baseline: {baseline_ll:.4f})")
    print(f"  EN goal rate: {y_te.mean():.3f} | Mean pred: {y_pred.mean():.3f}")

    return model, scaler, features


def apply_empty_net_xg(df_en_all, en_model, en_scaler, en_features,
                        shot_type_encoder, pos_map):
    df = df_en_all.copy()
    df["is_forward"] = df["shooter_id"].map(pos_map).fillna(1)
    df["shot_type"] = df["shot_type"].fillna("unknown")
    known = set(shot_type_encoder.classes_)
    df["shot_type_enc"] = df["shot_type"].apply(
        lambda x: shot_type_encoder.transform([x])[0] if x in known else 0)
    for col in ["speed_from_prev","is_rebound","rebound_angle_change"]:
        df[col] = df[col].fillna(0)
    df["period_adj"] = df["period"].clip(upper=4)
    df["distance"] = df["distance"].fillna(30)
    df["angle"]    = df["angle"].fillna(15)
    X = df[en_features].fillna(0)
    X_s = en_scaler.transform(X)
    df["xg"] = en_model.predict_proba(X_s)[:, 1]
    print(f"  EN xG mean: {df['xg'].mean():.3f} | Actual goal rate: {df['is_goal'].mean():.3f}")
    return df


def main():
    pos_map = build_position_map(PLAYER_PATH)

    df, df_missed, df_empty_net, df_empty_net_missed, \
        shot_type_encoder, prev_event_encoder = load_and_prepare(DATA_PATH, pos_map)

    print("\nBuilding features...")
    X, y = build_features(df)
    feature_names = X.columns.tolist()
    print(f"  Features: {feature_names}")

    # Train/test split
    test_mask = df["season"] == 20252026
    X_test, y_test = X[test_mask], y[test_mask]

    # Compare training windows
    print("\nComparing training windows...")
    windows = {
        "3 seasons (20222023-20242025)": df["season"].isin([20222023,20232024,20242025]),
        "2 seasons (20232024-20242025)": df["season"].isin([20232024,20242025]),
        "1 season  (20242025)":          df["season"] == 20242025,
    }

    best_auc, best_model, best_label = 0, None, None
    for label, train_mask in windows.items():
        train_mask = train_mask & ~test_mask
        X_tr, y_tr = X[train_mask], y[train_mask]
        m = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_tr==0).sum()/(y_tr==1).sum(),
            eval_metric="logloss", early_stopping_rounds=20,
            random_state=42, n_jobs=-1,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = m.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        ll  = log_loss(y_test, y_pred)
        print(f"  [{label}] AUC: {auc:.4f} | LogLoss: {ll:.4f}")
        if auc > best_auc:
            best_auc, best_model, best_label = auc, m, label

    print(f"\nBest window: {best_label} (AUC: {best_auc:.4f})")
    model = best_model

    # Feature importances
    print("\nFeature importances:")
    fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    for f, v in fi.items():
        print(f"  {f:<30} {v:.4f}  {'█'*int(v*100)}")

    # Retrain on all non-test data
    print(f"\nRetraining on all non-test data...")
    final_mask = ~test_mask
    X_final, y_final = X[final_mask], y[final_mask]
    model.set_params(early_stopping_rounds=None)
    model.fit(X_final, y_final, verbose=False)

    # Platt calibration (better tail calibration than isotonic)
    print("\nApplying Platt scaling calibration...")
    cal_split = int(len(X_final) * 0.8)
    X_cal, y_cal = X_final.iloc[cal_split:], y_final.iloc[cal_split:]
    calibrator = platt_calibrate(model, X_cal, y_cal)

    # Evaluate calibration before and after
    raw_test  = model.predict_proba(X_test)[:, 1]
    cal_test  = apply_platt(calibrator, raw_test)
    print(f"\nTest set metrics:")
    print(f"  AUC (raw):        {roc_auc_score(y_test, raw_test):.4f}")
    print(f"  AUC (calibrated): {roc_auc_score(y_test, cal_test):.4f}")
    print(f"  LogLoss (raw):    {log_loss(y_test, raw_test):.4f}")
    print(f"  LogLoss (cal):    {log_loss(y_test, cal_test):.4f}")
    print(f"  Mean xG (raw):    {raw_test.mean():.4f}")
    print(f"  Mean xG (cal):    {cal_test.mean():.4f}")
    print(f"  Actual goal rate: {y_test.mean():.4f}")

    evaluate_calibration(y_test.values, raw_test, label="(raw)")
    evaluate_calibration(y_test.values, cal_test, label="(Platt calibrated)")

    # Apply calibrated xG to all shots
    print("\nApplying xG to full dataset...")
    raw_all = model.predict_proba(X)[:, 1]
    df["xg"] = np.clip(apply_platt(calibrator, raw_all), 0.001, 0.95)

    # Apply to missed shots
    df_missed = df_missed[df_missed["distance"].notna() & df_missed["angle"].notna()].copy()
    df_missed["is_forward"] = df_missed["shooter_id"].map(pos_map).fillna(1)
    df_missed["is_rush"] = (
        (df_missed["speed_from_prev"].fillna(0) > RUSH_SPEED_THRESHOLD) &
        (df_missed["prev_distance"].fillna(0)   > RUSH_PREV_DIST_THRESHOLD)
    ).astype(int)
    df_missed["shot_type"] = df_missed["shot_type"].fillna("unknown")
    df_missed["shot_type_enc"] = df_missed["shot_type"].map(
        dict(zip(shot_type_encoder.classes_,
                 shot_type_encoder.transform(shot_type_encoder.classes_)))).fillna(0)
    df_missed["prev_event_type"] = df_missed["prev_event_type"].fillna("unknown")
    df_missed["prev_event_type_enc"] = df_missed["prev_event_type"].map(
        dict(zip(prev_event_encoder.classes_,
                 prev_event_encoder.transform(prev_event_encoder.classes_)))).fillna(0)
    df_missed["period_adj"] = df_missed["period"].clip(upper=4)
    for col in ["speed_from_prev","prev_distance","rebound_angle_change","is_rebound"]:
        df_missed[col] = df_missed[col].fillna(0)
    X_missed, _ = build_features(df_missed)
    raw_missed = model.predict_proba(X_missed)[:, 1]
    df_missed["xg"] = np.clip(apply_platt(calibrator, raw_missed), 0.001, 0.95)

    # Empty net model
    df_en_all = pd.concat([df_empty_net, df_empty_net_missed], ignore_index=True)
    en_model, en_scaler, en_features = build_empty_net_model(
        df_en_all, shot_type_encoder, pos_map)
    df_en_xg = apply_empty_net_xg(
        df_en_all, en_model, en_scaler, en_features, shot_type_encoder, pos_map)

    # League calibration check
    test_sog = df[test_mask & (df["is_on_goal"]==1)]
    print(f"\nLeague calibration check (test season SOG):")
    print(f"  Total xG:    {test_sog['xg'].sum():.1f}")
    print(f"  Total goals: {test_sog['is_goal'].sum():.1f}")
    print(f"  xG/goal:     {test_sog['xg'].sum()/test_sog['is_goal'].sum():.3f}")

    # Combine all shots
    df_all = pd.concat([df, df_missed, df_en_xg], ignore_index=True)
    df_all = df_all.sort_values(["game_id","period","time_seconds"]).reset_index(drop=True)

    print(f"\nFinal dataset: {len(df_all):,} shots")

    # Save everything
    print("\nSaving models...")
    model.save_model(os.path.join(MODEL_DIR, "xg_model.json"))

    with open(os.path.join(MODEL_DIR, "xg_calibrator.pkl"), "wb") as f:
        pickle.dump(calibrator, f)

    with open(os.path.join(MODEL_DIR, "xg_en_model.pkl"), "wb") as f:
        pickle.dump({"model": en_model, "scaler": en_scaler, "features": en_features}, f)

    with open(os.path.join(MODEL_DIR, "xg_encoders.pkl"), "wb") as f:
        pickle.dump({
            "shot_type_encoder":   shot_type_encoder,
            "prev_event_encoder":  prev_event_encoder,
            "pos_map":             pos_map,
        }, f)

    out_path = os.path.expanduser("~/nhl-props/data/processed/shot_data_with_xg.csv")
    df_all.to_csv(out_path, index=False)
    print(f"  Saved shot_data_with_xg.csv ({len(df_all):,} rows)")
    print("\nDone!")


if __name__ == "__main__":
    main()