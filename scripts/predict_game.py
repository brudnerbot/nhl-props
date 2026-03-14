import pandas as pd
import numpy as np
import os
import pickle
from scipy import stats
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
DATA_PATH = os.path.expanduser("~/nhl-props/data/processed/team_features.csv")
MODEL_DIR = os.path.expanduser("~/nhl-props/models/team")

# League average SH shots per 60 (fallback if no team data)
LEAGUE_AVG_SH_SHOTS_PER60 = 11.5


def load_models():
    """Load all trained models and metadata."""
    print("Loading models...")
    models = {}
    CLASSIFIER_MODELS = {"home_won"}

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".json")]
    for f in model_files:
        name = f.replace(".json", "")
        model_path = os.path.join(MODEL_DIR, f)
        if name in CLASSIFIER_MODELS:
            m = xgb.XGBClassifier()
        else:
            m = xgb.XGBRegressor()
        m.load_model(model_path)
        models[name] = m

    with open(os.path.join(MODEL_DIR, "feature_lists.pkl"), "rb") as f:
        feature_lists = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "residual_stds.pkl"), "rb") as f:
        residual_stds = pickle.load(f)

    print(f"  Loaded {len(models)} models")
    return models, feature_lists, residual_stds


def build_matchup_row(df, home_team, away_team, game_date=None):
    """
    Build a single matchup row for prediction.
    Gets the most recent feature row for each team regardless of home/away,
    then correctly assigns home_ and away_ prefixes for the new matchup.
    """
    if game_date:
        candidates = df[df["date"] < game_date].copy()
    else:
        candidates = df.copy()

    def get_team_stats(team):
        """Get most recent stats for a team, from either home or away perspective."""
        home_rows = candidates[candidates["home_team"] == team]
        away_rows = candidates[candidates["away_team"] == team]

        if len(home_rows) == 0 and len(away_rows) == 0:
            raise ValueError(f"No data found for team: {team}")

        latest_home = home_rows.sort_values("date").iloc[-1] if len(home_rows) > 0 else None
        latest_away = away_rows.sort_values("date").iloc[-1] if len(away_rows) > 0 else None

        if latest_home is not None and latest_away is not None:
            if latest_home["date"] >= latest_away["date"]:
                source = "home"
                row = latest_home
            else:
                source = "away"
                row = latest_away
        elif latest_home is not None:
            source = "home"
            row = latest_home
        else:
            source = "away"
            row = latest_away

        return row, source

    home_row, home_source = get_team_stats(home_team)
    away_row, away_source = get_team_stats(away_team)

    home_date = home_row["date"]
    away_date = away_row["date"]
    if hasattr(home_date, "strftime"):
        home_date = home_date.strftime("%Y-%m-%d")
    if hasattr(away_date, "strftime"):
        away_date = away_date.strftime("%Y-%m-%d")

    print(f"  {home_team}: using {home_source} game from {home_date}")
    print(f"  {away_team}: using {away_source} game from {away_date}")

    def extract_team_stats(row, source, new_prefix):
        """Extract team stats and rename prefix."""
        result = {}
        for col in row.index:
            if col.startswith(f"{source}_"):
                new_col = col.replace(f"{source}_", f"{new_prefix}_", 1)
                result[new_col] = row[col]
        return result

    home_stats = extract_team_stats(home_row, home_source, "home")
    away_stats = extract_team_stats(away_row, away_source, "away")

    home_stats["home_team"] = home_team
    away_stats["away_team"] = away_team

    combined = {**home_stats, **away_stats}
    combined["date"] = game_date or df["date"].max()
    combined["season"] = home_row["season"]

    return pd.Series(combined)


def predict_stat(model, feature_list, row, model_type):
    """Run prediction for a single stat."""
    X = pd.DataFrame([row])[feature_list].fillna(0)
    if model_type == "classifier":
        return float(model.predict_proba(X)[0, 1])
    else:
        return float(model.predict(X)[0])


def calculate_total_shots(ev_shots_per60, ev_toi,
                          weighted_pp_shots_per60, pp_toi,
                          weighted_sh_shots_per60, sh_toi):
    """Calculate total shots from rate x TOI at each strength."""
    ev_shots = ev_shots_per60 * (ev_toi / 60)
    pp_shots = weighted_pp_shots_per60 * (pp_toi / 60)
    sh_shots = weighted_sh_shots_per60 * (sh_toi / 60)
    total = ev_shots + pp_shots + sh_shots
    return total, ev_shots, pp_shots, sh_shots


def poisson_distribution(lam, max_k=15):
    """Generate Poisson probability distribution."""
    lam = max(lam, 0.01)
    return [stats.poisson.pmf(k, lam) for k in range(max_k + 1)]


def normal_thresholds(mu, sigma, thresholds):
    """Calculate P(X > threshold) for Normal distribution."""
    return {f"over_{t}": float(1 - stats.norm.cdf(t, mu, sigma))
            for t in thresholds}


def format_poisson(lam, probs):
    """Format Poisson distribution for display."""
    lines = [f"  Projected: {lam:.2f}"]
    dist_str = "  Distribution: "
    for k in range(8):
        dist_str += f"P({k})={probs[k]:.1%}  "
    lines.append(dist_str)
    lines.append(
        f"  Over 0.5: {1-probs[0]:.1%}  "
        f"Over 1.5: {1-probs[0]-probs[1]:.1%}  "
        f"Over 2.5: {1-sum(probs[:3]):.1%}  "
        f"Over 3.5: {1-sum(probs[:4]):.1%}  "
        f"Over 4.5: {1-sum(probs[:5]):.1%}"
    )
    return "\n".join(lines)


def format_normal(mu, sigma, thresholds):
    """Format Normal distribution for display."""
    lines = [
        f"  Projected: {mu:.1f} (±{sigma:.1f})",
        f"  68% range: {mu-sigma:.1f} to {mu+sigma:.1f}",
        f"  90% range: {mu-1.645*sigma:.1f} to {mu+1.645*sigma:.1f}",
    ]
    thresh_str = "  "
    for t in thresholds:
        prob = float(1 - stats.norm.cdf(t, mu, sigma))
        thresh_str += f"Over {t}: {prob:.1%}  "
    lines.append(thresh_str)
    return "\n".join(lines)


def predict_game(home_team, away_team, game_date=None, verbose=True):
    """
    Generate full game predictions for a matchup.

    Args:
        home_team: 3-letter team abbreviation (e.g. 'EDM')
        away_team: 3-letter team abbreviation (e.g. 'VAN')
        game_date: optional date string 'YYYY-MM-DD' to use features before that date
        verbose: print full output

    Returns:
        dict of predictions
    """
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    if game_date:
        game_date = pd.to_datetime(game_date)

    models, feature_lists, residual_stds = load_models()

    print(f"\nBuilding features for {home_team} (home) vs {away_team} (away)...")
    row = build_matchup_row(df, home_team, away_team, game_date)

    predictions = {}

    # --- WIN PROBABILITY ---
    if "home_won" in models:
        win_prob = predict_stat(models["home_won"], feature_lists["home_won"], row, "classifier")
        predictions["home_win_prob"] = win_prob
        predictions["away_win_prob"] = 1 - win_prob

    # --- GOALS ---
    for side in ["home", "away"]:
        target = f"{side}_goals_for"
        if target in models:
            lam = predict_stat(models[target], feature_lists[target], row, "regression")
            probs = poisson_distribution(lam)
            predictions[f"{side}_goals"] = {
                "lambda": lam,
                "probs": probs,
                "over_0.5": 1 - probs[0],
                "over_1.5": 1 - probs[0] - probs[1],
                "over_2.5": 1 - sum(probs[:3]),
                "over_3.5": 1 - sum(probs[:4]),
                "over_4.5": 1 - sum(probs[:5]),
            }

    # --- EV TOI ---
    for side in ["home", "away"]:
        target = f"{side}_ev_toi"
        if target in models:
            predictions[f"{side}_ev_toi"] = predict_stat(
                models[target], feature_lists[target], row, "regression")

    # --- PP TOI ---
    for side in ["home", "away"]:
        target = f"{side}_pp_toi"
        if target in models:
            predictions[f"{side}_pp_toi"] = predict_stat(
                models[target], feature_lists[target], row, "regression")
        else:
            predictions[f"{side}_pp_toi"] = 5.0

    # Enforce SH TOI = opponent PP TOI
    predictions["home_sh_toi"] = predictions.get("away_pp_toi", 5.0)
    predictions["away_sh_toi"] = predictions.get("home_pp_toi", 5.0)

    # --- EV SHOTS PER 60 ---
    for side in ["home", "away"]:
        target = f"{side}_ev_shots_on_goal_for_per60"
        if target in models:
            predictions[f"{side}_ev_shots_per60"] = predict_stat(
                models[target], feature_lists[target], row, "regression")

    # --- TOTAL SHOTS (rate x TOI) ---
    for side in ["home", "away"]:
        ev_shots_per60 = predictions.get(f"{side}_ev_shots_per60", 27.0)
        ev_toi = predictions.get(f"{side}_ev_toi", 50.0)
        pp_toi = predictions.get(f"{side}_pp_toi", 5.0)
        sh_toi = predictions.get(f"{side}_sh_toi", 5.0)

        weighted_pp = row.get(f"{side}_weighted_pp_shots_on_goal_for_per60", 70.0)
        weighted_sh = row.get(f"{side}_weighted_sh_shots_on_goal_for_per60",
                              LEAGUE_AVG_SH_SHOTS_PER60)

        if pd.isna(weighted_pp):
            weighted_pp = 70.0
        if pd.isna(weighted_sh):
            weighted_sh = LEAGUE_AVG_SH_SHOTS_PER60

        total, ev_shots, pp_shots, sh_shots = calculate_total_shots(
            ev_shots_per60, ev_toi, weighted_pp, pp_toi, weighted_sh, sh_toi)

        sigma = residual_stds.get(f"{side}_ev_shots_on_goal_for_per60", total * 0.15)
        sigma_total = sigma * (ev_toi / 60)

        predictions[f"{side}_total_shots"] = {
            "total": total,
            "ev_shots": ev_shots,
            "pp_shots": pp_shots,
            "sh_shots": sh_shots,
            "sigma": sigma_total,
            **normal_thresholds(total, sigma_total, [19.5, 24.5, 27.5, 29.5, 32.5, 34.5])
        }

    # --- xG ---
    for side in ["home", "away"]:
        target = f"{side}_xgf_total"
        if target in models:
            xg = predict_stat(models[target], feature_lists[target], row, "regression")
            sigma = residual_stds.get(target, xg * 0.20)
            predictions[f"{side}_xg"] = {
                "total": xg,
                "sigma": sigma,
                **normal_thresholds(xg, sigma, [1.5, 2.5, 3.5, 4.5, 5.5])
            }

    # --- PRINT OUTPUT ---
    if verbose:
        print("\n" + "="*65)
        print(f"  GAME PREDICTION: {home_team} vs {away_team}")
        if game_date:
            print(f"  Date: {game_date.strftime('%Y-%m-%d')}")
        print("="*65)

        # Win probability
        home_wp = predictions.get("home_win_prob", 0.5)
        away_wp = predictions.get("away_win_prob", 0.5)
        print(f"\nWIN PROBABILITY")
        print(f"  {home_team}: {home_wp:.1%}  |  {away_team}: {away_wp:.1%}")

        # Goals
        print(f"\nGOALS")
        for side, team in [("home", home_team), ("away", away_team)]:
            g = predictions.get(f"{side}_goals", {})
            if g:
                print(f"\n  {team}:")
                print(format_poisson(g["lambda"], g["probs"]))

        # Total shots
        print(f"\nTOTAL SHOTS ON GOAL")
        for side, team in [("home", home_team), ("away", away_team)]:
            s = predictions.get(f"{side}_total_shots", {})
            if s:
                print(f"\n  {team}:")
                print(f"  Breakdown: EV {s['ev_shots']:.1f} + "
                      f"PP {s['pp_shots']:.1f} + "
                      f"SH {s['sh_shots']:.1f} = {s['total']:.1f} total")
                print(format_normal(s["total"], s["sigma"],
                                    [19.5, 24.5, 27.5, 29.5, 32.5, 34.5]))

        # xG
        print(f"\nEXPECTED GOALS (xG)")
        for side, team in [("home", home_team), ("away", away_team)]:
            xg = predictions.get(f"{side}_xg", {})
            if xg:
                print(f"\n  {team}:")
                print(format_normal(xg["total"], xg["sigma"],
                                    [1.5, 2.5, 3.5, 4.5, 5.5]))

        # Strength TOI
        print(f"\nSTRENGTH TOI (minutes, includes OT if applicable)")
        for side, team in [("home", home_team), ("away", away_team)]:
            ev = predictions.get(f"{side}_ev_toi", 0)
            pp = predictions.get(f"{side}_pp_toi", 0)
            sh = predictions.get(f"{side}_sh_toi", 0)
            total_toi = ev + pp + sh
            print(f"  {team}: EV {ev:.1f}min  PP {pp:.1f}min  SH {sh:.1f}min  "
                  f"(total {total_toi:.1f}min)")

        print("\n" + "="*65)

    return predictions


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python predict_game.py HOME_TEAM AWAY_TEAM [DATE]")
        print("Example: python predict_game.py EDM VAN 2026-03-15")
        sys.exit(1)

    home = sys.argv[1].upper()
    away = sys.argv[2].upper()
    date = sys.argv[3] if len(sys.argv) > 3 else None

    predict_game(home, away, date)