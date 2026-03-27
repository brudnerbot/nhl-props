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

LEAGUE_AVG_SH_SHOTS_PER60 = 11.5

def load_models():
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
    if game_date:
        candidates = df[df["date"] < game_date].copy()
    else:
        candidates = df.copy()

    def get_latest_row(team):
        home_rows = candidates[candidates["home_team"] == team]
        away_rows = candidates[candidates["away_team"] == team]
        if len(home_rows) == 0 and len(away_rows) == 0:
            raise ValueError(f"No data found for team: {team}")
        latest_home = home_rows.sort_values("date").iloc[-1] if len(home_rows) > 0 else None
        latest_away = away_rows.sort_values("date").iloc[-1] if len(away_rows) > 0 else None
        if latest_home is not None and latest_away is not None:
            if latest_home["date"] >= latest_away["date"]:
                return latest_home, "home"
            else:
                return latest_away, "away"
        elif latest_home is not None:
            return latest_home, "home"
        else:
            return latest_away, "away"

    home_row, home_source = get_latest_row(home_team)
    away_row, away_source = get_latest_row(away_team)
    print(f"  {home_team}: features from {home_source} game on {str(home_row['date'])[:10]}")
    print(f"  {away_team}: features from {away_source} game on {str(away_row['date'])[:10]}")

    ROLLING_SUFFIXES = [
        "_last10", "_last20", "_last30", "_season_avg",
        "_opp_adj_last10", "_opp_adj_last20", "_opp_adj_last30",
        "_cumulative_season", "_per60_last10", "_per60_last20",
        "_per60_last30", "_per60_cumulative_season",
    ]
    STABLE_FEATURES = [
        "days_rest", "is_back_to_back",
        "save_pct_last20", "save_pct_last40",
        "save_pct_current_season", "save_pct_career",
        "gsax_per60_last20", "gsax_per60_last40",
        "gsax_per60_current_season", "gsax_per60_career",
    ]
    WEIGHTED_PREFIX = "weighted_"
    PER60_SUFFIX = "_per60"
    FUNNEL_RATE_KEYS = [
        "block_rate", "sog_fenwick_rate", "fenwick_rate",
        "xg_per_sog", "save_pct_team", "shooting_pct"
    ]
    COUNT_STATS = [
        "total_minor_penalties_taken", "total_minor_penalties_drawn",
        "total_penalties_taken", "total_penalties_drawn",
        "ev_minor_penalties_taken", "ev_minor_penalties_drawn",
        "pp_minor_penalties_taken", "pp_minor_penalties_drawn",
        "sh_minor_penalties_taken", "sh_minor_penalties_drawn",
        "ev_major_penalties_taken", "ev_major_penalties_drawn",
        "pp_major_penalties_taken", "pp_major_penalties_drawn",
        "sh_major_penalties_taken", "sh_major_penalties_drawn",
    ]

    def extract_rolling_features(row, source_prefix, new_prefix):
        result = {}
        for col in row.index:
            if not col.startswith(f"{source_prefix}_"):
                continue
            col_suffix = col[len(f"{source_prefix}_"):]

            if any(col_suffix.endswith(s) for s in ROLLING_SUFFIXES):
                result[f"{new_prefix}_{col_suffix}"] = row[col]
                continue

            if any(col_suffix == f for f in STABLE_FEATURES):
                result[f"{new_prefix}_{col_suffix}"] = row[col]
                continue

            if col_suffix.startswith(WEIGHTED_PREFIX):
                result[f"{new_prefix}_{col_suffix}"] = row[col]
                continue

            if PER60_SUFFIX in col_suffix:
                if any(s in col_suffix for s in ["_last10", "_last20", "_last30",
                                                   "_season_avg", "_cumulative",
                                                   "weighted_"]):
                    result[f"{new_prefix}_{col_suffix}"] = row[col]
                else:
                    for suffix in ["_per60_cumulative_season", "_per60_last20",
                                   "_per60_last30", "_season_avg"]:
                        sub_col = f"{source_prefix}_{col_suffix.replace('_per60','')}{suffix}"
                        if sub_col in row.index and not pd.isna(row[sub_col]) and row[sub_col] != 0:
                            result[f"{new_prefix}_{col_suffix}"] = row[sub_col]
                            break
                continue

            if any(key in col_suffix for key in FUNNEL_RATE_KEYS):
                result[f"{new_prefix}_{col_suffix}"] = row[col]
                continue

            if any(col_suffix == s for s in COUNT_STATS):
                season_col = f"{source_prefix}_{col_suffix}_season_avg"
                last20_col = f"{source_prefix}_{col_suffix}_last20"
                if season_col in row.index and not pd.isna(row[season_col]):
                    result[f"{new_prefix}_{col_suffix}"] = row[season_col]
                elif last20_col in row.index and not pd.isna(row[last20_col]):
                    result[f"{new_prefix}_{col_suffix}"] = row[last20_col]
                else:
                    result[f"{new_prefix}_{col_suffix}"] = row[col]
                continue

        return result

    home_stats = extract_rolling_features(home_row, home_source, "home")
    away_stats = extract_rolling_features(away_row, away_source, "away")
    home_stats["home_team"] = home_team
    away_stats["away_team"] = away_team
    combined = {**home_stats, **away_stats}
    combined["date"] = game_date or df["date"].max()
    combined["season"] = home_row["season"]
    return pd.Series(combined)


def predict_stat(model, feature_list, row, model_type):
    X = pd.DataFrame([row])[feature_list].fillna(0)
    if model_type == "classifier":
        return float(model.predict_proba(X)[0, 1])
    else:
        return float(model.predict(X)[0])


def poisson_distribution(lam, max_k=15):
    lam = max(lam, 0.01)
    return [stats.poisson.pmf(k, lam) for k in range(max_k + 1)]


def format_normal(mu, sigma, thresholds):
    lines = [
        f"  Projected: {mu:.1f} (±{sigma:.1f})",
        f"  68% range: {mu-sigma:.1f} to {mu+sigma:.1f}",
    ]
    thresh_str = "  "
    for t in thresholds:
        prob = float(1 - stats.norm.cdf(t, mu, sigma))
        thresh_str += f"Over {t}: {prob:.1%}  "
    lines.append(thresh_str)
    return "\n".join(lines)


def format_shots(mu, sigma, book_lines=None):
    """Format shots output with over/under probs centered on projection."""
    if book_lines is None:
        book_lines = [24.5, 29.5, 34.5]
    lines = [
        f"  Projected: {mu:.1f} (±{sigma:.1f})",
        f"  68% range: {mu-sigma:.1f} to {mu+sigma:.1f}",
    ]
    fixed_str = "  Book lines:  "
    for t in book_lines:
        over  = float(1 - stats.norm.cdf(t, mu, sigma))
        under = 1 - over
        fixed_str += f"o{t}: {over:.1%} / u{t}: {under:.1%}    "
    lines.append(fixed_str)

    center  = round(mu * 2) / 2
    dynamic = [center + 0.5*i for i in range(-5, 6) if (center + 0.5*i) % 1 == 0.5]
    lines.append(f"  {'Line':<8} {'Over':>7} {'Under':>7}")
    lines.append(f"  {'-'*24}")
    for t in dynamic:
        over  = float(1 - stats.norm.cdf(t, mu, sigma))
        under = 1 - over
        marker = " ◀" if abs(t - mu) < 0.5 else ""
        lines.append(f"  {t:<8.1f} {over:>7.1%} {under:>7.1%}{marker}")
    return "\n".join(lines)


def format_poisson(lam, probs):
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


def predict_game(home_team, away_team, game_date=None, verbose=True):
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    if game_date:
        game_date = pd.to_datetime(game_date)

    models, feature_lists, residual_stds = load_models()

    AVG_GAME_TOI = float(
        (df["home_ev_toi"] + df["home_pp_toi"] + df["home_sh_toi"]).mean()
    )
    AVG_EN_TOI = float(df["home_en_toi"].mean())
    LEAGUE_AVG_PP_TOI   = float(df["home_pp_toi"].mean())
    LEAGUE_AVG_PP_SHOTS = float(df["home_pp_shots_on_goal_for"].mean())
    LEAGUE_AVG_PP_PER60 = LEAGUE_AVG_PP_SHOTS / (LEAGUE_AVG_PP_TOI / 60)
    LEAGUE_AVG_MINOR_DRAWN = float(
        df["home_total_minor_penalties_drawn_season_avg"].dropna().mean()
    )
    LEAGUE_AVG_MINOR_TAKEN = float(
        df["away_total_minor_penalties_taken_season_avg"].dropna().mean()
    )

    print(f"\nBuilding features for {home_team} (home) vs {away_team} (away)...")
    row = build_matchup_row(df, home_team, away_team, game_date)

    predictions = {}

    # --- Step 1: Win probability ---
    if "home_won" in models:
        win_prob = predict_stat(models["home_won"], feature_lists["home_won"], row, "classifier")
        predictions["home_win_prob"] = win_prob
        predictions["away_win_prob"] = 1 - win_prob

    # --- Step 2: Goals ---
    for side in ["home", "away"]:
        target = f"{side}_goals_for"
        if target in models:
            lam = predict_stat(models[target], feature_lists[target], row, "regression")
            probs = poisson_distribution(lam)
            predictions[f"{side}_goals"] = {
                "lambda": lam, "probs": probs,
                "over_0.5": 1 - probs[0],
                "over_1.5": 1 - probs[0] - probs[1],
                "over_2.5": 1 - sum(probs[:3]),
                "over_3.5": 1 - sum(probs[:4]),
                "over_4.5": 1 - sum(probs[:5]),
            }

    # --- Step 3: PP TOI ---
    def calc_pp_toi(own_drawn, opp_taken, own_pp_toi_season):
        own_drawn = float(own_drawn) if own_drawn is not None and not pd.isna(own_drawn) else LEAGUE_AVG_MINOR_DRAWN
        opp_taken = float(opp_taken) if opp_taken is not None and not pd.isna(opp_taken) else LEAGUE_AVG_MINOR_TAKEN
        own_pp_toi_season = float(own_pp_toi_season) if own_pp_toi_season is not None and not pd.isna(own_pp_toi_season) else LEAGUE_AVG_PP_TOI
        return 0.87 + (own_drawn * 0.219) + (opp_taken * 0.606) + (own_pp_toi_season * 0.251)

    predictions["home_pp_toi"] = calc_pp_toi(
        row.get("home_total_minor_penalties_drawn_season_avg"),
        row.get("away_total_minor_penalties_taken_season_avg"),
        row.get("home_pp_toi_season_avg"),
    )
    predictions["away_pp_toi"] = calc_pp_toi(
        row.get("away_total_minor_penalties_drawn_season_avg"),
        row.get("home_total_minor_penalties_taken_season_avg"),
        row.get("away_pp_toi_season_avg"),
    )

    # --- Step 4: SH TOI = opponent PP TOI ---
    predictions["home_sh_toi"] = predictions["away_pp_toi"]
    predictions["away_sh_toi"] = predictions["home_pp_toi"]

    # --- Step 5: EV TOI ---
    for side in ["home", "away"]:
        pp = predictions[f"{side}_pp_toi"]
        sh = predictions[f"{side}_sh_toi"]
        predictions[f"{side}_ev_toi"] = max(AVG_GAME_TOI - pp - sh, 30.0)

    # --- Step 6: Individual team total shots ---
    LEAGUE_AVG_SHOTS = float(df["home_total_shots_on_goal_for"].mean())

    for side, opp_side in [("home", "away"), ("away", "home")]:
        own_l30 = row.get(f"{side}_total_shots_on_goal_for_last30")
        opp_against_l30 = row.get(f"{opp_side}_total_shots_on_goal_against_last30")

        if own_l30 is None or pd.isna(own_l30):
            own_l30 = row.get(f"{side}_total_shots_on_goal_for_season_avg", LEAGUE_AVG_SHOTS)
        if opp_against_l30 is None or pd.isna(opp_against_l30):
            opp_against_l30 = row.get(f"{opp_side}_total_shots_on_goal_against_season_avg", LEAGUE_AVG_SHOTS)

        total = 0.55 * float(own_l30) + 0.45 * float(opp_against_l30)

        # PP shots: cumulative per60 × predicted PP TOI
        pp_per60_cum = row.get(f"{side}_pp_shots_on_goal_for_per60_cumulative_season")
        if pp_per60_cum is None or pd.isna(pp_per60_cum):
            pp_per60_cum = LEAGUE_AVG_PP_PER60
        pp_toi = predictions[f"{side}_pp_toi"]
        pp_shots = float(pp_per60_cum) * (pp_toi / 60)

        # SH shots
        sh_toi = predictions[f"{side}_sh_toi"]
        weighted_sh = row.get(f"{side}_weighted_sh_shots_on_goal_for_per60")
        if weighted_sh is None or pd.isna(weighted_sh):
            weighted_sh = LEAGUE_AVG_SH_SHOTS_PER60
        sh_shots = float(weighted_sh) * (sh_toi / 60)

        ev_shots = max(total - pp_shots - sh_shots, 0)
        ev_toi = predictions[f"{side}_ev_toi"]
        sigma = residual_stds.get(f"{side}_ev_shots_on_goal_for_per60", 5.0)
        sigma_total = sigma * (ev_toi / 60)

        predictions[f"{side}_total_shots"] = {
            "total": total, "ev_shots": ev_shots,
            "pp_shots": pp_shots, "sh_shots": sh_shots,
            "sigma": sigma_total,
        }

    # --- Step 7: xG ---
    for side in ["home", "away"]:
        target = f"{side}_xgf_total"
        if target in models:
            xg = predict_stat(models[target], feature_lists[target], row, "regression")
            sigma = residual_stds.get(target, xg * 0.20)
            predictions[f"{side}_xg"] = {
                "total": xg, "sigma": sigma,
                **{f"over_{t}": float(1 - stats.norm.cdf(t, xg, sigma))
                   for t in [1.5, 2.5, 3.5, 4.5, 5.5]}
            }

   # --- Step 8: Game total shots (naive sum of individual predictions) ---
    # XGBoost game total model tested but pred std collapses to ~1.1 (unusable).
    # Naive sum of empirical formula matches near-optimal blend.
    # Residual std 8.34 from test set (2025-26, 1,136 games).
    GAME_TOTAL_SIGMA = 8.34
    home_total = predictions.get("home_total_shots", {}).get("total", 0)
    away_total = predictions.get("away_total_shots", {}).get("total", 0)
    game_total = home_total + away_total
    predictions["game_total_shots"] = {
        "total": game_total,
        "sigma": GAME_TOTAL_SIGMA,
    }

    # --- Output ---
    if verbose:
        print("\n" + "="*65)
        print(f"  GAME PREDICTION: {home_team} vs {away_team}")
        if game_date:
            print(f"  Date: {game_date.strftime('%Y-%m-%d')}")
        print("="*65)

        home_wp = predictions.get("home_win_prob", 0.5)
        away_wp = predictions.get("away_win_prob", 0.5)
        print(f"\nWIN PROBABILITY")
        print(f"  {home_team}: {home_wp:.1%}  |  {away_team}: {away_wp:.1%}")

        print(f"\nGOALS")
        for side, team in [("home", home_team), ("away", away_team)]:
            g = predictions.get(f"{side}_goals", {})
            if g:
                print(f"\n  {team}:")
                print(format_poisson(g["lambda"], g["probs"]))

        print(f"\nINDIVIDUAL SHOTS ON GOAL")
        for side, team in [("home", home_team), ("away", away_team)]:
            s = predictions.get(f"{side}_total_shots", {})
            if s:
                print(f"\n  {team}:")
                print(f"  Breakdown: EV {s['ev_shots']:.1f} + "
                      f"PP {s['pp_shots']:.1f} + "
                      f"SH {s['sh_shots']:.1f} = {s['total']:.1f} total")
                print(format_shots(s["total"], s["sigma"]))

        print(f"\nGAME TOTAL SHOTS")
        gt = predictions.get("game_total_shots", {})
        if gt:
            print(format_shots(gt["total"], gt["sigma"],
                               book_lines=[49.5, 54.5, 57.5, 59.5, 64.5]))

        print(f"\nEXPECTED GOALS (xG)")
        for side, team in [("home", home_team), ("away", away_team)]:
            xg = predictions.get(f"{side}_xg", {})
            if xg:
                print(f"\n  {team}:")
                print(format_normal(xg["total"], xg["sigma"],
                                    [1.5, 2.5, 3.5, 4.5, 5.5]))

        print(f"\nSTRENGTH TOI (minutes)")
        for side, team in [("home", home_team), ("away", away_team)]:
            ev = predictions[f"{side}_ev_toi"]
            pp = predictions[f"{side}_pp_toi"]
            sh = predictions[f"{side}_sh_toi"]
            print(f"  {team}: EV {ev:.1f}min  PP {pp:.1f}min  SH {sh:.1f}min  "
                  f"EN ~{AVG_EN_TOI:.1f}min  (total {ev+pp+sh+AVG_EN_TOI:.1f}min)")

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