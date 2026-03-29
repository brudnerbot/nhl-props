"""
predict_player_props.py

Generates player prop predictions for a given game using:
  - Today's lineup from data/raw/lineups/lineups_YYYYMMDD.json
  - Player features from data/processed/player_features.csv
  - Trained models from models/player/
  - Team PP baselines computed on-the-fly from recent PBP data

PP TOI uses formula: team_pp_toi × player_pp_share
  PP share priority:
  1. Lineup PP unit assignment (primary) → team PP1/PP2 baseline
  2. Recent games (last 10 with PP time) → nudge from baseline
  3. Season history → small additional nudge

Usage:
    python scripts/predict_player_props.py TOR VAN
    python scripts/predict_player_props.py TOR VAN --date 2026-03-28
"""

import argparse
import json
import pickle
import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models/player"

PLAYER_FEATURES = DATA_DIR / "processed/player_features.csv"
LINEUP_DIR      = DATA_DIR / "raw/lineups"
PP_SHARES_FILE  = DATA_DIR / "raw/player_pp_shares.csv"
PBP_FILE        = DATA_DIR / "raw/player_pbp_stats/player_pbp_stats.csv"
TEAM_LOGS_FILE  = DATA_DIR / "raw/team_game_logs/team_game_logs.csv"

SHOT_LINES  = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
POINT_LINES = [0.5, 1.5, 2.5]

# League average fallback PP share by unit
PP_SHARE_FALLBACK = {
    1: 0.670,
    2: 0.330,
    0: 0.030,
}

NICKNAMES = {
    "matt": "matthew", "mike": "michael", "alex": "alexander",
    "nick": "nicholas", "mitch": "mitchell", "cal": "callan",
    "zach": "zachary", "jake": "jacob", "pat": "patrick",
    "phil": "philip", "will": "william", "bill": "william",
    "rick": "richard", "dan": "daniel", "jon": "jonathan",
    "tom": "thomas", "rob": "robert", "cam": "cameron",
    "nate": "nathan", "drew": "andrew", "steve": "steven",
    "chris": "christopher", "tony": "anthony", "brad": "bradley",
    "sam": "samuel", "mat": "matthew",
}
NICKNAMES_REVERSE = {v: k for k, v in NICKNAMES.items()}


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models():
    with open(MODEL_DIR / "feature_lists.pkl", "rb") as f:
        feature_lists = pickle.load(f)
    with open(MODEL_DIR / "model_types.pkl", "rb") as f:
        model_types = pickle.load(f)
    with open(MODEL_DIR / "residual_stds.pkl", "rb") as f:
        residual_stds = pickle.load(f)
    calibrators = {}
    if (MODEL_DIR / "calibrators.pkl").exists():
        with open(MODEL_DIR / "calibrators.pkl", "rb") as f:
            calibrators = pickle.load(f)

    models = {}
    for name, mtype in model_types.items():
        path = MODEL_DIR / f"{name}.json"
        if not path.exists():
            continue
        m = xgb.XGBClassifier() if mtype == "classifier" else xgb.XGBRegressor()
        m.load_model(str(path))
        models[name] = m

    print(f"  Loaded {len(models)} models: {list(models.keys())}")
    return models, feature_lists, model_types, calibrators, residual_stds


# ── PP data loading and computation ──────────────────────────────────────────
def load_pp_shares():
    """Season-level PP share per player per team per season."""
    pp = pd.read_csv(PP_SHARES_FILE, low_memory=False)
    lookup = {}
    for _, row in pp.iterrows():
        key = (int(row["player_id"]), str(row["team"]), int(row["season"]))
        lookup[key] = {
            "avg_share":     float(row["avg_share"]),
            "games_with_pp": float(row["games_with_pp"]),
            "pp_unit_est":   int(row["pp_unit_est"]),
        }
    return lookup


def compute_team_pp_baselines(pbp_path, team_logs_path, last_n_games=20):
    """
    Compute each team's PP1/PP2 usage rate from recent games.
    PP1 usage = avg share of ranks 2-4 players (excludes QB skew).
    PP2 usage = 1 - PP1 usage.
    """
    print(f"Computing team PP baselines (last {last_n_games} games)...")
    pbp = pd.read_csv(pbp_path, low_memory=False)
    tgl = pd.read_csv(team_logs_path, low_memory=False)
    tgl["date"] = pd.to_datetime(tgl["date"])

    team_pp  = tgl[["game_id","team","pp_toi","date"]].rename(
        columns={"pp_toi":"team_pp_toi"})
    df = pbp.merge(team_pp, on=["game_id","team"], how="left")
    df = df[df["team_pp_toi"] > 1.0]

    game_dates = tgl[["game_id","date"]].drop_duplicates().set_index("game_id")["date"]
    df["date"] = df["game_id"].map(game_dates)

    baselines = {}
    for team, team_grp in df.groupby("team"):
        recent_game_ids = (
            team_grp[["game_id","date"]]
            .drop_duplicates()
            .sort_values("date")
            .tail(last_n_games)["game_id"]
        )
        team_recent = team_grp[team_grp["game_id"].isin(recent_game_ids)]

        game_pp1_usages = []
        for game_id, game_grp in team_recent[team_recent["toi_pp"] > 0].groupby("game_id"):
            game_grp    = game_grp.sort_values("toi_pp", ascending=False).reset_index(drop=True)
            team_pp_tot = game_grp["team_pp_toi"].iloc[0]
            if team_pp_tot < 1 or len(game_grp) < 4:
                continue
            game_grp["share"] = game_grp["toi_pp"] / (team_pp_tot + 1e-6)
            pp1_usage = game_grp.iloc[1:4]["share"].mean()
            if pd.notna(pp1_usage):
                game_pp1_usages.append(pp1_usage)

        if game_pp1_usages:
            pp1 = float(np.mean(game_pp1_usages))
            baselines[team] = {
                "pp1_usage": pp1,
                "pp2_usage": 1.0 - pp1,
                "games":     len(game_pp1_usages),
            }

    print(f"  Baselines computed for {len(baselines)} teams")
    sample = sorted(baselines.items(), key=lambda x: -x[1]["pp1_usage"])[:5]
    for team, bl in sample:
        print(f"    {team}: PP1={bl['pp1_usage']:.3f}  PP2={bl['pp2_usage']:.3f}  "
              f"games={bl['games']}")
    return baselines


def compute_recent_player_pp_shares(pbp_path, team_logs_path, last_n_games=10):
    """
    Compute each player's PP TOI share from their last N games with PP time.
    Captures recent role changes faster than season averages.
    """
    print(f"Computing recent player PP shares (last {last_n_games} PP games)...")
    pbp = pd.read_csv(pbp_path, low_memory=False)
    tgl = pd.read_csv(team_logs_path, low_memory=False)
    tgl["date"] = pd.to_datetime(tgl["date"])

    team_pp = tgl[["game_id","team","pp_toi"]].rename(columns={"pp_toi":"team_pp_toi"})
    df = pbp.merge(team_pp, on=["game_id","team"], how="left")
    df = df[df["team_pp_toi"] > 1.0]

    game_dates = tgl[["game_id","date"]].drop_duplicates().set_index("game_id")["date"]
    df["date"] = df["game_id"].map(game_dates)
    df["pp_share"] = df["toi_pp"] / (df["team_pp_toi"] + 1e-6)

    # Only games where player had meaningful PP time
    df = df[df["toi_pp"] > 0.3]

    recent_shares = {}
    for player_id, grp in df.groupby("player_id"):
        grp = grp.sort_values("date").tail(last_n_games)
        if len(grp) < 3:
            continue
        recent_shares[int(player_id)] = {
            "avg_share": float(grp["pp_share"].mean()),
            "games":     len(grp),
        }

    print(f"  Recent shares for {len(recent_shares):,} players")
    return recent_shares


def get_pp_share(player_id, team, pp_unit, feature_row,
                 pp_shares, team_pp_baselines, recent_shares):
    """
    Get player's expected PP TOI share of team PP time.

    Logic (lineup role is primary):
    1. Start with team's PP1 or PP2 baseline (from lineup assignment)
    2. Nudge based on recent games (last 10 with PP time) — moderate weight
    3. Nudge based on season history — small weight, only if same role
    """
    if pp_unit == 0:
        return PP_SHARE_FALLBACK[0]

    # Step 1: team baseline for this unit (primary signal from lineup)
    team_bl       = team_pp_baselines.get(str(team), {})
    team_baseline = team_bl.get(
        "pp1_usage" if pp_unit == 1 else "pp2_usage",
        PP_SHARE_FALLBACK[pp_unit]
    )

    # Start from team baseline
    share = team_baseline

    # Step 2: recent games nudge (last 10 games with PP time)
    recent = recent_shares.get(int(player_id))
    if recent and recent["games"] >= 3:
        # Weight recent more as sample grows, max 30% adjustment
        recent_weight = min(recent["games"] / 10.0, 1.0) * 0.30
        share = (1 - recent_weight) * share + recent_weight * recent["avg_share"]

    # Step 3: season history nudge (only if same role, small weight)
    try:
        current_season = int(feature_row.get("season", 20252026))
    except (TypeError, ValueError):
        current_season = 20252026
    s           = int(current_season)
    prev_season = (s // 10000 - 1) * 10000 + (s % 10000 - 1)

    hist = (pp_shares.get((int(player_id), str(team), current_season)) or
            pp_shares.get((int(player_id), str(team), prev_season)))

    if hist and int(hist["pp_unit_est"]) == pp_unit:
        hist_games  = float(hist["games_with_pp"])
        hist_share  = float(hist["avg_share"])
        # Max 20% weight from history, only when same role
        hist_weight = min(hist_games / 30.0, 1.0) * 0.20
        share = (1 - hist_weight) * share + hist_weight * hist_share

    return float(share)


# ── Lineup loading ────────────────────────────────────────────────────────────
def load_lineup(date_str=None):
    if date_str:
        path = LINEUP_DIR / f"lineups_{date_str.replace('-','')}.json"
    else:
        files = sorted(LINEUP_DIR.glob("lineups_*.json"))
        if not files:
            raise FileNotFoundError("No lineup files found")
        path = files[-1]

    with open(path) as f:
        data = json.load(f)

    print(f"Lineup date: {data.get('date', 'unknown')}")
    return data["teams"]


def extract_players_from_lineup(lineup, team, is_home):
    players   = []
    team_data = lineup.get(team, {})

    for line in team_data.get("forward_lines", []):
        line_num = line["line"]
        for pos in ["lw","c","rw"]:
            p = line.get(pos)
            if p and p.get("name"):
                players.append({
                    "name":           p["name"],
                    "dfo_id":         p.get("dfo_id"),
                    "position_group": "F",
                    "line_num":       line_num,
                    "pp_unit":        0,
                    "pk_unit":        0,
                    "team":           team,
                    "is_home":        is_home,
                })

    for pair in team_data.get("defense_pairs", []):
        pair_num = pair["pair"]
        for pos in ["ld","rd"]:
            p = pair.get(pos)
            if p and p.get("name"):
                players.append({
                    "name":           p["name"],
                    "dfo_id":         p.get("dfo_id"),
                    "position_group": "D",
                    "line_num":       pair_num,
                    "pp_unit":        0,
                    "pk_unit":        0,
                    "team":           team,
                    "is_home":        is_home,
                })

    pp1_names = {p["name"] for p in team_data.get("pp1", [])}
    pp2_names = {p["name"] for p in team_data.get("pp2", [])}
    pk1_names = {p["name"] for p in team_data.get("pk1", [])}
    pk2_names = {p["name"] for p in team_data.get("pk2", [])}

    for p in players:
        if p["name"] in pp1_names:
            p["pp_unit"] = 1
        elif p["name"] in pp2_names:
            p["pp_unit"] = 2
        if p["name"] in pk1_names:
            p["pk_unit"] = 1
        elif p["name"] in pk2_names:
            p["pk_unit"] = 2

    return players


# ── Name matching ─────────────────────────────────────────────────────────────
def normalize_name(name):
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower().strip()
    name = re.sub(r"[-']", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name


def expand_name_variants(name):
    parts    = name.strip().split()
    if not parts:
        return [name]
    first    = parts[0].lower()
    rest     = " ".join(parts[1:])
    variants = {name}
    if first in NICKNAMES:
        variants.add(f"{NICKNAMES[first].capitalize()} {rest}")
    if first in NICKNAMES_REVERSE:
        variants.add(f"{NICKNAMES_REVERSE[first].capitalize()} {rest}")
    return list(variants)


def build_name_lookup(features_df):
    current   = features_df[features_df["season"] == features_df["season"].max()].copy()
    primary   = {}
    secondary = {}
    tertiary  = {}

    for _, row in current.drop_duplicates(subset=["player_id"]).iterrows():
        name  = row["full_name"]
        entry = {
            "player_id": row["player_id"],
            "position":  row["position"],
            "team":      row["team"],
        }
        primary.setdefault(name, []).append(entry)
        secondary.setdefault(normalize_name(name), []).append(entry)
        tertiary.setdefault(normalize_name(name.split()[-1]), []).append(entry)

    return primary, secondary, tertiary


def match_player(dfo_name, position_group, team, primary, secondary, tertiary):
    def pick_best(entries):
        if len(entries) == 1:
            return entries[0]["player_id"]
        pos_matches = [e for e in entries
                       if (position_group == "D" and e["position"] == "D") or
                          (position_group == "F" and e["position"] != "D")]
        if len(pos_matches) == 1:
            return pos_matches[0]["player_id"]
        team_matches = [e for e in entries if e["team"] == team]
        if len(team_matches) == 1:
            return team_matches[0]["player_id"]
        return entries[0]["player_id"]

    parts    = dfo_name.split()
    variants = expand_name_variants(dfo_name)
    if len(parts) > 2:
        short = f"{parts[0]} {parts[-1]}"
        variants.append(short)
        variants.extend(expand_name_variants(short))

    for v in variants:
        if v in primary:
            return pick_best(primary[v])
    for v in variants:
        norm = normalize_name(v)
        if norm in secondary:
            return pick_best(secondary[norm])
    last = normalize_name(parts[-1])
    if last in tertiary:
        team_matches = [e for e in tertiary[last] if e["team"] == team]
        if team_matches:
            return pick_best(team_matches)
    return None


# ── Feature preparation ───────────────────────────────────────────────────────
def get_player_features(player_id, features_df, player_info):
    rows = features_df[features_df["player_id"] == player_id].sort_values("date")
    if len(rows) == 0:
        return None
    row = rows.iloc[-1].copy()
    row["is_home"]    = int(player_info["is_home"])
    row["is_defense"] = int(player_info["position_group"] == "D")
    row["is_forward"] = int(player_info["position_group"] == "F")
    row["is_center"]  = int(str(row.get("position","")) == "C")
    return row


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_player(feature_row, player_info, models, feature_lists,
                   model_types, calibrators, pp_shares,
                   team_pp_baselines, recent_shares):
    preds      = {}
    is_defense = int(feature_row.get("is_defense", 0))
    player_id  = feature_row.get("player_id", 0)
    team       = player_info.get("team", "")
    pp_unit    = int(player_info.get("pp_unit", 0))

    for model_name, model in models.items():
        if model_name == "scored_ev_goal_f" and is_defense:
            continue
        if model_name == "scored_ev_goal_d" and not is_defense:
            continue
        if model_name == "toi_pp":
            continue

        feats = feature_lists[model_name]
        x     = np.array([feature_row.get(f, 0) or 0 for f in feats]).reshape(1, -1)
        x     = np.nan_to_num(x, nan=0.0)
        X     = pd.DataFrame(x, columns=feats)

        mtype = model_types[model_name]
        if mtype == "classifier":
            raw_prob = model.predict_proba(X)[0, 1]
            if model_name in calibrators:
                prob = float(calibrators[model_name].transform([raw_prob])[0])
            else:
                prob = raw_prob
            preds["scored_ev_goal"] = prob
        else:
            preds[model_name] = float(model.predict(X)[0])

    # PP TOI via formula: team_pp_toi × player_pp_share
    pp_share = get_pp_share(player_id, team, pp_unit, feature_row,
                            pp_shares, team_pp_baselines, recent_shares)

    # Team PP TOI — use last20 rolling average (most recent)
    team_pp_toi = (
        feature_row.get("team_pp_toi_last20") or
        feature_row.get("team_pp_toi_last10") or
        feature_row.get("team_pp_toi_season_avg") or
        5.15
    )
    try:
        team_pp_toi = float(team_pp_toi)
        if np.isnan(team_pp_toi) or team_pp_toi < 0.5:
            team_pp_toi = 5.15
    except (TypeError, ValueError):
        team_pp_toi = 5.15

    preds["toi_pp"] = pp_share * team_pp_toi
    return preds


def compute_goals(preds, feature_row):
    ev_shots = max(preds.get("ev_shots", 0), 0)
    pp_shots = max(preds.get("pp_shots", 0), 0)
    toi_sh   = float(feature_row.get("toi_sh_last20", 0) or 0)
    sh_per60 = float(feature_row.get("regressed_sh_shots_per60", 5.44) or 5.44)
    sh_shots = toi_sh / 60 * sh_per60

    ev_sh_pct = float(feature_row.get("regressed_ev_shooting_pct", 0.098) or 0.098)
    finishing = float(feature_row.get("regressed_finishing_talent", 1.097) or 1.097)
    pp_sh_pct = float(feature_row.get("regressed_pp_shooting_pct", 0.144) or 0.144)

    ev_goals_lambda    = ev_shots * ev_sh_pct * finishing
    pp_goals_lambda    = pp_shots * pp_sh_pct
    total_goals_lambda = ev_goals_lambda + pp_goals_lambda

    return {
        "ev_goals_lambda":    ev_goals_lambda,
        "pp_goals_lambda":    pp_goals_lambda,
        "total_goals_lambda": total_goals_lambda,
        "ev_shots":           ev_shots,
        "pp_shots":           pp_shots,
        "sh_shots":           sh_shots,
        "total_shots_lambda": ev_shots + pp_shots + sh_shots,
    }


def compute_assists(preds):
    return max(preds.get("ev_assists", 0), 0) + max(preds.get("pp_assists", 0), 0)


def poisson_over_prob(lam, line):
    k = int(line + 0.5)
    return 1 - poisson.cdf(k - 1, lam)


def format_prob(p):
    p = max(0.001, min(0.999, p))
    if p >= 0.5:
        odds = -round((p / (1 - p)) * 100)
    else:
        odds = round(((1 - p) / p) * 100)
    return f"{p:.1%} ({odds:+d})"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("home_team", type=str)
    parser.add_argument("away_team", type=str)
    parser.add_argument("--date",      type=str,   default=None)
    parser.add_argument("--min-shots", type=float, default=1.0)
    args = parser.parse_args()

    home_team = args.home_team.upper()
    away_team = args.away_team.upper()
    date_str  = args.date.replace("-","") if args.date else None

    print(f"\n{'='*60}")
    print(f"PLAYER PROPS: {away_team} @ {home_team}")
    print(f"{'='*60}")

    print("\nLoading models...")
    models, feature_lists, model_types, calibrators, residual_stds = load_models()

    print("Loading PP shares...")
    pp_shares = load_pp_shares()
    print(f"  {len(pp_shares):,} player-team-season records")

    print("Computing team PP baselines...")
    team_pp_baselines = compute_team_pp_baselines(PBP_FILE, TEAM_LOGS_FILE,
                                                   last_n_games=20)

    print("Computing recent player PP shares...")
    recent_shares = compute_recent_player_pp_shares(PBP_FILE, TEAM_LOGS_FILE,
                                                     last_n_games=10)

    print("Loading player features...")
    features_df = pd.read_csv(PLAYER_FEATURES, low_memory=False)
    features_df["date"]       = pd.to_datetime(features_df["date"])
    features_df["is_defense"] = (features_df["position"] == "D").astype(int)
    features_df["is_forward"] = (features_df["position"] != "D").astype(int)
    features_df["is_center"]  = (features_df["position"] == "C").astype(int)
    features_df["is_home"]    = features_df["is_home"].astype(int)
    print(f"  {len(features_df):,} rows loaded")

    print("Loading lineups...")
    lineups = load_lineup(date_str)
    primary, secondary, tertiary = build_name_lookup(features_df)

    team_results = {}

    for team, is_home in [(home_team, True), (away_team, False)]:
        if team not in lineups:
            print(f"  WARNING: {team} not found in lineup file")
            continue

        players = extract_players_from_lineup(lineups, team, is_home)
        print(f"\n{'='*60}")
        print(f"  {team} ({'HOME' if is_home else 'AWAY'}) — {len(players)} players")
        print(f"{'='*60}")

        bl = team_pp_baselines.get(team, {})
        print(f"  Team PP usage: PP1={bl.get('pp1_usage',0.670):.3f}  "
              f"PP2={bl.get('pp2_usage',0.330):.3f}  "
              f"games={bl.get('games',0)}")

        team_shots      = 0.0
        team_goals      = 0.0
        unmatched_names = []
        player_rows     = []

        for p in players:
            pid = match_player(
                p["name"], p["position_group"], team,
                primary, secondary, tertiary
            )
            if pid is None:
                unmatched_names.append(p["name"])
                continue

            feat_row = get_player_features(pid, features_df, p)
            if feat_row is None:
                unmatched_names.append(p["name"])
                continue

            preds    = predict_player(feat_row, p, models, feature_lists,
                                      model_types, calibrators, pp_shares,
                                      team_pp_baselines, recent_shares)
            computed = compute_goals(preds, feat_row)
            assists  = compute_assists(preds)

            pos_label  = p["position_group"]
            line_label = f"L{p['line_num']}"
            pp_label   = f" PP{p['pp_unit']}" if p["pp_unit"] > 0 else ""
            role       = f"{pos_label} {line_label}{pp_label}"

            team_shots += computed["total_shots_lambda"]
            team_goals += computed["total_goals_lambda"]

            player_rows.append({
                "name":     p["name"],
                "role":     role,
                "toi_ev":   preds.get("toi_ev", 0),
                "toi_pp":   preds.get("toi_pp", 0),
                "shots":    computed["total_shots_lambda"],
                "ev_shots": computed["ev_shots"],
                "pp_shots": computed["pp_shots"],
                "goals":    computed["total_goals_lambda"],
                "assists":  assists,
                "points":   computed["total_goals_lambda"] + assists,
                "p_goal":   preds.get("scored_ev_goal", 0),
                "preds":    preds,
                "computed": computed,
            })

        player_rows.sort(key=lambda x: x["shots"], reverse=True)

        print(f"\n  {'Player':<22} {'Role':<12} {'TOI_EV':>6} {'TOI_PP':>6} "
              f"{'Shots':>6} {'Goals':>6} {'Assists':>7} {'Points':>7}")
        print(f"  {'-'*22} {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")

        for r in player_rows:
            if r["shots"] < args.min_shots:
                continue
            print(f"  {r['name']:<22} {r['role']:<12} "
                  f"{r['toi_ev']:>6.1f} {r['toi_pp']:>6.1f} "
                  f"{r['shots']:>6.1f} {r['goals']:>6.3f} "
                  f"{r['assists']:>7.3f} {r['points']:>7.3f}")

        print(f"\n  Team totals: shots={team_shots:.1f}  goals={team_goals:.3f}")

        if unmatched_names:
            print(f"\n  Unmatched ({len(unmatched_names)}): {', '.join(unmatched_names)}")

        team_results[team] = {
            "players":    player_rows,
            "team_shots": team_shots,
            "team_goals": team_goals,
            "is_home":    is_home,
        }

    if len(team_results) == 2:
        print(f"\n{'='*60}")
        print(f"GAME SUMMARY")
        print(f"{'='*60}")
        for team, res in team_results.items():
            label = "HOME" if res["is_home"] else "AWAY"
            print(f"  {team} ({label}): {res['team_shots']:.1f} shots  "
                  f"{res['team_goals']:.2f} goals")
        total_shots = sum(r["team_shots"] for r in team_results.values())
        total_goals = sum(r["team_goals"] for r in team_results.values())
        print(f"  Game total: {total_shots:.1f} shots  {total_goals:.2f} goals")

        print(f"\n  TOP SHOT PROPS (o2.5):")
        all_players = []
        for res in team_results.values():
            all_players.extend(res["players"])
        all_players.sort(key=lambda x: x["shots"], reverse=True)
        for r in all_players[:10]:
            p_over = poisson_over_prob(r["shots"], 2.5)
            print(f"    {r['name']:<22} {r['shots']:.1f} shots  "
                  f"o2.5: {format_prob(p_over)}")

        print(f"\n  TOP GOAL PROPS (anytime scorer):")
        all_players.sort(key=lambda x: x["goals"], reverse=True)
        for r in all_players[:10]:
            p_goal = poisson_over_prob(r["goals"], 0.5)
            print(f"    {r['name']:<22} λ={r['goals']:.3f}  "
                  f"anytime: {format_prob(p_goal)}")

        print(f"\n  TOP POINT PROPS (o0.5):")
        all_players.sort(key=lambda x: x["points"], reverse=True)
        for r in all_players[:10]:
            p_point = poisson_over_prob(r["points"], 0.5)
            print(f"    {r['name']:<22} {r['points']:.3f} pts  "
                  f"o0.5: {format_prob(p_point)}")


if __name__ == "__main__":
    main()