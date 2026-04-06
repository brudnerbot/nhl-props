"""
predict_player_props.py

Generates player prop predictions for a given game using:
  - Today's lineup from data/raw/lineups/lineups_YYYYMMDD.json
  - Player features from data/processed/player_features.csv
  - Trained models from models/player/
  - Team PP baselines computed on-the-fly from recent PBP data
  - Opponent goalie GSAx adjustment in goal formula

PP TOI:   formula — team_pp_toi × player_pp_share
EV shots: formula — rate_last10 × pred_toi_ev (60% recent, 40% season)
PP shots: formula — rate_last10 × pred_toi_pp (50% recent, 50% season)
Goals:    formula — shots × regressed_sh% × finishing × goalie_adj
Assists:  XGBoost models (ev_assists, pp_assists)
Goals classifier: scored_ev_goal_f (forwards only, calibrated)

Usage:
    python scripts/predict_player_props.py ANA TOR
    python scripts/predict_player_props.py ANA TOR --date 2026-03-30
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
from scipy.stats import nbinom

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models/player"

PLAYER_FEATURES  = DATA_DIR / "processed/player_features.csv"
LINEUP_DIR       = DATA_DIR / "raw/lineups"
PP_SHARES_FILE   = DATA_DIR / "raw/player_pp_shares.csv"
PBP_FILE         = DATA_DIR / "raw/player_pbp_stats/player_pbp_stats.csv"
TEAM_LOGS_FILE   = DATA_DIR / "raw/team_game_logs/team_game_logs.csv"
GOALIE_FEAT_FILE = DATA_DIR / "processed/goalie_features.csv"
GOALIE_LOGS_FILE = DATA_DIR / "raw/goalie_game_logs/goalie_game_logs.csv"

SHOT_LINES  = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
POINT_LINES = [0.5, 1.5, 2.5]

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


# ── Goalie loading ────────────────────────────────────────────────────────────
def load_goalie_features():
    """Load goalie features, add names, compute league avg shots faced."""
    gf = pd.read_csv(GOALIE_FEAT_FILE, low_memory=False)
    gf["date"] = pd.to_datetime(gf["date"])
    gf = gf[gf["toi"] > 30].sort_values("date")

    raw = pd.read_csv(
        GOALIE_LOGS_FILE,
        usecols=["player_id", "first_name", "last_name"],
        low_memory=False
    ).drop_duplicates("player_id")
    gf = gf.merge(raw, on="player_id", how="left")

    recent = gf[gf["season"] >= 20222023]
    avg_shots_faced = float(recent["shots_against"].mean())
    print(f"  League avg shots faced by starters: {avg_shots_faced:.1f}")

    return gf, avg_shots_faced


def get_goalie_gsax(goalie_df, team):
    """Get most recent starting goalie's GSAx dict for a team. Always returns dict."""
    team_gf = goalie_df[goalie_df["team"] == team]
    if len(team_gf) == 0:
        return {"gsax_last20": 0.0, "gsax_last30": 0.0,
                "regressed_gsax": 0.0, "name": "Unknown"}
    latest = team_gf.iloc[-1]
    return {
        "gsax_last20":    float(latest["gsax_last20"])
                          if pd.notna(latest["gsax_last20"]) else 0.0,
        "gsax_last30":    float(latest["gsax_last30"])
                          if pd.notna(latest["gsax_last30"]) else 0.0,
        "regressed_gsax": float(latest["regressed_gsax_per_game"])
                          if pd.notna(latest["regressed_gsax_per_game"]) else 0.0,
        "name":           str(latest["last_name"])
                          if "last_name" in latest.index and pd.notna(latest["last_name"])
                          else "Unknown",
    }


# ── PP data loading and computation ──────────────────────────────────────────
def load_pp_shares():
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
    print(f"Computing team PP baselines (last {last_n_games} games)...")
    pbp = pd.read_csv(pbp_path, low_memory=False)
    tgl = pd.read_csv(team_logs_path, low_memory=False)
    tgl["date"] = pd.to_datetime(tgl["date"])

    team_pp = tgl[["game_id","team","pp_toi","date"]].rename(
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
    if pp_unit == 0:
        return PP_SHARE_FALLBACK[0]

    team_bl       = team_pp_baselines.get(str(team), {})
    team_baseline = team_bl.get(
        "pp1_usage" if pp_unit == 1 else "pp2_usage",
        PP_SHARE_FALLBACK[pp_unit]
    )
    share = team_baseline

    recent = recent_shares.get(int(player_id))
    if recent and recent["games"] >= 3:
        recent_weight = min(recent["games"] / 10.0, 1.0) * 0.30
        share = (1 - recent_weight) * share + recent_weight * recent["avg_share"]

    try:
        current_season = int(feature_row.get("season", 20252026))
    except (TypeError, ValueError):
        current_season = 20252026
    s           = int(current_season)
    prev_season = (s // 10000 - 1) * 10000 + (s % 10000 - 1)

    hist = (pp_shares.get((int(player_id), str(team), current_season)) or
            pp_shares.get((int(player_id), str(team), prev_season)))

    if hist and int(hist["pp_unit_est"]) == pp_unit:
        hist_weight = min(float(hist["games_with_pp"]) / 30.0, 1.0) * 0.20
        share = (1 - hist_weight) * share + hist_weight * float(hist["avg_share"])

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

def load_prop_calibration():
    """Load empirical calibration curves for prop probabilities."""
    path = DATA_DIR / "processed/prop_calibration.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def empirical_over_prob(lam, line, stat, calibration):
    """
    Look up empirical P(stat > line | season_avg=lam) from calibration table.
    Falls back to NegBin if calibration not available.
    NegBin dispersion parameters fitted from empirical data:
      shots r=3.87, goals r=2.32, ev_assists r=6.46, points r=3.18
    """
    R_BY_STAT = {"shots": 3.87, "goals": 2.32,
                 "ev_assists": 6.46, "points": 3.18}
    r = R_BY_STAT.get(stat, 3.5)
    k = int(line + 0.5)

    # NegBin fallback
    def negbin_prob(lam, r, k):
        if lam <= 0: return 0.0
        p = r / (r + lam)
        return 1.0 - sum(nbinom.pmf(i, r, p) for i in range(k))

    if not calibration or stat not in calibration:
        return negbin_prob(lam, r, k)

    line_str  = str(float(line))
    line_data = calibration[stat].get(line_str, {})
    if not line_data:
        return negbin_prob(lam, r, k)

    # Find nearest bucket
    buckets = {float(k): v for k, v in line_data.items()}
    if not buckets:
        return negbin_prob(lam, r, k)

    keys     = sorted(buckets.keys())
    nearest  = min(keys, key=lambda x: abs(x - lam))
    distance = abs(nearest - lam)

    # If within 0.5 of a bucket with good sample, use empirical
    if distance <= 0.5 and buckets[nearest]["n"] >= 50:
        # Interpolate between two nearest buckets
        lower = max([k for k in keys if k <= lam], default=None)
        upper = min([k for k in keys if k >= lam], default=None)
        if lower is not None and upper is not None and lower != upper:
            w = (lam - lower) / (upper - lower)
            p_lower = buckets[lower]["actual"]
            p_upper = buckets[upper]["actual"]
            return float(p_lower * (1-w) + p_upper * w)
        return float(buckets[nearest]["actual"])

    return negbin_prob(lam, r, k)

# ── Prediction ────────────────────────────────────────────────────────────────
def predict_player(feature_row, player_info, models, feature_lists,
                   model_types, calibrators, pp_shares,
                   team_pp_baselines, recent_shares):
    preds      = {}
    is_defense = int(feature_row.get("is_defense", 0))
    player_id  = feature_row.get("player_id", 0)
    team       = player_info.get("team", "")
    pp_unit    = int(player_info.get("pp_unit", 0))
    eps        = 1e-6

    for model_name, model in models.items():
        if model_name == "scored_ev_goal_f" and is_defense:
            continue
        if model_name == "scored_ev_goal_d" and not is_defense:
            continue
        # toi_pp, ev_shots, pp_shots replaced by formulas below
        if model_name in ("toi_pp", "ev_shots", "pp_shots"):
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
            model_pred = float(model.predict(X)[0])

            # For assist models, blend with recent rate x TOI
            # to capture hot/cold streaks not reflected in season avg
            if model_name in ("ev_assists", "pp_assists"):
                pass  # XGBoost prediction used directly — blending hurts MAE

            preds[model_name] = model_pred

    # ── EV shots: rate x TOI formula ─────────────────────────────────────────
    # 60% weight on last10 rate, 40% on season rate
    # Preserves tail distribution better than XGBoost regression
    pred_toi_ev      = max(float(
        feature_row.get("toi_ev_last10") or
        feature_row.get("toi_ev_season_avg") or 18.0), 1.0)

    # Predicted TOI from XGBoost toi_ev model
    if "toi_ev" in models:
        feats = feature_lists["toi_ev"]
        x = np.array([feature_row.get(f, 0) or 0 for f in feats]).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0)
        pred_toi_ev = max(float(models["toi_ev"].predict(
            pd.DataFrame(x, columns=feats))[0]), 1.0)
        preds["toi_ev"] = pred_toi_ev

    ev_shots_last10  = float(feature_row.get("ev_shots_last10", 0) or 0)
    toi_ev_last10    = float(feature_row.get("toi_ev_last10", 1) or 1)
    ev_rate_last10   = min(ev_shots_last10 / (toi_ev_last10 / 60 + eps), 25.0)

    ev_shots_season  = float(feature_row.get("ev_shots_season_avg", 0) or 0)
    toi_ev_season    = float(feature_row.get("toi_ev_season_avg", 1) or 1)
    ev_rate_season   = min(ev_shots_season / (toi_ev_season / 60 + eps), 25.0)

    ev_rate_blended  = 0.60 * ev_rate_last10 + 0.40 * ev_rate_season
    preds["ev_shots"] = ev_rate_blended * (pred_toi_ev / 60)

    # ── PP TOI: team baseline × player share ──────────────────────────────────
    pp_share = get_pp_share(player_id, team, pp_unit, feature_row,
                            pp_shares, team_pp_baselines, recent_shares)

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

    pred_toi_pp       = pp_share * team_pp_toi
    preds["toi_pp"]   = pred_toi_pp

    # ── PP shots: rate x TOI formula ─────────────────────────────────────────
    # 50% weight on last10 rate, 50% on season rate (PP more stable than EV)
    pp_shots_last10  = float(feature_row.get("pp_shots_last10", 0) or 0)
    toi_pp_last10    = float(feature_row.get("toi_pp_last10", 0.5) or 0.5)
    pp_rate_last10   = min(pp_shots_last10 / (toi_pp_last10 / 60 + eps), 40.0)

    pp_shots_season  = float(feature_row.get("pp_shots_season_avg", 0) or 0)
    toi_pp_season    = float(feature_row.get("toi_pp_season_avg", 0.5) or 0.5)
    pp_rate_season   = min(pp_shots_season / (toi_pp_season / 60 + eps), 40.0)

    pp_rate_blended  = 0.50 * pp_rate_last10 + 0.50 * pp_rate_season
    preds["pp_shots"] = pp_rate_blended * (pred_toi_pp / 60)

    return preds


def compute_goals(preds, feature_row, opp_goalie, avg_shots_faced):
    ev_shots = max(preds.get("ev_shots", 0), 0)
    pp_shots = max(preds.get("pp_shots", 0), 0)
    toi_sh   = float(feature_row.get("toi_sh_last20", 0) or 0)
    sh_per60 = float(feature_row.get("regressed_sh_shots_per60", 5.44) or 5.44)
    sh_shots = toi_sh / 60 * sh_per60

    ev_sh_pct = float(feature_row.get("regressed_ev_shooting_pct", 0.098) or 0.098)
    finishing = float(feature_row.get("regressed_finishing_talent", 1.097) or 1.097)
    pp_sh_pct = float(feature_row.get("regressed_pp_shooting_pct", 0.144) or 0.144)

    gsax       = float(opp_goalie.get("gsax_last20", 0) or 0)
    goalie_adj = 1.0 - (gsax / avg_shots_faced)
    goalie_adj = max(0.70, min(1.30, goalie_adj))

    ev_goals_lambda    = ev_shots * ev_sh_pct * finishing * goalie_adj
    # Recent goals rate adjustment (captures hot streaks)
    goals_last10  = float(feature_row.get("goals_last10", 0) or 0)
    # goals_last10 is already per-game avg from rolling window
    # Blend: 60% formula, 40% recent rate
    ev_goals_lambda = 0.60 * ev_goals_lambda + 0.40 * goals_last10
    pp_goals_lambda    = pp_shots * pp_sh_pct * goalie_adj
    total_goals_lambda = ev_goals_lambda + pp_goals_lambda

    return {
        "ev_goals_lambda":    ev_goals_lambda,
        "pp_goals_lambda":    pp_goals_lambda,
        "total_goals_lambda": total_goals_lambda,
        "ev_shots":           ev_shots,
        "pp_shots":           pp_shots,
        "sh_shots":           sh_shots,
        "total_shots_lambda": ev_shots + pp_shots + sh_shots,
        "goalie_adj":         goalie_adj,
    }


def compute_assists(preds):
    return max(preds.get("ev_assists", 0), 0) + max(preds.get("pp_assists", 0), 0)


def nb_over_prob(lam, line, r=3.88):
    """
    Negative Binomial probability of exceeding line.
    r=3.88 fitted from empirical shot data (variance/mean ratio = 1.405).
    NegBin better accounts for overdispersion in shot counts vs Poisson.
    """
    if lam <= 0:
        return 0.0
    k   = int(line + 0.5)
    p   = r / (r + lam)
    cdf = sum(nbinom.pmf(i, r, p) for i in range(k))
    return 1.0 - cdf


def format_prob(p):
    p = max(0.001, min(0.999, p))
    if p >= 0.5:
        odds = -round((p / (1 - p)) * 100)
    else:
        odds = round(((1 - p) / p) * 100)
    return f"{p:.1%} ({odds:+d})"

# ── Zone profile loading ──────────────────────────────────────────────────────
def load_zone_data():
    """Load zone averages, player zone shots, team zone defense."""
    zone_avgs_path = DATA_DIR / "processed/zone_averages.json"
    if not zone_avgs_path.exists():
        return None, None, None
    with open(zone_avgs_path) as f:
        zone_avgs = json.load(f)

    pz = pd.read_csv(DATA_DIR / "processed/player_zone_shots.csv", low_memory=False)
    pz["player_id"] = pz["player_id"].astype("Int64")

    tz = pd.read_csv(DATA_DIR / "processed/team_zone_defense.csv", low_memory=False)

    return zone_avgs, pz, tz


def print_zone_matchup(home_team, away_team, player_rows_home, player_rows_away,
                       features_df, zone_avgs, pz_df, tz_df):
    """Print zone shot profiles for top players and team defense context."""
    if zone_avgs is None:
        return

    ZONES        = ["net_front","slot","left_flank","right_flank",
                    "left_point","mid_point","right_point"]
    ZONE_SHORT   = ["net","slot","l.flk","r.flk","l.pt","mid","r.pt"]
    pos_props    = zone_avgs["position_zone_props"]
    league_def   = zone_avgs["league_defense_avg"]
    league_total = sum(league_def.values())
    league_def_p = {z: league_def[z]/league_total for z in ZONES}

    def get_player_zone_props(player_id):
        rows = pz_df[pz_df["player_id"]==player_id].sort_values("date")
        if len(rows) == 0:
            return None
        latest = rows.iloc[-1]
        vals = {z: float(latest.get(f"{z}_season_avg") or 0) for z in ZONES}
        total = sum(vals.values()) + 1e-6
        return {z: v/total for z,v in vals.items()}

    def get_team_zone_defense(team):
        rows = tz_df[tz_df["defending_team"]==team].sort_values("date")
        if len(rows) == 0:
            return None
        latest = rows.iloc[-1]
        vals = {z: float(latest.get(f"opp_{z}_allowed_last30") or 0) for z in ZONES}
        total = sum(vals.values()) + 1e-6
        return {z: v/total for z,v in vals.items()}

    def fmt_diff(p, avg):
        d = p - avg
        sign = "+" if d >= 0 else ""
        return f"{p:.0%}({sign}{d:+.0%})"

    print(f"\n{'='*60}")
    print(f"ZONE MATCHUP CONTEXT")
    print(f"{'='*60}")

    # Header
    print(f"\n  {'Player':<22} {'Pos':>3} | " +
          " | ".join(f"{z:>10}" for z in ZONE_SHORT))
    print(f"  {'-'*22} {'-'*3}-" + "-+-".join(["-"*10]*len(ZONES)))

    for team, player_rows, label in [
        (home_team, player_rows_home, "HOME"),
        (away_team, player_rows_away, "AWAY"),
    ]:
        print(f"\n  -- {team} ({label}) players vs position avg --")
        top = sorted(player_rows, key=lambda r: r["shots"], reverse=True)[:6]
        for r in top:
            pid = features_df[features_df["full_name"]==r["name"]]["player_id"]
            if len(pid) == 0:
                continue
            pid = int(pid.iloc[0])
            props = get_player_zone_props(pid)
            if props is None:
                continue
            is_def = "D" in r["role"]
            pos_key = "defense" if is_def else "forward"
            pp = pos_props[pos_key]
            parts = [fmt_diff(props[z], pp[z]) for z in ZONES]
            print(f"  {r['name']:<22} {'D' if is_def else 'F':>3} | " +
                  " | ".join(f"{p:>10}" for p in parts))

    # Team zone defense
    print(f"\n  -- Team zone shots ALLOWED vs league avg (last 30) --")
    print(f"  {'Team':<6} | " + " | ".join(f"{z:>10}" for z in ZONE_SHORT))
    for team in [home_team, away_team]:
        props = get_team_zone_defense(team)
        if props is None:
            continue
        parts = [fmt_diff(props[z], league_def_p[z]) for z in ZONES]
        print(f"  {team:<6} | " + " | ".join(f"{p:>10}" for p in parts))

# ── B2B loading and detection ─────────────────────────────────────────────────
def load_b2b_effects():
    path = DATA_DIR / "processed/b2b_effects.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def detect_b2b(team, date_str=None):
    """Check if team played yesterday based on team game logs."""
    tgl = pd.read_csv(TEAM_LOGS_FILE, low_memory=False)
    tgl["date"] = pd.to_datetime(tgl["date"])
    team_games = tgl[tgl["team"]==team].sort_values("date")
    if len(team_games) == 0:
        return False
    last_game = team_games.iloc[-1]["date"]
    if date_str:
        target = pd.to_datetime(date_str)
    else:
        target = pd.Timestamp.now().normalize()
    days_rest = (target - last_game).days
    return days_rest == 1


def get_team_b2b_ratio(team, stat, b2b_data):
    """Get team-specific B2B ratio, fall back to league average."""
    team_ratios = b2b_data.get("team_b2b", {}).get(team, {})
    if stat in team_ratios:
        return float(team_ratios[stat]["ratio"])
    league = b2b_data.get("league_team_b2b", {})
    if stat in league:
        return float(league[stat]["ratio"])
    return 1.0


def get_player_b2b_ratio(player_id, stat, is_defense, b2b_data):
    """Get player-specific B2B ratio, fall back to position league average."""
    player_ratios = b2b_data.get("player_b2b", {}).get(str(int(player_id)), {})
    if stat in player_ratios:
        return float(player_ratios[stat]["ratio"])
    pos_key = "defense" if is_defense else "forward"
    league = b2b_data.get("league_player_b2b", {}).get(pos_key, {})
    if stat in league:
        return float(league[stat]["ratio"])
    return 1.0

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

    print("Loading prop calibration...")
    prop_calibration = load_prop_calibration()
    print(f"  Stats calibrated: {list(prop_calibration.keys())}")

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

    print("Loading goalie features...")
    goalie_df, avg_shots_faced = load_goalie_features()
    print(f"  {len(goalie_df):,} starter game records loaded")

    print("Loading zone data...")
    zone_avgs, pz_df, tz_df = load_zone_data()

    print("Loading B2B effects...")
    b2b_data = load_b2b_effects()

    print("Loading lineups...")
    lineups = load_lineup(date_str)
    primary, secondary, tertiary = build_name_lookup(features_df)

    team_results = {}

    for team, is_home in [(home_team, True), (away_team, False)]:
        opp_team   = away_team if is_home else home_team
        opp_goalie = get_goalie_gsax(goalie_df, opp_team)
        goalie_adj = 1.0 - (opp_goalie["gsax_last20"] / avg_shots_faced)
        goalie_adj = max(0.70, min(1.30, goalie_adj))

        # Detect if this team is on a B2B
        is_b2b = detect_b2b(team, date_str)
        team_shot_b2b_ratio = get_team_b2b_ratio(team, "ev_shots_on_goal_for", b2b_data)
        team_goal_b2b_ratio = get_team_b2b_ratio(team, "goals_for", b2b_data)

        players = extract_players_from_lineup(lineups, team, is_home)

        print(f"\n{'='*60}")
        print(f"  {team} ({'HOME' if is_home else 'AWAY'}) — {len(players)} players")
        print(f"  Opp goalie: {opp_goalie['name']} ({opp_team})  "
              f"GSAx_L20={opp_goalie['gsax_last20']:+.2f}  adj={goalie_adj:.3f}")
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

            # Apply B2B adjustments
            if is_b2b:
                pid_int    = int(feat_row.get("player_id", 0))
                is_def     = int(feat_row.get("is_defense", 0))
                shot_ratio = get_player_b2b_ratio(pid_int, "shots", is_def, b2b_data)
                # Blend player ratio with team ratio (60/40)
                combined_shot_ratio = 0.60 * shot_ratio + 0.40 * team_shot_b2b_ratio
                preds["ev_shots"] = preds.get("ev_shots", 0) * combined_shot_ratio
                preds["pp_shots"] = preds.get("pp_shots", 0) * combined_shot_ratio

            computed = compute_goals(preds, feat_row,
                                     opp_goalie=opp_goalie,
                                     avg_shots_faced=avg_shots_faced)
            assists  = compute_assists(preds)

            pos_label  = p["position_group"]
            line_label = f"L{p['line_num']}"
            pp_label   = f" PP{p['pp_unit']}" if p["pp_unit"] > 0 else ""
            role       = f"{pos_label} {line_label}{pp_label}"

            team_shots += computed["total_shots_lambda"]
            team_goals += computed["total_goals_lambda"]

            player_rows.append({
                "name":             p["name"],
                "role":             role,
                "toi_ev":           preds.get("toi_ev", 0),
                "toi_pp":           preds.get("toi_pp", 0),
                "shots":            computed["total_shots_lambda"],
                "shots_season_avg": float(feat_row.get("shots_season_avg", 0) or 0),
                "goals_season_avg": float(feat_row.get("goals_season_avg", 0) or 0),
                "points_season_avg":float(feat_row.get("points_season_avg", 0) or 0),
                "ev_shots":         computed["ev_shots"],
                "pp_shots":         computed["pp_shots"],
                "goals":            computed["total_goals_lambda"],
                "assists":          assists,
                "points":           computed["total_goals_lambda"] + assists,
                "p_goal":           preds.get("scored_ev_goal", 0),
                "preds":            preds,
                "computed":         computed,
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
            b2b_label = " ⚡ B2B" if detect_b2b(team, date_str) else ""
            print(f"  {team} ({label}){b2b_label}: {res['team_shots']:.1f} shots  "
                  f"{res['team_goals']:.2f} goals")
        total_shots = sum(r["team_shots"] for r in team_results.values())
        total_goals = sum(r["team_goals"] for r in team_results.values())
        print(f"  Game total: {total_shots:.1f} shots  {total_goals:.2f} goals")

        all_players = []
        for res in team_results.values():
            all_players.extend(res["players"])

        # ── Shot props — full line ladder ─────────────────────────────────
        print(f"\n  SHOT PROPS (projected / line ladder):")
        print(f"  {'Player':<22} {'Proj':>5} {'u2.5':>8} {'o2.5':>8} {'u3.5':>8} "
              f"{'o3.5':>8} {'o4.5':>8} {'o5.5':>8} {'o6.5':>8}")
        print(f"  {'-'*22} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        all_players.sort(key=lambda x: x["shots"], reverse=True)
        for r in all_players[:12]:
            cal_lam = r.get("shots_season_avg") or r["shots"]
            o25 = empirical_over_prob(cal_lam, 2.5, "shots", prop_calibration)
            o35 = empirical_over_prob(cal_lam, 3.5, "shots", prop_calibration)
            o45 = empirical_over_prob(cal_lam, 4.5, "shots", prop_calibration)
            o55 = empirical_over_prob(cal_lam, 5.5, "shots", prop_calibration)
            o65 = empirical_over_prob(cal_lam, 6.5, "shots", prop_calibration)
            u25 = 1 - o25
            u35 = 1 - o35
            def fmt(p):
                p = max(0.001, min(0.999, p))
                odds = -round((p/(1-p))*100) if p >= 0.5 else round(((1-p)/p)*100)
                return f"{p:.0%}({odds:+d})"
            print(f"  {r['name']:<22} {r['shots']:>5.1f} "
                  f"{fmt(u25):>8} {fmt(o25):>8} {fmt(u35):>8} "
                  f"{fmt(o35):>8} {fmt(o45):>8} {fmt(o55):>8} {fmt(o65):>8}")

        # ── Goal props ────────────────────────────────────────────────────
        print(f"\n  GOAL PROPS (anytime scorer):")
        print(f"  {'Player':<22} {'Proj':>6} {'anytime':>10} {'2+ goals':>10}")
        print(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*10}")
        all_players.sort(key=lambda x: x["goals"], reverse=True)
        for r in all_players[:12]:
            cal_lam = r.get("goals_season_avg") or r["goals"]
            p1 = empirical_over_prob(cal_lam, 0.5, "goals", prop_calibration)
            p2 = empirical_over_prob(cal_lam, 1.5, "goals", prop_calibration)
            print(f"  {r['name']:<22} {r['goals']:>6.3f} "
                  f"{format_prob(p1):>10} {format_prob(p2):>10}")

        # ── Point props ───────────────────────────────────────────────────
        print(f"\n  POINT PROPS:")
        print(f"  {'Player':<22} {'Proj':>6} {'o0.5':>10} {'o1.5':>10} {'o2.5':>10}")
        print(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
        all_players.sort(key=lambda x: x["points"], reverse=True)
        for r in all_players[:12]:
            cal_lam = r.get("points_season_avg") or r["points"]
            p05 = empirical_over_prob(cal_lam, 0.5,  "points", prop_calibration)
            p15 = empirical_over_prob(cal_lam, 1.5,  "points", prop_calibration)
            p25 = empirical_over_prob(cal_lam, 2.5,  "points", prop_calibration)
            print(f"  {r['name']:<22} {r['points']:>6.3f} "
                  f"{format_prob(p05):>10} {format_prob(p15):>10} {format_prob(p25):>10}")

        print_zone_matchup(
            home_team, away_team,
            team_results[home_team]["players"],
            team_results[away_team]["players"],
            features_df, zone_avgs, pz_df, tz_df,
        )


if __name__ == "__main__":
    main()