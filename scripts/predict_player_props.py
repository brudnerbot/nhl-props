"""
predict_player_props.py

Generates player prop predictions for a given game using:
  - Today's lineup from data/raw/lineups/lineups_YYYYMMDD.json
  - Player features from data/processed/player_features.csv
  - Trained models from models/player/

Usage:
    python scripts/predict_player_props.py TOR VAN
    python scripts/predict_player_props.py TOR VAN --date 2026-03-28

Output:
    - Per-player projections: TOI, shots, goals, assists, points
    - Team aggregates: total shots, goals
    - Prop lines with over/under probabilities
"""

import argparse
import pickle
import re
import warnings
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson, norm

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models/player"

PLAYER_FEATURES = DATA_DIR / "processed/player_features.csv"
LINEUP_DIR      = DATA_DIR / "raw/lineups"

# Book lines for common props
SHOT_LINES   = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
POINT_LINES  = [0.5, 1.5, 2.5]
GOAL_LINES   = [0.5]
ASSIST_LINES = [0.5, 1.5]


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models():
    models      = {}
    model_types = {}
    calibrators = {}

    with open(MODEL_DIR / "feature_lists.pkl", "rb") as f:
        feature_lists = pickle.load(f)
    with open(MODEL_DIR / "model_types.pkl", "rb") as f:
        model_types = pickle.load(f)
    with open(MODEL_DIR / "residual_stds.pkl", "rb") as f:
        residual_stds = pickle.load(f)

    if (MODEL_DIR / "calibrators.pkl").exists():
        with open(MODEL_DIR / "calibrators.pkl", "rb") as f:
            calibrators = pickle.load(f)

    for target, mtype in model_types.items():
        path = MODEL_DIR / f"{target}.json"
        if not path.exists():
            continue
        if mtype == "classifier":
            m = xgb.XGBClassifier()
        else:
            m = xgb.XGBRegressor()
        m.load_model(str(path))
        models[target] = m

    return models, feature_lists, model_types, calibrators, residual_stds


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
        data = __import__("json").load(f)

    print(f"Lineup date: {data.get('date', 'unknown')}")
    return data["teams"]


def extract_players_from_lineup(lineup, team, is_home):
    """
    Extract all players with their role context.
    Returns list of dicts with name, position_group, line_num, pp_unit.
    """
    players = []
    team_data = lineup.get(team, {})

    # Forwards
    for line in team_data.get("forward_lines", []):
        line_num = line["line"]
        for pos in ["lw", "c", "rw"]:
            p = line.get(pos)
            if p and p.get("name"):
                players.append({
                    "name":     p["name"],
                    "dfo_id":   p.get("dfo_id"),
                    "position_group": "F",
                    "line_num": line_num,
                    "pp_unit":  0,  # filled below
                    "pk_unit":  0,
                    "team":     team,
                    "is_home":  is_home,
                })

    # Defense
    for pair in team_data.get("defense_pairs", []):
        pair_num = pair["pair"]
        for pos in ["ld", "rd"]:
            p = pair.get(pos)
            if p and p.get("name"):
                players.append({
                    "name":     p["name"],
                    "dfo_id":   p.get("dfo_id"),
                    "position_group": "D",
                    "line_num": pair_num,
                    "pp_unit":  0,
                    "pk_unit":  0,
                    "team":     team,
                    "is_home":  is_home,
                })

    # Assign PP units
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
    """Lowercase, remove accents/hyphens, strip middle names."""
    import unicodedata
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower().strip()
    name = re.sub(r"[-']", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name


def build_name_lookup(features_df):
    """
    Build multiple lookup dicts for flexible name matching.
    Primary: full_name → {player_id, position, team}
    Secondary: normalized name → player_id
    """
    current = features_df[features_df["season"] == features_df["season"].max()].copy()

    # Primary lookup: exact name
    primary = {}
    for _, row in current.drop_duplicates(subset=["player_id"]).iterrows():
        name = row["full_name"]
        if name not in primary:
            primary[name] = []
        primary[name].append({
            "player_id": row["player_id"],
            "position":  row["position"],
            "team":      row["team"],
        })

    # Secondary: normalized name → list of player_ids
    secondary = {}
    for name, entries in primary.items():
        norm = normalize_name(name)
        if norm not in secondary:
            secondary[norm] = []
        secondary[norm].extend(entries)

    # Tertiary: last name only (for partial matches)
    tertiary = {}
    for name, entries in primary.items():
        last = normalize_name(name.split()[-1])
        if last not in tertiary:
            tertiary[last] = []
        tertiary[last].extend(entries)

    return primary, secondary, tertiary


def match_player(dfo_name, position_group, team, primary, secondary, tertiary):
    """Try multiple strategies to match a DFO name to a player_id."""

    # Common nickname <-> formal name mappings (bidirectional)
NICKNAMES = {
    "matt": "matthew",
    "mike": "michael",
    "alex": "alexander",
    "nick": "nicholas",
    "mitch": "mitchell",
    "cal": "callan",
    "zach": "zachary",
    "jake": "jacob",
    "pat": "patrick",
    "tj": "timothy",
    "jp": "jean-pierre",
    "j.t.": "john",
    "jt": "john",
    "phil": "philip",
    "will": "william",
    "bill": "william",
    "rick": "richard",
    "dan": "daniel",
    "max": "maxim",
    "jon": "jonathan",
    "tom": "thomas",
    "rob": "robert",
    "bob": "robert",
    "cam": "cameron",
    "nate": "nathan",
    "drew": "andrew",
    "steve": "steven",
    "chris": "christopher",
    "tony": "anthony",
    "brad": "bradley",
    "sam": "samuel",
    "mat": "matthew",
}
# Build reverse mapping too
NICKNAMES_REVERSE = {v: k for k, v in NICKNAMES.items()}


def expand_name_variants(name):
    """
    Generate all plausible name variants for a given name.
    e.g. "Matt Savoie" → ["Matt Savoie", "Matthew Savoie"]
         "Matthew Savoie" → ["Matthew Savoie", "Matt Savoie"]
    """
    parts = name.strip().split()
    if not parts:
        return [name]

    first = parts[0].lower()
    rest  = " ".join(parts[1:])
    variants = set()

    # Original
    variants.add(name)

    # Nickname → formal
    if first in NICKNAMES:
        formal_first = NICKNAMES[first].capitalize()
        variants.add(f"{formal_first} {rest}")

    # Formal → nickname
    if first in NICKNAMES_REVERSE:
        nick_first = NICKNAMES_REVERSE[first].capitalize()
        variants.add(f"{nick_first} {rest}")

    # Also try normalized versions of all variants
    result = list(variants)
    return result


def match_player(dfo_name, position_group, team, primary, secondary, tertiary):
    """Try multiple strategies to match a DFO name to a player_id."""

    def pick_best(entries):
        """If multiple entries, pick by position then team."""
        if len(entries) == 1:
            return entries[0]["player_id"]
        # Try position match
        pos_matches = [e for e in entries
                       if (position_group == "D" and e["position"] == "D") or
                          (position_group == "F" and e["position"] != "D")]
        if len(pos_matches) == 1:
            return pos_matches[0]["player_id"]
        # Try team match
        team_matches = [e for e in entries if e["team"] == team]
        if len(team_matches) == 1:
            return team_matches[0]["player_id"]
        # Fall back to first
        return entries[0]["player_id"]

    # Generate all name variants to try
    variants = expand_name_variants(dfo_name)

    # Also add middle-name-stripped variant
    parts = dfo_name.split()
    if len(parts) > 2:
        short = f"{parts[0]} {parts[-1]}"
        variants.append(short)
        variants.extend(expand_name_variants(short))

    # Strategy 1: exact match on any variant
    for variant in variants:
        if variant in primary:
            return pick_best(primary[variant])

    # Strategy 2: normalized match on any variant
    for variant in variants:
        norm = normalize_name(variant)
        if norm in secondary:
            return pick_best(secondary[norm])

    # Strategy 3: last name + team
    last = normalize_name(parts[-1])
    if last in tertiary:
        team_matches = [e for e in tertiary[last] if e["team"] == team]
        if len(team_matches) == 1:
            return team_matches[0]["player_id"]
        if team_matches:
            return pick_best(team_matches)

    return None


# ── Feature preparation ───────────────────────────────────────────────────────
def get_player_features(player_id, features_df, player_info):
    """Get most recent feature row for a player, add lineup context."""
    rows = features_df[features_df["player_id"] == player_id].sort_values("date")
    if len(rows) == 0:
        return None

    row = rows.iloc[-1].copy()

    # Override with today's lineup context
    row["is_home"]    = int(player_info["is_home"])
    row["is_defense"] = int(player_info["position_group"] == "D")
    row["is_forward"] = int(player_info["position_group"] == "F")
    row["is_center"]  = int(row.get("position", "") == "C")

    # PP unit as numeric feature
    row["pp_unit"]    = player_info["pp_unit"]
    row["line_num"]   = player_info["line_num"]

    return row


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_player(feature_row, models, feature_lists, model_types, calibrators):
    """Run all models for one player, return raw predictions."""
    preds = {}

    for target, model in models.items():
        feats = feature_lists[target]
        # Build input row — fill missing features with 0
        x = np.array([feature_row.get(f, 0) or 0 for f in feats]).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0)
        import pandas as pd
        X = pd.DataFrame(x, columns=feats)

        mtype = model_types[target]
        if mtype == "classifier":
            raw_prob = model.predict_proba(X)[0, 1]
            if target in calibrators:
                prob = float(calibrators[target].transform([raw_prob])[0])
            else:
                prob = raw_prob
            preds[target] = prob
        else:
            preds[target] = float(model.predict(X)[0])

    return preds


def compute_goals(preds, feature_row):
    """
    Compute expected goals using formula:
      ev_goals = ev_shots × regressed_ev_shooting_pct × regressed_finishing_talent
      pp_goals = pp_shots × career_pp_shooting_pct (fallback to league mean)
    """
    ev_shots    = max(preds.get("ev_shots", 0), 0)
    pp_shots    = max(preds.get("pp_shots", 0), 0)
    sh_shots    = ev_shots * 0.03  # approximate SH shots

    ev_sh_pct   = float(feature_row.get("regressed_ev_shooting_pct", 0.098) or 0.098)
    finishing   = float(feature_row.get("regressed_finishing_talent", 1.097) or 1.097)

    # Career PP shooting% — compute from career stats if available
    career_pp_goals  = float(feature_row.get("career_ev_goals_sum", 0) or 0)  # approximation
    pp_sh_pct        = 0.128  # league average PP shooting%

    ev_goals_lambda = ev_shots * ev_sh_pct * finishing
    pp_goals_lambda = pp_shots * pp_sh_pct
    total_goals_lambda = ev_goals_lambda + pp_goals_lambda

    return {
        "ev_goals_lambda": ev_goals_lambda,
        "pp_goals_lambda": pp_goals_lambda,
        "total_goals_lambda": total_goals_lambda,
        "ev_shots":  ev_shots,
        "pp_shots":  pp_shots,
        "sh_shots":  sh_shots,
        "total_shots_lambda": ev_shots + pp_shots + sh_shots,
    }


def compute_assists(preds):
    ev_assists = max(preds.get("ev_assists", 0), 0)
    pp_assists = max(preds.get("pp_assists", 0), 0)
    return ev_assists + pp_assists


def poisson_over_prob(lam, line):
    """P(X > line) where X ~ Poisson(lam), line is typically x.5."""
    k = int(line + 0.5)  # e.g. line=0.5 → k=1, line=1.5 → k=2
    return 1 - poisson.cdf(k - 1, lam)


def format_prob(p):
    """Convert probability to American odds string."""
    p = max(0.001, min(0.999, p))
    if p >= 0.5:
        odds = -round((p / (1 - p)) * 100)
    else:
        odds = round(((1 - p) / p) * 100)
    return f"{p:.1%} ({odds:+d})"


# ── Main output ───────────────────────────────────────────────────────────────
def print_player_props(player_name, role, computed, preds, lines_config=None):
    shots_lam  = computed["total_shots_lambda"]
    goals_lam  = computed["total_goals_lambda"]
    assists_lam = preds.get("ev_assists", 0) + preds.get("pp_assists", 0)
    points_lam  = goals_lam + assists_lam
    p_goal     = preds.get("scored_ev_goal", goals_lam / max(goals_lam + 0.01, 1))

    print(f"\n  {player_name} ({role})")
    print(f"    TOI:     EV {preds.get('toi_ev',0):.1f}  PP {preds.get('toi_pp',0):.1f}")
    print(f"    Shots:   EV {computed['ev_shots']:.1f}  PP {computed['pp_shots']:.1f}  Total {shots_lam:.1f}")
    print(f"    Goals:   λ={goals_lam:.3f}  P(anytime scorer)={format_prob(p_goal)}")
    print(f"    Assists: {assists_lam:.2f}")
    print(f"    Points:  {points_lam:.2f}")

    # Shot lines
    print(f"    Shots o/u:")
    for line in SHOT_LINES:
        p = poisson_over_prob(shots_lam, line)
        if 0.15 < p < 0.85:
            print(f"      o{line}: {format_prob(p)}")

    # Point lines
    print(f"    Points o/u:")
    for line in POINT_LINES:
        p = poisson_over_prob(points_lam, line)
        if 0.10 < p < 0.90:
            print(f"      o{line}: {format_prob(p)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("home_team", type=str)
    parser.add_argument("away_team", type=str)
    parser.add_argument("--date", type=str, default=None,
                        help="Date string YYYY-MM-DD (default: latest lineup file)")
    parser.add_argument("--min-shots", type=float, default=1.0,
                        help="Minimum projected shots to include player (default: 1.0)")
    args = parser.parse_args()

    home_team = args.home_team.upper()
    away_team = args.away_team.upper()
    date_str  = args.date.replace("-","") if args.date else None

    print(f"\n{'='*60}")
    print(f"PLAYER PROPS: {away_team} @ {home_team}")
    print(f"{'='*60}")

    # Load everything
    print("\nLoading models...")
    models, feature_lists, model_types, calibrators, residual_stds = load_models()
    print(f"  Loaded {len(models)} models")

    print("Loading player features...")
    features_df = pd.read_csv(PLAYER_FEATURES, low_memory=False)
    features_df["date"] = pd.to_datetime(features_df["date"])
    # Encode position
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

        team_shots   = 0.0
        team_goals   = 0.0
        matched      = 0
        unmatched_names = []
        player_rows  = []

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

            matched += 1
            preds    = predict_player(feat_row, models, feature_lists,
                                      model_types, calibrators)
            computed = compute_goals(preds, feat_row)
            assists  = compute_assists(preds)

            # Build role label
            pos_label = p["position_group"]
            line_label = f"L{p['line_num']}"
            pp_label   = f" PP{p['pp_unit']}" if p["pp_unit"] > 0 else ""
            role = f"{pos_label} {line_label}{pp_label}"

            team_shots += computed["total_shots_lambda"]
            team_goals += computed["total_goals_lambda"]

            player_rows.append({
                "name":    p["name"],
                "role":    role,
                "toi_ev":  preds.get("toi_ev", 0),
                "toi_pp":  preds.get("toi_pp", 0),
                "shots":   computed["total_shots_lambda"],
                "ev_shots":computed["ev_shots"],
                "pp_shots":computed["pp_shots"],
                "goals":   computed["total_goals_lambda"],
                "assists": assists,
                "points":  computed["total_goals_lambda"] + assists,
                "p_goal":  preds.get("scored_ev_goal", 0),
                "preds":   preds,
                "computed":computed,
            })

        # Sort by projected shots descending
        player_rows.sort(key=lambda x: x["shots"], reverse=True)

        # Print table
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
            "players": player_rows,
            "team_shots": team_shots,
            "team_goals": team_goals,
            "is_home": is_home,
        }

    # Game summary
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

        # Top shot props
        print(f"\n  TOP SHOT PROPS (o2.5):")
        all_players = []
        for res in team_results.values():
            all_players.extend(res["players"])
        all_players.sort(key=lambda x: x["shots"], reverse=True)
        for r in all_players[:10]:
            p_over = poisson_over_prob(r["shots"], 2.5)
            print(f"    {r['name']:<22} {r['shots']:.1f} shots  "
                  f"o2.5: {format_prob(p_over)}")


if __name__ == "__main__":
    main()