"""
build_zone_features.py

Builds shot zone features for players and teams.
Must be run after build_xg_model.py (requires shot_data_with_xg.csv).

Zones (normalized coords, net at x=89, y=0 center):
  net_front:   x > 74 (within 15ft of goal, full width including behind net)
  slot:        x=46.5-74, |y| < 22 (between faceoff dots, inside net front)
  left_flank:  x > 46.5, y < -22
  right_flank: x > 46.5, y > +22
  left_point:  x=25-46.5 (42.5ft from goal), y < -8.5
  mid_point:   x=25-46.5, |y| < 8.5
  right_point: x=25-46.5, y > +8.5
  out_of_ozone: x < 25

Outputs:
  data/processed/shot_data_with_xg.csv     — zone column added/updated
  data/processed/player_zone_shots.csv     — per-player per-game zone shots + rolling avgs
  data/processed/team_zone_defense.csv     — per-team shots allowed by zone (rolling)
  data/processed/team_zone_offense.csv     — per-team shots for by zone (rolling)
  data/processed/zone_averages.json        — league and position avg proportions

Usage:
    python scripts/build_zone_features.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

ZONES = ["net_front","slot","left_flank","right_flank",
         "left_point","mid_point","right_point"]


def classify_zone(x, y):
    """
    Classify a shot into a zone based on normalized coordinates.
    Net at x=89, blue line at x=25, y=0 center, y=±42.5 boards.
    Faceoff dots at x=69, y=±22. Point zone = 42.5ft from goal (x=46.5).
    """
    if x < 25:        return "out_of_ozone"
    if x <= 46.5:
        if y < -8.5:  return "left_point"
        elif y > 8.5: return "right_point"
        else:         return "mid_point"
    if x > 74:        return "net_front"
    if abs(y) < 22:   return "slot"
    if y < -22:       return "left_flank"
    return "right_flank"


def build_zone_column(shots):
    print("Classifying shot zones...")
    shots["zone"] = shots.apply(
        lambda r: classify_zone(r["x_coord_norm"], r["y_coord_norm"]), axis=1)
    sog = shots[shots["is_on_goal"]==1]
    print(f"\n  Zone distribution (SOG only, {len(sog):,} shots):")
    print(f"  {'Zone':<18} {'shots':>8} {'pct':>7} {'goal_rate':>10} {'avg_xg':>8}")
    for z in ZONES:
        b = sog[sog["zone"]==z]
        if len(b)==0: continue
        print(f"  {z:<18} {len(b):>8,} {len(b)/len(sog):>7.1%} "
              f"{b['is_goal'].mean():>10.3f} {b['xg'].mean():>8.3f}")
    return shots


def build_player_zone_features(shots):
    print("\nComputing per-player zone features...")
    sog = shots[shots["is_on_goal"]==1].copy()
    game_meta = sog[["game_id","season","date"]].drop_duplicates("game_id")

    player_zone = sog.groupby(["game_id","shooter_id","zone"]).size().reset_index(name="n")
    player_zone = player_zone.rename(columns={"shooter_id":"player_id"})
    player_zone["player_id"] = player_zone["player_id"].astype("Int64")

    wide = player_zone.pivot_table(
        index=["game_id","player_id"], columns="zone", values="n", fill_value=0
    ).reset_index()
    wide.columns.name = None
    for z in ZONES:
        if z not in wide.columns:
            wide[z] = 0

    wide = wide.merge(game_meta, on="game_id", how="left")
    wide = wide.sort_values(["player_id","date"])

    for z in ZONES:
        wide[f"{z}_season_avg"] = (
            wide.groupby(["player_id","season"])[z]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        wide[f"{z}_last10"] = (
            wide.groupby("player_id")[z]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        )

    out = DATA_DIR / "processed/player_zone_shots.csv"
    wide.to_csv(out, index=False)
    print(f"  Saved {out.name}: {len(wide):,} rows")
    return wide


def build_team_zone_features(shots):
    print("\nComputing per-team zone features...")
    sog = shots[shots["is_on_goal"]==1].copy()
    game_meta = sog[["game_id","season","date"]].drop_duplicates("game_id")

    for side, team_col, prefix in [
        ("defense", "defending_team", "opp_"),
        ("offense", "shooting_team",  ""),
    ]:
        grp = sog.groupby(["game_id", team_col, "zone"]).size().reset_index(name="n")
        wide = grp.pivot_table(
            index=["game_id", team_col], columns="zone", values="n", fill_value=0
        ).reset_index()
        wide.columns.name = None
        for z in ZONES:
            if z not in wide.columns:
                wide[z] = 0
        wide = wide.merge(game_meta, on="game_id", how="left")
        wide = wide.sort_values([team_col, "date"])

        suffix = "allowed_last30" if side=="defense" else "for_last30"
        for z in ZONES:
            col = f"{prefix}{z}_{suffix}"
            wide[col] = (
                wide.groupby(team_col)[z]
                .transform(lambda x: x.shift(1).rolling(30, min_periods=5).mean())
            )

        out = DATA_DIR / f"processed/team_zone_{side}.csv"
        wide.to_csv(out, index=False)
        print(f"  Saved {out.name}: {len(wide):,} rows")


def build_zone_averages(shots, player_zone_wide):
    print("\nComputing league and position averages...")
    sog = shots[shots["is_on_goal"]==1].copy()
    recent = sog[sog["season"] >= 20222023]

    # League avg shots allowed per game per zone
    league_avg = {}
    for z in ZONES:
        league_avg[z] = float(
            recent.groupby(["game_id","defending_team"])
            .apply(lambda x: (x["zone"]==z).sum(), include_groups=False)
            .mean()
        )

    # Position averages
    pbp = pd.read_csv(
        DATA_DIR / "raw/player_pbp_stats/player_pbp_stats.csv",
        usecols=["player_id","position"], low_memory=False
    ).drop_duplicates("player_id")
    pbp["is_defense"] = (pbp["position"]=="D").astype(int)

    pz = player_zone_wide.copy()
    pz["player_id"] = pz["player_id"].astype("Int64")
    pbp["player_id"] = pbp["player_id"].astype("Int64")
    pz = pz.merge(pbp[["player_id","is_defense"]], on="player_id", how="left")
    pz_recent = pz[pz["season"] >= 20222023].dropna(subset=["is_defense"])

    pos_zone_avg = {
        "forward": {z: float(pz_recent[pz_recent["is_defense"]==0][z].mean()) for z in ZONES},
        "defense": {z: float(pz_recent[pz_recent["is_defense"]==1][z].mean()) for z in ZONES},
    }

    def proportions(d):
        total = sum(d.values()) + 1e-6
        return {k: v/total for k,v in d.items()}

    averages = {
        "league_defense_avg":  league_avg,
        "league_offense_avg":  league_avg,
        "position_zone_avg":   pos_zone_avg,
        "position_zone_props": {
            "forward": proportions(pos_zone_avg["forward"]),
            "defense": proportions(pos_zone_avg["defense"]),
        },
    }

    out = DATA_DIR / "processed/zone_averages.json"
    with open(out, "w") as f:
        json.dump(averages, f, indent=2)
    print(f"  Saved {out.name}")

    print(f"\n  League avg shots/game per zone:")
    for z,v in league_avg.items():
        print(f"    {z:<18}: {v:.3f}")
    print(f"\n  Position zone proportions:")
    print(f"    {'Zone':<18} {'Forward':>10} {'Defense':>10}")
    for z in ZONES:
        fp = averages["position_zone_props"]["forward"][z]
        dp = averages["position_zone_props"]["defense"][z]
        print(f"    {z:<18} {fp:>10.1%} {dp:>10.1%}")

    return averages


def main():
    shot_path = DATA_DIR / "processed/shot_data_with_xg.csv"
    print(f"Loading shot data...")
    shots = pd.read_csv(shot_path, low_memory=False)
    shots["date"] = pd.to_datetime(shots["date"])
    print(f"  {len(shots):,} shots loaded")

    shots = build_zone_column(shots)
    shots.to_csv(shot_path, index=False)
    print(f"\n  Saved updated shot_data_with_xg.csv")

    player_zone_wide = build_player_zone_features(shots)
    build_team_zone_features(shots)
    build_zone_averages(shots, player_zone_wide)

    print("\nDone!")


if __name__ == "__main__":
    main()
