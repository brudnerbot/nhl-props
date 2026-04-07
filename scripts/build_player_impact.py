"""
build_player_impact.py

Builds per-player impact scores for lineup adjustment in team predictions.

Method: Blend of
  - 60% with/without team GF/GA/win rate (when player plays vs sits)
  - 40% individual xG above position average

Both signals regressed toward 0 based on sample size.
With/without: full trust at 15 missed games
xG: full trust at 2 seasons (164 games)

Future improvement: on-ice xG from shift data will improve defensive signal.

Output: data/processed/player_impact.json

Usage:
    python scripts/build_player_impact.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PLAYER_FEATURES = DATA_DIR / "processed/player_features.csv"
TEAM_LOGS       = DATA_DIR / "raw/team_game_logs/team_game_logs.csv"
OUT_FILE        = DATA_DIR / "processed/player_impact.json"

SEASONS   = [20202021,20212022,20222023,20232024,20242025,20252026]
SEASON_W  = {20252026:0.45,20242025:0.30,20232024:0.15,
             20222023:0.07,20212022:0.02,20202021:0.01}


def blend(row, col_s, col_l20, w=0.5):
    s   = row.get(col_s,   np.nan)
    l20 = row.get(col_l20, np.nan)
    if pd.isna(s) and pd.isna(l20): return np.nan
    if pd.isna(s):   return float(l20)
    if pd.isna(l20): return float(s)
    return (1-w)*float(s) + w*float(l20)


def wavg(grp, col, w_col):
    valid = grp.dropna(subset=[col])
    if len(valid) == 0: return np.nan
    weights = valid[w_col] * valid["season_w"]
    if weights.sum() == 0: return np.nan
    return float(np.average(valid[col], weights=weights))


def compute_with_without(pf_r, tf_r):
    print("Computing with/without team outcomes per player-season...")
    player_season_games = (
        pf_r.groupby(["player_id","team","season"])["game_id"]
        .apply(set).reset_index()
    )
    player_season_games.columns = ["player_id","team","season","played_games"]

    team_outcomes = tf_r[["game_id","team","season",
                           "goals_for","goals_against","won"]].copy()

    rows = []
    for _, row in player_season_games.iterrows():
        pid    = row["player_id"]
        team   = row["team"]
        season = row["season"]
        played = row["played_games"]
        sw     = SEASON_W.get(season, 0.01)

        tg = team_outcomes[(team_outcomes["team"]==team) &
                            (team_outcomes["season"]==season)]
        if len(tg) < 10: continue

        all_games = set(tg["game_id"])
        missed    = all_games - played
        n_played  = len(played & all_games)
        n_missed  = len(missed)
        n_team    = len(all_games)

        # Require player played at least 50% of team games
        # Filters mid-season trades and injury-shortened stints
        if n_played < max(5, n_team * 0.50): continue

        with_p    = tg[tg["game_id"].isin(played)]
        without_p = tg[tg["game_id"].isin(missed)]

        gf_with  = float(with_p["goals_for"].mean())
        ga_with  = float(with_p["goals_against"].mean())
        won_with = float(with_p["won"].mean())

        gf_without  = float(without_p["goals_for"].mean())    if n_missed >= 2 else np.nan
        ga_without  = float(without_p["goals_against"].mean()) if n_missed >= 2 else np.nan
        won_without = float(without_p["won"].mean())           if n_missed >= 2 else np.nan

        rows.append({
            "player_id":   int(pid),
            "team":        team,
            "season":      int(season),
            "season_w":    sw,
            "n_played":    n_played,
            "n_missed":    n_missed,
            "gf_impact":   gf_with  - gf_without  if not pd.isna(gf_without)  else np.nan,
            "ga_impact":   ga_with  - ga_without  if not pd.isna(ga_without)  else np.nan,
            "win_impact":  won_with - won_without if not pd.isna(won_without) else np.nan,
            "ww_trust":    min(n_missed / 15.0, 1.0),
        })

    ww = pd.DataFrame(rows)
    print(f"  {len(ww):,} player-season rows computed")
    return ww


def compute_xg_contrib(pf_r):
    print("Computing individual xG contribution above position average...")
    latest = pf_r.sort_values(["player_id","date"]).groupby("player_id").last().reset_index()
    latest["is_defense"] = (latest["position"]=="D").astype(int)

    fwd  = pf_r[pf_r["position"]!="D"]
    defs = pf_r[pf_r["position"]=="D"]

    league = {
        "ev_ixg_per60_fwd": float(fwd["ev_ixg_per60_season_avg"].dropna().mean()),
        "ev_ixg_per60_def": float(defs["ev_ixg_per60_season_avg"].dropna().mean()),
        "pp_ixg_per60_fwd": float(fwd["pp_ixg_per60_season_avg"].dropna().mean()),
        "pp_ixg_per60_def": float(defs["pp_ixg_per60_season_avg"].dropna().mean()),
    }
    print(f"  League EV ixG/60: fwd={league['ev_ixg_per60_fwd']:.3f} def={league['ev_ixg_per60_def']:.3f}")

    latest["b_ev_ixg_per60"] = latest.apply(
        lambda r: blend(r,"ev_ixg_per60_season_avg","ev_ixg_per60_last20"), axis=1)
    latest["b_pp_ixg_per60"] = latest.apply(
        lambda r: blend(r,"pp_ixg_per60_season_avg","pp_ixg_per60_last20"), axis=1)
    latest["b_toi_ev"] = latest.apply(
        lambda r: blend(r,"toi_ev_season_avg","toi_ev_last20"), axis=1)
    latest["b_toi_pp"] = latest.apply(
        lambda r: blend(r,"toi_pp_season_avg","toi_pp_last20"), axis=1)

    latest["league_ev_ixg"] = np.where(latest["is_defense"]==1,
        league["ev_ixg_per60_def"], league["ev_ixg_per60_fwd"])
    latest["league_pp_ixg"] = np.where(latest["is_defense"]==1,
        league["pp_ixg_per60_def"], league["pp_ixg_per60_fwd"])

    latest["xg_gf_contrib"] = (
        (latest["b_ev_ixg_per60"] - latest["league_ev_ixg"]) * latest["b_toi_ev"] / 60 +
        (latest["b_pp_ixg_per60"] - latest["league_pp_ixg"]) * latest["b_toi_pp"] / 60
    )

    return latest, league


def main():
    print("="*60)
    print("BUILDING PLAYER IMPACT SCORES")
    print("="*60)

    print("\nLoading data...")
    pf = pd.read_csv(PLAYER_FEATURES, low_memory=False)
    pf["full_name"] = pf["first_name"] + " " + pf["last_name"]
    pf["date"] = pd.to_datetime(pf["date"])
    tf = pd.read_csv(TEAM_LOGS, low_memory=False)
    tf["date"] = pd.to_datetime(tf["date"])

    pf_r = pf[pf["season"].isin(SEASONS)].copy()
    tf_r = tf[tf["season"].isin(SEASONS)].copy()
    pf_r["season_w"] = pf_r["season"].map(SEASON_W).fillna(0.01)

    league_team = {
        "team_avg_gf":  float(tf_r["goals_for"].mean()),
        "team_avg_ga":  float(tf_r["goals_against"].mean()),
        "team_avg_won": float(tf_r["won"].mean()),
    }

    # ── With/without ──────────────────────────────────────────────────────────
    ww = compute_with_without(pf_r, tf_r)

    agg = ww.groupby("player_id").apply(lambda g: pd.Series({
        "gf_impact_ww":  wavg(g, "gf_impact",  "ww_trust"),
        "ga_impact_ww":  wavg(g, "ga_impact",  "ww_trust"),
        "win_impact_ww": wavg(g, "win_impact", "ww_trust"),
        "total_missed":  int(g["n_missed"].fillna(0).sum()),
        "total_played":  int(g["n_played"].fillna(0).sum()),
        "max_ww_trust":  float(g["ww_trust"].fillna(0).max()),
    }), include_groups=False).reset_index()

    # ── xG contribution ───────────────────────────────────────────────────────
    latest, league_xg = compute_xg_contrib(pf_r)

    # ── Merge and blend ───────────────────────────────────────────────────────
    final = latest.merge(agg, on="player_id", how="left")
    final["career_games"]   = final["career_games"].fillna(0)
    final["xg_trust"]       = (final["career_games"] / 164.0).clip(0, 1)
    final["ww_trust_final"] = final["max_ww_trust"].fillna(0)

    final["gf_impact_xg"]     = final["xg_gf_contrib"].fillna(0) * final["xg_trust"]
    final["gf_impact_ww_reg"] = final["gf_impact_ww"].fillna(0)  * final["ww_trust_final"]

    # Blend: when ww_trust is high use 60% ww + 40% xG
    # When ww_trust is 0 fall back entirely to xG
    final["gf_impact_blend"] = (
        final["ww_trust_final"] * (
            0.60 * final["gf_impact_ww_reg"] +
            0.40 * final["gf_impact_xg"]
        ) +
        (1 - final["ww_trust_final"]) * final["gf_impact_xg"]
    )
    final["win_impact_reg"] = final["win_impact_ww"].fillna(0) * final["ww_trust_final"]
    final["ga_impact_reg"]  = final["ga_impact_ww"].fillna(0)  * final["ww_trust_final"]

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "league_avg": {**league_xg, **league_team},
        "method": "60pct with/without team GF/win + 40pct individual xG above position avg",
        "future_improvement": "on-ice xG from shift data will improve defensive signal",
        "players": {}
    }

    for _, r in final.iterrows():
        pid = int(r["player_id"])
        out["players"][pid] = {
            "full_name":    r["full_name"],
            "team":         str(r.get("team","") or ""),
            "position":     str(r.get("position","") or ""),
            "gf_impact":    round(float(r["gf_impact_blend"]),  5),
            "ga_impact":    round(float(r["ga_impact_reg"]),     5),
            "win_impact":   round(float(r["win_impact_reg"]),    5),
            "xg_contrib":   round(float(r["gf_impact_xg"]),      5),
            "total_missed": int(r["total_missed"]) if pd.notna(r.get("total_missed")) else 0,
            "career_games": int(r["career_games"]) if pd.notna(r["career_games"]) else 0,
            "ww_trust":     round(float(r["ww_trust_final"]),   3),
            "xg_trust":     round(float(r["xg_trust"]),         3),
        }

    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_FILE.name} ({len(out['players']):,} players)")

    # ── Verify key players ────────────────────────────────────────────────────
    check = ["Connor McDavid","Leon Draisaitl","Auston Matthews",
             "Cutter Gauthier","Nathan MacKinnon","Zach Hyman",
             "Nikita Kucherov","William Nylander","Jack Eichel",
             "Evan Bouchard","Cale Makar","Roman Josi"]
    print(f"\n{'Player':<22} {'GF_impact':>10} {'Win_impact':>11} "
          f"{'WW_trust':>9} {'xG_trust':>9} {'Missed':>7}")
    print("-"*72)
    for pid, d in out["players"].items():
        if d["full_name"] in check:
            print(f"  {d['full_name']:<20} {d['gf_impact']:>+10.4f} "
                  f"{d['win_impact']:>+11.4f} {d['ww_trust']:>9.3f} "
                  f"{d['xg_trust']:>9.3f} {d['total_missed']:>7}")

    print("\nDone!")


if __name__ == "__main__":
    main()
