"""
process_player_features.py

Builds per-player per-game feature dataset for player modeling.

Sources:
  - data/raw/game_logs/player_game_logs.csv       (7 seasons basic stats)
  - data/raw/player_pbp_stats/player_pbp_stats.csv (2 seasons strength splits + on-ice)
  - data/processed/shot_data_with_xg.csv           (4 seasons individual xG)
  - data/raw/team_game_logs/team_game_logs.csv      (team context)

Output: data/processed/player_features.csv

Rolling windows computed: last10, last20, last30, season_avg, career_avg
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- CONFIG ---
ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PLAYER_GL   = DATA_DIR / "raw/game_logs/player_game_logs.csv"
PLAYER_PBP  = DATA_DIR / "raw/player_pbp_stats/player_pbp_stats.csv"
SHOT_XG     = DATA_DIR / "processed/shot_data_with_xg.csv"
TEAM_GL     = DATA_DIR / "raw/team_game_logs/team_game_logs.csv"
OUT_FILE    = DATA_DIR / "processed/player_features.csv"

ROLLING_WINDOWS = [10, 20, 30]
MIN_SEASON      = 20192020  # earliest season to include


# ── Step 1: Load and merge base data ─────────────────────────────────────────
def load_base(gl_path, pbp_path):
    print("Loading player game logs...")
    gl = pd.read_csv(gl_path, low_memory=False)
    gl["date"] = pd.to_datetime(gl["date"])
    gl = gl[gl["season"] >= MIN_SEASON].copy()
    gl["full_name"] = gl["first_name"] + " " + gl["last_name"]
    print(f"  Game logs: {len(gl):,} rows, {gl['season'].nunique()} seasons")

    print("Loading player PBP stats...")
    pbp = pd.read_csv(pbp_path, low_memory=False)
    pbp["date"] = pd.to_datetime(pbp["date"])
    print(f"  PBP stats: {len(pbp):,} rows, {pbp['season'].nunique()} seasons")

    # Merge PBP onto game logs (left join — keep all gl rows)
    pbp_cols = [
        "game_id", "player_id",
        "toi_ev", "toi_pp", "toi_sh", "toi_en",
        "ev_shots", "pp_shots", "sh_shots",
        "ev_goals", "pp_goals", "sh_goals",
        "ev_assists", "pp_assists", "sh_assists",
        "ev_onice_sf", "ev_onice_sa",
        "pp_onice_sf", "pp_onice_sa",
        "sh_onice_sf", "sh_onice_sa",
        "ev_onice_gf", "ev_onice_ga",
        "pp_onice_gf", "pp_onice_ga",
        "sh_onice_gf", "sh_onice_ga",
        "hits", "blocks", "faceoffs_won", "faceoffs_taken",
        "giveaways", "takeaways",
    ]
    df = gl.merge(pbp[pbp_cols], on=["game_id","player_id"], how="left")
    print(f"  After merge: {len(df):,} rows")
    print(f"  Rows with PBP data: {df['toi_ev'].notna().sum():,}")

    # For rows without PBP data, fill TOI splits from total TOI
    mask = df["toi_ev"].isna()

    # Rename gl pp/sh goal columns to avoid confusion (they came from game_logs)
    # player_game_logs has: pp_goals, sh_goals, pp_points, sh_points
    gl_pp_goals  = df["pp_goals"].fillna(0)  if "pp_goals"  in df.columns else pd.Series(0, index=df.index)
    gl_sh_goals  = df["sh_goals"].fillna(0)  if "sh_goals"  in df.columns else pd.Series(0, index=df.index)
    gl_pp_points = df["pp_points"].fillna(0) if "pp_points" in df.columns else pd.Series(0, index=df.index)
    gl_sh_points = df["sh_points"].fillna(0) if "sh_points" in df.columns else pd.Series(0, index=df.index)

    df.loc[mask, "toi_ev"]    = df.loc[mask, "toi"] * 0.85
    df.loc[mask, "toi_pp"]    = df.loc[mask, "toi"] * 0.10
    df.loc[mask, "toi_sh"]    = df.loc[mask, "toi"] * 0.05
    df.loc[mask, "toi_en"]    = 0.0
    df.loc[mask, "ev_shots"]  = df.loc[mask, "shots"] * 0.80
    df.loc[mask, "pp_shots"]  = df.loc[mask, "shots"] * 0.15
    df.loc[mask, "sh_shots"]  = df.loc[mask, "shots"] * 0.05
    df.loc[mask, "pp_goals"]  = gl_pp_goals[mask]
    df.loc[mask, "sh_goals"]  = gl_sh_goals[mask]
    df.loc[mask, "ev_goals"]  = (df.loc[mask, "goals"] - gl_pp_goals[mask] - gl_sh_goals[mask]).clip(lower=0)
    df.loc[mask, "pp_assists"] = (gl_pp_points[mask] - gl_pp_goals[mask]).clip(lower=0)
    df.loc[mask, "sh_assists"] = (gl_sh_points[mask] - gl_sh_goals[mask]).clip(lower=0)
    df.loc[mask, "ev_assists"] = (df.loc[mask, "assists"] - df.loc[mask, "pp_assists"] - df.loc[mask, "sh_assists"]).clip(lower=0)

    # Fill remaining nulls
    onice_cols = [c for c in df.columns if "onice" in c]
    misc_cols  = ["hits","blocks","faceoffs_won","faceoffs_taken","giveaways","takeaways"]
    for c in onice_cols + misc_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df


# ── Step 2: Join individual xG ────────────────────────────────────────────────
def add_individual_xg(df, shot_path):
    print("Computing individual xG per player per game...")
    shots = pd.read_csv(shot_path, low_memory=False)
    shots = shots[
        shots["event_type"].isin(["shot-on-goal","goal"]) &
        (shots["is_empty_net"] == 0)
    ].copy()

    # Individual xG total
    ixg_total = shots.groupby(["game_id","shooter_id"]).agg(
        ixg=("xg","sum")
    ).reset_index().rename(columns={"shooter_id":"player_id"})

    # Individual xG by strength
    ixg_str = shots.groupby(["game_id","shooter_id","strength"]).agg(
        xg_sum=("xg","sum")
    ).reset_index().rename(columns={"shooter_id":"player_id"})
    ixg_str = ixg_str.pivot_table(
        index=["game_id","player_id"], columns="strength",
        values="xg_sum", fill_value=0
    ).reset_index()
    ixg_str.columns.name = None
    ixg_str = ixg_str.rename(columns={
        "ev": "ev_ixg", "pp": "pp_ixg", "sh": "sh_ixg"
    })
    for c in ["ev_ixg","pp_ixg","sh_ixg"]:
        if c not in ixg_str.columns:
            ixg_str[c] = 0.0

    # Merge
    df = df.merge(ixg_total, on=["game_id","player_id"], how="left")
    df = df.merge(ixg_str[["game_id","player_id","ev_ixg","pp_ixg","sh_ixg"]],
                  on=["game_id","player_id"], how="left")

    df["ixg"]    = df["ixg"].fillna(0.0)
    df["ev_ixg"] = df["ev_ixg"].fillna(0.0)
    df["pp_ixg"] = df["pp_ixg"].fillna(0.0)
    df["sh_ixg"] = df["sh_ixg"].fillna(0.0)

    print(f"  Rows with xG data: {(df['ixg']>0).sum():,}")
    return df


# ── Step 3: Add team context ──────────────────────────────────────────────────
def add_team_context(df, team_gl_path):
    print("Adding team context...")
    tgl = pd.read_csv(team_gl_path, low_memory=False)
    tgl["date"] = pd.to_datetime(tgl["date"])

    # We need: team EV TOI, PP TOI, SH TOI per game (for rate normalization)
    team_toi = tgl[["game_id","team","ev_toi","pp_toi","sh_toi",
                    "ev_shots_on_goal_for","pp_shots_on_goal_for"]].copy()
    team_toi.columns = ["game_id","team","team_ev_toi","team_pp_toi",
                        "team_sh_toi","team_ev_sog","team_pp_sog"]

    df = df.merge(team_toi, on=["game_id","team"], how="left")
    print(f"  Team context added: {df['team_ev_toi'].notna().sum():,} rows")
    return df


# ── Step 4: Derived per-game rates ────────────────────────────────────────────
def add_game_rates(df):
    print("Computing per-game rates...")
    eps = 1e-6

    # Shooting percentage (individual)
    df["ev_shooting_pct"] = df["ev_goals"] / (df["ev_shots"] + eps)
    df["pp_shooting_pct"] = df["pp_goals"] / (df["pp_shots"] + eps)

    # TOI share of team
    df["ev_toi_share"] = df["toi_ev"] / (df["team_ev_toi"].fillna(60) + eps)
    df["pp_toi_share"] = df["toi_pp"] / (df["team_pp_toi"].fillna(5)  + eps)

    # On-ice rates per 60 (EV)
    df["ev_onice_sf_per60"] = df["ev_onice_sf"] / (df["toi_ev"] / 60 + eps)
    df["ev_onice_sa_per60"] = df["ev_onice_sa"] / (df["toi_ev"] / 60 + eps)
    df["ev_onice_gf_per60"] = df["ev_onice_gf"] / (df["toi_ev"] / 60 + eps)
    df["ev_onice_ga_per60"] = df["ev_onice_ga"] / (df["toi_ev"] / 60 + eps)

    # Individual rates per 60
    df["ev_shots_per60"] = df["ev_shots"] / (df["toi_ev"] / 60 + eps)
    df["pp_shots_per60"] = df["pp_shots"] / (df["toi_pp"] / 60 + eps)
    df["ev_ixg_per60"]   = df["ev_ixg"]   / (df["toi_ev"] / 60 + eps)
    df["pp_ixg_per60"]   = df["pp_ixg"]   / (df["toi_pp"] / 60 + eps)

    # IPP: individual points as share of on-ice goals
    df["ipp"] = (df["ev_goals"] + df["ev_assists"]) / (df["ev_onice_gf"] + eps)
    df["ipp"] = df["ipp"].clip(0, 1)

    # CF% proxy (Corsi For %)
    df["ev_cf_pct"] = df["ev_onice_sf"] / (df["ev_onice_sf"] + df["ev_onice_sa"] + eps)

    # Cap extreme per-60 rates (short TOI games inflate rates)
    for col in ["ev_shots_per60","pp_shots_per60","ev_ixg_per60","pp_ixg_per60",
                "ev_onice_sf_per60","ev_onice_sa_per60"]:
        df[col] = df[col].clip(0, 200)

    return df


# ── Step 5: Rolling features ──────────────────────────────────────────────────
ROLL_STATS = [
    # TOI
    "toi", "toi_ev", "toi_pp", "toi_sh",
    # Individual counts
    "goals", "assists", "points", "shots",
    "ev_goals", "pp_goals", "ev_assists", "pp_assists",
    "ev_shots", "pp_shots",
    "ixg", "ev_ixg", "pp_ixg",
    # Per-60 rates
    "ev_shots_per60", "pp_shots_per60",
    "ev_ixg_per60", "pp_ixg_per60",
    # On-ice
    "ev_onice_sf_per60", "ev_onice_sa_per60",
    "ev_onice_gf_per60", "ev_onice_ga_per60",
    "ev_cf_pct",
    # Misc
    "ipp", "ev_toi_share", "pp_toi_share",
    "hits", "blocks", "faceoffs_won", "faceoffs_taken",
]


def add_rolling_features(df):
    print("Computing rolling features...")
    df = df.sort_values(["player_id","date"]).reset_index(drop=True)

    all_rows = []
    n_players = df["player_id"].nunique()

    for i, (player_id, pdf) in enumerate(df.groupby("player_id")):
        if i % 200 == 0:
            print(f"  {i}/{n_players} players...")

        pdf = pdf.sort_values("date").reset_index(drop=True)

        for idx in range(len(pdf)):
            row  = pdf.iloc[idx].copy()
            past = pdf.iloc[:idx]

            # Rolling windows
            for w in ROLLING_WINDOWS:
                past_w = past.tail(w)
                for stat in ROLL_STATS:
                    if stat in pdf.columns:
                        val = past_w[stat].mean() if len(past_w) > 0 else np.nan
                        row[f"{stat}_last{w}"] = val

            # Season average (current season, excluding current game)
            current_season = row["season"]
            season_past = past[past["season"] == current_season]
            for stat in ROLL_STATS:
                if stat in pdf.columns:
                    val = season_past[stat].mean() if len(season_past) > 0 else np.nan
                    row[f"{stat}_season_avg"] = val

            # Career cumulative shooting% (for regression to mean)
            # Use all past games regardless of season
            if len(past) > 0:
                career_shots = past["shots"].sum()
                career_goals = past["goals"].sum()
                row["career_shooting_pct"] = (
                    career_goals / career_shots if career_shots > 0 else np.nan
                )
                row["career_ev_shots"] = career_shots
                row["career_games"]    = len(past)

                # Career EV shooting%
                ev_shots_career = past["ev_shots"].sum()
                ev_goals_career = past["ev_goals"].sum()
                row["career_ev_shooting_pct"] = (
                    ev_goals_career / ev_shots_career
                    if ev_shots_career > 0 else np.nan
                )

                # Regressed shooting% (career + league mean blend)
                # Use 100-shot regression: weighted toward career as shots increase
                LEAGUE_SH_PCT = 0.098  # from our xG calibration
                REGRESSION_N  = 100    # shots to regress toward mean
                w_career = career_shots / (career_shots + REGRESSION_N)
                w_league = 1 - w_career
                if pd.notna(row.get("career_ev_shooting_pct")):
                    row["regressed_ev_shooting_pct"] = (
                        w_career * row["career_ev_shooting_pct"] +
                        w_league * LEAGUE_SH_PCT
                    )
                else:
                    row["regressed_ev_shooting_pct"] = LEAGUE_SH_PCT

            else:
                row["career_shooting_pct"]     = np.nan
                row["career_ev_shooting_pct"]  = np.nan
                row["regressed_ev_shooting_pct"] = 0.098
                row["career_ev_shots"]         = 0
                row["career_games"]            = 0

            all_rows.append(row)

    result = pd.DataFrame(all_rows)
    print(f"  Rolling features computed: {len(result):,} rows")
    return result


# ── Step 6: Cumulative per-60 rates (correct method) ─────────────────────────
def add_cumulative_rates(df):
    """
    Cumulative per-60 = sum(stat) / sum(toi/60) for current season.
    More stable than averaging per-game rates.
    """
    print("Computing cumulative per-60 rates...")
    df = df.sort_values(["player_id","date"]).reset_index(drop=True)

    CUM_PAIRS = [
        ("ev_shots",  "toi_ev",  "ev_shots_per60_cumulative"),
        ("pp_shots",  "toi_pp",  "pp_shots_per60_cumulative"),
        ("ev_goals",  "toi_ev",  "ev_goals_per60_cumulative"),
        ("pp_goals",  "toi_pp",  "pp_goals_per60_cumulative"),
        ("ev_ixg",    "toi_ev",  "ev_ixg_per60_cumulative"),
        ("pp_ixg",    "toi_pp",  "pp_ixg_per60_cumulative"),
        ("ev_onice_sf","toi_ev", "ev_onice_sf_per60_cumulative"),
        ("ev_onice_sa","toi_ev", "ev_onice_sa_per60_cumulative"),
        ("ev_onice_gf","toi_ev", "ev_onice_gf_per60_cumulative"),
        ("ev_onice_ga","toi_ev", "ev_onice_ga_per60_cumulative"),
        ("hits",      "toi",     "hits_per60_cumulative"),
        ("blocks",    "toi",     "blocks_per60_cumulative"),
    ]

    for stat, toi_col, out_col in CUM_PAIRS:
        if stat not in df.columns or toi_col not in df.columns:
            continue

        vals = []
        for player_id, pdf in df.groupby("player_id"):
            pdf = pdf.sort_values("date")
            for idx in range(len(pdf)):
                row          = pdf.iloc[idx]
                season_past  = pdf.iloc[:idx][pdf.iloc[:idx]["season"] == row["season"]]
                sum_stat     = season_past[stat].sum()
                sum_toi_hrs  = season_past[toi_col].sum() / 60
                rate = sum_stat / sum_toi_hrs if sum_toi_hrs > 0 else np.nan
                vals.append((pdf.index[idx], rate))

        idx_arr, rate_arr = zip(*vals) if vals else ([], [])
        df.loc[list(idx_arr), out_col] = list(rate_arr)

    print(f"  Cumulative rates added")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("BUILDING PLAYER FEATURES")
    print("="*60)

    # Step 1: Load and merge
    df = load_base(PLAYER_GL, PLAYER_PBP)

    # Step 2: Individual xG
    df = add_individual_xg(df, SHOT_XG)

    # Step 3: Team context
    df = add_team_context(df, TEAM_GL)

    # Step 4: Game rates
    df = add_game_rates(df)

    # Ensure points column exists
    if "points" not in df.columns:
        df["points"] = df["goals"] + df["assists"]

    print(f"\nBase dataset: {len(df):,} rows, {df.shape[1]} columns")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Players: {df['player_id'].nunique():,}")

    # Step 5: Rolling features (slow — iterates per player)
    df = add_rolling_features(df)

    # Step 6: Cumulative per-60 rates
    df = add_cumulative_rates(df)

    # Final cleanup
    df = df.sort_values(["date","player_id"]).reset_index(drop=True)

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Saved:   {OUT_FILE}")
    print(f"\nColumn groups:")
    base_cols    = [c for c in df.columns if not any(s in c for s in ["last10","last20","last30","season_avg","career","cumulative"])]
    rolling_cols = [c for c in df.columns if any(s in c for s in ["last10","last20","last30","season_avg"])]
    career_cols  = [c for c in df.columns if "career" in c]
    cumul_cols   = [c for c in df.columns if "cumulative" in c]
    print(f"  Base game stats:  {len(base_cols)}")
    print(f"  Rolling features: {len(rolling_cols)}")
    print(f"  Career features:  {len(career_cols)}")
    print(f"  Cumulative rates: {len(cumul_cols)}")

    # Sample output
    sample = df[df["player_id"] == df["player_id"].iloc[0]].iloc[-1]
    print(f"\nSample row ({sample.get('full_name','?')}):")
    key_cols = ["season","date","team","toi_ev","toi_pp","ev_shots","pp_shots",
                "ev_goals","ixg","ev_onice_sf","ev_onice_gf",
                "ev_shots_per60_last20","pp_shots_per60_last20",
                "regressed_ev_shooting_pct","career_games"]
    for c in key_cols:
        if c in sample.index:
            print(f"  {c}: {sample[c]}")


if __name__ == "__main__":
    main()