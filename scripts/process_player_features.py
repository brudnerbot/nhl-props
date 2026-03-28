"""
process_player_features.py

Builds per-player per-game feature dataset for player modeling.

Sources:
  - data/raw/game_logs/player_game_logs.csv       (11 seasons basic stats)
  - data/raw/player_pbp_stats/player_pbp_stats.csv (2 seasons strength splits + on-ice)
  - data/processed/shot_data_with_xg.csv           (4 seasons individual xG)
  - data/raw/team_game_logs/team_game_logs.csv      (team context)

Output: data/processed/player_features.csv

Rolling windows: last5/10/20/30 — within season only (reset at season boundary)
Season avg: expanding mean within season
Career features: cross-season (no reset)
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PLAYER_GL   = DATA_DIR / "raw/game_logs/player_game_logs.csv"
PLAYER_PBP  = DATA_DIR / "raw/player_pbp_stats/player_pbp_stats.csv"
SHOT_XG     = DATA_DIR / "processed/shot_data_with_xg.csv"
TEAM_GL     = DATA_DIR / "raw/team_game_logs/team_game_logs.csv"
OUT_FILE    = DATA_DIR / "processed/player_features.csv"

ROLLING_WINDOWS = [5, 10, 20, 30]
MIN_SEASON      = 20152016

ROLL_STATS = [
    "toi", "toi_ev", "toi_pp", "toi_sh",
    "goals", "assists", "points", "shots",
    "ev_goals", "pp_goals", "ev_assists", "pp_assists",
    "ev_shots", "pp_shots",
    "ixg", "ev_ixg", "pp_ixg",
    "ev_shots_per60", "pp_shots_per60",
    "ev_ixg_per60", "pp_ixg_per60",
    "ev_goals_per60", "pp_goals_per60",
    "ev_onice_sf_per60", "ev_onice_sa_per60",
    "ev_onice_gf_per60", "ev_onice_ga_per60",
    "ev_cf_pct",
    "ipp", "ev_toi_share", "pp_toi_share",
    "hits", "blocks", "faceoffs_won", "faceoffs_taken",
]


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

    mask = df["toi_ev"].isna()
    gl_pp_goals  = df["pp_goals"].fillna(0)  if "pp_goals"  in df.columns else pd.Series(0, index=df.index)
    gl_sh_goals  = df["sh_goals"].fillna(0)  if "sh_goals"  in df.columns else pd.Series(0, index=df.index)
    gl_pp_points = df["pp_points"].fillna(0) if "pp_points" in df.columns else pd.Series(0, index=df.index)
    gl_sh_points = df["sh_points"].fillna(0) if "sh_points" in df.columns else pd.Series(0, index=df.index)

    df.loc[mask, "toi_ev"]     = df.loc[mask, "toi"] * 0.85
    df.loc[mask, "toi_pp"]     = df.loc[mask, "toi"] * 0.10
    df.loc[mask, "toi_sh"]     = df.loc[mask, "toi"] * 0.05
    df.loc[mask, "toi_en"]     = 0.0
    df.loc[mask, "ev_shots"]   = df.loc[mask, "shots"] * 0.80
    df.loc[mask, "pp_shots"]   = df.loc[mask, "shots"] * 0.15
    df.loc[mask, "sh_shots"]   = df.loc[mask, "shots"] * 0.05
    df.loc[mask, "pp_goals"]   = gl_pp_goals[mask]
    df.loc[mask, "sh_goals"]   = gl_sh_goals[mask]
    df.loc[mask, "ev_goals"]   = (
        df.loc[mask, "goals"] - gl_pp_goals[mask] - gl_sh_goals[mask]
    ).clip(lower=0)
    df.loc[mask, "pp_assists"] = (gl_pp_points[mask] - gl_pp_goals[mask]).clip(lower=0)
    df.loc[mask, "sh_assists"] = (gl_sh_points[mask] - gl_sh_goals[mask]).clip(lower=0)
    df.loc[mask, "ev_assists"] = (
        df.loc[mask, "assists"] -
        df.loc[mask, "pp_assists"] -
        df.loc[mask, "sh_assists"]
    ).clip(lower=0)

    fill_zero = [c for c in df.columns if "onice" in c] + [
        "hits","blocks","faceoffs_won","faceoffs_taken","giveaways","takeaways"
    ]
    for c in fill_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df


def add_individual_xg(df, shot_path):
    print("Computing individual xG per player per game...")
    shots = pd.read_csv(shot_path, low_memory=False)
    shots = shots[
        shots["event_type"].isin(["shot-on-goal","goal"]) &
        (shots["is_empty_net"] == 0)
    ].copy()

    ixg_total = shots.groupby(["game_id","shooter_id"]).agg(
        ixg=("xg","sum")
    ).reset_index().rename(columns={"shooter_id":"player_id"})

    ixg_str = shots.groupby(["game_id","shooter_id","strength"]).agg(
        xg_sum=("xg","sum")
    ).reset_index().rename(columns={"shooter_id":"player_id"})
    ixg_str = ixg_str.pivot_table(
        index=["game_id","player_id"], columns="strength",
        values="xg_sum", fill_value=0
    ).reset_index()
    ixg_str.columns.name = None
    ixg_str = ixg_str.rename(columns={"ev":"ev_ixg","pp":"pp_ixg","sh":"sh_ixg"})
    for c in ["ev_ixg","pp_ixg","sh_ixg"]:
        if c not in ixg_str.columns:
            ixg_str[c] = 0.0

    df = df.merge(ixg_total, on=["game_id","player_id"], how="left")
    df = df.merge(ixg_str[["game_id","player_id","ev_ixg","pp_ixg","sh_ixg"]],
                  on=["game_id","player_id"], how="left")
    for c in ["ixg","ev_ixg","pp_ixg","sh_ixg"]:
        df[c] = df[c].fillna(0.0)

    print(f"  Rows with xG data: {(df['ixg']>0).sum():,}")
    return df


def add_team_context(df, team_gl_path):
    print("Adding team context...")
    tgl = pd.read_csv(team_gl_path, low_memory=False)
    team_toi = tgl[["game_id","team","ev_toi","pp_toi","sh_toi",
                     "ev_shots_on_goal_for","pp_shots_on_goal_for"]].copy()
    team_toi.columns = ["game_id","team","team_ev_toi","team_pp_toi",
                         "team_sh_toi","team_ev_sog","team_pp_sog"]
    df = df.merge(team_toi, on=["game_id","team"], how="left")
    print(f"  Team context added: {df['team_ev_toi'].notna().sum():,} rows")
    return df


def add_game_rates(df):
    print("Computing per-game rates...")
    eps = 1e-6

    df["ev_shooting_pct"] = df["ev_goals"] / (df["ev_shots"] + eps)
    df["pp_shooting_pct"] = df["pp_goals"] / (df["pp_shots"] + eps)
    df["ev_toi_share"]    = df["toi_ev"] / (df["team_ev_toi"].fillna(60) + eps)
    df["pp_toi_share"]    = df["toi_pp"] / (df["team_pp_toi"].fillna(5) + eps)

    df["ev_onice_sf_per60"] = df["ev_onice_sf"] / (df["toi_ev"] / 60 + eps)
    df["ev_onice_sa_per60"] = df["ev_onice_sa"] / (df["toi_ev"] / 60 + eps)
    df["ev_onice_gf_per60"] = df["ev_onice_gf"] / (df["toi_ev"] / 60 + eps)
    df["ev_onice_ga_per60"] = df["ev_onice_ga"] / (df["toi_ev"] / 60 + eps)

    df["ev_shots_per60"]   = df["ev_shots"] / (df["toi_ev"] / 60 + eps)
    df["pp_shots_per60"]   = df["pp_shots"] / (df["toi_pp"] / 60 + eps)
    df["ev_ixg_per60"]     = df["ev_ixg"]   / (df["toi_ev"] / 60 + eps)
    df["pp_ixg_per60"]     = df["pp_ixg"]   / (df["toi_pp"] / 60 + eps)

    # Goals per 60 (more stable than raw goals for modeling)
    df["ev_goals_per60"]   = df["ev_goals"] / (df["toi_ev"] / 60 + eps)
    df["pp_goals_per60"]   = df["pp_goals"] / (df["toi_pp"] / 60 + eps)
    df["ev_goals_per60"]   = df["ev_goals_per60"].clip(0, 10)
    df["pp_goals_per60"]   = df["pp_goals_per60"].clip(0, 10)

    df["ipp"]      = (df["ev_goals"] + df["ev_assists"]) / (df["ev_onice_gf"] + eps)
    df["ipp"]      = df["ipp"].clip(0, 1)
    df["ev_cf_pct"] = df["ev_onice_sf"] / (df["ev_onice_sf"] + df["ev_onice_sa"] + eps)

    # Goals vs xG ratio (finishing talent — per game)
    df["ev_goals_per_ixg"] = df["ev_goals"] / (df["ev_ixg"] + eps)
    df["ev_goals_per_ixg"] = df["ev_goals_per_ixg"].clip(0, 10)

    df["scored_ev_goal"] = (df["ev_goals"] > 0).astype(int)
    df["scored_pp_goal"] = (df["pp_goals"] > 0).astype(int)

    for col in ["ev_shots_per60","pp_shots_per60","ev_ixg_per60","pp_ixg_per60",
                "ev_onice_sf_per60","ev_onice_sa_per60"]:
        df[col] = df[col].clip(0, 200)

    if "points" not in df.columns:
        df["points"] = df["goals"] + df["assists"]

    return df


def add_trend_features(df):
    print("Computing trend and EWM features...")
    df = df.sort_values(["player_id","season","date"]).reset_index(drop=True)

    TREND_STATS = ["toi_ev", "toi_pp", "ev_shots", "pp_shots", "ixg"]

    for stat in TREND_STATS:
        if stat not in df.columns:
            continue
        grp = df.groupby(["player_id","season"])[stat]

        df[f"{stat}_ewm10"] = grp.transform(
            lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
        )

        l5_col  = f"{stat}_last5"
        l20_col = f"{stat}_last20"
        if l5_col in df.columns and l20_col in df.columns:
            df[f"{stat}_trend_ratio"] = (
                df[l5_col] / (df[l20_col] + 1e-6)
            ).clip(0, 5)
            df[f"{stat}_trend_delta"] = df[l5_col] - df[l20_col]

    print("  Trend features added")
    return df


def add_rolling_features(df):
    print("Computing rolling features...")
    df = df.sort_values(["player_id","season","date"]).reset_index(drop=True)

    print("  Computing within-season rolling windows...")
    for stat in ROLL_STATS:
        if stat not in df.columns:
            continue
        grp = df.groupby(["player_id","season"])[stat]
        for w in ROLLING_WINDOWS:
            df[f"{stat}_last{w}"] = (
                grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

    print("  Computing season averages...")
    for stat in ROLL_STATS:
        if stat not in df.columns:
            continue
        df[f"{stat}_season_avg"] = (
            df.groupby(["player_id","season"])[stat]
            .transform(lambda x: x.shift(1).expanding().mean())
        )

    print("  Computing career features...")
    df = df.sort_values(["player_id","date"]).reset_index(drop=True)

    df["career_games"] = (
        df.groupby("player_id")["game_id"]
        .transform(lambda x: x.shift(1).expanding().count())
        .fillna(0).astype(int)
    )

    career_ev_shots = df.groupby("player_id")["ev_shots"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    career_ev_goals = df.groupby("player_id")["ev_goals"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    career_shots = df.groupby("player_id")["shots"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    career_goals = df.groupby("player_id")["goals"].transform(
        lambda x: x.shift(1).expanding().sum()
    )

    df["career_shooting_pct"]    = career_goals    / (career_shots    + 1e-6)
    df["career_ev_shooting_pct"] = career_ev_goals / (career_ev_shots + 1e-6)

    LEAGUE_SH_PCT = 0.098
    REGRESSION_N  = 100
    w_career = career_ev_shots / (career_ev_shots + REGRESSION_N)
    df["regressed_ev_shooting_pct"] = (
        w_career * df["career_ev_shooting_pct"] +
        (1 - w_career) * LEAGUE_SH_PCT
    ).fillna(LEAGUE_SH_PCT)

    # Finishing talent (xG era only)
    LEAGUE_FINISHING = 1.097
    career_ev_ixg_xg = df.groupby("player_id").apply(
        lambda x: x["ev_ixg"].where(x["ev_ixg"] > 0).shift(1).expanding().sum()
    ).reset_index(level=0, drop=True)
    career_ev_goals_xg = df.groupby("player_id").apply(
        lambda x: x["ev_goals"].where(x["ev_ixg"] > 0).shift(1).expanding().sum()
    ).reset_index(level=0, drop=True)

    df["career_ev_ixg_xg"]   = career_ev_ixg_xg.fillna(0)
    df["career_ev_goals_xg"]  = career_ev_goals_xg.fillna(0)
    df["career_finishing_talent"] = (
        df["career_ev_goals_xg"] / (df["career_ev_ixg_xg"] + 1e-6)
    )
    w_finish = df["career_ev_ixg_xg"] / (df["career_ev_ixg_xg"] + 20)
    df["regressed_finishing_talent"] = (
        w_finish * df["career_finishing_talent"] + (1 - w_finish) * LEAGUE_FINISHING
    ).fillna(LEAGUE_FINISHING)
    df = df.drop(columns=["career_ev_ixg_xg","career_ev_goals_xg"])

    print(f"  Rolling features computed: {len(df):,} rows")
    return df


def add_cumulative_rates(df):
    print("Computing cumulative per-60 rates...")
    df = df.sort_values(["player_id","season","date"]).reset_index(drop=True)

    CUM_PAIRS = [
        ("ev_shots",    "toi_ev", "ev_shots_per60_cumulative"),
        ("pp_shots",    "toi_pp", "pp_shots_per60_cumulative"),
        ("ev_goals",    "toi_ev", "ev_goals_per60_cumulative"),
        ("pp_goals",    "toi_pp", "pp_goals_per60_cumulative"),
        ("ev_ixg",      "toi_ev", "ev_ixg_per60_cumulative"),
        ("pp_ixg",      "toi_pp", "pp_ixg_per60_cumulative"),
        ("ev_onice_sf", "toi_ev", "ev_onice_sf_per60_cumulative"),
        ("ev_onice_sa", "toi_ev", "ev_onice_sa_per60_cumulative"),
        ("ev_onice_gf", "toi_ev", "ev_onice_gf_per60_cumulative"),
        ("ev_onice_ga", "toi_ev", "ev_onice_ga_per60_cumulative"),
        ("hits",        "toi",    "hits_per60_cumulative"),
        ("blocks",      "toi",    "blocks_per60_cumulative"),
    ]

    for stat, toi_col, out_col in CUM_PAIRS:
        if stat not in df.columns or toi_col not in df.columns:
            continue
        grp     = df.groupby(["player_id","season"])
        cum_stat = grp[stat].transform(lambda x: x.shift(1).expanding().sum())
        cum_toi  = grp[toi_col].transform(lambda x: x.shift(1).expanding().sum())
        df[out_col] = cum_stat / (cum_toi / 60 + 1e-6)
        df.loc[cum_toi == 0, out_col] = np.nan

    print("  Cumulative rates added")
    return df


def main():
    print("="*60)
    print("BUILDING PLAYER FEATURES")
    print("="*60)

    df = load_base(PLAYER_GL, PLAYER_PBP)
    df = add_individual_xg(df, SHOT_XG)
    df = add_team_context(df, TEAM_GL)
    df = add_game_rates(df)

    print(f"\nBase dataset: {len(df):,} rows, {df.shape[1]} columns")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Players: {df['player_id'].nunique():,}")

    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_cumulative_rates(df)

    df = df.sort_values(["date","player_id"]).reset_index(drop=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Saved:   {OUT_FILE}")

    base_cols    = [c for c in df.columns if not any(s in c for s in
                    ["last5","last10","last20","last30","season_avg","career","cumulative","ewm","trend"])]
    rolling_cols = [c for c in df.columns if any(s in c for s in
                    ["last5","last10","last20","last30","season_avg"])]
    career_cols  = [c for c in df.columns if "career" in c]
    cumul_cols   = [c for c in df.columns if "cumulative" in c]
    trend_cols   = [c for c in df.columns if "ewm" in c or "trend" in c]
    print(f"  Base game stats:  {len(base_cols)}")
    print(f"  Rolling features: {len(rolling_cols)}")
    print(f"  Career features:  {len(career_cols)}")
    print(f"  Cumulative rates: {len(cumul_cols)}")
    print(f"  Trend/EWM:        {len(trend_cols)}")

    mcd = df[df["full_name"]=="Connor McDavid"].sort_values("date")
    if len(mcd) > 0:
        last = mcd.iloc[-1]
        print(f"\nMcDavid sample (last game):")
        for c in ["season","date","toi_ev","toi_pp","ev_shots","ixg",
                  "ev_shots_per60_last20","pp_shots_per60_last20",
                  "regressed_ev_shooting_pct","regressed_finishing_talent",
                  "career_games","ev_onice_sf_per60_last20","ipp_last20"]:
            if c in last.index:
                print(f"  {c}: {last[c]}")


if __name__ == "__main__":
    main()