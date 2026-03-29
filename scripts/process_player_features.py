"""
process_player_features.py

Builds per-player per-game feature dataset for player modeling.

Sources:
  - data/raw/game_logs/player_game_logs.csv         (11 seasons basic stats)
  - data/raw/player_pbp_stats/player_pbp_stats.csv  (6 seasons strength splits + on-ice + Corsi)
  - data/processed/shot_data_with_xg.csv            (4 seasons individual xG)
  - data/raw/team_game_logs/team_game_logs.csv       (team context)
  - data/raw/player_pp_stats.csv                    (11 seasons PP shots/goals from NHL API)
  - data/raw/player_corsi_stats.csv                 (11 seasons Corsi from NHL API)

Output: data/processed/player_features.csv

Rolling windows: last5/10/20/30 — within season only (reset at season boundary)
Season avg: expanding mean within season
Career features: cross-season (no reset)
Corsi rolling features: only from 20202021+ (real per-game PBP data)
PP TOI share: weighted EWM over past seasons (recent weighted more)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PLAYER_GL    = DATA_DIR / "raw/game_logs/player_game_logs.csv"
PLAYER_PBP   = DATA_DIR / "raw/player_pbp_stats/player_pbp_stats.csv"
SHOT_XG      = DATA_DIR / "processed/shot_data_with_xg.csv"
TEAM_GL      = DATA_DIR / "raw/team_game_logs/team_game_logs.csv"
PLAYER_PP    = DATA_DIR / "raw/player_pp_stats.csv"
PLAYER_CORSI = DATA_DIR / "raw/player_corsi_stats.csv"
OUT_FILE     = DATA_DIR / "processed/player_features.csv"

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
    "is_pp_player",
    # Corsi — only populated for 20202021+
    "indiv_shot_attempts", "indiv_missed_shots", "indiv_shots_blocked",
    "ev_shot_attempts", "pp_shot_attempts",
    "ev_missed_shots", "pp_missed_shots",
    "ev_shots_blocked_by_opp", "pp_shots_blocked_by_opp",
    "indiv_shot_attempts_per60", "ev_shot_attempts_per60", "pp_shot_attempts_per60",
    "team_pp_toi", "team_ev_toi",
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
        "ev_missed_shots", "pp_missed_shots", "sh_missed_shots",
        "ev_shots_blocked_by_opp", "pp_shots_blocked_by_opp", "sh_shots_blocked_by_opp",
        "ev_shot_attempts", "pp_shot_attempts", "sh_shot_attempts",
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

    # Fix merge column collisions
    for base in ["pp_goals", "sh_goals"]:
        x_col, y_col = f"{base}_x", f"{base}_y"
        if x_col in df.columns:
            df[base] = df[x_col].fillna(0)
            df = df.drop(columns=[x_col, y_col], errors="ignore")

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
        "hits","blocks","faceoffs_won","faceoffs_taken","giveaways","takeaways",
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
    tgl["date"] = pd.to_datetime(tgl["date"])
    tgl = tgl.sort_values(["team","date"])

    # Team's own PP/EV/SH TOI (already there)
    team_toi = tgl[["game_id","team","ev_toi","pp_toi","sh_toi",
                     "ev_shots_on_goal_for","pp_shots_on_goal_for"]].copy()
    team_toi.columns = ["game_id","team","team_ev_toi","team_pp_toi",
                         "team_sh_toi","team_ev_sog","team_pp_sog"]
    df = df.merge(team_toi, on=["game_id","team"], how="left")

    # Opponent defensive rolling stats (L30)
    # These represent: how many shots/attempts does this opponent ALLOW per game
    opp_def_cols = [
        "ev_shots_on_goal_against",   # shots they allow at EV
        "ev_shot_attempts_against",   # Corsi they allow at EV
        "ev_blocked_shots_for",       # shots they block (shot suppression skill)
        "ev_goals_against",           # EV goals they allow
        "ev_shots_on_goal_for",       # their own EV shot volume (pace proxy)
    ]

    opp_rolling = tgl[["game_id","team"] + opp_def_cols].copy()

    # Compute L30 rolling averages per team (shift=1 to avoid leakage)
    for col in opp_def_cols:
        opp_rolling[f"opp_{col}_last30"] = (
            tgl.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(30, min_periods=8).mean())
        )

    # Rename team → opponent so we can join on opponent
    opp_rolling = opp_rolling[["game_id","team"] +
                               [f"opp_{c}_last30" for c in opp_def_cols]].copy()
    opp_rolling = opp_rolling.rename(columns={"team":"opponent"})

    df = df.merge(opp_rolling, on=["game_id","opponent"], how="left")
    print(f"  Team context added: {df['team_ev_toi'].notna().sum():,} rows")
    print(f"  Opponent context added: {df['opp_ev_shots_on_goal_against_last30'].notna().sum():,} rows")
    return df


def add_pp_career_stats(df, pp_path):
    """Join accurate PP shots/goals from NHL stats API."""
    print("Adding career PP stats...")
    pp = pd.read_csv(pp_path, low_memory=False)
    pp = pp[["player_id","season","pp_goals","pp_shots","pp_toi_sec"]].copy()
    pp.columns = ["player_id","season","pp_goals_api","pp_shots_api","pp_toi_sec_api"]
    pp = pp.groupby(["player_id","season"]).sum().reset_index()

    df = df.merge(pp, on=["player_id","season"], how="left")

    games_per_season = df.groupby(["player_id","season"])["game_id"].transform("count")
    df["pp_shots_api_per_game"] = df["pp_shots_api"] / games_per_season.clip(lower=1)

    # Use API per-game for pre-PBP seasons, actual PBP for 20202021+
    pre_pbp = df["season"] < 20202021
    df["pp_shots_api_filled"] = np.where(
        pre_pbp,
        df["pp_shots_api_per_game"].fillna(0),
        df["pp_shots"].fillna(0)
    )

    print(f"  PP API data joined: {df['pp_shots_api'].notna().sum():,} rows")
    df = df.drop(columns=["pp_shots_api_per_game"], errors="ignore")
    return df


def add_corsi_stats(df, corsi_path):
    """
    Add individual Corsi components.
    For 20202021+: use actual per-game PBP values.
    For pre-20202021: set to NaN — season API averages are constants
    that pollute rolling window features.
    """
    print("Adding individual shot attempt stats...")

    pre_pbp = df["season"] < 20202021

    df["indiv_shot_attempts"] = np.where(
        pre_pbp, np.nan,
        df["ev_shot_attempts"].fillna(0) +
        df["pp_shot_attempts"].fillna(0) +
        df["sh_shot_attempts"].fillna(0)
    )
    df["indiv_missed_shots"] = np.where(
        pre_pbp, np.nan,
        df["ev_missed_shots"].fillna(0) +
        df["pp_missed_shots"].fillna(0) +
        df["sh_missed_shots"].fillna(0)
    )
    df["indiv_shots_blocked"] = np.where(
        pre_pbp, np.nan,
        df["ev_shots_blocked_by_opp"].fillna(0) +
        df["pp_shots_blocked_by_opp"].fillna(0) +
        df["sh_shots_blocked_by_opp"].fillna(0)
    )

    for col in ["ev_shot_attempts","pp_shot_attempts","sh_shot_attempts",
                "ev_missed_shots","pp_missed_shots","sh_missed_shots",
                "ev_shots_blocked_by_opp","pp_shots_blocked_by_opp","sh_shots_blocked_by_opp"]:
        if col in df.columns:
            df.loc[pre_pbp, col] = np.nan

    pbp_rows = (~pre_pbp & df["indiv_shot_attempts"].notna()).sum()
    print(f"  Corsi data available: {pbp_rows:,} rows (20202021+)")
    return df


def add_game_rates(df):
    print("Computing per-game rates...")
    eps = 1e-6

    df["ev_shooting_pct"] = df["ev_goals"] / (df["ev_shots"] + eps)
    df["pp_shooting_pct"] = df["pp_goals"] / (df["pp_shots"] + eps)
    df["ev_toi_share"]    = df["toi_ev"] / (df["team_ev_toi"].fillna(60) + eps)
    df["pp_toi_share"]    = df["toi_pp"] / (df["team_pp_toi"].fillna(5)  + eps)

    df["ev_onice_sf_per60"] = df["ev_onice_sf"] / (df["toi_ev"] / 60 + eps)
    df["ev_onice_sa_per60"] = df["ev_onice_sa"] / (df["toi_ev"] / 60 + eps)
    df["ev_onice_gf_per60"] = df["ev_onice_gf"] / (df["toi_ev"] / 60 + eps)
    df["ev_onice_ga_per60"] = df["ev_onice_ga"] / (df["toi_ev"] / 60 + eps)

    df["ev_shots_per60"]   = df["ev_shots"] / (df["toi_ev"] / 60 + eps)
    df["pp_shots_per60"]   = df["pp_shots"] / (df["toi_pp"] / 60 + eps)
    df["ev_ixg_per60"]     = df["ev_ixg"]   / (df["toi_ev"] / 60 + eps)
    df["pp_ixg_per60"]     = df["pp_ixg"]   / (df["toi_pp"] / 60 + eps)

    df["ev_goals_per60"]   = (df["ev_goals"] / (df["toi_ev"] / 60 + eps)).clip(0, 10)
    df["pp_goals_per60"]   = (df["pp_goals"] / (df["toi_pp"] / 60 + eps)).clip(0, 10)

    df["indiv_shot_attempts_per60"] = df["indiv_shot_attempts"] / (df["toi"] / 60 + eps)
    df["ev_shot_attempts_per60"]    = df["ev_shot_attempts"]    / (df["toi_ev"] / 60 + eps)
    df["pp_shot_attempts_per60"]    = df["pp_shot_attempts"]    / (df["toi_pp"] / 60 + eps)
    for col in ["indiv_shot_attempts_per60","ev_shot_attempts_per60","pp_shot_attempts_per60"]:
        df[col] = df[col].clip(0, 200)

    df["ipp"]       = (df["ev_goals"] + df["ev_assists"]) / (df["ev_onice_gf"] + eps)
    df["ipp"]       = df["ipp"].clip(0, 1)
    df["ev_cf_pct"] = df["ev_onice_sf"] / (df["ev_onice_sf"] + df["ev_onice_sa"] + eps)

    df["ev_goals_per_ixg"] = (df["ev_goals"] / (df["ev_ixg"] + eps)).clip(0, 10)

    df["scored_ev_goal"] = (df["ev_goals"] > 0).astype(int)
    df["scored_pp_goal"] = (df["pp_goals"] > 0).astype(int)

    # PP player indicator
    df["is_pp_player"] = (df["toi_pp"] > 0.5).astype(float)

    for col in ["ev_shots_per60","pp_shots_per60","ev_ixg_per60","pp_ixg_per60",
                "ev_onice_sf_per60","ev_onice_sa_per60"]:
        df[col] = df[col].clip(0, 200)

    if "points" not in df.columns:
        df["points"] = df["goals"] + df["assists"]

    return df


def add_trend_features(df):
    print("Computing trend and EWM features...")
    df = df.sort_values(["player_id","season","date"]).reset_index(drop=True)

    TREND_STATS = ["toi_ev", "toi_pp", "ev_shots", "pp_shots", "ixg",
                   "indiv_shot_attempts", "ev_shot_attempts"]

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

    # Career PP shooting%
    LEAGUE_PP_SH_PCT = 0.144
    PP_REGRESSION_N  = 30
    pp_shots_col = "pp_shots_api_filled" if "pp_shots_api_filled" in df.columns else "pp_shots"
    career_pp_shots = df.groupby("player_id")[pp_shots_col].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    career_pp_goals = df.groupby("player_id")["pp_goals"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    df["career_pp_shooting_pct"] = career_pp_goals / (career_pp_shots + 1e-6)
    w_pp = career_pp_shots / (career_pp_shots + PP_REGRESSION_N)
    df["regressed_pp_shooting_pct"] = (
        w_pp * df["career_pp_shooting_pct"] +
        (1 - w_pp) * LEAGUE_PP_SH_PCT
    ).fillna(LEAGUE_PP_SH_PCT)

    # Career SH shots per 60
    LEAGUE_SH_SHOTS_PER60 = 5.44
    SH_REGRESSION_N       = 20
    career_sh_shots = df.groupby("player_id")["sh_shots"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    career_sh_toi = df.groupby("player_id")["toi_sh"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    df["career_sh_shots_per60"] = career_sh_shots / (career_sh_toi / 60 + 1e-6)
    w_sh = career_sh_toi / (career_sh_toi + SH_REGRESSION_N)
    df["regressed_sh_shots_per60"] = (
        w_sh * df["career_sh_shots_per60"] +
        (1 - w_sh) * LEAGUE_SH_SHOTS_PER60
    ).fillna(LEAGUE_SH_SHOTS_PER60)

    # Finishing talent (xG era only)
    LEAGUE_FINISHING = 1.097
    career_ev_ixg_xg = df.groupby("player_id").apply(
        lambda x: x["ev_ixg"].where(x["ev_ixg"] > 0).shift(1).expanding().sum()
    ).reset_index(level=0, drop=True)
    career_ev_goals_xg = df.groupby("player_id").apply(
        lambda x: x["ev_goals"].where(x["ev_ixg"] > 0).shift(1).expanding().sum()
    ).reset_index(level=0, drop=True)

    df["career_ev_ixg_xg"]       = career_ev_ixg_xg.fillna(0)
    df["career_ev_goals_xg"]      = career_ev_goals_xg.fillna(0)
    df["career_finishing_talent"] = (
        df["career_ev_goals_xg"] / (df["career_ev_ixg_xg"] + 1e-6)
    )
    w_finish = df["career_ev_ixg_xg"] / (df["career_ev_ixg_xg"] + 20)
    df["regressed_finishing_talent"] = (
        w_finish * df["career_finishing_talent"] + (1 - w_finish) * LEAGUE_FINISHING
    ).fillna(LEAGUE_FINISHING)
    df = df.drop(columns=["career_ev_ixg_xg","career_ev_goals_xg"])

    # Career home vs away shot tendency
    # home_shot_ratio > 1.0 means player shoots more at home
    # Regressed toward 1.0 (no home advantage) with 50-game prior
    print("  Computing home/away shot tendency...")
    df = df.sort_values(["player_id","date"]).reset_index(drop=True)

    career_home_shots = df[df["is_home"]==1].groupby("player_id")["ev_shots"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    career_away_shots = df[df["is_home"]==0].groupby("player_id")["ev_shots"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Fill forward so away rows have the home career avg and vice versa
    df["_career_home_shots"] = career_home_shots
    df["_career_away_shots"] = career_away_shots
    df["_career_home_shots"] = df.groupby("player_id")["_career_home_shots"].ffill().bfill()
    df["_career_away_shots"] = df.groupby("player_id")["_career_away_shots"].ffill().bfill()

    # Home shot ratio: how much more does this player shoot at home vs away?
    df["career_home_shot_ratio"] = (
        df["_career_home_shots"] / (df["_career_away_shots"] + 1e-6)
    ).clip(0.5, 2.0)

    # Regress toward 1.0 (no home advantage) based on games played each side
    career_home_games = (df["is_home"]==1).groupby(df["player_id"]).transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)
    career_away_games = (df["is_home"]==0).groupby(df["player_id"]).transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)
    min_games = career_home_games.clip(upper=career_away_games)
    # Full trust at 50+ games each side
    trust_weight = (min_games / 50.0).clip(0, 1)
    df["regressed_home_shot_ratio"] = (
        trust_weight * df["career_home_shot_ratio"] + (1 - trust_weight) * 1.0
    ).fillna(1.0)

    df = df.drop(columns=["_career_home_shots","_career_away_shots"], errors="ignore")

    # Career home vs away goal tendency
    career_home_goals = df[df["is_home"]==1].groupby("player_id")["ev_goals"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    career_away_goals = df[df["is_home"]==0].groupby("player_id")["ev_goals"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["_career_home_goals"] = career_home_goals
    df["_career_away_goals"] = career_away_goals
    df["_career_home_goals"] = df.groupby("player_id")["_career_home_goals"].ffill().bfill()
    df["_career_away_goals"] = df.groupby("player_id")["_career_away_goals"].ffill().bfill()
    df["career_home_goal_ratio"] = (
        df["_career_home_goals"] / (df["_career_away_goals"] + 1e-6)
    ).clip(0.3, 3.0)
    # Need more games to trust goal splits (rarer events) — 100 game prior each side
    min_goal_games = career_home_games.clip(upper=career_away_games)
    goal_trust_weight = (min_goal_games / 100.0).clip(0, 1)
    df["regressed_home_goal_ratio"] = (
        goal_trust_weight * df["career_home_goal_ratio"] + (1 - goal_trust_weight) * 1.0
    ).fillna(1.0)
    df = df.drop(columns=["_career_home_goals","_career_away_goals"], errors="ignore")

    # Career home vs away assist tendency
    career_home_assists = df[df["is_home"]==1].groupby("player_id")["ev_assists"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    career_away_assists = df[df["is_home"]==0].groupby("player_id")["ev_assists"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["_career_home_assists"] = career_home_assists
    df["_career_away_assists"] = career_away_assists
    df["_career_home_assists"] = df.groupby("player_id")["_career_home_assists"].ffill().bfill()
    df["_career_away_assists"] = df.groupby("player_id")["_career_away_assists"].ffill().bfill()
    df["career_home_assist_ratio"] = (
        df["_career_home_assists"] / (df["_career_away_assists"] + 1e-6)
    ).clip(0.5, 2.0)
    df["regressed_home_assist_ratio"] = (
        trust_weight * df["career_home_assist_ratio"] + (1 - trust_weight) * 1.0
    ).fillna(1.0)
    df = df.drop(columns=["_career_home_assists","_career_away_assists"], errors="ignore")

    # Weighted PP TOI share across past seasons (EWM, recent weighted more)
    print("  Computing weighted PP TOI share...")
    season_pp = df.groupby(["player_id","season"]).agg(
        total_pp_toi  = ("toi_pp","sum"),
        total_team_pp = ("team_pp_toi","sum"),
    ).reset_index()
    season_pp["season_pp_share"] = (
        season_pp["total_pp_toi"] / (season_pp["total_team_pp"] + 1e-6)
    ).clip(0, 1)
    season_pp = season_pp.sort_values(["player_id","season"])
    # EWM over past seasons — shift(1) so current season excluded
    season_pp["weighted_pp_share"] = (
        season_pp.groupby("player_id")["season_pp_share"]
        .transform(lambda x: x.shift(1).ewm(span=2, min_periods=1).mean())
    )
    df = df.merge(
        season_pp[["player_id","season","weighted_pp_share"]],
        on=["player_id","season"], how="left"
    )
    # Home tendency interaction features
    # These directly encode: "is this player at home AND do they benefit from home?"
    df["home_shot_boost"]   = df["is_home"] * (df["regressed_home_shot_ratio"] - 1.0)
    df["home_goal_boost"]   = df["is_home"] * (df["regressed_home_goal_ratio"] - 1.0)
    df["home_assist_boost"] = df["is_home"] * (df["regressed_home_assist_ratio"] - 1.0)

    print(f"  Rolling features computed: {len(df):,} rows")
    return df


def add_cumulative_rates(df):
    print("Computing cumulative per-60 rates...")
    df = df.sort_values(["player_id","season","date"]).reset_index(drop=True)

    CUM_PAIRS = [
        ("ev_shots",         "toi_ev", "ev_shots_per60_cumulative"),
        ("pp_shots",         "toi_pp", "pp_shots_per60_cumulative"),
        ("ev_goals",         "toi_ev", "ev_goals_per60_cumulative"),
        ("pp_goals",         "toi_pp", "pp_goals_per60_cumulative"),
        ("ev_ixg",           "toi_ev", "ev_ixg_per60_cumulative"),
        ("pp_ixg",           "toi_pp", "pp_ixg_per60_cumulative"),
        ("ev_onice_sf",      "toi_ev", "ev_onice_sf_per60_cumulative"),
        ("ev_onice_sa",      "toi_ev", "ev_onice_sa_per60_cumulative"),
        ("ev_onice_gf",      "toi_ev", "ev_onice_gf_per60_cumulative"),
        ("ev_onice_ga",      "toi_ev", "ev_onice_ga_per60_cumulative"),
        ("hits",             "toi",    "hits_per60_cumulative"),
        ("blocks",           "toi",    "blocks_per60_cumulative"),
        ("ev_shot_attempts", "toi_ev", "ev_shot_attempts_per60_cumulative"),
        ("pp_shot_attempts", "toi_pp", "pp_shot_attempts_per60_cumulative"),
        ("indiv_missed_shots","toi",   "indiv_missed_shots_per60_cumulative"),
    ]

    for stat, toi_col, out_col in CUM_PAIRS:
        if stat not in df.columns or toi_col not in df.columns:
            continue
        grp      = df.groupby(["player_id","season"])
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
    df = add_pp_career_stats(df, PLAYER_PP)
    df = add_corsi_stats(df, PLAYER_CORSI)
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
                    ["last5","last10","last20","last30","season_avg",
                     "career","cumulative","ewm","trend"])]
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
                  "regressed_pp_shooting_pct","weighted_pp_share",
                  "career_games","ev_shot_attempts_last20"]:
            if c in last.index:
                print(f"  {c}: {last[c]}")


if __name__ == "__main__":
    main()