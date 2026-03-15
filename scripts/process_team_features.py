import pandas as pd
import numpy as np
import os

# --- CONFIG ---
DATA_DIR = os.path.expanduser("~/nhl-props/data")
TEAM_LOGS = os.path.join(DATA_DIR, "raw/team_game_logs/team_game_logs.csv")
SHOT_XG = os.path.join(DATA_DIR, "processed/shot_data_with_xg.csv")
GOALIE_LOGS = os.path.join(DATA_DIR, "raw/goalie_logs/goalie_game_logs.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "processed/team_features.csv")

ROLLING_WINDOWS = [10, 20, 30]  # L5 dropped - empirically adds noise not signal

TEAM_STATS = [
    "goals_for", "goals_against",
    "ev_shots_on_goal_for", "ev_shots_on_goal_against",
    "ev_shot_attempts_for", "ev_shot_attempts_against",
    "ev_missed_shots_for", "ev_missed_shots_against",
    "ev_blocked_shots_for", "ev_blocked_shots_against",
    "ev_goals_for", "ev_goals_against",
    "ev_hits_for", "ev_hits_against",
    "ev_giveaways", "ev_takeaways",
    "ev_faceoffs_won", "ev_faceoffs_taken",
    "pp_shots_on_goal_for", "pp_shots_on_goal_against",
    "pp_shot_attempts_for", "pp_shot_attempts_against",
    "pp_blocked_shots_for", "pp_blocked_shots_against",
    "pp_goals_for", "pp_goals_against",
    "pp_penalties_drawn", "pp_penalty_minutes",
    "sh_shots_on_goal_for", "sh_shots_on_goal_against",
    "sh_shot_attempts_for", "sh_shot_attempts_against",
    "sh_blocked_shots_for", "sh_blocked_shots_against",
    "sh_goals_for", "sh_goals_against",
    "sh_penalties_taken", "sh_penalty_minutes",
    # Totals
    "total_shots_on_goal_for", "total_shots_on_goal_against",
    "total_shot_attempts_for", "total_shot_attempts_against",
    "total_fenwick_for", "total_fenwick_against",
    # TOI
    "ev_toi", "pp_toi", "sh_toi", "en_toi",
    # Per-60 rates
    "ev_shots_on_goal_for_per60", "ev_shots_on_goal_against_per60",
    "ev_shot_attempts_for_per60", "ev_shot_attempts_against_per60",
    "ev_goals_for_per60", "ev_goals_against_per60",
    "ev_fenwick_for_per60", "ev_fenwick_against_per60",
    "ev_hits_for_per60", "ev_hits_against_per60",
    "ev_giveaways_per60", "ev_takeaways_per60",
    "pp_shots_on_goal_for_per60", "pp_shots_on_goal_against_per60",
    "pp_shot_attempts_for_per60", "pp_shot_attempts_against_per60",
    "pp_goals_for_per60", "pp_goals_against_per60",
    "pp_fenwick_for_per60", "pp_fenwick_against_per60",
    "sh_shots_on_goal_for_per60", "sh_shots_on_goal_against_per60",
    "sh_shot_attempts_for_per60", "sh_shot_attempts_against_per60",
    "sh_goals_for_per60", "sh_goals_against_per60",
    "sh_fenwick_for_per60", "sh_fenwick_against_per60",
    # xG
    "xgf_sog_total", "xga_sog_total",
    # PP/SH hits, giveaways, takeaways per60 (needed for TOI models)
    "pp_hits_for_per60", "pp_hits_against_per60",
    "sh_hits_for_per60", "sh_hits_against_per60",
    "pp_giveaways_per60", "pp_takeaways_per60",
    "sh_giveaways_per60", "sh_takeaways_per60",
    "pp_shot_attempts_for_per60", "pp_shot_attempts_against_per60",
    "sh_shot_attempts_for_per60", "sh_shot_attempts_against_per60",
    # Shot funnel rates (rolling avgs computed for these)
    "ev_fenwick_rate_for", "ev_block_rate_against",
    "ev_sog_fenwick_rate_for", "ev_shooting_pct",
    "ev_fenwick_rate_against", "ev_block_rate_for",
    "ev_sog_fenwick_rate_against", "ev_save_pct_team",
    "ev_xg_per_sog_for", "ev_xg_per_sog_against",
    "pp_fenwick_rate_for", "pp_sog_fenwick_rate_for", "pp_shooting_pct",
    "pp_fenwick_rate_against", "pp_sog_fenwick_rate_against", "pp_save_pct_team",
    "sh_fenwick_rate_for", "sh_sog_fenwick_rate_for", "sh_shooting_pct",
    "sh_fenwick_rate_against", "sh_sog_fenwick_rate_against", "sh_save_pct_team",
]


def aggregate_xg(shot_df):
    """Aggregate shot-level xG into team-game-level totals by strength."""
    print("  Aggregating xG per team per game...")
    shot_df = shot_df[shot_df["is_empty_net"] == 0].copy()

    xgf = shot_df.groupby(["game_id", "shooting_team", "strength"])["xg"].sum().reset_index()
    xgf.columns = ["game_id", "team", "strength", "xgf"]
    xga = shot_df.groupby(["game_id", "defending_team", "strength"])["xg"].sum().reset_index()
    xga.columns = ["game_id", "team", "strength", "xga"]
    xg = xgf.merge(xga, on=["game_id", "team", "strength"], how="outer").fillna(0)
    xg_pivot = xg.pivot_table(
        index=["game_id", "team"], columns="strength",
        values=["xgf", "xga"], aggfunc="sum"
    ).fillna(0)
    xg_pivot.columns = [f"{stat}_{strength}" for stat, strength in xg_pivot.columns]
    xg_pivot = xg_pivot.reset_index()
    xgf_cols = [c for c in xg_pivot.columns if c.startswith("xgf_")]
    xga_cols = [c for c in xg_pivot.columns if c.startswith("xga_")]
    xg_pivot["xgf_total"] = xg_pivot[xgf_cols].sum(axis=1)
    xg_pivot["xga_total"] = xg_pivot[xga_cols].sum(axis=1)

    sog_df = shot_df[shot_df["is_on_goal"] == 1].copy()
    xgf_sog = sog_df.groupby(["game_id", "shooting_team", "strength"])["xg"].sum().reset_index()
    xgf_sog.columns = ["game_id", "team", "strength", "xgf_sog"]
    xga_sog = sog_df.groupby(["game_id", "defending_team", "strength"])["xg"].sum().reset_index()
    xga_sog.columns = ["game_id", "team", "strength", "xga_sog"]
    xg_sog = xgf_sog.merge(xga_sog, on=["game_id", "team", "strength"], how="outer").fillna(0)
    xg_sog_pivot = xg_sog.pivot_table(
        index=["game_id", "team"], columns="strength",
        values=["xgf_sog", "xga_sog"], aggfunc="sum"
    ).fillna(0)
    xg_sog_pivot.columns = [f"{stat}_{strength}" for stat, strength in xg_sog_pivot.columns]
    xg_sog_pivot = xg_sog_pivot.reset_index()
    xgf_sog_cols = [c for c in xg_sog_pivot.columns if c.startswith("xgf_sog_")]
    xga_sog_cols = [c for c in xg_sog_pivot.columns if c.startswith("xga_sog_")]
    xg_sog_pivot["xgf_sog_total"] = xg_sog_pivot[xgf_sog_cols].sum(axis=1)
    xg_sog_pivot["xga_sog_total"] = xg_sog_pivot[xga_sog_cols].sum(axis=1)

    result = xg_pivot.merge(xg_sog_pivot, on=["game_id", "team"], how="left").fillna(0)
    print(f"  xG aggregated for {len(result)} team-game rows")
    print(f"  CSD mean: {result['xgf_total'].mean():.2f}  "
          f"xG (SOG only) mean: {result['xgf_sog_total'].mean():.2f}")
    return result


def aggregate_goalie_stats(goalie_df):
    """Compute primary goalie save% and GSAx across multiple time windows."""
    print("  Processing goalie stats...")
    goalie_df = goalie_df.copy()
    goalie_df["date"] = pd.to_datetime(goalie_df["date"])
    goalie_df = goalie_df.sort_values(["player_id", "date"]).reset_index(drop=True)

    results = []
    for player_id, g in goalie_df.groupby("player_id"):
        g = g.copy().reset_index(drop=True)
        g["gsax"] = g["saves"] - (g["shots_against"] * 0.906)

        for i in range(len(g)):
            row = g.iloc[i].copy()
            past = g.iloc[:i]

            for window in [20, 40]:
                past_w = past.tail(window)
                if len(past_w) > 0:
                    total_shots = past_w["shots_against"].sum()
                    total_saves = past_w["saves"].sum()
                    row[f"save_pct_last{window}"] = (
                        total_saves / total_shots if total_shots > 0 else None
                    )
                    row[f"gsax_per60_last{window}"] = (
                        past_w["gsax"].sum() / (past_w["toi"].sum() / 60)
                        if past_w["toi"].sum() > 0 else None
                    )
                else:
                    row[f"save_pct_last{window}"] = None
                    row[f"gsax_per60_last{window}"] = None

            current_season = row["season"]
            past_season = past[past["season"] == current_season]
            if len(past_season) > 0:
                total_shots = past_season["shots_against"].sum()
                total_saves = past_season["saves"].sum()
                row["save_pct_current_season"] = (
                    total_saves / total_shots if total_shots > 0 else None
                )
                row["gsax_per60_current_season"] = (
                    past_season["gsax"].sum() / (past_season["toi"].sum() / 60)
                    if past_season["toi"].sum() > 0 else None
                )
            else:
                row["save_pct_current_season"] = None
                row["gsax_per60_current_season"] = None

            if len(past) > 0:
                total_shots = past["shots_against"].sum()
                total_saves = past["saves"].sum()
                row["save_pct_career"] = (
                    total_saves / total_shots if total_shots > 0 else None
                )
                row["gsax_per60_career"] = (
                    past["gsax"].sum() / (past["toi"].sum() / 60)
                    if past["toi"].sum() > 0 else None
                )
            else:
                row["save_pct_career"] = None
                row["gsax_per60_career"] = None

            results.append(row)

    goalie_features = pd.DataFrame(results)
    primary_goalies = goalie_features.sort_values("toi", ascending=False).drop_duplicates(
        subset=["game_id", "team"]
    )[["game_id", "team", "save_pct_last20", "save_pct_last40",
       "save_pct_current_season", "save_pct_career",
       "gsax_per60_last20", "gsax_per60_last40",
       "gsax_per60_current_season", "gsax_per60_career"]]

    print(f"  Goalie features computed for {len(primary_goalies)} team-game rows")
    return primary_goalies


def add_fenwick(df):
    """Fenwick = shot attempts - blocked shots."""
    new_cols = {}
    for strength in ["ev", "pp", "sh"]:
        new_cols[f"{strength}_fenwick_for"] = (
            df[f"{strength}_shot_attempts_for"] - df[f"{strength}_blocked_shots_against"]
        )
        new_cols[f"{strength}_fenwick_against"] = (
            df[f"{strength}_shot_attempts_against"] - df[f"{strength}_blocked_shots_for"]
        )

    new_cols["total_shots_on_goal_for"] = (
        df["ev_shots_on_goal_for"] + df["pp_shots_on_goal_for"] + df["sh_shots_on_goal_for"]
    )
    new_cols["total_shots_on_goal_against"] = (
        df["ev_shots_on_goal_against"] + df["pp_shots_on_goal_against"] + df["sh_shots_on_goal_against"]
    )
    new_cols["total_shot_attempts_for"] = (
        df["ev_shot_attempts_for"] + df["pp_shot_attempts_for"] + df["sh_shot_attempts_for"]
    )
    new_cols["total_shot_attempts_against"] = (
        df["ev_shot_attempts_against"] + df["pp_shot_attempts_against"] + df["sh_shot_attempts_against"]
    )
    new_cols["total_fenwick_for"] = (
        new_cols["ev_fenwick_for"] + new_cols["pp_fenwick_for"] + new_cols["sh_fenwick_for"]
    )
    new_cols["total_fenwick_against"] = (
        new_cols["ev_fenwick_against"] + new_cols["pp_fenwick_against"] + new_cols["sh_fenwick_against"]
    )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_per60_rates(df):
    """Calculate per-60 rate stats at each strength."""
    new_cols = {}
    for strength in ["ev", "pp", "sh"]:
        toi_col = f"{strength}_toi"
        toi_60 = (df[toi_col] / 60).replace(0, np.nan)
        for stat in ["shots_on_goal_for", "shots_on_goal_against",
                     "shot_attempts_for", "shot_attempts_against",
                     "goals_for", "goals_against",
                     "hits_for", "hits_against",
                     "giveaways", "takeaways"]:
            col = f"{strength}_{stat}"
            if col in df.columns:
                new_cols[f"{strength}_{stat}_per60"] = df[col] / toi_60
        if f"{strength}_fenwick_for" in df.columns:
            new_cols[f"{strength}_fenwick_for_per60"] = df[f"{strength}_fenwick_for"] / toi_60
            new_cols[f"{strength}_fenwick_against_per60"] = df[f"{strength}_fenwick_against"] / toi_60

    new_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    per60_cols = [c for c in new_df.columns if c.endswith("_per60")]
    new_df[per60_cols] = new_df[per60_cols].fillna(0)
    return new_df


def add_pp_pk(df):
    """Add power play % and penalty kill %."""
    new_cols = {
        "pp_pct": (df["pp_goals_for"] / df["pp_penalties_drawn"].replace(0, np.nan)).fillna(0),
        "pk_pct": (1 - (df["pp_goals_against"] / df["sh_penalties_taken"].replace(0, np.nan))).fillna(1),
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_shot_funnel_rates(df):
    """
    Calculate team-specific shot funnel conversion rates.
    Shot Attempts → Fenwick (unblocked) → SOG → Goals
    """
    new_cols = {}

    for strength in ["ev", "pp", "sh"]:
        att_for = df[f"{strength}_shot_attempts_for"].replace(0, np.nan)
        att_against = df[f"{strength}_shot_attempts_against"].replace(0, np.nan)
        fen_for = df[f"{strength}_fenwick_for"].replace(0, np.nan)
        fen_against = df[f"{strength}_fenwick_against"].replace(0, np.nan)
        sog_for = df[f"{strength}_shots_on_goal_for"].replace(0, np.nan)
        sog_against = df[f"{strength}_shots_on_goal_against"].replace(0, np.nan)
        goals_for = df[f"{strength}_goals_for"].replace(0, np.nan)
        goals_against = df[f"{strength}_goals_against"].replace(0, np.nan)

        # Offensive funnel rates
        new_cols[f"{strength}_fenwick_rate_for"] = fen_for / att_for
        new_cols[f"{strength}_block_rate_against"] = 1 - (fen_for / att_for)
        new_cols[f"{strength}_sog_fenwick_rate_for"] = sog_for / fen_for
        new_cols[f"{strength}_shooting_pct"] = goals_for / sog_for

        # Defensive funnel rates
        new_cols[f"{strength}_fenwick_rate_against"] = fen_against / att_against
        new_cols[f"{strength}_block_rate_for"] = 1 - (fen_against / att_against)
        new_cols[f"{strength}_sog_fenwick_rate_against"] = sog_against / fen_against
        new_cols[f"{strength}_save_pct_team"] = 1 - (goals_against / sog_against)

        # xG quality rates
        xgf_col = f"xgf_{strength}"
        xga_col = f"xga_{strength}"
        if xgf_col in df.columns:
            new_cols[f"{strength}_xg_per_sog_for"] = df[xgf_col] / sog_for
        if xga_col in df.columns:
            new_cols[f"{strength}_xg_per_sog_against"] = df[xga_col] / sog_against

    result_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    funnel_cols = list(new_cols.keys())
    result_df[funnel_cols] = result_df[funnel_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result_df


def add_weighted_funnel_rates(df):
    """
    Calculate weighted multi-season shot funnel rates.
    Different weights based on how mean-reverting each stat is:
    - Shooting%: most mean-reverting (40/35/25)
    - SOG/Fenwick rate: somewhat mean-reverting (45/35/20)
    - Block rate: more stable, system-driven (55/30/15)
    """
    print("  Computing weighted funnel rates...")

    SEASONS = sorted(df["season"].unique())

   # Weights derived empirically from predictability analysis:
    # SOG/Fenwick and block rate: 3yr avg most predictive → equal weight across seasons
    # Shooting%: barely predictable → shrink heavily, equal weight for stability
    # xG per SOG: similar to shooting%, needs regression to mean
    WEIGHT_CONFIGS = {
        # Shooting% - barely predictable, shrink toward mean, use all seasons equally
        "shooting_pct":              {0: 0.34, -1: 0.33, -2: 0.33},
        # SOG/Fenwick - 3yr avg best, system trait
        "sog_fenwick_rate_for":      {0: 0.34, -1: 0.33, -2: 0.33},
        "sog_fenwick_rate_against":  {0: 0.34, -1: 0.33, -2: 0.33},
        # Block rate - 3yr avg best, coaching/system trait
        "block_rate_for":            {0: 0.34, -1: 0.33, -2: 0.33},
        "block_rate_against":        {0: 0.34, -1: 0.33, -2: 0.33},
        "fenwick_rate_for":          {0: 0.34, -1: 0.33, -2: 0.33},
        "fenwick_rate_against":      {0: 0.34, -1: 0.33, -2: 0.33},
        # xG per SOG - shot quality, needs regression to mean
        "xg_per_sog_for":            {0: 0.34, -1: 0.33, -2: 0.33},
        "xg_per_sog_against":        {0: 0.34, -1: 0.33, -2: 0.33},
        # Team save% - goalie quality, somewhat persistent
        "save_pct_team":             {0: 0.34, -1: 0.33, -2: 0.33},
    }

    # Build list of stats to weight
    stats_to_weight = []
    for strength in ["ev", "pp", "sh"]:
        for stat_key in WEIGHT_CONFIGS.keys():
            col = f"{strength}_{stat_key}"
            if col in df.columns:
                stats_to_weight.append((col, stat_key))

    results = []
    for team, team_df in df.groupby("team"):
        team_df = team_df.sort_values("date").reset_index(drop=True)

        for i in range(len(team_df)):
            row = team_df.iloc[i].copy()
            current_season = row["season"]
            season_idx = list(SEASONS).index(current_season) if current_season in SEASONS else -1

            for col, stat_key in stats_to_weight:
                weights = WEIGHT_CONFIGS[stat_key]
                weighted_sum = 0.0
                weight_total = 0.0

                for offset, weight in weights.items():
                    target_season_idx = season_idx + offset
                    if target_season_idx < 0 or target_season_idx >= len(SEASONS):
                        continue
                    target_season = SEASONS[target_season_idx]
                    past_season_games = team_df[
                        (team_df["season"] == target_season) &
                        (team_df["date"] < row["date"])
                    ]
                    if len(past_season_games) >= 5 and col in past_season_games.columns:
                        # Filter out zeros (games with no data)
                        valid = past_season_games[col].replace(0, np.nan).dropna()
                        if len(valid) >= 5:
                            season_avg = valid.mean()
                            if not np.isnan(season_avg):
                                weighted_sum += season_avg * weight
                                weight_total += weight

                row[f"weighted_{col}"] = weighted_sum / weight_total if weight_total > 0 else None

            results.append(row)

    result_df = pd.DataFrame(results)
    weighted_cols = [c for c in result_df.columns if c.startswith("weighted_") and
                     any(k in c for k in WEIGHT_CONFIGS.keys())]
    print(f"  Added {len(weighted_cols)} weighted funnel rate columns")
    return result_df


def add_rolling_features(df):
    """Compute rolling averages over last 5, 10, 20 games with opponent adjustment."""
    print("  Computing rolling averages...")
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    season_avgs = df.groupby(["team", "season"])[TEAM_STATS].mean().reset_index()
    season_avgs.columns = ["team", "season"] + [f"season_avg_{c}" for c in TEAM_STATS]

    all_rows = []
    for team, team_df in df.groupby("team"):
        team_df = team_df.sort_values("date").reset_index(drop=True)

        for i in range(len(team_df)):
            row = team_df.iloc[i].copy()
            past = team_df.iloc[:i]

            for window in ROLLING_WINDOWS:
                past_w = past.tail(window)
                for stat in TEAM_STATS:
                    if stat in team_df.columns:
                        row[f"{stat}_last{window}"] = (
                            past_w[stat].mean() if len(past_w) > 0 else None
                        )

                if len(past_w) > 0:
                    opp_adjustments = []
                    for _, game in past_w.iterrows():
                        opp = game["opponent"]
                        season = game["season"]
                        opp_avg = season_avgs[
                            (season_avgs["team"] == opp) &
                            (season_avgs["season"] == season)
                        ]
                        if len(opp_avg) > 0:
                            opp_adjustments.append(opp_avg.iloc[0].to_dict())
                    if opp_adjustments:
                        opp_df = pd.DataFrame(opp_adjustments)
                        for stat in ["ev_shots_on_goal_against",
                                     "ev_shot_attempts_against", "goals_against"]:
                            opp_col = f"season_avg_{stat}"
                            if opp_col in opp_df.columns:
                                row[f"{stat}_opp_adj_last{window}"] = (
                                    row[f"{stat}_last{window}"] - opp_df[opp_col].mean()
                                )

            current_season = row["season"]
            season_past = past[past["season"] == current_season]
            for stat in TEAM_STATS:
                if stat in team_df.columns:
                    row[f"{stat}_season_avg"] = (
                        season_past[stat].mean() if len(season_past) > 0 else None
                    )

            if i > 0:
                prev_date = pd.to_datetime(team_df.iloc[i - 1]["date"])
                curr_date = pd.to_datetime(row["date"])
                row["days_rest"] = (curr_date - prev_date).days
            else:
                row["days_rest"] = 3

            row["is_back_to_back"] = int(row["days_rest"] == 1)
            all_rows.append(row)

    result = pd.DataFrame(all_rows)
    print(f"  Rolling features computed for {len(result)} rows")
    return result


def add_weighted_pp_sh_rates(df):
    """Weighted multi-season PP/SH shots per 60 (50/35/15)."""
    print("  Computing weighted PP/SH rate stats...")

    SEASONS = sorted(df["season"].unique())
   # Current season dominates for shots/goals per 60
    # (empirically: current season ≈ last30 >> multi-season for these stats)
    WEIGHTS = {0: 0.60, -1: 0.30, -2: 0.10}

    stats_to_weight = [
        "pp_shots_on_goal_for_per60", "pp_shots_on_goal_against_per60",
        "sh_shots_on_goal_for_per60", "sh_shots_on_goal_against_per60",
        "pp_fenwick_for_per60", "pp_fenwick_against_per60",
        "sh_fenwick_for_per60", "sh_fenwick_against_per60",
        "pp_shot_attempts_for_per60", "pp_shot_attempts_against_per60",
        "sh_shot_attempts_for_per60", "sh_shot_attempts_against_per60",
        "pp_goals_for_per60", "pp_goals_against_per60",
        "sh_goals_for_per60", "sh_goals_against_per60",
    ]

    results = []
    for team, team_df in df.groupby("team"):
        team_df = team_df.sort_values("date").reset_index(drop=True)

        for i in range(len(team_df)):
            row = team_df.iloc[i].copy()
            current_season = row["season"]
            season_idx = list(SEASONS).index(current_season) if current_season in SEASONS else -1

            for stat in stats_to_weight:
                if stat not in team_df.columns:
                    continue
                weighted_sum = 0.0
                weight_total = 0.0
                for offset, weight in WEIGHTS.items():
                    target_season_idx = season_idx + offset
                    if target_season_idx < 0 or target_season_idx >= len(SEASONS):
                        continue
                    target_season = SEASONS[target_season_idx]
                    past_season_games = team_df[
                        (team_df["season"] == target_season) &
                        (team_df["date"] < row["date"])
                    ]
                    if len(past_season_games) >= 5:
                        season_avg = past_season_games[stat].mean()
                        if not np.isnan(season_avg):
                            weighted_sum += season_avg * weight
                            weight_total += weight
                row[f"weighted_{stat}"] = weighted_sum / weight_total if weight_total > 0 else None

            results.append(row)

    result_df = pd.DataFrame(results)
    weighted_cols = [c for c in result_df.columns if c.startswith("weighted_")]
    print(f"  Added {len(weighted_cols)} weighted rate columns")
    return result_df


def build_matchup_features(df):
    """Combine home and away team features into a single matchup row."""
    print("  Building matchup features...")

    home = df[df["is_home"] == True].copy()
    away = df[df["is_home"] == False].copy()

    feature_cols = [c for c in df.columns if c not in
                    ["game_id", "season", "date", "team", "opponent",
                     "is_home", "went_to_ot", "won"]]

    home_renamed = home[["game_id", "season", "date", "went_to_ot", "won"] +
                        feature_cols].copy()
    home_renamed.columns = (["game_id", "season", "date", "went_to_ot", "home_won"] +
                            [f"home_{c}" for c in feature_cols])
    home_renamed["home_team"] = home["team"].values
    home_renamed["away_team"] = home["opponent"].values

    away_renamed = away[["game_id"] + feature_cols].copy()
    away_renamed.columns = ["game_id"] + [f"away_{c}" for c in feature_cols]

    matchup = home_renamed.merge(away_renamed, on="game_id", how="inner")
    matchup = matchup.sort_values("date").reset_index(drop=True)

    goalie_check = [c for c in matchup.columns if "save_pct" in c]
    if not goalie_check:
        print("  WARNING: Goalie columns missing!")
    else:
        print(f"  Goalie columns included: {len(goalie_check)}")

    weighted_funnel = [c for c in matchup.columns if "weighted_ev_" in c]
    print(f"  Weighted funnel rate columns: {len(weighted_funnel)}")

    print(f"  Matchup features built: {len(matchup)} games")
    return matchup


def main():
    print("Loading data...")
    team_df = pd.read_csv(TEAM_LOGS)
    shot_df = pd.read_csv(SHOT_XG)
    goalie_df = pd.read_csv(GOALIE_LOGS)
    print(f"  Team logs: {len(team_df)} rows")
    print(f"  Shot xG data: {len(shot_df)} rows")
    print(f"  Goalie logs: {len(goalie_df)} rows")

    team_df["date"] = pd.to_datetime(team_df["date"])

    print("\nStep 1: Aggregating xG...")
    xg_features = aggregate_xg(shot_df)

    print("\nStep 2: Merging xG into team logs...")
    team_df = team_df.merge(xg_features, on=["game_id", "team"], how="left")
    xg_cols = [c for c in xg_features.columns if c not in ["game_id", "team"]]
    team_df[xg_cols] = team_df[xg_cols].fillna(0)
    print(f"  xG columns added: {len(xg_cols)}")

    print("\nStep 3: Adding Fenwick and per-60 rates...")
    team_df = add_fenwick(team_df)
    team_df = add_per60_rates(team_df)

    print("\nStep 4: Adding PP%/PK%...")
    team_df = add_pp_pk(team_df)

    print("\nStep 4b: Adding shot funnel rates...")
    team_df = add_shot_funnel_rates(team_df)

    print("\nStep 5: Computing rolling averages...")
    team_df = add_rolling_features(team_df)

    print("\nStep 6: Computing weighted PP/SH rate stats...")
    team_df = add_weighted_pp_sh_rates(team_df)

    print("\nStep 6b: Computing weighted funnel rates...")
    team_df = add_weighted_funnel_rates(team_df)

    print("\nStep 7: Computing goalie features...")
    goalie_features = aggregate_goalie_stats(goalie_df)
    team_df = team_df.merge(goalie_features, on=["game_id", "team"], how="left")
    goalie_cols = ["save_pct_last20", "save_pct_last40",
                   "save_pct_current_season", "save_pct_career",
                   "gsax_per60_last20", "gsax_per60_last40",
                   "gsax_per60_current_season", "gsax_per60_career"]
    print(f"  Goalie cols: {[c for c in goalie_cols if c in team_df.columns]}")
    print(f"  Goalie null count: {team_df['save_pct_last20'].isnull().sum()}")

    print("\nStep 8: Building matchup features...")
    matchup_df = build_matchup_features(team_df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    matchup_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone! {len(matchup_df)} matchup rows saved to {OUTPUT_PATH}")
    print(f"Total features per matchup: {len(matchup_df.columns)}")


if __name__ == "__main__":
    main()