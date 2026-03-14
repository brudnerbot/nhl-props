import pandas as pd
import numpy as np
import os

# --- CONFIG ---
DATA_DIR = os.path.expanduser("~/nhl-props/data")
TEAM_LOGS = os.path.join(DATA_DIR, "raw/team_game_logs/team_game_logs.csv")
SHOT_XG = os.path.join(DATA_DIR, "processed/shot_data_with_xg.csv")
GOALIE_LOGS = os.path.join(DATA_DIR, "raw/goalie_logs/goalie_game_logs.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "processed/team_features.csv")

ROLLING_WINDOWS = [5, 10, 20]

# Stats to compute rolling averages for
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
    # Totals across all strengths
    "total_shots_on_goal_for", "total_shots_on_goal_against",
    "total_shot_attempts_for", "total_shot_attempts_against",
    "total_fenwick_for", "total_fenwick_against",
    # TOI at each strength
    # TOI at each strength
    "ev_toi", "pp_toi", "sh_toi", "en_toi",
    # Per-60 rate stats at each strength
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
]


# --- STEP 1: Aggregate xG per game per team ---
def aggregate_xg(shot_df):
    """Aggregate shot-level xG into team-game-level totals by strength."""
    print("  Aggregating xG per team per game...")

    # Exclude empty net shots from xG totals
    shot_df = shot_df[shot_df["is_empty_net"] == 0].copy()

    # Aggregate xGF (shooting team) and xGA (defending team)
    xgf = shot_df.groupby(["game_id", "shooting_team", "strength"])["xg"].sum().reset_index()
    xgf.columns = ["game_id", "team", "strength", "xgf"]

    xga = shot_df.groupby(["game_id", "defending_team", "strength"])["xg"].sum().reset_index()
    xga.columns = ["game_id", "team", "strength", "xga"]

    # Merge xGF and xGA
    xg = xgf.merge(xga, on=["game_id", "team", "strength"], how="outer").fillna(0)

    # Pivot strength into columns
    xg_pivot = xg.pivot_table(
        index=["game_id", "team"],
        columns="strength",
        values=["xgf", "xga"],
        aggfunc="sum"
    ).fillna(0)

    # Flatten column names
    xg_pivot.columns = [f"{stat}_{strength}" for stat, strength in xg_pivot.columns]
    xg_pivot = xg_pivot.reset_index()

    # Add total xGF and xGA
    xgf_cols = [c for c in xg_pivot.columns if c.startswith("xgf_")]
    xga_cols = [c for c in xg_pivot.columns if c.startswith("xga_")]
    xg_pivot["xgf_total"] = xg_pivot[xgf_cols].sum(axis=1)
    xg_pivot["xga_total"] = xg_pivot[xga_cols].sum(axis=1)

    print(f"  xG aggregated for {len(xg_pivot)} team-game rows")
    return xg_pivot


# --- STEP 2: Aggregate goalie stats per game ---
def aggregate_goalie_stats(goalie_df):
    """
    For each game, get the primary goalie's stats and compute
    weighted save% and GSAx across multiple time windows.
    """
    print("  Processing goalie stats...")

    goalie_df = goalie_df.copy()
    goalie_df["date"] = pd.to_datetime(goalie_df["date"])
    goalie_df = goalie_df.sort_values(["player_id", "date"]).reset_index(drop=True)

    # Calculate rolling goalie metrics
    results = []
    for player_id, g in goalie_df.groupby("player_id"):
        g = g.copy().reset_index(drop=True)

        # GSAx = saves - (shots_against * league_avg_save_pct)
        # We use save_pct as proxy since we don't have xGA per goalie yet
        # Will be replaced with proper GSAx once goalie xG is built
        g["gsax"] = g["saves"] - (g["shots_against"] * 0.906)  # 0.906 = NHL avg save pct

        for i in range(len(g)):
            row = g.iloc[i].copy()
            past = g.iloc[:i]  # all games before this one

            # Rolling windows
            for window in [20, 40]:
                past_w = past.tail(window)
                if len(past_w) > 0:
                    total_shots = past_w["shots_against"].sum()
                    total_saves = past_w["saves"].sum()
                    row[f"save_pct_last{window}"] = total_saves / total_shots if total_shots > 0 else None
                    row[f"gsax_per60_last{window}"] = (
                        past_w["gsax"].sum() /
                        (past_w["toi"].sum() / 60)
                        if past_w["toi"].sum() > 0 else None
                    )
                else:
                    row[f"save_pct_last{window}"] = None
                    row[f"gsax_per60_last{window}"] = None

            # Current season save%
            current_season = row["season"]
            past_season = past[past["season"] == current_season]
            if len(past_season) > 0:
                total_shots = past_season["shots_against"].sum()
                total_saves = past_season["saves"].sum()
                row["save_pct_current_season"] = total_saves / total_shots if total_shots > 0 else None
                row["gsax_per60_current_season"] = (
                    past_season["gsax"].sum() /
                    (past_season["toi"].sum() / 60)
                    if past_season["toi"].sum() > 0 else None
                )
            else:
                row["save_pct_current_season"] = None
                row["gsax_per60_current_season"] = None

            # Multi-season save% (all available history)
            if len(past) > 0:
                total_shots = past["shots_against"].sum()
                total_saves = past["saves"].sum()
                row["save_pct_career"] = total_saves / total_shots if total_shots > 0 else None
                row["gsax_per60_career"] = (
                    past["gsax"].sum() /
                    (past["toi"].sum() / 60)
                    if past["toi"].sum() > 0 else None
                )
            else:
                row["save_pct_career"] = None
                row["gsax_per60_career"] = None

            results.append(row)

    goalie_features = pd.DataFrame(results)

    # For each game, get the primary goalie (most TOI)
    primary_goalies = goalie_features.sort_values("toi", ascending=False).drop_duplicates(
        subset=["game_id", "team"]
    )[["game_id", "team", "save_pct_last20", "save_pct_last40",
       "save_pct_current_season", "save_pct_career",
       "gsax_per60_last20", "gsax_per60_last40",
       "gsax_per60_current_season", "gsax_per60_career"]]

    print(f"  Goalie features computed for {len(primary_goalies)} team-game rows")
    return primary_goalies


# --- STEP 3: Calculate Fenwick ---
def add_fenwick(df):
    """Fenwick = shot attempts - blocked shots (unblocked shot attempts)."""
    for strength in ["ev", "pp", "sh"]:
        df[f"{strength}_fenwick_for"] = (
            df[f"{strength}_shot_attempts_for"] - df[f"{strength}_blocked_shots_against"]
        )
        df[f"{strength}_fenwick_against"] = (
            df[f"{strength}_shot_attempts_against"] - df[f"{strength}_blocked_shots_for"]
        )

    # Total shots (EV + PP + SH)
    df["total_shots_on_goal_for"] = (
        df["ev_shots_on_goal_for"] +
        df["pp_shots_on_goal_for"] +
        df["sh_shots_on_goal_for"]
    )
    df["total_shots_on_goal_against"] = (
        df["ev_shots_on_goal_against"] +
        df["pp_shots_on_goal_against"] +
        df["sh_shots_on_goal_against"]
    )
    df["total_shot_attempts_for"] = (
        df["ev_shot_attempts_for"] +
        df["pp_shot_attempts_for"] +
        df["sh_shot_attempts_for"]
    )
    df["total_shot_attempts_against"] = (
        df["ev_shot_attempts_against"] +
        df["pp_shot_attempts_against"] +
        df["sh_shot_attempts_against"]
    )
    df["total_fenwick_for"] = (
        df["ev_fenwick_for"] +
        df["pp_fenwick_for"] +
        df["sh_fenwick_for"]
    )
    df["total_fenwick_against"] = (
        df["ev_fenwick_against"] +
        df["pp_fenwick_against"] +
        df["sh_fenwick_against"]
    )
    return df

def add_per60_rates(df):
    """
    Calculate per-60 rate stats at each strength.
    These are normalized by TOI so they're not circular with TOI predictions.
    """
    for strength in ["ev", "pp", "sh"]:
        toi_col = f"{strength}_toi"
        toi_60 = df[toi_col] / 60  # convert minutes to per-60 denominator

        # Avoid division by zero
        toi_60 = toi_60.replace(0, np.nan)

        for stat in ["shots_on_goal_for", "shots_on_goal_against",
                     "shot_attempts_for", "shot_attempts_against",
                     "goals_for", "goals_against",
                     "hits_for", "hits_against",
                     "giveaways", "takeaways"]:
            col = f"{strength}_{stat}"
            if col in df.columns:
                df[f"{strength}_{stat}_per60"] = df[col] / toi_60

        # Fenwick per 60
        if f"{strength}_fenwick_for" in df.columns:
            df[f"{strength}_fenwick_for_per60"] = df[f"{strength}_fenwick_for"] / toi_60
            df[f"{strength}_fenwick_against_per60"] = df[f"{strength}_fenwick_against"] / toi_60

    # Fill any remaining nulls from zero TOI games
    per60_cols = [c for c in df.columns if c.endswith("_per60")]
    df[per60_cols] = df[per60_cols].fillna(0)

    return df

# --- STEP 4: Calculate PP% and PK% ---
def add_pp_pk(df):
    """Add power play % and penalty kill %."""
    df["pp_pct"] = df["pp_goals_for"] / df["pp_penalties_drawn"].replace(0, np.nan)
    df["pk_pct"] = 1 - (df["pp_goals_against"] / df["sh_penalties_taken"].replace(0, np.nan))
    df["pp_pct"] = df["pp_pct"].fillna(0)
    df["pk_pct"] = df["pk_pct"].fillna(1)
    return df


# --- STEP 5: Rolling averages with opponent adjustment ---
def add_rolling_features(df):
    """
    For each team, compute rolling averages over last 5, 10, 20 games.
    Also compute opponent-adjusted versions by comparing to opponent's
    season average against all other teams.
    """
    print("  Computing rolling averages...")
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    # Compute season averages per team (used for opponent adjustment)
    season_avgs = df.groupby(["team", "season"])[TEAM_STATS].mean().reset_index()
    season_avgs.columns = ["team", "season"] + [f"season_avg_{c}" for c in TEAM_STATS]

    all_rows = []
    for team, team_df in df.groupby("team"):
        team_df = team_df.sort_values("date").reset_index(drop=True)

        for i in range(len(team_df)):
            row = team_df.iloc[i].copy()
            past = team_df.iloc[:i]

            # Rolling windows
            for window in ROLLING_WINDOWS:
                past_w = past.tail(window)
                for stat in TEAM_STATS:
                    col = f"{stat}_last{window}"
                    row[col] = past_w[stat].mean() if len(past_w) > 0 else None

                # Opponent adjustment
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
                        for stat in ["ev_shots_on_goal_against", "ev_shot_attempts_against",
                                     "goals_against"]:
                            opp_col = f"season_avg_{stat}"
                            if opp_col in opp_df.columns:
                                row[f"{stat}_opp_adj_last{window}"] = (
                                    row[f"{stat}_last{window}"] -
                                    opp_df[opp_col].mean()
                                )

            # Current season average
            current_season = row["season"]
            season_past = past[past["season"] == current_season]
            for stat in TEAM_STATS:
                row[f"{stat}_season_avg"] = (
                    season_past[stat].mean() if len(season_past) > 0 else None
                )

            # Rest days
            if i > 0:
                prev_date = pd.to_datetime(team_df.iloc[i - 1]["date"])
                curr_date = pd.to_datetime(row["date"])
                row["days_rest"] = (curr_date - prev_date).days
            else:
                row["days_rest"] = 3  # default for first game

            row["is_back_to_back"] = int(row["days_rest"] == 1)

            all_rows.append(row)

    result = pd.DataFrame(all_rows)
    print(f"  Rolling features computed for {len(result)} rows")
    return result


# --- STEP 6: Build matchup features ---
def build_matchup_features(df):
    """
    Combine home and away team features into a single matchup row.
    Each row represents one game with both teams' features.
    """
    print("  Building matchup features...")

    home = df[df["is_home"] == True].copy()
    away = df[df["is_home"] == False].copy()

    # Rename columns with home_ and away_ prefix
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

    # Step 1: aggregate xG
    print("\nStep 1: Aggregating xG...")
    xg_features = aggregate_xg(shot_df)

    # Step 2: merge xG into team logs
    print("\nStep 2: Merging xG into team logs...")
    team_df = team_df.merge(xg_features, on=["game_id", "team"], how="left")
    xg_cols = [c for c in xg_features.columns if c not in ["game_id", "team"]]
    team_df[xg_cols] = team_df[xg_cols].fillna(0)
    print(f"  xG columns added: {xg_cols}")

    # Step 3: add Fenwick and per-60 rates
    print("\nStep 3: Adding Fenwick and per-60 rates...")
    team_df = add_fenwick(team_df)
    team_df = add_per60_rates(team_df)

    # Step 4: add PP%/PK%
    print("\nStep 4: Adding PP%/PK%...")
    team_df = add_pp_pk(team_df)

    # Step 5: rolling averages with opponent adjustment
    print("\nStep 5: Computing rolling averages...")
    team_df = add_rolling_features(team_df)

    # Step 6: goalie features
    print("\nStep 6: Computing goalie features...")
    goalie_features = aggregate_goalie_stats(goalie_df)
    team_df = team_df.merge(goalie_features, on=["game_id", "team"], how="left")

    # Step 7: build matchup features
    print("\nStep 7: Building matchup features...")
    matchup_df = build_matchup_features(team_df)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    matchup_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone! {len(matchup_df)} matchup rows saved to {OUTPUT_PATH}")
    print(f"Total features per matchup: {len(matchup_df.columns)}")


if __name__ == "__main__":
    main()