import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

# Import our existing parse functions
import sys
sys.path.append(os.path.dirname(__file__))
from fetch_team_game_logs import fetch_play_by_play, parse_game
from fetch_player_game_logs import fetch_player_game_log, parse_game_log
from fetch_goalie_game_logs import fetch_goalie_game_log, parse_goalie_game_log

# --- CONFIG ---
BASE_URL = "https://api-web.nhle.com/v1"
DATA_DIR = os.path.expanduser("~/nhl-props/data/raw")

TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WPG", "WSH"
]

CURRENT_SEASON = "20252026"


def get_date_range(lookback_days):
    """Get start and end dates for the update window."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def get_recent_game_ids(start_date, end_date):
    """Get all game IDs from the current season that fall in the date range."""
    print(f"  Scanning schedule for games between {start_date} and {end_date}...")
    all_game_ids = set()
    for team in TEAMS:
        try:
            url = f"{BASE_URL}/club-schedule-season/{team}/{CURRENT_SEASON}"
            response = requests.get(url)
            response.raise_for_status()
            games = response.json().get("games", [])
            for game in games:
                if game.get("gameType") != 2:
                    continue
                if game.get("gameState") not in ["OFF", "FINAL"]:
                    continue
                game_date = game.get("gameDate", "")
                if start_date <= game_date <= end_date:
                    all_game_ids.add(game.get("id"))
            time.sleep(0.3)
        except Exception as e:
            print(f"    ERROR {team}: {e}")
    return all_game_ids


def get_players_from_games(game_ids):
    """
    Use rosterSpots from play-by-play to get only players/goalies
    who actually played in the target games.
    Returns (player_dict, goalie_dict).
    Note: play-by-play data is already fetched here and reused in update_team_logs.
    """
    player_dict = {}
    goalie_dict = {}
    pbp_cache = {}  # cache so we don't fetch twice

    for game_id in game_ids:
        try:
            data = fetch_play_by_play(game_id)
            pbp_cache[game_id] = data
            for spot in data.get("rosterSpots", []):
                pid = spot.get("playerId")
                position = spot.get("positionCode")
                if not pid:
                    continue
                if position == "G":
                    goalie_dict[pid] = {
                        "player_id": pid,
                        "first_name": spot.get("firstName", {}).get("default", ""),
                        "last_name": spot.get("lastName", {}).get("default", ""),
                        "catches": None,
                        "height_in": None,
                        "weight_lbs": None,
                        "birth_date": None,
                        "birth_country": None,
                    }
                else:
                    player_dict[pid] = {
                        "player_id": pid,
                        "first_name": spot.get("firstName", {}).get("default", ""),
                        "last_name": spot.get("lastName", {}).get("default", ""),
                        "position": position,
                        "shoots": None,
                        "height_in": None,
                        "weight_lbs": None,
                        "birth_date": None,
                        "birth_country": None,
                    }
            time.sleep(0.3)
        except Exception as e:
            print(f"    ERROR fetching roster for game {game_id}: {e}")

    return player_dict, goalie_dict, pbp_cache


def update_team_logs(pbp_cache):
    """Fetch and merge new team game logs using cached play-by-play data."""
    path = os.path.join(DATA_DIR, "team_game_logs", "team_game_logs.csv")
    existing = pd.read_csv(path)
    print(f"  Existing team rows: {len(existing)}")

    new_rows = []
    for game_id, data in pbp_cache.items():
        try:
            rows = parse_game(data)
            new_rows.extend(rows)
        except Exception as e:
            print(f"    ERROR parsing game {game_id}: {e}")

    if not new_rows:
        print("  No new team data found.")
        return

    new_df = pd.DataFrame(new_rows)
    existing = existing[~existing["game_id"].isin(new_df["game_id"])]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(["team", "date"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    print(f"  Team logs updated: {len(new_rows)} new rows added, {len(combined)} total rows")


def update_player_logs(start_date, end_date, player_dict, existing_players):
    """Fetch and merge new player game logs."""
    path = os.path.join(DATA_DIR, "game_logs", "player_game_logs.csv")
    existing = existing_players
    print(f"  Existing player rows: {len(existing)}")
    print(f"  Found {len(player_dict)} unique skaters in target games")

    # Preserve bio data from existing records
    bio_cols = ["player_id", "shoots", "height_in", "weight_lbs", "birth_date", "birth_country"]
    bio_data = existing[bio_cols].drop_duplicates(subset=["player_id"])

    new_rows = []
    for i, (player_id, player_info) in enumerate(player_dict.items()):
        if i % 50 == 0:
            print(f"    {i}/{len(player_dict)} players checked...")
        try:
            game_log = fetch_player_game_log(player_id, CURRENT_SEASON)
            if not game_log:
                continue
            recent = [g for g in game_log if start_date <= g.get("gameDate", "") <= end_date]
            if recent:
                rows = parse_game_log(recent, player_id, player_info, CURRENT_SEASON)
                new_rows.extend(rows)
            time.sleep(0.3)
        except Exception as e:
            print(f"    ERROR player {player_id}: {e}")

    if not new_rows:
        print("  No new player data found.")
        return

    new_df = pd.DataFrame(new_rows)

    # Fill in bio data from existing records
    new_df = new_df.drop(columns=["shoots", "height_in", "weight_lbs", "birth_date", "birth_country"])
    new_df = new_df.merge(bio_data, on="player_id", how="left")

    overlap_keys = existing["game_id"].astype(str) + "_" + existing["player_id"].astype(str)
    new_keys = new_df["game_id"].astype(str) + "_" + new_df["player_id"].astype(str)
    existing = existing[~overlap_keys.isin(new_keys)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(["player_id", "season", "date"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    print(f"  Player logs updated: {len(new_rows)} new rows added, {len(combined)} total rows")


def update_goalie_logs(start_date, end_date, goalie_dict, existing_goalies):
    """Fetch and merge new goalie game logs."""
    path = os.path.join(DATA_DIR, "goalie_logs", "goalie_game_logs.csv")
    existing = existing_goalies
    print(f"  Existing goalie rows: {len(existing)}")
    print(f"  Found {len(goalie_dict)} unique goalies in target games")

    # Preserve bio data from existing records
    bio_cols = ["player_id", "catches", "height_in", "weight_lbs", "birth_date", "birth_country"]
    bio_data = existing[bio_cols].drop_duplicates(subset=["player_id"])

    new_rows = []
    for player_id, goalie_info in goalie_dict.items():
        try:
            game_log = fetch_goalie_game_log(player_id, CURRENT_SEASON)
            if not game_log:
                continue
            recent = [g for g in game_log if start_date <= g.get("gameDate", "") <= end_date]
            if recent:
                rows = parse_goalie_game_log(recent, player_id, goalie_info, CURRENT_SEASON)
                new_rows.extend(rows)
            time.sleep(0.3)
        except Exception as e:
            print(f"    ERROR goalie {player_id}: {e}")

    if not new_rows:
        print("  No new goalie data found.")
        return

    new_df = pd.DataFrame(new_rows)

    # Fill in bio data from existing records
    new_df = new_df.drop(columns=["catches", "height_in", "weight_lbs", "birth_date", "birth_country"])
    new_df = new_df.merge(bio_data, on="player_id", how="left")

    overlap_keys = existing["game_id"].astype(str) + "_" + existing["player_id"].astype(str)
    new_keys = new_df["game_id"].astype(str) + "_" + new_df["player_id"].astype(str)
    existing = existing[~overlap_keys.isin(new_keys)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(["player_id", "season", "date"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    print(f"  Goalie logs updated: {len(new_rows)} new rows added, {len(combined)} total rows")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Update NHL data CSVs")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to look back (default: 7)"
    )
    args = parser.parse_args()

    start_date, end_date = get_date_range(args.days)
    print(f"\nUpdating data for {start_date} to {end_date} ({args.days} day lookback)")
    print("=" * 50)

    # Step 1: get game IDs in date range
    print("\n[1/4] Scanning for recent games...")
    game_ids = get_recent_game_ids(start_date, end_date)
    print(f"  Found {len(game_ids)} games in date range")

    if not game_ids:
        print("  No games found in date range, exiting.")
        return

    # Step 2: fetch play-by-play once for all games, extract rosters
    print("\n[2/4] Fetching play-by-play and building player lists...")
    player_dict, goalie_dict, pbp_cache = get_players_from_games(game_ids)
    print(f"  {len(player_dict)} unique skaters, {len(goalie_dict)} unique goalies found")

    # Step 3: load existing CSVs once
    existing_players = pd.read_csv(os.path.join(DATA_DIR, "game_logs", "player_game_logs.csv"))
    existing_goalies = pd.read_csv(os.path.join(DATA_DIR, "goalie_logs", "goalie_game_logs.csv"))

    # Step 4: update all three files
    print("\n[3/4] Updating team game logs...")
    update_team_logs(pbp_cache)

    print("\n[4/4a] Updating player game logs...")
    update_player_logs(start_date, end_date, player_dict, existing_players)

    print("\n[4/4b] Updating goalie game logs...")
    update_goalie_logs(start_date, end_date, goalie_dict, existing_goalies)

    print("\nDone! All data updated.")


if __name__ == "__main__":
    main()