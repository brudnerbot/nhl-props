import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

# Import our existing parse functions
import sys
sys.path.append(os.path.dirname(__file__))
from fetch_team_game_logs import fetch_team_schedule, fetch_play_by_play, parse_game
from fetch_player_game_logs import fetch_roster, fetch_player_game_log, parse_game_log
from fetch_goalie_game_logs import fetch_goalie_roster, fetch_goalie_game_log, parse_goalie_game_log

# --- CONFIG ---
BASE_URL = "https://api-web.nhle.com/v1"
DATA_DIR = os.path.expanduser("~/nhl-props/data/raw")
LOOKBACK_DAYS = 7

TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WPG", "WSH"
]

CURRENT_SEASON = "20252026"

def get_date_range():
    """Get start and end dates for the update window."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def get_recent_game_ids(start_date, end_date):
    """Get all game IDs from the current season that fall in the date range."""
    print(f"  Scanning schedule for games between {start_date} and {end_date}...")
    all_game_ids = set()
    for team in TEAMS:
        try:
            ids_with_dates = []
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

def update_team_logs(game_ids):
    """Fetch and merge new team game logs."""
    path = os.path.join(DATA_DIR, "team_game_logs", "team_game_logs.csv")
    existing = pd.read_csv(path)
    print(f"  Existing team rows: {len(existing)}")

    new_rows = []
    for i, game_id in enumerate(sorted(game_ids)):
        try:
            data = fetch_play_by_play(game_id)
            rows = parse_game(data)
            new_rows.extend(rows)
            time.sleep(0.3)
        except Exception as e:
            print(f"    ERROR game {game_id}: {e}")

    if not new_rows:
        print("  No new team data found.")
        return

    new_df = pd.DataFrame(new_rows)

    # Remove any overlap then append
    existing = existing[~existing["game_id"].isin(new_df["game_id"])]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(["team", "date"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    print(f"  Team logs updated: {len(new_rows)} new rows added, {len(combined)} total rows")

def update_player_logs(start_date, end_date):
    """Fetch and merge new player game logs."""
    path = os.path.join(DATA_DIR, "game_logs", "player_game_logs.csv")
    existing = pd.read_csv(path)
    print(f"  Existing player rows: {len(existing)}")

    # Build current player list
    player_dict = {}
    for team in TEAMS:
        try:
            players = fetch_roster(team, CURRENT_SEASON)
            for p in players:
                player_dict[p["player_id"]] = p
            time.sleep(0.3)
        except Exception as e:
            print(f"    ERROR roster {team}: {e}")

    print(f"  Found {len(player_dict)} active skaters")

    new_rows = []
    for i, (player_id, player_info) in enumerate(player_dict.items()):
        if i % 50 == 0:
            print(f"    {i}/{len(player_dict)} players checked...")
        try:
            game_log = fetch_player_game_log(player_id, CURRENT_SEASON)
            if not game_log:
                continue
            # Filter to date range only
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

    # Remove overlap then append
    overlap_keys = existing["game_id"].astype(str) + "_" + existing["player_id"].astype(str)
    new_keys = new_df["game_id"].astype(str) + "_" + new_df["player_id"].astype(str)
    existing = existing[~overlap_keys.isin(new_keys)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(["player_id", "season", "date"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    print(f"  Player logs updated: {len(new_rows)} new rows added, {len(combined)} total rows")

def update_goalie_logs(start_date, end_date):
    """Fetch and merge new goalie game logs."""
    path = os.path.join(DATA_DIR, "goalie_logs", "goalie_game_logs.csv")
    existing = pd.read_csv(path)
    print(f"  Existing goalie rows: {len(existing)}")

    # Build current goalie list
    goalie_dict = {}
    for team in TEAMS:
        try:
            goalies = fetch_goalie_roster(team, CURRENT_SEASON)
            for g in goalies:
                goalie_dict[g["player_id"]] = g
            time.sleep(0.3)
        except Exception as e:
            print(f"    ERROR roster {team}: {e}")

    print(f"  Found {len(goalie_dict)} active goalies")

    new_rows = []
    for player_id, goalie_info in goalie_dict.items():
        try:
            game_log = fetch_goalie_game_log(player_id, CURRENT_SEASON)
            if not game_log:
                continue
            # Filter to date range only
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

    # Remove overlap then append
    overlap_keys = existing["game_id"].astype(str) + "_" + existing["player_id"].astype(str)
    new_keys = new_df["game_id"].astype(str) + "_" + new_df["player_id"].astype(str)
    existing = existing[~overlap_keys.isin(new_keys)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(["player_id", "season", "date"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    print(f"  Goalie logs updated: {len(new_rows)} new rows added, {len(combined)} total rows")

def main():
    start_date, end_date = get_date_range()
    print(f"\nUpdating data for {start_date} to {end_date}")
    print("=" * 50)

    print("\n[1/3] Updating team game logs...")
    game_ids = get_recent_game_ids(start_date, end_date)
    print(f"  Found {len(game_ids)} games in date range")
    update_team_logs(game_ids)

    print("\n[2/3] Updating player game logs...")
    update_player_logs(start_date, end_date)

    print("\n[3/3] Updating goalie game logs...")
    update_goalie_logs(start_date, end_date)

    print("\nDone! All data updated.")

if __name__ == "__main__":
    main()