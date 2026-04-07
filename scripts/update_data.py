"""
update_data.py

Incrementally updates all NHL data files for recent games.
Reuses play-by-play fetch across all data types to minimize API calls.

Usage:
    python scripts/update_data.py          # default 7 days
    python scripts/update_data.py --days 14
"""

import argparse
import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.dirname(__file__))
from fetch_team_game_logs import fetch_play_by_play, parse_game
from fetch_player_game_logs import fetch_player_game_log, parse_game_log
from fetch_goalie_game_logs import fetch_goalie_game_log, parse_goalie_game_log
from fetch_shot_data import parse_shots
from fetch_player_pbp_stats import process_game as process_pbp_game

BASE_URL       = "https://api-web.nhle.com/v1"
DATA_DIR       = os.path.expanduser("~/nhl-props/data/raw")
CURRENT_SEASON = "20252026"

TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WPG", "WSH"
]


def get_date_range(lookback_days):
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def get_recent_game_ids(start_date, end_date):
    print(f"  Scanning schedule for games between {start_date} and {end_date}...")
    all_game_ids = set()
    for team in TEAMS:
        try:
            url      = f"{BASE_URL}/club-schedule-season/{team}/{CURRENT_SEASON}"
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
    player_dict = {}
    goalie_dict = {}
    pbp_cache   = {}
    game_meta   = {}

    for game_id in sorted(game_ids):
        try:
            data = fetch_play_by_play(game_id)
            pbp_cache[game_id] = data
            home = data.get("homeTeam", {})
            away = data.get("awayTeam", {})
            game_meta[game_id] = {
                "home_team": home.get("abbrev", ""),
                "away_team": away.get("abbrev", ""),
                "date":      data.get("gameDate", ""),
                "season":    data.get("season", int(CURRENT_SEASON)),
            }
            for spot in data.get("rosterSpots", []):
                pid      = spot.get("playerId")
                position = spot.get("positionCode")
                if not pid:
                    continue
                info = {
                    "player_id":     pid,
                    "first_name":    spot.get("firstName", {}).get("default", ""),
                    "last_name":     spot.get("lastName", {}).get("default", ""),
                    "catches":       None,
                    "height_in":     None,
                    "weight_lbs":    None,
                    "birth_date":    None,
                    "birth_country": None,
                }
                if position == "G":
                    goalie_dict[pid] = info
                else:
                    info["position"] = position
                    info["shoots"]   = None
                    player_dict[pid] = info
            time.sleep(0.3)
        except Exception as e:
            print(f"    ERROR fetching PBP for game {game_id}: {e}")

    return player_dict, goalie_dict, pbp_cache, game_meta


def update_team_logs(pbp_cache):
    path         = os.path.join(DATA_DIR, "team_game_logs", "team_game_logs.csv")
    existing     = pd.read_csv(path)
    existing_ids = set(existing["game_id"].unique())
    print(f"  Existing team rows: {len(existing)}")

    new_rows = []
    for game_id, data in pbp_cache.items():
        if game_id in existing_ids:
            continue
        try:
            new_rows.extend(parse_game(data))
        except Exception as e:
            print(f"    ERROR parsing team game {game_id}: {e}")

    if not new_rows:
        print("  No new team data.")
        return

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.sort_values(["team", "date"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    print(f"  Team logs: +{len(new_rows)} rows ({len(combined)} total)")


def update_shot_data(pbp_cache, game_meta):
    path = os.path.join(DATA_DIR, "shot_data", "shot_data.csv")
    existing_ids = set()
    if os.path.exists(path):
        existing_ids = set(pd.read_csv(path, usecols=["game_id"])["game_id"].unique())
    print(f"  Existing shot games: {len(existing_ids)}")

    new_rows = []
    for game_id, data in pbp_cache.items():
        if game_id in existing_ids:
            continue
        try:
            new_rows.extend(parse_shots(data))
        except Exception as e:
            print(f"    ERROR parsing shots game {game_id}: {e}")

    if not new_rows:
        print("  No new shot data.")
        return

    df_new = pd.DataFrame(new_rows)
    if os.path.exists(path):
        df_new = pd.concat([pd.read_csv(path), df_new], ignore_index=True)
    df_new = df_new.sort_values(["game_id", "period", "time_seconds"]).reset_index(drop=True)
    df_new.to_csv(path, index=False)
    print(f"  Shot data: +{len(new_rows)} rows ({len(df_new)} total)")


def update_pbp_stats(pbp_cache, game_meta):
    path = os.path.join(DATA_DIR, "player_pbp_stats", "player_pbp_stats.csv")
    existing_ids = set()
    if os.path.exists(path):
        existing_ids = set(pd.read_csv(path, usecols=["game_id"])["game_id"].unique())
    print(f"  Existing PBP games: {len(existing_ids)}")

    new_rows = []
    for game_id, data in pbp_cache.items():
        if game_id in existing_ids:
            continue
        meta = game_meta.get(game_id, {})
        try:
            rows = process_pbp_game(
                int(game_id),
                int(meta.get("season", CURRENT_SEASON)),
                str(meta.get("date", "")),
                str(meta.get("home_team", "")),
                str(meta.get("away_team", "")),
            )
            new_rows.extend(rows)
        except Exception as e:
            print(f"    ERROR parsing PBP stats game {game_id}: {e}")

    if not new_rows:
        print("  No new PBP stats.")
        return

    df_new = pd.DataFrame(new_rows)
    if os.path.exists(path):
        df_new = pd.concat([pd.read_csv(path), df_new], ignore_index=True)
    df_new.drop_duplicates(["game_id", "player_id"]).to_csv(path, index=False)
    print(f"  PBP stats: +{len(new_rows)} rows ({len(df_new)} total)")


def update_player_logs(start_date, end_date, player_dict, existing_players):
    path     = os.path.join(DATA_DIR, "game_logs", "player_game_logs.csv")
    existing = existing_players
    print(f"  Existing player rows: {len(existing)}")
    print(f"  Found {len(player_dict)} unique skaters in target games")

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
            recent = [g for g in game_log
                      if start_date <= g.get("gameDate", "") <= end_date]
            if recent:
                new_rows.extend(parse_game_log(recent, player_id,
                                               player_info, CURRENT_SEASON))
            time.sleep(0.3)
        except Exception as e:
            print(f"    ERROR player {player_id}: {e}")

    if not new_rows:
        print("  No new player data.")
        return

    df_new = df_new = pd.DataFrame(new_rows)
    df_new = df_new.drop(columns=["shoots", "height_in", "weight_lbs",
                                   "birth_date", "birth_country"], errors="ignore")
    df_new = df_new.merge(bio_data, on="player_id", how="left")

    overlap  = existing["game_id"].astype(str) + "_" + existing["player_id"].astype(str)
    new_keys = df_new["game_id"].astype(str)   + "_" + df_new["player_id"].astype(str)
    combined = pd.concat([existing[~overlap.isin(new_keys)], df_new], ignore_index=True)
    combined = combined.sort_values(["player_id", "season", "date"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    print(f"  Player logs: +{len(new_rows)} rows ({len(combined)} total)")


def update_goalie_logs(start_date, end_date, goalie_dict, existing_goalies):
    path     = os.path.join(DATA_DIR, "goalie_game_logs", "goalie_game_logs.csv")
    existing = existing_goalies
    print(f"  Existing goalie rows: {len(existing)}")
    print(f"  Found {len(goalie_dict)} unique goalies in target games")

    bio_cols = ["player_id", "catches", "height_in", "weight_lbs", "birth_date", "birth_country"]
    bio_data = existing[bio_cols].drop_duplicates(subset=["player_id"])

    new_rows = []
    for player_id, goalie_info in goalie_dict.items():
        try:
            game_log = fetch_goalie_game_log(player_id, CURRENT_SEASON)
            if not game_log:
                continue
            recent = [g for g in game_log
                      if start_date <= g.get("gameDate", "") <= end_date]
            if recent:
                new_rows.extend(parse_goalie_game_log(recent, player_id,
                                                       goalie_info, CURRENT_SEASON))
            time.sleep(0.3)
        except Exception as e:
            print(f"    ERROR goalie {player_id}: {e}")

    if not new_rows:
        print("  No new goalie data.")
        return

    df_new = pd.DataFrame(new_rows)
    df_new = df_new.drop(columns=["catches", "height_in", "weight_lbs",
                                   "birth_date", "birth_country"], errors="ignore")
    df_new = df_new.merge(bio_data, on="player_id", how="left")

    overlap  = existing["game_id"].astype(str) + "_" + existing["player_id"].astype(str)
    new_keys = df_new["game_id"].astype(str)   + "_" + df_new["player_id"].astype(str)
    combined = pd.concat([existing[~overlap.isin(new_keys)], df_new], ignore_index=True)
    combined = combined.sort_values(["player_id", "season", "date"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    print(f"  Goalie logs: +{len(new_rows)} rows ({len(combined)} total)")


def main():
    parser = argparse.ArgumentParser(description="Update NHL data CSVs incrementally")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days to look back (default: 7)")
    args = parser.parse_args()

    start_date, end_date = get_date_range(args.days)
    print(f"\nUpdating data: {start_date} to {end_date} ({args.days} day lookback)")
    print("=" * 55)

    print("\n[1/6] Scanning for recent games...")
    game_ids = get_recent_game_ids(start_date, end_date)
    print(f"  Found {len(game_ids)} games in date range")
    if not game_ids:
        print("  No games found, exiting.")
        return

    print("\n[2/6] Fetching play-by-play (shared across all updates)...")
    player_dict, goalie_dict, pbp_cache, game_meta = get_players_from_games(game_ids)
    print(f"  {len(pbp_cache)} games fetched, "
          f"{len(player_dict)} skaters, {len(goalie_dict)} goalies")

    print("\n[3/6] Updating team game logs...")
    update_team_logs(pbp_cache)

    print("\n[4/6] Updating shot data...")
    update_shot_data(pbp_cache, game_meta)

    print("\n[5/6] Updating player PBP stats...")
    update_pbp_stats(pbp_cache, game_meta)

    print("\n[6a/6] Updating player game logs...")
    existing_players = pd.read_csv(
        os.path.join(DATA_DIR, "game_logs", "player_game_logs.csv"))
    update_player_logs(start_date, end_date, player_dict, existing_players)

    print("\n[6b/6] Updating goalie game logs...")
    existing_goalies = pd.read_csv(
        os.path.join(DATA_DIR, "goalie_game_logs", "goalie_game_logs.csv"))
    update_goalie_logs(start_date, end_date, goalie_dict, existing_goalies)

    print("\nDone! All data updated.")
    print("\nNext steps:")
    print("  python scripts/build_zone_features.py")
    print("  python scripts/process_player_features.py")
    print("  python scripts/fetch_daily_lineups.py")
    print("  python scripts/predict_player_props.py HOME AWAY")


if __name__ == "__main__":
    main()
