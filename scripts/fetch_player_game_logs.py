import requests
import pandas as pd
import time
import os

# --- CONFIG ---
SEASONS = [
    "20152016", "20162017", "20172018", "20182019",
    "20192020", "20202021", "20212022", "20222023",
    "20232024", "20242025", "20252026"
]
OUTPUT_DIR = os.path.expanduser("~/nhl-props/data/raw/game_logs")
BASE_URL = "https://api-web.nhle.com/v1"

TEAMS = [
    "ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL",
    "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR",
    "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN",
    "VGK", "WPG", "WSH"
]

def fetch_roster(team, season):
    """Get all players on a team's roster for a given season."""
    url = f"{BASE_URL}/roster/{team}/{season}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    players = []
    for position_group in ["forwards", "defensemen"]:
        for player in data.get(position_group, []):
            players.append({
                "player_id": player.get("id"),
                "first_name": player.get("firstName", {}).get("default", ""),
                "last_name": player.get("lastName", {}).get("default", ""),
                "position": player.get("positionCode"),
                "shoots": player.get("shootsCatches"),
                "height_in": player.get("heightInInches"),
                "weight_lbs": player.get("weightInPounds"),
                "birth_date": player.get("birthDate"),
                "birth_country": player.get("birthCountry"),
            })
    return players

def fetch_player_game_log(player_id, season):
    """Get game-by-game log for a player in a given season."""
    url = f"{BASE_URL}/player/{player_id}/game-log/{season}/2"
    response = requests.get(url)
    response.raise_for_status()
    return response.json().get("gameLog", [])

def parse_game_log(game_log, player_id, player_info, season):
    """Parse raw game log into rows."""
    rows = []
    for game in game_log:
        # Parse TOI from "MM:SS" to float minutes
        toi_str = game.get("toi", "0:00")
        try:
            toi_parts = toi_str.split(":")
            toi_minutes = int(toi_parts[0]) + int(toi_parts[1]) / 60
        except:
            toi_minutes = None

        row = {
            "player_id": player_id,
            "first_name": player_info.get("first_name"),
            "last_name": player_info.get("last_name"),
            "position": player_info.get("position"),
            "shoots": player_info.get("shoots"),
            "height_in": player_info.get("height_in"),
            "weight_lbs": player_info.get("weight_lbs"),
            "birth_date": player_info.get("birth_date"),
            "birth_country": player_info.get("birth_country"),
            "season": season,
            "game_id": game.get("gameId"),
            "date": game.get("gameDate"),
            "team": game.get("teamAbbrev"),
            "opponent": game.get("opponentAbbrev"),
            "is_home": game.get("homeRoadFlag") == "H",
            "toi": toi_minutes,
            "goals": game.get("goals", 0),
            "assists": game.get("assists", 0),
            "points": game.get("points", 0),
            "plus_minus": game.get("plusMinus", 0),
            "shots": game.get("shots", 0),
            "pim": game.get("pim", 0),
            "shifts": game.get("shifts", 0),
            "pp_goals": game.get("powerPlayGoals", 0),
            "pp_points": game.get("powerPlayPoints", 0),
            "sh_goals": game.get("shorthandedGoals", 0),
            "sh_points": game.get("shorthandedPoints", 0),
            "gw_goals": game.get("gameWinningGoals", 0),
            "ot_goals": game.get("otGoals", 0),
        }
        rows.append(row)
    return rows

def main(test_mode=False):
    # Step 1: build master player list
    print("Step 1: Building player list...")
    player_dict = {}  # player_id -> player_info

    seasons_for_roster = ["20232024", "20242025", "20252026"] if not test_mode else ["20232024"]
    teams_for_roster = TEAMS if not test_mode else ["EDM"]

    for season in seasons_for_roster:
        print(f"  Season {season}:")
        for team in teams_for_roster:
            try:
                players = fetch_roster(team, season)
                for p in players:
                    player_dict[p["player_id"]] = p
                time.sleep(0.3)
            except Exception as e:
                print(f"    ERROR {team} {season}: {e}")
        print(f"    Unique players so far: {len(player_dict)}")

    print(f"\nTotal unique skaters found: {len(player_dict)}")

    # Step 2: fetch game logs for each player
    print("\nStep 2: Fetching game logs...")
    all_rows = []
    seasons_to_fetch = SEASONS if not test_mode else ["20232024"]
    player_ids = list(player_dict.keys()) if not test_mode else list(player_dict.keys())[:3]

    for i, player_id in enumerate(player_ids):
        player_info = player_dict[player_id]
        if i % 50 == 0:
            print(f"  {i}/{len(player_ids)} players fetched...")
        for season in seasons_to_fetch:
            try:
                game_log = fetch_player_game_log(player_id, season)
                if game_log:
                    rows = parse_game_log(game_log, player_id, player_info, season)
                    all_rows.extend(rows)
                time.sleep(0.3)
            except Exception as e:
                print(f"    ERROR player {player_id} {season}: {e}")

    # Step 3: save
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["player_id", "season", "date"]).reset_index(drop=True)

    if test_mode:
        print("\n--- TEST OUTPUT ---")
        print(df[["first_name", "last_name", "season", "date", "team", "goals", "assists", "points", "shots", "toi"]].to_string())
    else:
        output_path = os.path.join(OUTPUT_DIR, "player_game_logs.csv")
        df.to_csv(output_path, index=False)
        print(f"\nDone! {len(df)} rows saved to {output_path}")

if __name__ == "__main__":
    import sys
    test = "--test" in sys.argv
    main(test_mode=test)