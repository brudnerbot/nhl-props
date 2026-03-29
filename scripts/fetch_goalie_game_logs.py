import requests
import pandas as pd
import time
import os

# --- CONFIG ---
SEASONS = ["20152016", "20162017", "20172018", "20182019", "20192020", "20202021", "20212022", "20222023", "20232024", "20242025", "20252026"]
OUTPUT_DIR = os.path.expanduser("~/nhl-props/data/raw/goalie_game_logs")
BASE_URL = "https://api-web.nhle.com/v1"

TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WPG", "WSH"
]

def fetch_goalie_roster(team, season):
    """Get all goalies on a team's roster for a given season."""
    url = f"{BASE_URL}/roster/{team}/{season}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    goalies = []
    for goalie in data.get("goalies", []):
        goalies.append({
            "player_id": goalie.get("id"),
            "first_name": goalie.get("firstName", {}).get("default", ""),
            "last_name": goalie.get("lastName", {}).get("default", ""),
            "catches": goalie.get("shootsCatches"),
            "height_in": goalie.get("heightInInches"),
            "weight_lbs": goalie.get("weightInPounds"),
            "birth_date": goalie.get("birthDate"),
            "birth_country": goalie.get("birthCountry"),
        })
    return goalies

def fetch_goalie_game_log(player_id, season):
    """Get game-by-game log for a goalie in a given season."""
    url = f"{BASE_URL}/player/{player_id}/game-log/{season}/2"
    response = requests.get(url)
    response.raise_for_status()
    return response.json().get("gameLog", [])

def parse_goalie_game_log(game_log, player_id, goalie_info, season):
    """Parse raw goalie game log into rows."""
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
            "first_name": goalie_info.get("first_name"),
            "last_name": goalie_info.get("last_name"),
            "catches": goalie_info.get("catches"),
            "height_in": goalie_info.get("height_in"),
            "weight_lbs": goalie_info.get("weight_lbs"),
            "birth_date": goalie_info.get("birth_date"),
            "birth_country": goalie_info.get("birth_country"),
            "season": season,
            "game_id": game.get("gameId"),
            "date": game.get("gameDate"),
            "team": game.get("teamAbbrev"),
            "opponent": game.get("opponentAbbrev"),
            "is_home": game.get("homeRoadFlag") == "H",
            "toi": toi_minutes,
            "games_started": game.get("gamesStarted", 0),
            "decision": game.get("decision", None),
            "shots_against": game.get("shotsAgainst", 0),
            "goals_against": game.get("goalsAgainst", 0),
            "saves": game.get("shotsAgainst", 0) - game.get("goalsAgainst", 0),
            "save_pct": game.get("savePctg", None),
            "shutouts": game.get("shutouts", 0),
            "goals": game.get("goals", 0),
            "assists": game.get("assists", 0),
            "pim": game.get("pim", 0),
            "ot_loss": 1 if game.get("decision") == "O" else 0,
        }
        rows.append(row)
    return rows

def main(test_mode=False):
    # Step 1: build master goalie list
    print("Step 1: Building goalie list...")
    goalie_dict = {}  # player_id -> goalie_info

    seasons_for_roster = SEASONS if not test_mode else ["20232024"]
    teams_for_roster = TEAMS if not test_mode else ["EDM", "TOR", "BOS"]

    for season in seasons_for_roster:
        print(f"  Season {season}:")
        for team in teams_for_roster:
            try:
                goalies = fetch_goalie_roster(team, season)
                for g in goalies:
                    goalie_dict[g["player_id"]] = g
                time.sleep(0.3)
            except Exception as e:
                print(f"    ERROR {team} {season}: {e}")
        print(f"    Unique goalies so far: {len(goalie_dict)}")

    print(f"\nTotal unique goalies found: {len(goalie_dict)}")

    # Step 2: fetch game logs for each goalie
    print("\nStep 2: Fetching goalie game logs...")
    all_rows = []
    seasons_to_fetch = SEASONS if not test_mode else ["20232024"]
    goalie_ids = list(goalie_dict.keys()) if not test_mode else list(goalie_dict.keys())[:3]

    for i, player_id in enumerate(goalie_ids):
        goalie_info = goalie_dict[player_id]
        if i % 20 == 0:
            print(f"  {i}/{len(goalie_ids)} goalies fetched...")
        for season in seasons_to_fetch:
            try:
                game_log = fetch_goalie_game_log(player_id, season)
                if game_log:
                    rows = parse_goalie_game_log(game_log, player_id, goalie_info, season)
                    all_rows.extend(rows)
                time.sleep(0.3)
            except Exception as e:
                print(f"    ERROR goalie {player_id} {season}: {e}")

    # Step 3: save
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["player_id", "season", "date"]).reset_index(drop=True)

    if test_mode:
        print("\n--- TEST OUTPUT ---")
        print(df[["first_name", "last_name", "season", "date", "team", "decision", "shots_against", "goals_against", "saves", "save_pct", "toi"]].to_string())
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "goalie_game_logs.csv")
        df.to_csv(output_path, index=False)
        print(f"\nDone! {len(df)} rows saved to {output_path}")

if __name__ == "__main__":
    import sys
    test = "--test" in sys.argv
    main(test_mode=test)