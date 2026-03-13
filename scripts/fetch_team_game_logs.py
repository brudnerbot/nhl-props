import requests
import pandas as pd
import time
import os

# --- CONFIG ---
SEASONS = ["20232024", "20242025", "20252026"]
OUTPUT_DIR = os.path.expanduser("~/nhl-props/data/raw/team_game_logs")
BASE_URL = "https://api-web.nhle.com/v1"

TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WPG", "WSH"
]

# Situation codes where each team is on the power play
# Format is: away_goalie | away_skaters | home_skaters | home_goalie
def get_strength(situation_code, event_team_is_home):
    if not situation_code or len(situation_code) != 4:
        return "other"
    away_skaters = int(situation_code[1])
    home_skaters = int(situation_code[2])

    if away_skaters == home_skaters:
        if away_skaters == 5:
            return "ev"
        elif away_skaters == 4:
            return "ev"  # 4v4 OT, treat as EV
        else:
            return "other"
    elif event_team_is_home:
        if home_skaters > away_skaters:
            return "pp"
        else:
            return "sh"
    else:
        if away_skaters > home_skaters:
            return "pp"
        else:
            return "sh"

def fetch_team_schedule(team, season):
    """Get all completed regular season game IDs for a team."""
    url = f"{BASE_URL}/club-schedule-season/{team}/{season}"
    response = requests.get(url)
    response.raise_for_status()
    games = response.json().get("games", [])
    game_ids = []
    for game in games:
        if game.get("gameType") != 2:
            continue
        if game.get("gameState") not in ["OFF", "FINAL"]:
            continue
        game_ids.append(game.get("id"))
    return game_ids

def fetch_play_by_play(game_id):
    """Get full play-by-play for a single game."""
    url = f"{BASE_URL}/gamecenter/{game_id}/play-by-play"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def parse_game(data):
    """Aggregate play-by-play events into team-level stats by strength."""
    game_id = data.get("id")
    date = data.get("gameDate")
    season = data.get("season")

    home = data.get("homeTeam", {})
    away = data.get("awayTeam", {})
    home_id = home.get("id")
    away_id = away.get("id")
    home_abbrev = home.get("abbrev")
    away_abbrev = away.get("abbrev")

    # Check OT/shootout
    periods = data.get("periodDescriptor", {})
    all_periods = set(p.get("periodDescriptor", {}).get("number", 0) for p in data.get("plays", []))
    went_to_ot = any(p > 3 for p in all_periods)

    # Base row template
    def empty_stats():
        return {
            "shots_on_goal": 0, "missed_shots": 0, "blocked_shots": 0,
            "goals": 0, "hits": 0, "giveaways": 0, "takeaways": 0,
            "faceoffs_won": 0, "faceoffs_taken": 0,
            "penalties_drawn": 0, "penalties_taken": 0, "penalty_minutes": 0,
        }

    stats = {
        "home": {"ev": empty_stats(), "pp": empty_stats(), "sh": empty_stats()},
        "away": {"ev": empty_stats(), "pp": empty_stats(), "sh": empty_stats()},
    }

    for play in data.get("plays", []):
        event_type = play.get("typeDescKey")
        situation_code = play.get("situationCode", "")
        details = play.get("details", {})
        owner_team_id = details.get("eventOwnerTeamId")

        if not owner_team_id:
            continue

        is_home = owner_team_id == home_id
        side = "home" if is_home else "away"
        opp_side = "away" if is_home else "home"
        strength = get_strength(situation_code, is_home)

        if strength == "other":
            continue

        s = stats[side][strength]
        o = stats[opp_side][strength]

        if event_type == "shot-on-goal":
            s["shots_on_goal"] += 1
        elif event_type == "missed-shot":
            s["missed_shots"] += 1
        elif event_type == "blocked-shot":
            # blocked shots are credited to the blocking team in NHL data
            o["blocked_shots"] += 1
        elif event_type == "goal":
            s["goals"] += 1
            s["shots_on_goal"] += 1
        elif event_type == "hit":
            s["hits"] += 1
        elif event_type == "giveaway":
            s["giveaways"] += 1
        elif event_type == "takeaway":
            s["takeaways"] += 1
        elif event_type == "faceoff":
            winning_player = details.get("winningPlayerId")
            s["faceoffs_taken"] += 1
            o["faceoffs_taken"] += 1
            if winning_player:
                s["faceoffs_won"] += 1
        elif event_type == "penalty":
            pim = details.get("duration", 0)
            s["penalties_taken"] += 1
            s["penalty_minutes"] += pim
            o["penalties_drawn"] += 1

    # Flatten into rows, one per team
    rows = []
    for side, team_abbrev, opp_abbrev in [
        ("home", home_abbrev, away_abbrev),
        ("away", away_abbrev, home_abbrev)
    ]:
        row = {
            "game_id": game_id,
            "season": season,
            "date": date,
            "team": team_abbrev,
            "opponent": opp_abbrev,
            "is_home": side == "home",
            "went_to_ot": went_to_ot,
            "won": 1 if home.get("score", 0) > away.get("score", 0) and side == "home"
                   else 1 if away.get("score", 0) > home.get("score", 0) and side == "away"
                   else 0,
            "goals_for": home.get("score", 0) if side == "home" else away.get("score", 0),
            "goals_against": away.get("score", 0) if side == "home" else home.get("score", 0),
        }

        for strength in ["ev", "pp", "sh"]:
            s = stats[side][strength]
            o = stats["away" if side == "home" else "home"][strength]
            prefix = f"{strength}_"
            row[f"{prefix}shots_on_goal_for"] = s["shots_on_goal"]
            row[f"{prefix}shots_on_goal_against"] = o["shots_on_goal"]
            row[f"{prefix}missed_shots_for"] = s["missed_shots"]
            row[f"{prefix}missed_shots_against"] = o["missed_shots"]
            row[f"{prefix}blocked_shots_for"] = s["blocked_shots"]
            row[f"{prefix}blocked_shots_against"] = o["blocked_shots"]
            row[f"{prefix}shot_attempts_for"] = s["shots_on_goal"] + s["missed_shots"] + s["blocked_shots"]
            row[f"{prefix}shot_attempts_against"] = o["shots_on_goal"] + o["missed_shots"] + o["blocked_shots"]
            row[f"{prefix}goals_for"] = s["goals"]
            row[f"{prefix}goals_against"] = o["goals"]
            row[f"{prefix}hits_for"] = s["hits"]
            row[f"{prefix}hits_against"] = o["hits"]
            row[f"{prefix}giveaways"] = s["giveaways"]
            row[f"{prefix}takeaways"] = s["takeaways"]
            row[f"{prefix}faceoffs_won"] = s["faceoffs_won"]
            row[f"{prefix}faceoffs_taken"] = s["faceoffs_taken"]
            row[f"{prefix}penalties_taken"] = s["penalties_taken"]
            row[f"{prefix}penalties_drawn"] = s["penalties_drawn"]
            row[f"{prefix}penalty_minutes"] = s["penalty_minutes"]

        rows.append(row)
    return rows

def main(test_mode=False):
    # Step 1: collect game IDs
    print("Step 1: Collecting game IDs...")
    all_game_ids = set()

    if test_mode:
        # Just use one known game ID for testing
        all_game_ids = {2023020001}
        print("  TEST MODE: using game 2023020001 only")
    else:
        for season in SEASONS:
            print(f"  Season {season}:")
            for team in TEAMS:
                try:
                    ids = fetch_team_schedule(team, season)
                    all_game_ids.update(ids)
                    time.sleep(0.3)
                except Exception as e:
                    print(f"    ERROR {team} {season}: {e}")
            print(f"    Total unique games so far: {len(all_game_ids)}")

    print(f"\nTotal unique games to fetch: {len(all_game_ids)}")

    # Step 2: fetch play-by-play for each game
    print("\nStep 2: Fetching play-by-play...")
    all_rows = []
    for i, game_id in enumerate(sorted(all_game_ids)):
        try:
            if i % 100 == 0:
                print(f"  {i}/{len(all_game_ids)} games fetched...")
            data = fetch_play_by_play(game_id)
            rows = parse_game(data)
            all_rows.extend(rows)
            time.sleep(0.3)
        except Exception as e:
            print(f"  ERROR game {game_id}: {e}")

    # Step 3: save
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    if test_mode:
        print("\n--- TEST OUTPUT ---")
        print(df.T)  # print transposed so all columns are visible
    else:
        output_path = os.path.join(OUTPUT_DIR, "team_game_logs.csv")
        df.to_csv(output_path, index=False)
        print(f"\nDone! {len(df)} rows saved to {output_path}")

if __name__ == "__main__":
    import sys
    test = "--test" in sys.argv
    main(test_mode=test)