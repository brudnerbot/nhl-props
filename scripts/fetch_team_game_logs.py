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

def fetch_boxscore(game_id):
    """Get full boxscore for a single game."""
    url = f"{BASE_URL}/gamecenter/{game_id}/boxscore"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def parse_boxscore(data):
    """Extract team-level stats for both home and away from a boxscore."""
    rows = []
    game_id = data.get("id")
    date = data.get("gameDate")
    season = data.get("season")

    home_team = data.get("homeTeam", {})
    away_team = data.get("awayTeam", {})

    # Check for overtime/shootout
    period_descriptor = data.get("periodDescriptor", {})
    periods = data.get("linescore", {}).get("byPeriod", [])
    went_to_ot = len(periods) > 3

    for side, team, opp in [("home", home_team, away_team), ("away", away_team, home_team)]:
        team_stats = team.get("commonTeamStats", {})
        opp_stats = opp.get("commonTeamStats", {})

        row = {
            "game_id": game_id,
            "season": season,
            "date": date,
            "team": team.get("abbrev"),
            "opponent": opp.get("abbrev"),
            "is_home": side == "home",
            "went_to_ot": went_to_ot,
            "won": 1 if team.get("score", 0) > opp.get("score", 0) else 0,

            # Goals
            "goals_for": team.get("score", 0),
            "goals_against": opp.get("score", 0),

            # Shots on goal
            "shots_for": team_stats.get("sog", None),
            "shots_against": opp_stats.get("sog", None),

            # Shot attempts (Corsi)
            "shot_attempts_for": team_stats.get("shotAttempts", None),
            "shot_attempts_against": opp_stats.get("shotAttempts", None),

            # Missed shots
            "missed_shots_for": team_stats.get("missedShots", None),
            "missed_shots_against": opp_stats.get("missedShots", None),

            # Blocked shots
            "blocked_shots_for": team_stats.get("blockedShots", None),
            "blocked_shots_against": opp_stats.get("blockedShots", None),

            # Power play
            "pp_goals_for": team_stats.get("ppGoals", None),
            "pp_opportunities_for": team_stats.get("ppOpportunities", None),
            "pp_goals_against": opp_stats.get("ppGoals", None),
            "pp_opportunities_against": opp_stats.get("ppOpportunities", None),

            # Faceoffs
            "faceoffs_won": team_stats.get("faceoffWinningPctg", None),

            # Hits, giveaways, takeaways, PIM
            "hits_for": team_stats.get("hits", None),
            "hits_against": opp_stats.get("hits", None),
            "giveaways": team_stats.get("giveaways", None),
            "takeaways": team_stats.get("takeaways", None),
            "pim": team_stats.get("pim", None),
        }
        rows.append(row)
    return rows

def main():
    # Step 1: collect all unique game IDs across all teams and seasons
    print("Step 1: Collecting game IDs...")
    all_game_ids = set()
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

    # Step 2: fetch boxscore for each game
    print("\nStep 2: Fetching boxscores...")
    all_rows = []
    for i, game_id in enumerate(sorted(all_game_ids)):
        try:
            if i % 100 == 0:
                print(f"  {i}/{len(all_game_ids)} games fetched...")
            data = fetch_boxscore(game_id)
            rows = parse_boxscore(data)
            all_rows.extend(rows)
            time.sleep(0.3)
        except Exception as e:
            print(f"  ERROR game {game_id}: {e}")

    # Step 3: save to CSV
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["team", "date"]).reset_index(drop=True)
    output_path = os.path.join(OUTPUT_DIR, "team_game_logs.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDone! {len(df)} rows saved to {output_path}")

if __name__ == "__main__":
    main()