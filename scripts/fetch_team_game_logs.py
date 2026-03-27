import requests
import pandas as pd
import time
import os

# --- CONFIG ---
SEASONS = [
    "20202021", "20212022", "20222023",
    "20232024", "20242025", "20252026"
]
OUTPUT_DIR = os.path.expanduser("~/nhl-props/data/raw/team_game_logs")
BASE_URL = "https://api-web.nhle.com/v1"

# All franchises by season - VGK joined 2017, SEA joined 2021, ARI->UTA 2024
TEAMS = [
    "ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL",
    "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR",
    "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN",
    "VGK", "WPG", "WSH"
]

# Penalty type codes that create PP time
MINOR_CODES = {"MIN", "DBL"}  # minor (2 min) and double minor (4 min)
MAJOR_CODES = {"MAJ"}         # major (5 min) - rare but does create PP


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


def get_strength(situation_code, event_team_is_home):
    """Determine strength state from situation code."""
    if not situation_code or len(situation_code) != 4:
        return "other"
    away_skaters = int(situation_code[1])
    home_skaters = int(situation_code[2])

    if away_skaters == home_skaters:
        return "ev"
    elif event_team_is_home:
        return "pp" if home_skaters > away_skaters else "sh"
    else:
        return "pp" if away_skaters > home_skaters else "sh"


def calculate_strength_toi(plays, home_id):
    """Calculate time on ice at each strength state for both teams."""
    toi = {
        "home": {"ev": 0, "pp": 0, "sh": 0, "en": 0},
        "away": {"ev": 0, "pp": 0, "sh": 0, "en": 0},
    }

    def time_to_seconds(t):
        try:
            parts = t.split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except:
            return 0

    valid_plays = [p for p in plays if p.get("situationCode")]
    valid_plays = sorted(valid_plays,
                         key=lambda p: (p.get("periodDescriptor", {}).get("number", 0),
                                        time_to_seconds(p.get("timeInPeriod", "0:00"))))

    for i, play in enumerate(valid_plays):
        situation = play.get("situationCode", "")
        if len(situation) != 4:
            continue

        period = play.get("periodDescriptor", {}).get("number", 1)
        current_time = time_to_seconds(play.get("timeInPeriod", "0:00"))

        if i < len(valid_plays) - 1:
            next_play = valid_plays[i + 1]
            next_period = next_play.get("periodDescriptor", {}).get("number", 1)
            next_time = time_to_seconds(next_play.get("timeInPeriod", "0:00"))
            if next_period == period:
                elapsed = next_time - current_time
            else:
                period_length = 300 if period > 3 else 1200
                elapsed = period_length - current_time
        else:
            period_length = 300 if period > 3 else 1200
            elapsed = period_length - current_time

        if elapsed <= 0:
            continue

        away_goalie = situation[0]
        away_skaters = int(situation[1])
        home_skaters = int(situation[2])
        home_goalie = situation[3]

        home_en = home_goalie == "0"
        away_en = away_goalie == "0"

        if home_en or away_en:
            toi["home"]["en"] += elapsed
            toi["away"]["en"] += elapsed
        elif home_skaters == away_skaters:
            toi["home"]["ev"] += elapsed
            toi["away"]["ev"] += elapsed
        elif home_skaters > away_skaters:
            toi["home"]["pp"] += elapsed
            toi["away"]["sh"] += elapsed
        else:
            toi["home"]["sh"] += elapsed
            toi["away"]["pp"] += elapsed

    return {
        "home_ev_toi": round(toi["home"]["ev"] / 60, 2),
        "home_pp_toi": round(toi["home"]["pp"] / 60, 2),
        "home_sh_toi": round(toi["home"]["sh"] / 60, 2),
        "home_en_toi": round(toi["home"]["en"] / 60, 2),
        "away_ev_toi": round(toi["away"]["ev"] / 60, 2),
        "away_pp_toi": round(toi["away"]["pp"] / 60, 2),
        "away_sh_toi": round(toi["away"]["sh"] / 60, 2),
        "away_en_toi": round(toi["away"]["en"] / 60, 2),
    }


def parse_game(data):
    """Aggregate play-by-play events into team-level stats by strength."""
    game_id = data.get("id")
    date = data.get("gameDate")
    season = data.get("season")

    home = data.get("homeTeam", {})
    away = data.get("awayTeam", {})
    home_id = home.get("id")
    home_abbrev = home.get("abbrev")
    away_abbrev = away.get("abbrev")

    all_periods = set(
        p.get("periodDescriptor", {}).get("number", 0)
        for p in data.get("plays", [])
    )
    went_to_ot = any(p > 3 for p in all_periods)

    def empty_stats():
        return {
            "shots_on_goal": 0, "missed_shots": 0, "blocked_shots": 0,
            "goals": 0, "hits": 0, "giveaways": 0, "takeaways": 0,
            "faceoffs_won": 0, "faceoffs_taken": 0,
            "penalties_drawn": 0, "penalties_taken": 0, "penalty_minutes": 0,
            "minor_penalties_taken": 0, "minor_penalties_drawn": 0,
            "major_penalties_taken": 0, "major_penalties_drawn": 0,
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
            type_code = details.get("typeCode", "")
            s["penalties_taken"] += 1
            s["penalty_minutes"] += pim
            o["penalties_drawn"] += 1
            if type_code in MINOR_CODES:
                s["minor_penalties_taken"] += 1
                o["minor_penalties_drawn"] += 1
            elif type_code in MAJOR_CODES:
                s["major_penalties_taken"] += 1
                o["major_penalties_drawn"] += 1

    # Calculate strength TOI
    toi_data = calculate_strength_toi(data.get("plays", []), home_id)

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
            row[f"{prefix}shot_attempts_for"] = (
                s["shots_on_goal"] + s["missed_shots"] + s["blocked_shots"]
            )
            row[f"{prefix}shot_attempts_against"] = (
                o["shots_on_goal"] + o["missed_shots"] + o["blocked_shots"]
            )
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
            row[f"{prefix}minor_penalties_taken"] = s["minor_penalties_taken"]
            row[f"{prefix}minor_penalties_drawn"] = s["minor_penalties_drawn"]
            row[f"{prefix}major_penalties_taken"] = s["major_penalties_taken"]
            row[f"{prefix}major_penalties_drawn"] = s["major_penalties_drawn"]

        # Total minor penalties across all strengths
        row["total_minor_penalties_taken"] = sum(
            stats[side][st]["minor_penalties_taken"] for st in ["ev", "pp", "sh"]
        )
        row["total_minor_penalties_drawn"] = sum(
            stats[side][st]["minor_penalties_drawn"] for st in ["ev", "pp", "sh"]
        )
        row["total_penalties_taken"] = sum(
            stats[side][st]["penalties_taken"] for st in ["ev", "pp", "sh"]
        )
        row["total_penalties_drawn"] = sum(
            stats[side][st]["penalties_drawn"] for st in ["ev", "pp", "sh"]
        )

        # Add strength TOI
        row["ev_toi"] = toi_data[f"{side}_ev_toi"]
        row["pp_toi"] = toi_data[f"{side}_pp_toi"]
        row["sh_toi"] = toi_data[f"{side}_sh_toi"]
        row["en_toi"] = toi_data[f"{side}_en_toi"]

        rows.append(row)
    return rows


def main(test_mode=False):
    print("Step 1: Collecting game IDs...")
    all_game_ids = set()

    if test_mode:
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

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    if test_mode:
        print("\n--- TEST OUTPUT ---")
        print(df.T)
    else:
        output_path = os.path.join(OUTPUT_DIR, "team_game_logs.csv")
        df.to_csv(output_path, index=False)
        print(f"\nDone! {len(df)} rows saved to {output_path}")


if __name__ == "__main__":
    import sys
    test = "--test" in sys.argv
    main(test_mode=test)