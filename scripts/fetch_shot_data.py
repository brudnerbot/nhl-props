import requests
import pandas as pd
import time
import os
import math

# --- CONFIG ---
SEASONS = ["20222023", "20232024", "20242025", "20252026"]
OUTPUT_DIR = os.path.expanduser("~/nhl-props/data/raw")
BASE_URL = "https://api-web.nhle.com/v1"

TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WPG", "WSH"
]

# NHL net locations (x, y) - nets are at x=+89 and x=-89
NET_X = 89.0
NET_Y = 0.0

SHOT_EVENTS = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}


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


def normalize_coords(x, y, shooting_right):
    """
    Normalize coordinates so all shots are attacking right (positive x).
    This ensures distance/angle calculations are consistent.
    """
    if not shooting_right:
        return -x, -y
    return x, y


def calc_distance(x, y):
    """Calculate distance from net."""
    return math.sqrt((x - NET_X) ** 2 + (y - NET_Y) ** 2)


def calc_angle(x, y):
    """
    Calculate shot angle from net centerline.
    0 = straight on, 90 = from the side.
    """
    angle = math.degrees(math.atan2(abs(y), abs(NET_X - x)))
    return angle


def time_to_seconds(time_str):
    """Convert MM:SS to total seconds."""
    try:
        parts = time_str.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except:
        return None


def parse_shots(data):
    """Extract all shot events from a game's play-by-play."""
    game_id = data.get("id")
    season = data.get("season")
    date = data.get("gameDate")
    home_team = data.get("homeTeam", {})
    away_team = data.get("awayTeam", {})
    home_id = home_team.get("id")
    home_abbrev = home_team.get("abbrev")
    away_abbrev = away_team.get("abbrev")

    plays = data.get("plays", [])
    rows = []
    prev_event = None  # track previous event for speed/rebound calculations

    for play in plays:
        event_type = play.get("typeDescKey")
        details = play.get("details", {})
        period = play.get("periodDescriptor", {}).get("number")
        period_type = play.get("periodDescriptor", {}).get("periodType")
        time_in_period = play.get("timeInPeriod", "0:00")
        situation_code = play.get("situationCode", "")
        home_defending_side = play.get("homeTeamDefendingSide", "")

        # Process previous event info regardless of shot type
        # so we can calculate speed/context for the next shot
        if event_type not in SHOT_EVENTS:
            prev_event = {
                "type": event_type,
                "x": details.get("xCoord"),
                "y": details.get("yCoord"),
                "time_seconds": time_to_seconds(time_in_period),
                "period": period,
            }
            continue

        x = details.get("xCoord")
        y = details.get("yCoord")
        if x is None or y is None:
            prev_event = None
            continue

        owner_team_id = details.get("eventOwnerTeamId")
        is_home = owner_team_id == home_id
        shooting_team = home_abbrev if is_home else away_abbrev
        defending_team = away_abbrev if is_home else home_abbrev

        # Determine if shooting team is attacking right this period
        # Home team defends left in period 1 = attacks right
        if is_home:
            attacking_right = home_defending_side == "left"
        else:
            attacking_right = home_defending_side == "right"

        # Normalize coords so we always calculate vs right net
        norm_x, norm_y = normalize_coords(x, y, attacking_right)

        # Distance and angle
        distance = calc_distance(norm_x, norm_y)
        angle = calc_angle(norm_x, norm_y)

        # Situation
        is_goal = event_type == "goal"
        is_on_goal = event_type in {"shot-on-goal", "goal"}
        is_missed = event_type == "missed-shot"
        is_blocked = event_type == "blocked-shot"
        is_empty_net = situation_code[3] == "0" if len(situation_code) == 4 and is_home else \
                       situation_code[0] == "0" if len(situation_code) == 4 else False

        # Strength state
        away_skaters = int(situation_code[1]) if len(situation_code) == 4 else None
        home_skaters = int(situation_code[2]) if len(situation_code) == 4 else None
        if away_skaters and home_skaters:
            if away_skaters == home_skaters:
                strength = "ev"
            elif is_home and home_skaters > away_skaters:
                strength = "pp"
            elif not is_home and away_skaters > home_skaters:
                strength = "pp"
            else:
                strength = "sh"
        else:
            strength = None

        # Previous event features
        current_seconds = time_to_seconds(time_in_period)
        prev_event_type = None
        prev_distance = None
        speed_from_prev = None
        is_rebound = False
        rebound_angle_change = None

        if prev_event and prev_event.get("period") == period:
            prev_event_type = prev_event.get("type")
            px = prev_event.get("x")
            py = prev_event.get("y")
            prev_time = prev_event.get("time_seconds")

            if px is not None and py is not None:
                prev_distance = math.sqrt((x - px) ** 2 + (y - py) ** 2)
                time_diff = (current_seconds or 0) - (prev_time or 0)
                if time_diff > 0:
                    speed_from_prev = prev_distance / time_diff
                elif time_diff == 0:
                    speed_from_prev = prev_distance / 0.001  # instant rebound

            # Rebound: previous event was a shot on goal within 3 seconds
            if prev_event_type in {"shot-on-goal", "goal"} and \
               prev_time is not None and current_seconds is not None and \
               (current_seconds - prev_time) <= 3:
                is_rebound = True
                if px is not None and py is not None:
                    prev_norm_x, prev_norm_y = normalize_coords(px, py, attacking_right)
                    prev_angle = calc_angle(prev_norm_x, prev_norm_y)
                    rebound_angle_change = abs(angle - prev_angle)

        row = {
            "game_id": game_id,
            "season": season,
            "date": date,
            "period": period,
            "period_type": period_type,
            "time_in_period": time_in_period,
            "time_seconds": current_seconds,
            "shooting_team": shooting_team,
            "defending_team": defending_team,
            "is_home": is_home,
            "shooter_id": details.get("shootingPlayerId") or details.get("scoringPlayerId"),
            "goalie_id": details.get("goalieInNetId"),
            "event_type": event_type,
            "is_goal": int(is_goal),
            "is_on_goal": int(is_on_goal),
            "is_missed": int(is_missed),
            "is_blocked": int(is_blocked),
            "is_empty_net": int(is_empty_net),
            "is_rebound": int(is_rebound),
            "shot_type": details.get("shotType"),
            "x_coord": x,
            "y_coord": y,
            "x_coord_norm": norm_x,
            "y_coord_norm": norm_y,
            "distance": round(distance, 2),
            "angle": round(angle, 2),
            "strength": strength,
            "situation_code": situation_code,
            "prev_event_type": prev_event_type,
            "prev_distance": round(prev_distance, 2) if prev_distance else None,
            "speed_from_prev": round(speed_from_prev, 2) if speed_from_prev else None,
            "rebound_angle_change": round(rebound_angle_change, 2) if rebound_angle_change else None,
            "miss_reason": details.get("reason") if is_missed else None,
        }
        rows.append(row)

        # Update prev event
        prev_event = {
            "type": event_type,
            "x": x,
            "y": y,
            "time_seconds": current_seconds,
            "period": period,
        }

    return rows


def main(test_mode=False):
    # Step 1: collect game IDs
    print("Step 1: Collecting game IDs...")
    all_game_ids = set()

    if test_mode:
        all_game_ids = {2023020001, 2023020002, 2023020003}
        print(f"  TEST MODE: using {len(all_game_ids)} games")
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

    # Step 2: fetch play-by-play and parse shots
    print("\nStep 2: Fetching play-by-play and extracting shots...")
    all_rows = []
    for i, game_id in enumerate(sorted(all_game_ids)):
        try:
            if i % 100 == 0:
                print(f"  {i}/{len(all_game_ids)} games processed...")
            data = fetch_play_by_play(game_id)
            rows = parse_shots(data)
            all_rows.extend(rows)
            time.sleep(0.3)
        except Exception as e:
            print(f"  ERROR game {game_id}: {e}")

    # Step 3: save
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["game_id", "period", "time_seconds"]).reset_index(drop=True)

    if test_mode:
        print("\n--- TEST OUTPUT ---")
        print(f"Total shots: {len(df)}")
        print()
        print(df[["date", "shooting_team", "event_type", "shot_type",
                   "distance", "angle", "is_goal", "is_rebound",
                   "strength", "prev_event_type"]].head(20).to_string())
        print()
        print("Goals only:")
        print(df[df["is_goal"] == 1][["shooting_team", "shot_type",
                                       "distance", "angle", "is_rebound",
                                       "strength"]].head(10).to_string())
    else:
        output_path = os.path.join(OUTPUT_DIR, "shot_data", "shot_data.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nDone! {len(df)} shots saved to {output_path}")


if __name__ == "__main__":
    import sys
    test = "--test" in sys.argv
    main(test_mode=test)