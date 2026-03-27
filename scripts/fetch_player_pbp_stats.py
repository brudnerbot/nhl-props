"""
fetch_player_pbp_stats.py

Fetches player-level stats from NHL API play-by-play and shifts data.
Computes per-game per-player:
  - EV/PP/SH TOI (from shifts + situation codes)
  - Individual shots, goals, assists by strength (from PBP events)
  - Hits, blocks, faceoffs won/taken (from PBP events + boxscore)
  - Giveaways, takeaways

Situation code format: away_goalie|away_skaters|home_skaters|home_goalie
Verified: 1560 = away penalized, home has extra skater
          1451 = home penalized, away has extra skater (away on PP)
          1541 = away penalized, home has extra skater (home on PP)
          1551 = 5v5 EV

Output: data/raw/player_pbp_stats/player_pbp_stats.csv

Usage:
    python scripts/fetch_player_pbp_stats.py --seasons 20242025 20252026
    python scripts/fetch_player_pbp_stats.py --days 7
    python scripts/fetch_player_pbp_stats.py --test
"""

import argparse
import os
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import requests

# --- CONFIG ---
ROOT      = Path(__file__).resolve().parents[1]
OUT_DIR   = ROOT / "data/raw/player_pbp_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE  = OUT_DIR / "player_pbp_stats.csv"
TEAM_LOGS = ROOT / "data/raw/team_game_logs/team_game_logs.csv"

BASE_URL   = "https://api-web.nhle.com/v1"
SHIFTS_URL = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}"

HEADERS = {"User-Agent": "Mozilla/5.0"}


# --- Situation code helpers ---
def decode_situation(code: str):
    """
    Decode 4-digit situation code.
    Format: away_goalie | away_skaters | home_skaters | home_goalie
    e.g. 1560: away_g=1, away_s=5, home_s=6, home_g=0 → home EN, away penalized
         1451: away_g=1, away_s=4, home_s=5, home_g=1 → home penalized, away on PP
         1541: away_g=1, away_s=5, home_s=4, home_g=1 → away penalized, home on PP
         1551: 5v5 EV
    """
    if not code or len(code) != 4:
        return None
    try:
        ag, as_, hs, hg = int(code[0]), int(code[1]), int(code[2]), int(code[3])
        is_en = (hg == 0 or ag == 0)
        return hg, hs, as_, ag, is_en
    except:
        return None


def classify_strength(code: str, is_home: bool) -> str:
    """Return 'ev', 'pp', 'sh', 'en', or 'ev' as default."""
    decoded = decode_situation(code)
    if decoded is None:
        return "ev"
    hg, hs, as_, ag, is_en = decoded
    if is_en:
        return "en"
    if hs == as_:
        return "ev"
    if is_home:
        return "pp" if hs > as_ else "sh"
    else:
        return "pp" if as_ > hs else "sh"


def time_to_seconds(period: int, time_str: str) -> int:
    """Convert period + MM:SS to absolute game seconds."""
    try:
        mins, secs = map(int, time_str.split(":"))
        return (period - 1) * 1200 + mins * 60 + secs
    except:
        return 0


# --- API fetchers ---
def fetch_pbp(game_id: int):
    url = f"{BASE_URL}/gamecenter/{game_id}/play-by-play"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"    PBP error {game_id}: {e}")
    return None


def fetch_shifts(game_id: int):
    url = SHIFTS_URL.format(game_id=game_id)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            return resp.json().get("data", [])
    except Exception as e:
        print(f"    Shifts error {game_id}: {e}")
    return None


def fetch_boxscore(game_id: int):
    url = f"{BASE_URL}/gamecenter/{game_id}/boxscore"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"    Boxscore error {game_id}: {e}")
    return None


# --- Core processing ---
def build_situation_map(pbp: dict) -> dict:
    """
    Build dense second-by-second situation map from all PBP events.
    Forward-fills situation code from most recent event.
    """
    raw_sit = {}
    for play in pbp.get("plays", []):
        sit    = play.get("situationCode", "")
        period = play["periodDescriptor"]["number"]
        if period > 4:
            continue
        t_sec = time_to_seconds(period, play.get("timeInPeriod", "0:00"))
        if sit:
            raw_sit[t_sec] = sit

    if not raw_sit:
        return {}

    time_sit = {}
    sorted_times = sorted(raw_sit.keys())
    current_sit  = "1551"
    t_idx = 0
    max_t = sorted_times[-1] + 120

    for t in range(0, max_t + 1):
        if t_idx < len(sorted_times) and t >= sorted_times[t_idx]:
            current_sit = raw_sit[sorted_times[t_idx]]
            t_idx += 1
        time_sit[t] = current_sit

    return time_sit


def process_game(game_id: int, season: int, date: str,
                 home_team: str, away_team: str) -> list:
    """Process one game, return list of player-game stat rows."""

    pbp    = fetch_pbp(game_id)
    shifts = fetch_shifts(game_id)
    box    = fetch_boxscore(game_id)

    if not pbp or not shifts:
        return []

    home_id   = pbp["homeTeam"]["id"]
    time_sit  = build_situation_map(pbp)

    # --- Build roster from rosterSpots ---
    roster = {}
    for spot in pbp.get("rosterSpots", []):
        pid       = spot["playerId"]
        team_id   = spot["teamId"]
        is_home   = (team_id == home_id)
        team_abbr = home_team if is_home else away_team
        roster[pid] = {
            "player_id": pid,
            "name":      f"{spot['firstName']['default']} {spot['lastName']['default']}",
            "position":  spot.get("positionCode", "?"),
            "team":      team_abbr,
            "is_home":   is_home,
        }

    # Supplement roster from shifts for players missing from rosterSpots
    for shift in shifts:
        pid = shift["playerId"]
        if pid not in roster:
            team_abbr = shift.get("teamAbbrev", "")
            is_home   = (team_abbr == home_team)
            roster[pid] = {
                "player_id": pid,
                "name":      f"{shift.get('firstName','')} {shift.get('lastName','')}",
                "position":  "?",
                "team":      team_abbr,
                "is_home":   is_home,
            }

    # --- Compute TOI by strength from shifts ---
    player_toi = defaultdict(lambda: {
        "ev": 0.0, "pp": 0.0, "sh": 0.0, "en": 0.0, "total": 0.0
    })

    for shift in shifts:
        pid    = shift["playerId"]
        period = shift["period"]
        if period > 4:
            continue

        start_sec = time_to_seconds(period, shift["startTime"])
        end_sec   = time_to_seconds(period, shift["endTime"])
        if end_sec <= start_sec:
            continue

        is_home = roster.get(pid, {}).get("is_home", True)

        # Accumulate TOI second by second (exclude end second)
        for t in range(start_sec, end_sec):
            sit      = time_sit.get(t, "1551")
            strength = classify_strength(sit, is_home)
            player_toi[pid][strength] += 1 / 60.0
            player_toi[pid]["total"]  += 1 / 60.0

    # --- Parse PBP events for individual stats ---
    player_stats = defaultdict(lambda: {
        "ev_shots": 0, "pp_shots": 0, "sh_shots": 0,
        "ev_goals": 0, "pp_goals": 0, "sh_goals": 0,
        "ev_assists": 0, "pp_assists": 0, "sh_assists": 0,
        "hits": 0, "blocks": 0,
        "faceoffs_won": 0, "faceoffs_taken": 0,
        "giveaways": 0, "takeaways": 0,
    })

    for play in pbp.get("plays", []):
        etype  = play.get("typeDescKey", "")
        sit    = play.get("situationCode", "")
        period = play["periodDescriptor"]["number"]
        if period > 4:
            continue
        det = play.get("details", {})

        # Decode situation for strength classification
        decoded = decode_situation(sit)
        if not decoded:
            continue
        hg, hs, as_, ag, is_en = decoded

        def get_strength(player_id):
            """Get pp/sh/ev/en for a specific player."""
            if is_en:
                return "en"
            if hs == as_:
                return "ev"
            is_home = roster.get(player_id, {}).get("is_home", True)
            if is_home:
                return "pp" if hs > as_ else "sh"
            else:
                return "pp" if as_ > hs else "sh"

        if etype in ("shot-on-goal", "goal"):
            shooter = det.get("shootingPlayerId") or det.get("scoringPlayerId")
            if shooter and shooter in roster:
                s = get_strength(shooter)
                if s != "en":
                    player_stats[shooter][f"{s}_shots"] += 1

            if etype == "goal":
                scorer = det.get("scoringPlayerId")
                if scorer and scorer in roster:
                    s = get_strength(scorer)
                    if s != "en":
                        player_stats[scorer][f"{s}_goals"] += 1

                for assist_key in ["assist1PlayerId", "assist2PlayerId"]:
                    assister = det.get(assist_key)
                    if assister and assister in roster:
                        s = get_strength(assister)
                        if s != "en":
                            player_stats[assister][f"{s}_assists"] += 1

        elif etype == "hit":
            hitter = det.get("hittingPlayerId")
            if hitter and hitter in roster:
                player_stats[hitter]["hits"] += 1

        elif etype == "blocked-shot":
            blocker = det.get("blockingPlayerId")
            if blocker and blocker in roster:
                player_stats[blocker]["blocks"] += 1

        elif etype == "faceoff":
            winner = det.get("winningPlayerId")
            loser  = det.get("losingPlayerId")
            if winner and winner in roster:
                player_stats[winner]["faceoffs_won"]   += 1
                player_stats[winner]["faceoffs_taken"] += 1
            if loser and loser in roster:
                player_stats[loser]["faceoffs_taken"] += 1

        elif etype == "giveaway":
            pid = det.get("playerId")
            if pid and pid in roster:
                player_stats[pid]["giveaways"] += 1

        elif etype == "takeaway":
            pid = det.get("playerId")
            if pid and pid in roster:
                player_stats[pid]["takeaways"] += 1

    # Override hits and blocks with boxscore (more reliable)
    if box:
        for side in ["homeTeam", "awayTeam"]:
            team_data = box.get("playerByGameStats", {}).get(side, {})
            for pos in ["forwards", "defense"]:
                for p in team_data.get(pos, []):
                    pid = p["playerId"]
                    player_stats[pid]["hits"]   = p.get("hits", 0)
                    player_stats[pid]["blocks"] = p.get("blockedShots", 0)

    # --- Combine into rows ---
    rows = []
    all_pids = set(roster.keys()) | set(player_toi.keys())

    for pid in all_pids:
        if pid not in roster:
            continue
        r     = roster[pid]
        toi   = player_toi[pid]
        stats = player_stats[pid]

        rows.append({
            "game_id":        game_id,
            "season":         season,
            "date":           date,
            "player_id":      pid,
            "name":           r["name"],
            "position":       r["position"],
            "team":           r["team"],
            "opponent":       away_team if r["is_home"] else home_team,
            "is_home":        r["is_home"],
            "toi_total":      round(toi["total"], 3),
            "toi_ev":         round(toi["ev"],    3),
            "toi_pp":         round(toi["pp"],    3),
            "toi_sh":         round(toi["sh"],    3),
            "toi_en":         round(toi["en"],    3),
            "ev_shots":       stats["ev_shots"],
            "pp_shots":       stats["pp_shots"],
            "sh_shots":       stats["sh_shots"],
            "ev_goals":       stats["ev_goals"],
            "pp_goals":       stats["pp_goals"],
            "sh_goals":       stats["sh_goals"],
            "ev_assists":     stats["ev_assists"],
            "pp_assists":     stats["pp_assists"],
            "sh_assists":     stats["sh_assists"],
            "hits":           stats["hits"],
            "blocks":         stats["blocks"],
            "faceoffs_won":   stats["faceoffs_won"],
            "faceoffs_taken": stats["faceoffs_taken"],
            "giveaways":      stats["giveaways"],
            "takeaways":      stats["takeaways"],
        })

    return rows


# --- Game list ---
def get_games_to_fetch(seasons=None, days=None):
    logs = pd.read_csv(TEAM_LOGS, low_memory=False)
    logs["date"] = pd.to_datetime(logs["date"])

    if days:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        logs = logs[logs["date"] >= cutoff]
    elif seasons:
        logs = logs[logs["season"].isin(seasons)]

    # Regular season only
    logs = logs[logs["game_id"].astype(str).str[4:6] == "02"]

    home = logs[logs["is_home"] == True][["game_id","season","date","team"]].copy()
    home.columns = ["game_id","season","date","home_team"]
    away = logs[logs["is_home"] == False][["game_id","team"]].copy()
    away.columns = ["game_id","away_team"]

    games = home.merge(away, on="game_id").drop_duplicates("game_id")
    games = games.sort_values("date")

    return games[["game_id","season","date","home_team","away_team"]].values.tolist()


# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int,
                        default=[20242025, 20252026])
    parser.add_argument("--days",    type=int, default=None)
    parser.add_argument("--test",    action="store_true",
                        help="Process only 3 games for testing")
    args = parser.parse_args()

    games = get_games_to_fetch(
        seasons=args.seasons if not args.days else None,
        days=args.days
    )
    print(f"Games to process: {len(games):,}")

    if args.test:
        games = games[:3]
        print(f"TEST MODE: processing {len(games)} games")

    # Skip already-processed games
    existing_games = set()
    if OUT_FILE.exists():
        existing = pd.read_csv(OUT_FILE, usecols=["game_id"])
        existing_games = set(existing["game_id"].unique())
        print(f"Already processed: {len(existing_games):,} games")

    games = [g for g in games if int(g[0]) not in existing_games]
    print(f"New games to fetch: {len(games):,}")

    if not games:
        print("Nothing to fetch.")
        return

    all_rows = []
    for i, (game_id, season, date, home_team, away_team) in enumerate(games):
        if i % 50 == 0:
            print(f"  [{i}/{len(games)}] {game_id} ({str(date)[:10]} "
                  f"{home_team} vs {away_team})...")

        rows = process_game(int(game_id), int(season), str(date)[:10],
                            str(home_team), str(away_team))
        all_rows.extend(rows)

        # Checkpoint every 200 games
        if i % 200 == 199:
            df_new = pd.DataFrame(all_rows)
            if OUT_FILE.exists():
                df_new = pd.concat([pd.read_csv(OUT_FILE), df_new],
                                   ignore_index=True)
            df_new.drop_duplicates(subset=["game_id","player_id"]).to_csv(
                OUT_FILE, index=False)
            print(f"    Checkpoint: {len(df_new):,} rows saved")
            all_rows = []

        time.sleep(0.3)

    # Final save
    if all_rows:
        df_new = pd.DataFrame(all_rows)
        if OUT_FILE.exists():
            df_new = pd.concat([pd.read_csv(OUT_FILE), df_new],
                               ignore_index=True)
        df_new.drop_duplicates(subset=["game_id","player_id"]).to_csv(
            OUT_FILE, index=False)

    final = pd.read_csv(OUT_FILE)
    print(f"\nDone! {len(final):,} player-game rows")
    print(f"  Seasons: {sorted(final['season'].unique())}")
    print(f"  Columns: {list(final.columns)}")
    print(f"\nSample row:")
    print(final.iloc[0].to_string())


if __name__ == "__main__":
    main()