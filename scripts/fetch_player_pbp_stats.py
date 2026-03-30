"""
fetch_player_pbp_stats.py

Fetches player-level stats from NHL API play-by-play and shifts data.
Computes per-game per-player:
  - EV/PP/SH TOI (from shifts + situation codes)
  - Individual shots, goals, assists by strength (from PBP events)
  - Missed shots and shots blocked by opponent (individual Corsi components)
  - On-ice shots for/against by strength
  - On-ice goals for/against by strength
  - Hits, blocks, faceoffs won/taken, giveaways, takeaways

Situation code format: away_goalie|away_skaters|home_skaters|home_goalie
Verified: 1451 = home penalized (away on PP)
          1541 = away penalized (home on PP)
          1551 = 5v5 EV

Output: data/raw/player_pbp_stats/player_pbp_stats.csv

Usage:
    python scripts/fetch_player_pbp_stats.py --seasons 20242025 20252026
    python scripts/fetch_player_pbp_stats.py --days 7
    python scripts/fetch_player_pbp_stats.py --test
"""

import argparse
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
import requests

# --- CONFIG ---
ROOT      = Path(__file__).resolve().parents[1]
OUT_DIR   = ROOT / "data/raw/player_pbp_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE  = OUT_DIR / "player_pbp_stats.csv"
TEAM_LOGS = ROOT / "data/raw/team_game_logs/team_game_logs.csv"

BASE_URL   = "https://api-web.nhle.com/v1"
SHIFTS_URL = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}"
HEADERS    = {"User-Agent": "Mozilla/5.0"}


# --- Situation code helpers ---
def decode_situation(code: str):
    """
    Format: away_goalie | away_skaters | home_skaters | home_goalie
    e.g. 1451: ag=1 as=4 hs=5 hg=1 → home penalized, away on PP
         1541: ag=1 as=5 hs=4 hg=1 → away penalized, home on PP
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
    decoded = decode_situation(code)
    if decoded is None:
        return "ev"
    hg, hs, as_, ag, is_en = decoded

    if is_en:
        # Empty net — classify by skater differential (ignore missing goalie)
        # 6v5 = EV, 6v4 = PP for team with extra skater, 5v6 = SH
        if hs == as_:
            return "ev"
        if is_home:
            return "pp" if hs > as_ else "sh"
        else:
            return "pp" if as_ > hs else "sh"

    if hs == as_:
        return "ev"
    if is_home:
        return "pp" if hs > as_ else "sh"
    else:
        return "pp" if as_ > hs else "sh"


def time_to_seconds(period: int, time_str: str) -> int:
    try:
        mins, secs = map(int, time_str.split(":"))
        return (period - 1) * 1200 + mins * 60 + secs
    except:
        return 0


# --- API fetchers ---
def fetch_pbp(game_id: int):
    try:
        resp = requests.get(f"{BASE_URL}/gamecenter/{game_id}/play-by-play",
                            headers=HEADERS, timeout=20)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        print(f"    PBP error {game_id}: {e}"); return None


def fetch_shifts(game_id: int):
    try:
        resp = requests.get(SHIFTS_URL.format(game_id=game_id),
                            headers=HEADERS, timeout=20)
        return resp.json().get("data", []) if resp.status_code == 200 else None
    except Exception as e:
        print(f"    Shifts error {game_id}: {e}"); return None


def fetch_boxscore(game_id: int):
    try:
        resp = requests.get(f"{BASE_URL}/gamecenter/{game_id}/boxscore",
                            headers=HEADERS, timeout=20)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        print(f"    Boxscore error {game_id}: {e}"); return None


# --- Situation map ---
def build_situation_map(pbp: dict) -> dict:
    """Dense second-by-second situation map, forward-filled from all PBP events."""
    raw_sit = {}
    for play in pbp.get("plays", []):
        sit    = play.get("situationCode", "")
        period = play["periodDescriptor"]["number"]
        if period > 4: continue
        t = time_to_seconds(period, play.get("timeInPeriod", "0:00"))
        if sit:
            raw_sit[t] = sit

    if not raw_sit:
        return {}

    time_sit     = {}
    sorted_times = sorted(raw_sit.keys())
    current_sit  = "1551"
    t_idx        = 0
    max_t        = sorted_times[-1] + 120

    for t in range(0, max_t + 1):
        if t_idx < len(sorted_times) and t >= sorted_times[t_idx]:
            current_sit = raw_sit[sorted_times[t_idx]]
            t_idx += 1
        time_sit[t] = current_sit

    return time_sit


# --- Shift intervals for on-ice lookups ---
def build_shift_intervals(shifts: list) -> dict:
    """player_id -> list of (start_sec, end_sec) tuples."""
    intervals = defaultdict(list)
    for s in shifts:
        period = s["period"]
        if period > 4: continue
        sm, ss = map(int, s["startTime"].split(":"))
        em, es = map(int, s["endTime"].split(":"))
        start  = (period - 1) * 1200 + sm * 60 + ss
        end    = (period - 1) * 1200 + em * 60 + es
        if end > start:
            intervals[s["playerId"]].append((start, end))
    return intervals


def players_on_ice_at(t: int, shift_intervals: dict) -> set:
    """Return set of player_ids on ice at absolute second t."""
    return {pid for pid, ivs in shift_intervals.items()
            if any(s <= t < e for s, e in ivs)}


# --- Core processing ---
def process_game(game_id: int, season: int, date: str,
                 home_team: str, away_team: str) -> list:

    pbp    = fetch_pbp(game_id)
    shifts = fetch_shifts(game_id)
    box    = fetch_boxscore(game_id)

    if not pbp or not shifts:
        return []

    home_id   = pbp["homeTeam"]["id"]
    time_sit  = build_situation_map(pbp)
    shift_ivs = build_shift_intervals(shifts)

    # --- Roster ---
    roster = {}
    for spot in pbp.get("rosterSpots", []):
        pid     = spot["playerId"]
        is_home = (spot["teamId"] == home_id)
        roster[pid] = {
            "player_id": pid,
            "name":      f"{spot['firstName']['default']} {spot['lastName']['default']}",
            "position":  spot.get("positionCode", "?"),
            "team":      home_team if is_home else away_team,
            "is_home":   is_home,
        }
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

    # --- TOI by strength ---
    player_toi = defaultdict(lambda: {
        "ev": 0.0, "pp": 0.0, "sh": 0.0, "en": 0.0, "total": 0.0
    })
    for shift in shifts:
        pid    = shift["playerId"]
        period = shift["period"]
        if period > 4: continue
        sm, ss = map(int, shift["startTime"].split(":"))
        em, es = map(int, shift["endTime"].split(":"))
        start  = (period - 1) * 1200 + sm * 60 + ss
        end    = (period - 1) * 1200 + em * 60 + es
        if end <= start: continue
        is_home = roster.get(pid, {}).get("is_home", True)
        for t in range(start, end):
            s = classify_strength(time_sit.get(t, "1551"), is_home)
            player_toi[pid][s]       += 1 / 60.0
            player_toi[pid]["total"] += 1 / 60.0

    # --- Individual + on-ice stats ---
    player_stats = defaultdict(lambda: {
        # Individual shots by strength
        "ev_shots": 0, "pp_shots": 0, "sh_shots": 0,
        # Individual Corsi components by strength
        "ev_missed_shots": 0, "pp_missed_shots": 0, "sh_missed_shots": 0,
        "ev_shots_blocked_by_opp": 0, "pp_shots_blocked_by_opp": 0, "sh_shots_blocked_by_opp": 0,
        # Goals and assists by strength
        "ev_goals": 0, "pp_goals": 0, "sh_goals": 0,
        "ev_assists": 0, "pp_assists": 0, "sh_assists": 0,
        # On-ice shots for/against by strength
        "ev_onice_sf": 0, "ev_onice_sa": 0,
        "pp_onice_sf": 0, "pp_onice_sa": 0,
        "sh_onice_sf": 0, "sh_onice_sa": 0,
        # On-ice goals for/against by strength
        "ev_onice_gf": 0, "ev_onice_ga": 0,
        "pp_onice_gf": 0, "pp_onice_ga": 0,
        "sh_onice_gf": 0, "sh_onice_ga": 0,
        # Misc
        "hits": 0, "blocks": 0,
        "faceoffs_won": 0, "faceoffs_taken": 0,
        "giveaways": 0, "takeaways": 0,
    })

    for play in pbp.get("plays", []):
        etype  = play.get("typeDescKey", "")
        sit    = play.get("situationCode", "")
        period = play["periodDescriptor"]["number"]
        if period > 4: continue
        det   = play.get("details", {})
        t_sec = time_to_seconds(period, play.get("timeInPeriod", "0:00"))

        decoded = decode_situation(sit)
        if not decoded: continue
        hg, hs, as_, ag, is_en = decoded

        def get_strength(pid):
            if is_en: return "en"
            if hs == as_: return "ev"
            ih = roster.get(pid, {}).get("is_home", True)
            if ih: return "pp" if hs > as_ else "sh"
            else:  return "pp" if as_ > hs else "sh"

        if etype in ("shot-on-goal", "goal"):
            shooter = det.get("shootingPlayerId") or det.get("scoringPlayerId")
            if shooter and shooter in roster:
                s = get_strength(shooter)
                if s != "en":
                    player_stats[shooter][f"{s}_shots"] += 1

            # On-ice shots for/against
            on_ice = players_on_ice_at(t_sec, shift_ivs)
            shooting_team_id = det.get("eventOwnerTeamId")
            shooting_is_home = (shooting_team_id == home_id)
            for pid in on_ice:
                if pid not in roster: continue
                pid_is_home = roster[pid]["is_home"]
                s = classify_strength(sit, pid_is_home)
                if s == "en": continue
                if pid_is_home == shooting_is_home:
                    player_stats[pid][f"{s}_onice_sf"] += 1
                else:
                    player_stats[pid][f"{s}_onice_sa"] += 1

            if etype == "goal":
                scorer = det.get("scoringPlayerId")
                if scorer and scorer in roster:
                    s = get_strength(scorer)
                    if s != "en":
                        player_stats[scorer][f"{s}_goals"] += 1

                for ak in ["assist1PlayerId", "assist2PlayerId"]:
                    a = det.get(ak)
                    if a and a in roster:
                        s = get_strength(a)
                        if s != "en":
                            player_stats[a][f"{s}_assists"] += 1

                # On-ice goals for/against
                for pid in on_ice:
                    if pid not in roster: continue
                    pid_is_home = roster[pid]["is_home"]
                    s = classify_strength(sit, pid_is_home)
                    if s == "en": continue
                    if pid_is_home == shooting_is_home:
                        player_stats[pid][f"{s}_onice_gf"] += 1
                    else:
                        player_stats[pid][f"{s}_onice_ga"] += 1

        elif etype == "missed-shot":
            shooter = det.get("shootingPlayerId")
            if shooter and shooter in roster:
                s = get_strength(shooter)
                if s != "en":
                    player_stats[shooter][f"{s}_missed_shots"] += 1

        elif etype == "blocked-shot":
            shooter = det.get("shootingPlayerId")
            if shooter and shooter in roster:
                s = get_strength(shooter)
                if s != "en":
                    player_stats[shooter][f"{s}_shots_blocked_by_opp"] += 1
            blocker = det.get("blockingPlayerId")
            if blocker and blocker in roster:
                player_stats[blocker]["blocks"] += 1

        elif etype == "hit":
            hitter = det.get("hittingPlayerId")
            if hitter and hitter in roster:
                player_stats[hitter]["hits"] += 1

        elif etype == "faceoff":
            w = det.get("winningPlayerId")
            l = det.get("losingPlayerId")
            if w and w in roster:
                player_stats[w]["faceoffs_won"]   += 1
                player_stats[w]["faceoffs_taken"] += 1
            if l and l in roster:
                player_stats[l]["faceoffs_taken"] += 1

        elif etype == "giveaway":
            pid = det.get("playerId")
            if pid and pid in roster:
                player_stats[pid]["giveaways"] += 1

        elif etype == "takeaway":
            pid = det.get("playerId")
            if pid and pid in roster:
                player_stats[pid]["takeaways"] += 1

    # Override hits/blocks with boxscore (more reliable)
    if box:
        for side in ["homeTeam", "awayTeam"]:
            for pos in ["forwards", "defense"]:
                for p in box.get("playerByGameStats", {}).get(side, {}).get(pos, []):
                    pid = p["playerId"]
                    player_stats[pid]["hits"]   = p.get("hits", 0)
                    player_stats[pid]["blocks"] = p.get("blockedShots", 0)

    # --- Combine into rows ---
    rows = []
    for pid in set(roster.keys()) | set(player_toi.keys()):
        if pid not in roster: continue
        r   = roster[pid]
        toi = player_toi[pid]
        st  = player_stats[pid]
        rows.append({
            "game_id":              game_id,
            "season":               season,
            "date":                 date,
            "player_id":            pid,
            "name":                 r["name"],
            "position":             r["position"],
            "team":                 r["team"],
            "opponent":             away_team if r["is_home"] else home_team,
            "is_home":              r["is_home"],
            # TOI
            "toi_total":            round(toi["total"], 3),
            "toi_ev":               round(toi["ev"],    3),
            "toi_pp":               round(toi["pp"],    3),
            "toi_sh":               round(toi["sh"],    3),
            "toi_en":               round(toi["en"],    3),
            # Individual shots by strength
            "ev_shots":             st["ev_shots"],
            "pp_shots":             st["pp_shots"],
            "sh_shots":             st["sh_shots"],
            # Individual Corsi components by strength
            "ev_missed_shots":          st["ev_missed_shots"],
            "pp_missed_shots":          st["pp_missed_shots"],
            "sh_missed_shots":          st["sh_missed_shots"],
            "ev_shots_blocked_by_opp":  st["ev_shots_blocked_by_opp"],
            "pp_shots_blocked_by_opp":  st["pp_shots_blocked_by_opp"],
            "sh_shots_blocked_by_opp":  st["sh_shots_blocked_by_opp"],
            # Total shot attempts by strength (Corsi)
            "ev_shot_attempts": st["ev_shots"] + st["ev_missed_shots"] + st["ev_shots_blocked_by_opp"],
            "pp_shot_attempts": st["pp_shots"] + st["pp_missed_shots"] + st["pp_shots_blocked_by_opp"],
            "sh_shot_attempts": st["sh_shots"] + st["sh_missed_shots"] + st["sh_shots_blocked_by_opp"],
            # Goals and assists by strength
            "ev_goals":             st["ev_goals"],
            "pp_goals":             st["pp_goals"],
            "sh_goals":             st["sh_goals"],
            "ev_assists":           st["ev_assists"],
            "pp_assists":           st["pp_assists"],
            "sh_assists":           st["sh_assists"],
            # On-ice shots
            "ev_onice_sf":          st["ev_onice_sf"],
            "ev_onice_sa":          st["ev_onice_sa"],
            "pp_onice_sf":          st["pp_onice_sf"],
            "pp_onice_sa":          st["pp_onice_sa"],
            "sh_onice_sf":          st["sh_onice_sf"],
            "sh_onice_sa":          st["sh_onice_sa"],
            # On-ice goals
            "ev_onice_gf":          st["ev_onice_gf"],
            "ev_onice_ga":          st["ev_onice_ga"],
            "pp_onice_gf":          st["pp_onice_gf"],
            "pp_onice_ga":          st["pp_onice_ga"],
            "sh_onice_gf":          st["sh_onice_gf"],
            "sh_onice_ga":          st["sh_onice_ga"],
            # Misc
            "hits":                 st["hits"],
            "blocks":               st["blocks"],
            "faceoffs_won":         st["faceoffs_won"],
            "faceoffs_taken":       st["faceoffs_taken"],
            "giveaways":            st["giveaways"],
            "takeaways":            st["takeaways"],
        })
    return rows


# --- Game list ---
def get_games_to_fetch(seasons=None, days=None):
    logs = pd.read_csv(TEAM_LOGS, low_memory=False)
    logs["date"] = pd.to_datetime(logs["date"])
    if days:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        logs   = logs[logs["date"] >= cutoff]
    elif seasons:
        logs = logs[logs["season"].isin(seasons)]
    logs = logs[logs["game_id"].astype(str).str[4:6] == "02"]
    home = logs[logs["is_home"] == True][["game_id","season","date","team"]].copy()
    home.columns = ["game_id","season","date","home_team"]
    away = logs[logs["is_home"] == False][["game_id","team"]].copy()
    away.columns = ["game_id","away_team"]
    games = home.merge(away, on="game_id").drop_duplicates("game_id").sort_values("date")
    return games[["game_id","season","date","home_team","away_team"]].values.tolist()


# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[20242025, 20252026])
    parser.add_argument("--days",    type=int,   default=None)
    parser.add_argument("--test",    action="store_true")
    args = parser.parse_args()

    games = get_games_to_fetch(
        seasons=args.seasons if not args.days else None,
        days=args.days)
    print(f"Games to process: {len(games):,}")

    if args.test:
        games = games[:3]
        print(f"TEST MODE: {len(games)} games")

    existing_games = set()
    if OUT_FILE.exists():
        existing_games = set(pd.read_csv(OUT_FILE, usecols=["game_id"])["game_id"].unique())
        print(f"Already processed: {len(existing_games):,} games")

    games = [g for g in games if int(g[0]) not in existing_games]
    print(f"New games to fetch: {len(games):,}")
    if not games:
        print("Nothing to fetch."); return

    all_rows = []
    for i, (game_id, season, date, home_team, away_team) in enumerate(games):
        if i % 50 == 0:
            print(f"  [{i}/{len(games)}] {game_id} ({str(date)[:10]} "
                  f"{home_team} vs {away_team})...")
        rows = process_game(int(game_id), int(season), str(date)[:10],
                            str(home_team), str(away_team))
        all_rows.extend(rows)

        if i % 200 == 199:
            df_new = pd.DataFrame(all_rows)
            if OUT_FILE.exists():
                df_new = pd.concat([pd.read_csv(OUT_FILE), df_new], ignore_index=True)
            df_new.drop_duplicates(["game_id","player_id"]).to_csv(OUT_FILE, index=False)
            print(f"    Checkpoint: {len(df_new):,} rows")
            all_rows = []

        time.sleep(0.3)

    if all_rows:
        df_new = pd.DataFrame(all_rows)
        if OUT_FILE.exists():
            df_new = pd.concat([pd.read_csv(OUT_FILE), df_new], ignore_index=True)
        df_new.drop_duplicates(["game_id","player_id"]).to_csv(OUT_FILE, index=False)

    final = pd.read_csv(OUT_FILE)
    print(f"\nDone! {len(final):,} rows | Columns: {list(final.columns)}")


if __name__ == "__main__":
    main()