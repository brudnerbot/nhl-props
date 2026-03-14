# NHL Props Model — Project Notes

## Setup
- Python 3.14.3, venv at ~/nhl-props/venv
- GitHub: github.com/brudnerbot/nhl-props
- Libraries: pandas, requests, gspread, google-auth, scikit-learn, xgboost, openpyxl

## Project Goals
1. **Phase 2 - Team Model:** Project team event outcomes per game (shot attempts, shots on
   goal, goals scored, PP opportunities, all for/against, win probability, OT probability).
   Output: probability distribution + most likely outcome for each stat.
2. **Phase 3 - Player Game Model:** Project every player's game performance (goals, assists,
   points, shots, hits, blocks, PIM, faceoffs, TOI, PPP, SHP). Account for deployment,
   linemate quality, opponent. Output: probability distribution + most likely outcome.
   Also used for props betting (e.g. over/under 1.5 shots, over 0.5 points, etc.)
3. **Phase 4 - Goalie Model:** Project goalie saves and save percentage per game, based on
   projected shots against (from Phase 2) and goalie recent/career performance.
   Output: probability distribution + most likely outcome.
4. **Phase 5 - Season Projection Model (by July):** Per-60 projections for every player,
   per strength state. Manual TOI override capability. Probability distributions per stat.

## Modeling Approach
- **Validation:** Test all models on games already played in 20252026 season
- **Output format:** For every predicted stat, produce:
  - Full probability distribution (e.g. P(goals=0), P(goals=1), P(goals=2+))
  - Most likely projected outcome (point estimate)
  - Props thresholds (e.g. P(shots > 1.5), P(points > 0.5), P(points > 1.5))
- **Feature weighting:** Model should learn optimal weights for:
  - Last 5 games
  - Last 10 games
  - Full current season
  - Multi-season history (for regression to mean)
- **Regression to mean:** Key for shooting % - if player shoots 30% last 10 games
  but 10% over 5 seasons, model should not project 30% forward
- **Strength state splits:** EV, PP, SH tracked separately throughout

## Data Collected
All raw data saved to ~/nhl-props/data/raw/

| File | Rows | Seasons | Notes |
|------|------|---------|-------|
| team_game_logs.csv | 7,328 | 20232024-20252026 | 67 cols, full strength state splits via play-by-play |
| player_game_logs.csv | 251,573 | 20192020-20252026 | 29 cols, goals/assists/shots/TOI/PP/SH splits |
| goalie_game_logs.csv | 14,714 | 20192020-20252026 | 25 cols, saves/shots against/save pct/decision |

## Scripts
| Script | Purpose | Run Time |
|--------|---------|----------|
| fetch_team_game_logs.py | Full pull - team play-by-play stats | ~30-45 min |
| fetch_player_game_logs.py | Full pull - skater game logs | ~45-60 min |
| fetch_goalie_game_logs.py | Full pull - goalie game logs | ~5-10 min |
| update_data.py | Incremental update (--days 1/2/7/14) | ~3-5 min |

## Current Status
- ✅ Phase 1 complete - all raw data pulled, verified, committed
- ✅ Update script working - zero duplicates verified
- 🔲 Phase 2 - Team model (next)
- 🔲 Phase 3 - Player game model
- 🔲 Phase 4 - Goalie model
- 🔲 Phase 5 - Season projection model (by July)

## Key Design Decisions
- Team logs only go back 3 seasons (systems/personnel change year-to-year)
- Player/goalie logs go back 7 seasons (regression baselines, shooting % stabilization)
- Strength state splits (EV/PP/SH) pulled from play-by-play for team logs
- CSVs for storage now, can migrate to SQLite later if needed
- Update script uses rosterSpots to only check players who actually played
- Update script pulls last N days and overwrites overlap to avoid duplicates

## Key NHL API Endpoints
- Base URL: https://api-web.nhle.com/v1
- No authentication required
- /club-schedule-season/{team}/{season} — game IDs and dates
- /gamecenter/{game_id}/play-by-play — full event data + roster spots
- /roster/{team}/{season} — player/goalie IDs and bio info
- /player/{player_id}/game-log/{season}/2 — player/goalie game logs

## situationCode Reference
Format: away_goalie | away_skaters | home_skaters | home_goalie
- 1551 = even strength (5v5)
- 1451 = home power play (5v4)
- 1541 = away power play (4v5)
- 1441 = 4v4 (OT)
- 0651 = home empty net (6v5)
- 1560 = away empty net (5v6)

## Technical Details for Onboarding

### Team Game Logs (team_game_logs.csv)
- One row per team per game (2 rows per game)
- Built by fetching play-by-play for every game and aggregating events
- Strength splits: ev_, pp_, sh_ prefix for every stat column
- Key columns: game_id, season, date, team, opponent, is_home, went_to_ot, won,
  goals_for, goals_against, then for each strength (ev/pp/sh):
  shots_on_goal_for/against, missed_shots_for/against, blocked_shots_for/against,
  shot_attempts_for/against, goals_for/against, hits_for/against,
  giveaways, takeaways, faceoffs_won, faceoffs_taken,
  penalties_taken, penalties_drawn, penalty_minutes

### Player Game Logs (player_game_logs.csv)
- One row per player per game
- Built from /player/{id}/game-log/{season}/2 endpoint
- Key columns: player_id, first_name, last_name, position, shoots, height_in,
  weight_lbs, birth_date, birth_country, season, game_id, date, team, opponent,
  is_home, toi, goals, assists, points, plus_minus, shots, pim, shifts,
  pp_goals, pp_points, sh_goals, sh_points, gw_goals, ot_goals
- TOI stored as float minutes (e.g. 18:30 = 18.5)
- Missing: hits, blocks, faceoffs, shot attempts (need play-by-play enrichment later)

### Goalie Game Logs (goalie_game_logs.csv)
- One row per goalie per game
- Built from same game-log endpoint as players
- Key columns: player_id, first_name, last_name, catches, height_in, weight_lbs,
  birth_date, birth_country, season, game_id, date, team, opponent, is_home,
  toi, games_started, decision, shots_against, goals_against, saves,
  save_pct, shutouts, goals, assists, pim
- decision is null when goalie entered mid-game and didn't get a W/L/O

### Update Script Logic (update_data.py)
- Run with: python scripts/update_data.py --days N
- Step 1: scan schedules for game IDs in date range
- Step 2: fetch play-by-play for each game, cache it, extract rosterSpots
- Step 3: reuse cached play-by-play to update team logs (no extra API calls)
- Step 4: only fetch game logs for players/goalies who appeared in target games
- Overlap removal: drops existing rows matching game_id+team or game_id+player_id
  before appending new rows, guaranteeing zero duplicates