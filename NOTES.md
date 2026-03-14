# NHL Props Model — Project Notes

## Setup
- Python 3.14.3, venv at ~/nhl-props/venv
- GitHub: github.com/brudnerbot/nhl-props
- Libraries: pandas, requests, gspread, google-auth, scikit-learn, xgboost, openpyxl

## Project Goals
1. **Phase 2 - Team Model:** Project team event outcomes per game (shot attempts, goals,
   PP opportunities, win probability, OT probability) with probability distributions
2. **Phase 3 - Player Game Model:** Project every player's game performance per stat
   with probability distributions for props betting
3. **Phase 4 - Goalie Model:** Project goalie save totals based on projected shots against
   and save percentage
4. **Phase 5 - Season Projection Model (by July):** Per-60 projections for every player,
   per strength state, with manual TOI override capability

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
| update_data.py | Incremental daily update - last 7 days | ~5-10 min |

## Current Status
- ✅ Phase 1 complete - all raw data pulled and saved
- ✅ Incremental update script built (update_data.py)
- 🔲 Phase 2 - Team model (next)
- 🔲 Phase 3 - Player game model
- 🔲 Phase 4 - Goalie model
- 🔲 Phase 5 - Season projection model (by July)

## Key Design Decisions
- Team logs only go back 3 seasons (systems/personnel change too much year-to-year)
- Player/goalie logs go back 7 seasons (for regression baselines and shooting % stabilization)
- Strength state splits (EV/PP/SH) pulled from play-by-play for team logs
- CSVs used for storage now, can migrate to SQLite later if needed
- Update script pulls last 7 days and overwrites any overlap to avoid duplicates

## Notes on NHL API
- Base URL: https://api-web.nhle.com/v1
- No authentication required
- Key endpoints:
  - /club-schedule-season/{team}/{season} — game IDs
  - /gamecenter/{game_id}/play-by-play — full event data
  - /roster/{team}/{season} — player/goalie IDs
  - /player/{player_id}/game-log/{season}/2 — player game logs
- situationCode format: away_goalie | away_skaters | home_skaters | home_goalie
  - 1551 = even strength, 1451 = home PP, 1541 = away PP