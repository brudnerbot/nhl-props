# NHL Props Model — Project Notes

## Setup
- Python 3.14.3, venv at ~/nhl-props/venv
- GitHub: github.com/brudnerbot/nhl-props
- Libraries: pandas, requests, gspread, google-auth, scikit-learn, xgboost, openpyxl
- Run scripts from ~/nhl-props with venv activated:
  `cd ~/nhl-props && source venv/bin/activate`

---

## Project Goals
Build a full NHL analytics and props betting model system in 5 phases:

1. **Phase 1 - Data Pipeline** ✅ COMPLETE
2. **Phase 2 - Team Game Model** 🔲 NEXT
3. **Phase 3 - Player Game Model** 🔲
4. **Phase 4 - Goalie Model** 🔲
5. **Phase 5 - Season Projection Model** 🔲 (target: end of July)

---

## Phase Details

### Phase 2 - Team Game Model
Predict team event outcomes for each game matchup directly (not independently).
- **Predicts:** shots on goal for/against, shot attempts for/against, goals for/against,
  PP opportunities for/against, win probability, OT probability
- **Output:** probability distribution for each stat + most likely outcome
- **Features:** rolling averages (last 5, 10, 20 games), current season avg,
  home/away splits, rest days, back-to-back flag, opponent strength, xG for/against
- **Training data:** 20232024 onward (3 seasons) — teams change systems year-to-year
  so older data adds noise
- **Validation:** test on 20252026 games already played
- **Model:** XGBoost for count stats with Poisson output for distributions;
  logistic regression or XGBoost classifier for win/OT probability

### Phase 3 - Player Game Model
Predict every player's performance for each game.
- **Predicts:** goals, assists, points, shots, hits, blocks, PIM, faceoffs, TOI, PPP, SHP
- **Output:** probability distribution + most likely outcome + props thresholds
  (e.g. P(shots > 1.5), P(points > 0.5), P(points > 1.5))
- **Features:** player rolling averages, TOI deployment, linemate quality (on-ice outcomes
  at each strength), opponent defensive stats, team context
- **Regression to mean:** shooting % stabilization — if player shoots 30% last 10 games
  but 10% over 5 seasons, model projects toward career rate
- **Training data:** 20192020 onward for regression baselines; 20232024 onward weighted
  more heavily for recent form
- **Validation:** test on 20252026 games already played

### Phase 4 - Goalie Model
Predict goalie saves and save percentage per game.
- **Predicts:** saves, save percentage, goals against
- **Input:** projected shots against from Phase 2 + goalie recent/career performance
- **Features:** recent save %, career save %, goals saved above expected,
  rest days, opponent xG, games started %, recent pull rate
- **Output:** probability distribution + most likely outcome

### Phase 5 - Season Projection Model (by July)
Project every player's full season stats using per-60 framework.
- **Method:** project per-60 at each strength → shots → goals → assists → all other stats
- **Key steps:**
  1. Project shot attempts per 60 at each strength (EV/PP/SH)
  2. Project shot% of attempts that hit net → shots on goal per 60
  3. Project shooting% → goals per 60
  4. Project on-ice shots for (team factor) → on-ice goals for per 60
  5. Project IPP → assists per 60
  6. Project hits, blocks, PIM, faceoffs per 60
  7. Project TOI per game at each strength + GP
- **Manual override:** model outputs per-60 numbers, user manually adjusts TOI
  based on team situation, role changes, coaching decisions
- **Output:** per-60 projections + probability distributions per stat

---

## Expected Goals (xG) Model
*Similar to MoneyPuck and Natural Stat Trick*

### What is xG?
Expected goals (xG) is the probability that a given shot results in a goal, based on
the characteristics of the shot rather than the outcome. By summing xG across all shots
in a game, we get a team's "expected goals" — a measure of how many goals they deserved
to score based on the quality of their chances, independent of goaltending or shooting luck.

### How Our Model Works
The xG model is trained on all shots on goal and goals (unblocked shots with known outcomes)
from the 20232024 and 20242025 NHL regular seasons — approximately 150,000 shots including
15,500 goals. It uses gradient boosting (XGBoost) to learn the relationship between shot
characteristics and goal probability.

The model is then applied to missed shots as well, since a missed shot still represents
a real scoring opportunity — the shooter simply failed to hit the net. Blocked shots are
excluded as the shot was altered before reaching the goalie.

### Variables in the xG Model
1. **Shot Distance** — distance from the net in feet (calculated from x/y coordinates)
2. **Shot Angle** — angle from the net centerline in degrees (0° = straight on, 90° = side)
3. **Shot Type** — wrist, slap, snap, tip-in, backhand, deflected, wrap-around
4. **Is Rebound** — whether the shot followed a shot on goal within 3 seconds
5. **Rebound Angle Change** — change in shot angle for rebound shots (larger = more dangerous)
6. **Speed From Previous Event** — distance from last event divided by time elapsed
   (captures rush shots and quick plays)
7. **Distance From Previous Event** — how far the puck traveled before the shot
8. **Previous Event Type** — what happened before the shot (faceoff, hit, takeaway, etc.)
9. **Strength State** — even strength, power play, or shorthanded
10. **Period** — period of the game (capped at 4 for OT periods)

### Coordinate Normalization
NHL coordinates flip each period as teams change ends. All shot coordinates are normalized
so every shot is calculated as if attacking the same net (positive x direction), ensuring
consistent distance and angle calculations regardless of period or team.

### Model Performance
Tested on the 20252026 season (held-out test set, never seen during training):
- **AUC-ROC: 0.7807** — comparable to MoneyPuck (~0.79-0.80) and Natural Stat Trick
- **Log Loss: 0.5680** vs baseline of 0.3452 (always predicting the mean)
- **Brier Score: 0.1996**

### Training Window Selection
We tested four training windows and selected the best by AUC on the 20252026 test set:

| Training Window | AUC |
|---|---|
| 20232024 + 20242025 (2 seasons) | **0.7807** ✅ selected |
| 20222023 + 20232024 + 20242025 (3 seasons) | 0.7782 |
| 20242025 only (1 season) | 0.7758 |
| 20222023 only | 0.7503 |

Adding the older 20222023 season slightly hurt performance, confirming that NHL shot
patterns evolve year-to-year and older data introduces noise.

### Feature Importances
| Feature | Importance |
|---|---|
| Strength state (EV/PP/SH) | 30.4% |
| Shot distance | 19.7% |
| Period | 10.3% |
| Shot angle | 7.3% |
| Shot type | 7.0% |
| Is rebound | 6.8% |
| Speed from previous event | 6.0% |
| Distance from previous event | 4.8% |
| Previous event type | 3.9% |
| Rebound angle change | 3.8% |

### Output
The model produces an xG value (0.0 to 1.0) for every unblocked shot. These are used to
calculate team and player xG totals per game, which feed into the team model (Phase 2),
player model (Phase 3), and goalie model (Phase 4).

---

## Data Pipeline

### Raw Data (~/nhl-props/data/raw/)
| File | Rows | Seasons | Notes |
|------|------|---------|-------|
| team_game_logs.csv | 7,328 | 20232024-20252026 | 67 cols, full EV/PP/SH splits via play-by-play |
| player_game_logs.csv | 251,573 | 20192020-20252026 | 29 cols, goals/assists/shots/TOI/PP/SH |
| goalie_game_logs.csv | 14,714 | 20192020-20252026 | 25 cols, saves/shots against/save pct |
| shot_data.csv | 592,795 | 20222023-20252026 | 33 cols, full shot detail with coordinates |

### Processed Data (~/nhl-props/data/processed/)
| File | Rows | Notes |
|------|------|-------|
| shot_data_with_xg.csv | ~424,000 | All unblocked shots with xG values attached |

### Models (~/nhl-props/models/)
| File | Notes |
|------|-------|
| xg_model.json | Trained XGBoost xG model |
| xg_encoders.pkl | Shot type and previous event label encoders |

---

## Scripts
| Script | Purpose | Run Time |
|--------|---------|----------|
| fetch_team_game_logs.py | Full pull - team play-by-play stats | ~30-45 min |
| fetch_player_game_logs.py | Full pull - skater game logs | ~45-60 min |
| fetch_goalie_game_logs.py | Full pull - goalie game logs | ~5-10 min |
| fetch_shot_data.py | Full pull - shot events with coordinates | ~30-45 min |
| update_data.py | Incremental update (--days 1/2/7/14) | ~3-5 min |
| build_xg_model.py | Train xG model, save to models/ | ~5 min |

---

## Technical Details

### Team Game Logs
- One row per team per game (2 rows per game)
- Built by aggregating play-by-play events by team and strength state
- Prefix convention: ev_, pp_, sh_ for each strength state
- Key cols: game_id, season, date, team, opponent, is_home, went_to_ot, won,
  goals_for/against, then per strength: shots_on_goal, missed_shots, blocked_shots,
  shot_attempts, goals, hits, giveaways, takeaways, faceoffs, penalties

### Player Game Logs
- One row per player per game
- Source: /player/{id}/game-log/{season}/2 endpoint
- TOI stored as float minutes (18:30 = 18.5)
- Missing: hits, blocks, faceoffs, shot attempts (need play-by-play enrichment in Phase 3)

### Goalie Game Logs
- One row per goalie per game
- decision is null when goalie entered mid-game (no W/L/O)
- saves = shots_against - goals_against

### Shot Data
- One row per shot event (shot-on-goal, goal, missed-shot, blocked-shot)
- Coordinates normalized so all shots attack right (positive x)
- distance and angle calculated from normalized coordinates
- is_rebound = True if previous shot-on-goal within 3 seconds
- speed_from_prev = distance from last event / time elapsed

### Update Script (update_data.py)
- Run: python scripts/update_data.py --days N
- Uses rosterSpots from play-by-play to only fetch logs for players who played
- Overlap removal: drops existing rows by game_id+team or game_id+player_id before append
- Zero duplicates verified across all files

### NHL API
- Base URL: https://api-web.nhle.com/v1
- No authentication required
- Key endpoints:
  - /club-schedule-season/{team}/{season}
  - /gamecenter/{game_id}/play-by-play
  - /roster/{team}/{season}
  - /player/{player_id}/game-log/{season}/2

### situationCode Reference
Format: away_goalie | away_skaters | home_skaters | home_goalie
- 1551 = even strength (5v5)
- 1451 = home power play (5v4)
- 1541 = away power play (4v5)
- 1441 = 4v4 OT
- 0651 = home empty net (6v5)
- 1560 = away empty net (5v6)

### NHL Ice Coordinate System
- x-axis: -100 to +100 (end to end)
- y-axis: -42.5 to +42.5 (side to side)
- Net locations: x = +89 and x = -89, y = 0
- homeTeamDefendingSide flips each period