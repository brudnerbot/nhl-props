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
2. **Phase 2 - Team Game Model** 🔲 IN PROGRESS
3. **Phase 3 - Player Game Model** 🔲
4. **Phase 4 - Goalie Model** 🔲
5. **Phase 5 - Season Projection Model** 🔲 (target: end of July)

---

## Current Status
- ✅ All raw data pulled and verified
- ✅ xG model built (AUC 0.7807)
- ✅ Team features processed (3,666 matchups, 895 features)
- ✅ Team model built (win prob AUC 0.6971, goals MAE 1.349)
- 🔲 Fix PP/SH shot prediction approach — NEXT
- 🔲 Build total shots output (rate × TOI)
- 🔲 Build prediction output script
- 🔲 Build player model
- 🔲 Build goalie model
- 🔲 Build season projection model

---

## Phase Details

### Phase 2 - Team Game Model
Predict team event outcomes for each game as a direct matchup.
Each training row = one game with both teams' features combined.

**Current Model Performance (test set = 20252026 season):**
| Target | MAE / AUC | Notes |
|---|---|---|
| home_goals_for | MAE 1.349 | Good, hockey goals are random |
| away_goals_for | MAE 1.316 | Good |
| home_ev_shots_on_goal_for_per60 | MAE 5.318 | Reasonable |
| away_ev_shots_on_goal_for_per60 | MAE 5.189 | Reasonable |
| home_pp_shots_on_goal_for_per60 | MAE 23.614 | Too noisy - fix needed |
| away_pp_shots_on_goal_for_per60 | MAE 26.082 | Too noisy - fix needed |
| home_sh_shots_on_goal_for_per60 | MAE 7.516 | Noisy |
| home_ev_fenwick_for_per60 | MAE 8.124 | Reasonable |
| home_pp_fenwick_for_per60 | MAE 41.914 | Too noisy - fix needed |
| home_xgf_total | MAE 2.885 | Good |
| home_ev_toi | MAE 2.433 | Good |
| home_pp_toi | MAE 0.856 | Good |
| away_pp_toi | MAE 0.782 | Good |
| home_won | AUC 0.6971 | Excellent - matches MoneyPuck |

**Known Issues / Next Fixes:**
- PP and SH per-60 shots are too noisy on a per-game basis
- Fix: predict PP TOI (already working) × season avg PP shots per 60 = total PP shots
- Need a prediction output script that converts per-60 rates × TOI into total shots
- PerformanceWarning in process_team_features.py (harmless, fix with pd.concat)

**Output approach (total shots):**
1. Predict EV shots per 60 → multiply by predicted EV TOI → EV shots
2. Use season avg PP shots per 60 → multiply by predicted PP TOI → PP shots
3. Use season avg SH shots per 60 → multiply by predicted SH TOI → SH shots
4. Sum for total shots on goal
5. Apply Normal distribution around total for probability thresholds

**Key features used:**
- Rolling averages last 5, 10, 20 games (opponent-adjusted)
- Current season averages
- Per-60 rate stats at each strength (EV/PP/SH)
- TOI at each strength
- xG for/against (total + by strength)
- Fenwick for/against per 60
- Goalie save% and GSAx (last 20, 40 games, season, career)
- Home/away flag, days rest, back-to-back flag

**Overfitting protection:**
- Feature selection: top 80 features from 670 available
- XGBoost early stopping (30 rounds)
- Validation split: last 20% of training data

### Phase 3 - Player Game Model
Predict every player's performance for each game.
- **Predicts:** goals, assists, points, shots, hits, blocks, PIM, faceoffs, TOI, PPP, SHP
- **Output:** probability distribution + most likely outcome + props thresholds
- **Features:** player rolling averages, TOI deployment, linemate quality,
  opponent defensive stats, team context, goalie quality
- **Regression to mean:** shooting % stabilization over 5+ seasons
- **Training data:** 20192020 onward for regression baselines
- **Validation:** test on 20252026 games already played

### Phase 4 - Goalie Model
Predict goalie saves and save percentage per game.
- **Input:** projected shots against from Phase 2 + goalie performance history
- **Features:** recent save%, career save%, GSAx, rest days, opponent xG
- **Output:** probability distribution + most likely outcome

### Phase 5 - Season Projection Model (by July)
Per-60 projections for every player using the framework:
1. Shot attempts per 60 → shots on goal per 60 → goals per 60
2. On-ice shots for (team factor) → on-ice goals → IPP → assists per 60
3. Hits, blocks, PIM, faceoffs per 60
4. TOI per game at each strength + GP projection
Manual TOI override capability built in.

---

## Expected Goals (xG) Model

### What is xG?
Expected goals (xG) = probability a shot results in a goal based on shot
characteristics, independent of goaltending or shooting luck.

### How Our Model Works
Trained on ~150,000 shots on goal + goals from 20232024 and 20242025 seasons.
Applied to missed shots too (missed shot = real scoring opportunity, shooter missed net).
Blocked shots excluded (shot was altered before reaching goalie).

### Variables
1. Shot distance from net
2. Shot angle from net centerline
3. Shot type (wrist, slap, snap, tip-in, backhand, deflected, wrap-around)
4. Is rebound (previous shot on goal within 3 seconds)
5. Rebound angle change
6. Speed from previous event (rush shots)
7. Distance from previous event
8. Previous event type (faceoff, hit, takeaway, etc.)
9. Strength state (EV/PP/SH)
10. Period (capped at 4 for OT)

### Performance
- **AUC-ROC: 0.7807** (MoneyPuck ~0.79-0.80)
- **Log Loss: 0.5680** vs baseline 0.3452
- **Best training window:** 20232024 + 20242025 (3 seasons hurt performance)

### Feature Importances
| Feature | Importance |
|---|---|
| Strength state | 30.4% |
| Shot distance | 19.7% |
| Period | 10.3% |
| Shot angle | 7.3% |
| Shot type | 7.0% |
| Is rebound | 6.8% |
| Speed from previous event | 6.0% |
| Distance from previous event | 4.8% |
| Previous event type | 3.9% |
| Rebound angle change | 3.8% |

---

## Data Pipeline

### Raw Data (~/nhl-props/data/raw/)
| File | Rows | Seasons | Notes |
|------|------|---------|-------|
| team_game_logs.csv | 7,332 | 20232024-20252026 | 71 cols, EV/PP/SH splits + strength TOI |
| player_game_logs.csv | 251,573 | 20192020-20252026 | 29 cols |
| goalie_game_logs.csv | 14,714 | 20192020-20252026 | 25 cols |
| shot_data.csv | 592,795 | 20222023-20252026 | 33 cols, full shot detail |

### Processed Data (~/nhl-props/data/processed/)
| File | Rows | Notes |
|------|------|-------|
| shot_data_with_xg.csv | 424,041 | Unblocked shots with xG values |
| team_features.csv | 3,666 | One row per game, 895 features, both teams |

### Models (~/nhl-props/models/)
| File | Notes |
|------|-------|
| xg_model.json | XGBoost xG model |
| xg_encoders.pkl | Shot type and prev event encoders |
| team/home_goals_for.json | Goals model (home) |
| team/away_goals_for.json | Goals model (away) |
| team/home_won.json | Win probability model |
| team/home_pp_toi.json | PP TOI model (home) |
| team/away_pp_toi.json | PP TOI model (away) |
| team/home_ev_toi.json | EV TOI model |
| team/[other targets].json | Per-60 rate models |
| team/feature_lists.pkl | Top 80 features per model |
| team/residual_stds.pkl | Sigma values for Normal distributions |

---

## Scripts
| Script | Purpose | Run Time |
|--------|---------|----------|
| fetch_team_game_logs.py | Full pull - team play-by-play + TOI | ~30-45 min |
| fetch_player_game_logs.py | Full pull - skater game logs | ~45-60 min |
| fetch_goalie_game_logs.py | Full pull - goalie game logs | ~5-10 min |
| fetch_shot_data.py | Full pull - shot events with coordinates | ~30-45 min |
| update_data.py | Incremental update (--days 1/2/7/14) | ~3-5 min |
| build_xg_model.py | Train xG model | ~5 min |
| process_team_features.py | Build team matchup feature dataset | ~5 min |
| build_team_model.py | Train team prediction models | ~10 min |

---

## Technical Details

### Team Game Logs (71 cols)
- One row per team per game (2 rows per game)
- EV/PP/SH prefix for strength state splits
- New: ev_toi, pp_toi, sh_toi, en_toi (minutes at each strength)
- Calculated by tracking situationCode transitions in play-by-play
- situationCode: away_goalie|away_skaters|home_skaters|home_goalie
  - 1551=EV, 1451=home PP, 1541=away PP, 0651=home EN

### Team Features (895 cols)
- One row per game, home_ and away_ prefix
- Rolling windows: last 5, 10, 20 games (opponent-adjusted for key stats)
- Per-60 rates: ev_, pp_, sh_ prefix + _per60 suffix
- Goalie features: save% and GSAx per 60 over last 20, 40 games, season, career
- Primary goalie = most TOI in that game

### Total Shots Calculation (planned)
- EV shots = predicted EV shots/60 × predicted EV TOI / 60
- PP shots = season avg PP shots/60 × predicted PP TOI / 60
- SH shots = season avg SH shots/60 × predicted SH TOI / 60
- Total = EV + PP + SH shots

### Player Game Logs (29 cols)
- TOI as float minutes (18:30 = 18.5)
- Missing: hits, blocks, faceoffs, shot attempts (play-by-play enrichment in Phase 3)

### Goalie Game Logs (25 cols)
- decision null when goalie entered mid-game
- saves = shots_against - goals_against
- GSAx = saves - (shots_against × 0.906) [NHL avg save pct]

### Update Script
- python scripts/update_data.py --days N
- Uses rosterSpots to only check players who played
- Zero duplicates verified

### NHL API
- Base: https://api-web.nhle.com/v1
- /club-schedule-season/{team}/{season}
- /gamecenter/{game_id}/play-by-play
- /roster/{team}/{season}
- /player/{player_id}/game-log/{season}/2

### NHL Ice Coordinates
- x: -100 to +100, y: -42.5 to +42.5
- Nets at x=±89, y=0
- homeTeamDefendingSide flips each period