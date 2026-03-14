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
2. **Phase 2 - Team Game Model** ✅ COMPLETE
3. **Phase 3 - Player Game Model** 🔲 NEXT
4. **Phase 4 - Goalie Model** 🔲
5. **Phase 5 - Season Projection Model** 🔲 (target: end of July)

---

## Current Status
- ✅ All raw data pulled and verified
- ✅ xG model built, calibrated, empty net model added (AUC 0.7610)
- ✅ Team features processed (3,666 matchups, 959 features)
- ✅ Team model built and validated (win prob AUC 0.7079)
- ✅ predict_game.py working — outputs win prob, goals, shots, xG, TOI
- 🔲 Build player game model (next)
- 🔲 Build goalie model
- 🔲 Build season projection model (by July)

---

## How To Run A Game Prediction
```bash
cd ~/nhl-props
source venv/bin/activate
python scripts/predict_game.py HOME_TEAM AWAY_TEAM [DATE]
# Example:
python scripts/predict_game.py EDM VAN
python scripts/predict_game.py TOR BOS 2026-03-20
```

## How To Update Data Daily
```bash
cd ~/nhl-props
source venv/bin/activate
python scripts/update_data.py --days 1   # yesterday only
python scripts/update_data.py --days 7   # last week
```

## How To Rebuild Everything From Scratch
```bash
python scripts/fetch_team_game_logs.py      # ~30-45 min
python scripts/fetch_player_game_logs.py    # ~45-60 min
python scripts/fetch_goalie_game_logs.py    # ~5-10 min
python scripts/fetch_shot_data.py           # ~30-45 min
python scripts/build_xg_model.py            # ~5 min
python scripts/process_team_features.py     # ~5 min
python scripts/build_team_model.py          # ~10 min
```

---

## Phase 2 - Team Model (COMPLETE)

### Model Performance (test set = 20252026 season)
| Target | MAE / AUC | Notes |
|---|---|---|
| home_goals_for | MAE 1.353 | Goals are random — this is near ceiling |
| away_goals_for | MAE 1.317 | Good |
| home_ev_shots_on_goal_for_per60 | MAE 5.386 | Reasonable |
| home_ev_fenwick_for_per60 | MAE 8.174 | Reasonable |
| home_xgf_total | MAE 0.918 | Good (scale ~4-5) |
| home_xgf_sog_total | MAE 0.772 | Good (scale ~3) |
| home_ev_toi | MAE 2.429 | Good |
| home_pp_toi | MAE 0.852 | Good |
| home_won | AUC 0.7079 | Solid — comparable to public models |

### Key Design Decisions
- Model the matchup directly (not each team independently)
- Exclude all raw single-game stats as features — use rolling averages only
  (raw per-60 from last game causes data leakage)
- Use rolling windows: last 5, 10, 20 games + season average
- Opponent-adjusted rolling averages for shots and goals
- Weighted multi-season PP/SH rates (50% current, 35% last, 15% two ago)
- Total shots = EV shots/60 × EV TOI + weighted PP shots/60 × PP TOI + weighted SH shots/60 × SH TOI
- SH TOI enforced = opponent PP TOI (they must be equal)
- Goalie features: save% and GSAx per 60 over last 20, 40 games, season, career

### predict_game.py Output
- Win probability (home and away)
- Goals: Poisson distribution + P(over 0.5/1.5/2.5/3.5/4.5)
- Total shots: EV+PP+SH breakdown + Normal distribution + P(over thresholds)
- xG: Normal distribution + P(over 1.5/2.5/3.5/4.5/5.5)
- Strength TOI: EV, PP, SH minutes for each team

---

## xG Model

### Two xG metrics
1. **Cumulative Shot Danger (CSD)** — sum of xG across all unblocked shots
   (shots on goal + missed shots). Scale ~4-5 per team per game.
   Used as a feature in team/player models.
2. **xG SOG** — sum of xG for shots on goal only. Scale ~3 per team per game.
   Closer to traditional xG. Used as model target and output.

### Empty Net Model
Separate logistic regression model for empty net shots.
Features: distance, angle, shot type, is_rebound, speed_from_prev, period.
AUC 0.7226, calibrated mean 0.128 vs actual 0.129.

### Main xG Model
XGBoost trained on shots on goal + goals (non-empty-net).
Calibrated with isotonic regression.
- **AUC: 0.7610** (MoneyPuck ~0.79-0.80)
- **Mean xG per SOG: 0.1078** (30 SOG × 0.1078 = 3.23 expected goals)
- **Training window:** 20232024 + 20242025 (best of 4 windows tested)
- **Strength_enc excluded** — xG should be location/context based only

### Feature Importances
| Feature | Importance |
|---|---|
| Shot distance | 29.8% |
| Period | 11.7% |
| Shot angle | 11.6% |
| Shot type | 10.9% |
| Speed from previous event | 8.9% |
| Is rebound | 7.9% |
| Distance from previous event | 7.3% |
| Previous event type | 6.5% |
| Rebound angle change | 5.4% |

---

## Phase 3 - Player Game Model (NEXT)

### What to predict per player per game
Goals, assists, points, shots on goal, hits, blocks, PIM, faceoffs won/taken, TOI, PPP, SHP

### Output per stat
- Probability distribution (Poisson for counts)
- Most likely projected outcome
- Props thresholds: P(goals > 0.5), P(points > 1.5), P(shots > 2.5), etc.

### Key modeling considerations
- **Regression to mean for shooting%:** if player shoots 30% last 10 games but
  10% over 5 seasons, project toward career rate
- **Deployment:** TOI at each strength (EV/PP/SH) is a key input
- **Linemate quality:** on-ice shot rates and goal rates at each strength
- **Opponent:** defensive shot suppression, goals against, xG against
- **Goalie quality:** starting goalie save% and GSAx affects goal probability
- **Training data:** 20192020 onward for regression baselines
- **Validation:** test on 20252026 games already played

### Missing player data (need play-by-play enrichment)
- Hits, blocks, faceoffs, shot attempts not in player_game_logs.csv
- These need to be pulled from play-by-play per game per player
- Will build fetch_player_pbp_stats.py in Phase 3

---

## Data Pipeline

### Raw Data (~/nhl-props/data/raw/)
| File | Rows | Seasons | Notes |
|------|------|---------|-------|
| team_game_logs.csv | 7,332 | 20232024-20252026 | 71 cols, EV/PP/SH splits + TOI |
| player_game_logs.csv | 251,573 | 20192020-20252026 | 29 cols, goals/assists/shots/TOI |
| goalie_game_logs.csv | 14,714 | 20192020-20252026 | 25 cols, saves/shots/save pct |
| shot_data.csv | 592,795 | 20222023-20252026 | 33 cols, full shot detail |

### Processed Data (~/nhl-props/data/processed/)
| File | Rows | Notes |
|------|------|-------|
| shot_data_with_xg.csv | 432,713 | All shots with calibrated xG values |
| team_features.csv | 3,666 | One row per game, 959 features |

### Models (~/nhl-props/models/)
| File | Notes |
|------|-------|
| xg_model.json | XGBoost xG model (non-empty-net) |
| xg_calibrator.pkl | Isotonic regression calibrator |
| xg_encoders.pkl | Shot type and prev event encoders |
| xg_en_model.pkl | Empty net xG logistic regression |
| team/home_won.json | Win probability classifier |
| team/home_goals_for.json | Home goals Poisson model |
| team/away_goals_for.json | Away goals Poisson model |
| team/home_ev_toi.json | EV TOI regression |
| team/home_pp_toi.json | PP TOI regression |
| team/away_pp_toi.json | Away PP TOI regression |
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
| build_xg_model.py | Train xG + empty net models | ~5 min |
| process_team_features.py | Build team matchup feature dataset | ~5 min |
| build_team_model.py | Train team prediction models | ~10 min |
| predict_game.py | Generate game predictions | <1 min |

---

## Technical Details

### Team Game Logs (71 cols)
- One row per team per game
- EV/PP/SH prefix for strength state splits
- ev_toi, pp_toi, sh_toi, en_toi = minutes at each strength
- Calculated by tracking situationCode transitions in play-by-play

### situationCode Reference
- 1551 = even strength (5v5)
- 1451 = home power play (5v4)
- 1541 = away power play (4v5)
- 1441 = 4v4 OT
- 0651 = home empty net
- 1560 = away empty net

### NHL Ice Coordinates
- x: -100 to +100, y: -42.5 to +42.5
- Nets at x=±89, y=0
- homeTeamDefendingSide flips each period
- All xG coordinates normalized to attack right (positive x)

### Team Features (959 cols)
- One row per game, home_ and away_ prefix
- Rolling windows: last 5, 10, 20 games
- Opponent-adjusted: raw stat minus opponent season avg
- Per-60 rolling avgs: _per60_last5, _per60_last10, _per60_season_avg
- Weighted PP/SH rates: 50% current, 35% last season, 15% two seasons ago
- Goalie: save_pct and gsax_per60 over last 20, 40, season, career

### Player Game Logs (29 cols)
- TOI as float minutes (18:30 = 18.5)
- Missing: hits, blocks, faceoffs, shot attempts (Phase 3)

### Goalie Game Logs (25 cols)
- decision null when goalie entered mid-game
- saves = shots_against - goals_against
- GSAx = saves - (shots_against × 0.906)

### NHL API
- Base: https://api-web.nhle.com/v1
- No auth required
- /club-schedule-season/{team}/{season}
- /gamecenter/{game_id}/play-by-play
- /roster/{team}/{season}
- /player/{player_id}/game-log/{season}/2