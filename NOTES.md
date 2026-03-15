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
2. **Phase 2 - Team Game Model** 🔲 IN PROGRESS — testing and refining
3. **Phase 3 - Player Game Model** 🔲
4. **Phase 4 - Goalie Model** 🔲
5. **Phase 5 - Season Projection Model** 🔲 (target: end of July)

---

## How To Run A Game Prediction
```bash
cd ~/nhl-props && source venv/bin/activate
python scripts/predict_game.py HOME_TEAM AWAY_TEAM [DATE]
python scripts/predict_game.py EDM VAN
python scripts/predict_game.py TOR BOS 2026-03-20
```

## How To Update Data Daily
```bash
python scripts/update_data.py --days 1
python scripts/update_data.py --days 7
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

## Current Status
- ✅ All raw data pulled and verified
- ✅ xG model built and calibrated (AUC 0.7610, mean xG/SOG = 0.108)
- ✅ Shot funnel rates added (block rate, SOG/Fenwick, shooting%)
- ✅ Empirically-derived feature weights (L30 best for most stats, 3yr avg for funnel rates)
- ✅ Win model beating MoneyPuck (AUC 0.7306 vs MP 0.658-0.661)
- 🔲 Fix PP TOI prediction — CRITICAL NEXT STEP
- 🔲 Fix predict_game.py single-game noise contamination
- 🔲 Build player game model
- 🔲 Build goalie model
- 🔲 Build season projection model (by July)

---

## Known Issues / Next Fixes

### CRITICAL: PP TOI Prediction Is Wrong
**Problem:** PP TOI model is predicting 10+ minutes for both teams in TOR vs MIN.
Root cause: The model uses raw single-game PP/SH per-60 stats as features,
which get contaminated by outlier games (e.g. TOR had 12.43 min PP vs ANA).
Even with rolling average substitution, the underlying issue persists.

**Correct approach:**
PP TOI is driven by penalty rates, not recent PP TOI.
- PP TOI ≈ penalties_drawn_per_game × avg_penalty_duration (~2 min)
- Penalty rates are stable season-level stats, not noisy game-to-game
- Should use: season_avg penalties drawn + opponent season_avg penalties taken
- Should NOT use: recent game PP TOI, recent PP shot attempts

**Plan:**
1. Compute team penalty draw rate (per game, season avg) → stable predictor
2. Compute team penalty take rate (per game, season avg) → stable predictor
3. PP TOI = f(home_penalty_draw_rate, away_penalty_take_rate) — simple linear model
4. Remove all noisy PP/SH per-60 features from PP TOI model

### Single-Game Noise in predict_game.py
**Problem:** predict_game.py pulls features from team's most recent game row.
If that game was an outlier (e.g. 12 min PP), it contaminates predictions.
**Fix needed:** Always use rolling averages (last20/last30/season_avg) for
per-60 rate stats, never raw single-game values.

### Data Leakage Vigilance
Raw single-game stats (shooting%, block rate, etc.) computed FROM the same
game's outcomes must be in EXCLUDE_COLS. Only rolling averages of these stats
should be used as features. This was fixed for most stats but PP/SH per-60
stats need further attention.

---

## Empirical Feature Analysis Results

### Which Windows Are Most Predictive?
From analysis of 6,279+ team-game observations:

| Stat | L5 | L10 | L20 | L30 | Season | Best |
|---|---|---|---|---|---|---|
| EV Shot Attempts | 0.156 | 0.185 | 0.202 | 0.202 | 0.195 | L20/L30 |
| EV Shots on Goal | 0.181 | 0.214 | 0.243 | 0.247 | 0.229 | L30 |
| EV Goals | 0.027 | 0.038 | 0.066 | 0.071 | 0.054 | L30 |
| EV Shooting% | 0.018 | 0.027 | 0.048 | 0.055 | 0.029 | L30 |
| Block Rate | 0.083 | 0.089 | 0.113 | 0.126 | 0.111 | L30/3yr |
| SOG/Fenwick Rate | 0.095 | 0.117 | 0.146 | 0.166 | 0.134 | L30/3yr |

**Key conclusions:**
- L5 window adds noise, not signal — DROPPED from ROLLING_WINDOWS
- Current ROLLING_WINDOWS = [10, 20, 30]
- Shooting% is nearly unpredictable (max correlation 0.055)
- SOG/Fenwick and block rate benefit from 3yr weighted averages
- Multi-season data helps for system/coaching traits, not for goals/shots

### Weighted Funnel Rate Weights (Empirically Derived)
- **Shooting%:** 34/33/33 across 3 seasons (most mean-reverting, equal weight)
- **SOG/Fenwick rate:** 34/33/33 (system trait, 3yr avg best)
- **Block rate:** 34/33/33 (coaching/system trait, 3yr avg best)
- **PP/SH shots per 60:** 60/30/10 (current season dominates)

---

## Phase 2 - Team Model

### Current Performance (test set = 20252026, 1,042 games)
| Metric | Our Model | MoneyPuck Benchmark |
|---|---|---|
| Win Accuracy | 67.4% | ~60-61% |
| Win Log Loss | 0.5991 | 0.658-0.661 |
| Win AUC | 0.7306 | ~0.68-0.70 |
| Goals MAE | 1.319 | N/A |
| EV SOG/60 MAE | 5.288 | N/A |
| PP TOI MAE | 0.862 | N/A |

### Model Architecture
- One XGBoost model per target stat
- Train on 20232024+20242025, test on 20252026
- Feature selection: top 80 features from ~1,000+ available
- Early stopping: 30 rounds on validation set
- All raw single-game stats excluded (data leakage prevention)
- Only rolling averages (L10/L20/L30/season_avg/weighted) used as features

### Targets Predicted
- Goals for (home + away) — Poisson distribution output
- EV shots on goal per 60 (home + away)
- PP/SH shots on goal per 60 (home + away)
- EV/PP Fenwick per 60 (home + away)
- EV shot attempts per 60 (home + away)
- xG total CSD (home + away)
- xG SOG total (home + away)
- EV TOI (home + away)
- PP TOI (home + away) ← currently broken, needs penalty rate fix
- Win probability (classifier)

### Total Shots Calculation
EV shots = predicted EV shots/60 × predicted EV TOI / 60
PP shots = weighted PP shots/60 × predicted PP TOI / 60
SH shots = weighted SH shots/60 × predicted SH TOI / 60
Total = EV + PP + SH

### Key Design Decisions
- Matchup modeled directly (not teams independently)
- SH TOI enforced = opponent PP TOI
- Goalie save% and GSAx included as features
- Weighted multi-season funnel rates (block%, SOG/Fenwick%, shooting%)
- Opponent-adjusted rolling averages for shots/goals

---

## xG Model

### Two xG Metrics
1. **CSD (Cumulative Shot Danger)** — all unblocked shots, scale ~4-5/game
2. **xG SOG** — shots on goal only, scale ~3/game (traditional xG)

### Main xG Model (non-empty-net)
- XGBoost + isotonic regression calibration
- AUC: 0.7610, mean xG/SOG: 0.1078 (30 SOG × 0.108 = 3.23 xG)
- Training: 20232024+20242025 (best of 4 windows tested)
- strength_enc EXCLUDED — xG is location/context based only

### Empty Net Model
- Logistic regression on distance, angle, shot type, rebound
- AUC 0.7226, calibrated mean 0.128 vs actual 0.129

### Feature Importances
| Feature | Importance |
|---|---|
| Shot distance | 29.8% |
| Period | 11.7% |
| Shot angle | 11.6% |
| Shot type | 10.9% |
| Speed from previous event | 8.9% |
| Is rebound | 7.9% |

---

## Data Pipeline

### Raw Data (~/nhl-props/data/raw/)
| File | Rows | Seasons | Notes |
|------|------|---------|-------|
| team_game_logs.csv | 7,332 | 20232024-20252026 | 71 cols, EV/PP/SH splits + TOI |
| player_game_logs.csv | 251,573 | 20192020-20252026 | 29 cols |
| goalie_game_logs.csv | 14,714 | 20192020-20252026 | 25 cols |
| shot_data.csv | 592,795 | 20222023-20252026 | 33 cols |

### Processed Data (~/nhl-props/data/processed/)
| File | Rows | Notes |
|------|------|-------|
| shot_data_with_xg.csv | 432,713 | All shots with calibrated xG |
| team_features.csv | 3,666 | One row per game, 1,319 features |

### Models (~/nhl-props/models/)
| File | Notes |
|------|-------|
| xg_model.json | XGBoost xG model |
| xg_calibrator.pkl | Isotonic calibrator |
| xg_encoders.pkl | Shot type + prev event encoders |
| xg_en_model.pkl | Empty net logistic regression |
| team/home_won.json | Win probability |
| team/home_goals_for.json | Home goals Poisson |
| team/[targets].json | Per-60 rate + TOI models |
| team/feature_lists.pkl | Top 80 features per model |
| team/residual_stds.pkl | Sigma for Normal distributions |

---

## Scripts
| Script | Purpose | Run Time |
|--------|---------|----------|
| fetch_team_game_logs.py | Full pull - team play-by-play + TOI | ~30-45 min |
| fetch_player_game_logs.py | Full pull - skater game logs | ~45-60 min |
| fetch_goalie_game_logs.py | Full pull - goalie game logs | ~5-10 min |
| fetch_shot_data.py | Full pull - shot events | ~30-45 min |
| update_data.py | Incremental update (--days N) | ~3-5 min |
| build_xg_model.py | Train xG + empty net models | ~5 min |
| process_team_features.py | Build team matchup feature dataset | ~5 min |
| build_team_model.py | Train team prediction models | ~10 min |
| predict_game.py | Generate game predictions | <1 min |

---

## Technical Details

### Shot Funnel Rates (computed per game, rolling avgs used as features)
Shot Attempts (Corsi) → Fenwick (unblocked) → SOG → Goals
- ev_fenwick_rate_for = fenwick_for / shot_attempts_for
- ev_block_rate_against = 1 - fenwick_rate_for
- ev_sog_fenwick_rate_for = sog_for / fenwick_for
- ev_shooting_pct = goals_for / sog_for
- ev_xg_per_sog_for = xgf / sog_for
Defensive versions computed for all of the above.

### situationCode Reference
- 1551 = even strength (5v5)
- 1451 = home PP (5v4)
- 1541 = away PP (4v5)
- 0651/1560 = empty net

### NHL Ice Coordinates
- x: -100 to +100, y: -42.5 to +42.5
- Nets at x=±89, y=0
- Normalized: all shots calculated attacking right (positive x)

### GSAx Calculation
GSAx = saves - (shots_against × 0.906)
where 0.906 = NHL average save percentage