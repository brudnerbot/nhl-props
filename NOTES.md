# NHL Props Model — Project Notes

## Setup
- Python 3.14.3, venv at ~/nhl-props/venv
- GitHub: github.com/brudnerbot/nhl-props
- Run: `cd ~/nhl-props && source venv/bin/activate`

---

## Project Phases
1. **Phase 1 - Data Pipeline** ✅ COMPLETE
2. **Phase 2 - Team Game Model** ✅ FUNCTIONALLY COMPLETE — testing/refining
3. **Phase 3 - Player Game Model** 🔲 NEXT
4. **Phase 4 - Goalie Model** 🔲
5. **Phase 5 - Season Projection Model** 🔲 (by July)

---

## How To Run
```bash
cd ~/nhl-props && source venv/bin/activate
python scripts/predict_game.py HOME AWAY [DATE]   # e.g. python scripts/predict_game.py EDM VAN
python scripts/update_data.py --days 1             # daily update
```

## Full Rebuild
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

## Current Model Performance (test set = 20252026, 1,057 games)

| Metric | Our Model | MoneyPuck Benchmark |
|---|---|---|
| Win AUC | **0.7430** | ~0.68-0.70 |
| Win Accuracy | **67.4%** | ~60-61% |
| Win Log Loss | **0.5991** | 0.658-0.661 |
| Goals MAE | 1.316 | N/A |
| EV SOG/60 MAE | 5.199 | N/A |
| Total Shots MAE (full set) | 5.378 | N/A |
| Total Shots MAE (batch) | 4.30 | N/A |

---

## Shot Prediction Architecture

### Flow
```
EV shots = XGBoost EV shots/60 × predicted EV TOI
PP shots = team season avg PP shots directly (no rate × TOI)
SH shots = weighted SH shots/60 × predicted SH TOI
Total shots = EV + PP + SH
```

### Why PP shots use season avg directly
PP shots per-60 rate is unreliable — short PP games inflate the rate enormously.
Example: CAR gets 6 shots in 3 min PP = 118/60, vs 6 shots in 6 min = 60/60.
The season average total PP shots (e.g. CAR = 5.4/game) is stable and accurate.

### TOI Framework
```
PP TOI = 0.87 + own_drawn×0.219 + opp_taken×0.606 + own_pp_toi_season×0.251
SH TOI = opponent PP TOI (enforced equal)
EV TOI = AVG_GAME_TOI (59.73) - PP TOI - SH TOI
EN TOI = ~1.72 min avg (displayed only, shots already in EV/PP/SH buckets)
Total displayed = EV + PP + SH + EN ≈ 61.5 min
```

### Known Limitation: COL PP TOI
Colorado draws many penalties AND has a bad PP (full 2 min used per penalty).
Their actual PP TOI (~7.2 min) is systematically under-predicted (~5.0 min).
This causes EV TOI to be over-predicted → EV shots slightly over-predicted.
Root cause: PP TOI prediction has hard ceiling (MAE ~2.0, barely beats baseline 2.07).
No formula can predict COL-style outliers from pre-game data alone.

### Cumulative Per-60 Rates (KEY FIX)
Old approach: average per-game per-60 rates → inflated by short-TOI games
New approach: sum(stat) / (sum(TOI) / 60) → matches NaturalStatTrick
Example: CAR PP SOG/60 was 102.5 (wrong) → now correctly ~56/60

---

## Empirical Analysis Results

### Rolling Window Predictability
| Stat | L10 | L20 | L30 | Season | Best |
|---|---|---|---|---|---|
| EV Shot Attempts | 0.185 | 0.202 | 0.202 | 0.195 | L20/L30 |
| EV Shots on Goal | 0.214 | 0.243 | 0.247 | 0.229 | L30 |
| EV Goals | 0.038 | 0.066 | 0.071 | 0.054 | L30 |
| EV Shooting% | 0.027 | 0.048 | 0.055 | 0.029 | L30 |
| Block Rate | 0.089 | 0.113 | 0.126 | 0.111 | 3yr avg |
| SOG/Fenwick | 0.117 | 0.146 | 0.166 | 0.134 | 3yr avg |

Key: L5 dropped (adds noise). ROLLING_WINDOWS = [10, 20, 30]

### PP TOI Predictability
Best predictor: own PP TOI season avg (corr 0.104)
Opponent minor penalties taken (corr 0.115) adds small signal
Ridge formula MAE 2.043 vs baseline 2.066 — barely better than mean
PP TOI is nearly random game-to-game (CV = 0.528)

### Weighted Funnel Rate Weights (empirical)
- Shooting%, SOG/Fenwick, block rate: 34/33/33 (all stable system traits)
- PP/SH shots per 60: 60/30/10 (current season dominates)

---

## xG Model
- XGBoost + isotonic calibration, AUC 0.7610
- Mean xG/SOG: 0.108 (30 SOG × 0.108 = 3.23 xG)
- Empty net: separate logistic regression, AUC 0.7226
- strength_enc excluded (xG is location/context based)
- Top features: distance 29.8%, period 11.7%, angle 11.6%, shot type 10.9%

---

## Data Pipeline

### Raw Data
| File | Rows | Seasons |
|---|---|---|
| team_game_logs.csv | 7,362 | 20232024-20252026 |
| player_game_logs.csv | 251,573 | 20192020-20252026 |
| goalie_game_logs.csv | 14,714 | 20192020-20252026 |
| shot_data.csv | 592,795 | 20222023-20252026 |

### Processed Data
| File | Rows | Features |
|---|---|---|
| shot_data_with_xg.csv | 432,713 | calibrated xG |
| team_features.csv | 3,681 | 1,503 features |

### Key Feature Types
- Rolling averages: last10, last20, last30, season_avg
- Cumulative per-60 rates: sum(stat)/sum(TOI in hrs) — correct, not avg of ratios
- Weighted funnel rates: 3yr weighted block%, SOG/Fenwick%, shooting%
- Goalie: save% and GSAx over last20, last40, season, career
- Penalty tracking: minor/major by strength (ev/pp/sh)
- Won (rolling win%) added to TEAM_STATS for form signal

### Minor Penalty Tracking (added Mar 2026)
fetch_team_game_logs.py now tracks:
- ev/pp/sh minor_penalties_taken/drawn
- ev/pp/sh major_penalties_taken/drawn  
- total_minor_penalties_taken/drawn (all strengths)
Used as primary input to PP TOI formula

---

## Phase 3 - Player Game Model (NEXT)

### Stats to predict per player per game
Goals, assists, points, shots on goal, hits, blocks, TOI, PPP, SHP, faceoffs

### Key modeling considerations
- Regression to mean for shooting% (career + multi-season weighted)
- Deployment: EV/PP/SH TOI as key inputs
- Linemate quality: on-ice shot rates at each strength
- Opponent defensive quality + goalie GSAx
- Training: 20192020 onward

### Missing player data (need PBP enrichment)
Hits, blocks, faceoffs, shot attempts not in player_game_logs.csv
Need: fetch_player_pbp_stats.py

---

## Scripts
| Script | Purpose | Time |
|---|---|---|
| fetch_team_game_logs.py | Team PBP + TOI + penalties | ~30-45 min |
| fetch_player_game_logs.py | Skater game logs | ~45-60 min |
| fetch_goalie_game_logs.py | Goalie game logs | ~5-10 min |
| fetch_shot_data.py | Shot events with coords | ~30-45 min |
| update_data.py | Incremental update | ~3-5 min |
| build_xg_model.py | Train xG models | ~5 min |
| process_team_features.py | Build feature dataset | ~5 min |
| build_team_model.py | Train prediction models | ~10 min |
| predict_game.py | Generate predictions | <1 min |

---

## NHL API Reference
- Base: https://api-web.nhle.com/v1
- No auth required
- Situation codes: 1551=EV, 1451=home PP, 1541=away PP, 0651/1560=EN
- Ice coords: x±100, y±42.5, nets at x=±89

## Bug Fixes & Improvements (Mar 2026)

### Fix 1: Same-game penalty leakage in TOI model
Raw single-game penalty counts (home/away_ev/pp/sh/total_minor/major_penalties_taken/drawn)
were in the feature pool and being selected by the TOI model. These are same-game actuals
(mean ~3.1/game, max 17-21) that wouldn't be available pre-game.
Added all 28 columns to EXCLUDE_COLS in build_team_model.py and retrained.

Results after fix:
| Model        | Before  | After   |
|---|---|---|
| home_pp_toi MAE | ~2.043 | 0.795 |
| away_pp_toi MAE | ~2.092 | 0.717 |
| home_won AUC    | 0.7430 | 0.7530 |

### Fix 2: PP shots back-calculation method
predict_game.py was using flat pp_shots_season_avg (~5.2) regardless of predicted PP TOI.
This over-predicted low PP TOI games (+2.9 shots) and under-predicted high PP TOI games (-3.1).
Switched to: pp_shots = pp_shots_per60_cumulative_season × (pred_pp_toi / 60)

Results:
| Method | MAE | Bias |
|---|---|---|
| Flat season_avg (old) | 2.739 | +0.203 |
| cumul_per60 × TOI (new) | 2.644 | -0.156 |

### Empirical finding: total shots formula near ceiling
Grid searched 3-way blends of own_last30 / opp_against_last30 / season_avg.
Best combo (own=0.38, opp=0.46, season=0.15) improved test MAE by only 0.001.
Current formula (own_last30×0.55 + opp_against_last30×0.45) is near-optimal.
Per-team bias in extreme teams (COL, VGK under; PHI, CHI over) is irreducible
from pre-game data — it reflects true game-to-game variance in shot volume.