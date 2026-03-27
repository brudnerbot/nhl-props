# NHL Props Model — Project Notes

## Setup
- Python 3.14.3, venv at ~/nhl-props/venv
- GitHub: github.com/brudnerbot/nhl-props
- Run: `cd ~/nhl-props && source venv/bin/activate`

---

## Project Phases
1. **Phase 1 - Data Pipeline** COMPLETE
2. **Phase 2 - Team Game Model** COMPLETE (solid baseline, ceiling without player data)
3. **Phase 3 - Player Game Model** NEXT — will inform Phase 2 rebuild
4. **Phase 4 - Goalie Model** 
5. **Phase 5 - Season Projection Model** (by July)

### Architecture vision
Current: team rolling averages -> team model -> predictions
Target:  player lineup -> player models -> team aggregates -> team model -> predictions
Player models will unlock: lineup-aware projections, injury adjustments,
goalie matchup quality, individual hot/cold streaks.

---

## How To Run

    cd ~/nhl-props && source venv/bin/activate
    python scripts/predict_game.py HOME AWAY [DATE]
    python scripts/update_data.py --days 1

## Full Rebuild

    python scripts/fetch_team_game_logs.py      # ~30-45 min
    python scripts/fetch_player_game_logs.py    # ~45-60 min
    python scripts/fetch_goalie_game_logs.py    # ~5-10 min
    python scripts/fetch_shot_data.py           # ~30-45 min
    python scripts/build_xg_model.py            # ~5 min
    python scripts/process_team_features.py     # ~25-30 min
    python scripts/build_team_model.py          # ~10 min

---

## Current Model Performance (test set = 20252026, 1,136 games)

| Metric                        | Our Model       | MoneyPuck Benchmark |
|-------------------------------|-----------------|---------------------|
| Win AUC                       | 0.7514          | ~0.68-0.70          |
| Win Log Loss                  | 0.5890          | 0.658-0.661         |
| Goals MAE (home/away)         | 1.319 / 1.277   | N/A                 |
| xG model AUC                  | 0.7635          | N/A                 |
| Total Shots MAE (live)        | 5.002           | N/A                 |
| EV Shots MAE (live)           | 4.512           | N/A                 |
| EV Shots Bias (live)          | +0.055          | N/A                 |
| PP TOI MAE                    | 0.776 / 0.755   | N/A                 |
| Game Total Shots MAE          | 6.671           | N/A                 |
| Game Total Shots Sigma        | 8.34            | N/A                 |

---

## Prediction Output (predict_game.py)

### What is shown
- Win probability (home/away)
- Goals per team: Poisson lambda + distribution + over thresholds
- Individual shots on goal per team: EV/PP/SH breakdown, book lines, dynamic over/under table
- Game total shots: naive sum with empirical sigma, book lines, dynamic table
- xG per team: projected total + over thresholds
- Strength TOI: EV/PP/SH/EN minutes per team

### Shot output format
- Book lines: fixed thresholds (individual: 24.5/29.5/34.5, game total: 49.5/54.5/57.5/59.5/64.5)
- Dynamic table: +-5 lines in 0.5 increments centered on projection, .5 lines only
- Arrow marks lines closest to projection

### Known model limitations
- Win prob and xG can disagree — they use separate feature sets
- Per-team shot bias: high-volume teams (COL/VGK/CAR) under-predicted 2-3 shots,
  low-volume (CHI/PHI/SEA) over-predicted 2+ shots
- Goals model pred std 0.492 vs actual 1.707 — variance compression is normal
  for XGBoost regression, Poisson distribution handles uncertainty correctly
- All limitations fixable with Phase 3 player lineup data

---

## Shot Prediction Architecture

### Flow

    Total shots = own_last30 x 0.55 + opp_against_last30 x 0.45
    PP shots    = pp_shots_per60_cumulative_season x pred_pp_toi / 60
    SH shots    = weighted_sh_shots_per60 x pred_sh_toi / 60
    EV shots    = total - PP - SH
    Game total  = home_total + away_total (naive sum, sigma=8.34)

### Why total shots uses empirical formula not XGBoost
Grid searched all weight combos for own/opp/season blend — best combo improved
MAE by 0.001 over current formula. Near ceiling for pre-game prediction.

### Why game total uses naive sum not XGBoost
Dedicated XGBoost game total model tested — pred std collapsed to 1.1 shots.
No pre-game features add signal beyond naive sum. MAE 6.671, sigma 8.34.

### Why PP shots use cumulative per60 x TOI
Flat season avg over-predicts low PP TOI games (+2.9), under-predicts high (+3.1).
cumul_per60 x pred_pp_toi scales correctly with predicted ice time.

| Method                  | MAE   | Bias   |
|-------------------------|-------|--------|
| Flat season_avg (old)   | 2.739 | +0.203 |
| cumul_per60 x TOI (new) | 2.644 | -0.156 |

### TOI Framework

    PP TOI = 0.87 + own_drawn x 0.219 + opp_taken x 0.606 + own_pp_toi_season x 0.251
    SH TOI = opponent PP TOI (enforced equal)
    EV TOI = AVG_GAME_TOI (59.73) - PP TOI - SH TOI
    EN TOI = ~1.72 min avg (displayed only)
    Total displayed ~ 61.5 min

---

## Bug Fixes & Improvements (Mar 2026)

### Fix 1: Same-game penalty leakage in TOI model
Raw single-game penalty counts selected by EV TOI models.
Added 28 columns to EXCLUDE_COLS in build_team_model.py and retrained.

| Model                | Before | After        |
|----------------------|--------|--------------|
| home/away PP TOI MAE | ~2.04  | 0.776 / 0.755|
| home_won AUC         | 0.7430 | 0.7514       |

### Fix 2: PP shots back-calculation
Switched from flat season_avg to cumul_per60 x pred_pp_toi.
EV bias improved -0.307 to +0.055.

### Fix 3: xG/SOG feature inflation
ev/pp/sh_xg_per_sog_for/against were in TEAM_STATS — rolling averages
computed as average of per-game ratios = inflated 0.155 vs correct 0.104.
Removed from TEAM_STATS and WEIGHT_CONFIGS. Added correct cumulative
rolling versions: sum(xG)/sum(shots) for last10/20/30 and cumulative season.
Goals MAE improved: home 1.328->1.319, away 1.288->1.277.

### Finding: Total shots and game total near ceiling
Grid searched 3-way blends, XGBoost game total model tested.
Both near-optimal with current team-level features.
Player lineup data needed to break through ceiling.

---

## xG Model

### Definition
xG = probability that a given shot results in a goal given all observable
shot characteristics. Summed across all shots, xG equals actual goals
scored in the long run. A shot from a spot where 50% of shots go in = 0.50 xG.

### Current model (rebuilt Mar 2026)
- XGBoost + Platt scaling calibration
- AUC 0.7635 (up from 0.7610)
- Mean xG/SOG: 0.104 (correctly calibrated)
- xG/goal ratio: 1.027 (near-perfect league calibration)
- Empty net: separate logistic regression, AUC 0.7223
- Top features: distance 0.213, is_forward 0.211, period 0.089, angle 0.086
- Best training window: 2 seasons (20232024-20242025)

### Features
- distance, angle, shot_type_enc (location/geometry)
- is_rebound, rebound_angle_change (rebound context)
- speed_from_prev, prev_distance, prev_event_type_enc (play context)
- period_adj (game state proxy)
- is_forward (forward vs defenseman — 2nd most important feature)
- is_rush (speed>20 ft/s AND prev_dist>75 ft)
- strength excluded (location implicitly captures this)
- score state excluded (affects shot selection not shot quality)

### Calibration (test season, non-EN SOG)
| xG bucket | Actual rate | Predicted | Error  |
|-----------|-------------|-----------|--------|
| 0.0-0.05  | 2.8%        | 2.0%      | -0.8%  |
| 0.05-0.10 | 7.8%        | 7.3%      | -0.5%  |
| 0.10-0.15 | 12.2%       | 12.4%     | +0.2%  |
| 0.15-0.20 | 17.1%       | 17.4%     | +0.4%  |
| 0.20-0.30 | 21.5%       | 24.3%     | +2.8%  |
| 0.30-0.40 | 30.2%       | 33.9%     | +3.7%  |
| 0.40-0.60 | 41.4%       | 46.2%     | +4.8%  |
| 0.60-1.00 | 66.0%       | 73.4%     | +7.4%  |

High danger still slightly over-predicted — acceptable given data limitations.
Further improvement requires tracking data (traffic, screens, pass quality).

---

## Empirical Analysis Results

### Rolling Window Predictability
| Stat              | L10   | L20   | L30   | Season | Best    |
|-------------------|-------|-------|-------|--------|---------|
| EV Shot Attempts  | 0.185 | 0.202 | 0.202 | 0.195  | L20/L30 |
| EV Shots on Goal  | 0.214 | 0.243 | 0.247 | 0.229  | L30     |
| EV Goals          | 0.038 | 0.066 | 0.071 | 0.054  | L30     |
| EV Shooting%      | 0.027 | 0.048 | 0.055 | 0.029  | L30     |
| Block Rate        | 0.089 | 0.113 | 0.126 | 0.111  | 3yr avg |
| SOG/Fenwick       | 0.117 | 0.146 | 0.166 | 0.134  | 3yr avg |

### PP TOI Predictability
Ridge formula MAE 0.776 vs baseline 2.164 after leakage fix.
PP TOI is nearly random game-to-game (CV = 0.528).

### Weighted Funnel Rate Weights (empirical)
- Shooting%, SOG/Fenwick, block rate: 34/33/33
- PP/SH shots per 60: 60/30/10 (current season dominates)

---

## Data Pipeline

### Raw Data
| File                  | Rows    | Seasons           |
|-----------------------|---------|-------------------|
| team_game_logs.csv    | 14,504  | 20232024-20252026 |
| player_game_logs.csv  | 254,957 | 20192020-20252026 |
| goalie_game_logs.csv  | 14,910  | 20192020-20252026 |
| shot_data.csv         | 592,795 | 20222023-20252026 |

### Processed Data
| File                   | Rows    | Features       |
|------------------------|---------|----------------|
| shot_data_with_xg.csv  | 432,713 | calibrated xG  |
| team_features.csv      | 7,252   | 1,511 features |

### Key Feature Types
- Rolling averages: last10, last20, last30, season_avg
- Cumulative per-60 rates: sum(stat)/sum(TOI in hrs)
- Cumulative xG/SOG rates: sum(xG)/sum(shots) — correct, not avg of ratios
- Weighted funnel rates: 3yr weighted block%, SOG/Fenwick%, shooting%
- Goalie: save% and GSAx over last20, last40, season, career
- Penalty tracking: minor/major by strength (ev/pp/sh)

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
- Output feeds back into team model as bottom-up aggregates

### Missing player data (need PBP enrichment)
Hits, blocks, faceoffs, shot attempts not in player_game_logs.csv
Need: fetch_player_pbp_stats.py

### Why player models unlock team model ceiling
Current team model: rolling team averages (ignores who is playing)
Target: player lineup -> individual projections -> team aggregates
Unlocks: injury adjustments, goalie matchups, individual streaks,
lineup changes, home/away player splits

---

## Scripts
| Script                          | Purpose                        | Time       |
|---------------------------------|--------------------------------|------------|
| fetch_team_game_logs.py         | Team PBP + TOI + penalties     | ~30-45 min |
| fetch_player_game_logs.py       | Skater game logs               | ~45-60 min |
| fetch_goalie_game_logs.py       | Goalie game logs               | ~5-10 min  |
| fetch_shot_data.py              | Shot events with coords        | ~30-45 min |
| update_data.py                  | Incremental update             | ~3-5 min   |
| build_xg_model.py               | Train xG models                | ~5 min     |
| process_team_features.py        | Build feature dataset          | ~25-30 min |
| build_team_model.py             | Train prediction models        | ~10 min    |
| predict_game.py                 | Generate predictions           | <1 min     |
| diagnose_shot_model.py          | Shot model diagnostics         | <1 min     |
| build_game_total_shots_model.py | Game total shots (reference)   | ~2 min     |

---

## NHL API Reference
- Base: https://api-web.nhle.com/v1
- No auth required
- Situation codes: 1551=EV, 1451=home PP, 1541=away PP, 0651/1560=EN
- Ice coords: x+-100, y+-42.5, nets at x=+-89