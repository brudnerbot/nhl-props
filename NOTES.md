# NHL Props Model — Project Notes

## Setup
- Python 3.14.3, venv at ~/nhl-props/venv
- GitHub: github.com/brudnerbot/nhl-props
- Run: `cd ~/nhl-props && source venv/bin/activate`

---

## Project Phases
1. **Phase 1 - Data Pipeline** COMPLETE
2. **Phase 2 - Team Game Model** COMPLETE
3. **Phase 3 - Player Props Model** MOSTLY COMPLETE — props pipeline working end-to-end, shot/goal/point calibration next
4. **Phase 4 - Goalie Model** NEXT
5. **Phase 5 - Season Projection Model** (by July)

### Architecture Vision
Current: team rolling averages → team model → predictions
Target:  player lineup → player models → team aggregates → team model → predictions
Player models unlock: lineup-aware projections, injury adjustments, goalie matchup quality, individual hot/cold streaks.

---

## How To Run

    cd ~/nhl-props && source venv/bin/activate
    python scripts/predict_game.py HOME AWAY [DATE]
    python scripts/predict_player_props.py HOME AWAY [--date YYYY-MM-DD]
    python scripts/update_data.py --days 1

## Daily Update Workflow

    python scripts/fetch_daily_lineups.py                     # scrape today's lineups (~1 min)
    python scripts/fetch_player_pbp_stats.py --days 3        # update recent games (~5 min)
    python scripts/process_player_features.py                # rebuild features (~10 min)
    python scripts/predict_player_props.py HOME AWAY         # generate player props
    python scripts/predict_game.py HOME AWAY                 # generate game prediction

## Full Rebuild (all seasons from scratch)

    python scripts/fetch_team_game_logs.py                   # ~30-45 min
    python scripts/fetch_player_game_logs.py                 # ~45-60 min
    python scripts/fetch_goalie_game_logs.py                 # ~5-10 min
    python scripts/fetch_shot_data.py                        # ~30-45 min
    python scripts/build_xg_model.py                         # ~5 min
    python scripts/fetch_player_pbp_stats.py --seasons 20202021 20212022 20222023 20232024 20242025 20252026  # ~3 hrs
    python scripts/process_team_features.py                  # ~25-30 min
    python scripts/build_team_model.py                       # ~10 min
    python scripts/process_player_features.py                # ~10 min
    python scripts/build_player_model.py                     # ~10 min

---

## Key File Paths

    data/raw/game_logs/player_game_logs.csv          339,853 rows, 11 seasons (20152016-20252026)
    data/raw/player_pbp_stats/player_pbp_stats.csv   268,051 rows, 6 seasons (20202021-20252026)
    data/raw/team_game_logs/team_game_logs.csv        team game logs, 6 seasons
    data/raw/lineups/lineups_YYYYMMDD.json            daily lineup files from Daily Faceoff
    data/raw/player_pp_stats.csv                      10,442 rows, PP shots/goals from NHL API
    data/raw/player_corsi_stats.csv                   individual shot attempts from NHL realtime API
    data/raw/player_pp_shares.csv                     5,158 rows, per-player-team-season PP TOI share
    data/processed/player_features.csv                339,853 rows, ~300 features
    data/processed/shot_data_with_xg.csv              432,713 rows, calibrated xG
    data/processed/team_features.csv                  7,252 rows, 1,511 features
    models/team/                                       23 XGBoost models
    models/player/                                     7 XGBoost models + calibrators

---

## Scripts Reference

    Script                              Purpose                              Time
    fetch_team_game_logs.py             Team PBP + TOI + penalties           ~30-45 min
    fetch_player_game_logs.py           Skater game logs (11 seasons)        ~45-60 min
    fetch_goalie_game_logs.py           Goalie game logs                     ~5-10 min
    fetch_shot_data.py                  Shot events with coordinates         ~30-45 min
    fetch_player_pbp_stats.py           Player PBP stats (6 seasons)         ~3 hrs full, ~5 min --days 3
    fetch_daily_lineups.py              Scrape Daily Faceoff lineups         ~1 min
    update_data.py                      Incremental update all data          ~3-5 min
    build_xg_model.py                   Train xG model                       ~5 min
    process_team_features.py            Build team feature dataset           ~25-30 min
    build_team_model.py                 Train team prediction models         ~10 min
    process_player_features.py          Build player feature dataset         ~10 min
    build_player_model.py               Train player prediction models       ~10 min
    predict_game.py                     Generate team game predictions       <1 min
    predict_player_props.py             Generate player prop predictions     ~1 min

---

## NHL API Reference
- Base: https://api-web.nhle.com/v1
- No auth required
- Situation codes: 1551=EV, 1451=away PP (home penalized), 1541=home PP (away penalized), 0651/1560=EN
- Ice coords: x+-100, y+-42.5, nets at x=+-89
- Realtime stats: https://api.nhle.com/stats/rest/en/skater/realtime
- PP stats: https://api.nhle.com/stats/rest/en/skater/powerplay
- Shift charts: https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}

---

## Phase 2 — Team Model (COMPLETE)

### Performance (test set = 20252026, 1,136 games)

    Win AUC:              0.7544    (MoneyPuck benchmark: ~0.68-0.70)
    Win LogLoss:          0.5890    (MoneyPuck benchmark: 0.658-0.661)
    Goals MAE:            1.319 / 1.277 (home/away)
    xG model AUC:         0.7635
    Total Shots MAE:      5.002
    EV Shots MAE:         4.512
    EV Shots Bias:        +0.055
    PP TOI MAE:           0.776 / 0.755
    Game Total Shots MAE: 6.671 (sigma=8.34)

### Shot Prediction Architecture

    Total shots = own_last30 x 0.55 + opp_against_last30 x 0.45
    PP shots    = pp_shots_per60_cumulative_season x pred_pp_toi / 60
    SH shots    = weighted_sh_shots_per60 x pred_sh_toi / 60
    EV shots    = total - PP - SH
    Game total  = home_total + away_total (naive sum, sigma=8.34)

    PP TOI = 0.87 + own_drawn x 0.219 + opp_taken x 0.606 + own_pp_toi_season x 0.251
    SH TOI = opponent PP TOI (enforced equal)
    EV TOI = AVG_GAME_TOI (59.73) - PP TOI - SH TOI

### predict_game.py Output Format
- Win probability (home/away)
- Goals per team: Poisson lambda + distribution + over thresholds
- Individual shots: EV/PP/SH breakdown, book lines (24.5/29.5/34.5), dynamic +-5 table in 0.5 increments
- Game total shots: naive sum, book lines (49.5/54.5/57.5/59.5/64.5)
- xG per team, strength TOI breakdown

### Bug Fixes Applied (Mar 2026)
- Same-game penalty leakage: 28 raw penalty count columns added to EXCLUDE_COLS. PP TOI MAE 2.04→0.776, win AUC 0.743→0.754.
- PP shots back-calculation: switched from flat season_avg to cumul_per60 x pred_pp_toi. EV bias -0.307→+0.055.
- xG/SOG feature inflation: rolling averages were averaging per-game ratios. Switched to cumulative sum(xG)/sum(shots). Goals MAE 1.328→1.319.

---

## Phase 3 — Player Props Model (MOSTLY COMPLETE)

### What Works End-to-End
predict_player_props.py generates per-player predictions for any game using today's Daily Faceoff lineup. Output includes EV TOI, PP TOI, shots, goals, assists, points per player plus game-level totals and top prop probabilities.

### Data Sources

    player_game_logs.csv        339,853 rows   11 seasons   basic per-game stats, total TOI
    player_pbp_stats.csv        268,051 rows   6 seasons    EV/PP/SH TOI, Corsi, on-ice shots/goals by strength
    shot_data_with_xg.csv       432,713 rows   4 seasons    individual xG per shot
    player_pp_stats.csv         10,442 rows    11 seasons   PP shots/goals per season from NHL API
    player_corsi_stats.csv      ~10,000 rows   11 seasons   individual shot attempts from NHL realtime API
    player_pp_shares.csv        5,158 rows     6 seasons    per-player-team-season PP TOI share
    lineups_YYYYMMDD.json       32 teams       daily        forward lines, defense pairs, PP units, injuries

### player_pbp_stats.csv Columns (268,051 rows, 20202021-20252026)
    TOI: toi_total, toi_ev, toi_pp, toi_sh, toi_en
    Individual shots by strength: ev/pp/sh_shots
    Corsi by strength: ev/pp/sh_missed_shots, ev/pp/sh_shots_blocked_by_opp, ev/pp/sh_shot_attempts
    Goals by strength: ev/pp/sh_goals
    Assists by strength: ev/pp/sh_assists
    On-ice shots: ev/pp/sh_onice_sf/sa (shots for/against)
    On-ice goals: ev/pp/sh_onice_gf/ga (goals for/against)
    Misc: hits, blocks, faceoffs_won, faceoffs_taken, giveaways, takeaways

    Note: Corsi (missed/blocked/attempts) only available 20202021+ via PBP.
    Pre-20202021 Corsi set to NaN deliberately — season API constants pollute rolling windows.

### Situation Code Format (verified)
    away_goalie | away_skaters | home_skaters | home_goalie
    1551 = 5v5 EV
    1451 = home penalized (away on PP)
    1541 = away penalized (home on PP)
    1560 = away EN
    0651 = home EN
    TOI variance vs NatStatTrick: ~16s per player per game (API vs official shift report — irreducible)

---

## xG Model (COMPLETE)

### What It Does
Estimates the probability that a given shot results in a goal based on observable shot characteristics. Used to compute individual xG (ixg) and on-ice xG for player features.

### Architecture
- XGBoost classifier + Platt scaling calibration
- Trained on 20232024-20242025 (2-season window best)
- AUC: 0.7635, calibration: league xG/goal ratio 1.027

### Features
    distance          0.213   distance from net in feet
    is_forward        0.211   forward vs defenseman shot
    period            0.089   game period (proxy for game state)
    angle             0.086   shot angle from center
    shot_type         0.072   wrist/snap/slap/tip/backhand etc
    is_rebound        0.065   shot within 3 seconds of previous shot
    rebound_angle     0.043   angle change on rebound
    speed_from_prev   0.038   puck speed proxy (distance/time from last event)
    is_rush           0.031   speed>20 ft/s AND prev_dist>75 ft

- Strength excluded: shot location implicitly captures this
- Score state excluded: affects shot selection, not shot quality
- Empty net: separate logistic regression (AUC 0.7223)

### Calibration (test season, non-EN SOG)
    xG bucket   Actual   Predicted
    0.00-0.05   2.8%     2.0%
    0.05-0.10   7.8%     7.3%
    0.10-0.15   12.2%    12.4%
    0.15-0.20   17.1%    17.4%
    0.20-0.30   21.5%    24.3%
    0.30-0.40   30.2%    33.9%
    0.40-0.60   41.4%    46.2%
    0.60-1.00   66.0%    73.4%

High danger slightly over-predicted — acceptable without tracking data (traffic, screens, pass quality).

---

## Player Feature Engineering (process_player_features.py)

### Pipeline
    1. player_game_logs (11 seasons) — base stats
    2. LEFT JOIN player_pbp_stats (6 seasons) — strength splits + Corsi
       Pre-20202021 rows: approximate fills (toi_ev=toi*0.85, ev_shots=shots*0.80 etc)
    3. JOIN shot_data_with_xg — individual xG by strength per game
    4. JOIN team_game_logs — team EV/PP/SH TOI context per game
    5. JOIN player_pp_stats — accurate career PP shots/goals from NHL API
    6. Compute game rates (per-60, shares, binary indicators)
    7. Rolling features (within-season windows)
    8. Trend/EWM features
    9. Career features (cross-season)
    10. Cumulative per-60 rates (sum/sum method)
    11. Weighted PP TOI share (EWM across past seasons)

### Key Derived Features

    ev_toi_share              player EV TOI / team EV TOI
    pp_toi_share              player PP TOI / team PP TOI
    ev_cf_pct                 ev_onice_sf / (ev_onice_sf + ev_onice_sa)
    ipp                       (ev_goals + ev_assists) / ev_onice_gf (clipped 0-1)
    scored_ev_goal            binary: did player score an EV goal this game
    is_pp_player              binary: did player get >0.5 min PP TOI this game
    ev_shot_attempts_per60    ev_shot_attempts / (toi_ev/60)
    indiv_shot_attempts       total Corsi per game (NaN pre-20202021)
    weighted_pp_share         EWM of past seasons' PP TOI share (span=2, shift=1)

### Rolling Windows
    Within-season only (reset at season boundary): last5, last10, last20, last30, season_avg
    EWM (within-season): span=10, shift=1 for toi_ev, toi_pp, ev_shots, pp_shots, ixg, shot_attempts
    Trend ratio: last5 / last20 (clipped 0-5) — role change detector
    Corsi rolling features: only populated for 20202021+ (real per-game data, not constants)

### Career Features (cross-season, all seasons)

    career_games                      total NHL games played (prior to current game)
    regressed_ev_shooting_pct         Bayesian regression: career ev_goals/ev_shots toward league mean 0.098
                                      Prior strength: 100 shots. At 100 career EV shots → 50% weight on career rate.
    regressed_pp_shooting_pct         career pp_goals/pp_shots (using accurate NHL API data) toward 0.144
                                      Prior strength: 30 PP shots.
    regressed_sh_shots_per60          career sh_shots per 60 SH TOI toward league mean 5.44
                                      Prior strength: 20 SH TOI minutes.
    regressed_finishing_talent        career ev_goals/ev_ixg (xG-era only) toward league mean 1.097
                                      Prior strength: 20 career xG. Draisaitl ~1.65x, Kucherov ~1.45x.

    Why 1.097 finishing talent baseline (not 1.0):
    xG model under-predicts aggregate goals at player level by 9.7%.
    Using 1.097 as the regression anchor correctly calibrates goal predictions.

### PP Shooting% Uses NHL API Data (not PBP)
The player_pp_stats.csv from NHL API has accurate career PP shot/goal totals
going back to 20142015. This is used instead of PBP because:
- PBP only covers 20202021+ (6 seasons)
- API covers full career (more data = better regression)
- Validated: Draisaitl 22.0%, Kucherov 13.1%, McDavid 14.4%, Matthews 17.0%

---

## Player Models (build_player_model.py)

### Architecture
    Train: 20152016-20242025 (10 seasons)
    Test:  20252026
    Algorithm: XGBoost (1000 estimators, lr=0.03, max_depth=4, early stopping=30)
    Feature selection: XGBoost importance, top 60 features per model
    Validation: last 20% of training data (time-ordered)

### Models Trained

    toi_ev (regression)
    Predicts: minutes of even-strength ice time
    MAE: 1.855 min (baseline: 3.066, 40% improvement)
    Corr: 0.783
    Key features: toi_ev_ewm10 (0.38), toi_ev_last30 (0.23), toi_ev_season_avg (0.11)
    Why EWM dominates: TOI is highly persistent — a player's role changes slowly.
    The exponentially weighted average captures recent deployment better than raw last-N windows.

    toi_pp (regression)
    NOTE: This model is trained but NOT used in predictions.
    PP TOI is predicted via formula instead (see PP TOI Formula below).
    MAE: 0.590 min (baseline: 0.951) — kept for reference/validation
    Key features: is_pp_player (0.49), toi_pp_last30 (0.08)
    Why formula wins: PP TOI is almost entirely determined by lineup PP unit assignment.
    XGBoost can't see today's lineup, but the formula can.

    ev_shots (regression)
    Predicts: even-strength shots on goal
    MAE: 0.876 shots (baseline: 0.960, 9% improvement)
    Corr: 0.390
    Key features: shots_season_avg (0.27), shots_last30 (0.16), ev_shots_season_avg (0.12)
    Why shot volume is hard to predict: high game-to-game variance (CV ~1.0).
    Season averages dominate because per-game shot counts are noisy.
    Corsi features (indiv_shot_attempts) add modest signal for recent seasons.

    pp_shots (regression)
    Predicts: power play shots on goal
    MAE: 0.091 shots (baseline: 0.298, 69% improvement) — best model by far
    Corr: 0.955
    Key features: pp_shots_api_filled (0.32), indiv_shot_attempts_last5 (0.14)
    Why so accurate: PP shots are driven by PP deployment (predictable) and
    player shot rate on PP (stable). The accurate NHL API PP shot totals
    (pp_shots_api_filled) provide a strong season-level anchor.

    ev_assists (regression)
    Predicts: even-strength assists
    MAE: 0.334 assists (baseline: 0.342, marginal improvement)
    Corr: 0.172
    Key features: points_season_avg (0.16), assists_season_avg (0.09), points_last30 (0.07)
    Why assists are hard: assists are extremely noisy per game (most games = 0).
    The model barely beats baseline. Points/assists season averages capture
    player quality but can't predict when assists cluster.

    pp_assists (regression)
    Predicts: power play assists
    MAE: 0.128 assists (baseline: 0.124, slightly worse than baseline)
    Corr: 0.264
    Key features: pp_assists_season_avg (0.15), pp_shots_api_filled (0.14)
    Note: Slightly worse than baseline — PP assists are too rare and random
    per game to model reliably. Consider switching to formula:
    pp_assists = pp_points_expected - pp_goals_expected

    scored_ev_goal_f (classifier, forwards only)
    Predicts: probability that a forward scored an EV goal this game
    AUC: 0.6257, LogLoss: 0.4010 (baseline: 0.4126)
    Calibration: isotonic regression on validation set
    Key features: shots_season_avg (0.15), shots_last30 (0.06), toi_ev_ewm10 (0.03)
    Why classifier not regression: EV goals are binary (scored or not) with
    Poisson-like distribution. Classifier + Poisson CDF for probability thresholds
    is more principled than regression for rare binary events.
    Why forward-only: defense score so rarely (6% positive rate vs 14% for forwards)
    that a combined model is dominated by the is_defense flag.
    Defense EV goals use formula only: ev_shots x regressed_ev_shooting_pct x regressed_finishing_talent.
    Calibration check (test season):
      0-5% bucket:   actual 4.0% vs pred 4.0%  (perfect)
      5-10% bucket:  actual 7.7% vs pred 7.3%  (excellent)
      10-15% bucket: actual 12.1% vs pred 12.2% (excellent)
      15-20% bucket: actual 18.2% vs pred 17.2% (excellent)

### What Is Excluded (EXCLUDE set)
All same-game raw stats are excluded to prevent leakage:
- Same-game TOI, goals, assists, shots, points
- Same-game Corsi components (ev/pp/sh_missed_shots, shot_attempts, shots_blocked)
- Same-game per-60 rates derived from the game itself
- Same-game on-ice shot/goal counts
- Raw NHL API season totals (pp_shots_api, pp_goals_api, pp_toi_sec_api) — same-season leakage
- Columns ending in _x or _y (merge artifacts)

### Saved Model Files (models/player/)
    toi_ev.json, toi_pp.json
    ev_shots.json, pp_shots.json
    ev_assists.json, pp_assists.json
    scored_ev_goal_f.json
    feature_lists.pkl      {model_name: [feature_col_names]}
    residual_stds.pkl      {model_name: float} for confidence intervals
    model_types.pkl        {model_name: "regression"|"classifier"}
    calibrators.pkl        {model_name: IsotonicRegression} for scored_ev_goal_f

---

## predict_player_props.py — Prediction Pipeline

### Usage
    python scripts/predict_player_props.py EDM VAN
    python scripts/predict_player_props.py EDM VAN --date 2026-03-28

### Pipeline (per player)
    1. Load lineup from Daily Faceoff JSON (forward lines, defense pairs, PP units)
    2. Match player names to player_ids (nickname expansion, unicode normalization,
       middle name stripping, position disambiguation)
    3. Pull latest feature row from player_features.csv
    4. Run XGBoost models → toi_ev, ev_shots, pp_shots, ev_assists, pp_assists, p_ev_goal
    5. Compute PP TOI via formula (not model)
    6. Compute goals via formula
    7. Sum to team totals

### PP TOI Formula (replaces XGBoost toi_pp model)
PP TOI is predicted as:

    predicted_pp_toi = team_pp_toi_last20 x player_pp_share

Where player_pp_share is computed via get_pp_share() with this priority:

    Step 1 (PRIMARY): Team baseline for today's PP unit assignment
      - compute_team_pp_baselines(): reads last 20 games of PBP
      - PP1 usage = avg share of ranks 2-4 players per game (excludes QB skew)
      - PP2 usage = 1 - PP1 usage
      - Example: EDM PP1=0.770, PP2=0.230 | TBL PP1=0.796, PP2=0.204 | STL PP1=0.575, PP2=0.425
      - This is the DOMINANT signal — lineup assignment drives ~50-70% of the prediction

    Step 2 (NUDGE, up to 30%): Recent games adjustment
      - compute_recent_player_pp_shares(): last 10 games where player had PP time
      - Weight scales with sample: min(games/10, 1.0) x 0.30
      - Captures role changes not yet reflected in season history

    Step 3 (NUDGE, up to 20%): Season history adjustment
      - player_pp_shares.csv: per-player-team-season average share
      - Only applies if historical role matches today's role (PP1 vs PP2)
      - Weight: min(hist_games/30, 1.0) x 0.20
      - Ignored for role changes (e.g. Savoie promoted from PP2 to PP1)

    Example (EDM, 4.81 min team PP last20):
      McDavid (PP1, 38 hist games at 0.805 share): 0.770 + hist nudge → 0.77 x 4.81 = 3.7 min
      Savoie (PP2 hist, listed as PP2 today): team PP2 baseline → 0.23 x 4.81 = 1.1 min
      Podkolzin (PP2 hist, listed as PP1 today): role change → PP1 baseline → 2.9 min

### Goals Formula

    ev_goals_lambda = ev_shots x regressed_ev_shooting_pct x regressed_finishing_talent
    pp_goals_lambda = pp_shots x regressed_pp_shooting_pct
    total_goals     = ev_goals_lambda + pp_goals_lambda

    Poisson CDF used for threshold probabilities (o0.5, o1.5 etc)

### Assists Formula

    total_assists = ev_assists (from model) + pp_assists (from model)

### Name Matching (97%+ success rate)
    Priority: exact → unicode normalized → middle name stripped → nickname expansion
    Nicknames: Matt↔Matthew, Jake↔Jacob, Alex↔Alexander, Mitch↔Mitchell, etc.
    Disambiguation: Elias Pettersson F vs D resolved by position_group from lineup
    Fallback: last name + team if all else fails

### Output Format
    Player table: name, role (F/D L1-L4 PP1/PP2), TOI_EV, TOI_PP, Shots, Goals, Assists, Points
    Game summary: team shot/goal totals
    Top props: shot props (o2.5), goal props (anytime scorer), point props (o0.5)
    Probabilities shown as: XX.X% (±odds)

### Known Issues / Next Steps
- Shot props need calibration: model predicts shots well at aggregate level
  but individual o2.5 probabilities need validation against historical lines
- Goal props may be under-predicted: regressed_finishing_talent pulling toward mean
  for hot streaks — consider adding recent shooting% feature
- pp_assists slightly worse than baseline — consider formula approach
- No opponent defensive context yet (goalie quality, team shot suppression)
- No home/away splits in player features yet

---

## PP TOI Share Data (player_pp_shares.csv)

Generated at runtime by compute_team_pp_baselines() and compute_recent_player_pp_shares()
from player_pbp_stats.csv — no static CSV needed for team baselines.

player_pp_shares.csv (static, regenerate with process_player_features.py):
- Per player per team per season: avg_share, games_with_pp, avg_rank, pp_unit_est
- pp_unit_est: 1 = PP1 (avg_rank < 5.5), 2 = PP2 (avg_rank >= 5.5)
- Only players with 5+ PP games included

Team PP usage rates (20252026, last 20 games):
    EDM: PP1=0.770  PP2=0.230   (dominant PP1 unit)
    TBL: PP1=0.796  PP2=0.204
    VGK: PP1=0.771  PP2=0.229
    STL: PP1=0.575  PP2=0.425   (most balanced — uses PP2 heavily)
    VAN: PP1=0.659  PP2=0.341

---

## Corsi Data Coverage

    Season       total_shot_attempts   missed_shots   shots_blocked   Source
    20152016      935 players           available      935 players     NHL realtime API
    20162017      919 players           available      919 players     NHL realtime API
    20172018-     NOT available         available      NOT available   API gap (NaN in features)
    20212022
    20222023+     997+ players          available      997+ players    NHL realtime API
    20202021+     per-game PBP          per-game PBP   per-game PBP    PBP fetch (used for rolling features)

    Decision: Pre-20202021 Corsi set to NaN in features. Season API totals are constants
    per player per season — distributing them per game creates leakage (same value every row).
    Rolling windows of a constant add zero signal. NaN is the correct choice.

---

## Key Modeling Decisions & Rationale

    PP TOI via formula not model
    XGBoost can't see today's lineup. PP TOI is almost entirely determined by
    PP unit assignment (PP1 vs PP2). Formula using team baseline + player history
    is more accurate and more interpretable than a rolling-average-based model.

    Separate forward/defense goal models
    is_defense had 0.44 importance in combined model — model was mostly learning
    "D scores less than F" rather than who among forwards will score.
    Forward model (AUC 0.625) is better calibrated. Defense uses formula only.

    pp_shots_api_filled feature
    For pre-20202021: NHL API season total / games_in_season (distributed per game)
    For 20202021+: actual per-game PBP pp_shots
    Used as a feature (not target) — represents player's season-level PP deployment.
    Highest importance feature in pp_shots model (0.32).

    Regressed shooting% toward league mean not raw career
    Raw career shooting% is noisy for players with small samples (rookies, part-time players).
    Bayesian regression toward league mean (100-shot prior for EV, 30-shot for PP)
    prevents extreme predictions for low-sample players while preserving elite shooter signal.

    Rolling windows reset at season boundary
    A player's stats from 3 seasons ago are not informative about tonight's game.
    Within-season rolling windows capture current-season form.
    Career features (cross-season) capture stable player quality separately.

    Corsi only from 20202021+
    Pre-20202021 Corsi from NHL API is a season total distributed evenly across games.
    This creates a constant feature per player per season — rolling windows of a constant
    are identical at every window length (last5 = last30 = season_avg).
    Setting to NaN forces the model to use real per-game Corsi only where available.

---

## Phase 4 — Goalie Model (NEXT)

### Plan
- Input: goalie game logs + save% rolling averages + opponent shot quality (xG)
- Predict: goals allowed, save%, GSAx (goals saved above expected)
- Integration: feed into team win probability and team goals model
- Key feature: goalie quality affects shot volume (teams shoot more vs bad goalies)

### Data Available
- goalie_game_logs.csv: basic stats per game
- shot_data_with_xg.csv: per-shot xG, can compute xG faced per goalie per game
- Need: goalie-level rolling save% by shot quality tier

---

## Empirical Analysis Results

### Rolling Window Predictability (team level)
    Stat                L10     L20     L30     Season  Best
    EV Shot Attempts    0.185   0.202   0.202   0.195   L20/L30
    EV Shots on Goal    0.214   0.243   0.247   0.229   L30
    EV Goals            0.038   0.066   0.071   0.054   L30
    EV Shooting%        0.027   0.048   0.055   0.029   L30
    Block Rate          0.089   0.113   0.126   0.111   3yr avg
    SOG/Fenwick         0.117   0.146   0.166   0.134   3yr avg

### Player Model Performance Summary

    Model              MAE/AUC    Baseline   Improvement
    toi_ev             1.855 min  3.066      40%
    toi_pp             0.590 min  0.951      38% (trained, not used — formula used instead)
    ev_shots           0.876      0.960      9%
    pp_shots           0.091      0.298      69%
    ev_assists         0.334      0.342      2%
    pp_assists         0.128      0.124      -3% (slightly worse than baseline)
    scored_ev_goal_f   AUC 0.626  —          well-calibrated

### PP Shooting% Validation (from NHL API career data)
    Player          Raw career%   Regressed%
    Draisaitl       23.5%         22.0%
    Kucherov        14.6%         13.1%
    McDavid         15.7%         14.4%
    Matthews        17.5%         17.0%
    Hyman           20.7%         17.7%
    Tkachuk         11.9%         12.1%
    League mean: 14.4% (regression anchor)
## Pipeline Order (Phase 3)
1. fetch_shot_data.py         → data/raw/shot_data/shot_data.csv
2. build_xg_model.py          → data/processed/shot_data_with_xg.csv
3. build_zone_features.py     → data/processed/player_zone_shots.csv, team_zone_*.csv, zone_averages.json
4. fetch_player_pbp_stats.py  → data/raw/player_pbp_stats/player_pbp_stats.csv
5. fetch_goalie_game_logs.py  → data/raw/goalie_game_logs/goalie_game_logs.csv
6. process_player_features.py → data/processed/player_features.csv
7. build_player_model.py      → models/player/
8. fetch_daily_lineups.py     → data/raw/lineups/lineups_YYYYMMDD.json
9. predict_player_props.py    → predictions output
