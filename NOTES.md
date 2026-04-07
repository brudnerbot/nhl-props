# NHL Props Model — Master Reference Document
# Last updated: April 2026
# GitHub: github.com/brudnerbot/nhl-props
# Stack: Python 3.14.3, venv, XGBoost, pandas, numpy, scipy
# Workflow: chat -> code in Cursor -> run in terminal

==============================================================
PROJECT OVERVIEW
==============================================================

Goal: Build NHL player prop prediction models that predict true
probabilities better than sportsbooks, exploiting their recency
bias and public over-betting tendencies.

Key insight: Books use last 5-10 game rolling windows and exploit
recency bias. We use season averages + recent blend + zone context
+ goalie quality + B2B effects to predict true mean outcomes.
Books maximize profit via psychology; we maximize accuracy.

==============================================================
ARCHITECTURE — NO CIRCULAR REFERENCES
==============================================================

Prediction flows in this order:

1. GOALIE MODEL (fully independent)
   Inputs:  goalie GSAx history, team defense zone profile,
            opponent offense zone profile, B2B, home/away
   Outputs: predicted saves, save%, goals allowed, GSAx adjustment
   Note:    Does NOT use player predictions

2. PLAYER MODEL (uses goalie output only)
   Inputs:  player features, zone profiles, B2B ratios,
            goalie adj from step 1
   Outputs: shots, goals, assists, points per player

3. TEAM MODEL (uses player + goalie outputs)
   Inputs:  aggregated player predictions (sum of shots/xG),
            team features, goalie adj from step 1
   Outputs: team shots, goals, win probability, game total
   Note:    Uses AGGREGATED player output, not individual preds
            so no circular reference

4. GAME SUMMARY
   Inputs:  all of the above
   Outputs: prop probabilities, zone matchup context, B2B flags

==============================================================
PIPELINE SCRIPTS (run in this order)
==============================================================

--- DATA FETCHING ---
fetch_shot_data.py
  -> data/raw/shot_data/shot_data.csv
  -> 1,548,612 shots, 11 seasons (20152016-20252026)
  -> includes x_coord_norm, y_coord_norm, is_on_goal, is_goal,
     shooter_id, defending_team, shooting_team, strength, etc.

fetch_player_pbp_stats.py
  -> data/raw/player_pbp_stats/player_pbp_stats.csv
  -> 267,811 rows, 6 seasons (20202021-20252026)
  -> strength-split TOI, shots, goals, assists, Corsi, on-ice
  -> IMPORTANT: EN situation fix applied
     6v5 EN = EV, 6v4 EN = PP for attacking team, 5v6 EN = SH
     Fix: adj_hs = hs - (1 if hg==0 else 0)
     Verified: Gauthier March 8 correct EV=10.03, PP=8.03, SH=0.87

fetch_goalie_game_logs.py
  -> data/raw/goalie_game_logs/goalie_game_logs.csv
  -> per-game goalie stats, all seasons

fetch_daily_lineups.py
  -> data/raw/lineups/lineups_YYYYMMDD.json
  -> DailyFaceoff scraper, forward lines + D pairs + PP/PK units

--- DATA PROCESSING ---
build_xg_model.py
  -> data/processed/shot_data_with_xg.csv
  -> XGBoost xG model trained on shot location + type + strength
  -> Features: x_coord_norm, y_coord_norm, shot_type, strength,
     angle, distance, is_rush, is_rebound
  -> Adds xg column and zone column to shot data

build_zone_features.py
  -> data/processed/player_zone_shots.csv  (315,654 rows)
  -> data/processed/team_zone_defense.csv  (26,734 rows)
  -> data/processed/team_zone_offense.csv  (26,734 rows)
  -> data/processed/zone_averages.json
  -> Must run after build_xg_model.py

process_player_features.py
  -> data/processed/player_features.csv
  -> 339,853 rows, 493 features, 11 seasons
  -> Merges: game_logs + pbp_stats + xG + team context +
     goalie context + zone features + PP stats + Corsi
  -> Rolling windows: last5/10/15/20/30 within season
  -> Season avg: expanding mean within season
  -> Career features: cross-season
  -> Takes ~10-15 min to run

build_player_model.py
  -> models/player/*.json  (7 XGBoost models)
  -> models/player/feature_lists.pkl
  -> models/player/model_types.pkl
  -> models/player/calibrators.pkl
  -> models/player/residual_stds.pkl

--- PREDICTION ---
predict_player_props.py
  -> Usage: python scripts/predict_player_props.py HOME AWAY
  -> Optional: --date 2026-03-30  --min-shots 1.0

==============================================================
KEY DATA FILES
==============================================================

data/raw/shot_data/shot_data.csv
  1,548,612 shots, 11 seasons

data/processed/shot_data_with_xg.csv
  631,950 SOG rows with xG + zone classification

data/processed/player_zone_shots.csv
  315,654 rows. Per-player per-game zone shots + rolling avgs.
  Columns: game_id, player_id, net_front, slot, left_flank,
  right_flank, left_point, mid_point, right_point,
  {zone}_season_avg, {zone}_last10

data/processed/team_zone_defense.csv
  26,734 rows. Per-team shots allowed by zone per game.
  Columns: game_id, defending_team, {zone}_allowed_last30

data/processed/team_zone_offense.csv
  26,734 rows. Per-team shots for by zone per game.
  Columns: game_id, shooting_team, {zone}_for_last30

data/processed/zone_averages.json
  League avg shots/game per zone (20222023+)
  Position zone proportions (forward vs defense)
  NOT hardcoded — always loaded from file

data/processed/player_features.csv
  339,853 rows, 493 features. Main training dataset.

data/processed/goalie_features.csv
  Rolling GSAx, save%, shots_against per goalie per game.
  gsax_last20, gsax_last30, gsax_season_avg,
  regressed_gsax_per_game

data/processed/prop_calibration.json
  Empirical P(stat > line | season_avg) curves.
  Built from 20222023+ data. Buckets of 0.10 width.
  Stats: shots, goals, ev_assists, points

data/processed/b2b_effects.json
  Team and player B2B shot/goal adjustment ratios.
  Weighted by season (see B2B section below).
  Keys: league_team_b2b, team_b2b, league_player_b2b, player_b2b

data/raw/player_pbp_stats/player_pbp_stats.csv
  267,811 rows, 6 seasons strength-split stats

data/raw/goalie_game_logs/goalie_game_logs.csv
  Per-game goalie stats all seasons

data/raw/team_game_logs/team_game_logs.csv
  Per-team per-game stats all seasons

data/raw/game_logs/player_game_logs.csv
  Per-player per-game basic stats all seasons

data/raw/lineups/lineups_YYYYMMDD.json
  Daily lineup files from DailyFaceoff

data/raw/player_pp_shares.csv
  PP TOI share per player per team per season

==============================================================
SHOT ZONE CLASSIFICATION
==============================================================

NHL rink: 200ft long, 85ft wide
Net at x=89 (normalized coords), blue line at x=25 (64ft from goal)
Goal line 11ft from end boards
Faceoff dots: x=69 (20ft from goal), y=+/-22
Faceoff circle radius: 15ft
Blue line: 64ft from goal line (NOT 25ft — common mistake)

def classify_zone(x, y):
    # x=89 goal line, x=25 blue line, y=0 center, y=+/-42.5 boards
    if x < 25:          return "out_of_ozone"
    if x <= 46.5:       # point zone: 42.5ft from goal
        if y < -8.5:    return "left_point"
        elif y > 8.5:   return "right_point"
        else:           return "mid_point"
    if x > 74:          return "net_front"  # within 15ft of goal
    if abs(y) < 22:     return "slot"       # between faceoff dots
    if y < -22:         return "left_flank"
    return "right_flank"

Point zone width split: total 85ft, mid = half width of each side
  side width = 34ft (y=+/-8.5 to +/-42.5)
  mid width  = 17ft (y=-8.5 to +8.5)
Net front includes behind-the-net (x > 74, all y)

Zone Stats (SOG only, all seasons):
  net_front:   32.9%  goal_rate=15.7%  avg_xg=0.164
  slot:        29.8%  goal_rate=12.8%  avg_xg=0.133
  left_flank:   7.3%  goal_rate=4.0%   avg_xg=0.039
  right_flank:  9.0%  goal_rate=4.3%   avg_xg=0.044
  left_point:   8.2%  goal_rate=3.0%   avg_xg=0.028
  mid_point:    4.5%  goal_rate=5.0%   avg_xg=0.052
  right_point:  8.3%  goal_rate=2.9%   avg_xg=0.028

Position Zone Proportions (20222023+):
  Zone          Forward   Defense
  net_front      43.0%    10.8%
  slot           32.9%    18.1%
  left_flank      7.3%     8.7%
  right_flank     9.2%    11.0%
  left_point      2.7%    20.9%
  mid_point       2.1%    10.4%
  right_point     2.9%    20.1%

League avg shots/game per zone (20222023+):
  net_front:    9.649
  slot:         8.125
  left_flank:   2.148
  right_flank:  2.714
  left_point:   2.131
  mid_point:    1.216
  right_point:  2.116

Notable player zone profiles (20242025 + 20252026):
  McDavid:    59% net_front (+16% vs F avg)
  Hyman:      78% net_front (+35%) -- extreme net-front shooter
  Bouchard:   35% slot (+17%), 59% total point shots
  Eichel:     30% net_front (-13%), heavy right_flank (+10%)
  Gauthier:   slot-heavy (+4%), right_flank (+10%)
  Ovechkin:   B2B shot ratio 1.161 (unique -- improves on B2B)

==============================================================
CURRENT PLAYER MODEL PERFORMANCE
==============================================================

Model               MAE/AUC   Notes
toi_ev              1.809     XGBoost
toi_pp              0.562     Formula (not model)
ev_shots            0.625     XGBoost + zone features (-28% vs pre-zone)
pp_shots            0.047     XGBoost + zone features (-45% vs pre-zone)
ev_assists          0.349     XGBoost + recent rate blend
pp_assists          0.119     XGBoost + recent rate blend
scored_ev_goal_f    AUC 0.6950  Classifier, zone features +7pts AUC

History of ev_shots MAE:
  Before zone features:  0.865
  After zone features:   0.625  (-28%)

==============================================================
PREDICTION FORMULAS (predict_player_props.py)
==============================================================

--- EV SHOTS (formula, not XGBoost) ---
Reason: XGBoost regresses to mean, underestimates tails.
Formula preserves true player rate and game-to-game variance.

ev_rate_last10 = ev_shots_last10 / (toi_ev_last10 / 60)
ev_rate_season = ev_shots_season_avg / (toi_ev_season_avg / 60)
ev_rate = 0.60 * ev_rate_last10 + 0.40 * ev_rate_season
ev_shots = ev_rate * pred_toi_ev / 60

pred_toi_ev comes from XGBoost toi_ev model

--- PP SHOTS (formula) ---
pp_rate_last10 = pp_shots_last10 / (toi_pp_last10 / 60)
pp_rate_season = pp_shots_season_avg / (toi_pp_season_avg / 60)
pp_rate = 0.50 * pp_rate_last10 + 0.50 * pp_rate_season
pp_shots = pp_rate * pred_toi_pp / 60

pred_toi_pp = team_pp_toi * player_pp_share
player_pp_share: team baseline (60%) + recent share (30%) + hist (20%)

--- PP TOI ---
team_pp_toi: from team game logs rolling avg
player_pp_share: blended from team PP baselines + recent PBP shares
  PP1 fallback: 0.670, PP2 fallback: 0.330, non-PP: 0.030

--- EV GOALS ---
ev_sh_pct = regressed_ev_shooting_pct  (Bayes regressed to 9.8%)
finishing = regressed_finishing_talent  (Bayes regressed to 1.097)
goalie_adj = 1 - (opp_gsax_last20 / avg_shots_faced)  # capped +/-30%
avg_shots_faced = 28.8 (computed from 20222023+ data)

ev_goals_formula = ev_shots * ev_sh_pct * finishing * goalie_adj
ev_goals = 0.60 * ev_goals_formula + 0.40 * goals_last10
  (40% recent rate captures hot/cold streaks)

--- PP GOALS ---
pp_sh_pct = regressed_pp_shooting_pct  (Bayes regressed to 14.4%)
pp_goals = pp_shots * pp_sh_pct * goalie_adj

--- TOTAL GOALS ---
total_goals = ev_goals + pp_goals

--- ASSISTS (XGBoost blend) ---
ev_assists = 0.50 * XGBoost_pred + 0.50 * (assists_last10_rate * pred_toi_ev)
pp_assists = 0.50 * XGBoost_pred + 0.50 * (pp_assists_last10_rate * pred_toi_pp)

--- PROBABILITY CALCULATION ---
Method: empirical calibration table with NegBin fallback
NegBin r values (fitted from data):
  shots=3.87, goals=2.32, ev_assists=6.46, points=3.18
Empirical table: prop_calibration.json buckets of 0.10 width
Interpolation between nearest buckets when sample >= 50

==============================================================
GOALIE GSAx
==============================================================

GSAx = Goals Saved Above Expected
Computed per-game from shot xG vs actual goals allowed.

Rolling windows: gsax_last20, gsax_last30, gsax_season_avg
regressed_gsax_per_game: Bayes regression toward 0

20252026 top goalies (GSAx season):
  Thompson (VGK):    +42.7
  Sorokin (NYI):     +42.1
  Vasilevskiy (TBL): +32.8

20252026 bottom goalies:
  Binnington (STL):  -18.1
  Merilainen (OTT):  -17.2

goalie_adj applied to goal predictions:
  goalie_adj = 1 - (gsax_last20 / avg_shots_faced)
  Capped at +/-30% (0.70 to 1.30)
  avg_shots_faced = 28.8

==============================================================
B2B EFFECTS
==============================================================

File: data/processed/b2b_effects.json
Script to rebuild: run the B2B computation script in terminal

Season weights:
  Team:   current=0.50, prev=0.30, 2 seasons ago=0.20
  Player: current=0.40, prev=0.25, -2=0.20, -3=0.10, -4=0.05

Trust blending: raw_ratio blended toward 1.0 (no effect)
  Full trust at 30+ B2B games, blends to 1.0 with fewer

League-wide effects (20202021-20252026, weighted):
  ev_shots_on_goal_for:      ratio=0.9720  (-2.8%)
  ev_shots_on_goal_against:  ratio=1.0296  (+3.0%)
  goals_for:                 ratio=0.9287  (-7.1%)
  goals_against:             ratio=1.0423  (+4.2%)
  ev_shot_attempts_for:      ratio=0.9913  (-0.9%)
  ev_shot_attempts_against:  ratio=1.0201  (+2.0%)
  pp_toi:                    ratio=0.9779  (-2.2%)

By position (league-wide, weighted):
  Forwards: shots -1.2%, goals -4.3%, points -4.2%, TOI +1.4%
  Defense:  shots +0.2%, goals -6.8%, points -3.8%, TOI +1.7%

Team shot_for ratios on B2B (sorted, weighted):
  Most affected:  UTA 0.910, SJS 0.917, EDM 0.917
  Least affected: TBL 1.030, DET 1.025, MTL 1.020

Key player shot ratios (weighted, blended with league avg):
  Gauthier:    0.860 (-14%)
  MacKinnon:   0.847 (-15%)
  Ovechkin:    1.161 (+16%)  -- unique, improves on B2B
  McDavid:     0.942
  Nylander:    0.932
  Matthews:    0.950
  Draisaitl:   0.949
  Kucherov:    1.056
  Eichel:      1.019

B2B Integration in predict_player_props.py:
  detect_b2b(team, date_str): checks team game logs for yesterday's game
  get_team_b2b_ratio(team, stat, b2b_data): team or league fallback
  get_player_b2b_ratio(pid, stat, is_defense, b2b_data): player or pos fallback
  Combined ratio: 0.60 * player_ratio + 0.40 * team_ratio
  Applied to ev_shots and pp_shots when is_b2b=True
  B2B label shown in per-team header in prediction output

==============================================================
TEAM MODEL (Phase 2, COMPLETED)
==============================================================

Performance:
  Win probability AUC: 0.7544
  Goals MAE:           1.319
  EV Shots MAE:        4.512
  PP TOI MAE:          0.776

Status: Separate model, not yet integrated with player predictions.
Next step: Feed aggregated player shot predictions into team model
as features to improve win probability.

==============================================================
SPORTSBOOK RESEARCH & EDGE ANALYSIS
==============================================================

How books set prop lines:
- Use last 5-10 game rolling windows (NOT season averages)
- Exploit recency bias deliberately -- set lines high after hot
  streaks knowing public money follows overs
- Simulate 10,000 games per matchup
- Prop lines are softer than game lines (less sharp money, lower limits)
- Public over-bias: fans want big performances, pushes over-lines higher

Academic backing:
- Paul & Weinbach: sportsbooks do NOT price to balance the book
- Shank: books actively use recency bias to get bettors on losing side
- Wizard of Odds: with only 5-10 games of hot performance, career data
  should be weighted 80-90% -- most bettors do the opposite

Our edge:
1. Zone matchup context (we have this, books mostly don't)
2. Goalie GSAx integration (more rigorous than most books)
3. B2B adjustments (player + team specific)
4. Regression to mean (we anchor to season avg, books over-react)

Gauthier o2.5 case study (20252026):
  Actual hit rate:   66.2% (47/71 games)
  Book implied prob: 76.2% (priced at -320) -- OVERPRICED ~10%
  Our model:         53.7% -- UNDER-PREDICTING ~12%
  True probability:  ~66%
  Conclusion: books over-exploit recency bias, we under-predict
  Fix needed: calibration improvement for high-volume shooters
  
Gauthier season avg shots: 3.73/game
  Distribution: 0=4, 1=4, 2=16, 3=14, 4=10, 5=9, 6=6, 7=3

==============================================================
ZONE PROFILE DISPLAY (predict_player_props.py)
==============================================================

print_zone_matchup() output:
- Top 6 players per team: zone proportions vs position avg
- Format: 43%(+0%) showing actual% and diff from position avg
- Team zone shots ALLOWED vs league avg (last 30 games)
- Loaded from: zone_averages.json, player_zone_shots.csv,
  team_zone_defense.csv

Example output (EDM vs VGK):
  McDavid  F: net=59%(+16%) slot=34%(+1%)  -- net-front specialist
  Hyman    F: net=78%(+35%)               -- extreme net-front
  Bouchard D: slot=35%(+17%) l.pt=26%(+5%) -- unusual slot D
  Eichel   F: net=30%(-13%) r.flk=19%(+10%) -- flank shooter
  EDM allows: net=34%(-0%) slot=32%(+3%)  -- slightly slot-heavy
  VGK allows: net=32%(-2%) slot=32%(+3%)  -- similar profile

==============================================================
NAME MATCHING SYSTEM
==============================================================

Three-tier fuzzy matching:
  primary:   exact full_name match
  secondary: normalized name (no accents, lowercase, no hyphens)
  tertiary:  last name only, filtered by team

Nickname expansion (both directions):
  matt <-> matthew, mike <-> michael, alex <-> alexander, etc.

Multi-word last names: tries "First Last" shortening

==============================================================
PP SHARE CALCULATION
==============================================================

Layer 1 (60%): Team PP baseline from last 20 PP games
  PP1 top players get ~67% of PP TOI
  PP2 players get ~33%
Layer 2 (30%): Recent player PP share (last 10 PP games from PBP)
Layer 3 (20%): Historical season PP share from pp_shares.csv
Fallbacks: PP1=0.670, PP2=0.330, non-PP=0.030

==============================================================
ROADMAP
==============================================================

COMPLETED
[x] Phase 1: Data infrastructure
    - fetch_shot_data.py (11 seasons PBP shots)
    - fetch_player_pbp_stats.py (6 seasons strength-split)
    - fetch_goalie_game_logs.py
    - fetch_daily_lineups.py (DailyFaceoff scraper)
    - EN TOI classification fix (6v5=EV, 6v4=PP, 5v6=SH)
[x] Phase 2: Team model
    - Win AUC 0.7544, Goals MAE 1.319
[x] Phase 3 (in progress):
    - xG model (XGBoost on shot location/type/strength)
    - Shot zone classification (7 zones + out_of_ozone)
    - build_zone_features.py script
    - Zone features in process_player_features.py
    - Zone profile display in predict_player_props.py
    - Goalie GSAx integration in goal formula
    - B2B effects computed (team + player, weighted by season)
    - B2B integration in predict_player_props.py
    - Empirical prop calibration (prop_calibration.json)
    - Player model: 7 XGBoost models

IN PROGRESS
[ ] Calibration fix for high-volume shooters
    Problem: lambda=3.73 -> we predict 54%, truth is 66%
    Hypothesis: calibration uses season_avg as lookup key but
    we should use predicted lambda (slightly different)
    Fix: rebuild calibration stratified by shot tier,
    verify lookup key matches predicted lambda

NEXT STEPS (priority order)

1. GOALIE SAVE MODEL (new script: build_goalie_model.py)
   - Predict: saves, save_pct, goals_allowed per game
   - Features: gsax_last20/30, team_zone_defense (opponent),
     team_zone_offense (own team), B2B, home/away,
     shots_against_last10, opponent_ev_shots_for_last20
   - No circular reference (does not use player preds)
   - Output: predicted_saves, predicted_gaa
   - Feeds into: player goal formula, team model
   - Add save prop to prediction output

2. PLAYER -> TEAM INTEGRATION
   - In predict_player_props.py: sum player shot predictions
     per team -> feed into team win probability
   - team_expected_shots = sum(player_shots)
   - team_expected_xg = sum(player_goals)  
   - Pass these into team model as override/supplement

3. CALIBRATION FIX
   - Rebuild prop_calibration.json using predicted lambda
     not season_avg as the lookup key
   - Stratify by position (F vs D) and shot tier
   - Use 20232024 + 20252026 only (cleaner recent data)
   - Verify Gauthier: lambda=3.73 should map to ~66%

4. IS_B2B AS ROLLING FEATURE IN PROCESS_PLAYER_FEATURES
   - Add is_b2b flag per game to player_features.csv
   - Let XGBoost learn B2B effects from data directly
   - Complements explicit B2B ratio adjustments

5. BACKTEST (20242025 season)
   - Run predict_player_props.py on each game day
   - Compare predicted probs to actual outcomes
   - Compute: Brier score, calibration curve, ROI simulation
   - Identify props with best historical edge
   - Validate B2B adjustments improve accuracy

6. BOOK LINE INTEGRATION (last)
   - Manually input or scrape closing lines
   - Edge calculator: our_prob vs book_implied_prob
   - Flag bets with >5% edge
   - Track results over time

DEFERRED (good ideas, lower priority)
- Score effects model: good for live betting, not pre-game
  (we can't know pre-game score so not useful for props)
- Sub-position splits (C vs LW vs RW):
  data structure supports it, need more samples per bucket
- Venue/arena effects (some rinks track shots differently)
- Referee effects on PP frequency

==============================================================
KEY ARCHITECTURAL DECISIONS & RATIONALE
==============================================================

WHY FORMULA FOR SHOTS INSTEAD OF XGBOOST?
XGBoost regresses toward mean, systematically underestimates
high-volume shooters. McDavid/Gauthier get pulled to average.
Formula (rate * TOI) preserves true talent and captures
game-to-game variance. ev_shots MAE: 0.865 -> 0.625 (-28%)
after adding zone features to the formula approach.

WHY NEGBIN INSTEAD OF POISSON?
Shot counts are overdispersed (variance > mean). NegBin r=3.87
fitted from empirical shot data gives better tail probabilities.
Poisson would underestimate P(>4 shots) for volume shooters.

WHY 60/40 BLEND FOR SHOTS?
60% last10 / 40% season. Captures hot/cold streaks while
anchoring to true talent. Books use last 5-10 games only --
they over-react to streaks (exploitable edge for us).
PP shots: 50/50 because PP role is more stable than EV.

WHY EMPIRICAL CALIBRATION OVER PURE NEGBIN?
NegBin under-predicts o2.5 for high-average players.
Empirical curves from historical data capture actual hit rates.
Gauthier: NegBin gives 54%, empirical truth 66%.
Problem: we're still off, suggesting calibration lookup fix needed.

WHY COMBINE PLAYER + TEAM B2B RATIOS?
Individual player B2B samples are noisy (Gauthier: 23 games).
60% player + 40% team gives more stable estimates while
preserving player-specific tendencies (Ovechkin improves,
Gauthier drops significantly). Trust-blended toward 1.0
when sample < 30 games.

WHY ZONE FEATURES HELP SO MUCH?
Zone proportions tell the model WHERE a player shoots from,
not just how many. A net-front forward facing a team that
allows lots of slot shots has fundamentally different
expected output than a point-shooting defenseman.
Zone matchup: player_zone_profile vs team_zone_defense.

WHY GOALIE ADJ CAPPED AT +/-30%?
Prevents extreme adjustments from small sample GSAx.
+30% means goalie adds 30% more goals against expected.
-30% means goalie prevents 30% of expected goals.
Real range in data is roughly -20% to +20% for starters.

==============================================================
ENVIRONMENT & DEPENDENCIES
==============================================================

Python: 3.14.3
venv:   ~/nhl-props/venv
Activate: source venv/bin/activate

Key packages:
  xgboost, pandas, numpy, scipy, scikit-learn
  requests, beautifulsoup4, matplotlib

Data sources:
  NHL API (stats.nhl.com) -- game logs, player stats, goalie stats
  MoneyPuck -- shot coordinate data (PBP)
  DailyFaceoff -- daily lineup scraping

GitHub: github.com/brudnerbot/nhl-props
Branch: master
Data files: gitignored (too large)
Scripts:    committed

To run a prediction:
  cd ~/nhl-props
  source venv/bin/activate
  python scripts/fetch_daily_lineups.py
  python scripts/predict_player_props.py EDM VGK
  python scripts/predict_player_props.py ANA TOR --date 2026-03-30
==============================================================
UPDATES — April 2026
==============================================================

COMPLETED SINCE LAST NOTES UPDATE:

[x] Calibration fix (prop_calibration.json)
    - Was using season_avg as lookup key -> under-predicted high-volume players
    - Now uses blended key: 50% season_avg + 50% last20
    - Gauthier o2.5: 53.7% -> 58% (actual truth 66.2%)
    - Kreider points o0.5: 63.6% -> 53.9% (was massively over-predicted)
    - Killorn points o0.5: 31.9% -> 39.4%

[x] Removed manual assist blend
    - Was blending XGBoost 50/50 with recent rate formula
    - Tested: XGBoost alone MAE 0.337 beats all manual blends
    - Season avg alone MAE 0.334 also beats blends
    - Conclusion: manual blending hurts, trust the model

[x] Full prop line ladder in prediction output
    - Shots: u2.5, o2.5, u3.5, o3.5, o4.5, o5.5, o6.5
    - Goals: anytime (o0.5), 2+ goals (o1.5)
    - Points: o0.5, o1.5, o2.5
    - All use blended calibration key

[x] is_b2b as rolling feature in process_player_features.py
    - Added flag per game, added to ROLL_STATS
    - 15.2% of games are B2B
    - Model performance unchanged (XGBoost learns average pattern)
    - Columns: 493 -> 500

[x] fetch_team_game_logs.py extended to 20152016
    - Was only fetching 20202021+
    - Now covers all 11 seasons matching goalie/player data
    - 26,856 rows (13,428 games x 2 teams)

[x] build_goalie_model.py created
    - XGBoost models for saves and goals_against
    - saves MAE: 5.276 vs baseline 5.243 (barely beats)
    - goals_against MAE: 1.231 vs baseline 1.242 (slightly better)

[x] Goalie save props in prediction output
    - pred_saves = opp_team_shots * regressed_save_pct
    - Shows o24.5, o27.5, o29.5 lines per goalie
    - regressed_save_pct = league_avg (0.906) + gsax_adj

KEY FINDING — Goalie save% is unpredictable game-to-game:
    save_pct_last30 corr with actual save_pct: 0.070
    Baseline MAE (predict mean): 0.0450
    Best rolling predictor MAE:  0.0466
    Single-game save% dominated by shot quality variance and luck
    
    Correct architecture:
    - Predict shots_against from opponent shot model (already doing)
    - Use GSAx as quality adjustment in player goal formula (already doing)
    - saves = opp_shots * league_avg_save_pct (formula, no model needed)
    - No separate goalie save model needed

KEY FINDING — save% -> goals_against sensitivity:
    0.005 error in save_pct -> 0.15 error in goals_against (30 shots)
    0.010 error in save_pct -> 0.30 error in goals_against
    This is why GSAx (cumulative quality signal) beats single-game save%

CURRENT ROADMAP STATUS:

COMPLETED:
[x] xG model
[x] Shot zone classification + features
[x] Zone profile display in predictions
[x] EN TOI fix
[x] Goalie GSAx integration
[x] B2B effects (team + player, weighted, blended)
[x] B2B integration in predictions
[x] Prop calibration (empirical, blended key)
[x] Full prop line ladder output
[x] is_b2b as rolling feature
[x] Team game logs extended to 20152016
[x] Goalie save props output

NEXT STEPS:
1. BACKTEST (20242025 season)
   - Run predictions on past games
   - Compare predicted probs to actual outcomes
   - Brier score, calibration curve, ROI simulation
   - Identify props with best historical edge

2. PLAYER -> TEAM INTEGRATION
   - Aggregate player shot predictions -> team expected shots
   - Feed into team win probability model
   - Avoid circular reference: team uses aggregated player output

3. BOOK LINE INTEGRATION (last)
   - Input closing lines manually or via scraper
   - Edge calculator: our_prob vs book_implied_prob
   - Track results over time

DEFERRED:
- Score effects (live betting only, not pre-game)
- Sub-position splits (C vs LW vs RW)
- Venue/arena shot-tracking effects
- Referee PP frequency effects
