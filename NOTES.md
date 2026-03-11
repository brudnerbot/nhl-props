# NHL Props Model — Project Notes

## Setup Complete
- Python 3.14.3, venv at ~/nhl-props/venv
- GitHub: github.com/brudnerbot/nhl-props
- Libraries: pandas, requests, gspread, google-auth, scikit-learn, xgboost, openpyxl

## Goal
Game-by-game data for all players and teams, 5 seasons back.
Stats at each strength (EV, PP, SH): goals, assists, shots, hits, blocks, TOI, and more.
Eventually build expected goals (xG) model.

## Current Status
- Project structure created
- .gitignore configured
- Next: write scripts/fetch_game_logs.py to pull from NHL API