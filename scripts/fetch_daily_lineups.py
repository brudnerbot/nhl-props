"""
fetch_daily_lineups.py

Scrapes Daily Faceoff for all 32 NHL teams' current line combinations.
Stores structured lineup data including:
  - Forward lines (4 lines, LW/C/RW)
  - Defensive pairings (3 pairs)
  - PP units 1 and 2
  - PK units 1 and 2
  - Goalies (order = depth chart)
  - Injuries with status (IR/DTD/OUT)

Output: data/raw/lineups/lineups_YYYYMMDD.json

Usage:
    python scripts/fetch_daily_lineups.py           # today
    python scripts/fetch_daily_lineups.py 2026-03-27 # specific date
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

# --- CONFIG ---
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data/raw/lineups"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.dailyfaceoff.com/teams/{slug}/line-combinations"

# Team slug map: NHL abbreviation -> Daily Faceoff URL slug
TEAM_SLUGS = {
    "ANA": "anaheim-ducks",
    "BOS": "boston-bruins",
    "BUF": "buffalo-sabres",
    "CGY": "calgary-flames",
    "CAR": "carolina-hurricanes",
    "CHI": "chicago-blackhawks",
    "COL": "colorado-avalanche",
    "CBJ": "columbus-blue-jackets",
    "DAL": "dallas-stars",
    "DET": "detroit-red-wings",
    "EDM": "edmonton-oilers",
    "FLA": "florida-panthers",
    "LAK": "los-angeles-kings",
    "MIN": "minnesota-wild",
    "MTL": "montreal-canadiens",
    "NSH": "nashville-predators",
    "NJD": "new-jersey-devils",
    "NYI": "new-york-islanders",
    "NYR": "new-york-rangers",
    "OTT": "ottawa-senators",
    "PHI": "philadelphia-flyers",
    "PIT": "pittsburgh-penguins",
    "SJS": "san-jose-sharks",
    "SEA": "seattle-kraken",
    "STL": "st-louis-blues",
    "TBL": "tampa-bay-lightning",
    "TOR": "toronto-maple-leafs",
    "UTA": "utah-mammoth",
    "VAN": "vancouver-canucks",
    "VGK": "vegas-golden-knights",
    "WSH": "washington-capitals",
    "WPG": "winnipeg-jets",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def parse_lineup(html: str, team: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # --- Last updated ---
    last_updated = None
    for tag in soup.find_all("span"):
        text = tag.get_text(strip=True)
        if text.startswith("20") and "T" in text and len(text) < 35:
            last_updated = text
            break

    # --- Source ---
    source = None
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if "naturalstattrick" in href or "x.com" in href or "twitter.com" in href:
            source = a.get_text(strip=True) or href
            break

    # --- Walk DOM linearly, bucket players by section ---
    SECTION_IDS = {"rw-forwards": "forwards", "defense": "defense",
                   "powerplay": "powerplay", "goalies": "goalies"}

    current_section = None
    section_players = {"forwards": [], "defense": [], "powerplay": [], "goalies": [], "injuries": []}
    pending_badge = None

    for tag in soup.descendants:
        if not isinstance(tag, Tag):
            continue

        # Section detection
        if tag.name == "span" and tag.get("id") in SECTION_IDS:
            current_section = SECTION_IDS[tag.get("id")]
            pending_badge = None
            continue

        # Injury badges
        if tag.name in ["span", "div"]:
            text = tag.get_text(strip=True).upper()
            if text in ["IR", "DTD", "OUT", "GTD"] and len(text) <= 3:
                pending_badge = text
                continue

        # Player links
        if tag.name == "a" and "/players/news/" in tag.get("href", ""):
            name = tag.get_text(strip=True)
            if not name or not current_section:
                continue
            href = tag.get("href", "")
            dfo_id = href.rstrip("/").split("/")[-1]
            p = {"name": name, "dfo_id": dfo_id}

            if current_section == "goalies":
                # First 2-3 are goalies, rest are injuries
                if len(section_players["goalies"]) < 3 and pending_badge is None:
                    section_players["goalies"].append(p)
                else:
                    p["status"] = pending_badge or "OUT"
                    pending_badge = None
                    # Avoid duplicates
                    existing = {x["name"] for x in section_players["injuries"]}
                    if name not in existing:
                        section_players["injuries"].append(p)
            else:
                existing = {x["name"] for x in section_players[current_section]}
                if name not in existing:
                    section_players[current_section].append(p)

    # --- Structure forward lines (row order: LW,C,RW per line) ---
    fwd = section_players["forwards"]
    forward_lines = []
    for i in range(0, min(len(fwd), 15), 3):
        group = fwd[i:i+3]
        if not group:
            break
        line = {"line": len(forward_lines) + 1}
        line["lw"] = group[0] if len(group) > 0 else None
        line["c"]  = group[1] if len(group) > 1 else None
        line["rw"] = group[2] if len(group) > 2 else None
        forward_lines.append(line)

    # --- Defensive pairings ---
    dfn = section_players["defense"]
    defense_pairs = []
    for i in range(0, min(len(dfn), 8), 2):
        group = dfn[i:i+2]
        if not group:
            break
        pair = {"pair": len(defense_pairs) + 1}
        pair["ld"] = group[0] if len(group) > 0 else None
        pair["rd"] = group[1] if len(group) > 1 else None
        defense_pairs.append(pair)

    # --- Split powerplay: PP1(5) PP2(5) PK1(4) PK2(4) ---
    pp = section_players["powerplay"]
    pp1 = pp[0:5]
    pp2 = pp[5:10]
    pk1 = pp[10:14]
    pk2 = pp[14:18]

    return {
        "team":          team,
        "last_updated":  last_updated,
        "source":        source,
        "forward_lines": forward_lines,
        "defense_pairs": defense_pairs,
        "pp1": pp1, "pp2": pp2,
        "pk1": pk1, "pk2": pk2,
        "goalies":   section_players["goalies"],
        "injuries":  section_players["injuries"],
    }


def fetch_team_lineup(team: str, slug: str) -> dict | None:
    """Fetch and parse lineup for one team."""
    url = BASE_URL.format(slug=slug)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  [{team}] HTTP {resp.status_code}")
            return None
        return parse_lineup(resp.text, team)
    except Exception as e:
        print(f"  [{team}] Error: {e}")
        return None


def fetch_all_lineups(date_str: str = None) -> dict:
    """Fetch lineups for all 32 teams."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"Fetching NHL lineups for {date_str}...")
    print(f"  Teams: {len(TEAM_SLUGS)}")

    all_lineups = {}
    for i, (team, slug) in enumerate(TEAM_SLUGS.items()):
        print(f"  [{i+1:2d}/32] {team}...", end=" ", flush=True)
        lineup = fetch_team_lineup(team, slug)
        if lineup:
            all_lineups[team] = lineup
            forwards_count = len(lineup["forward_lines"]) * 3
            injuries_count = len(lineup["injuries"])
            goalie = lineup["goalies"][0]["name"] if lineup["goalies"] else "?"
            print(f"OK  ({forwards_count}F, G:{goalie}, inj:{injuries_count})")
        else:
            print("FAILED")
        time.sleep(0.5)  # polite rate limiting

    return {
        "date":    date_str,
        "fetched": datetime.now().isoformat(),
        "teams":   all_lineups,
    }


def save_lineups(data: dict, date_str: str):
    path = OUTPUT_DIR / f"lineups_{date_str.replace('-', '')}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved → {path}")
    print(f"  Teams fetched: {len(data['teams'])}/32")
    return path


def print_summary(data: dict, team: str = "EDM"):
    """Print a formatted summary for one team to verify parsing."""
    if team not in data["teams"]:
        print(f"  {team} not found")
        return
    t = data["teams"][team]
    print(f"\n{'='*50}")
    print(f"  {team} LINEUP  (updated: {t.get('last_updated','?')})")
    print(f"{'='*50}")

    print("\nFORWARD LINES:")
    for line in t["forward_lines"]:
        lw = line["lw"]["name"] if line["lw"] else "—"
        c  = line["c"]["name"]  if line["c"]  else "—"
        rw = line["rw"]["name"] if line["rw"] else "—"
        print(f"  L{line['line']}: {lw:<22} {c:<22} {rw}")

    print("\nDEFENSE PAIRS:")
    for pair in t["defense_pairs"]:
        ld = pair["ld"]["name"] if pair["ld"] else "—"
        rd = pair["rd"]["name"] if pair["rd"] else "—"
        print(f"  P{pair['pair']}: {ld:<22} {rd}")

    print("\nPOWER PLAY:")
    pp1 = [p["name"] for p in t["pp1"]]
    pp2 = [p["name"] for p in t["pp2"]]
    print(f"  PP1: {', '.join(pp1)}")
    print(f"  PP2: {', '.join(pp2)}")

    print("\nGOALIES:")
    for i, g in enumerate(t["goalies"]):
        role = "STARTER" if i == 0 else "BACKUP"
        print(f"  {role}: {g['name']}")

    if t["injuries"]:
        print("\nINJURIES:")
        for p in t["injuries"]:
            print(f"  {p['status']:4s} — {p['name']}")


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else None
    data = fetch_all_lineups(date_str)
    path = save_lineups(data, data["date"])

    # Print sample for verification
    print_summary(data, "EDM")
    print_summary(data, "VGK")


if __name__ == "__main__":
    main()