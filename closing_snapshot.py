"""
Closing-line snapshot from ParlayAPI (costs ~3 credits per run, whole slate in one call).

Run shortly before kickoff windows. Keeps only real sportsbooks (no DFS sites), only games that
start within the next CLOSE_WINDOW_MIN minutes, one main line per player per book.
Writes closing_YYYYMMDD_HHMM.csv; score_week.py uses these as the closing line for CLV.

Needs PARLAY_API_KEY. Output folder: NFL_OUTPUT_DIR if set, else Google Drive, else here.
"""
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
import requests

BASES = [os.environ["PARLAY_BASE_URL"].rstrip("/")] if os.environ.get("PARLAY_BASE_URL") \
    else ["https://parlay-api.com", "https://api.parlay-api.com"]
SPORT = "americanfootball_nfl"
MARKET = "player_rec_yds"
CLOSE_WINDOW_MIN = int(os.environ.get("CLOSE_WINDOW_MIN", "150"))   # games starting within this long
# Real sportsbooks only. DFS/pick'em sites (PrizePicks, Pick6, Underdog, Sleeper) list several
# fixed-price lines per player; exchanges and sweepstakes books are left out too.
SPORTSBOOKS = {"draftkings", "fanduel", "betmgm", "caesars", "pinnacle", "bovada", "betonlineag",
               "espnbet", "hardrockbet", "betrivers", "fanatics", "bet365", "williamhill_us",
               "mybookieag", "betus", "lowvig", "prophetx"}
DRIVE_ROOT = "/content/drive/MyDrive"
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def out_dir() -> str:
    env = os.environ.get("NFL_OUTPUT_DIR")
    if env:
        os.makedirs(env, exist_ok=True)
        return env
    if os.path.isdir(DRIVE_ROOT):
        d = DRIVE_ROOT + "/nfl_projections"
        os.makedirs(d, exist_ok=True)
        return d
    return "."


def name_key(name) -> str:
    s = re.sub(r"\([^)]*\)", " ", str(name))                  # "AJ Barner (SEA)" -> "AJ Barner"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`\-]", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(p for p in s.split() if p not in _SUFFIXES)


def dec(a):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / -a)


def fetch(key: str):
    last = None
    for base in BASES:
        try:
            r = requests.get(f"{base}/v1/sports/{SPORT}/props",
                             params={"markets": MARKET, "limit": 10000},
                             headers={"X-API-Key": key}, timeout=60)
        except requests.RequestException as e:
            last = e
            continue
        if r.status_code == 404 and base != BASES[-1]:
            continue
        return r
    raise SystemExit(f"Could not reach ParlayAPI: {last}")


def early_game_today() -> bool:
    """True if an NFL game kicks off before noon ET today (London/Madrid/Munich-type games)."""
    try:
        import nflreadpy as nfl
        today = pd.Timestamp.now(tz="America/New_York")
        sched = nfl.load_schedules([today.year if today.month > 2 else today.year - 1]).to_pandas()
        g = sched[(sched["game_type"] == "REG") & (sched["gameday"].astype(str) == today.strftime("%Y-%m-%d"))]
        return bool((g["gametime"].astype(str) < "12:00").any())
    except Exception as e:                       # if the schedule can't be read, take the snapshot
        print(f"Could not check the schedule ({e}); taking the snapshot anyway.")
        return True


def main():
    if os.environ.get("EARLY_GATE") == "1" and not early_game_today():
        print("No early-morning game today; skipping (no credits used).")
        return
    key = os.environ.get("PARLAY_API_KEY", "").strip()
    if not key:
        print("PARLAY_API_KEY is not set; skipping closing snapshot.")
        return
    r = fetch(key)
    print(f"HTTP {r.status_code} | credits used {r.headers.get('x-requests-last')} "
          f"| remaining {r.headers.get('x-requests-remaining')}")
    if r.status_code != 200:
        print(r.text[:300])
        return
    body = r.json()
    rows = pd.DataFrame(body if isinstance(body, list) else body.get("data", body.get("props", [])))
    if rows.empty:
        print("No prop rows returned.")
        return

    now = pd.Timestamp.now(tz="UTC")
    rows["commence"] = pd.to_datetime(rows["commence_time"], utc=True, errors="coerce")
    rows = rows[rows["bookmaker"].isin(SPORTSBOOKS)
                & rows["over_price"].notna() & rows["under_price"].notna()]
    rows = rows[(rows["commence"] >= now - pd.Timedelta(minutes=5))
                & (rows["commence"] <= now + pd.Timedelta(minutes=CLOSE_WINDOW_MIN))]
    if rows.empty:
        print(f"No sportsbook lines for games starting in the next {CLOSE_WINDOW_MIN} minutes.")
        return

    rows["name_key"] = rows["player"].map(name_key)
    do, du = dec(rows["over_price"]), dec(rows["under_price"])
    rows["fair_p_over"] = (1 / do) / (1 / do + 1 / du)
    rows["_balance"] = (rows["fair_p_over"] - 0.5).abs()
    # main line = the one with the most balanced prices, per book/player/game
    main_line = (rows.sort_values("_balance")
                 .drop_duplicates(["bookmaker", "name_key", "canonical_event_id"]))
    df = main_line.rename(columns={"bookmaker": "book"})[
        ["book", "player", "name_key", "line", "over_price", "under_price", "fair_p_over",
         "commence_time", "age_seconds"]].copy()
    df["commence_time"] = main_line["commence"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()
    df["snap_time"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    path = os.path.join(out_dir(), f"closing_{now.strftime('%Y%m%d_%H%M')}.csv")
    df.sort_values(["name_key", "book"]).to_csv(path, index=False)
    print(f"Saved {len(df)} lines ({df['name_key'].nunique()} players, {df['book'].nunique()} books: "
          f"{sorted(df['book'].unique())}) -> {path}")
    print(f"Median line age {df['age_seconds'].median():.0f}s")


if __name__ == "__main__":
    sys.exit(main())
