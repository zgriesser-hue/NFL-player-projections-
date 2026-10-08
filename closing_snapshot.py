"""
Closing-line snapshot from ParlayAPI (~3 credits per run, whole slate and all three stats in one call).

Run shortly before kickoff windows. Keeps only real sportsbooks (no DFS sites), only games that
start within the next CLOSE_WINDOW_MIN minutes, one main line per player per book per stat
(receiving yards, rushing yards, receptions; see lines_parlay.py).
Writes closing_YYYYMMDD_HHMM.csv; score_week.py / score_extra.py use these as the closing line for CLV.

Needs PARLAY_API_KEY. Output folder: NFL_OUTPUT_DIR if set, else Google Drive, else here.
"""
import os
import sys

import pandas as pd

import lines_parlay as L

CLOSE_WINDOW_MIN = int(os.environ.get("CLOSE_WINDOW_MIN", "150"))   # games starting within this long
DRIVE_ROOT = "/content/drive/MyDrive"


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
    key = L.get_key()
    if not key:
        print("PARLAY_API_KEY is not set; skipping closing snapshot.")
        return
    raw, resp = L.fetch_raw(key)
    if raw is None:
        print("No closing snapshot this run. CLV falls back to our last Odds API snapshot "
              "(receiving yards) for these games.")
        return
    print(f"HTTP {resp.status_code} | credits used {resp.headers.get('x-requests-last')} "
          f"| remaining {resp.headers.get('x-requests-remaining')}")
    df = L.tidy(raw)
    now = pd.Timestamp.now(tz="UTC")
    t = pd.to_datetime(df["commence_time"], utc=True)
    df = df[(t >= now - pd.Timedelta(minutes=5)) & (t <= now + pd.Timedelta(minutes=CLOSE_WINDOW_MIN))]
    if df.empty:
        print(f"No sportsbook lines for games starting in the next {CLOSE_WINDOW_MIN} minutes.")
        return
    df = df.assign(snap_time=now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    path = os.path.join(out_dir(), f"closing_{now.strftime('%Y%m%d_%H%M')}.csv")
    df.sort_values(["stat", "name_key", "book"]).to_csv(path, index=False)
    per = df.groupby("stat")["name_key"].nunique().to_dict()
    print(f"Saved {len(df)} lines (players per stat {per}; books {sorted(df['book'].unique())}) -> {path}")
    print(f"Median line age {df['age_seconds'].median():.0f}s")


if __name__ == "__main__":
    sys.exit(main())
