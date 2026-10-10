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


def log_status(msg: str) -> None:
    """Append one line per run to closing_status.txt so a missing snapshot can be diagnosed afterward."""
    print(msg)
    try:
        with open(os.path.join(out_dir(), "closing_status.txt"), "a", encoding="utf-8") as f:
            f.write(f"{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%MZ')} {msg}\n")
    except Exception:
        pass


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
        log_status("skipped: no early-morning game today (no credits used)")
        return
    key = L.get_key()
    if not key:
        log_status("FAILED: PARLAY_API_KEY is not set")
        return
    raw, resp = L.fetch_raw(key)
    if raw is None:
        log_status("FAILED: ParlayAPI returned no data (request error or empty response)")
        return
    print(f"HTTP {resp.status_code} | credits used {resp.headers.get('x-requests-last')} "
          f"| remaining {resp.headers.get('x-requests-remaining')}")
    df = L.tidy(raw)
    now = pd.Timestamp.now(tz="UTC")
    t = pd.to_datetime(df["commence_time"], utc=True)
    df = df[(t >= now - pd.Timedelta(minutes=5)) & (t <= now + pd.Timedelta(minutes=CLOSE_WINDOW_MIN))]
    if df.empty:
        n_games = t.nunique() if len(t) else 0
        nxt = t[t > now].min() if (t > now).any() else None
        log_status(f"NO SNAPSHOT: no sportsbook lines for games starting within {CLOSE_WINDOW_MIN} min "
                   f"(API had {n_games} start times; next kickoff {nxt})")
        return
    df = df.assign(snap_time=now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    path = os.path.join(out_dir(), f"closing_{now.strftime('%Y%m%d_%H%M')}.csv")
    df.sort_values(["stat", "name_key", "book"]).to_csv(path, index=False)
    per = df.groupby("stat")["name_key"].nunique().to_dict()
    log_status(f"saved {len(df)} lines, players per stat {per}, books {sorted(df['book'].unique())}")
    print(f"Median line age {df['age_seconds'].median():.0f}s")


if __name__ == "__main__":
    sys.exit(main())
