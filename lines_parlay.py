"""
ParlayAPI player-prop lines for receiving yards, rushing yards and receptions, in one call
(3 credits whatever the number of markets).

What this module takes care of:
  * the same stat comes back under different market keys depending on the book
    (player_rec_yds / player_receiving_yards / player_reception_yds / ...): mapped to one `stat`
  * DFS and pick'em sites (PrizePicks, Pick6, Underdog, Sleeper), exchanges and sweepstakes books are
    dropped: they list several fixed-price lines per player. Real sportsbooks only.
  * one main line per book/player/game (the one with the most balanced prices)
  * player names like "AJ Barner (SEA)" -> "aj barner"
  * server errors (5xx) are retried

Stats: "rec_yds", "rush_yds", "receptions".
"""
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import requests

BASES = [os.environ["PARLAY_BASE_URL"].rstrip("/")] if os.environ.get("PARLAY_BASE_URL") \
    else ["https://parlay-api.com", "https://api.parlay-api.com"]
SPORT = "americanfootball_nfl"
MARKETS_ASKED = "player_rec_yds,player_rush_yds,player_receptions"
STATS = ("rec_yds", "rush_yds", "receptions")
SPORTSBOOKS = {"draftkings", "fanduel", "betmgm", "caesars", "pinnacle", "bovada", "betonlineag",
               "espnbet", "hardrockbet", "betrivers", "fanatics", "bet365", "williamhill_us",
               "mybookieag", "betus", "lowvig", "prophetx"}
LINES_CACHE_MINUTES = 60
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_EXCLUDE = ("alternate", "1st", "2nd", "3rd", "4th", "quarter", "half", "_td", "tds", "attempt",
            "longest", "first", "anytime", "combo", "passing", "pass_", "+")


def get_key():
    key = os.environ.get("PARLAY_API_KEY", "").strip()
    if not key:
        try:
            from google.colab import userdata
            key = (userdata.get("PARLAY_API_KEY") or "").strip()
        except Exception:
            key = ""
    return key


def name_key(name) -> str:
    s = re.sub(r"\([^)]*\)", " ", str(name))                  # "AJ Barner (SEA)" -> "AJ Barner"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`\-]", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(p for p in s.split() if p not in _SUFFIXES)


def classify(market_key):
    """Map a book-specific market key to one of STATS, or None (alternates, periods, TDs...)."""
    m = str(market_key).lower()
    if any(t in m for t in _EXCLUDE):
        return None
    if "rush" in m and ("recept" in m or "receiv" in m or "rec_" in m):   # rush+rec combo markets
        return None
    if "rush" in m and ("yd" in m or "yard" in m):
        return "rush_yds"
    if "recept" in m and "yd" not in m and "yard" not in m:      # player_receptions
        return "receptions"
    if ("rec_yd" in m or "receiv" in m or "reception_yd" in m or "recept" in m) and ("yd" in m or "yard" in m):
        return "rec_yds"
    return None


def _dec(a):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / -a)


def fetch_raw(key: str, markets: str = MARKETS_ASKED, tries: int = 4):
    """GET the props slate. Returns (DataFrame, response) or (None, None) after repeated failures."""
    last = None
    for attempt in range(tries):
        for base in BASES:
            try:
                r = requests.get(f"{base}/v1/sports/{SPORT}/props",
                                 params={"markets": markets, "limit": 10000},
                                 headers={"X-API-Key": key}, timeout=60)
            except requests.RequestException as e:
                last = e
                continue
            if r.status_code == 404 and base != BASES[-1]:
                continue
            if r.status_code < 500:
                if r.status_code != 200:
                    print(f"ParlayAPI HTTP {r.status_code}: {r.text[:200]}")
                    return None, r
                body = r.json()
                rows = body if isinstance(body, list) else body.get("data", body.get("props", []))
                return pd.DataFrame(rows), r
            last = f"HTTP {r.status_code} {r.text[:120]}"
            break
        if attempt < tries - 1:
            wait = 5 * (attempt + 1)
            print(f"ParlayAPI error ({last}); retrying in {wait}s ({attempt + 1}/{tries - 1})")
            time.sleep(wait)
    print(f"ParlayAPI still failing after {tries} tries ({last}).")
    return None, None


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw rows -> one main line per (book, stat, player, game), sportsbooks only."""
    cols = ["book", "player", "name_key", "stat", "line", "over_price", "under_price",
            "fair_p_over", "commence_time", "event_id", "age_seconds", "market_key"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)
    d = raw.copy()
    if "period" in d.columns:
        d = d[d["period"].isna() | d["period"].astype(str).str.upper().isin(["FULL", "UNKNOWN"])]
    d = d[d["bookmaker"].isin(SPORTSBOOKS)]
    d["stat"] = d["market_key"].map(classify)
    d = d.dropna(subset=["stat", "line", "over_price", "under_price", "player"])
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["name_key"] = d["player"].map(name_key)
    do, du = _dec(d["over_price"]), _dec(d["under_price"])
    d["fair_p_over"] = (1 / do) / (1 / do + 1 / du)
    d["_bal"] = (d["fair_p_over"] - 0.5).abs()
    ev = d["canonical_event_id"] if "canonical_event_id" in d.columns else d.get("event_id", "")
    d["event_id"] = ev
    d["commence_time"] = pd.to_datetime(d["commence_time"], utc=True, errors="coerce")
    d = (d.sort_values("_bal")
         .drop_duplicates(["bookmaker", "stat", "name_key", "event_id"])
         .rename(columns={"bookmaker": "book"}))
    if "age_seconds" not in d.columns:
        d["age_seconds"] = np.nan
    d["commence_time"] = d["commence_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return d[cols].reset_index(drop=True)


def get_lines(key: str, season: int, week: int, out_dir: str, window=None) -> pd.DataFrame:
    """Tidy lines for the target week's games (window = (start, end) UTC timestamps), cached for an hour."""
    raw_path = os.path.join(out_dir, f"lines_parlay_{season}_wk{week}_raw.csv")
    if os.path.exists(raw_path):
        cached = pd.read_csv(raw_path)
        if len(cached) and "fetched_at" in cached:
            age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(cached["fetched_at"].iloc[0])).total_seconds() / 60
            if age < LINES_CACHE_MINUTES:
                print(f"Using cached ParlayAPI lines fetched {age:.0f} min ago")
                return cached
    raw, resp = fetch_raw(key)
    if raw is None:
        return pd.DataFrame()
    if resp is not None:
        print(f"ParlayAPI: {len(raw)} raw rows | credits used {resp.headers.get('x-requests-last')} "
              f"| remaining {resp.headers.get('x-requests-remaining')}")
    df = tidy(raw)
    if window is not None and len(df):
        t = pd.to_datetime(df["commence_time"], utc=True)
        df = df[(t >= window[0]) & (t <= window[1])].reset_index(drop=True)
    if len(df):
        df["fetched_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        df.to_csv(raw_path, index=False)
        counts = df.groupby("stat")["name_key"].nunique().to_dict()
        print(f"ParlayAPI sportsbook lines kept (players per stat): {counts}")
    return df
