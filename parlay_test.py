"""
One-off check of ParlayAPI for NFL receiving-yards props (and its closing-odds endpoint).

Needs PARLAY_API_KEY (env var / GitHub secret). Costs about 3 credits for the props call,
plus 10 if you also test closing odds (set PARLAY_TEST_CLOSING_DATE=YYYY-MM-DD, a day in the
last 48 hours that had NFL games). Prints what came back so we can judge compatibility.
"""
import json
import os
import sys
import time

import requests

KEY = os.environ.get("PARLAY_API_KEY", "").strip()
BASES = [os.environ.get("PARLAY_BASE_URL", "").rstrip("/")] if os.environ.get("PARLAY_BASE_URL") \
    else ["https://parlay-api.com", "https://api.parlay-api.com"]
SPORT = "americanfootball_nfl"
MARKET = "player_rec_yds"
OUT = os.environ.get("NFL_OUTPUT_DIR", ".")


def credit_headers(r):
    return {k: v for k, v in r.headers.items()
            if any(t in k.lower() for t in ("request", "credit", "ratelimit", "total", "count", "complete"))}


def get(path, params=None):
    last = None
    for base in BASES:
        try:
            r = requests.get(base + path, params=params or {}, headers={"X-API-Key": KEY}, timeout=60)
        except requests.RequestException as e:
            last = f"{base}{path}: {e}"
            continue
        if r.status_code == 404 and base != BASES[-1]:
            last = f"{base}{path}: 404"
            continue
        for wait in (3, 8):                      # server errors: retry a couple of times
            if r.status_code < 500:
                break
            print(f"  HTTP {r.status_code} from {path}; retrying in {wait}s ...")
            time.sleep(wait)
            r = requests.get(base + path, params=params or {}, headers={"X-API-Key": KEY}, timeout=60)
        return base, r
    print("Could not reach the API:", last)
    sys.exit(0)


def main():
    if not KEY:
        print("PARLAY_API_KEY is not set (add it as a GitHub secret). Nothing to test.")
        return

    print("== 1. Market list (should include a receiving-yards market) ==")
    base, r = get("/v1/markets", {"sport": SPORT})
    print(f"base {base} | HTTP {r.status_code} | credits: {credit_headers(r)}")
    try:
        txt = json.dumps(r.json())
        hits = sorted({w.strip('"') for w in txt.replace(",", " ").split() if "rec" in w.lower() and "yd" in w.lower()})
        print("receiving-yard market keys seen:", hits[:15] or "none found")
    except ValueError:
        print(r.text[:300])

    print(f"\n== 2. Props: {MARKET} ==")
    base, r = get(f"/v1/sports/{SPORT}/props", {"markets": MARKET, "limit": 5000})
    print(f"HTTP {r.status_code} | credits: {credit_headers(r)}")
    rows = []
    if r.status_code != 200:
        print(r.text[:500])
    else:
        rows = r.json()
        rows = rows if isinstance(rows, list) else rows.get("data", rows.get("props", []))
        with open(os.path.join(OUT, "parlay_props_sample.json"), "w") as f:
            json.dump(rows[:300], f, indent=1)
        print(f"{len(rows)} rows")
    if r.status_code == 200 and not rows:
        print("No receiving-yards rows returned: either the market is not covered or lines are not posted yet.")
    elif rows:
        books = {}
        for x in rows:
            books[x.get("bookmaker")] = books.get(x.get("bookmaker"), 0) + 1
        games = {x.get("canonical_event_id") for x in rows}
        players = {x.get("player") for x in rows}
        periods = {x.get("period") for x in rows}
        print(f"games: {len(games)} | players: {len(players)} | periods: {periods}")
        print("rows per book:", dict(sorted(books.items(), key=lambda kv: -kv[1])))
        for x in rows[:6]:
            print({k: x.get(k) for k in ("bookmaker", "player", "market_key", "line",
                                         "over_price", "under_price", "commence_time", "age_seconds")})
        names = sorted(p for p in players if p)[:12]
        print("sample player names (check they match nflverse style):", names)

    print("\n== 2b. Several markets in ONE call (credit cost, and rushing/receptions coverage) ==")
    try:
        mk_txt = json.dumps(get("/v1/markets", {"sport": SPORT})[1].json())
        cands = sorted({w.strip('"[]{}') for w in mk_txt.replace(",", " ").replace(":", " ").split()
                        if ("rush" in w.lower() or "recept" in w.lower()) and w.lower().startswith('"player')})
        print("rushing/receptions market keys listed:", cands[:25] or "none found")
    except Exception as e:
        print("could not list markets:", e)
    multi = os.environ.get("PARLAY_TEST_MARKETS", "player_rec_yds,player_rush_yds,player_receptions")
    base, r = get(f"/v1/sports/{SPORT}/props", {"markets": multi, "limit": 10000})
    print(f"markets asked: {multi} | HTTP {r.status_code} | credits: {credit_headers(r)}")
    if r.status_code == 200:
        rows = r.json()
        rows = rows if isinstance(rows, list) else rows.get("data", rows.get("props", []))
        by_mk = {}
        for x in rows:
            d = by_mk.setdefault(x.get("market_key"), {"rows": 0, "books": set()})
            d["rows"] += 1
            d["books"].add(x.get("bookmaker"))
        for mk, d in sorted(by_mk.items(), key=lambda kv: -kv[1]["rows"]):
            print(f"  {mk}: {d['rows']} rows | books: {sorted(d['books'])}")
    else:
        print(r.text[:300])

    date = os.environ.get("PARLAY_TEST_CLOSING_DATE", "").strip()
    if date:
        print(f"\n== 3. Closing odds for {date} (10 credits) ==")
        base, r = get(f"/v1/historical/sports/{SPORT}/closing-odds", {"markets": MARKET, "date": date})
        print(f"HTTP {r.status_code} | credits: {credit_headers(r)}")
        try:
            body = r.json()
            rows = body if isinstance(body, list) else body.get("data", body.get("rows", body))
            print(f"{len(rows) if hasattr(rows, '__len__') else '?'} rows")
            with open(os.path.join(OUT, "parlay_closing_sample.json"), "w") as f:
                json.dump(rows[:300] if isinstance(rows, list) else rows, f, indent=1)
            print(json.dumps(rows[:3] if isinstance(rows, list) else rows, indent=1)[:1500])
        except ValueError:
            print(r.text[:500])
    else:
        print("\n(Closing-odds check skipped: set PARLAY_TEST_CLOSING_DATE=YYYY-MM-DD to try it, 10 credits.)")


if __name__ == "__main__":
    main()
