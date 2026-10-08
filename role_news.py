"""
Role and depth-chart news for the rushing projections.

The simulation only knows what happened in past games plus the official injury report. It cannot know
that a starter was cut, traded or moved to IR, or that a backup was promoted, until those changes show up
in game data. Books know immediately, which is where huge model-vs-book gaps come from.

This module gathers whatever it can reach (each source is optional and failure-tolerant) and writes a
short plain-English `role_note` per player, shown beside CHECK NEWS leans and in the rushing email:

  1. Our own game data: recent carries vs long-run form, rank among teammates, teammates who were
     carrying the ball in recent games but are Out or gone from the roster.
  2. nflverse depth charts (nflreadpy.load_depth_charts): this week's RB depth rank, if published.
  3. Sleeper's public player feed (no key): current team, roster status (IR / PUP / suspended / inactive),
     injury designation and depth-chart order. Catches trades and cuts the same day.

Notes are context for you, not model inputs. (A model input would need years of history for each source.)
"""
import pandas as pd
import requests

import projections as P

SLEEPER_URL = "https://api.sleeper.app/v1/players/nfl"
BAD_STATUS = {"injured reserve", "pup", "suspended", "inactive", "practice squad", "nfi", "reserve/covid"}


def fetch_sleeper() -> pd.DataFrame:
    try:
        r = requests.get(SLEEPER_URL, timeout=60)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        print(f"Role news: Sleeper feed unavailable ({type(e).__name__}); skipping it.")
        return pd.DataFrame()
    rows = []
    for v in raw.values():
        if v.get("position") not in ("RB", "QB", "FB"):
            continue
        rows.append(dict(name_key=P.name_key(v.get("full_name") or f"{v.get('first_name', '')} {v.get('last_name', '')}"),
                         s_team=v.get("team"), s_status=v.get("status"), s_injury=v.get("injury_status"),
                         s_depth=v.get("depth_chart_order"), s_pos=v.get("position")))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # prefer the row of a player that still has a team (duplicates exist for retired namesakes)
    df["_has_team"] = df["s_team"].notna()
    df = df.sort_values("_has_team", ascending=False).drop_duplicates("name_key").drop(columns="_has_team")
    print(f"Role news: Sleeper feed loaded ({len(df)} RB/QB/FB rows).")
    return df


def fetch_depth(season: int, week: int) -> pd.DataFrame:
    """RB depth rank (1 = starter) from nflverse depth charts. The schema changed in 2025: older files have
    week / depth_position / depth_team, newer ones have a snapshot date `dt` and pos_abb / pos_rank."""
    try:
        import nflreadpy as nfl
        d = nfl.load_depth_charts([season]).to_pandas()
        pick = lambda *names: next((c for c in names if c in d.columns), None)
        ide = pick("gsis_id", "player_id")
        pos = pick("depth_position", "pos_abb", "position", "pos_name")
        rank = pick("depth_team", "pos_rank", "pos_slot", "rank")
        if not all([ide, pos, rank]):
            raise ValueError(f"unexpected columns {list(d.columns)}")
        d = d[d[pos].astype(str).isin(["RB", "HB", "Running Back"])]
        wk = pick("week")
        if wk:
            d = d[d[wk] <= week]
            d = d[d[wk] == d[wk].max()]
            tag = f"week {int(d[wk].max()) if len(d) else '?'}"
        elif "dt" in d.columns:
            dt = pd.to_datetime(d["dt"], errors="coerce", utc=True)
            d = d[dt == dt.max()]
            tag = f"snapshot {dt.max():%Y-%m-%d}"
        else:
            tag = "latest"
        d = d.assign(depth_rank=pd.to_numeric(d[rank], errors="coerce")).dropna(subset=["depth_rank", ide])
        d = d.sort_values("depth_rank").drop_duplicates(ide)
        print(f"Role news: nflverse depth chart loaded ({tag}, {len(d)} RBs).")
        return d.rename(columns={ide: "player_id"})[["player_id", "depth_rank"]]
    except Exception as e:
        print(f"Role news: nflverse depth chart unavailable ({type(e).__name__}: {str(e)[:400]}); skipping it.")
        return pd.DataFrame()


def build_notes(hist: pd.DataFrame, fut: pd.DataFrame, sleeper=None, depth=None) -> pd.DataFrame:
    """One row per future player: player_id, role_note. `hist` = played rows, `fut` = this week's rows."""
    sleeper = sleeper if sleeper is not None else pd.DataFrame()
    depth = depth if depth is not None else pd.DataFrame()
    h = hist.sort_values(["season", "week"])
    t = h["season"] * 100 + h["week"]
    last3 = (h.assign(_t=t).groupby("player_id").tail(3).groupby("player_id")["carries"].mean())
    last1 = h.groupby("player_id").tail(1).set_index("player_id")["carries"]

    f = fut.copy()
    f["name_key"] = f["player_display_name"].map(P.name_key)
    rb = f[f["position"] == "RB"]
    f["form_rank"] = rb.groupby("team")["ewm_carries"].rank(ascending=False, method="first")
    if len(depth):
        f = f.merge(depth, on="player_id", how="left")
    if len(sleeper):
        f = f.merge(sleeper, on="name_key", how="left")

    # teammates who carried recently but are Out now or gone from this week's roster
    recent_t = t.max()
    recent = h[(t >= recent_t - 3)]            # roughly the last couple of weeks
    recent = recent[(recent["position"] == "RB") & (recent["carries"] >= 5)]
    rec_form = recent.groupby(["team", "player_id", "player_display_name"])["carries"].mean().reset_index()
    here = set(zip(fut["team"], fut["player_id"]))
    out_ids = set(fut.loc[fut["status"] == "Out", "player_id"])
    missing = {}
    for r in rec_form.itertuples():
        if (r.team, r.player_id) not in here:
            why = "not on roster"
        elif r.player_id in out_ids:
            why = "Out"
        else:
            continue
        missing.setdefault(r.team, []).append(f"{r.player_display_name} ({r.carries:.0f} car/g, {why})")

    notes = []
    for r in f.itertuples():
        n = []
        if r.position == "RB":
            l3, ew = last3.get(r.player_id), getattr(r, "ewm_carries", float("nan"))
            if l3 == l3 and ew == ew:
                if l3 >= 1.5 * ew + 2:
                    n.append(f"carries up lately (last 3 avg {l3:.0f} vs form {ew:.0f})")
                elif ew >= 8 and l3 <= 0.6 * ew - 2:
                    n.append(f"carries down lately (last 3 avg {l3:.0f} vs form {ew:.0f})")
            if r.form_rank == r.form_rank:
                n.append(f"RB{int(r.form_rank)} on team by recent carries")
            dr = getattr(r, "depth_rank", float("nan"))
            if dr == dr:
                n.append(f"depth chart RB{int(dr)}")
            for m in missing.get(r.team, [])[:2]:
                if not m.startswith(str(r.player_display_name)):
                    n.append(f"teammate {m}")
        sd, st, si = getattr(r, "s_depth", None), getattr(r, "s_team", None), getattr(r, "s_status", None)
        if isinstance(si, str) and si.lower() in BAD_STATUS:
            n.append(f"Sleeper status: {si}")
        if isinstance(getattr(r, "s_injury", None), str):
            n.append(f"Sleeper injury: {r.s_injury}")
        if isinstance(st, str) and P.fix_team(pd.Series([st])).iloc[0] != r.team:
            n.append(f"Sleeper lists team {st} (we have {r.team})")
        if sd == sd and sd is not None and r.position == "RB":
            n.append(f"Sleeper depth #{int(sd)}")
            ours = [x for x in (getattr(r, "depth_rank", float("nan")), r.form_rank) if x == x]
            if int(sd) == 1 and ours and min(ours) >= 2:
                n.append("LIKELY PROMOTED to starter: our projection may be too low")
            elif int(sd) >= 2 and r.form_rank == 1:
                n.append("possible DEMOTION: Sleeper has him behind another back")
        notes.append("; ".join(n))
    return pd.DataFrame({"player_id": f["player_id"].to_numpy(), "role_note": notes})


def get_notes(hist, fut, season: int, week: int) -> pd.DataFrame:
    try:
        return build_notes(hist, fut, fetch_sleeper(), fetch_depth(season, week))
    except Exception as e:                      # notes are optional context; never break the run
        print(f"Role news skipped ({type(e).__name__}: {e})")
        return pd.DataFrame(columns=["player_id", "role_note"])
