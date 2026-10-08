"""
Bet tracker for player props: the ledger of bets actually placed, graded automatically.

Ledger: data/bets.csv  (one row per bet; never edit the result columns by hand, `grade` fills them)

    python bets.py add --book draftkings --player "George Pickens" --stat rec_yds --side under \
        --line 65.5 --odds -113 --stake 25 --payout 47.12 --game "TB@DAL" --kickoff 2026-10-09T00:15Z \
        --placed 2026-10-08T22:07:06Z --ticket DK639270940253045494
    python bets.py grade      # grade finished games, compute closing-line value, print the summary
    python bets.py clv        # only fill in closing-line value for bets placed (no grading)
    python bets.py summary    # just print the summary

`add` also copies the model's numbers at the time of the bet (projection, probability, EV) from the saved
line snapshots, so you can later see whether the bets the model liked did better than the ones it did not.

stat is one of: rec_yds, rush_yds, receptions.   odds are American.   payout is the total returned if it wins
(stake included), as printed on the sportsbook receipt; leave it out and it is computed from the odds.
Results: win / loss / push / void (player did not play = stake returned).
"""
import argparse
import os
import re
import unicodedata

import numpy as np
import pandas as pd

LEDGER_NAME = "bets.csv"
SUMMARY_NAME = "bets_summary.txt"
COLS = [
    "ticket", "placed_at", "book", "season", "week", "game", "kickoff", "player", "team", "stat", "side",
    "line", "odds", "stake", "payout",
    # the model at the time of the bet (latest saved snapshot before it, same book if possible)
    "model_mean", "model_median", "model_snap_line", "model_p_side", "model_ev",
    # filled by grade
    "status", "actual", "profit", "close_line", "close_odds", "close_p_side", "clv_line", "clv_prob", "clv_source",
]
EDGE_PREFIX = {"rec_yds": "projections", "rush_yds": "rushing", "receptions": "receptions"}
ACTUAL_COL = {"rec_yds": "receiving_yards", "rush_yds": "rushing_yards", "receptions": "receptions"}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


# ------------------------------------------------------------------ small helpers
def out_dir() -> str:
    env = os.environ.get("NFL_OUTPUT_DIR")
    if env:
        os.makedirs(env, exist_ok=True)
        return env
    drive = "/content/drive/MyDrive/nfl_projections"
    if os.path.isdir(drive):
        return drive
    return "data" if os.path.isdir("data") else "."


def name_key(name) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`\-]", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(p for p in s.split() if p not in _SUFFIXES)


def decimal(odds: float) -> float:
    return 1 + odds / 100 if odds > 0 else 1 + 100 / -odds


def implied(odds: float) -> float:
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def load_ledger(d: str) -> pd.DataFrame:
    path = os.path.join(d, LEDGER_NAME)
    if os.path.exists(path):
        df = pd.read_csv(path)
        for c in COLS:
            if c not in df.columns:
                df[c] = np.nan
        for c in ("clv_source", "status", "team", "game", "kickoff"):
            df[c] = df[c].astype(object)                    # an all-empty column loads as float and would reject text
        return df[COLS]
    return pd.DataFrame(columns=COLS)


def save_ledger(d: str, df: pd.DataFrame) -> None:
    df[COLS].to_csv(os.path.join(d, LEDGER_NAME), index=False)


# ------------------------------------------------------------------ add
def infer_game(d: str, stat: str, player: str):
    """season, week, team, kickoff for a player from the newest saved line snapshots (None if he is not in them)."""
    prefix = EDGE_PREFIX[stat]
    pat = re.compile(rf"{prefix}_(\d{{4}})_wk(\d+)_edges_\d{{8}}_\d{{4}}\.csv$")
    found = sorted({(int(m.group(1)), int(m.group(2))) for f in os.listdir(d) if (m := pat.match(f))})
    for season, week in reversed(found[-2:]):
        key = name_key(player)
        for f in sorted(os.listdir(d), reverse=True):
            m = pat.match(f)
            if not m or (int(m.group(1)), int(m.group(2))) != (season, week):
                continue
            df = pd.read_csv(os.path.join(d, f))
            hit = df[df["name_key"] == key]
            if len(hit):
                r = hit.iloc[0]
                return season, week, r.get("team"), r.get("commence_time")
    return None


def model_at_bet(d: str, stat: str, season, week, player: str, book: str, side: str, line: float, placed_at):
    """The model's numbers from the latest saved snapshot taken before the bet was placed."""
    prefix = EDGE_PREFIX[stat]
    pat = re.compile(rf"{prefix}_{season}_wk{week}_edges_(\d{{8}}_\d{{4}})\.csv$")
    frames = []
    for f in sorted(os.listdir(d)):
        m = pat.match(f)
        if m:
            df = pd.read_csv(os.path.join(d, f))
            df["snap_time"] = pd.to_datetime(m.group(1), format="%Y%m%d_%H%M", utc=True)
            frames.append(df)
    if not frames:
        return {}
    e = pd.concat(frames, ignore_index=True)
    e = e[e["name_key"] == name_key(player)]
    e = e[e["snap_time"] <= placed_at]
    if e.empty:
        return {}
    e = e[e["snap_time"] == e["snap_time"].max()].copy()
    same = e[e["book"] == book]
    e = same if len(same) else e
    e["dist"] = (e["line"] - line).abs()
    r = e.sort_values("dist").iloc[0]
    p_side = r["model_p_over"] if side == "over" else r["model_p_under"]
    ev = r["ev_over"] if side == "over" else r["ev_under"]
    return {"team": r.get("team"), "model_mean": r.get("model_mean"), "model_median": r.get("model_median"),
            "model_snap_line": r["line"], "model_p_side": p_side, "model_ev": ev}


def cmd_add(a) -> None:
    d = out_dir()
    led = load_ledger(d)
    ticket = a.ticket or f"manual-{len(led) + 1}"
    if ticket in set(led["ticket"].astype(str)):
        print(f"Ticket {ticket} is already in the ledger; nothing added.")
        return
    placed = pd.Timestamp(a.placed) if a.placed else pd.Timestamp.now(tz="UTC")
    placed = placed.tz_localize("UTC") if placed.tzinfo is None else placed.tz_convert("UTC")
    payout = a.payout if a.payout else round(a.stake * decimal(a.odds), 2)
    if not (a.season and a.week):
        g = infer_game(d, a.stat, a.player)
        if g:
            a.season, a.week = g[0], g[1]
            a.game = a.game or (f"{g[2]} game" if isinstance(g[2], str) else "")
            a.kickoff = a.kickoff or (g[3] if isinstance(g[3], str) else "")
            print(f"  matched {a.player} to {g[2]}, season {g[0]} week {g[1]}")
        else:
            print(f"  WARNING: {a.player} is not in the latest {a.stat} line snapshots, so season/week are blank "
                  f"and this bet cannot be graded until they are filled in.")
    row = {"ticket": ticket, "placed_at": placed.strftime("%Y-%m-%dT%H:%M:%SZ"), "book": a.book.lower(),
           "season": a.season, "week": a.week, "game": a.game, "kickoff": a.kickoff, "player": a.player,
           "stat": a.stat, "side": a.side.lower(), "line": a.line, "odds": a.odds, "stake": a.stake,
           "payout": payout, "status": "pending"}
    if a.season and a.week:
        row.update(model_at_bet(d, a.stat, a.season, a.week, a.player, a.book.lower(), row["side"], a.line, placed))
    led = pd.concat([led, pd.DataFrame([row])], ignore_index=True)
    save_ledger(d, led)
    print(f"Added: {a.side} {a.line} {a.player} {a.stat} at {a.odds:+d} (${a.stake:.2f} to return ${payout:.2f})")
    if "model_mean" in row:
        print(f"  model at the time: mean {row['model_mean']:.1f}, median {row['model_median']:.1f}, "
              f"{row['model_p_side']:.0%} on this side at {row['model_snap_line']} (EV {row['model_ev']:+.0%})")
    else:
        print("  (no model snapshot found for this player, so the model columns are blank)")


# ------------------------------------------------------------------ grade
def closing_for(d: str, S, bet: pd.Series):
    """Closing line for a placed bet, from the ParlayAPI closing snapshots (nearest to kickoff, at or before it).
    Returns (line, over_price, under_price, source): the bet's own book when it is in the snapshot, otherwise the
    median across books ("consensus"). None if there is no usable closing snapshot yet. A snapshot taken before the
    bet was placed is not a closing line for it, so it is skipped."""
    cl = S.load_closing(d, bet["stat"])
    if cl.empty:
        return None
    cl = cl[cl["name_key"] == name_key(bet["player"])]
    cl = cl[(cl["snap_time"] <= cl["commence"]) & (cl["snap_time"] >= pd.Timestamp(bet["placed_at"]))]
    if cl.empty:
        return None
    cl = cl[cl["snap_time"] == cl["snap_time"].max()]
    mine = cl[cl["book"] == bet["book"]]
    if len(mine):
        r = mine.iloc[0]
        return r["line"], r["over_price"], r["under_price"], "own book"
    return cl["line"].median(), cl["over_price"].median(), cl["under_price"].median(), f"consensus of {len(cl)} books"


def fill_clv(d: str, S, led: pd.DataFrame) -> pd.DataFrame:
    """Closing-line value for placed bets only (separate from the projection scorecards' CLV on the model's picks).

    clv_line: points by which the line moved in our favour (+ = we got the better number).
    clv_prob: closing price minus our price as implied probability, only when the book and line are unchanged
              (+ = the market now rates our side as likelier than the price we paid implied)."""
    for i, bet in led.iterrows():
        if pd.notna(bet["clv_line"]) or pd.notna(bet["clv_prob"]):
            continue
        c = closing_for(d, S, bet)
        if c is None:
            continue
        line, po_price, pu_price, source = c
        over = bet["side"] == "over"
        po, pu = implied(po_price), implied(pu_price)
        close_price = po_price if over else pu_price
        same = source == "own book" and line == bet["line"]
        vals = {"close_line": line, "close_odds": close_price, "close_p_side": (po if over else pu) / (po + pu),
                "clv_line": (line - bet["line"]) if over else (bet["line"] - line),
                "clv_prob": (implied(close_price) - implied(bet["odds"])) if same else np.nan,
                "clv_source": source}
        for k, v in vals.items():
            led.loc[i, k] = v
    return led


def grade(d: str) -> pd.DataFrame:
    import score_week as S                                  # needs nflreadpy; imported here so `add` works anywhere
    led = load_ledger(d)
    pend = led[led["status"].isin(["pending"])]
    for (season, week), grp in pend.groupby(["season", "week"]):
        if pd.isna(season) or pd.isna(week):
            continue
        season, week = int(season), int(week)
        kick, done_teams, n_games, n_done = S.week_schedule(season, week)
        act = None
        for i, bet in grp.iterrows():
            team = bet["team"]
            if isinstance(team, str) and team not in done_teams:
                continue                                   # game not finished yet
            if not isinstance(team, str) and n_done == 0:
                continue
            if act is None:
                act = S.load_actuals(season, week)
            hit = act[act["name_key"] == name_key(bet["player"])]
            if hit.empty:
                led.loc[i, "status"], led.loc[i, "profit"] = "void", 0.0
                continue
            y = float(hit.iloc[0][ACTUAL_COL[bet["stat"]]])
            if y == bet["line"]:
                res, profit = "push", 0.0
            else:
                won = (y > bet["line"]) if bet["side"] == "over" else (y < bet["line"])
                res = "win" if won else "loss"
                profit = round(bet["payout"] - bet["stake"], 2) if won else -float(bet["stake"])
            led.loc[i, "status"], led.loc[i, "actual"], led.loc[i, "profit"] = res, y, profit
    led = fill_clv(d, S, led)
    save_ledger(d, led)
    return led


# ------------------------------------------------------------------ summary
def summary(led: pd.DataFrame) -> str:
    if led.empty:
        return "No bets in the ledger yet."
    out = []
    done = led[led["status"].isin(["win", "loss", "push"])]
    pend = led[led["status"] == "pending"]
    out.append(f"BET TRACKER: {len(led)} bets | {len(done)} graded | {len(pend)} pending | "
               f"{int((led['status'] == 'void').sum())} void")
    if len(done):
        staked, profit = done["stake"].sum(), done["profit"].sum()
        w, l, p = (done["status"] == "win").sum(), (done["status"] == "loss").sum(), (done["status"] == "push").sum()
        out.append(f"  record {w}-{l}-{p} | staked ${staked:,.2f} | profit {profit:+,.2f} | ROI {profit / staked:+.1%}")
        be = (done["odds"].map(implied)).mean()
        out.append(f"  break-even win rate at these prices {be:.1%}; actual {w / max(w + l, 1):.1%}")
        for col in ("stat", "side"):
            for k, g in done.groupby(col):
                out.append(f"    {col} {k}: {int((g['status'] == 'win').sum())}-{int((g['status'] == 'loss').sum())}"
                           f" | profit {g['profit'].sum():+,.2f} | ROI {g['profit'].sum() / g['stake'].sum():+.1%}")
        if done["model_ev"].notna().any():
            liked = done[done["model_ev"] >= 0.05]
            if len(liked) and len(liked) < len(done):
                out.append(f"  bets the model rated EV >= 5%: {len(liked)} | ROI {liked['profit'].sum() / liked['stake'].sum():+.1%}")
        if len(done) < 100:
            out.append("  (with this few bets, results are mostly luck: a +/-20% ROI swing is normal over 50 bets)")
    cl = led[led["clv_line"].notna()]
    out.append("")
    out.append("BET CLV (closing-line value on bets you placed; not the projection scorecards' CLV on the model's picks)")
    if len(cl):
        n = len(cl)
        out.append(f"  {n} of {len(led)} bets have a closing line | average line move in your favour {cl['clv_line'].mean():+.2f} pts"
                   f" | beat the close on {(cl['clv_line'] > 0).sum()}, matched {(cl['clv_line'] == 0).sum()}, lost to it {(cl['clv_line'] < 0).sum()}")
        pc = cl[cl["clv_prob"].notna()]
        if len(pc):
            out.append(f"  price edge where the line was unchanged ({len(pc)} bets): {pc['clv_prob'].mean():+.1%} implied probability")
        for k, g in cl.groupby("side"):
            out.append(f"    {k}s: {len(g)} bets, line move {g['clv_line'].mean():+.2f} pts")
        out.append("  " + "; ".join(f"{r.player} {r.side} {r.line:g} -> closed {r.close_line:g} ({r.clv_line:+.1f})"
                                     + ("" if r.clv_source == "own book" else f" [{r.clv_source}]") for r in cl.itertuples()))
        if n < 50:
            out.append("  (CLV is far less noisy than win/loss, but still wants about 50+ bets before it means much)")
    else:
        out.append("  no closing lines recorded for your bets yet (they come from the closing snapshots shortly before kickoff)")
    if len(pend):
        out.append("  pending: " + "; ".join(f"{r.side} {r.line:g} {r.player} {r.stat} ({r.odds:+.0f})" for r in pend.itertuples()))
    return "\n".join(out)


def cmd_summary(d: str, led: pd.DataFrame) -> None:
    text = summary(led)
    print("\n" + text)
    with open(os.path.join(d, SUMMARY_NAME), "w") as f:
        f.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--book", required=True)
    a.add_argument("--player", required=True)
    a.add_argument("--stat", required=True, choices=list(EDGE_PREFIX))
    a.add_argument("--side", required=True, choices=["over", "under"])
    a.add_argument("--line", type=float, required=True)
    a.add_argument("--odds", type=int, required=True)
    a.add_argument("--stake", type=float, required=True)
    a.add_argument("--payout", type=float)
    a.add_argument("--game", default="")
    a.add_argument("--kickoff", default="")
    a.add_argument("--placed")
    a.add_argument("--ticket")
    a.add_argument("--season", type=int)
    a.add_argument("--week", type=int)
    sub.add_parser("grade")
    sub.add_parser("clv")
    sub.add_parser("summary")
    args = ap.parse_args()
    d = out_dir()
    if args.cmd == "add":
        cmd_add(args)
        cmd_summary(d, load_ledger(d))
    elif args.cmd == "grade":
        cmd_summary(d, grade(d))
    elif args.cmd == "clv":
        import score_week as S
        led = fill_clv(d, S, load_ledger(d))
        save_ledger(d, led)
        cmd_summary(d, led)
    else:
        cmd_summary(d, load_ledger(d))


if __name__ == "__main__":
    main()
