"""Builds the post-game results email for the most recently graded week.

Reads the scorecard logs written by score_week.py / score_extra.py and the placed-bet ledger (bets.csv)
and writes data/results_email.html + data/results_subject.txt. Only the latest graded week is shown.
"""
import glob
import html
import os
import sys

import numpy as np
import pandas as pd

OUT = os.environ.get("OUT_DIR", "data")
esc = html.escape
TD = "padding:4px 8px;white-space:nowrap;border-bottom:1px solid #e5e5e5"
TH = "padding:6px 8px;text-align:left;background:#f3f3f3;white-space:nowrap"
LABELS = {"bets": "Receiving yards", "rushing": "Rushing yards", "receptions": "Receptions"}


def table(cols, rows):
    return ('<table style="border-collapse:collapse;font-size:13px"><tr>'
            + "".join(f'<th style="{TH}">{c}</th>' for c in cols) + "</tr>"
            + "".join("<tr>" + "".join(f'<td style="{TD}">{c}</td>' for c in r) + "</tr>" for r in rows) + "</table>")


def read(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def latest_week():
    best = None
    for p in glob.glob(os.path.join(OUT, "scorecard*bets.csv")):
        d = read(p)
        if len(d) and {"season", "week"} <= set(d.columns):
            k = (int(d["season"].max()), int(d[d["season"] == d["season"].max()]["week"].max()))
            best = k if best is None or k > best else best
    return best


def section(path, season, week):
    name = os.path.basename(path)[len("scorecard"):-len("bets.csv")].strip("_") or "bets"
    label = LABELS.get(name if name != "" else "bets", name.title())
    d = read(path)
    d = d[(d["season"] == season) & (d["week"] == week)]
    if d.empty:
        return ""
    g = d[d["result"].isin(["win", "loss", "push"])]
    w, l, p = [(g["result"] == r).sum() for r in ("win", "loss", "push")]
    units = g["profit"].sum()
    roi = units / max(len(g) - p, 1)
    html_ = (f"<h3 style='margin:18px 0 4px'>{label}: model leans, graded</h3>"
             f"<p style='margin:0 0 6px'><b>{w}-{l}" + (f"-{p}" if p else "") +
             f"</b>, {units:+.1f} units flat-staked ({roi:+.0%} per bet)</p>")
    rows = []
    for r in g.sort_values("ev", ascending=False).head(30).itertuples():
        col = "#1a7f37" if r.result == "win" else ("#b42318" if r.result == "loss" else "#555")
        rows.append([esc(str(r.player)), f"{r.side.title()} {r.line:g} ({r.price:+.0f})",
                     f"{r.actual:g}", f'<b style="color:{col}">{r.result}</b>', f"{r.profit:+.2f}"])
    html_ += table(["Player", "Lean", "Actual", "Result", "Units"], rows)
    if len(g) > 30:
        html_ += f"<p style='margin:4px 0 0;color:#777;font-size:12px'>Top 30 of {len(g)} by edge shown.</p>"
    pl = read(path.replace("bets.csv", "players.csv"))
    if len(pl) and {"model_median", "consensus_line", "actual"} <= set(pl.columns):
        pl = pl[(pl["season"] == season) & (pl["week"] == week)].dropna(subset=["actual", "model_median"])
        pl = pl[pl["consensus_line"].notna()]
        if len(pl) >= 10:
            m_err = (pl["model_median"] - pl["actual"]).abs().mean()
            l_err = (pl["consensus_line"] - pl["actual"]).abs().mean()
            bias = (pl["model_mean"] - pl["actual"]).mean() if "model_mean" in pl else float("nan")
            html_ += (f"<p style='margin:6px 0 0;color:#555;font-size:12px'>Accuracy on {len(pl)} players: model "
                      f"median off by {m_err:.1f} on average vs {l_err:.1f} for the book line; "
                      f"model average runs {bias:+.1f} vs actual.</p>")
    return html_


def placed(season, week):
    d = read(os.path.join(OUT, "bets.csv"))
    if d.empty:
        return ""
    d = d[(d["season"] == season) & (d["week"] == week)]
    if d.empty:
        return ""
    rows = []
    for r in d.itertuples():
        res = str(r.status)
        col = "#1a7f37" if res == "win" else ("#b42318" if res == "loss" else "#555")
        prof = f"{r.profit:+.2f}" if pd.notna(r.profit) else "&ndash;"
        clv = f"{r.clv_line:+.1f} pts" if pd.notna(r.clv_line) else "&ndash;"
        rows.append([esc(str(r.player)), f"{str(r.side).title()} {r.line:g} {str(r.stat).replace('_', ' ')} ({r.odds:+.0f})",
                     f"${r.stake:g}", "&ndash;" if pd.isna(r.actual) else f"{r.actual:g}",
                     f'<b style="color:{col}">{res}</b>', prof, clv])
    g = d[d["status"].isin(["win", "loss", "push"])]
    tot = f"{g['profit'].sum():+.2f}" if len(g) else "pending"
    return ("<h3 style='margin:18px 0 4px'>Your placed bets</h3>"
            f"<p style='margin:0 0 6px'>Net: <b>{tot}</b> (dollars). CLV = how many points better your number was "
            "than the close (positive is good).</p>"
            + table(["Player", "Bet", "Stake", "Actual", "Result", "Profit $", "CLV"], rows))


CLV_FILES = [("scorecard_clv.csv", "Receiving yards"), ("scorecard_clv_receptions.csv", "Receptions"),
             ("scorecard_clv_rush_yds.csv", "Rushing yards")]


def clv_block(season, week):
    """Did the lines move toward our picks? Pick line (when first flagged) vs closing line."""
    rows, movers = [], []
    for fn, label in CLV_FILES:
        d = read(os.path.join(OUT, fn))
        if d.empty or "clv_line" not in d.columns:
            continue
        d = d[(d["season"] == season) & (d["week"] == week)]
        if d.empty:
            continue
        m = d.dropna(subset=["clv_line"])
        c = d.dropna(subset=["clv_line_consensus"])
        r2 = lambda x: round(float(x), 2)
        rows.append([label, len(d), f"{int((m.clv_line.round(2) > 0).sum())} / {int((m.clv_line.round(2) == 0).sum())} / "
                     f"{int((m.clv_line.round(2) < 0).sum())}", f"{r2(m.clv_line.mean()):+.2f}" if len(m) else "&ndash;",
                     f"{int((c.clv_line_consensus.round(2) > 0).sum())} / {int((c.clv_line_consensus.round(2) < 0).sum())}",
                     f"{r2(c.clv_line_consensus.mean()):+.2f}" if len(c) else "&ndash;"])
        movers.append(d.assign(mkt=label))
    if not rows:
        return ""
    allm = pd.concat(movers).dropna(subset=["clv_line_consensus"]).sort_values("clv_line_consensus", ascending=False)
    top = [[esc(str(r.player)), r.mkt, f"{r.side.title()} {r.pick_line:g}",
            f"{r.close_consensus:g}", f"{r.clv_line_consensus:+.1f}"] for r in allm.head(6).itertuples()]
    return ("<h3 style='margin:18px 0 4px'>Did lines move toward our picks?</h3>"
            "<p style='margin:0 0 6px;color:#555;font-size:12px'>Our pick's line when first flagged vs the closing line "
            "(points; + means the line moved our way). <b>Same book</b> is the book we flagged; <b>consensus</b> is the median "
            "close across books, which also reflects that book being slow. One week is a small sample: look for the "
            "average staying positive over several weeks.</p>"
            + table(["Market", "Picks", "Same book: toward / same / against", "Same book avg",
                     "Consensus: toward / against", "Consensus avg"], rows)
            + "<p style='margin:8px 0 4px'><b>Biggest moves our way</b></p>"
            + table(["Player", "Market", "Our pick", "Closing consensus line", "Move (pts)"], top))


def main():
    k = latest_week()
    if not k:
        print("No graded week found; no results email.")
        return 0
    season, week = k
    parts = [section(p, season, week) for p in sorted(glob.glob(os.path.join(OUT, "scorecard*bets.csv")))]
    body = (f"<h2 style='margin:0 0 8px'>{season} Week {week} results</h2>" + placed(season, week) + clv_block(season, week) + "".join(parts))
    with open(os.path.join(OUT, "results_email.html"), "w", encoding="utf-8") as f:
        f.write(f'<div style="font-family:-apple-system,Segoe UI,Arial,sans-serif;font-size:14px">{body}</div>')
    with open(os.path.join(OUT, "results_subject.txt"), "w", encoding="utf-8") as f:
        f.write(f"NFL props: {season} Week {week} results")
    print(f"Results email written for {season} week {week}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
