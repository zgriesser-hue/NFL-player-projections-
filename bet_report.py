"""Weekly bet-results CSVs from the placed-bet ledger (data/bets.csv).

Writes to data/bet_reports/:
  Bet_Results_Summary.csv  one row per week + cumulative columns + an ALL WEEKS total row
  Bet_Results_Detail.csv   every bet with result, profit and closing-line value
Voided bets (player didn't play) are listed but not counted. Pending bets are listed and not counted.
"""
import os

import numpy as np
import pandas as pd

DATA = os.environ.get("OUT_DIR", "data")
OUT = os.path.join(DATA, "bet_reports")
MARKET = {"rec_yds": "Receiving Yards", "rush_yds": "Rushing Yards", "receptions": "Receptions"}
RESULT = {"win": "WIN", "loss": "LOSS", "push": "PUSH", "pending": "Pending", "void": "Void"}


def fnum(x, d=2):
    return np.nan if pd.isna(x) else round(float(x), d)


def detail(led: pd.DataFrame) -> pd.DataFrame:
    d = led.copy()
    alt = d["clv_source"].astype(str).str.startswith("alt")
    out = pd.DataFrame({
        "Season": d["season"], "Week": d["week"], "Game": d["game"], "Player": d["player"], "Team": d["team"],
        "Book": d["book"], "Market": d["stat"].map(MARKET).fillna(d["stat"]) + np.where(alt, " (Alt Line)", ""),
        "Bet": d["side"].str.upper() + " " + d["line"].map(lambda x: f"{x:g}"),
        "Odds": d["odds"].map(lambda x: f"{x:+.0f}"), "Stake $": d["stake"],
        "To Win $": (d["payout"] - d["stake"]).round(2), "Result": d["status"].map(RESULT),
        "Actual Result": d["actual"], "Profit $": d["profit"],
        "Our Chance at Bet Time (%)": (d["model_p_side"] * 100).round(0),
        "Our EV at Bet Time (%)": (d["model_ev"] * 100).round(0),
        "CLV: Line Move vs Close (pts, + is good)": d["clv_line"].round(1),
        "Closing Line": d["close_line"], "Closing Odds": d["close_odds"],
        "CLV: Price vs Close (% pts, + is good)": (d["clv_prob"] * 100).round(1),
        "CLV Source": d["clv_source"].where(~alt, "n/a (alt line)").fillna(""),
        "Placed (UTC)": d["placed_at"], "Ticket": d["ticket"]})
    return out.sort_values(["Season", "Week", "Placed (UTC)"]).reset_index(drop=True)


def summarize(g: pd.DataFrame) -> dict:
    done = g[g["status"].isin(["win", "loss", "push"])]
    w, l, p = [int((done["status"] == s).sum()) for s in ("win", "loss", "push")]
    staked, profit = float(done["stake"].sum()), float(done["profit"].sum())
    cl = g[g["clv_line"].notna()]
    exp = float((done["stake"] * done["model_ev"].fillna(0)).sum())
    return {"Bets": len(g), "Graded": len(done), "Pending": int((g["status"] == "pending").sum()),
            "Void": int((g["status"] == "void").sum()), "W": w, "L": l, "P": p, "Staked $": staked,
            "Profit $": profit, "Model-Expected Profit $": exp,
            "Bets With CLV": len(cl), "CLV Avg Line Move (pts)": cl["clv_line"].mean() if len(cl) else np.nan,
            "Beat Closing Line": int((cl["clv_line"] > 0).sum()), "Matched": int((cl["clv_line"] == 0).sum()),
            "Lost to Closing Line": int((cl["clv_line"] < 0).sum())}


def fmt_row(label: str, s: dict, cum: dict) -> dict:
    wl = lambda x: f"{x['W']}-{x['L']}" + (f"-{x['P']}" if x["P"] else "")
    return {
        "Week": label, "Bets Placed": s["Bets"], "Graded": s["Graded"], "Pending": s["Pending"], "Void": s["Void"],
        "Record (W-L-P)": wl(s), "Win %": round(100 * s["W"] / max(s["W"] + s["L"], 1), 1) if s["Graded"] else np.nan,
        "Staked $": round(s["Staked $"], 2), "Profit $": round(s["Profit $"], 2),
        "ROI %": round(100 * s["Profit $"] / s["Staked $"], 1) if s["Staked $"] else np.nan,
        "Model-Expected Profit $": round(s["Model-Expected Profit $"], 2),
        "CUMULATIVE Record": wl(cum), "CUMULATIVE Staked $": round(cum["Staked $"], 2),
        "CUMULATIVE Profit $": round(cum["Profit $"], 2),
        "CUMULATIVE ROI %": round(100 * cum["Profit $"] / cum["Staked $"], 1) if cum["Staked $"] else np.nan,
        "CLV Bets Measured": s["Bets With CLV"],
        "CLV Avg Line Move (pts)": fnum(s["CLV Avg Line Move (pts)"], 2),
        "Beat / Matched / Lost vs Close": f"{s['Beat Closing Line']} / {s['Matched']} / {s['Lost to Closing Line']}"
        if s["Bets With CLV"] else ""}


def summary(led: pd.DataFrame) -> pd.DataFrame:
    rows, run = [], led.iloc[0:0]
    for (season, week), g in led.sort_values(["season", "week"]).groupby(["season", "week"], dropna=False):
        run = pd.concat([run, g])
        label = f"{int(season)} Wk {int(week)}" if pd.notna(season) and pd.notna(week) else "No week"
        rows.append(fmt_row(label, summarize(g), summarize(run)))
    rows.append(fmt_row("ALL WEEKS", summarize(led), summarize(led)))
    return pd.DataFrame(rows)


def main():
    path = os.path.join(DATA, "bets.csv")
    if not os.path.exists(path):
        print("No bets.csv; nothing to report.")
        return
    led = pd.read_csv(path, dtype={"game": object, "kickoff": object, "clv_source": object, "team": object})
    if led.empty:
        print("Ledger is empty; nothing to report.")
        return
    os.makedirs(OUT, exist_ok=True)
    s, d = summary(led), detail(led)
    s.to_csv(os.path.join(OUT, "Bet_Results_Summary.csv"), index=False)
    d.to_csv(os.path.join(OUT, "Bet_Results_Detail.csv"), index=False)
    t = s.iloc[-1]
    print(f"Bet report: {len(d)} bets | record {t['Record (W-L-P)']} | profit ${t['Profit $']:+,.2f} | "
          f"ROI {t['ROI %']}% | CLV bets {t['CLV Bets Measured']}")


if __name__ == "__main__":
    main()
