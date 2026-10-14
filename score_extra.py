"""
Score the rushing-yards and receptions models after games finish: projections vs the book line vs what
happened, graded picks, and closing line value. Same idea as score_week.py (receiving yards).

Reads the {rushing,receptions}_{season}_wk{week}_edges_*.csv snapshots saved by rushing.py / projections.py,
uses each player's LAST snapshot before kickoff (no look-ahead), grades against nflverse results, and
appends to scorecard_rush_*.csv / scorecard_receptions_*.csv. CLV uses closing_*.csv snapshots when present.

Run:  python score_extra.py
"""
import os
import re

import numpy as np
import pandas as pd

import score_week as S

STATS = {
    "rush_yds": dict(label="Rushing yards", prefix="rushing", actual="rushing_yards", log="rush"),
    "receptions": dict(label="Receptions", prefix="receptions", actual="receptions", log="receptions"),
}
RECEPTIONS_EDGES = False      # receptions betting edges are switched off (see projections.RECEPTIONS_EDGES)
if not RECEPTIONS_EDGES:
    STATS.pop("receptions")


def latest_week(out_dir: str, prefix: str):
    found = set()
    for f in os.listdir(out_dir):
        m = re.match(rf"{prefix}_(\d{{4}})_wk(\d+)_edges_\d{{8}}_\d{{4}}\.csv$", f)
        if m:
            found.add((int(m.group(1)), int(m.group(2))))
    return max(found) if found else None


def load_edges(out_dir: str, prefix: str, season: int, week: int) -> pd.DataFrame:
    pat = re.compile(rf"{prefix}_{season}_wk{week}_edges_(\d{{8}}_\d{{4}})\.csv$")
    frames = []
    for f in sorted(os.listdir(out_dir)):
        m = pat.match(f)
        if m:
            df = pd.read_csv(os.path.join(out_dir, f))
            df["snap_time"] = pd.to_datetime(m.group(1), format="%Y%m%d_%H%M", utc=True)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out) and "line" in out.columns:           # old snapshots may hold passing-yard lines under rush
        cap = {"rushing": 150, "receptions": 20}.get(prefix)
        if cap:
            out = out[out["line"] < cap]
    return out


def score_stat(stat: str, cfg: dict, out_dir: str) -> None:
    label = cfg["label"]
    sw = latest_week(out_dir, cfg["prefix"])
    if sw is None:
        print(f"\n== {label}: no line snapshots yet ==")
        return
    season, week = sw
    print(f"\n==== {label}: {season} week {week} ====")
    kick, done_teams, n_games, n_done = S.week_schedule(season, week)
    print(f"{n_done}/{n_games} games finished (only finished games are scored)")
    if n_done == 0:
        return

    act = S.load_actuals(season, week)
    edges_all = load_edges(out_dir, cfg["prefix"], season, week)
    edges = S.before_kickoff(edges_all.copy(), kick)
    edges = edges[edges["team"].isin(done_teams)].copy()
    if edges.empty:
        print("No snapshots taken before kickoff for finished games.")
        return
    A = cfg["actual"]
    ef = edges.merge(act[["name_key", A]], on="name_key", how="left")
    ef = ef.rename(columns={A: "actual"})

    # ---- projection accuracy vs the book line (one row per player who played)
    ef["consensus_line"] = ef.groupby("name_key")["line"].transform("median")
    pl = ef.dropna(subset=["actual"]).drop_duplicates("name_key")
    print(f"Players with a line who played: {len(pl)} (did not play, props void: "
          f"{ef.drop_duplicates('name_key')['actual'].isna().sum()})")
    if len(pl):
        print(f"  {'':<14}{'MAE':>7}{'RMSE':>8}{'bias':>8}   (bias = actual - prediction)")
        for name, col in (("model median", "model_median"), ("model mean", "model_mean"),
                          ("book line", "consensus_line")):
            e = pl["actual"] - pl[col]
            print(f"  {name:<14}{e.abs().mean():>7.2f}{np.sqrt((e ** 2).mean()):>8.2f}{e.mean():>8.2f}")

    # ---- probability accuracy (Brier), one line per player closest to consensus
    pls = ef.dropna(subset=["actual"])
    dev = (pls["line"] - pls["consensus_line"]).abs()
    one = pls.assign(_d=dev).sort_values(["name_key", "_d"]).drop_duplicates("name_key")
    one = one[one["actual"] != one["line"]]
    if len(one):
        over = (one["actual"] > one["line"]).astype(float)
        print(f"Brier score on {len(one)} lines (lower is better): model "
              f"{((one['model_p_over'] - over) ** 2).mean():.4f} | book no-vig "
              f"{((one['fair_p_over'] - over) ** 2).mean():.4f}")

    # ---- picks
    ef = ef.sort_values("best_ev", ascending=False)
    bets_all = pd.DataFrame()
    for kind, subset in (("confirmed picks (mean and median agree)", ef[ef["agree"]]),
                         ("all picks", ef)):
        picks_all = subset.drop_duplicates("name_key")
        picks_all = picks_all[picks_all["best_ev"] >= S.EV_MIN]
        picks = picks_all[picks_all["actual"].notna()].copy()
        print(f"\n{kind}: {len(picks_all)} (EV >= {S.EV_MIN:.2f}) | voided: "
              f"{int(picks_all['actual'].isna().sum())} | graded: {len(picks)}")
        if picks.empty:
            continue
        yv, ln = picks["actual"].to_numpy(), picks["line"].to_numpy()
        side_over = (picks["best_side"] == "over").to_numpy()
        push = yv == ln
        win = np.where(side_over, yv > ln, yv < ln)
        dec = S.american_to_decimal(picks["best_price"])
        picks["result"] = np.where(push, "push", np.where(win, "win", "loss"))
        picks["profit"] = np.where(push, 0.0, np.where(win, dec - 1, -1.0))
        b = picks.drop(columns=["player"], errors="ignore").assign(season=season, week=week, confirmed=kind.startswith("confirmed")).rename(
            columns={"model_player": "player", "best_side": "side", "best_price": "price", "best_ev": "ev"})[
            ["season", "week", "player", "team", "opp", "book", "model_version", "side", "line", "price",
             "model_p", "fair_p", "ev", "actual", "result", "profit", "confirmed"]]
        S.summarize(b, "  all")
        for side, sub in b.groupby("side"):
            S.summarize(sub, f"  {side}s")
        if kind.startswith("confirmed"):
            bets_all = b

    # ---- logs (re-running a week replaces that week)
    def append(name, new):
        if new.empty:
            return
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            old = pd.read_csv(path)
            old = old.loc[:, ~old.columns.str.endswith(".1")]
            old = old[~((old["season"] == season) & (old["week"] == week))]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(path, index=False)

    append(f"scorecard_{cfg['log']}_bets.csv", bets_all)
    append(f"scorecard_{cfg['log']}_players.csv", pl.drop(columns=["player"], errors="ignore").assign(season=season, week=week).rename(
        columns={"model_player": "player"})[
        ["season", "week", "player", "team", "model_version", "model_median", "model_mean",
         "consensus_line", "actual"]])
    bpath = os.path.join(out_dir, f"scorecard_{cfg['log']}_bets.csv")
    if os.path.exists(bpath):
        allb = pd.read_csv(bpath)
        print(f"\nCumulative confirmed picks ({allb[['season', 'week']].drop_duplicates().shape[0]} weeks):")
        S.summarize(allb, "  all")
        print("  With under ~100 picks the confidence interval will usually still include break-even.")

    # ---- closing line value
    S.clv_report(edges_all, kick, season, week, out_dir, stat=stat)


def main():
    out_dir = S.resolve_out_dir()
    for stat, cfg in STATS.items():
        try:
            score_stat(stat, cfg, out_dir)
        except Exception as e:                      # one stat failing must not hide the other
            print(f"\n{cfg['label']}: scoring failed ({type(e).__name__}: {e})")
    print(f"\nLogs saved in {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
