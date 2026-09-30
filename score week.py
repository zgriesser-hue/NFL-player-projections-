"""
Score a finished week: projections and sportsbook picks vs. actual results.

Reads the snapshots that receiving_projection_v6.py saved to your Drive folder,
uses for each player the LAST snapshot taken before that player's game kicked off
(no look-ahead), and grades it against nflverse results. Only finished games are
scored, so you can run it mid-week and again after Monday night.

Outputs (printed, and logged to Drive so results accumulate across weeks):
    scorecard_players.csv   one row per player-week (model vs. consensus line vs. actual)
    scorecard_bets.csv      one row per pick (best book per player, EV >= EV_MIN)

Run:  python score_week.py     (auto-picks the latest week that has snapshots)
Output folder: NFL_OUTPUT_DIR if set (GitHub workflow), else Google Drive, else here.
"""
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import nflreadpy as nfl

DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_DIR = DRIVE_ROOT + "/nfl_projections"
SCORE_SEASON = None      # None = latest week that has projection snapshots
SCORE_WEEK = None
EV_MIN = 0.05            # a "pick" = best book per player with model EV at or above this

TEAM_FIX = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "JAC": "JAX",
    "LAR": "LA", "STL": "LA", "SL": "LA", "OAK": "LV", "SD": "LAC", "WSH": "WAS",
}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


# ------------------------------------------------------------ helpers
def resolve_out_dir() -> str:
    """NFL_OUTPUT_DIR (GitHub Actions) > mounted Google Drive (Colab) > current directory."""
    env = os.environ.get("NFL_OUTPUT_DIR")
    if env:
        os.makedirs(env, exist_ok=True)
        return env
    if os.path.isdir(DRIVE_ROOT):
        os.makedirs(DRIVE_DIR, exist_ok=True)
        return DRIVE_DIR
    print("Google Drive is not mounted; looking in the current directory instead.")
    return "."


def fix_team(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_FIX)


def name_key(name) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`\-]", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(p for p in s.split() if p not in _SUFFIXES)


def american_to_decimal(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / -a)


def latest_scored_week(out_dir: str):
    found = set()
    for f in os.listdir(out_dir):
        m = re.match(r"projections_(\d{4})_wk(\d+)_\d{8}_\d{4}\.csv$", f)
        if m:
            found.add((int(m.group(1)), int(m.group(2))))
    if not found:
        raise ValueError(f"No projection snapshots found in {out_dir}")
    return max(found)


def load_snapshots(out_dir: str, season: int, week: int, edges: bool) -> pd.DataFrame:
    pat = re.compile(rf"projections_{season}_wk{week}_{'edges_' if edges else ''}"
                     rf"(\d{{8}}_\d{{4}})\.csv$")
    frames = []
    for f in sorted(os.listdir(out_dir)):
        m = pat.match(f)
        if m:
            df = pd.read_csv(os.path.join(out_dir, f))
            df["snap_time"] = pd.to_datetime(m.group(1), format="%Y%m%d_%H%M", utc=True)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def week_schedule(season: int, week: int):
    """Kickoff time (UTC) per team, and which teams' games are finished."""
    sched = nfl.load_schedules([season]).to_pandas()
    g = sched[(sched["season"] == season) & (sched["week"] == week)
              & (sched["game_type"] == "REG")].copy()
    ko = pd.to_datetime(g["gameday"].astype(str) + " " + g["gametime"].astype(str),
                        errors="coerce").dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    ko = ko.fillna(pd.Timestamp.now(tz="UTC"))
    kick, done_teams = {}, set()
    for row, k in zip(g.itertuples(), ko):
        home, away = fix_team(pd.Series([row.home_team, row.away_team])).tolist()
        kick[home] = kick[away] = k
        if pd.notna(row.home_score) and pd.notna(row.away_score):
            done_teams |= {home, away}
    return kick, done_teams, len(g), int(g["home_score"].notna().sum())


def load_actuals(season: int, week: int) -> pd.DataFrame:
    """Everyone who played that week, with receiving yards (0 if no stat line)."""
    stats = nfl.load_player_stats([season]).to_pandas()
    stats = stats[(stats["season_type"] == "REG") & (stats["season"] == season)]
    names = (stats.sort_values("week").drop_duplicates("player_id", keep="last")
             .set_index("player_id")["player_display_name"])
    wk = stats[stats["week"] == week][["player_id", "receiving_yards"]]

    players = (nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]]
               .dropna().drop_duplicates("pfr_id"))
    snaps = nfl.load_snap_counts([season]).to_pandas()
    snaps = snaps[(snaps["game_type"] == "REG") & (snaps["season"] == season)
                  & (snaps["week"] == week) & (snaps["offense_snaps"] > 0)]
    snaps = (snaps.merge(players, left_on="pfr_player_id", right_on="pfr_id")
             .rename(columns={"gsis_id": "player_id"})[["player_id", "player"]]
             .drop_duplicates("player_id"))

    act = snaps.merge(wk, on="player_id", how="outer")      # a stat line also proves they played
    act["receiving_yards"] = act["receiving_yards"].fillna(0.0)
    act["name"] = act["player_id"].map(names).fillna(act["player"])
    act["name_key"] = act["name"].map(name_key)
    return act.drop_duplicates("name_key")[["name_key", "name", "receiving_yards"]]


def before_kickoff(df: pd.DataFrame, kick: dict) -> pd.DataFrame:
    """Keep only each player's snapshot(s) taken before their game started; latest one wins."""
    df = df.copy()
    df["kickoff"] = df["team"].map(kick)
    df = df[df["snap_time"] <= df["kickoff"]]
    latest = df.groupby("name_key")["snap_time"].transform("max")
    return df[df["snap_time"] == latest]


# ------------------------------------------------------------ reporting
def summarize(b: pd.DataFrame, label: str) -> None:
    if b.empty:
        print(f"  {label}: no bets")
        return
    n = len(b)
    graded = b[b["result"] != "push"]
    hit = (graded["result"] == "win").mean() if len(graded) else float("nan")
    be = (1 / american_to_decimal(graded["price"])).mean() if len(graded) else float("nan")
    ci = 1.96 * b["profit"].std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    ci_txt = f"±{ci:.1%}" if np.isfinite(ci) else "n/a"
    print(f"  {label}: {n} bets | hit {hit:.1%} vs break-even {be:.1%} "
          f"| ROI {b['profit'].mean():+.1%} (95% CI {ci_txt})")


def score_week(season=None, week=None):
    out_dir = resolve_out_dir()
    if season is None or week is None:
        try:
            season, week = latest_scored_week(out_dir)
        except ValueError as e:        # no snapshots yet: nothing to score
            print(e)
            return
    print(f"Scoring {season} week {week}\n")

    kick, done_teams, n_games, n_done = week_schedule(season, week)
    print(f"{n_done}/{n_games} games finished (only finished games are scored)")
    if n_done == 0:
        print("Nothing to score yet.")
        return

    act = load_actuals(season, week)
    proj = load_snapshots(out_dir, season, week, edges=False)
    if proj.empty:
        raise ValueError("No projection snapshots for that week.")
    proj["name_key"] = proj["player_display_name"].map(name_key)
    proj = before_kickoff(proj, kick)
    proj = proj[proj["team"].isin(done_teams)].drop_duplicates("name_key")

    edges = load_snapshots(out_dir, season, week, edges=True)
    if len(edges):
        edges = before_kickoff(edges, kick)
        edges = edges[edges["team"].isin(done_teams)].copy()
        consensus = edges.groupby("name_key")["line"].median()
    else:
        consensus = pd.Series(dtype=float)
    proj["consensus_line"] = proj["name_key"].map(consensus)

    pl = proj.merge(act[["name_key", "receiving_yards"]], on="name_key", how="left")
    pl["played"] = pl["receiving_yards"].notna()
    played = pl[pl["played"]].copy()
    dnp = pl[~pl["played"]]
    print(f"Projected players in finished games: {len(pl)} | played: {len(played)} "
          f"| did not play (props void): {len(dnp)}")
    if len(dnp):
        print("  Sample of non-players (check for name mismatches): "
              f"{sorted(dnp['player_display_name'])[:12]}")
    if played.empty:
        print("No projected players matched a played result.")
        return

    # ---------- projection accuracy
    both = played.dropna(subset=["consensus_line"])
    print(f"\nProjection accuracy (yards) on {len(both)} players with a book line:")
    print(f"  {'':<14}{'MAE':>7}{'RMSE':>8}{'bias':>8}   (bias = actual - prediction)")
    for label, col in (("model median", "median"), ("model mean", "mean"),
                       ("book line", "consensus_line")):
        if both.empty:
            break
        e = both["receiving_yards"] - both[col]
        print(f"  {label:<14}{e.abs().mean():>7.2f}{np.sqrt((e ** 2).mean()):>8.2f}{e.mean():>8.2f}")
    y = played["receiving_yards"]
    print(f"\nModel ranges on all {len(played)} players who played (expected: ≥50% and ≥80%, "
          f"zeros count as inside):")
    print(f"  inside q25-q75: {((y >= played['q25']) & (y <= played['q75'])).mean():.1%} "
          f"| inside q10-q90: {((y >= played['q10']) & (y <= played['q90'])).mean():.1%}")
    print(f"  zero-yard games: actual {(y == 0).mean():.1%} | predicted {played['p_zero'].mean():.1%}")

    players_log = played.assign(season=season, week=week, actual=played["receiving_yards"]).rename(
        columns={"player_display_name": "player", "median": "model_median",
                 "mean": "model_mean"})[
        ["season", "week", "player", "team", "position", "model_median", "model_mean",
         "consensus_line", "actual"]]

    # ---------- betting evaluation
    bets_log = pd.DataFrame()
    if edges.empty:
        print("\nNo sportsbook line snapshots for this week; skipping bet grading.")
    else:
        ef = edges.merge(act[["name_key", "receiving_yards"]], on="name_key", how="left").rename(
            columns={"player": "book_player"})
        played_lines = ef.dropna(subset=["receiving_yards"])

        # probability accuracy: one line per player (closest to that player's consensus)
        dev = (played_lines["line"] - played_lines.groupby("name_key")["line"].transform("median")).abs()
        one = played_lines.assign(_dev=dev).sort_values(["name_key", "_dev"]).drop_duplicates("name_key")
        one = one[one["receiving_yards"] != one["line"]]
        over = (one["receiving_yards"] > one["line"]).astype(float)
        if len(one):
            print(f"\nProbability accuracy on {len(one)} player lines (Brier score, lower is better):")
            print(f"  model {((one['model_p_over'] - over) ** 2).mean():.4f} "
                  f"| book no-vig {((one['fair_p_over'] - over) ** 2).mean():.4f}")
            bucket = pd.cut(one["model_p_over"], [0, .3, .4, .5, .6, .7, 1.0])
            cal = pd.DataFrame({"pred": one["model_p_over"], "over_rate": over, "b": bucket}).groupby(
                "b", observed=True).agg(n=("pred", "size"), pred=("pred", "mean"), actual=("over_rate", "mean"))
            print("  Model P(over) vs how often overs actually hit:")
            print(cal.round(2).to_string())

        picks_all = ef.sort_values("best_ev", ascending=False).drop_duplicates("name_key")
        picks_all = picks_all[picks_all["best_ev"] >= EV_MIN]
        void = picks_all["receiving_yards"].isna()
        picks = picks_all[~void].copy()
        print(f"\nPicks (best book per player, model EV ≥ {EV_MIN:.2f}): {len(picks_all)} "
              f"| voided (did not play): {int(void.sum())} | graded: {len(picks)}")
        if not picks.empty:
            yv, ln = picks["receiving_yards"].to_numpy(), picks["line"].to_numpy()
            side_over = (picks["best_side"] == "over").to_numpy()
            push = yv == ln
            win = np.where(side_over, yv > ln, yv < ln)
            dec = american_to_decimal(picks["best_price"])
            picks["result"] = np.where(push, "push", np.where(win, "win", "loss"))
            picks["profit"] = np.where(push, 0.0, np.where(win, dec - 1, -1.0))
            bets_log = picks.assign(season=season, week=week).rename(columns={
                "model_player": "player", "best_side": "side", "best_price": "price",
                "best_ev": "ev", "receiving_yards": "actual"})[
                ["season", "week", "player", "team", "opp", "book", "side", "line", "price",
                 "model_p", "fair_p", "ev", "actual", "result", "profit"]]
            print("Results this week:")
            summarize(bets_log, "all picks")
            for side, sub in bets_log.groupby("side"):
                summarize(sub, f"{side}s")
            tier = pd.cut(bets_log["ev"], [EV_MIN, 0.10, 0.20, 10], right=False,
                          labels=[f"EV {EV_MIN:.2f}-0.10", "EV 0.10-0.20", "EV 0.20+"])
            for name, sub in bets_log.groupby(tier, observed=True):
                summarize(sub, str(name))

    # ---------- append to running logs (re-running a week replaces that week)
    for name, new in (("scorecard_players.csv", players_log), ("scorecard_bets.csv", bets_log)):
        if new.empty:
            continue
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            old = pd.read_csv(path)
            old = old[~((old["season"] == season) & (old["week"] == week))]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(path, index=False)

    bpath = os.path.join(out_dir, "scorecard_bets.csv")
    if os.path.exists(bpath):
        allb = pd.read_csv(bpath)
        print(f"\nCumulative record ({allb[['season', 'week']].drop_duplicates().shape[0]} weeks):")
        summarize(allb, "all picks")
        print("  Small samples are noisy: with under ~100 bets the confidence interval "
              "usually still includes break-even.")
    ppath = os.path.join(out_dir, "scorecard_players.csv")
    if os.path.exists(ppath):
        allp = pd.read_csv(ppath).dropna(subset=["consensus_line"])
        if len(allp):
            m_err = (allp["actual"] - allp["model_median"]).abs().mean()
            l_err = (allp["actual"] - allp["consensus_line"]).abs().mean()
            print(f"  Cumulative MAE on {len(allp)} lined players: model median {m_err:.2f} "
                  f"| book line {l_err:.2f}")
    print(f"\nLogs saved in {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    score_week(SCORE_SEASON, SCORE_WEEK)
