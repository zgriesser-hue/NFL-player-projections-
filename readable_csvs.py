"""Writes clean, human-readable copies of the model outputs to data/readable/.

The raw files (projections_*, *_edges_*, ...) keep their internal column names because scoring and the bet
tracker read them. These copies are what gets emailed: games listed as "AWAY @ HOME" in kickoff order, plain-English
headers, no ids/timestamps, one row per player (the best book) for edges.

Run after a projections run:  python readable_csvs.py   (season/week come from the newest projections file)
"""
import glob
import os
import re

import numpy as np
import pandas as pd

DATA = os.environ.get("OUT_DIR", "data")
OUT = os.path.join(DATA, "readable")
ALIAS = {"LA": "LAR", "WSH": "WAS", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LAR"}


def norm(t):
    t = str(t)
    return ALIAS.get(t, t)


def load_games(season: int, week: int) -> dict:
    """{frozenset({team, opp}): (label 'AWAY @ HOME', kickoff datetime, 'Thu 8:15 PM')}"""
    try:
        import nflreadpy as nfl
        s = nfl.load_schedules([season]).to_pandas()
        s = s[(s["week"] == week) & (s["game_type"] == "REG")]
    except Exception as e:
        print(f"Schedule unavailable ({e}); games shown as 'A vs B'")
        return {}
    games = {}
    for r in s.itertuples():
        try:
            ko = pd.Timestamp(f"{r.gameday} {r.gametime}")
            when = ko.strftime("%a %-I:%M %p")
        except Exception:
            ko, when = pd.Timestamp.max, ""
        games[frozenset({norm(r.home_team), norm(r.away_team)})] = (
            f"{norm(r.away_team)} @ {norm(r.home_team)}", ko, when)
    return games


def games_from_edges(tag: str) -> dict:
    """Fallback when the schedule can't be loaded: kickoff times from the book lines (no home/away)."""
    games = {}
    for f in glob.glob(os.path.join(DATA, f"*_{tag}_edges_latest.csv")):
        d = pd.read_csv(f, usecols=lambda c: c in ("team", "opp", "commence_time"))
        if not {"team", "opp", "commence_time"} <= set(d.columns):
            continue
        for r in d.drop_duplicates(["team", "opp"]).itertuples():
            ko = pd.Timestamp(r.commence_time).tz_convert("America/New_York").tz_localize(None)
            a, b = sorted([norm(r.team), norm(r.opp)])
            games[frozenset({a, b})] = (f"{a} vs {b}", ko, ko.strftime("%a %-I:%M %p"))
    return games


def add_game(df: pd.DataFrame, games: dict) -> pd.DataFrame:
    def look(r):
        g = games.get(frozenset({norm(r["team"]), norm(r["opp"])}))
        if g:
            return g
        a, b = sorted([str(r["team"]), str(r["opp"])])
        return (f"{a} vs {b}", pd.Timestamp.max, "")
    info = df.apply(look, axis=1, result_type="expand")
    df = df.copy()
    df["Game"], df["_ko"], df["Kickoff (ET)"] = info[0], info[1], info[2]
    return df


def finish(df: pd.DataFrame, cols: list, sort_within: str, ascending=False) -> pd.DataFrame:
    df = df.sort_values(["_ko", "Game", sort_within], ascending=[True, True, ascending])
    return df[cols].reset_index(drop=True)


def pct(x, d=0):
    return (pd.to_numeric(x, errors="coerce") * 100).round(d)


def _imp(o):
    o = pd.to_numeric(o, errors="coerce")
    return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


def _hold(d):
    return _imp(d["over_price"]) + _imp(d["under_price"]) - 1


def edges(raw: pd.DataFrame, games: dict, unit: str) -> pd.DataFrame:
    d = raw.copy()
    d["_n_books"] = d.groupby("model_player")["book"].transform("nunique")
    d = d.sort_values("best_ev", ascending=False).drop_duplicates("model_player")     # best book per player
    d = add_game(d, games)
    odds = d["best_price"].round().astype(int)
    d["Bet"] = (d["best_side"].str.upper() + " " + d["line"].map(lambda x: f"{x:g}") + " ("
                + odds.map(lambda x: f"{x:+d}") + ")")
    p = 1 if unit == "rec" else 0
    out = pd.DataFrame({
        "Game": d["Game"], "Kickoff (ET)": d["Kickoff (ET)"], "Player": d["model_player"], "Team": d["team"],
        "Pos": d["position"], "Injury Status": d["status"].fillna(""), "Book": d["book"],
        "BET (Side, Line, Odds)": d["Bet"], "Book Line": d["line"],
        "Our Projection (Median)": d["model_median"].round(p), "Our Average": d["model_mean"].round(1),
        "Chance Bet Wins - Ours, Blended (%)": pct(d["model_p"], 1),
        "Chance Bet Wins - Book No-Vig (%)": pct(d["fair_p"], 1),
        "Book Hold (Vig %)": pct(_hold(d), 1),
        "Edge: EV per $1 (%)": pct(d["best_ev"]),
        "Median & Average Agree": np.where(d["agree"], "Yes", "No"),
        "CHECK NEWS": np.where(d["check_news"], "CHECK NEWS", ""),
        "Chance Bet Wins - Raw Model (%)": pct(np.where(d["best_side"] == "over", d["raw_p_over"], d["raw_p_under"]), 1),
        "Books Compared": d["_n_books"],
        "Notes": d["role_note"].fillna("") if "role_note" in d else "",
        "_ko": d["_ko"],
    })
    out = out.sort_values(["_ko", "Game", "Edge: EV per $1 (%)"], ascending=[True, True, False])
    return out.drop(columns="_ko").reset_index(drop=True)


def yard_proj(raw: pd.DataFrame, games: dict) -> pd.DataFrame:
    d = add_game(raw, games)
    out = pd.DataFrame({
        "Game": d["Game"], "Kickoff (ET)": d["Kickoff (ET)"], "Player": d["player_display_name"], "Team": d["team"],
        "Pos": d["position"], "Injury Status": d["status"].fillna(""), "Practice": d["practice"].fillna(""),
        "Projected Rec Yards (Median)": d["median"].round(0), "Low End (25th pct)": d["q25"].round(0),
        "High End (75th pct)": d["q75"].round(0), "Boom (90th pct)": d["q90"].round(0),
        "Average": d["mean"].round(1), "Chance of 0 Yards (%)": pct(d["p_zero"]),
        "Projected Receptions (Median)": d["rec_projection"], "Expected Targets": d["exp_targets"].round(1),
        "Catch Rate (%)": pct(d["p_catch"]), "Yards per Catch": d["ypr"].round(1),
        "Team Implied Points": d["implied_total"], "Chance Plays (%)": pct(d["p_active"]), "_ko": d["_ko"]})
    out = out.sort_values(["_ko", "Game", "Projected Rec Yards (Median)"], ascending=[True, True, False])
    return out.drop(columns="_ko").reset_index(drop=True)


def rec_proj(raw: pd.DataFrame, games: dict) -> pd.DataFrame:
    d = add_game(raw, games)
    out = pd.DataFrame({
        "Game": d["Game"], "Kickoff (ET)": d["Kickoff (ET)"], "Player": d["player"], "Team": d["team"],
        "Pos": d["position"], "Injury Status": d["status"].fillna(""),
        "Projected Receptions (Median)": d["receptions_median"], "Average": d["receptions_mean"].round(1),
        "Low End (25th pct)": d["receptions_q25"], "High End (75th pct)": d["receptions_q75"],
        "Chance of 0 Catches (%)": pct(d["p_zero_receptions"]), "Expected Targets": d["exp_targets"].round(1),
        "Catch Rate (%)": pct(d["p_catch"]), "Projected Rec Yards (Median)": d["receiving_yards_median"].round(0),
        "_ko": d["_ko"]})
    out = out.sort_values(["_ko", "Game", "Projected Receptions (Median)", "Average"],
                          ascending=[True, True, False, False])
    return out.drop(columns="_ko").reset_index(drop=True)


def rush_proj(raw: pd.DataFrame, games: dict) -> pd.DataFrame:
    d = add_game(raw, games)
    out = pd.DataFrame({
        "Game": d["Game"], "Kickoff (ET)": d["Kickoff (ET)"], "Player": d["player_display_name"], "Team": d["team"],
        "Pos": d["position"], "Injury Status": d["status"].fillna(""), "Practice": d["practice"].fillna(""),
        "Projected Rush Yards (Median)": d["median"].round(0), "Low End (25th pct)": d["q25"].round(0),
        "High End (75th pct)": d["q75"].round(0), "Boom (90th pct)": d["q90"].round(0),
        "Average": d["mean"].round(1), "Chance of 0 Yards (%)": pct(d["p_zero"]),
        "Projected Carries": d["carries_projection"], "Yards per Carry": d["ypc"].round(2),
        "Team Implied Points": d["implied_total"], "Chance Plays (%)": pct(d["p_active"]),
        "Role Notes": d["role_note"].fillna("") if "role_note" in d else "", "_ko": d["_ko"]})
    out = out.sort_values(["_ko", "Game", "Projected Rush Yards (Median)"], ascending=[True, True, False])
    return out.drop(columns="_ko").reset_index(drop=True)


def boom(raw: pd.DataFrame, games: dict) -> pd.DataFrame:
    d = add_game(raw, games)
    out = pd.DataFrame({
        "Game": d["Game"], "Kickoff (ET)": d["Kickoff (ET)"], "Player": d["player"], "Team": d["team"],
        "Pos": d["position"], "Injury Status": d["status"].fillna(""),
        "Book Line (Consensus)": d["book_line"], "Our Projection (Median)": d["model_median"].round(0),
        "Yards Above Book Line": d["plus_yards"], "ALT OVER LINE": d["over"],
        "Yards Needed (X+)": d["yards_needed"], "Our Chance It Hits (%)": pct(d["p_boom"], 1),
        "FAIR ODDS (American)": d["fair_odds"].round(0).astype(int),
        "Take It If Book Pays At Least": d["bet_if_at_least"].round(0).astype(int), "_ko": d["_ko"],
        "_p": d["player"]})
    out = out.sort_values(["_ko", "Game", "Player", "Yards Above Book Line"])
    return out.drop(columns=["_ko", "_p"]).reset_index(drop=True)


def main():
    pr = [f for f in glob.glob(os.path.join(DATA, "projections_*_wk*_latest.csv"))
          if re.search(r"projections_(\d{4})_wk(\d+)_latest\.csv$", f)]
    if not pr:
        print("No projections file; nothing to do.")
        return
    key = lambda f: tuple(int(x) for x in re.search(r"(\d{4})_wk(\d+)", f).groups())
    season, week = max(key(f) for f in pr)
    tag = f"{season}_wk{week}"
    games = load_games(season, week) or games_from_edges(tag)
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*.csv")):          # only the current week's files stay
        os.remove(f)
    jobs = [
        (f"projections_{tag}_edges_latest.csv", f"Wk{week}_1_Receiving_Yards_Edges.csv", lambda d: edges(d, games, "yds")),
        (f"receptions_{tag}_edges_latest.csv", f"Wk{week}_2_Receptions_Edges.csv", lambda d: edges(d, games, "rec")),
        (f"rushing_{tag}_edges_latest.csv", f"Wk{week}_3_Rushing_Yards_Edges.csv", lambda d: edges(d, games, "yds")),
        (f"projections_{tag}_boom_latest.csv", f"Wk{week}_4_Boom_Odds_Alt_Overs.csv", lambda d: boom(d, games)),
        (f"projections_{tag}_latest.csv", f"Wk{week}_5_Receiving_Yards_Projections.csv", lambda d: yard_proj(d, games)),
        (f"receptions_{tag}_latest.csv", f"Wk{week}_6_Receptions_Projections.csv", lambda d: rec_proj(d, games)),
        (f"rushing_{tag}_latest.csv", f"Wk{week}_7_Rushing_Projections.csv", lambda d: rush_proj(d, games)),
    ]
    for src, dst, fn in jobs:
        p = os.path.join(DATA, src)
        if not os.path.exists(p):
            continue
        try:
            res = fn(pd.read_csv(p))
            res.to_csv(os.path.join(OUT, dst), index=False)
            print(f"Wrote {dst} ({len(res)} rows)")
        except Exception as e:
            print(f"Skipped {dst}: {e}")


if __name__ == "__main__":
    main()
