"""Larger-sample check of the simulation's upper tail (feeds the boom-odds correction).

Runs the same walk-forward backtest as projections.py over more seasons (default 2021-2025, about 2.5x the
usual sample) into data/receiving_backtest_boom.csv, then prints how often the actual result beat the
simulation's 90th/95th/97th/99th percentile, overall and split by position, predicted yards per catch,
projected volume and by individual player. It does not change the live model or its cached backtest.

Run:  python boom_backtest.py            (BOOM_SEASONS=2019,2020,... to change the test seasons)
"""
import os

import numpy as np
import pandas as pd

LEVELS = [0.10, 0.05, 0.03, 0.01]


def tail_table(d: pd.DataFrame, by, label: str) -> None:
    rows = []
    for k, g in d.groupby(by, observed=True):
        rows.append({label: k, "n": len(g), **{f"top{int(p * 100)}%": round(float((g["pit"] > 1 - p).mean()), 3)
                                                   for p in LEVELS}})
    print(pd.DataFrame(rows).to_string(index=False))


def analyze(bt: pd.DataFrame) -> None:
    d = bt[(bt["sim_mean"] >= 25) & bt["pit"].notna()].copy()
    print(f"\n=== Upper-tail calibration: {len(d)} games of players projected 25+ yards, "
          f"seasons {sorted(d['season'].unique())} ===")
    print("(a calibrated simulation shows 0.10 / 0.05 / 0.03 / 0.01 in the columns)")
    d["all"] = "all"
    tail_table(d, "all", "group")
    print("\nBy season:");  tail_table(d, "season", "season")
    print("\nBy position:"); tail_table(d, "position", "position")
    d["ypr_q"] = pd.qcut(d["ypr_pred"], 4, labels=["lowest yds/catch", "2", "3", "highest yds/catch"])
    print("\nBy predicted yards per catch (quartiles):"); tail_table(d, "ypr_q", "ypr")
    d["vol_q"] = pd.qcut(d["sim_mean"], 4, labels=["lowest proj", "2", "3", "highest proj"])
    print("\nBy projected yards (quartiles):"); tail_table(d, "vol_q", "proj")
    # player-level: is the spread in per-player boom rates bigger than chance?
    g = d.groupby("player_id")["pit"].agg(n="size", p10=lambda x: (x > 0.90).mean())
    g = g[g["n"] >= 25]
    if len(g) >= 20:
        chance = np.sqrt(0.09 * 0.91 / g["n"].mean())
        print(f"\nPlayer-level: {len(g)} players with 25+ games. SD of per-player top-10% rate = "
              f"{g['p10'].std():.3f}; pure chance would give about {chance:.3f}.")
        print("  (clearly above chance => some players really do have fatter tails)")
    # proposed correction: realized/model chance by ypr quartile at each level
    print("\nSuggested shrink on the simulated chance (realized / model), by yards-per-catch quartile:")
    out = []
    for k, gr in d.groupby("ypr_q", observed=True):
        out.append({"ypr": k, **{f"at {int(p * 100)}%": round(float((gr["pit"] > 1 - p).mean() / p), 2)
                                 for p in LEVELS}})
    print(pd.DataFrame(out).to_string(index=False))


def main():
    import projections as P
    seasons = [int(s) for s in os.environ.get("BOOM_SEASONS", "2021,2022,2023,2024,2025").split(",")]
    P.TEST_SEASONS = seasons
    out_dir = P.resolve_out_dir()
    path = os.path.join(out_dir, "receiving_backtest_boom.csv")
    if os.path.exists(path) and not os.environ.get("BOOM_REFRESH"):
        print(f"Using cached {path} (set BOOM_REFRESH=1 to redo)")
        bt = pd.read_csv(path)
    else:
        played, parts, _ = P.build_table(project=None)
        played["is_future"] = False
        full = P.add_features(played)
        full = P.add_opponent_features(full, parts["ctx"])
        full = P.add_weather_features(full)
        bt, _ = P.walk_forward(full)
        bt.to_csv(path, index=False)
        print(f"Saved {path} ({len(bt)} rows)")
    analyze(bt)


if __name__ == "__main__":
    main()
