"""Boom probabilities: how likely a player is to clear his yardage line by 10, 20, 30 ... yards.

For each player with a sportsbook receiving-yards line, the ladder is the book's line + 10k yards
(the alternate "over" rungs books post), kept only while the fair price is no longer than +1000.
Chances come from the simulation, then get a tail correction learned from the walk-forward backtest:
for regular players the raw simulation overstates the far upper tail (see tail_map).
"""
import numpy as np
import pandas as pd

MAX_ODDS = 1000            # stop the ladder where fair odds pass +1000
STEP = 10
MAX_RUNGS = 12
P_FLOOR = 1 / (1 + MAX_ODDS / 100)
MIN_MEAN_FOR_CAL = 25      # backtest players used to learn the tail correction
CAL_POINTS = np.array([0.30, 0.20, 0.10, 0.05, 0.03, 0.015, 0.0075])


def to_american(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.where(p < 0.5, 100 * (1 - p) / p, -100 * p / (1 - p))


def tail_map(backtest: pd.DataFrame):
    """Returns f(p_model) -> p_realized for upper-tail chances, learned from backtest PIT values.
    PIT>1-p should happen with frequency p if the simulation is calibrated."""
    try:
        b = backtest[(backtest["sim_mean"] >= MIN_MEAN_FOR_CAL) & backtest["pit"].notna()]
        if len(b) < 500:
            raise ValueError("too few backtest rows")
        pit = b["pit"].to_numpy()
        real = np.array([(pit > 1 - p).mean() for p in CAL_POINTS])
        ratio = np.clip(real / CAL_POINTS, 0.4, 1.2)
        lr = np.log(ratio)

        def f(p):
            p = np.asarray(p, dtype=float)
            r = np.exp(np.interp(np.log(np.clip(p, 1e-6, 1)), np.log(CAL_POINTS[::-1]), lr[::-1]))
            return np.clip(p * r, 0, 1)
        info = {float(p): round(float(x), 4) for p, x in zip(CAL_POINTS, real)}
        print(f"Boom tail calibration (model chance -> realized, {len(b)} backtest games): {info}")
        return f
    except Exception as e:
        print(f"Boom tail calibration skipped ({e}); using raw simulation chances")
        return lambda p: np.asarray(p, dtype=float)


def consensus_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per player: median line across books, snapped to x.5."""
    g = lines.groupby("name_key").agg(book_line=("line", "median"), n_books=("book", "nunique"),
                                      player=("player", "first")).reset_index()
    g["book_line"] = np.floor(g["book_line"]) + 0.5
    return g


def boom_ladder(out: pd.DataFrame, sims: np.ndarray, lines: pd.DataFrame, cal, name_key) -> pd.DataFrame:
    idx = {}
    for i, k in enumerate(out["player_display_name"].map(name_key)):
        idx.setdefault(k, i)
    cons = consensus_lines(lines)
    rows = []
    for r in cons.itertuples():
        i = idx.get(r.name_key)
        if i is None:
            continue
        y = np.rint(sims[i])
        base_p = float((y > r.book_line).mean())
        for k in range(1, MAX_RUNGS + 1):
            thr = r.book_line + STEP * k
            p_raw = float((y > thr).mean())
            p = float(cal(p_raw))
            if p < P_FLOOR:
                break
            rows.append({
                "player": out.at[i, "player_display_name"], "team": out.at[i, "team"], "opp": out.at[i, "opp"],
                "position": out.at[i, "position"], "status": out.at[i, "status"],
                "book_line": r.book_line, "n_books": r.n_books, "model_median": out.at[i, "median"],
                "model_mean": out.at[i, "mean"], "p_over_book_line": base_p,
                "plus_yards": STEP * k, "over": thr, "yards_needed": int(np.ceil(thr)),
                "p_model": p_raw, "p_boom": p, "fair_odds": float(to_american(p)),
                "bet_if_at_least": float(to_american(min(p / 1.10, 0.999))),   # price that gives +10% EV
            })
    return pd.DataFrame(rows)


def boom_section(ladder: pd.DataFrame, max_players: int = 15, max_cols: int = 6) -> str:
    import html
    esc = html.escape
    if ladder.empty:
        return ""
    th = "padding:6px 8px;text-align:left;background:#f3f3f3;white-space:nowrap"
    td = "padding:4px 8px;white-space:nowrap;border-bottom:1px solid #e5e5e5"
    # rank by chance of clearing the line by 30+ (or the farthest rung available)
    rank = (ladder[ladder["plus_yards"] == 30].set_index("player")["p_boom"])
    players = rank.sort_values(ascending=False).head(max_players).index.tolist()
    head = ("<h3 style='margin:18px 0 4px'>Boom odds: chance to clear the yardage line by 10, 20, 30+ yards</h3>"
            "<p style='margin:0 0 6px;color:#555;font-size:12px'>Each cell is our chance and the <b>fair</b> American "
            "price for the alternate over at the book's line plus that many yards (consensus line; rungs past +1000 "
            "are dropped). The small number is the price you'd need for a 10% edge: if the book pays that much or more, "
            "it's worth a look. Chances include a haircut because the raw simulation overstates the far tail. "
            "Chances assume the player plays. Full ladder in the boom CSV.</p>")
    cols = [f"+{STEP * k}" for k in range(1, max_cols + 1)]
    rows = []
    for p in players:
        sub = ladder[ladder["player"] == p].set_index("plus_yards")
        first = sub.iloc[0]
        cells = [f'<td style="{td};font-weight:600">{esc(str(p))}</td>',
                 f'<td style="{td}">{esc(str(first["team"]))} vs {esc(str(first["opp"]))}</td>',
                 f'<td style="{td};text-align:right">{first["book_line"]:g}</td>']
        for k in range(1, max_cols + 1):
            y = STEP * k
            if y in sub.index:
                s = sub.loc[y]
                cells.append(f'<td style="{td};text-align:right"><b>{s["p_boom"]:.0%}</b> '
                             f'{s["fair_odds"]:+.0f}<div style="font-size:10px;color:#777">&ge;{s["bet_if_at_least"]:+.0f}</div></td>')
            else:
                cells.append(f'<td style="{td};text-align:right;color:#bbb">&ndash;</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    t = ('<table style="border-collapse:collapse;font-size:13px"><tr>'
         + "".join(f'<th style="{th}">{c}</th>' for c in ["Player", "Game", "Book line"] + cols) + "</tr>"
         + "".join(rows) + "</table>")
    return head + t
