"""
A/B backtest: do the opponent defense-vs-position features help?

Runs the identical walk-forward backtest twice (same rows, same settings, same random
draws), once without and once with the opponent features, then compares them. Results are
paired by week, so the confidence intervals reflect week-to-week noise.

Run:  python ab_opponent.py      (takes roughly twice as long as a normal backtest)
Adopt the features (set USE_OPP_FEATURES = True in projections.py) only if the verdict
at the bottom says the gain is clear.
"""
import os

import numpy as np
import pandas as pd

import projections as P

QUANTS = [(0.10, "sim_q10"), (0.25, "sim_q25"), (0.50, "sim_q50"),
          (0.75, "sim_q75"), (0.90, "sim_q90")]
LINES = []


def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


def prepare() -> pd.DataFrame:
    played, parts, _ = P.build_table(project=None)
    played["is_future"] = False
    full = P.add_features(played)
    return P.add_opponent_features(full, parts["ctx"])


def backtest(full: pd.DataFrame, use_opp: bool) -> pd.DataFrame:
    P.MODEL_FEATURES = P.FEATURES + P.RATIO_FEATURES + (P.OPP_FEATURES if use_opp else [])
    res, _ = P.walk_forward(full)
    return res


def pinball(res: pd.DataFrame) -> pd.Series:
    """Average pinball loss over the q10/q25/q50/q75/q90 forecasts (distribution quality)."""
    y = res[P.TARGET].to_numpy()
    total = np.zeros(len(res))
    for tau, col in QUANTS:
        d = y - res[col].to_numpy()
        total += np.maximum(tau * d, (tau - 1) * d)
    return pd.Series(total / len(QUANTS), index=res.index)


def losses(res: pd.DataFrame) -> dict:
    y = res[P.TARGET]
    return {
        "MAE (median)": (y - res["sim_q50"]).abs(),
        "RMSE (mean)": (y - res["sim_mean"]) ** 2,          # squared; sqrt'd for display
        "Pinball (5 quantiles)": pinball(res),
        "Targets MAE": (res["targets"] - res["mu_targets"]).abs(),
    }


def line(name: str, la: pd.Series, lb: pd.Series, sqrt: bool = False):
    """One comparison row; returns (relative gain, half-width of its 95% CI)."""
    d = (la - lb).groupby(level=["season", "week"]).mean()          # paired by week
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    ma, mb = la.mean(), lb.mean()
    if sqrt:
        va, vb, rel, ci = np.sqrt(ma), np.sqrt(mb), 1 - np.sqrt(mb / ma), 0.5 * 1.96 * se / ma
    else:
        va, vb, rel, ci = ma, mb, (ma - mb) / ma, 1.96 * se / ma
    say(f"  {name:<24}{va:>8.3f}{vb:>9.3f}{rel:>+9.2%}  ±{ci:.2%}   {(d > 0).mean():>5.0%}")
    return rel, ci


def compare(res_a: pd.DataFrame, res_b: pd.DataFrame) -> None:
    key = ["player_id", "season", "week"]
    a = res_a.set_index(key).sort_index()
    b = res_b.set_index(key).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Backtests evaluated different rows; cannot pair them.")
    la, lb = losses(a), losses(b)

    say(f"\n{len(a):,} player-weeks, {a.groupby(level=['season', 'week']).ngroups} weeks")
    say(f"  {'':<24}{'base':>8}{'+ opp':>9}{'gain':>9}  {'95% CI':<8}{'weeks better':>6}")
    gains = {}
    for name in la:
        gains[name] = line(name, la[name], lb[name], sqrt=(name == "RMSE (mean)"))

    say("\nMAE by position:")
    for pos in P.POSITIONS:
        m = a["position"] == pos
        line(pos, la["MAE (median)"][m], lb["MAE (median)"][m])

    mae_rel, mae_ci = gains["MAE (median)"]
    pin_rel, pin_ci = gains["Pinball (5 quantiles)"]
    say("\nVerdict:")
    if mae_rel - mae_ci > 0 and pin_rel - pin_ci > 0:
        say("  Clear gain on both MAE and pinball. Set USE_OPP_FEATURES = True in projections.py.")
    elif mae_rel + mae_ci < 0 or pin_rel + pin_ci < 0:
        say("  The opponent features make things worse. Leave USE_OPP_FEATURES = False.")
    else:
        say("  No clear difference (gain is within noise). Leave USE_OPP_FEATURES = False;")
        say("  extra features that don't clearly help mostly add overfitting risk.")


def main() -> None:
    out_dir = P.resolve_out_dir()
    say("Building data...")
    full = prepare()
    say("\nBacktest A: baseline features")
    res_a = backtest(full, use_opp=False)
    say("Backtest B: baseline + opponent features")
    res_b = backtest(full, use_opp=True)
    compare(res_a, res_b)
    with open(os.path.join(out_dir, "ab_opponent_latest.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
