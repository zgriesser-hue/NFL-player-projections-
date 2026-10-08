"""
A/B backtest for the rushing model: do the 'vacated carries' features help?

Vacated carries = recent carry form of backs who played for the team in its previous 3 games but are not
in the active group now (IR, cut, traded, or Out). The injury report only catches Out players, so the model
currently misses, for example, a lead back on IR. Same walk-forward backtest twice (same rows, same random
draws), paired by week.

Run:  python ab_rushing.py     (about twice as long as a normal backtest)
Adopt only if the verdict says the gain is clear: set USE_VAC_FEATURES = True at the top of rushing.py.
"""
import os

import numpy as np
import pandas as pd

import projections as P
import rushing as R

QUANTS = [(0.10, "sim_q10"), (0.25, "sim_q25"), (0.50, "sim_q50"), (0.75, "sim_q75"), (0.90, "sim_q90")]
LINES = []


def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


def losses(d: pd.DataFrame) -> dict:
    y = d[R.TARGET].to_numpy(dtype=float)
    pin = sum(np.maximum(q * (y - d[c]), (q - 1) * (y - d[c])) for q, c in QUANTS) / len(QUANTS)
    return {"MAE (median)": (y - d["sim_q50"]).abs(),
            "RMSE (mean)": (y - d["sim_mean"]) ** 2,
            "Pinball (5 quantiles)": pin}


def line(name, la, lb, sqrt=False):
    d = (la - lb).groupby(level=["season", "week"]).mean()
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    ma, mb = la.mean(), lb.mean()
    if sqrt:
        va, vb, rel, ci = np.sqrt(ma), np.sqrt(mb), 1 - np.sqrt(mb / ma), 0.5 * 1.96 * se / ma
    else:
        va, vb, rel, ci = ma, mb, (ma - mb) / ma, 1.96 * se / ma
    say(f"  {name:<30}{va:>8.3f}{vb:>9.3f}{rel:>+9.2%}  ±{ci:.2%}   {(d > 0).mean():>5.0%}")
    return rel, ci


def main() -> None:
    out_dir = P.resolve_out_dir()
    say("Building data...")
    played, parts, inj = R.build_rush_table(project=None)
    played["is_future"] = False
    full = R.add_features(R.add_vac_features(played))
    hist = full[~full["is_future"]].copy()
    vac = hist.set_index(["player_id", "season", "week"])["vac_all_carries"]

    res = {}
    for use in (False, True):
        say(f"\nBacktest {'B: + vacated-carries features' if use else 'A: baseline features'}")
        R.set_vac(use)
        res[use] = R.walk_forward(hist).set_index(["player_id", "season", "week"]).sort_index()
    a, b = res[False], res[True]
    if not a.index.equals(b.index):
        raise ValueError("Backtests evaluated different rows; cannot pair them.")
    la, lb = losses(a), losses(b)
    flag = vac.reindex(a.index).to_numpy() >= 8          # a meaningful amount of carries is missing

    say(f"\n{len(a):,} player-weeks, {a.groupby(level=['season', 'week']).ngroups} weeks; "
        f"{int(flag.sum())} with 8+ vacated carries")
    say(f"  {'':<30}{'base':>8}{'+ vac':>9}{'gain':>9}  {'95% CI':<8}{'weeks better':>6}")
    gains = {n: line(n, la[n], lb[n], sqrt=(n == "RMSE (mean)")) for n in la}
    say("\nMAE by position:")
    for pos in R.RUSH_POS:
        m = (a["position"] == pos).to_numpy()
        line(pos, la["MAE (median)"][m], lb["MAE (median)"][m])
    say("\nMAE where 8+ carries are vacated (the situation the feature targets):")
    if flag.sum() >= 30:
        line("vacated >= 8", la["MAE (median)"][flag], lb["MAE (median)"][flag])
        line("vacated < 8", la["MAE (median)"][~flag], lb["MAE (median)"][~flag])
    else:
        say("  too few cases to say")

    mae_rel, mae_ci = gains["MAE (median)"]
    pin_rel, pin_ci = gains["Pinball (5 quantiles)"]
    say("\nVerdict:")
    if mae_rel - mae_ci > 0 and pin_rel - pin_ci > 0:
        say("  Clear gain on both MAE and pinball. Set USE_VAC_FEATURES = True in rushing.py.")
    elif mae_rel + mae_ci < 0 or pin_rel + pin_ci < 0:
        say("  The vacated-carries features make things worse. Leave USE_VAC_FEATURES = False.")
    else:
        say("  No clear overall difference. If the 'vacated >= 8' rows improve clearly while the rest")
        say("  stay flat, the feature is still doing its job on the cases that matter; otherwise leave it off.")
    with open(os.path.join(out_dir, "ab_rushing_report.txt"), "w") as f:
        f.write("\n".join(LINES))


if __name__ == "__main__":
    main()
