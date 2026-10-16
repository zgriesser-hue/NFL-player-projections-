"""
A/B backtest of two new feature sets: practice participation and game-day weather.

Runs the identical walk-forward backtest three times (same rows, settings and random
draws): baseline, baseline + practice participation, baseline + weather. Each variant is
compared to the baseline, paired by week. Because these features only matter for a small
share of games, it also shows MAE in the subsets where they should help (windy/cold games;
players who were limited in practice).

Run:  python ab_features.py      (about 1.5x the time of ab_opponent.py)
Flip USE_PRACTICE_FEATURES / USE_WEATHER_FEATURES in projections.py only for a variant whose
verdict says the gain is clear.
"""
import os

import numpy as np
import pandas as pd

import ab_opponent as AB
import projections as P

KEY = ["player_id", "season", "week"]


def prepare() -> pd.DataFrame:
    played, _, _ = P.build_table(project=None)
    played["is_future"] = False
    full = P.add_features(played)
    return P.add_ngs_features(P.add_weather_features(full))


def backtest(full: pd.DataFrame, **sets) -> pd.DataFrame:
    P.MODEL_FEATURES = P.build_model_features(**sets)
    res, _ = P.walk_forward(full)
    return res


def wx_subsets(f: pd.DataFrame) -> dict:
    outdoor = f["wx_is_dome"] == 0
    windy = outdoor & (f["wx_wind_mph"] >= 15)
    cold = outdoor & (f["wx_temp_f"] <= 32)
    return {"windy (>= 15 mph)": windy, "cold (<= 32F)": cold,
            "windy or cold": windy | cold, "dome / closed roof": ~outdoor}


def prac_subsets(f: pd.DataFrame) -> dict:
    lim, dnp = f["prac_limited"] == 1, f["prac_dnp"] == 1
    return {"limited in practice": lim, "DNP in practice, played": dnp, "either": lim | dnp}


def ngs_subsets(f: pd.DataFrame) -> dict:
    has = f["ngs_avg_separation"].notna()
    hi = f["ngs_avg_separation"] >= f["ngs_avg_separation"].median()
    return {"has NGS history": has, "no NGS history": ~has, "above-median separation": has & hi,
            "below-median separation": has & ~hi}


def recent_subsets(f: pd.DataFrame) -> dict:
    jump = f["last_targets"] - f["ewm_targets"]
    return {"last game targets up 1.2+": jump >= 1.2, "last game targets down 1.3+": jump <= -1.3,
            "3 or fewer prior games": f["games_prior"] <= 2, "everyone else": ~((jump >= 1.2) | (jump <= -1.3))}


def subset_line(name: str, la: pd.Series, lb: pd.Series, mask) -> None:
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n < 30:
        AB.say(f"  {name:<28}{n:>6}   (too few rows to read)")
        return
    d = la[mask] - lb[mask]
    ma, mb = la[mask].mean(), lb[mask].mean()
    se = d.std(ddof=1) / np.sqrt(n)
    AB.say(f"  {name:<28}{n:>6}{ma:>9.3f}{mb:>9.3f}{(ma - mb) / ma:>+9.2%}  ±{1.96 * se / ma:.2%}")


def compare_variant(title: str, res_a: pd.DataFrame, res_b: pd.DataFrame,
                    feat: pd.DataFrame, subsets) -> None:
    a = res_a.set_index(KEY).sort_index()
    b = res_b.set_index(KEY).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Backtests evaluated different rows; cannot pair them.")
    la, lb = AB.losses(a), AB.losses(b)

    AB.say(f"\n=== Baseline vs + {title} ===")
    AB.say(f"{len(a):,} player-weeks, {a.groupby(level=['season', 'week']).ngroups} weeks")
    AB.say(f"  {'':<24}{'base':>8}{'+ feat':>9}{'gain':>9}  {'95% CI':<8}{'weeks better':>6}")
    gains = {name: AB.line(name, la[name], lb[name], sqrt=(name == "RMSE (mean)")) for name in la}

    AB.say("\nWhere this feature should matter (MAE; the ± is rough, row-level):")
    AB.say(f"  {'':<28}{'n':>6}{'base':>9}{'+ feat':>9}{'gain':>9}")
    sub = feat.reindex(a.index)
    for name, mask in subsets(sub).items():
        subset_line(name, la["MAE (median)"], lb["MAE (median)"], mask)

    mae_rel, mae_ci = gains["MAE (median)"]
    pin_rel, pin_ci = gains["Pinball (5 quantiles)"]
    AB.say("\nVerdict:")
    if mae_rel - mae_ci > 0 and pin_rel - pin_ci > 0:
        AB.say(f"  Clear overall gain. Turn this feature set on in projections.py.")
    elif mae_rel + mae_ci < 0 or pin_rel + pin_ci < 0:
        AB.say("  Makes things worse overall. Leave it off.")
    else:
        AB.say("  No clear overall difference. Leave it off unless the subset rows above are")
        AB.say("  convincingly better (they're small samples; treat them as a hint, not proof).")


def main() -> None:
    out_dir = P.resolve_out_dir()
    AB.say("Building data...")
    full = prepare()
    feat = full.set_index(KEY)[P.PRAC_FEATURES + P.WX_FEATURES + P.NGS_FEATURES + P.RECENT_FEATURES]

    AB.say("\nBacktest A: baseline features")
    res_a = backtest(full)
    variants = [("practice participation", dict(prac=True), prac_subsets),
                ("game-day weather", dict(wx=True), wx_subsets),
                ("Next Gen Stats tracking", dict(ngs=True), ngs_subsets),
                ("recent form (last game / last 3)", dict(recent=True), recent_subsets)]
    only = os.environ.get("AB_ONLY")
    if only:
        variants = [v for v in variants if v[0].lower().startswith({"ngs": "next", "recent": "recent"}.get(only, only))]
    for title, sets, subsets in variants:
        AB.say(f"Backtest: baseline + {title}")
        compare_variant(title, res_a, backtest(full, **sets), feat, subsets)

    with open(os.path.join(out_dir, "ab_features_latest.txt"), "w") as f:
        f.write("\n".join(AB.LINES) + "\n")


if __name__ == "__main__":
    main()
