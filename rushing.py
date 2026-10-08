"""
Rushing-yards projections (RB and QB), same two-stage simulation idea as projections.py.

    P(carries >= 1) -> carries - 1 ~ NegativeBinomial -> yards ~ Gamma(shape s*carries, scale ypc/s)

Features: recent form (EWMA of carries, rushing yards, targets, share of team carries, snap share),
game script (spread, implied total, home), carries vacated by injured teammates, own questionable tag.
Outputs: rushing_{season}_wk{week}_*.csv, plus rushing_email_part.html that projections.py
appends to the email when this runs first.

Run:  python rushing.py        (python projections.py afterwards if you want the email)
Output folder: NFL_OUTPUT_DIR if set, else Google Drive, else here.
"""
import html as _html
import os

import lightgbm as lgb
import nflreadpy as nfl
import numpy as np
import pandas as pd

import lines_parlay
import projections as P
import role_news

RUSH_POS = ["RB", "QB"]
RUSH_STATS = ["carries", "rushing_yards", "targets"]
FORM = RUSH_STATS + ["carry_share", "offense_pct"]
TARGET = "rushing_yards"
MODEL_VERSION = "rush-r1"
BACKTEST_CACHE = "rushing_backtest_r3.csv"      # r2: PIT treats negative yardage as 0, adds diagnostics
REFRESH_BACKTEST = False
MIN_SNAP_RB = 0.20          # skip RBs whose recent snap share is below this
MIN_SNAP_QB = 0.50          # QBs: only recent starters
CLIP_YPC = 1.0

BASE_FEATURES = (
    [f"ewm_{c}" for c in FORM]
    + ["games_prior", "implied_total", "spread", "is_home",
       "inj_out_carries", "inj_n_out_rb", "self_questionable", "is_qb"]
)
FEATURES = BASE_FEATURES + ["ewm_ypc"]
BASELINE_COL = f"ewm_{TARGET}"
MEDIAN_COL = "med_rushing_yards"


# ---------------------------------------------------------------- data
def load_rush_weeks():
    stats = nfl.load_player_stats(P.SEASONS).to_pandas()
    stats = stats[stats["season_type"] == "REG"].copy()

    tc = stats.assign(team=P.fix_team(stats["team"])).groupby(
        ["team", "season", "week"], as_index=False)["carries"].sum().rename(
        columns={"carries": "team_carries"})

    players = (nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]]
               .dropna().drop_duplicates("pfr_id"))
    snaps = nfl.load_snap_counts(P.SEASONS).to_pandas()
    snaps = snaps[(snaps["game_type"] == "REG")
                  & (snaps["position"].isin(RUSH_POS + ["FB"]))
                  & (snaps["offense_snaps"] > 0)]
    snaps = snaps.merge(players, left_on="pfr_player_id", right_on="pfr_id",
                        how="inner").rename(columns={"gsis_id": "player_id",
                                                     "player": "player_display_name"})
    snaps["team"] = P.fix_team(snaps["team"])
    snaps["position"] = snaps["position"].replace({"FB": "RB"})
    snaps = (snaps[["player_id", "player_display_name", "season", "week", "team",
                    "position", "offense_snaps", "offense_pct"]]
             .drop_duplicates(["player_id", "season", "week"]))

    pw = snaps.merge(stats[["player_id", "season", "week"] + RUSH_STATS],
                     on=["player_id", "season", "week"], how="left")
    pw[RUSH_STATS] = pw[RUSH_STATS].fillna(0)
    pw = pw.merge(tc, on=["team", "season", "week"], how="left")
    pw["carry_share"] = (pw["carries"] / pw["team_carries"]).replace(
        [np.inf, -np.inf], np.nan).fillna(0)
    return pw.drop(columns="team_carries"), stats


def vacated_carries(stats: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    """Per team-week: carries (recent form) belonging to RBs listed Out, and how many are out."""
    form = stats[stats["position"].isin(["RB", "FB"])][["player_id", "season", "week", "carries"]].copy()
    form = form.sort_values(["player_id", "season", "week"])
    form["form_carries"] = form.groupby("player_id")["carries"].transform(
        lambda s: s.ewm(halflife=P.HALFLIFE, min_periods=1).mean())
    form["t"] = form["season"].astype(int) * 100 + form["week"].astype(int)

    out = inj[(inj["report_status"] == "Out") & (inj["position"].isin(["RB", "FB"]))].copy()
    out["t"] = out["season"] * 100 + out["week"]
    out = pd.merge_asof(
        out.sort_values("t"), form[["player_id", "t", "form_carries"]].sort_values("t"),
        on="t", by="player_id", allow_exact_matches=False, tolerance=100)
    out["form_carries"] = out["form_carries"].fillna(0)
    return (out.groupby(["team", "season", "week"])
            .agg(inj_out_carries=("form_carries", "sum"), inj_n_out_rb=("player_id", "count"))
            .reset_index())


def build_rush_table(project=None):
    pw, stats = load_rush_weeks()
    _team_out_unused, q, _prac, inj = P.injury_features(stats, project)
    team_out = vacated_carries(stats, inj)
    ctx = P.load_context()

    df = pw.merge(ctx, on=["season", "week", "team"], how="left")
    df = df.merge(team_out, on=["team", "season", "week"], how="left")
    df = df.merge(q, on=["player_id", "season", "week"], how="left")
    cols = ["inj_out_carries", "inj_n_out_rb", "self_questionable"]
    df[cols] = df[cols].fillna(0)
    name_map = (stats.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
                .set_index("player_id")["player_display_name"])
    df["player_display_name"] = df["player_id"].map(name_map).fillna(df["player_display_name"])
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    return df, dict(pw=pw, ctx=ctx, team_out=team_out, q=q), inj


def make_future_rows(played, parts, inj, season: int, week: int) -> pd.DataFrame:
    pw, ctx = parts["pw"], parts["ctx"]
    last = (played.sort_values(["season", "week"]).groupby("player_id").tail(1)
            [["player_id", "player_display_name", "position", "team", "season", "week"]]
            .rename(columns={"team": "last_team", "season": "last_season", "week": "last_week"}))
    roster = P.current_roster(season, week)
    if roster is not None:
        cand = last.merge(roster, on="player_id", how="inner")
    else:
        cand = last[last["last_season"] * 100 + last["last_week"] >= (season - 1) * 100 + 15].copy()
        cand["team"] = cand["last_team"]
    cand = cand.drop(columns=["last_team", "last_season", "last_week"])
    cand["season"], cand["week"] = season, week

    c = ctx[(ctx["season"] == season) & (ctx["week"] == week)]
    cand = cand.merge(c, on=["season", "week", "team"], how="inner")
    done = pw[["player_id", "season", "week"]].assign(_done=1)
    cand = cand.merge(done, on=["player_id", "season", "week"], how="left")
    cand = cand[cand["_done"].isna()].drop(columns="_done")

    cand = cand.merge(parts["team_out"], on=["team", "season", "week"], how="left")
    cand = cand.merge(parts["q"], on=["player_id", "season", "week"], how="left")
    cols = ["inj_out_carries", "inj_n_out_rb", "self_questionable"]
    cand[cols] = cand[cols].fillna(0)
    st = (inj[(inj["season"] == season) & (inj["week"] == week)]
          .drop_duplicates("player_id")[["player_id", "report_status"]]
          .rename(columns={"report_status": "status"}))
    cand = cand.merge(st, on="player_id", how="left")
    # practice flags for status inference (same idea as projections.py)
    pr = inj[(inj["season"] == season) & (inj["week"] == week)].drop_duplicates("player_id")
    dnp, lim = P.practice_flags(pr)
    pr = pr.assign(prac_dnp=dnp.astype(int), prac_limited=lim.astype(int))[
        ["player_id", "prac_dnp", "prac_limited"]]
    cand = cand.merge(pr, on="player_id", how="left")
    cand[["prac_dnp", "prac_limited"]] = cand[["prac_dnp", "prac_limited"]].fillna(0)
    cand["is_future"] = True
    return cand


# ----------------------------------------------------------- features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("player_id", group_keys=False)
    for col in FORM:
        df[f"ewm_{col}"] = g[col].apply(
            lambda s: s.shift(1).ewm(halflife=P.HALFLIFE, min_periods=1).mean())
    df["games_prior"] = g.cumcount()
    df[MEDIAN_COL] = g[TARGET].apply(
        lambda s: s.shift(1).rolling(P.MEDIAN_WINDOW, min_periods=1).median())
    df["is_qb"] = (df["position"] == "QB").astype(int)
    df["ewm_ypc"] = (df["ewm_rushing_yards"] / df["ewm_carries"]).replace([np.inf, -np.inf], np.nan)
    return df


# ---------------------------------------------------------- simulation
def estimate_dispersion(hist: pd.DataFrame):
    """Per position: NB dispersion k for carries, gamma shape s for yards per carry,
    from PAST out-of-sample rows only (method of moments)."""
    k, s, est = {}, {}, {}
    for pos in RUSH_POS:
        h = hist[hist["position"] == pos] if len(hist) else hist
        if len(h) < P.MIN_CAL_ROWS:
            k[pos], s[pos], est[pos] = P.K_DEFAULT, 0.8, False
            continue
        hp = h[h["carries"] > 0]
        y = hp["carries"] - 1
        excess = np.mean((y - hp["mu_pos"]) ** 2) - np.mean(hp["mu_pos"])
        k[pos] = float(np.clip(np.mean(hp["mu_pos"] ** 2) / max(excess, 1e-6), 1.0, 100.0))
        inv_s = np.mean((hp[TARGET] - hp["carries"] * hp["ypc_pred"]) ** 2
                        / (hp["carries"] * hp["ypc_pred"] ** 2))
        s[pos] = float(np.clip(1.0 / inv_s, 0.2, 10.0))
        est[pos] = True
    return k, s, est


def estimate_adjust(hist: pd.DataFrame):
    """Per position, from PAST out-of-sample rows only: a level scale (actual total / raw simulated total,
    clipped to +-8%) and a spread widening factor (raises k and s down when the 80% range
    covers less than 80% of outcomes). Returns ({pos: scale}, {pos: widen})."""
    scale, widen = {p: 1.0 for p in RUSH_POS}, {p: 1.0 for p in RUSH_POS}
    if not len(hist) or "scale_used" not in hist.columns:
        return scale, widen
    for pos in RUSH_POS:
        h = hist[hist["position"] == pos]
        if len(h) < P.MIN_CAL_ROWS:
            continue
        raw = (h["sim_mean"] / h["scale_used"]).sum()
        scale[pos] = float(np.clip(h[TARGET].clip(lower=0).sum() / max(raw, 1e-9), 0.92, 1.08))
        cover = float(((h[TARGET].clip(lower=0) >= h["sim_q10"]) & (h[TARGET].clip(lower=0) <= h["sim_q90"])).mean())
        widen[pos] = float(np.clip(h["widen_used"].mean() * (1 + 2.0 * (0.80 - cover)), 1.0, 2.5))
    return scale, widen


def simulate(p_pos, mu_pos, ypc, k, s, rng, return_carries: bool = False, scale=None, widen=None):
    if widen is not None:
        k, s = k / widen, np.maximum(s / widen, 0.1)
    shp = (len(mu_pos), P.N_SIMS)
    mu_c, k_c, s_c = mu_pos[:, None], np.maximum(k, 0.3)[:, None], s[:, None]
    carried = rng.random(shp) < p_pos[:, None]
    extra = rng.negative_binomial(k_c, k_c / (k_c + mu_c), size=shp)
    carries = np.where(carried, 1 + extra, 0)
    yards = rng.gamma(np.maximum(s_c * carries, 1e-9), ypc[:, None] / s_c, size=shp)
    yards = np.where(carries > 0, yards, 0.0)
    if scale is not None:
        yards = yards * scale[:, None]
    return (yards, carries) if return_carries else yards


def fit_stage_models(train: pd.DataFrame):
    X = train[FEATURES]
    m_hit = lgb.LGBMClassifier(**P.PARAMS).fit(X, (train["carries"] > 0).astype(int))
    tr = train[train["carries"] > 0]
    m_car = lgb.LGBMRegressor(objective="poisson", **P.PARAMS).fit(
        tr[FEATURES], tr["carries"] - 1)
    m_ypc = lgb.LGBMRegressor(objective="regression", **P.PARAMS).fit(
        tr[FEATURES], tr[TARGET] / tr["carries"], sample_weight=tr["carries"])
    return m_hit, m_car, m_ypc


def predict_stages(models, X: pd.DataFrame):
    m_hit, m_car, m_ypc = models
    p_pos = np.clip(m_hit.predict_proba(X)[:, 1], 0.01, 0.99)
    mu_pos = np.clip(m_car.predict(X), 0.05, None)
    ypc = np.clip(m_ypc.predict(X), CLIP_YPC, None)
    return p_pos, mu_pos, ypc


def walk_forward(df: pd.DataFrame):
    df = df.dropna(subset=BASE_FEATURES + [TARGET]).copy()
    df = df[df["games_prior"] >= P.MIN_PRIOR_GAMES]
    rng = np.random.default_rng(P.SEED)
    preds = []
    test = df[df["season"].isin(P.TEST_SEASONS)]
    for (season, week), wk in test.groupby(["season", "week"]):
        train = df[(df["season"] < season) | ((df["season"] == season) & (df["week"] < week))]
        models = fit_stage_models(train)
        Xw = wk[FEATURES]
        p_pos, mu_pos, ypc = predict_stages(models, Xw)
        m_direct = lgb.LGBMRegressor(objective="quantile", alpha=0.5).fit(train[FEATURES], train[TARGET])

        hist = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
        k_by, s_by, est_by = estimate_dispersion(hist)
        k = wk["position"].map(k_by).to_numpy(dtype=float)
        s = wk["position"].map(s_by).to_numpy(dtype=float)
        sc_by, wd_by = estimate_adjust(hist)
        sc = wk["position"].map(sc_by).to_numpy(dtype=float)
        wd = wk["position"].map(wd_by).to_numpy(dtype=float)
        sims, car = simulate(p_pos, mu_pos, ypc, k, s, rng, return_carries=True, scale=sc, widen=wd)
        q10, q25, q50, q75, q90 = np.percentile(sims, [10, 25, 50, 75, 90], axis=1)

        # Simulated yards can't go below 0 but real games can (kneel-downs, a loss on one carry).
        # For calibration, treat negative actual yardage as 0 so it isn't counted as "below everything".
        y_act = np.clip(wk[TARGET].to_numpy(), 0, None)[:, None]
        pit_lo, pit_hi = (sims < y_act).mean(axis=1), (sims <= y_act).mean(axis=1)

        out = wk[["player_id", "player_display_name", "season", "week", "position", TARGET,
                  "carries", BASELINE_COL, MEDIAN_COL, "ewm_carries"]].copy()
        out["p_pos"], out["mu_pos"] = p_pos, mu_pos
        out["mu_carries"] = p_pos * (1 + mu_pos)
        out["ypc_pred"] = ypc
        out["pred_direct_median"] = m_direct.predict(Xw)
        out["sim_mean"] = sims.mean(axis=1)
        out["sim_q10"], out["sim_q25"], out["sim_q50"] = q10, q25, q50
        out["sim_q75"], out["sim_q90"] = q75, q90
        out["p_zero"] = (sims == 0).mean(axis=1)
        out["neg_actual"] = (wk[TARGET].to_numpy() < 0).astype(int)
        out["pit"] = pit_lo + rng.random(len(out)) * (pit_hi - pit_lo)
        out["scale_used"], out["widen_used"] = sc, wd
        out["disp_est"] = wk["position"].map(est_by).to_numpy(dtype=bool)
        preds.append(out)
    return pd.concat(preds, ignore_index=True)


def report(res: pd.DataFrame) -> None:
    mae = lambda c: np.mean(np.abs(res[TARGET] - res[c]))
    print(f"Rows evaluated: {len(res):,}")
    print("\nRushing yards MAE:")
    print(f"  EWMA mean baseline       : {mae(BASELINE_COL):.2f}")
    print(f"  Rolling median baseline  : {mae(MEDIAN_COL):.2f}")
    print(f"  Direct quantile-median   : {mae('pred_direct_median'):.2f}")
    print(f"  Two-stage (sim median)   : {mae('sim_q50'):.2f}")
    print("\nMAE by position (baseline | direct | two-stage):")
    for pos, sub in res.groupby("position"):
        f = lambda c: np.mean(np.abs(sub[TARGET] - sub[c]))
        print(f"  {pos}: {f(BASELINE_COL):.2f} | {f('pred_direct_median'):.2f} | {f('sim_q50'):.2f}  (n={len(sub):,})")
    print("\nCarries MAE: baseline "
          f"{np.mean(np.abs(res['carries'] - res['ewm_carries'])):.2f} "
          f"| model {np.mean(np.abs(res['carries'] - res['mu_carries'])):.2f}")

    if "neg_actual" in res.columns:
        print("\nShare of games with negative rushing yards (the simulation can't produce these):")
        for pos, sub in res.groupby("position"):
            print(f"  {pos}: {sub['neg_actual'].mean():.1%}")
    print("\nBias by position (actual mean | simulated mean):")
    for pos, sub in res.groupby("position"):
        print(f"  {pos}: {sub[TARGET].mean():.2f} | {sub['sim_mean'].mean():.2f}")
    cal = res[res["disp_est"]]
    if len(cal):
        print(f"\nRandomized-PIT calibration on {len(cal):,} rows "
              "(expected: <25% 25 | 25-75 50 | >75% 25  ||  <10% 10 | 10-90 80 | >90% 10)")
        for label, sub in [("All", cal)] + list(cal.groupby("position")):
            p = sub["pit"]
            b25, a75, b10, a90 = (p < .25).mean(), (p > .75).mean(), (p < .10).mean(), (p > .90).mean()
            print(f"  {label}: {b25:.1%} | {1 - b25 - a75:.1%} | {a75:.1%}  ||  "
                  f"{b10:.1%} | {1 - b10 - a90:.1%} | {a90:.1%}")
    print("\nZero-yard games (actual vs predicted P(0)):")
    for pos, sub in res.groupby("position"):
        print(f"  {pos}: actual {(sub[TARGET] == 0).mean():.1%} | predicted {sub['p_zero'].mean():.1%}")
    print(f"\nMean yards: actual {res[TARGET].mean():.2f} | sim mean {res['sim_mean'].mean():.2f}")


# ------------------------------------------------------- upcoming week
def play_rates(inj: pd.DataFrame, pw: pd.DataFrame, t_max: int) -> dict:
    x = inj[inj["position"].isin(RUSH_POS + ["FB"]) & (inj["season"] * 100 + inj["week"] < t_max)].copy()
    dnp, lim = P.practice_flags(x)
    key = x["report_status"].where(x["report_status"].isin(["Questionable", "Doubtful"]))
    key = key.mask(key.isna() & dnp, "PracDNP").mask(key.isna() & lim, "PracLimited")
    x = x.assign(key=key).dropna(subset=["key"])[["player_id", "season", "week", "key"]]
    x = x.drop_duplicates(["player_id", "season", "week"])
    played = pw[["player_id", "season", "week"]].assign(played=1)
    x = x.merge(played, on=["player_id", "season", "week"], how="left")
    x["played"] = x["played"].fillna(0)
    return x.groupby("key")["played"].mean().to_dict()


def project_upcoming(train, fut, backtest, rates):
    rng = np.random.default_rng(P.SEED)
    models = fit_stage_models(train)
    p_pos, mu_pos, ypc = predict_stages(models, fut[FEATURES])
    k_by, s_by, _ = estimate_dispersion(backtest)
    print(f"Dispersion used: k (carries) {k_by}, s (yards per carry) {s_by}")
    k = fut["position"].map(k_by).to_numpy(dtype=float)
    s = fut["position"].map(s_by).to_numpy(dtype=float)
    sc_by, wd_by = estimate_adjust(backtest)
    print(f"Calibration adjustments: level scale {sc_by}, spread widen {wd_by}")
    sims, car = simulate(p_pos, mu_pos, ypc, k, s, rng, return_carries=True,
                         scale=fut["position"].map(sc_by).to_numpy(dtype=float),
                         widen=fut["position"].map(wd_by).to_numpy(dtype=float))
    q10, q25, q50, q75, q90 = np.percentile(sims, [10, 25, 50, 75, 90], axis=1)

    pmap = {"Out": 0.0, "Doubtful": rates.get("Doubtful", 0.10),
            "Questionable": rates.get("Questionable", 0.75),
            "PracDNP": rates.get("PracDNP", 0.50), "PracLimited": rates.get("PracLimited", 0.92)}
    out = fut[["player_id", "player_display_name", "team", "opp", "position", "status", "implied_total", "spread"]].copy()
    out["status"] = out["status"].fillna("")
    key = fut["status"].where(fut["status"].isin(["Out", "Doubtful", "Questionable"]))
    key = key.mask(key.isna() & (fut["prac_dnp"] == 1), "PracDNP").mask(
        key.isna() & (fut["prac_limited"] == 1), "PracLimited")
    out["p_active"] = key.map(pmap).fillna(1.0).to_numpy()
    out["practice"] = np.where(fut["prac_dnp"] == 1, "DNP", np.where(fut["prac_limited"] == 1, "Limited", ""))
    out["q10"], out["q25"], out["median"], out["q75"], out["q90"] = q10, q25, q50, q75, q90
    out["projection"] = out["median"]            # OUR PROJECTION = median outcome
    out["mean"] = sims.mean(axis=1)
    out["p_zero"] = (sims == 0).mean(axis=1)
    out["exp_carries"] = p_pos * (1 + mu_pos)
    out["carries_projection"] = np.median(car, axis=1)
    out["ypc"] = ypc
    out["model_version"] = MODEL_VERSION
    order = np.argsort(-out["mean"].to_numpy())
    return out.iloc[order].reset_index(drop=True), sims[order]


def build_email_part(out: pd.DataFrame) -> str:
    esc = _html.escape
    rows = []
    for r in out.head(30).itertuples():
        rows.append("<tr>" + P._cell(esc(str(r.player_display_name)), bold=True)
                    + P._cell(f"{esc(str(r.team))} vs {esc(str(r.opp))}")
                    + P._cell(esc(str(r.position)))
                    + P._cell(f"{r.projection:.0f}", bold=True, align="right")
                    + P._cell(f"{r.q25:.0f}-{r.q75:.0f}", align="right")
                    + P._cell(f"{r.mean:.0f}", align="right")
                    + P._cell(f"{r.exp_carries:.1f}", align="right")
                    + P._cell(f"{r.p_zero:.0%}", align="right")
                    + P._cell(esc(str(r.status or "")), align="left")
                    + P._cell(esc(str(getattr(r, "role_note", "") or "")), align="left") + "</tr>")
    return ("<h3 style='margin:18px 0 4px'>Rushing yards: OUR PROJECTION (RB and QB)</h3>"
            "<p style='margin:0 0 6px;color:#555;font-size:12px'>Median outcome, with the middle half "
            "of outcomes as the range. This model is new and has no live track record yet.</p>"
            '<table style="border-collapse:collapse;font-size:13px">'
            + P._head(["Player", "Game", "Pos", "OUR PROJECTION (rush yds)", "Typical range (25th-75th pct)",
                       "Average (mean)", "Expected carries", "Chance of 0 yds", "Status", "Role notes"])
            + "".join(rows) + "</table>")


# ------------------------------------------------------------- main
def main():
    season, week = P.PROJECT_SEASON, P.PROJECT_WEEK
    if season is None or week is None:
        try:
            season, week = P.detect_target_week()
        except ValueError as e:
            print(e)
            return
    out_dir = P.resolve_out_dir()
    print(f"Rushing model {MODEL_VERSION}: target {season} week {week}; saving to {os.path.abspath(out_dir)}\n")

    played, parts, inj = build_rush_table(project=(season, week))
    played["is_future"] = False
    fut = make_future_rows(played, parts, inj, season, week)
    full = pd.concat([played, fut], ignore_index=True, sort=False)
    full = full.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    full = add_features(full)
    hist = full[~full["is_future"]].copy()
    fut = full[full["is_future"]].copy()

    bt_path = os.path.join(out_dir, BACKTEST_CACHE)
    if os.path.exists(bt_path) and not REFRESH_BACKTEST:
        print(f"Using cached backtest: {bt_path}")
        backtest = pd.read_csv(bt_path)
    else:
        backtest = walk_forward(hist)
        report(backtest)
        backtest.to_csv(bt_path, index=False)

    scale = 100.0 if hist["offense_pct"].max() > 1.5 else 1.0
    fut = fut.dropna(subset=BASE_FEATURES)
    min_snap = np.where(fut["position"] == "QB", MIN_SNAP_QB, MIN_SNAP_RB) * scale
    fut = fut[(fut["games_prior"] >= P.MIN_PRIOR_GAMES) & (fut["ewm_offense_pct"].to_numpy() >= min_snap)]
    out_names = sorted(fut.loc[fut["status"] == "Out", "player_display_name"])
    fut = fut[fut["status"] != "Out"]
    if out_names:
        print(f"Excluded (listed Out): {out_names}")
    if fut.empty:
        print("No rushers to project (check the week, rosters and betting lines).")
        return

    train = hist.dropna(subset=BASE_FEATURES + [TARGET])
    train = train[train["games_prior"] >= P.MIN_PRIOR_GAMES]
    rates = play_rates(inj, parts["pw"], season * 100 + week)
    out, sims = project_upcoming(train, fut, backtest, rates)
    notes = role_news.get_notes(hist, fut, season, week)
    out = out.merge(notes, on="player_id", how="left")
    out["role_note"] = out["role_note"].fillna("")

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M")
    base = os.path.join(out_dir, f"rushing_{season}_wk{week}")
    out.to_csv(f"{base}_{stamp}.csv", index=False)
    out.to_csv(f"{base}_latest.csv", index=False)
    part = build_email_part(out)

    # sportsbook comparison (ParlayAPI; falls back to projections only if no key / no lines)
    pkey = lines_parlay.get_key()
    if pkey:
        try:
            plines = lines_parlay.get_lines(pkey, season, week, out_dir, window=P.week_window(season, week))
        except Exception as e:
            print(f"ParlayAPI lines skipped ({e})")
            plines = pd.DataFrame()
        if not len(plines):
            print("\nParlayAPI: no usable rushing lines yet (see diagnostics above); projections only.")
        if len(plines):
            html_edges, _ = P.stat_edges(out, sims, plines[plines["stat"] == "rush_yds"], "Rushing yards",
                                         "rushing", "yds", season, week, out_dir, stamp)
            part += html_edges
    else:
        print("\nNo PARLAY_API_KEY found: rushing line comparison skipped.")
    with open(os.path.join(out_dir, "rushing_email_part.html"), "w", encoding="utf-8") as f:
        f.write(part)

    pd.set_option("display.width", 220)
    cols = ["player_display_name", "team", "opp", "position", "status", "q25", "projection",
            "q75", "mean", "exp_carries", "p_zero"]
    labels = {"player_display_name": "player", "projection": "OUR_PROJECTION", "q25": "range_low(25%)",
              "q75": "range_high(75%)", "mean": "average", "p_zero": "chance_0_yds"}
    print(f"\nTop rushing projections, {season} week {week} ({len(out)} players):")
    print(out[cols].head(30).round({"q25": 0, "projection": 0, "q75": 0, "mean": 1,
                                    "exp_carries": 1, "p_zero": 2}).rename(columns=labels).to_string(index=False))
    print(f"\nSaved {base}_latest.csv")


if __name__ == "__main__":
    main()
