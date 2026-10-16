"""
Receiving yards projection v6: upcoming-week projections + sportsbook line comparison.

Run it and it auto-detects the next unplayed NFL week, builds a row for every
relevant WR/TE, and simulates a yardage distribution for each. Results print and
save to projections_<season>_wk<week>.csv. See the config block below for manual
injury overrides.

Sportsbook lines (optional): get a free key at the-odds-api.com and add it as a
Colab Secret named ODDS_API_KEY (or set the ODDS_API_KEY environment variable).
The script then pulls receiving-yards props for that week's games, compares them to
the simulated distributions, and saves an edges file next to the projections.
Outputs go to NFL_OUTPUT_DIR if that environment variable is set (the GitHub workflow
uses data/), else to your Google Drive folder (DRIVE_DIR) when Drive is mounted.

Stage 1: targeted?      ~ Bernoulli(p from LightGBM classifier)   [the zero hurdle]
         targets        ~ 1 + NegativeBinomial(mean from LightGBM-Poisson, dispersion k)
                          (only for players who get targeted at all)
Stage 2: receptions     ~ Binomial(targets, catch prob from LightGBM)
         yards          ~ sum of per-catch yards ~ Gamma(shape = s * receptions,
                          scale = predicted yards-per-reception / s)
Simulating this chain gives a full distribution per player-week, including the
pile of exact-zero games, so ranges and P(over/under) come out naturally.

Self-contained (includes the v2 data pipeline).

Setup:
    pip install nflreadpy pandas numpy lightgbm scikit-learn
    (also needs lines_parlay.py next to this file)
"""
import html as _html
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import boom
import lines_parlay
import nflreadpy as nfl
import lightgbm as lgb

CURRENT_SEASON = 2026
SEASONS = list(range(2018, CURRENT_SEASON + 1))
TEST_SEASONS = [2024, 2025]
TARGET = "receiving_yards"
POSITIONS = ["WR", "TE"]
CATCHER_POS = ["WR", "TE", "RB"]
HALFLIFE = 3
MIN_PRIOR_GAMES = 3
MEDIAN_WINDOW = 6

# ---- Upcoming-week projection settings ----
PROJECT_SEASON = None       # None = auto-detect the next unplayed week from the schedule
PROJECT_WEEK = None
MANUAL_OUT = []             # display names to treat as Out this week, e.g. ["Player Name"]
MANUAL_QUESTIONABLE = []    # display names to treat as Questionable
PROP_LINES = {}             # {"Player Name": 64.5} adds P(over) / P(under) columns
MIN_SNAP_PCT = 0.20         # skip players whose recent snap share is below this fraction
REFRESH_BACKTEST = False    # True = re-run the walk-forward backtest instead of using the cache
BACKTEST_CACHE = "receiving_backtest_v6b.csv"   # v6b adds receptions columns

# ---- Opponent defense-vs-position ----
USE_OPP_FEATURES = False    # flip to True only after ab_opponent.py shows a real gain
OPP_HALFLIFE = 6            # games; decay for a defense's recent form

# ---- Practice participation + weather (off until ab_features.py shows a real gain) ----
USE_PRACTICE_FEATURES = False
USE_WEATHER_FEATURES = False
USE_NGS_FEATURES = False        # NFL Next Gen Stats (separation etc.); flip only if ab_features says it helps

# ---- Saving + sportsbook lines ----
DRIVE_ROOT = "/content/drive/MyDrive"                 # where Colab mounts Drive
DRIVE_DIR = DRIVE_ROOT + "/nfl_projections"           # outputs land here when Drive is mounted
ODDS_MARKET = "player_reception_yds"                  # receiving yards (the only stat this model simulates)
ODDS_REGION = "us"
ODDS_BOOKS = None                                     # e.g. ["draftkings", "fanduel"]; None = all in region
LINES_CACHE_MINUTES = 60                              # reuse a recent pull to save API credits
FORCE_REFRESH_LINES = False

N_SIMS = 20000
SEED = 42
MIN_CAL_ROWS = 300      # past out-of-sample rows (per position) before dispersion is estimated
K_DEFAULT = 4.0         # NB dispersion until enough history (larger = closer to Poisson)
S_DEFAULT = 1.2         # gamma shape for per-catch yards until enough history

PARAMS = dict(
    num_leaves=15, min_child_samples=50, n_estimators=300, learning_rate=0.03,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
)

STAT_COLS = [
    "targets", "receptions", "receiving_yards",
    "target_share", "air_yards_share", "receiving_air_yards",
]
FORM_COLS = STAT_COLS + ["offense_pct"]

TEAM_FIX = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "JAC": "JAX",
    "LAR": "LA", "STL": "LA", "SL": "LA", "OAK": "LV", "SD": "LAC", "WSH": "WAS",
}


def fix_team(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_FIX)


# ---------------------------------------------------------------- data
def load_context() -> pd.DataFrame:
    sched = nfl.load_schedules(SEASONS).to_pandas()
    sched = sched[sched["game_type"] == "REG"].copy()
    for c in ("roof", "temp", "wind", "location", "gametime", "gameday"):
        if c not in sched.columns:
            sched[c] = np.nan
    ko = pd.to_datetime(sched["gameday"].astype(str) + " " + sched["gametime"].astype(str),
                        errors="coerce")
    sched["kickoff_utc"] = ko.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    sched["venue_team"] = sched["home_team"]
    home = sched[["season", "week", "home_team", "away_team", "spread_line", "total_line",
                  "roof", "temp", "wind", "location", "kickoff_utc", "venue_team"]].copy()
    home.columns = ["season", "week", "team", "opp", "spread", "total",
                    "roof", "temp", "wind", "location", "kickoff_utc", "venue_team"]
    home["is_home"] = 1

    away = home.copy()
    away["team"], away["opp"] = home["opp"], home["team"]
    away["spread"] = -home["spread"]
    away["is_home"] = 0

    ctx = pd.concat([home, away], ignore_index=True)
    ctx["team"] = fix_team(ctx["team"])
    ctx["venue_team"] = fix_team(ctx["venue_team"])
    ctx["implied_total"] = (ctx["total"] + ctx["spread"]) / 2
    return ctx.drop(columns=["total"])


def load_player_weeks() -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = nfl.load_player_stats(SEASONS).to_pandas()
    stats = stats[stats["season_type"] == "REG"].copy()

    players = (nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]]
               .dropna().drop_duplicates("pfr_id"))
    snaps = nfl.load_snap_counts(SEASONS).to_pandas()
    snaps = snaps[(snaps["game_type"] == "REG")
                  & (snaps["position"].isin(POSITIONS))
                  & (snaps["offense_snaps"] > 0)]
    snaps = snaps.merge(players, left_on="pfr_player_id", right_on="pfr_id",
                        how="inner").rename(columns={"gsis_id": "player_id",
                                                     "player": "player_display_name"})
    snaps["team"] = fix_team(snaps["team"])
    snaps = (snaps[["player_id", "player_display_name", "season", "week",
                    "team", "position", "offense_snaps", "offense_pct"]]
             .drop_duplicates(["player_id", "season", "week"]))

    pw = snaps.merge(stats[["player_id", "season", "week"] + STAT_COLS],
                     on=["player_id", "season", "week"], how="left")
    pw[STAT_COLS] = pw[STAT_COLS].fillna(0)
    return pw, stats


def practice_flags(df: pd.DataFrame):
    """(did-not-participate, limited) booleans from the injury report's practice status."""
    if "practice_status" in df.columns:
        ps = df["practice_status"].astype(str).str.lower()
    else:
        ps = pd.Series("", index=df.index)
    return ps.str.contains("did not", na=False), ps.str.contains("limited", na=False)


def injury_features(stats: pd.DataFrame, project=None):
    inj = nfl.load_injuries(SEASONS).to_pandas()
    inj = inj[inj["game_type"] == "REG"].rename(columns={"gsis_id": "player_id"})
    inj["team"] = fix_team(inj["team"])
    inj["season"] = inj["season"].astype(int)
    inj["week"] = inj["week"].astype(int)

    # Manual overrides for the target week (e.g. before the official report is out)
    if project is not None and (MANUAL_OUT or MANUAL_QUESTIONABLE):
        latest = stats.sort_values(["season", "week"]).groupby("player_id").tail(1)
        extra = []
        for status, names in (("Out", MANUAL_OUT), ("Questionable", MANUAL_QUESTIONABLE)):
            for name in set(names) - set(latest["player_display_name"]):
                print(f"WARNING: manual injury name not found in stats: {name}")
            m = latest[latest["player_display_name"].isin(names)][
                ["player_id", "team", "position"]].copy()
            m["team"] = fix_team(m["team"])
            m["season"], m["week"] = project
            m["report_status"], m["game_type"] = status, "REG"
            extra.append(m)
        inj = pd.concat([inj] + extra, ignore_index=True)
        inj = inj.drop_duplicates(["player_id", "season", "week"], keep="last")

    form = stats[stats["position"].isin(CATCHER_POS)][
        ["player_id", "season", "week", "targets"]].copy()
    form = form.sort_values(["player_id", "season", "week"])
    form["form_targets"] = form.groupby("player_id")["targets"].transform(
        lambda s: s.ewm(halflife=HALFLIFE, min_periods=1).mean())
    form["t"] = form["season"].astype(int) * 100 + form["week"].astype(int)

    out = inj[(inj["report_status"] == "Out")
              & (inj["position"].isin(CATCHER_POS))].copy()
    out["t"] = out["season"] * 100 + out["week"]
    out = pd.merge_asof(
        out.sort_values("t"),
        form[["player_id", "t", "form_targets"]].sort_values("t"),
        on="t", by="player_id", allow_exact_matches=False, tolerance=100,
    )
    out["form_targets"] = out["form_targets"].fillna(0)

    team_out = (out.groupby(["team", "season", "week"])
                .agg(inj_out_targets=("form_targets", "sum"),
                     inj_n_out=("player_id", "count"))
                .reset_index())

    q = inj[inj["report_status"] == "Questionable"][
        ["player_id", "season", "week"]].drop_duplicates()
    q["self_questionable"] = 1

    dnp, lim = practice_flags(inj)
    prac = inj.assign(prac_dnp=dnp.astype(int), prac_limited=lim.astype(int))[
        ["player_id", "season", "week", "prac_dnp", "prac_limited"]]
    prac = prac[(prac["prac_dnp"] + prac["prac_limited"]) > 0].drop_duplicates(
        ["player_id", "season", "week"], keep="last")
    return team_out, q, prac, inj


def build_table(project=None):
    pw, stats = load_player_weeks()
    team_out, q, prac, inj = injury_features(stats, project)
    ctx = load_context()

    df = pw.merge(ctx, on=["season", "week", "team"], how="left")
    df = df.merge(team_out, on=["team", "season", "week"], how="left")
    df = df.merge(q, on=["player_id", "season", "week"], how="left")
    df = df.merge(prac, on=["player_id", "season", "week"], how="left")
    flag_cols = ["inj_out_targets", "inj_n_out", "self_questionable", "prac_dnp", "prac_limited"]
    df[flag_cols] = df[flag_cols].fillna(0)
    # Use nflverse display names everywhere (consistent with MANUAL_* and PROP_LINES)
    name_map = (stats.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
                .set_index("player_id")["player_display_name"])
    df["player_display_name"] = df["player_id"].map(name_map).fillna(df["player_display_name"])

    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    return df, dict(pw=pw, ctx=ctx, team_out=team_out, q=q, prac=prac), inj


# --------------------------------------------- upcoming-week rows
def detect_target_week():
    """Earliest week of the latest season that still has unplayed regular-season games."""
    sched = nfl.load_schedules(SEASONS).to_pandas()
    up = sched[(sched["game_type"] == "REG") & sched["home_score"].isna()]
    if up.empty:
        raise ValueError("No unplayed regular-season games found; "
                         "set PROJECT_SEASON and PROJECT_WEEK manually.")
    season = int(up["season"].max())
    return season, int(up[up["season"] == season]["week"].min())


def current_roster(season: int, week: int):
    """gsis_id -> current team from weekly rosters (handles trades/cuts). None if unavailable."""
    try:
        r = nfl.load_rosters_weekly(season).to_pandas()
        r = r[r["week"] <= week]
        if r.empty:
            raise ValueError("no roster rows yet")
    except Exception as e:
        print(f"Weekly rosters unavailable ({e}); falling back to each player's last team.")
        return None
    # Latest row per player FIRST, then keep only active/inactive (drops cut / IR / etc.)
    r = r.sort_values("week").groupby("gsis_id").tail(1)
    r = r[r["status"].isin(["ACT", "INA"])].copy()
    r["team"] = fix_team(r["team"])
    return r.rename(columns={"gsis_id": "player_id"})[["player_id", "team"]]


def play_rates(inj: pd.DataFrame, pw: pd.DataFrame, t_max: int) -> dict:
    """Historical share of WR/TEs who played, by report status. Players with a practice
    designation but no game status yet are keyed 'PracDNP' / 'PracLimited' (early-week runs)."""
    x = inj[inj["position"].isin(POSITIONS) & (inj["season"] * 100 + inj["week"] < t_max)].copy()
    dnp, lim = practice_flags(x)
    key = x["report_status"].where(x["report_status"].isin(["Questionable", "Doubtful"]))
    key = key.mask(key.isna() & dnp, "PracDNP").mask(key.isna() & lim, "PracLimited")
    x = x.assign(key=key).dropna(subset=["key"])[["player_id", "season", "week", "key"]]
    x = x.drop_duplicates(["player_id", "season", "week"])
    played = pw[["player_id", "season", "week"]].assign(played=1)
    x = x.merge(played, on=["player_id", "season", "week"], how="left")
    x["played"] = x["played"].fillna(0)
    return x.groupby("key")["played"].mean().to_dict()


# ---------------------------------------------------- weather (forecast for upcoming games)
# (lat, lon, roof type). Coordinates are approximate (from memory): fine for a weather grid.
STADIUMS = {
    "ARI": (33.5276, -112.2626, "retractable"), "ATL": (33.7554, -84.4008, "retractable"),
    "BAL": (39.2780, -76.6227, "outdoors"), "BUF": (42.7738, -78.7870, "outdoors"),
    "CAR": (35.2258, -80.8528, "outdoors"), "CHI": (41.8623, -87.6167, "outdoors"),
    "CIN": (39.0955, -84.5160, "outdoors"), "CLE": (41.5061, -81.6995, "outdoors"),
    "DAL": (32.7473, -97.0945, "retractable"), "DEN": (39.7439, -105.0201, "outdoors"),
    "DET": (42.3400, -83.0456, "dome"), "GB": (44.5013, -88.0622, "outdoors"),
    "HOU": (29.6847, -95.4107, "retractable"), "IND": (39.7601, -86.1639, "retractable"),
    "JAX": (30.3240, -81.6373, "outdoors"), "KC": (39.0489, -94.4839, "outdoors"),
    "LV": (36.0909, -115.1833, "dome"), "LAC": (33.9535, -118.3390, "dome"),
    "LA": (33.9535, -118.3390, "dome"), "MIA": (25.9580, -80.2389, "outdoors"),
    "MIN": (44.9738, -93.2575, "dome"), "NE": (42.0909, -71.2643, "outdoors"),
    "NO": (29.9511, -90.0812, "dome"), "NYG": (40.8135, -74.0745, "outdoors"),
    "NYJ": (40.8135, -74.0745, "outdoors"), "PHI": (39.9008, -75.1675, "outdoors"),
    "PIT": (40.4468, -80.0158, "outdoors"), "SEA": (47.5952, -122.3316, "outdoors"),
    "SF": (37.4030, -121.9700, "outdoors"), "TB": (27.9759, -82.5033, "outdoors"),
    "TEN": (36.1665, -86.7713, "outdoors"), "WAS": (38.9076, -76.8645, "outdoors"),
}


def effective_roof(roof, venue) -> str:
    """Schedule roof if given; otherwise the stadium default (retractable assumed closed)."""
    r = str(roof).lower()
    if r in ("dome", "closed", "open", "outdoors"):
        return r
    kind = STADIUMS.get(venue, (None, None, None))[2]
    return {"dome": "dome", "retractable": "closed", "outdoors": "outdoors"}.get(kind, "unknown")


def fetch_forecast(lat: float, lon: float, kickoff: pd.Timestamp):
    """(temp F, wind mph) averaged over the first ~3 hours of the game, from Open-Meteo."""
    import requests
    d0 = kickoff.strftime("%Y-%m-%d")
    d1 = (kickoff + pd.Timedelta(hours=4)).strftime("%Y-%m-%d")
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=dict(
        latitude=lat, longitude=lon, hourly="temperature_2m,wind_speed_10m",
        temperature_unit="fahrenheit", wind_speed_unit="mph", timezone="UTC",
        start_date=d0, end_date=d1), timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]
    t = pd.to_datetime(pd.Series(h["time"]), utc=True)
    start = kickoff.floor("h")
    m = ((t >= start) & (t <= start + pd.Timedelta(hours=3))).to_numpy()
    if not m.any():
        return None
    return (float(np.mean(np.array(h["temperature_2m"], dtype=float)[m])),
            float(np.mean(np.array(h["wind_speed_10m"], dtype=float)[m])))


def apply_forecast_weather(cand: pd.DataFrame) -> pd.DataFrame:
    """Fill temp/wind for upcoming outdoor games with forecasts (dome/closed stay neutral)."""
    cand = cand.copy()
    for c in ("temp", "wind"):
        cand[c] = pd.to_numeric(cand[c], errors="coerce")
    cand["roof"] = [effective_roof(r, v) for r, v in zip(cand["roof"], cand["venue_team"])]
    home_game = cand["location"].fillna("Home").astype(str).str.lower().eq("home")
    need = (cand["roof"].isin(["open", "outdoors"]) & cand["kickoff_utc"].notna()
            & cand["venue_team"].isin(STADIUMS) & cand["temp"].isna() & cand["wind"].isna()
            & home_game)
    games = cand.loc[need, ["venue_team", "kickoff_utc"]].drop_duplicates()
    notable = []
    for venue, ko in games.itertuples(index=False):
        try:
            res = fetch_forecast(STADIUMS[venue][0], STADIUMS[venue][1], ko)
        except Exception as e:
            print(f"Weather forecast failed for {venue}: {e}")
            continue
        if res is None:
            continue
        temp_f, wind_mph = res
        m = (cand["venue_team"] == venue) & (cand["kickoff_utc"] == ko)
        cand.loc[m, "temp"], cand.loc[m, "wind"] = temp_f, wind_mph
        if wind_mph >= 15 or temp_f <= 32:
            notable.append(f"{venue} ({wind_mph:.0f} mph wind, {temp_f:.0f}F)")
    print(f"Weather forecasts applied to {len(games)} outdoor games."
          + (f" Notable: {', '.join(notable)}" if notable else ""))
    return cand


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Game-day weather: wind, temperature, dome flag. Domes/closed roofs are neutral (0 mph, 70F).
    History uses recorded conditions; upcoming games use the forecast."""
    roof = df["roof"].astype(str).str.lower()
    dome = roof.isin(["dome", "closed"])
    df["wx_is_dome"] = dome.astype(int)
    df["wx_wind_mph"] = np.where(dome, 0.0, pd.to_numeric(df["wind"], errors="coerce"))
    df["wx_temp_f"] = np.where(dome, 70.0, pd.to_numeric(df["temp"], errors="coerce"))
    return df


def make_future_rows(played, parts, inj, season: int, week: int) -> pd.DataFrame:
    """One candidate row per WR/TE for the target week: outcome unknown, context known."""
    pw, ctx = parts["pw"], parts["ctx"]
    last = (played.sort_values(["season", "week"]).groupby("player_id").tail(1)
            [["player_id", "player_display_name", "position", "team", "season", "week"]]
            .rename(columns={"team": "last_team", "season": "last_season",
                             "week": "last_week"}))

    roster = current_roster(season, week)
    if roster is not None:
        cand = last.merge(roster, on="player_id", how="inner")
    else:   # fallback: last team played for; keep players seen since late last season
        cand = last[last["last_season"] * 100 + last["last_week"]
                    >= (season - 1) * 100 + 15].copy()
        cand["team"] = cand["last_team"]
    cand = cand.drop(columns=["last_team", "last_season", "last_week"])
    cand["season"], cand["week"] = season, week

    # Game context for that week; the inner merge drops teams on a bye
    c = ctx[(ctx["season"] == season) & (ctx["week"] == week)]
    cand = cand.merge(c, on=["season", "week", "team"], how="inner")

    # Skip anyone whose game already happened (e.g. Thursday night)
    done = pw[["player_id", "season", "week"]].assign(_done=1)
    cand = cand.merge(done, on=["player_id", "season", "week"], how="left")
    cand = cand[cand["_done"].isna()].drop(columns="_done")

    cand = cand.merge(parts["team_out"], on=["team", "season", "week"], how="left")
    cand = cand.merge(parts["q"], on=["player_id", "season", "week"], how="left")
    cand = cand.merge(parts["prac"], on=["player_id", "season", "week"], how="left")
    cols = ["inj_out_targets", "inj_n_out", "self_questionable", "prac_dnp", "prac_limited"]
    cand[cols] = cand[cols].fillna(0)

    st = (inj[(inj["season"] == season) & (inj["week"] == week)]
          .drop_duplicates("player_id")[["player_id", "report_status"]]
          .rename(columns={"report_status": "status"}))
    cand = cand.merge(st, on="player_id", how="left")
    cand = apply_forecast_weather(cand)
    cand["is_future"] = True
    return cand


# ----------------------------------------------------------- features
BASELINE_COL = f"ewm_{TARGET}"
MEDIAN_COL = "med_receiving_yards"

FEATURES = (
    [f"ewm_{c}" for c in FORM_COLS]
    + ["games_prior", "implied_total", "spread", "is_home",
       "inj_out_targets", "inj_n_out", "self_questionable", "is_te"]
)
# Ratio features can be NaN (e.g. no prior targets); LightGBM handles NaN natively
RATIO_FEATURES = ["ewm_catch_rate", "ewm_ypr", "ewm_ypt"]
OPP_FEATURES = ["opp_tgt_oe", "opp_yds_oe"]
PRAC_FEATURES = ["prac_dnp", "prac_limited"]
WX_FEATURES = ["wx_wind_mph", "wx_temp_f", "wx_is_dome"]
NGS_COLS = ["avg_separation", "avg_cushion", "avg_intended_air_yards",
            "percent_share_of_intended_air_yards", "avg_yac_above_expectation"]
NGS_FEATURES = [f"ngs_{c}" for c in NGS_COLS]
NGS_URL = "https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_receiving.csv.gz"


def build_model_features(opp: bool = False, prac: bool = False, wx: bool = False, ngs: bool = False) -> list:
    return (FEATURES + RATIO_FEATURES + (OPP_FEATURES if opp else [])
            + (PRAC_FEATURES if prac else []) + (WX_FEATURES if wx else [])
            + (NGS_FEATURES if ngs else []))


MODEL_FEATURES = build_model_features(USE_OPP_FEATURES, USE_PRACTICE_FEATURES, USE_WEATHER_FEATURES,
                                      USE_NGS_FEATURES)
_tags = [n for n, on in (("opp", USE_OPP_FEATURES), ("prac", USE_PRACTICE_FEATURES),
                         ("wx", USE_WEATHER_FEATURES), ("ngs", USE_NGS_FEATURES)) if on]
MODEL_VERSION = "v7-" + "-".join(_tags) if _tags else "v6"


def add_ngs_features(df: pd.DataFrame) -> pd.DataFrame:
    """Next Gen Stats receiving tracking (separation, cushion, air yards, share of air yards, YAC over expected).
    Each is a recency-weighted average over the player's PRIOR games that have NGS rows. NGS only lists a player
    in a week with at least 5 targets, so the average skips thin weeks; players with no history get NaN."""
    try:
        ngs = pd.read_csv(NGS_URL)
    except Exception as exc:                                 # network hiccup: features stay NaN, model still runs
        print(f"WARNING: could not load Next Gen Stats ({exc}); NGS features left empty.")
        for c in NGS_FEATURES:
            df[c] = np.nan
        return df
    ngs = ngs[(ngs["season_type"] == "REG") & (ngs["week"] > 0)].rename(columns={"player_gsis_id": "player_id"})
    ngs = ngs.sort_values(["player_id", "season", "week"]).copy()
    ngs["t"] = ngs["season"].astype(int) * 100 + ngs["week"].astype(int)
    g = ngs.groupby("player_id")
    for c in NGS_COLS:
        ngs[f"ngs_{c}"] = g[c].transform(lambda s: s.ewm(halflife=HALFLIFE, min_periods=1).mean())
    key = ngs[["player_id", "t"] + NGS_FEATURES].dropna(subset=["player_id"])
    base = df.drop(columns=[c for c in NGS_FEATURES if c in df.columns]).copy()
    base["_t"] = base["season"].astype(int) * 100 + base["week"].astype(int)
    base["_i"] = np.arange(len(base))
    m = pd.merge_asof(base.sort_values("_t"), key.sort_values("t"), left_on="_t", right_on="t",
                      by="player_id", allow_exact_matches=False, tolerance=300)     # strictly earlier games only
    m = m.sort_values("_i").drop(columns=["_t", "_i", "t"])
    m.index = df.index
    return m


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("player_id", group_keys=False)
    # shift(1) BEFORE smoothing so a game never sees its own outcome
    for col in FORM_COLS:
        df[f"ewm_{col}"] = g[col].apply(
            lambda s: s.shift(1).ewm(halflife=HALFLIFE, min_periods=1).mean())
    df["games_prior"] = g.cumcount()
    df[MEDIAN_COL] = g[TARGET].apply(
        lambda s: s.shift(1).rolling(MEDIAN_WINDOW, min_periods=1).median())
    df["is_te"] = (df["position"] == "TE").astype(int)

    def div(a, b):
        return (a / b).replace([np.inf, -np.inf], np.nan)

    df["ewm_catch_rate"] = div(df["ewm_receptions"], df["ewm_targets"])
    df["ewm_ypr"] = div(df["ewm_receiving_yards"], df["ewm_receptions"])
    df["ewm_ypt"] = div(df["ewm_receiving_yards"], df["ewm_targets"])
    return df


def add_opponent_features(df: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Defense-vs-position, adjusted for who each defense actually faced.

    For each player-game, measure how far the player landed above/below his OWN lagged
    baseline (targets and yards). Sum that by the defense faced and position group (WR/TE).
    A defense's feature is the lagged EWMA of those sums, i.e. the extra targets/yards it has
    been allowing versus expectation. Everything is shifted by one game, so a row only sees
    games that happened before it (including for the upcoming week).
    """
    hist = df[~df["is_future"] & (df["games_prior"] >= MIN_PRIOR_GAMES)].dropna(
        subset=["ewm_targets", BASELINE_COL, "opp"]).copy()
    hist["tgt_oe"] = hist["targets"] - hist["ewm_targets"]
    hist["yds_oe"] = hist[TARGET] - hist[BASELINE_COL]
    agg = (hist.groupby(["opp", "season", "week", "position"], as_index=False)[["tgt_oe", "yds_oe"]]
           .sum().rename(columns={"opp": "defense"}))

    # one row per defense-week-position, including the upcoming week (values NaN there)
    grid = (ctx[["season", "week", "opp"]].rename(columns={"opp": "defense"}).drop_duplicates()
            .merge(pd.DataFrame({"position": POSITIONS}), how="cross"))
    grid = grid.merge(agg, on=["defense", "season", "week", "position"], how="left")
    grid = grid.sort_values(["defense", "position", "season", "week"]).reset_index(drop=True)
    gb = grid.groupby(["defense", "position"], group_keys=False)
    grid["opp_tgt_oe"] = gb["tgt_oe"].apply(
        lambda s: s.shift(1).ewm(halflife=OPP_HALFLIFE, min_periods=1).mean())
    grid["opp_yds_oe"] = gb["yds_oe"].apply(
        lambda s: s.shift(1).ewm(halflife=OPP_HALFLIFE, min_periods=1).mean())

    keep = grid[["defense", "season", "week", "position"] + OPP_FEATURES].rename(
        columns={"defense": "opp"})
    df = df.drop(columns=[c for c in OPP_FEATURES if c in df.columns])   # safe to call twice
    return df.merge(keep, on=["opp", "season", "week", "position"], how="left")


# ---------------------------------------------------- two-stage pieces
def predict_prob(model, X) -> np.ndarray:
    p = model.predict(X)
    if p.min() < 0 or p.max() > 1:      # guard: some versions return raw scores
        p = 1 / (1 + np.exp(-p))
    return np.clip(p, 0.01, 0.99)


def estimate_dispersion(hist: pd.DataFrame):
    """Per-position NB dispersion k (targets) and gamma shape s (yards per catch),
    estimated by method of moments from PAST out-of-sample rows only."""
    k, s, est = {}, {}, {}
    for pos in POSITIONS:
        h = hist[hist["position"] == pos] if len(hist) else hist
        if len(h) < MIN_CAL_ROWS:
            k[pos], s[pos], est[pos] = K_DEFAULT, S_DEFAULT, False
            continue
        # Among targeted players: (targets - 1) ~ NB(mean mu_pos, k), Var = mu + mu^2 / k
        hp = h[h["targets"] > 0]
        y = hp["targets"] - 1
        excess = np.mean((y - hp["mu_pos"]) ** 2) - np.mean(hp["mu_pos"])
        k[pos] = float(np.clip(np.mean(hp["mu_pos"] ** 2) / max(excess, 1e-6), 1.0, 100.0))
        # Var(yards | R catches) = R * m^2 / s
        hr = h[h["receptions"] > 0]
        inv_s = np.mean((hr[TARGET] - hr["receptions"] * hr["ypr_pred"]) ** 2
                        / (hr["receptions"] * hr["ypr_pred"] ** 2))
        s[pos] = float(np.clip(1.0 / inv_s, 0.3, 10.0))
        est[pos] = True
    return k, s, est


def simulate(p_pos, mu_pos, p_catch, ypr, k, s, rng, return_catches: bool = False):
    """Returns (n_rows, N_SIMS) simulated receiving yards (hurdle for targets).
    With return_catches=True also returns the simulated receptions (same draws)."""
    shp = (len(mu_pos), N_SIMS)
    mu_c, k_c, s_c = mu_pos[:, None], k[:, None], s[:, None]
    targeted = rng.random(shp) < p_pos[:, None]
    extra = rng.negative_binomial(k_c, k_c / (k_c + mu_c), size=shp)
    targets = np.where(targeted, 1 + extra, 0)
    catches = rng.binomial(targets, p_catch[:, None])
    yards = rng.gamma(np.maximum(s_c * catches, 1e-9), (ypr[:, None] / s_c), size=shp)
    yards = np.where(catches > 0, yards, 0.0)
    return (yards, catches) if return_catches else yards


# ---------------------------------------------------------- modeling
def walk_forward(df: pd.DataFrame):
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    df = df[df["games_prior"] >= MIN_PRIOR_GAMES]
    rng = np.random.default_rng(SEED)

    preds, m_targets = [], None
    test = df[df["season"].isin(TEST_SEASONS)]
    for (season, week), wk in test.groupby(["season", "week"]):
        train = df[(df["season"] < season)
                   | ((df["season"] == season) & (df["week"] < week))]
        Xw = wk[MODEL_FEATURES]

        # Stage 1a: P(player is targeted at all)
        m_hit = lgb.LGBMClassifier(**PARAMS).fit(
            train[MODEL_FEATURES], (train["targets"] > 0).astype(int))

        # Stage 1b: target count among targeted players, shifted by 1 (Poisson mean)
        tr_p = train[train["targets"] > 0]
        m_targets = lgb.LGBMRegressor(objective="poisson", **PARAMS).fit(
            tr_p[MODEL_FEATURES], tr_p["targets"] - 1)

        # Stage 2a: catch probability (weighted by targets)
        tr_t = train[train["targets"] > 0]
        m_catch = lgb.LGBMRegressor(objective="cross_entropy", **PARAMS).fit(
            tr_t[MODEL_FEATURES],
            np.clip(tr_t["receptions"] / tr_t["targets"], 0, 1),
            sample_weight=tr_t["targets"])

        # Stage 2b: yards per reception (weighted by receptions)
        tr_r = train[train["receptions"] > 0]
        m_ypr = lgb.LGBMRegressor(objective="regression", **PARAMS).fit(
            tr_r[MODEL_FEATURES], tr_r[TARGET] / tr_r["receptions"],
            sample_weight=tr_r["receptions"])

        # Direct median model, for a fair head-to-head comparison
        m_direct = lgb.LGBMRegressor(objective="quantile", alpha=0.5).fit(
            train[MODEL_FEATURES], train[TARGET])

        p_pos = np.clip(m_hit.predict_proba(Xw)[:, 1], 0.01, 0.99)
        mu_pos = np.clip(m_targets.predict(Xw), 0.05, None)
        p_catch = predict_prob(m_catch, Xw)
        ypr = np.clip(m_ypr.predict(Xw), 1.0, None)

        # Dispersion from earlier weeks' out-of-sample results only
        hist = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
        k_by, s_by, est_by = estimate_dispersion(hist)
        k = wk["position"].map(k_by).to_numpy(dtype=float)
        s = wk["position"].map(s_by).to_numpy(dtype=float)

        sims, rec_sims = simulate(p_pos, mu_pos, p_catch, ypr, k, s, rng, return_catches=True)
        q10, q25, q50, q75, q90 = np.percentile(sims, [10, 25, 50, 75, 90], axis=1)

        # Randomized PIT: handles the point mass at zero (and integer-ish outcomes)
        y_act = wk[TARGET].to_numpy()[:, None]
        pit_lo = (sims < y_act).mean(axis=1)
        pit_hi = (sims <= y_act).mean(axis=1)

        out = wk[["player_id", "player_display_name", "season", "week", "position",
                  TARGET, "targets", "receptions", BASELINE_COL, MEDIAN_COL,
                  "ewm_targets", "ewm_receptions"]].copy()
        out["p_pos"], out["mu_pos"] = p_pos, mu_pos
        out["mu_targets"] = p_pos * (1 + mu_pos)       # expected targets
        out["p_catch"], out["ypr_pred"] = p_catch, ypr
        out["pred_direct_median"] = m_direct.predict(Xw)
        out["sim_mean"] = sims.mean(axis=1)
        out["sim_q10"], out["sim_q25"], out["sim_q50"] = q10, q25, q50
        out["sim_q75"], out["sim_q90"] = q75, q90
        out["p_zero"] = (sims == 0).mean(axis=1)
        out["sim_rec_mean"] = rec_sims.mean(axis=1)
        out["sim_rec_q50"] = np.median(rec_sims, axis=1)
        out["sim_rec_q25"] = np.percentile(rec_sims, 25, axis=1)
        out["sim_rec_q75"] = np.percentile(rec_sims, 75, axis=1)
        out["pit"] = pit_lo + rng.random(len(out)) * (pit_hi - pit_lo)
        out["disp_est"] = wk["position"].map(est_by).to_numpy(dtype=bool)
        preds.append(out)

    return pd.concat(preds, ignore_index=True), m_targets


def report(res: pd.DataFrame, m_targets) -> None:
    mae = lambda c: np.mean(np.abs(res[TARGET] - res[c]))
    rmse = lambda c: np.sqrt(np.mean((res[TARGET] - res[c]) ** 2))

    print(f"Rows evaluated: {len(res):,}")
    print("\nYards MAE:")
    print(f"  EWMA mean baseline       : {mae(BASELINE_COL):.2f}")
    print(f"  Rolling median baseline  : {mae(MEDIAN_COL):.2f}")
    print(f"  Direct quantile-median   : {mae('pred_direct_median'):.2f}")
    print(f"  Two-stage (sim median)   : {mae('sim_q50'):.2f}")
    print(f"Yards RMSE: baseline mean {rmse(BASELINE_COL):.2f} "
          f"| two-stage mean {rmse('sim_mean'):.2f}")

    print("\nMAE by position (baseline | direct | two-stage):")
    for pos, sub in res.groupby("position"):
        f = lambda c: np.mean(np.abs(sub[TARGET] - sub[c]))
        print(f"  {pos}: {f(BASELINE_COL):.2f} | {f('pred_direct_median'):.2f} "
              f"| {f('sim_q50'):.2f}")

    print("\nStage 1 (targets) MAE: "
          f"baseline {np.mean(np.abs(res['targets'] - res['ewm_targets'])):.2f} "
          f"| model {np.mean(np.abs(res['targets'] - res['mu_targets'])):.2f}")

    cal = res[res["disp_est"]]
    print(f"\nRandomized-PIT calibration on {len(cal):,} rows with estimated dispersion")
    print("(expected: <25% 25 | 25-75 50 | >75% 25  ||  <10% 10 | 10-90 80 | >90% 10)")

    def pit_summary(sub):
        p = sub["pit"]
        b25, a75 = (p < 0.25).mean(), (p > 0.75).mean()
        b10, a90 = (p < 0.10).mean(), (p > 0.90).mean()
        return (f"{b25:.1%} | {1 - b25 - a75:.1%} | {a75:.1%}  ||  "
                f"{b10:.1%} | {1 - b10 - a90:.1%} | {a90:.1%}")

    print(f"  All: {pit_summary(cal)}")
    for pos, sub in cal.groupby("position"):
        print(f"  {pos}: {pit_summary(sub)}")
    deciles = np.histogram(cal["pit"], bins=10, range=(0, 1))[0] / len(cal)
    print("  PIT decile shares (expected 10% each): "
          + " ".join(f"{d:.0%}" for d in deciles))

    if "sim_rec_q50" in res.columns:
        r = res.dropna(subset=["sim_rec_q50"])
        act = r["receptions"]
        base = r["ewm_receptions"] if "ewm_receptions" in r else None
        print(f"\nReceptions on {len(r):,} rows:")
        if base is not None:
            print(f"  MAE: EWMA baseline {np.mean(np.abs(act - base)):.3f} "
                  f"| two-stage median {np.mean(np.abs(act - r['sim_rec_q50'])):.3f} "
                  f"| two-stage mean {np.mean(np.abs(act - r['sim_rec_mean'])):.3f}")
        print(f"  inside q25-q75 {((act >= r['sim_rec_q25']) & (act <= r['sim_rec_q75'])).mean():.1%} "
              f"(integer outcomes make this run above 50%) | mean actual {act.mean():.2f} "
              f"vs predicted {r['sim_rec_mean'].mean():.2f}")

    print("\nZero-yard games (actual share vs mean predicted P(0)):")
    for pos, sub in res.groupby("position"):
        print(f"  {pos}: actual {(sub[TARGET] == 0).mean():.1%} "
              f"| predicted {sub['p_zero'].mean():.1%}")
    print("\nP(targeted at least once): actual vs predicted")
    for pos, sub in res.groupby("position"):
        print(f"  {pos}: actual {(sub['targets'] > 0).mean():.1%} "
              f"| predicted {sub['p_pos'].mean():.1%}")
    print(f"\nMean yards: actual {res[TARGET].mean():.2f} | sim mean {res['sim_mean'].mean():.2f}")

    imp = pd.Series(m_targets.booster_.feature_importance("gain"), index=MODEL_FEATURES)
    print("\nStage 1b (target count among targeted players) feature importance, last model:")
    print((imp / imp.sum()).sort_values(ascending=False).round(3).to_string())


# ------------------------------------------- upcoming-week model
def fit_stage_models(train: pd.DataFrame):
    """Fit the four stage models on all completed games."""
    m_hit = lgb.LGBMClassifier(**PARAMS).fit(
        train[MODEL_FEATURES], (train["targets"] > 0).astype(int))
    tr_p = train[train["targets"] > 0]
    m_targets = lgb.LGBMRegressor(objective="poisson", **PARAMS).fit(
        tr_p[MODEL_FEATURES], tr_p["targets"] - 1)
    m_catch = lgb.LGBMRegressor(objective="cross_entropy", **PARAMS).fit(
        tr_p[MODEL_FEATURES], np.clip(tr_p["receptions"] / tr_p["targets"], 0, 1),
        sample_weight=tr_p["targets"])
    tr_r = train[train["receptions"] > 0]
    m_ypr = lgb.LGBMRegressor(objective="regression", **PARAMS).fit(
        tr_r[MODEL_FEATURES], tr_r[TARGET] / tr_r["receptions"],
        sample_weight=tr_r["receptions"])
    return m_hit, m_targets, m_catch, m_ypr


def predict_stages(models, X: pd.DataFrame):
    m_hit, m_targets, m_catch, m_ypr = models
    p_pos = np.clip(m_hit.predict_proba(X)[:, 1], 0.01, 0.99)
    mu_pos = np.clip(m_targets.predict(X), 0.05, None)
    p_catch = predict_prob(m_catch, X)
    ypr = np.clip(m_ypr.predict(X), 1.0, None)
    return p_pos, mu_pos, p_catch, ypr


def project_upcoming(train, fut, backtest, rates) -> tuple:
    rng = np.random.default_rng(SEED)
    models = fit_stage_models(train)
    p_pos, mu_pos, p_catch, ypr = predict_stages(models, fut[MODEL_FEATURES])

    # Dispersion from the walk-forward backtest's out-of-sample results
    k_by, s_by, _ = estimate_dispersion(backtest)
    print(f"Dispersion used: k (targets) {k_by}, s (yards per catch) {s_by}")
    k = fut["position"].map(k_by).to_numpy(dtype=float)
    s = fut["position"].map(s_by).to_numpy(dtype=float)

    sims, rec_sims = simulate(p_pos, mu_pos, p_catch, ypr, k, s, rng, return_catches=True)
    q10, q25, q50, q75, q90 = np.percentile(sims, [10, 25, 50, 75, 90], axis=1)

    pmap = {"Out": 0.0,
            "Doubtful": rates.get("Doubtful", 0.10),
            "Questionable": rates.get("Questionable", 0.75),
            "PracDNP": rates.get("PracDNP", 0.50),
            "PracLimited": rates.get("PracLimited", 0.92)}
    out = fut[["player_display_name", "team", "opp", "position", "status",
               "implied_total"]].copy()
    out["status"] = out["status"].fillna("")
    key = fut["status"].where(fut["status"].isin(["Out", "Doubtful", "Questionable"]))
    key = key.mask(key.isna() & (fut["prac_dnp"] == 1), "PracDNP").mask(
        key.isna() & (fut["prac_limited"] == 1), "PracLimited")
    out["p_active"] = key.map(pmap).fillna(1.0).to_numpy()
    out["practice"] = np.where(fut["prac_dnp"] == 1, "DNP",
                               np.where(fut["prac_limited"] == 1, "Limited", ""))
    out["wind_mph"], out["temp_f"] = fut["wx_wind_mph"].to_numpy(), fut["wx_temp_f"].to_numpy()
    out["q10"], out["q25"], out["median"], out["q75"], out["q90"] = q10, q25, q50, q75, q90
    out["projection"] = out["median"]          # OUR PROJECTION = the median outcome (same as `median`)
    out["mean"] = sims.mean(axis=1)
    out["p_zero"] = (sims == 0).mean(axis=1)
    out["rec_projection"] = np.median(rec_sims, axis=1)       # OUR receptions projection (median)
    out["rec_mean"] = rec_sims.mean(axis=1)
    out["rec_q25"] = np.percentile(rec_sims, 25, axis=1)
    out["rec_q75"] = np.percentile(rec_sims, 75, axis=1)
    out["rec_p_zero"] = (rec_sims == 0).mean(axis=1)
    out["exp_targets"] = p_pos * (1 + mu_pos)
    out["p_catch"], out["ypr"] = p_catch, ypr
    out["model_version"] = MODEL_VERSION
    for c in OPP_FEATURES:
        out[c] = fut[c].to_numpy()

    if PROP_LINES:   # probabilities are conditional on the player playing
        line = fut["player_display_name"].map(PROP_LINES).to_numpy(dtype=float)
        has = ~np.isnan(line)
        out["line"] = line
        out["p_over"] = np.where(has, (sims > line[:, None]).mean(axis=1), np.nan)
        out["p_under"] = np.where(has, (sims < line[:, None]).mean(axis=1), np.nan)
        unmatched = set(PROP_LINES) - set(fut["player_display_name"])
        if unmatched:
            print(f"WARNING: PROP_LINES names not matched to any projected player: {sorted(unmatched)}")

    order = np.argsort(-out["mean"].to_numpy())
    return out.iloc[order].reset_index(drop=True), sims[order], rec_sims[order]


# ------------------------------------------------ sportsbook lines
ODDS_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def resolve_out_dir() -> str:
    """NFL_OUTPUT_DIR (GitHub Actions) > mounted Google Drive (Colab) > current directory."""
    env = os.environ.get("NFL_OUTPUT_DIR")
    if env:
        os.makedirs(env, exist_ok=True)
        return env
    if os.path.isdir(DRIVE_ROOT):
        os.makedirs(DRIVE_DIR, exist_ok=True)
        return DRIVE_DIR
    print("Google Drive is not mounted; saving to the current directory instead.")
    return "."


def get_api_key():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        try:
            from google.colab import userdata
            key = userdata.get("ODDS_API_KEY")
        except Exception:
            key = None
    return key


def name_key(name) -> str:
    """Normalize player names so 'Amon-Ra St. Brown' / 'Luther Burden III' match across sources."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`\-]", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(p for p in s.split() if p not in _SUFFIXES)


def week_window(season: int, week: int):
    """UTC window covering every game of the target week (from schedule game dates)."""
    sched = nfl.load_schedules([season]).to_pandas()
    g = sched[(sched["season"] == season) & (sched["week"] == week)
              & (sched["game_type"] == "REG")]
    days = pd.to_datetime(g["gameday"])
    return (days.min().tz_localize("UTC"),
            days.max().tz_localize("UTC") + pd.Timedelta(hours=30))


def fetch_lines(api_key: str, start, end) -> pd.DataFrame:
    """One row per (game, book, player) with the over/under line and prices."""
    import requests
    r = requests.get(f"{ODDS_BASE}/events", params={"apiKey": api_key}, timeout=30)
    r.raise_for_status()
    now = pd.Timestamp.now(tz="UTC")
    events = [e for e in r.json()
              if max(start, now) <= pd.Timestamp(e["commence_time"]) <= end]
    print(f"Fetching {ODDS_MARKET} for {len(events)} upcoming games "
          f"(roughly {len(events)} API credits)...")

    rows, remaining = [], None
    for ev in events:
        params = dict(apiKey=api_key, markets=ODDS_MARKET, oddsFormat="american")
        if ODDS_BOOKS:
            params["bookmakers"] = ",".join(ODDS_BOOKS)
        else:
            params["regions"] = ODDS_REGION
        resp = requests.get(f"{ODDS_BASE}/events/{ev['id']}/odds", params=params, timeout=30)
        if resp.status_code != 200:
            print(f"  skipped {ev.get('away_team')} @ {ev.get('home_team')}: HTTP {resp.status_code}")
            continue
        remaining = resp.headers.get("x-requests-remaining", remaining)
        for bk in resp.json().get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != ODDS_MARKET:
                    continue
                by_player = {}
                for oc in mk.get("outcomes", []):
                    d = by_player.setdefault(oc.get("description"), {})
                    d["line"] = oc.get("point")
                    d[str(oc.get("name")).lower()] = oc.get("price")
                for player, d in by_player.items():
                    if player and d.get("line") is not None \
                            and d.get("over") is not None and d.get("under") is not None:
                        rows.append(dict(
                            event_id=ev["id"], commence_time=ev["commence_time"],
                            book=bk.get("key"), player=player, line=float(d["line"]),
                            over_price=float(d["over"]), under_price=float(d["under"]),
                            name_key=name_key(player)))
    if remaining is not None:
        print(f"  API credits remaining this month: {remaining}")
    df = pd.DataFrame(rows)
    df["fetched_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    return df


def get_lines(api_key: str, season: int, week: int, out_dir: str) -> pd.DataFrame:
    raw = os.path.join(out_dir, f"lines_{season}_wk{week}_raw.csv")
    if os.path.exists(raw) and not FORCE_REFRESH_LINES:
        cached = pd.read_csv(raw)
        if len(cached):
            age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(cached["fetched_at"].iloc[0])
                   ).total_seconds() / 60
            if age < LINES_CACHE_MINUTES:
                print(f"Using cached lines fetched {age:.0f} min ago (saves API credits)")
                return cached
    start, end = week_window(season, week)
    df = fetch_lines(api_key, start, end)
    if len(df):
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M")
        df.to_csv(raw, index=False)                                   # latest pull (cache)
        df.to_csv(os.path.join(out_dir, f"lines_{season}_wk{week}_{stamp}.csv"), index=False)
    return df


def american_to_decimal(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / -a)


RECEPTIONS_EDGES = False    # receptions lines are too coarse to vet (half-catch steps): no edges/email/scoring when False
SHRINK_YARDS = 0.5          # weight on OUR probability for receiving-yards edges (rest = book no-vig); fitted to Wk 4-5 results
SHRINK_EXTRA = 0.6          # weight on OUR probability for rushing/receptions edges (rest = book's no-vig)
CHECK_EV, CHECK_GAP = 0.30, 0.35


def evaluate_lines(out: pd.DataFrame, sims: np.ndarray, lines: pd.DataFrame, shrink: float = 1.0):
    """Model P(over)/P(under) at each book's line, vs. the book's no-vig price and EV.

    Probabilities are conditional on the player playing (props normally void otherwise).
    Yards are rounded to integers so integer lines produce correct pushes.
    """
    idx = {}
    for i, k in enumerate(out["player_display_name"].map(name_key)):
        idx.setdefault(k, i)                    # rows are sorted by mean; first match wins
    li = lines.copy()
    li["row"] = li["name_key"].map(idx)
    unmatched = sorted(li.loc[li["row"].isna(), "player"].unique())
    li = li.dropna(subset=["row"]).reset_index(drop=True)
    li["row"] = li["row"].astype(int)
    if li.empty:
        return li, unmatched

    y = np.rint(sims[li["row"].to_numpy()])
    line = li["line"].to_numpy(dtype=float)[:, None]
    p_over, p_under = (y > line).mean(axis=1), (y < line).mean(axis=1)

    dec_o, dec_u = american_to_decimal(li["over_price"]), american_to_decimal(li["under_price"])
    imp_o, imp_u = 1 / dec_o, 1 / dec_u
    li["fair_p_over"] = imp_o / (imp_o + imp_u)          # book's own no-vig probability
    li["raw_p_over"], li["raw_p_under"] = p_over, p_under
    raw_ev_o, raw_ev_u = p_over * (dec_o - 1) - p_under, p_under * (dec_u - 1) - p_over
    li["raw_best_ev"] = np.maximum(raw_ev_o, raw_ev_u)
    if shrink < 1.0:        # blend toward the market: a big gap is more often missing info than edge
        p_over = shrink * p_over + (1 - shrink) * li["fair_p_over"].to_numpy()
        p_under = shrink * p_under + (1 - shrink) * (1 - li["fair_p_over"].to_numpy())
    li["model_p_over"], li["model_p_under"] = p_over, p_under
    li["ev_over"] = p_over * (dec_o - 1) - p_under       # per 1 staked; push returns stake
    li["ev_under"] = p_under * (dec_u - 1) - p_over

    over_best = li["ev_over"] >= li["ev_under"]
    li["best_side"] = np.where(over_best, "over", "under")
    li["best_ev"] = np.where(over_best, li["ev_over"], li["ev_under"])
    li["best_price"] = np.where(over_best, li["over_price"], li["under_price"])
    li["model_p"] = np.where(over_best, p_over, p_under)
    li["fair_p"] = np.where(over_best, li["fair_p_over"], 1 - li["fair_p_over"])

    info_cols = ["player_display_name", "team", "opp", "position", "status", "median", "mean",
                 "model_version"] + (["role_note"] if "role_note" in out.columns else [])
    info = out.iloc[li["row"].to_numpy()][info_cols].reset_index(drop=True).rename(columns={"player_display_name": "model_player",
                                             "median": "model_median", "mean": "model_mean"})
    li = pd.concat([li, info], axis=1)
    # Median and mean disagree on skewed stats. Only call a side "confirmed" when the
    # model's mean AND median are both on that side of the line.
    ov = li["best_side"] == "over"
    li["agree"] = np.where(ov, (li["model_mean"] > li["line"]) & (li["model_median"] > li["line"]),
                           (li["model_mean"] < li["line"]) & (li["model_median"] < li["line"]))
    gap = (li["model_median"] - li["line"]).abs() / li["line"].clip(lower=10)
    li["check_news"] = (li["raw_best_ev"] >= CHECK_EV) | (gap >= CHECK_GAP)
    return li.sort_values("best_ev", ascending=False).reset_index(drop=True), unmatched


# ------------------------------------------------ emailed summary
def _cell(x, bold=False, align="left"):
    style = f"padding:4px 8px;text-align:{align};white-space:nowrap;border-bottom:1px solid #e5e5e5"
    if bold:
        style += ";font-weight:600"
    return f'<td style="{style}">{x}</td>'


def _head(cols):
    return "<tr>" + "".join(
        f'<th style="padding:6px 8px;text-align:left;background:#f3f3f3;white-space:nowrap">{c}</th>'
        for c in cols) + "</tr>"


EXTRA_EMAIL: list = []      # extra HTML sections (e.g. receptions edges) appended to the email


def edges_section(title: str, edges: pd.DataFrame, unit: str, max_rows: int = 15) -> str:
    """Email table of confirmed leans (model mean AND median on the same side of the line)."""
    esc = _html.escape
    conf = edges[edges["agree"]]
    top = conf.drop_duplicates("model_player")
    if unit == "rec":       # reception lines are highly correlated within a game: show the best 2 per game
        top = top.assign(_g=top.apply(lambda r: "|".join(sorted([str(r["team"]), str(r["opp"])])), axis=1))
        top = top.groupby("_g", sort=False).head(2)
    top = top.head(max_rows)
    head = (f"<h3 style='margin:18px 0 4px'>{esc(title)}: our projection vs sportsbook lines</h3>"
            "<p style='margin:0 0 6px;color:#555;font-size:12px'>Only leans where our median AND average "
            f"are on the same side of the line ({conf['model_player'].nunique()} of "
            f"{edges['model_player'].nunique()} players). Largest gaps first, best book per player. "
            "Our chance is blended " + f"{SHRINK_EXTRA:.0%}" + " model / rest book price, so edges are modest. "
            "<b>CHECK NEWS</b> = very large gap from the book: usually injury or role news our model lacks, "
            "verify first. New model with no live track record yet: things to check, not bets.</p>")
    n_u = int((conf.drop_duplicates("model_player")["best_side"] == "under").sum())
    if len(conf) and n_u >= 0.7 * conf["model_player"].nunique():
        head += ("<p style='margin:0 0 6px;color:#a15c00;font-size:12px'>Most leans are unders: yardage is "
                 "right-skewed and book lines sit near our average, so our median is usually lower. "
                 "Wait for the scorecard before trusting the under lean.</p>")
    if top.empty:
        return head + "<p style='color:#555'>No confirmed leans this run.</p>"
    rows = []
    for r in top.itertuples():
        side = "Over" if r.best_side == "over" else "Under"
        flag = " &#9888; CHECK NEWS" if getattr(r, "check_news", False) else ""
        note = getattr(r, "role_note", "")
        note = (f"<div style='font-weight:400;font-size:11px;color:#666;white-space:normal;max-width:360px'>"
                f"{esc(str(note))}</div>") if getattr(r, "check_news", False) and isinstance(note, str) and note else ""
        rows.append("<tr>" + _cell(esc(str(r.model_player)) + flag + note, bold=True)
                    + _cell(f"{esc(str(r.team))} vs {esc(str(r.opp))}")
                    + _cell(esc(str(r.book)))
                    + _cell(f"{r.line:g}", align="right")
                    + _cell(f"{r.model_median:.1f}" if unit == "rec" else f"{r.model_median:.0f}",
                            bold=True, align="right")
                    + _cell(f"{side} ({r.best_price:+.0f})", bold=True)
                    + _cell(f"{r.model_p:.0%}", align="right")
                    + _cell(f"{r.fair_p:.0%}", align="right")
                    + _cell(f"{r.best_ev:+.2f}", align="right") + "</tr>")
    label = "OUR PROJECTION (receptions)" if unit == "rec" else "OUR PROJECTION (yds)"
    return (head + '<table style="border-collapse:collapse;font-size:13px">'
            + _head(["Player", "Game", "Book", "Book line", label, "Our lean (price)",
                     "Our chance it hits", "Book's implied chance", "Edge (EV per $1)"])
            + "".join(rows) + "</table>")


def stat_edges(stat_out: pd.DataFrame, sims: np.ndarray, lines_stat: pd.DataFrame, title: str,
               prefix: str, unit: str, season: int, week: int, out_dir: str, stamp: str):
    """Edges for one extra stat (receptions, rushing yards). Saves {prefix}_{season}_wk{week}_edges_*.csv.
    `stat_out` needs columns median/mean for this stat; `sims` is aligned with its rows."""
    if lines_stat is None or lines_stat.empty:
        print(f"\n{title}: no sportsbook lines yet.")
        return "", None
    edges, unmatched = evaluate_lines(stat_out, sims, lines_stat, shrink=SHRINK_EXTRA)
    if unmatched:
        print(f"\n{title}: book names not matched to a projected player ({len(unmatched)}): {unmatched[:12]}")
    if edges.empty:
        print(f"\n{title}: no book lines matched a projected player.")
        return "", None
    base = os.path.join(out_dir, f"{prefix}_{season}_wk{week}_edges")
    edges.to_csv(f"{base}_{stamp}.csv", index=False)
    edges.to_csv(f"{base}_latest.csv", index=False)
    conf = edges[edges["agree"]]
    top = conf.drop_duplicates("model_player").head(15)
    print(f"\n{title}: {conf['model_player'].nunique()} of {edges['model_player'].nunique()} players have a "
          f"confirmed lean ({len(edges)} book lines). Largest gaps:")
    if len(top):
        show = top[["model_player", "team", "opp", "book", "best_side", "line", "best_price",
                    "model_median", "model_p", "fair_p", "best_ev"]]
        print(show.round({"model_median": 1, "model_p": 2, "fair_p": 2, "best_ev": 3}).to_string(index=False))
    return edges_section(title, edges, unit), edges


def build_email(out: pd.DataFrame, edges, season: int, week: int, extra_html: str = ""):
    """(subject, html): clearly labelled projections plus model-vs-book table."""
    esc = _html.escape
    now = pd.Timestamp.now(tz="America/New_York")
    subject = f"NFL props: {season} Week {week} projections ({now.strftime('%a %-I:%M%p').lower()} ET)"

    rows = []
    for r in out.head(40).itertuples():
        note = r.status if isinstance(r.status, str) and r.status else ""
        if getattr(r, "practice", ""):
            note = (note + " | " if note else "") + f"Practice: {r.practice}"
        rows.append("<tr>" + _cell(esc(str(r.player_display_name)), bold=True)
                    + _cell(f"{esc(str(r.team))} vs {esc(str(r.opp))}")
                    + _cell(esc(str(r.position)))
                    + _cell(f"{r.projection:.0f}", bold=True, align="right")
                    + _cell(f"{r.q25:.0f}-{r.q75:.0f}", align="right")
                    + _cell(f"{r.mean:.0f}", align="right")
                    + _cell(f"{r.p_zero:.0%}", align="right")
                    + _cell(("" if pd.isna(getattr(r, "rec_projection", np.nan))
                             else f"{r.rec_projection:.0f} ({r.rec_q25:.0f}-{r.rec_q75:.0f})"),
                            align="right")
                    + _cell(esc(note)) + "</tr>")
    proj_table = ('<table style="border-collapse:collapse;font-size:13px">'
                  + _head(["Player", "Game", "Pos", "OUR PROJECTION (yds)",
                           "Typical range (25th-75th pct)", "Average (mean)",
                           "Chance of 0 yds", "OUR RECEPTIONS (25th-75th)", "Status"])
                  + "".join(rows) + "</table>")

    lines_html = ""
    if edges is not None and len(edges):
        conf = edges[edges["agree"]]
        top = conf.drop_duplicates("model_player").head(25)
        erows = []
        for r in top.itertuples():
            side = "Over" if r.best_side == "over" else "Under"
            price = f"{r.best_price:+.0f}"
            erows.append("<tr>" + _cell(esc(str(r.model_player)), bold=True)
                         + _cell(f"{esc(str(r.team))} vs {esc(str(r.opp))}")
                         + _cell(esc(str(r.book)))
                         + _cell(f"{r.line:g}", align="right")
                         + _cell(f"{r.model_median:.0f}", bold=True, align="right")
                         + _cell(f"{side} ({price})", bold=True)
                         + _cell(f"{r.model_p:.0%}", align="right")
                         + _cell(f"{r.fair_p:.0%}", align="right")
                         + _cell(f"{r.best_ev:+.2f}", align="right") + "</tr>")
        lines_html = (
            "<h3 style='margin:18px 0 4px'>Our projection vs sportsbook lines</h3>"
            "<p style='margin:0 0 6px;color:#555;font-size:12px'>Only leans where our median AND "
            f"average are on the same side of the line ({conf['model_player'].nunique()} of "
            f"{edges['model_player'].nunique()} players). Largest gaps first, best book per player. "
            f"Our chance is blended {SHRINK_YARDS:.0%} model / rest book price. Chances assume the player plays.</p>"
            '<table style="border-collapse:collapse;font-size:13px">'
            + _head(["Player", "Game", "Book", "Book line", "OUR PROJECTION (yds)",
                     "Our lean (price)", "Our chance it hits", "Book's implied chance", "Edge (EV per $1)"])
            + "".join(erows) + "</table>")
    else:
        lines_html = ("<p style='color:#555'>No sportsbook lines were available for this run "
                      "(props usually post a day or two before kickoff).</p>")

    body = f"""<div style="font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:900px">
<h2 style="margin:0 0 4px">NFL receiving-yard projections: {season} Week {week}</h2>
<p style="margin:0 0 10px;color:#555;font-size:12px">Model {esc(MODEL_VERSION)} | generated {now.strftime('%a %b %-d, %-I:%M %p')} ET |
injury statuses are as of this run; designations usually finalize Friday.</p>
<p style="font-size:13px;margin:0 0 8px"><b>How to read this:</b> <b>Our projection</b> is the median outcome:
half of 2,000 simulated games landed above it, half below. <b>Average</b> runs higher because big games pull it up.
The <b>typical range</b> holds the middle half of outcomes. All numbers assume the player plays.</p>
<div style="overflow-x:auto">
<h3 style="margin:12px 0 4px">Top 40 projections</h3>
{proj_table}
{lines_html}
{extra_html}
</div>
<p style="color:#777;font-size:12px;margin-top:14px">A big gap between our projection and a book's line more often means the
model is missing news than that there's an edge. Full tables are attached as CSV.</p>
</div>"""
    return subject, body


def write_email(out, edges, season: int, week: int, out_dir: str) -> None:
    extra = ""
    part = os.path.join(out_dir, "rushing_email_part.html")      # written by rushing.py if it ran first
    if os.path.exists(part):
        with open(part, encoding="utf-8") as f:
            extra = f.read()
    extra += "".join(EXTRA_EMAIL)
    subject, body = build_email(out, edges, season, week, extra_html=extra)
    with open(os.path.join(out_dir, "email_latest.html"), "w", encoding="utf-8") as f:
        f.write(body)
    with open(os.path.join(out_dir, "email_subject.txt"), "w", encoding="utf-8") as f:
        f.write(subject)
    print(f"Email summary written ({subject})")


def main():
    season, week = PROJECT_SEASON, PROJECT_WEEK
    if season is None or week is None:
        try:
            season, week = detect_target_week()
        except ValueError as e:        # offseason / playoffs: nothing to do, don't fail the workflow
            print(e)
            return
    print(f"Target: {season} week {week}\n")
    out_dir = resolve_out_dir()
    print(f"Saving to: {os.path.abspath(out_dir)}\n")
    print(f"Model version: {MODEL_VERSION}\n")

    played, parts, inj = build_table(project=(season, week))
    played["is_future"] = False
    fut = make_future_rows(played, parts, inj, season, week)

    # Features are computed on history + future rows together; they're lagged,
    # so a future row only sees games before it.
    full = pd.concat([played, fut], ignore_index=True, sort=False)
    full = full.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    full = add_features(full)
    full = add_opponent_features(full, parts["ctx"])
    full = add_weather_features(full)
    if USE_NGS_FEATURES:
        full = add_ngs_features(full)
    hist = full[~full["is_future"]].copy()
    fut = full[full["is_future"]].copy()

    # Walk-forward backtest (only needed for dispersion); cached between runs
    bt_name = BACKTEST_CACHE if MODEL_VERSION == "v6" else f"receiving_backtest_{MODEL_VERSION}.csv"
    bt_path = os.path.join(out_dir, bt_name)
    if os.path.exists(bt_path) and not REFRESH_BACKTEST:
        print(f"Using cached backtest: {bt_path} (set REFRESH_BACKTEST = True to redo)")
        backtest = pd.read_csv(bt_path)
    else:
        backtest, m_targets = walk_forward(hist)
        report(backtest, m_targets)
        backtest.to_csv(bt_path, index=False)

    # Which players get projected
    scale = 100.0 if hist["offense_pct"].max() > 1.5 else 1.0
    no_line = sorted(fut.loc[fut["implied_total"].isna(), "team"].unique())
    if no_line:
        print(f"No betting line yet, skipped teams: {no_line}")
    fut = fut.dropna(subset=FEATURES)
    fut = fut[(fut["games_prior"] >= MIN_PRIOR_GAMES)
              & (fut["ewm_offense_pct"] >= MIN_SNAP_PCT * scale)]
    out_names = sorted(fut.loc[fut["status"] == "Out", "player_display_name"])
    fut = fut[fut["status"] != "Out"]
    if out_names:
        print(f"Excluded (listed Out): {out_names}")
    if fut.empty:
        raise ValueError("No players to project. Check the week, rosters and betting lines.")

    train = hist.dropna(subset=FEATURES + [TARGET])
    train = train[train["games_prior"] >= MIN_PRIOR_GAMES]
    rates = play_rates(inj, parts["pw"], season * 100 + week)
    print(f"Historical play rate by report status: {({k: round(v, 2) for k, v in rates.items()})}")

    out, sims, rec_sims = project_upcoming(train, fut, backtest, rates)

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M")
    base = os.path.join(out_dir, f"projections_{season}_wk{week}")
    out.to_csv(f"{base}_{stamp}.csv", index=False)      # timestamped snapshot (keeps history)
    path = f"{base}_latest.csv"
    out.to_csv(path, index=False)

    cols = ["player_display_name", "team", "opp", "position", "status", "practice",
            "q25", "projection", "q75", "mean", "p_zero", "rec_projection"]
    if PROP_LINES:
        cols += ["line", "p_over", "p_under"]
    rounding = {"q10": 0, "q25": 0, "projection": 0, "q75": 0, "q90": 0, "mean": 1,
                "p_zero": 2, "p_over": 2, "p_under": 2}
    labels = {"player_display_name": "player", "projection": "OUR_PROJECTION", "q25": "range_low(25%)",
              "q75": "range_high(75%)", "mean": "average", "p_zero": "chance_0_yds",
              "rec_projection": "OUR_RECEPTIONS"}
    pd.set_option("display.width", 220)
    print(f"\nTop projections, {season} week {week} ({len(out)} players total):")
    print("OUR_PROJECTION = median outcome (half of simulated games land above, half below); "
          "average is higher because big games pull it up.")
    print(out[cols].head(40).round(rounding).rename(columns=labels).to_string(index=False))
    print(f"\nSaved {path} (plus a timestamped snapshot)")

    # ---- standalone receptions projections file (same simulation as the yards projection) ----
    rcols = ["player_display_name", "team", "opp", "position", "status", "rec_projection", "rec_mean", "rec_q25",
             "rec_q75", "rec_p_zero", "exp_targets", "p_catch", "projection"]
    if all(c in out.columns for c in rcols):
        rec_file = (out[rcols].rename(columns={
            "player_display_name": "player", "rec_projection": "receptions_median", "rec_mean": "receptions_mean",
            "rec_q25": "receptions_q25", "rec_q75": "receptions_q75", "rec_p_zero": "p_zero_receptions",
            "projection": "receiving_yards_median"})
            .sort_values(["receptions_median", "receptions_mean"], ascending=False)
            .round({"receptions_mean": 2, "p_zero_receptions": 3, "exp_targets": 2, "p_catch": 3, "receiving_yards_median": 1}))
        rbase = os.path.join(out_dir, f"receptions_{season}_wk{week}")
        rec_file.to_csv(f"{rbase}_{stamp}.csv", index=False)
        rec_file.to_csv(f"{rbase}_latest.csv", index=False)
        print(f"Saved {rbase}_latest.csv (receptions projections for {len(rec_file)} players)")

    # ---- receptions lines (ParlayAPI; one call covers receiving yards, rushing yards and receptions) ----
    EXTRA_EMAIL.clear()
    pkey = lines_parlay.get_key()
    if pkey:
        try:
            plines = lines_parlay.get_lines(pkey, season, week, out_dir, window=week_window(season, week))
        except Exception as e:
            print(f"ParlayAPI lines skipped ({e})")
            plines = pd.DataFrame()
        if len(plines) and RECEPTIONS_EDGES:
            rec_out = out.assign(median=out["rec_projection"], mean=out["rec_mean"])
            html_rec, _ = stat_edges(rec_out, rec_sims, plines[plines["stat"] == "receptions"],
                                     "Receptions", "receptions", "rec", season, week, out_dir, stamp)
            if html_rec:
                EXTRA_EMAIL.append(html_rec)
    else:
        print("\nNo PARLAY_API_KEY found: receptions and rushing line comparisons skipped.")

    # ---- sportsbook lines ----
    api_key = get_api_key()
    if not api_key:
        print("\nNo ODDS_API_KEY found, so sportsbook lines were skipped "
              "(add it as a Colab Secret or environment variable).")
        write_email(out, None, season, week, out_dir)
        return
    lines = get_lines(api_key, season, week, out_dir)
    if lines is None or lines.empty:
        print("\nNo lines returned yet (props usually post a day or two before kickoff).")
        write_email(out, None, season, week, out_dir)
        return
    edges, unmatched = evaluate_lines(out, sims, lines, shrink=SHRINK_YARDS)
    if unmatched:
        print(f"\nBook names not matched to a projected player ({len(unmatched)}): "
              f"{unmatched[:15]}{' ...' if len(unmatched) > 15 else ''}")
    if edges.empty:
        print("No book lines matched any projected player.")
        write_email(out, None, season, week, out_dir)
        return
    edges.to_csv(f"{base}_edges_{stamp}.csv", index=False)
    edges.to_csv(f"{base}_edges_latest.csv", index=False)

    try:
        bb_path = os.path.join(out_dir, "receiving_backtest_boom.csv")      # larger multi-season sample, if available
        bb = pd.read_csv(bb_path) if os.path.exists(bb_path) else backtest
        ladder = boom.boom_ladder(out, sims, lines, boom.tail_map(bb), name_key)
        if len(ladder):
            ladder.round({"p_model": 4, "p_boom": 4, "p_over_book_line": 4, "fair_odds": 0, "bet_if_at_least": 0,
                          "model_mean": 1, "model_median": 1}).to_csv(f"{base}_boom_{stamp}.csv", index=False)
            ladder.round({"p_model": 4, "p_boom": 4, "p_over_book_line": 4, "fair_odds": 0, "bet_if_at_least": 0,
                          "model_mean": 1, "model_median": 1}).to_csv(f"{base}_boom_latest.csv", index=False)
            EXTRA_EMAIL.append(boom.boom_section(ladder))
            print(f"\nSaved {base}_boom_latest.csv ({ladder['player'].nunique()} players, boom ladder to +{boom.MAX_ODDS})")
    except Exception as e:
        print(f"Boom ladder skipped ({e})")

    n_all = edges.drop_duplicates("model_player").shape[0]
    conf = edges[edges["agree"]]
    top = conf.drop_duplicates("model_player").head(25)     # best book per player
    print(f"\n{conf['model_player'].nunique()} of {n_all} players have a "
          "confirmed lean (model mean AND median on the same side of the line).")
    show = top[["model_player", "team", "opp", "book", "best_side", "line", "best_price",
                "model_median", "model_p", "fair_p", "best_ev"]]
    print(f"\nLargest model-vs-book gaps ({len(edges)} book lines, best book per player):")
    print(show.round({"model_median": 0, "model_p": 2, "fair_p": 2, "best_ev": 3})
          .to_string(index=False))
    print("EV is per 1 staked, from the model's probabilities. A big gap more often means "
          "the model is missing news than that you've found an edge; check before acting.")
    write_email(out, edges, season, week, out_dir)


if __name__ == "__main__":
    main()
