# NFL receiving-yards projections (automated)

Two scripts, run on a schedule by GitHub Actions (`.github/workflows/nfl.yml`):

- `projections.py` projects the upcoming week, pulls sportsbook lines, and saves snapshots to `data/`.
- `score_week.py` grades finished games against those snapshots and keeps a running scorecard in `data/`.

## Setup
1. Create a **private** repo and add these files (keep the `.github/workflows/nfl.yml` path).
2. Repo > Settings > Secrets and variables > Actions > New repository secret: name `ODDS_API_KEY`, value = your key from the-odds-api.com.
3. Repo > Settings > Actions > General > Workflow permissions: **Read and write**.
4. Actions tab > "NFL props" > Run workflow > `project`. The first run also builds the backtest cache, so it takes longer.

After that it runs itself. Each run's printed output appears on its Actions page (Summary), and the CSVs land in `data/`.

## Email (optional)
Each run can email you. Projection runs send a formatted summary (top 40 projections, then model vs book
lines) with the CSVs attached; scoring and A/B runs send their printed results. Add three repo secrets
(Settings > Secrets and variables > Actions):
- `EMAIL_USER`: the Gmail address that sends (a spare account is fine)
- `EMAIL_APP_PASSWORD`: a Google app password (Google Account > Security > 2-Step Verification > App passwords;
  your normal password won't work)
- `EMAIL_TO`: where to receive it (optional; defaults to `EMAIL_USER`)
No secrets = no email, and the run still succeeds. Test it with Actions > Run workflow > `score`.
To get fewer emails, delete cron lines in `nfl.yml`.

Manual runs: Actions > NFL props > Run workflow, pick `project`, `score`, `ab`, or `ab_features`.
`ab_features` A/B tests practice-participation and game-day-weather features; flip
`USE_PRACTICE_FEATURES` / `USE_WEATHER_FEATURES` in `projections.py` only if its verdict is a clear gain.
`ab` runs `ab_opponent.py`, an A/B backtest of the opponent defense-vs-position features. If it
shows a clear gain, set `USE_OPP_FEATURES = True` at the top of `projections.py`.
Snapshots and the scorecard carry a `model_version` column so results stay comparable.
Holiday weeks (Thanksgiving, Christmas) have odd kickoff times: trigger a manual run beforehand.
Injury overrides / sportsbook settings live in the config block at the top of `projections.py`.

## Closing lines for CLV (optional)

`closing_snapshot.py` pulls player receiving-yards lines from ParlayAPI (about 3 credits per run)
shortly before each kickoff window and saves `data/closing_YYYYMMDD_HHMM.csv`. `score_week.py` uses
those files as the closing line when it grades CLV (`scorecard_clv.csv`).

- Add the repo secret `PARLAY_API_KEY`. Without it the closing runs do nothing.
- Five scheduled runs a week (plus the Sunday-morning check) (Thu night, Sun 1pm, Sun 4pm, Sun night, Mon night), about 15 credits.
- Only real sportsbooks are kept; PrizePicks, Pick6, Underdog, Sleeper and exchanges are ignored.
- Cron times are UTC. After the clocks change in November, runs land an hour earlier relative to
  kickoff; the script's 150-minute window still covers it. A sixth run at about 8:50am ET Sunday covers 9:30am London-type games; it checks the NFL schedule first and uses no credits if no game kicks off before noon.
- Test by hand: Actions -> Run workflow -> `closing` (only finds games starting within 150 minutes).
