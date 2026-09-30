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

Manual runs: Actions > NFL props > Run workflow, pick `project`, `score`, or `ab`.
`ab` runs `ab_opponent.py`, an A/B backtest of the opponent defense-vs-position features. If it
shows a clear gain, set `USE_OPP_FEATURES = True` at the top of `projections.py`.
Snapshots and the scorecard carry a `model_version` column so results stay comparable.
Holiday weeks (Thanksgiving, Christmas) have odd kickoff times: trigger a manual run beforehand.
Injury overrides / sportsbook settings live in the config block at the top of `projections.py`.
