# BETter — MLB Game Prediction Engine

A high-accuracy MLB win-probability system targeting **59–62% pre-game accuracy** with real-time in-game updates and betting edge detection against live market odds.

## Target Performance

| Metric | Target | Benchmark |
|--------|--------|-----------|
| Accuracy | 59–62% | Vegas ~58%, home team ~54% |
| Log Loss | 0.66–0.67 | Coin flip = 0.693 |
| Brier Score | 0.22–0.23 | Coin flip = 0.25 |

---

## Architecture Overview

```
Data Layer (Phase 1)          Feature Engineering (Phase 2)       Models (Phase 3+)
──────────────────            ─────────────────────────────        ─────────────────
Retrosheet game logs    ──►   Rolling team stats (7/14/30d)  ──►  GBM Ensemble
Statcast pitch data     ──►   Pitcher matchup features        ──►  Bayesian State-Space
FanGraphs advanced      ──►   Park factors / weather          ──►  Monte Carlo (10K sims)
MLB Stats API (live)    ──►   Elo ratings                     ──►  Self-Supervised Transformer
The Odds API            ──►   Market-implied probs            ──►  Meta-Learner Stack
                                                               ──►  Market-Aware RL Agent
```

**Storage:** DuckDB (embedded, columnar — 10–100× faster than SQLite for analytics)
**Package manager:** `uv` (fast, reproducible)
**Python:** 3.11+

---

## Quick Start

### 1. Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone & Install

```bash
git clone https://github.com/RohithGSVV/BETter.git
cd BETter

# Install all dependencies (creates .venv automatically)
uv sync
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required: get a free key at https://the-odds-api.com
ODDS_API_KEY=your_key_here

# Optional overrides (defaults shown)
DUCKDB_PATH=data/better.duckdb
TRAIN_START_YEAR=2010
TRAIN_END_YEAR=2025
```

### 4. Run Tests

```bash
uv run pytest tests/ -v
```

All 16 unit tests should pass (odds math, Kelly criterion, Pythagorean WP, rolling stats).

---

## Phase 1 Data Ingestion

Run in order. Total time: ~5 minutes for steps 1–3; step 4 is optional and large.

### Step 1 — Player ID Crosswalk (~1 min)

Maps player IDs across MLB AM, FanGraphs, Retrosheet, and Baseball Reference.

```bash
uv run python -m better.data.ingest.player_ids
```

Expected: **~24,695 players** loaded.

### Step 2 — Team & Player Stats (~3 min)

Advanced metrics via FanGraphs: wRC+, wOBA, FIP, xFIP, SIERA, Stuff+, K%, BB%, WAR.

```bash
uv run python -m better.data.ingest.lahman
```

Expected: **450** team-season rows, **8,942** pitcher rows, **8,158** batter rows (2010–2024).

### Step 3 — Historical Game Logs (~15 sec)

15+ seasons of game-level results: scores, hits, errors, attendance, starting pitchers.

```bash
uv run python -m better.data.ingest.retrosheet
```

Expected: **~37,343 games** (2010–2025), ~52–56% home win rate per season ✓

### Step 4 — Statcast Pitch Data (optional, large)

Pitch-by-pitch: velocity, spin rate, launch angle, xwOBA, delta run expectancy.
⚠️ **Large download (~10–15 GB, several hours).** Run overnight or on Kaggle T4x2.

```bash
uv run python -m better.data.ingest.statcast
```

### Step 5 — Verify

```bash
uv run python -c "
from better.data.db import get_connection
conn = get_connection()
tables = ['games','player_ids','team_batting','team_pitching','player_batting','player_pitching']
for t in tables:
    n = conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'{t:<25}: {n:,} rows')
"
```

---

## Project Structure

```
BETter/
├── src/better/
│   ├── config.py              # Pydantic settings (loads .env)
│   ├── constants.py           # Team codes, park factors, Elo constants
│   ├── data/
│   │   ├── db.py              # DuckDB connection singleton
│   │   ├── schema.py          # 13-table schema definitions
│   │   └── ingest/
│   │       ├── retrosheet.py  # Historical game logs (2010–2025)
│   │       ├── lahman.py      # FanGraphs team + player stats
│   │       ├── player_ids.py  # Chadwick ID crosswalk
│   │       ├── statcast.py    # Pitch-by-pitch Statcast data
│   │       ├── mlb_api.py     # Live MLB Stats API client
│   │       └── odds.py        # The Odds API client
│   ├── data/live/
│   │   ├── game_feed.py       # Async live game state poller (10s intervals)
│   │   └── odds_feed.py       # Async live odds poller
│   └── utils/
│       ├── stats.py           # Pythagorean WP, EWMA, Kelly, odds math
│       ├── dates.py           # Date helpers
│       ├── logging.py         # Structured logging (structlog)
│       └── async_helpers.py   # Async utilities
├── tests/
│   ├── conftest.py
│   └── unit/
│       ├── test_stats.py
│       └── test_odds_converter.py
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Database Schema (13 Tables)

| Table | Description | Key Columns |
|-------|-------------|-------------|
| `games` | One row per MLB game | `game_pk`, `game_date`, `home_win`, `home_score`, `away_score` |
| `player_ids` | Cross-system ID map | `mlb_id`, `retrosheet_id`, `fangraphs_id`, `bbref_id` |
| `statcast_pitches` | Pitch-level Statcast data | `pitcher_id`, `pitch_type`, `release_speed`, `launch_speed`, `delta_run_exp` |
| `team_batting` | Season-level team batting | `wrc_plus`, `woba`, `obp`, `slg`, `ops` |
| `team_pitching` | Season-level team pitching | `era`, `fip`, `xfip`, `siera`, `k_pct`, `bb_pct` |
| `player_batting` | Season-level batter stats | `woba`, `wrc_plus`, `war`, `babip` |
| `player_pitching` | Season-level pitcher stats | `fip`, `xfip`, `siera`, `stuff_plus`, `war` |
| `team_features_daily` | Rolling game-day features | `pythag_win_pct_30`, `ewma_win_rate_7`, `bullpen_fip_composite` |
| `pitcher_features` | Pitcher matchup features | `siera`, `stuff_plus`, `days_rest`, `game_score_avg_5` |
| `predictions` | Model output per game | `bayesian_prob`, `gbm_prob`, `monte_carlo_prob`, `edge` |
| `odds_snapshots` | Live market odds captures | `bookmaker`, `home_implied_prob`, `away_fair_prob`, `overround` |
| `win_expectancy` | In-game WE lookup table | `inning`, `outs`, `runners`, `score_diff`, `win_prob` |
| `bet_log` | Bet tracking + P&L | `edge`, `kelly_fraction`, `bet_amount`, `pnl`, `bankroll_after` |

---

## Models (Phases 3–5)

| Model | Purpose | Status |
|-------|---------|--------|
| GBM Ensemble (XGBoost + LightGBM + CatBoost) | Pre-game win probability | Phase 3 |
| Bayesian State-Space (Kalman Filter) | Team strength with uncertainty | Phase 3 |
| Monte Carlo Simulator (10K runs) | Run distribution simulation | Phase 3 |
| Self-Supervised Transformer | Pretrained on Statcast, fine-tuned for WP | Phase 4 (Kaggle T4x2) |
| Meta-Learner Stack | Logistic blending of all model outputs | Phase 4 |
| Market-Aware RL Agent | Betting edge optimization vs. market | Phase 5 |

**Skipped:** Neural ODEs (baseball is discrete events), Diffusion Models, TFT, GNN — no evidence of gains in sports prediction tasks.

---

## Betting Framework

- **Edge threshold:** ≥ 3% (`model_prob − market_implied_prob ≥ 0.03`)
- **Kelly sizing:** Quarter-Kelly (0.25×) to manage variance
- **Vig removal:** Normalize home + away implied probs to sum to 1.0
- **Odds source:** The Odds API free tier (~500 req/month → 2 polls/day)
- **Validation:** Walk-forward only — never random train/test splits on time-series data
- **Calibration:** Platt scaling (`CalibratedClassifierCV`) + reliability diagrams

---

## Roadmap

- [x] **Phase 1** — Data pipeline (Retrosheet, FanGraphs, Statcast, MLB API, Odds API)
- [ ] **Phase 2** — Feature engineering (rolling stats, Elo ratings, park factors, pitcher matchups)
- [ ] **Phase 3** — GBM ensemble + Bayesian state-space + Monte Carlo simulator
- [ ] **Phase 4** — Transformer pretraining on Kaggle + meta-learner stacking
- [ ] **Phase 5** — Live prediction API (FastAPI) + Streamlit dashboard
- [ ] **Phase 6** — Market-aware RL betting agent + paper trading

---

## Data Sources & Attribution

| Source | Data | Cost |
|--------|------|------|
| [Retrosheet](https://www.retrosheet.org/) | Game logs 1898–present | Free |
| [FanGraphs](https://www.fangraphs.com/) via pybaseball | wRC+, FIP, SIERA, WAR | Free |
| [Chadwick Bureau](https://github.com/chadwickbureau/register) | Player ID crosswalk | Free |
| [MLB Stats API](https://statsapi.mlb.com/) | Live schedules, lineups, feeds | Free |
| [The Odds API](https://the-odds-api.com/) | Live odds from 40+ bookmakers | Free tier |

> **Retrosheet notice:** The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at [retrosheet.org](https://www.retrosheet.org).

---

## License

Private project — not for distribution.
