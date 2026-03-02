# BETter — Startup & Navigation Guide

## Quick Start

```bash
# 1. Install dependencies (first time only)
uv sync

# 2. Start the dashboard (recommended for most users)
uv run better dashboard

# 3. Or start the API server (for programmatic access)
uv run better api serve
```

---

## Starting the Dashboard

```bash
uv run better dashboard                  # default: http://localhost:8501
uv run better dashboard --port 3000      # custom port
```

Opens a NiceGUI web app in your browser with a dark theme. The dashboard loads your trained models on startup and talks directly to the Python backend — no API server needed.

### Dashboard Pages

#### Today's Games

What it shows:
- Today's MLB schedule with probable starting pitchers
- Model predictions: P(Home Win) from each model (Bayesian, Monte Carlo, Meta-Learner)
- Market odds comparison (if Odds API key is configured)
- Edge = Model probability minus market fair probability
- Bet recommendations: games that pass the minimum edge threshold with Kelly sizing

What's happening in the backend:
- Fetches the schedule from the MLB Stats API (free, no key needed)
- Loads saved models from the `models/` directory
- Bayesian Kalman uses learned team strengths to predict outcomes
- Monte Carlo simulates 10,000 games using team run-scoring distributions
- If an Odds API key is set, fetches live odds and computes edges
- The Refresh button clears cached data and re-fetches everything

When there are no games (off-season or off-day), it looks ahead up to 7 days to tell you when the next games are scheduled.

#### Backtest Results

What it shows:
- Summary metrics: total bets, win rate, yield, Sharpe ratio, max drawdown
- Bankroll curve chart (Plotly interactive) showing bankroll growth over time
- Detailed statistics table

Controls (inline above the chart):
- **Min Edge** (0.01–0.15): Only bet when model edge exceeds this. Higher = fewer bets, higher win rate, less volume. Default 3%.
- **Kelly Fraction** (0.05–1.0): How aggressively to size bets. 0.25 = quarter-Kelly (conservative). 1.0 = full Kelly (aggressive, high variance).
- **Model**: Which model's predictions to use for the backtest.
- **Run Backtest** button: re-runs with the selected parameters.

What's happening in the backend:
- Loads out-of-fold predictions from `results/oof_details.csv` (25K+ historical games)
- Uses Elo probabilities as a synthetic market (since historical odds aren't available)
- Simulates betting through each game chronologically with Kelly sizing
- Results are cached per parameter combination so switching is fast

#### Edge Analysis

What it shows:
- **Model Comparison table**: accuracy, log-loss, and lift vs Elo baseline for every model
- **Calibration Plot**: predicted probability vs actual win rate. Points on the diagonal = perfect calibration. ECE (Expected Calibration Error) shown below.
- **Win Rate by Edge Threshold**: how win rate changes at different edge cutoffs
- **Edge by Probability Range**: where the model has the most edge (favorites vs underdogs)
- **Edge by Month**: seasonal patterns in model performance

Controls:
- **Model dropdown**: choose which model to analyze. Switching models re-renders all charts instantly.

What's happening in the backend:
- Reads the OOF (out-of-fold) predictions CSV
- The EdgeAnalyzer compares model predictions against Elo baseline and actual outcomes
- Calibration bins predictions into probability ranges and checks accuracy in each

#### Model Status

What it shows:
- Load status for each model (green = loaded, red = not found)
- Last training date
- Accuracy metrics from `results/summary.csv` (log-loss, Brier score, accuracy)
- Per-model file sizes and storage info
- Per-fold results (expandable section)

What's happening in the backend:
- Checks which model files exist in the `models/` directory
- Reads training summary CSVs from the `results/` directory
- Scans model directories for file counts and sizes

---

## Starting the API Server

```bash
uv run better api serve                        # default: http://localhost:8000
uv run better api serve --port 9000            # custom port
uv run better api serve --reload               # auto-reload on code changes (dev mode)
```

### API Endpoints

Open http://localhost:8000/docs for the interactive Swagger UI.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns status and whether models are loaded |
| `/api/games/today` | GET | Today's schedule with probable pitchers |
| `/api/predictions/today` | GET | Model predictions for today's games |
| `/api/bets/recommendations` | GET | Bet recommendations (filtered to positive edge) |
| `/api/backtest/summary` | GET | Backtest results with configurable params |
| `/api/backtest/bankroll-curve` | GET | Bankroll time series data for charting |
| `/api/models/status` | GET | Which models are loaded, accuracy metrics |
| `/api/edge/calibration` | GET | Calibration and edge analysis data |

Query parameters for backtest endpoints:
- `edge_threshold` (float, default 0.03)
- `kelly_fraction` (float, default 0.25)
- `model` (string, default "meta_learner")

Example:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/backtest/summary?edge_threshold=0.05&model=gbm_ensemble
```

### What happens on API startup

1. FastAPI application initializes
2. All 4 models are loaded from `models/` directory into memory
3. CORS middleware is enabled (allows requests from any origin)
4. Routes are registered and the server starts accepting requests
5. Models stay in memory — no reloading between requests

---

## CLI Commands Reference

### Data & Features
```bash
uv run better features build           # Run full feature pipeline
uv run better features training-set    # Export training DataFrame
uv run better features elo             # Compute and display Elo ratings
```

### Model Training
```bash
uv run better model train              # Train all models (with Optuna tuning)
uv run better model train --skip-tuning   # Train without hyperparameter tuning (faster)
uv run better model train --n-trials 100  # More Optuna trials (better but slower)
uv run better model evaluate           # Show saved OOF prediction stats
uv run better model predict NYY BOS    # Quick prediction for a matchup
```

### Betting & Backtesting
```bash
uv run better bet backtest             # Run backtest with default params
uv run better bet backtest --edge-threshold 0.05 --model gbm_ensemble
uv run better bet sweep                # Compare results across edge thresholds
uv run better bet edge                 # Calibration and edge analysis
uv run better bet generate-oof         # Rebuild OOF CSV without retraining
```

### Server & Dashboard
```bash
uv run better api serve                # Start FastAPI server on :8000
uv run better dashboard                # Start NiceGUI dashboard on :8501
```

---

## Configuration

Settings are loaded from environment variables or a `.env` file in the project root.

Key settings:
- `ODDS_API_KEY`: Your API key from [The Odds API](https://the-odds-api.com/) (free tier: 500 requests/month). Without this, odds/edge features are unavailable but predictions still work.
- `INITIAL_BANKROLL`: Starting bankroll for backtesting (default: $1,000)
- `MIN_EDGE_THRESHOLD`: Minimum edge to recommend a bet (default: 0.03 = 3%)
- `KELLY_FRACTION`: Fraction of Kelly criterion to use (default: 0.25 = quarter-Kelly)

Example `.env`:
```
ODDS_API_KEY=your_key_here
INITIAL_BANKROLL=1000
```

---

## Architecture Overview

```
User
  |
  +-- uv run better dashboard  -->  NiceGUI App (port 8501)
  |                                    |
  |                                    +-- imports Python modules directly
  |                                    |     (no HTTP calls)
  |                                    v
  |                              PredictionService
  |                                    |
  |                                    +-- loads models from models/
  |                                    +-- fetches MLB schedule (free API)
  |                                    +-- fetches odds (Odds API, optional)
  |                                    +-- runs predictions through models
  |                                    +-- computes edges via BettingEngine
  |                                    +-- runs backtests via Backtester
  |
  +-- uv run better api serve  -->  FastAPI Server (port 8000)
                                       |
                                       +-- same PredictionService
                                       +-- REST endpoints for external access
                                       +-- Swagger docs at /docs
```

The dashboard and API both use the same `PredictionService` class but run as separate processes. You can run either one independently — the dashboard does not need the API server.

---

## Troubleshooting

**"No games scheduled today"**
- Normal during off-season (Nov–March) or on off-days. The page will show when the next games are.

**Models not loading**
- Make sure you've trained models first: `uv run better model train --skip-tuning`
- Check that `models/` directory contains subdirectories like `gbm_ensemble/final/`, `bayesian_kalman/final/`, etc.

**No edge/odds data**
- Set your `ODDS_API_KEY` in `.env`. Without it, predictions work but edge calculations are skipped.
- The free tier allows ~16 requests/day. The scheduler fetches twice daily (10AM and 5PM ET).

**Dashboard seems slow on first load**
- Model loading takes a few seconds on first request. After that, models are cached in memory.
- Backtest with default params is cached after the first run.

**Port already in use**
- Use `--port` flag: `uv run better dashboard --port 3000` or `uv run better api serve --port 9000`
