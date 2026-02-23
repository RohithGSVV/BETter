# BETter

MLB game prediction system with real-time win probability and betting optimization.

## Overview

BETter predicts MLB game outcomes using an ensemble of models:

- **Gradient-Boosted Ensemble** (XGBoost + LightGBM + CatBoost) — core prediction engine
- **Bayesian State-Space Model** (Kalman filter) — tracks evolving team strength
- **Monte Carlo Simulator** — full score distributions from 10K game simulations
- **Self-Supervised Transformer** — player embeddings from pitch-level Statcast data
- **Meta-Learner Stack** — calibrated ensemble combining all model outputs

The system provides:
- Pre-game win probabilities for every MLB game
- Dynamic in-game win probability updates via live feed polling
- Betting edge detection by comparing model probabilities against market odds
- Fractional Kelly criterion bet sizing recommendations

## Target Performance

| Metric | Target | Benchmark |
|--------|--------|-----------|
| Accuracy | 59–62% | Vegas ~58%, home team ~54% |
| Log Loss | 0.66–0.67 | Coin flip = 0.693 |
| Brier Score | 0.22–0.23 | Coin flip = 0.25 |

## Quick Start

```bash
# Install dependencies
uv sync

# Create .env from template
cp .env.example .env
# Edit .env with your Odds API key

# Initialize database and load historical data
python -m better.data.schema
python -m better.data.ingest.retrosheet
python -m better.data.ingest.lahman

# Run tests
uv run pytest
```

## Data Sources

| Source | Data | Cost |
|--------|------|------|
| Retrosheet | Game logs 1898–present | Free |
| Statcast/Baseball Savant | Pitch-level tracking 2015+ | Free |
| Lahman Database | Season aggregates 1871–present | Free |
| FanGraphs | wRC+, FIP, SIERA, WAR | Free |
| MLB Stats API | Schedules, lineups, live feeds | Free |
| The Odds API | Odds from 40+ bookmakers | Free tier |

## Project Structure

```
src/better/
├── data/           # Ingestion and database layer
├── features/       # Feature engineering pipeline
├── models/         # ML models (Bayesian, GBM, Transformer, Monte Carlo, Meta)
├── betting/        # Odds conversion, edge detection, Kelly sizing
├── api/            # FastAPI backend
├── dashboard/      # Streamlit frontend
├── training/       # Model training and validation
├── jobs/           # Scheduled data ingestion and prediction
└── utils/          # Logging, stats, date helpers
```

## License

Private project — not for distribution.
