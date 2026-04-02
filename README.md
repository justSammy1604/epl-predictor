# EPL Prediction Engine
### Julia XGBoost Backend · HTML/CSS/JS Frontend

---

## 📁 File Architecture

Every file has a single, focused responsibility. Here's exactly what lives where:

```
epl-predictor/
├── backend/
│   ├── server.jl          ← Entry point. HTTP server, all API routes
│   ├── model.jl           ← XGBoost training, bootstrap CI prediction
│   ├── features.jl        ← ELO ratings, rolling averages, feature matrix
│   ├── monte_carlo.jl     ← Season simulator (N × round-robin)
│   ├── betting.jl         ← EV calculator, no-vig odds, Kelly criterion
│   ├── Project.toml       ← Julia package dependencies
│   └── data/
│       └── epl_final.csv  ← 9,380 EPL matches (2000–2024)
│
└── frontend/
    ├── index.html         ← Full UI shell, tabs, all panels
    ├── css/
    │   └── style.css      ← Dark theme, all component styles
    └── js/
        ├── api.js         ← All HTTP calls to Julia backend
        ├── app.js         ← UI logic, state, event handlers
        └── charts.js      ← Chart.js: prediction bars, EV, MC standings
```

---

## 🧩 What Each File Does

### Backend (Julia)

| File | Responsibility |
|------|---------------|
| `server.jl` | HTTP server on `:8080`. Defines all 7 REST endpoints. Imports all other modules. Entry point — run this. |
| `model.jl` | Loads CSV → trains XGBoost via MLJ with 5-fold CV grid search on `max_depth`. Exports `predict_match()` and `predict_with_confidence()` (bootstrap CI). Saves/loads `.jlso` model. |
| `features.jl` | `compute_elo!()` — chronological ELO updates (K=20, start=1500). `compute_rolling_averages!()` — 5-game rolling goal average. `get_team_stats()` — aggregated per-team lookup used by both prediction and API. |
| `monte_carlo.jl` | `simulate_season()` — pre-computes all fixture probabilities from model, then runs N simulations using `Threads.@threads`. Returns per-team: mean/std points, 5th–95th CI, title/top4/top6/relegation probabilities. |
| `betting.jl` | `compute_betting_ev()` — compares model probability to bookmaker odds. `no_vig_probability()` removes bookmaker margin. `kelly_criterion()` computes optimal half-Kelly stake size. |

### Frontend (HTML/CSS/JS)

| File | Responsibility |
|------|---------------|
| `index.html` | App shell. Four-tab layout: Match Predictor, Betting EV, Season Simulator, ELO Rankings. All DOM structure. Loads Chart.js from CDN. |
| `css/style.css` | Complete dark-theme design system. CSS custom properties, all component styles (cards, tables, badges, probability cards, CI bars, charts). Responsive. |
| `js/api.js` | **All** fetch calls to `http://localhost:8080/api`. One function per endpoint. Import this before using any API data. |
| `js/app.js` | Main application. Server health polling, team loading, tab routing, all event listeners, all DOM rendering (prediction cards, EV table, MC standings, H2H table). |
| `js/charts.js` | Three Chart.js renderers: `renderPredictionChart()` (bar + custom CI error bars), `renderEVChart()` (EV bars + probability line), `renderMonteCarloChart()` (horizontal multi-dataset). |

---

## 🚀 Quick Start

### 1. Install Julia
Download from https://julialang.org/downloads/ (Julia 1.9+)

### 2. Install dependencies
```bash
cd epl-predictor/backend
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### 3. Start the backend
```bash
julia --project=. server.jl
```
- **First run**: trains XGBoost + saves model to `data/footballmodel.jlso` (~5–10 min)
- **Subsequent runs**: loads saved model (~30 seconds)
- Watch for: `Starting HTTP server on http://127.0.0.1:8080`

### 4. Open the frontend
Open `frontend/index.html` in your browser.
The status dot turns **green** when the model is ready.

---

## 🔌 REST API Reference

All endpoints return JSON. CORS is open (`*`).

### `GET /api/health`
```json
{ "status": "ok", "model": "ready", "teams": 47, "timestamp": "..." }
```

### `GET /api/teams`
Returns full team list + ELO ratings dictionary.

### `POST /api/predict`
```json
// Request
{ "home": "Arsenal", "away": "Chelsea", "confidence": true }

// Response
{
  "home_win": 0.4821, "draw": 0.2634, "away_win": 0.2545,
  "confidence_intervals": {
    "home_win": { "low": 0.41, "high": 0.55 },
    "draw":     { "low": 0.21, "high": 0.32 },
    "away_win": { "low": 0.19, "high": 0.31 }
  },
  "elo_home": 1623.4, "elo_away": 1589.1,
  "predicted_result": "Home Win",
  "n_bootstrap": 300
}
```
`confidence: true` runs 300 bootstrap samples → 95% CI. Set `false` for faster point estimates.

### `POST /api/betting-ev`
```json
// Request
{ "home": "Arsenal", "away": "Chelsea", "odds": { "home": 2.10, "draw": 3.50, "away": 3.80 } }

// Response
{
  "vig_pct": 5.84,
  "has_value": true,
  "analyses": [
    { "outcome": "Home Win", "expected_value": 0.0122, "kelly_fraction": 0.0241,
      "is_value_bet": true, "edge": 0.037, ... }
  ],
  "best_bet": { "outcome": "Home Win", "expected_value": 0.0122, "kelly_fraction": 0.024 }
}
```

### `POST /api/monte-carlo`
```json
// Request
{ "n_sims": 5000 }

// Response
{
  "n_simulations": 5000,
  "standings": [
    { "position": 1, "team": "Man City", "mean_points": 87.3, "std_points": 4.1,
      "ci_low": 81, "ci_high": 95,
      "prob_title": 58.2, "prob_top4": 96.1, "prob_relegation": 0.0 },
    ...
  ]
}
```

### `GET /api/team-stats?team=Arsenal`
Returns aggregated historical stats: win rates, goal averages, SOT, ELO.

### `GET /api/head-to-head?home=Arsenal&away=Chelsea`
Returns H2H record + last 5 meetings.

---

## 🤖 ML Model Details

### Features (10 total — matching your Pluto notebook exactly)
| Feature | Description |
|---------|-------------|
| `EloHome` | Home team's ELO rating before the match |
| `EloAway` | Away team's ELO rating before the match |
| `HomeShotsOnTarget` | Home team's recent avg shots on target |
| `AwayShotsOnTarget` | Away team's recent avg shots on target |
| `HalfTimeHomeGoals` | Home team's recent avg HT goals |
| `HalfTimeAwayGoals` | Away team's recent avg HT goals |
| `HomeRecentGoals` | 5-game rolling avg of home goals scored |
| `AwayRecentGoals` | 5-game rolling avg of away goals scored |
| `HomeCorners` | Home team's recent avg corners |
| `AwayCorners` | Away team's recent avg corners |

### Training pipeline
1. Load 9,380 matches (2000–2024)
2. Sort chronologically by `MatchDate`
3. Compute ELO ratings in order (no leakage)
4. Compute rolling 5-game averages (no leakage)
5. 80/20 train/test split (shuffled, seed=42)
6. `XGBoostClassifier(num_round=100, eta=0.1, subsample=0.8)`
7. Grid search: `max_depth ∈ {3,4,5,6,7,8,9}` with 5-fold CV
8. Evaluate on held-out test set

### Confidence Intervals
Bootstrap resampling (n=300): for each bootstrap draw, resample from the team's recent 20 games, add Gaussian noise to ELO (σ=30), run model inference. Take 2.5th–97.5th percentile of the 300 probability draws as the 95% CI.

---

## 💰 Betting Math

**Expected Value** per £1 staked:
```
EV = (model_prob × decimal_odds) − 1
```
Positive EV = value bet (book is offering more than fair odds).

**No-Vig Probability** (removing bookmaker margin):
```
overround = 1/home_odds + 1/draw_odds + 1/away_odds   (typically ~1.06)
fair_home_prob = raw_home_prob / overround
```

**Half-Kelly Criterion** (optimal stake):
```
full_kelly = (b×p − q) / b    where b = odds−1, p = model_prob, q = 1−p
stake = 0.5 × full_kelly       (half-Kelly for risk management)
stake = clamp(stake, 0, 0.25)  (max 25% of bankroll)
```

---

## 🎲 Monte Carlo Details

- Round-robin: every team plays every other team home and away (380 fixtures for 20 teams)
- All probabilities pre-computed from XGBoost before simulations begin
- Simulations run in parallel with `Threads.@threads`
- Default: 3,000 simulations (~30 seconds). Max: 10,000
- Points CI: 5th–95th percentile of simulated final point totals
- Position determined by total points only (no goal difference tiebreaker)

---

## 🔧 Configuration

In `server.jl`:
- `HOST = "127.0.0.1"` — change to `"0.0.0.0"` to accept external connections
- `PORT = 8080`

In `monte_carlo.jl`:
- `CURRENT_EPL_TEAMS` — update for each season's actual clubs

In `js/api.js`:
- `API_BASE = "http://127.0.0.1:8080/api"` — update if deploying remotely
