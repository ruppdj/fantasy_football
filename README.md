# Dynasty Fantasy Football Player Value Predictor

A data science project that predicts next-season PPR fantasy football output for skill-position players (QB, RB, WR, TE) — designed as a dynasty draft and roster management tool.

This project is structured as an **explicit iteration**, showing how a data science model evolves as new data sources are identified, gaps are addressed, and the approach is refined.

## Iterations

### v1 — Base Model (seasonal stats only)

The initial model built the core pipeline: pull NFL seasonal stats, merge with Sleeper player metadata and snap count data, engineer lagged production features, and train a regression model to predict next-season PPR output.

**Data sources:** nfl_data_py seasonal stats, Sleeper player API, snap counts  
**Features (~19):** prior-season PPR (1yr, 2yr), target share, air yards share, snap %, age, years experience, EPA metrics, position dummies  
**Best model:** Ridge Regression (validation MAE ≈ 46.3)  
**Gap identified:** Rookies were excluded entirely — their lagged PPR was null so they were dropped. Dynasty values rookie upside highly. No injury data. No pre-NFL signal (draft capital, combine, college production).

### v2 — Unified Dynasty Model (rookies + pre-NFL data)

v2 addresses the gaps identified in v1 by adding four new data sources and a revised rookie inclusion strategy.

**New data sources:**
- NFL draft picks (`nfl_data_py.import_draft_picks`) — draft capital, age at draft, college player ID
- NFL combine data (`nfl_data_py.import_combine_data`) — athleticism profile (40-yard dash, weight, vertical)
- Injury reports (`nfl_data_py.import_injuries`) — aggregated to games missed and IR flag per player-season
- College Football Data API — final college season receiving/rushing production, dominator rate

**Rookie inclusion strategy:** Rather than dropping players with no prior NFL stats, their lagged stats are set to 0 (accurate — they scored 0 NFL PPR before their first season). An `is_rookie` flag is added. Draft capital and college production carry the predictive signal for these players.

**Features (~40):** All v1 features + `is_rookie`, injury history, draft capital, combine athleticism, college production  
**Best model:** Evaluated in `v2_04_modeling.ipynb`  
**Documented in:** `v2_05_iteration_comparison.ipynb` — side-by-side v1 vs v2 results, feature importance comparison, gap analysis, and v3 roadmap

---

## Project Structure

```
fantasy_football/
├── data/
│   ├── raw/          # parquet cache files (per-season, interrupt-safe)
│   └── processed/    # reserved for intermediate outputs
├── db/
│   └── fantasy.db    # SQLite database (model_ready table)
├── notebooks/
│   ├── v1_01_data_gathering.ipynb       # v1: pull NFL stats + Sleeper data
│   ├── v1_02_eda.ipynb                  # v1: EDA — age curves, correlations
│   ├── v1_03_pipeline_cleaning.ipynb    # v1: merge, clean, build model_ready
│   ├── v1_04_modeling.ipynb             # v1: train models, dynasty rankings
│   ├── v2_01_data_gathering.ipynb       # v2: add draft, combine, injuries, college
│   ├── v2_03_pipeline_cleaning.ipynb    # v2: updated pipeline with new sources
│   ├── v2_04_modeling.ipynb             # v2: model with expanded feature set
│   └── v2_05_iteration_comparison.ipynb # v1 vs v2: what changed, what improved
├── src/
│   ├── fetchers/
│   │   ├── nfl_fetcher.py      # nfl_data_py wrappers (stats, snaps, draft, combine, injuries)
│   │   ├── sleeper_fetcher.py  # Sleeper REST API wrappers
│   │   └── college_fetcher.py  # College Football Data API (CFBD)
│   ├── pipeline/
│   │   ├── cleaner.py          # merge all sources, compute age, load to SQLite
│   │   └── features.py         # feature engineering, rookie flag, target variable
│   └── models/
│       └── predictor.py        # model training, evaluation, feature importance
├── test_fetchers.py             # smoke tests for all 9 data fetchers
└── requirements.txt
```

## Data Sources

| Source | What it provides | Auth |
|--------|-----------------|------|
| [nfl_data_py](https://github.com/nflverse/nfl_data_py) | Seasonal stats, weekly data, snap counts, player IDs, draft picks, combine, injuries | None (public) |
| [Sleeper API](https://docs.sleeper.com/) | Player metadata, dynasty context | None (public) |
| [College Football Data API](https://collegefootballdata.com) | Final college season production per drafted player | None (public) |

## Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Project

Run notebooks in version order. v1 notebooks must run before v2.

```bash
cd notebooks
jupyter notebook
```

### v1 (base model)

| Notebook | What it does | Prerequisite |
|----------|-------------|--------------|
| `v1_01_data_gathering` | Fetch NFL stats + Sleeper data, cache to parquet, load to SQLite | None |
| `v1_02_eda` | Age curves, correlations, YoY consistency — loads from parquet cache | Run v1_01 |
| `v1_03_pipeline_cleaning` | Merge sources, engineer features, build `model_ready` table | Run v1_01 |
| `v1_04_modeling` | Train models, compare results, output dynasty rankings | Run v1_03 |

### v2 (unified dynasty model)

| Notebook | What it does | Prerequisite |
|----------|-------------|--------------|
| `v2_01_data_gathering` | Fetch draft picks, combine, injuries, college stats | Run v1_01 |
| `v2_03_pipeline_cleaning` | Rebuild pipeline with all v2 sources, validate rookie inclusion | Run v2_01 |
| `v2_04_modeling` | Train with expanded feature set, rookie-specific analysis | Run v2_03 |
| `v2_05_iteration_comparison` | Side-by-side v1 vs v2 results, gap analysis, v3 roadmap | Run v1_04 + v2_04 |

## Testing

Smoke tests validate all 9 data fetchers before running the full pipeline:

```bash
python test_fetchers.py
```

Each test fetches a single season (2023) and validates shape, columns, value ranges, and known player presence. Run this before kicking off data gathering to confirm all APIs are reachable.

**Tests covered:**
- NFL seasonal stats, rosters, weekly data, snap counts
- NFL draft picks, combine data, injury reports
- Sleeper player metadata
- College Football Data API (CFBD)

## Checkpointing & Rate Limiting

Each season's data is saved to its own parquet file (e.g. `data/raw/seasonal_stats_2023.parquet`) immediately after download. If the fetch is interrupted, already-saved seasons are skipped on retry.

A 1-second delay is inserted between season requests to stay polite to the nflverse CDN and Sleeper API. The College Football Data API uses 1.5s delays. Configurable via `REQUEST_DELAY_SECONDS` at the top of each fetcher.

## Model

All versions train and compare:
- **Naive baseline** — predict next year = this year
- **Linear / Ridge Regression** — interpretable
- **Random Forest** — handles age curves and non-linearities
- **Gradient Boosting / XGBoost** — typically best performance

**Train/Val/Test split:** 2016–2021 / 2022 / 2023 (time-based to prevent data leakage)  
**Target variable:** `ppr_pts_next` — PPR fantasy points scored in the following season

## Output

The final output of each modeling notebook is a ranked player list with predicted next-season PPR points. v2 adds a separate rookie rankings table — useful for dynasty startup drafts and devy leagues.

## Dependencies

See [requirements.txt](requirements.txt). Key packages:

- `nfl_data_py` — NFL stats ingestion
- `pandas`, `numpy` — data manipulation
- `sqlalchemy` — SQLite ORM
- `scikit-learn` — models and evaluation
- `xgboost` — gradient boosting
- `matplotlib`, `seaborn` — visualization
- `requests` — Sleeper API + College Football Data API
- `jupyter` — notebook interface
