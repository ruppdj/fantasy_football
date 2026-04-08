"""
predictor.py
Model training, evaluation, and prediction for dynasty player value.
Implements baseline, linear regression, random forest, and XGBoost models.
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed — XGBoost model will be skipped.")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "db", "fantasy.db")

FEATURE_COLS = [
    # Rookie flag
    "is_rookie",
    # Core NFL stats
    "age_at_season", "years_exp",
    "ppr_pts_prev1", "ppr_pts_prev2",
    "fantasy_points_ppr", "ppr_per_game",
    "td_rate",
    # Usage
    "target_share", "air_yards_share", "wopr",
    "targets_per_game", "carries_per_game",
    "avg_snap_pct",
    # Advanced NFL
    "receiving_epa", "rushing_epa", "passing_epa",
    "racr",
    # Injury history
    "games_missed_prev1", "career_games_missed_rate", "ir_flag_prev1",
    # Draft capital
    "draft_pick", "draft_round", "age_at_draft", "is_undrafted",
    # Combine athleticism
    "combine_forty", "combine_weight", "combine_height",
    "combine_vertical", "combine_bench",
    # College production
    "college_rec_yards", "college_rec_tds", "college_targets",
    "college_rush_yards", "college_rush_tds",
    "college_dominator_rate", "college_years",
    # Position dummies
    "pos_QB", "pos_RB", "pos_WR", "pos_TE",
]

TARGET_COL = "ppr_pts_next"

# Train: 2016-2021 | Val: 2022 | Test: 2023
TRAIN_END = 2021
VAL_YEAR = 2022
TEST_YEAR = 2023


def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}")


def load_model_data() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text("SELECT * FROM model_ready"), conn)


def split_data(df: pd.DataFrame):
    """
    Time-based train/val/test split to avoid data leakage.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test, meta_test)
    """
    available_features = [c for c in FEATURE_COLS if c in df.columns]

    train = df[df["season"] <= TRAIN_END]
    val = df[df["season"] == VAL_YEAR]
    test = df[df["season"] == TEST_YEAR]

    X_train = train[available_features]
    y_train = train[TARGET_COL]
    X_val = val[available_features]
    y_val = val[TARGET_COL]
    X_test = test[available_features]
    y_test = test[TARGET_COL]

    # Keep metadata for prediction output
    meta_cols = ["player_id", "full_name", "position", "season"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta_test = test[meta_cols].copy()

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, meta_test


def evaluate(name: str, y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:30s} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | R²: {r2:.3f}")
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2}


def build_pipeline(model):
    """Wrap a model in an impute → scale → model sklearn Pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def train_and_evaluate(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Train all models, evaluate on val and test sets.
    Returns (results_dict, test_predictions_df).
    """
    X_train, y_train, X_val, y_val, X_test, y_test, meta_test = split_data(df)

    models = {
        "Naive Baseline (prev year)": None,  # special case
        "Linear Regression": build_pipeline(LinearRegression()),
        "Ridge Regression": build_pipeline(Ridge(alpha=1.0)),
        "Random Forest": build_pipeline(
            RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        ),
        "Gradient Boosting": build_pipeline(
            GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = build_pipeline(
            XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                         random_state=42, verbosity=0)
        )

    results = []
    best_model = None
    best_val_mae = float("inf")
    best_preds = None

    print("\n=== Validation Results ===")
    for name, pipeline in models.items():
        if name == "Naive Baseline (prev year)":
            available = [c for c in FEATURE_COLS if c in X_val.columns]
            if "ppr_pts_prev1" in available:
                val_preds = X_val["ppr_pts_prev1"].fillna(0)
                test_preds = X_test["ppr_pts_prev1"].fillna(0)
            else:
                val_preds = pd.Series([y_train.mean()] * len(y_val), index=y_val.index)
                test_preds = pd.Series([y_train.mean()] * len(y_test), index=y_test.index)
        else:
            pipeline.fit(X_train, y_train)
            val_preds = pipeline.predict(X_val)
            test_preds = pipeline.predict(X_test)

        res = evaluate(name, y_val, val_preds)
        results.append(res)

        if res["mae"] < best_val_mae:
            best_val_mae = res["mae"]
            best_model = name
            best_preds = test_preds

    print(f"\nBest model on validation: {best_model} (MAE={best_val_mae:.2f})")

    print("\n=== Test Results (best model) ===")
    evaluate(best_model, y_test, best_preds)

    # Build prediction output DataFrame
    pred_df = meta_test.copy()
    pred_df["actual_ppr_next"] = y_test.values
    pred_df["predicted_ppr_next"] = best_preds
    pred_df["error"] = pred_df["predicted_ppr_next"] - pred_df["actual_ppr_next"]
    pred_df = pred_df.sort_values("predicted_ppr_next", ascending=False).reset_index(drop=True)
    pred_df["predicted_rank"] = pred_df.index + 1

    return {"results": results, "best_model": best_model}, pred_df


def get_feature_importance(df: pd.DataFrame, model_name: str = "Random Forest") -> pd.DataFrame:
    """
    Fit a Random Forest on full training data and return feature importances.
    """
    X_train, y_train, _, _, _, _, _ = split_data(df)
    available_features = [c for c in FEATURE_COLS if c in X_train.columns]

    pipeline = build_pipeline(
        RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    )
    pipeline.fit(X_train, y_train)

    importances = pipeline.named_steps["model"].feature_importances_
    feat_df = pd.DataFrame({
        "feature": available_features,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return feat_df
