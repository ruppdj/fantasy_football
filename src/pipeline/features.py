"""
features.py
Feature engineering for the dynasty player value model (v2).
Takes the cleaned merged DataFrame and produces a model-ready dataset
with lagged stats, target variable, rookie flags, injury history,
draft capital, combine athleticism, and college production features.

v2 change: rookies are INCLUDED with NFL stat features set to 0 (true zero —
they had no prior NFL production). Draft capital and college stats carry
the predictive signal for rookies.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "db", "fantasy.db")
POSITIONS = ["QB", "RB", "WR", "TE"]
MIN_GAMES_TARGET = 4  # target year must also have enough games


def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}")


def add_rookie_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_rookie flag: 1 for a player's first NFL season, 0 otherwise.
    Uses draft_season (from draft picks join) when available — this correctly
    identifies rookies even when the dataset starts mid-career (e.g. 2012 dataset
    would otherwise tag Brady/Manning as rookies via min(season)).
    Falls back to min(season) per player if draft_season is absent.
    """
    df = df.copy()
    if "draft_season" in df.columns:
        df["is_rookie"] = (df["season"] == df["draft_season"]).astype(int)
    else:
        first_season = df.groupby("player_id")["season"].transform("min")
        df["is_rookie"] = (df["season"] == first_season).astype(int)
    return df


def add_lagged_ppr(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player-season, add PPR points from the prior 1 and 2 seasons.
    Rookies have no prior NFL history — their lagged values are set to 0
    (true zero, not missing) so they remain in the model dataset.
    """
    df = df.sort_values(["player_id", "season"]).copy()
    df["ppr_pts_prev1"] = df.groupby("player_id")["fantasy_points_ppr"].shift(1)
    df["ppr_pts_prev2"] = df.groupby("player_id")["fantasy_points_ppr"].shift(2)
    # Fill rookie zeros — they genuinely scored 0 in prior NFL seasons
    df["ppr_pts_prev1"] = df["ppr_pts_prev1"].fillna(0)
    df["ppr_pts_prev2"] = df["ppr_pts_prev2"].fillna(0)
    return df


def add_lagged_injury(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add prior-season injury features:
    - games_missed_prev1: games missed last season (0 for rookies)
    - ir_flag_prev1: 1 if on IR last season (0 for rookies)
    Also compute career_games_missed_rate: avg games missed per season to date.
    """
    df = df.sort_values(["player_id", "season"]).copy()

    if "games_missed" in df.columns:
        df["games_missed_prev1"] = df.groupby("player_id")["games_missed"].shift(1).fillna(0)
        df["career_games_missed_rate"] = (
            df.groupby("player_id")["games_missed"]
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
            .shift(1)
            .fillna(0)
        )
    else:
        df["games_missed_prev1"] = 0
        df["career_games_missed_rate"] = 0

    if "ir_flag" in df.columns:
        df["ir_flag_prev1"] = df.groupby("player_id")["ir_flag"].shift(1).fillna(0)
    else:
        df["ir_flag_prev1"] = 0

    return df


def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add next-season PPR points as the regression target.
    Only rows where the next season also meets the MIN_GAMES_TARGET threshold
    are kept as valid training samples.
    """
    df = df.sort_values(["player_id", "season"]).copy()
    df["ppr_pts_next"] = df.groupby("player_id")["fantasy_points_ppr"].shift(-1)
    df["games_next"] = df.groupby("player_id")["games"].shift(-1)

    # Mark rows where next season is valid (not the final year, enough games)
    df["has_next_season"] = (
        df["ppr_pts_next"].notna() & (df["games_next"] >= MIN_GAMES_TARGET)
    )
    return df


def add_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute team-level context features:
    - td_rate: TDs per game
    - team_pass_volume: passing attempts per game (proxy via targets per game)
    """
    df = df.copy()

    # TDs per game (position-appropriate)
    total_tds = df.get("receiving_tds", 0).fillna(0) + \
                df.get("rushing_tds", 0).fillna(0) + \
                df.get("passing_tds", 0).fillna(0)
    df["td_rate"] = total_tds / df["games"].replace(0, np.nan)

    # Targets per game (WR/TE/RB receiving usage proxy)
    if "targets" in df.columns:
        df["targets_per_game"] = df["targets"] / df["games"].replace(0, np.nan)

    # Carries per game (RB rushing usage proxy)
    if "carries" in df.columns:
        df["carries_per_game"] = df["carries"] / df["games"].replace(0, np.nan)

    # PPR points per game (efficiency measure)
    df["ppr_per_game"] = df["fantasy_points_ppr"] / df["games"].replace(0, np.nan)

    return df


def encode_position(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the position column."""
    df = df.copy()
    pos_dummies = pd.get_dummies(df["position"], prefix="pos")
    # Ensure all four positions are present as columns
    for pos in POSITIONS:
        col = f"pos_{pos}"
        if col not in pos_dummies.columns:
            pos_dummies[col] = 0
    df = pd.concat([df, pos_dummies], axis=1)
    return df


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select rows and columns for modeling.
    v2: rookies included (is_rookie=1, lagged NFL stats=0).
    Only drops rows missing the target or age.
    """
    feature_cols = [
        # Identifiers
        "player_id", "full_name", "position", "season",
        # Target
        "ppr_pts_next",
        # Rookie flag
        "is_rookie",
        # Core NFL stats
        "age_at_season",
        "ppr_pts_prev1", "ppr_pts_prev2",
        "fantasy_points_ppr",
        "ppr_per_game",
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

    available = [c for c in feature_cols if c in df.columns]
    model_df = df[df["has_next_season"] == True][available].copy()

    # Only require target and age — everything else handled by imputer
    model_df = model_df.dropna(subset=["ppr_pts_next", "age_at_season"])

    rookie_count = model_df["is_rookie"].sum() if "is_rookie" in model_df.columns else 0
    print(f"Model-ready dataset: {len(model_df)} rows, {len(available)} features "
          f"({int(rookie_count)} rookies included)")
    return model_df


def build_model_dataset(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full v2 feature engineering pipeline. Returns model-ready DataFrame
    and saves it to SQLite as 'model_ready'.
    """
    df = clean_df.copy()

    print("Adding rookie flag...")
    df = add_rookie_flag(df)

    print("Adding lagged PPR features (rookies → 0)...")
    df = add_lagged_ppr(df)

    print("Adding lagged injury features...")
    df = add_lagged_injury(df)

    print("Adding target variable...")
    df = add_target_variable(df)

    print("Adding usage features...")
    df = add_usage_features(df)

    print("Encoding position...")
    df = encode_position(df)

    print("Selecting model features...")
    model_df = select_model_features(df)

    print("Saving model_ready table to SQLite...")
    engine = get_engine()
    with engine.connect() as conn:
        model_df.to_sql("model_ready", conn, if_exists="replace", index=False)
        conn.commit()
    print(f"Saved {len(model_df)} rows to model_ready table")

    return model_df
