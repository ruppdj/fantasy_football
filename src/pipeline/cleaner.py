"""
cleaner.py
Merges raw nfl_data_py and Sleeper data, normalizes fields,
handles nulls, and loads into SQLite tables.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text

POSITIONS = ["QB", "RB", "WR", "TE"]
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "db", "fantasy.db")
MIN_GAMES = 4  # minimum games played to be included in model data


def get_engine():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}")


def load_to_db(df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> None:
    engine = get_engine()
    with engine.connect() as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        conn.commit()
    print(f"Loaded {len(df)} rows into table '{table_name}'")


def read_from_db(table_name: str) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(f"SELECT * FROM {table_name}"), conn)


def clean_seasonal_stats(seasonal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and normalize season-level stats.
    - Keep only skill positions
    - Remove players with fewer than MIN_GAMES
    - Standardize column types
    """
    df = seasonal_df.copy()

    # Filter to skill positions
    if "position" in df.columns:
        df = df[df["position"].isin(POSITIONS)]

    # Minimum games filter
    if "games" in df.columns:
        df = df[df["games"] >= MIN_GAMES]

    # Ensure key numeric columns are float
    numeric_cols = [
        "fantasy_points_ppr", "targets", "receptions", "receiving_yards",
        "receiving_tds", "carries", "rushing_yards", "rushing_tds",
        "passing_yards", "passing_tds", "interceptions",
        "target_share", "air_yards_share", "wopr", "racr",
        "receiving_epa", "rushing_epa", "passing_epa",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)
    return df


def merge_rosters(seasonal_df: pd.DataFrame, rosters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join seasonal stats with roster metadata to get position, birth_date, IDs.
    rosters_df is a static player snapshot (one row per player, no season column),
    so the join key is player_id only.
    """
    roster_cols = [
        "player_id", "name", "position", "birth_date",
        "height", "weight", "college", "sleeper_id", "pfr_id",
    ]
    roster_cols = [c for c in roster_cols if c in rosters_df.columns]
    roster_slim = rosters_df[roster_cols].drop_duplicates(subset=["player_id"])

    merged = seasonal_df.merge(roster_slim, on="player_id", how="left", suffixes=("", "_roster"))

    # Use roster position if seasonal doesn't already have it
    if "position_roster" in merged.columns:
        merged["position"] = merged["position"].fillna(merged["position_roster"])
        merged.drop(columns=["position_roster"], inplace=True)

    # Use roster name as full_name if not present
    if "full_name" not in merged.columns and "name" in merged.columns:
        merged = merged.rename(columns={"name": "full_name"})
    elif "name" in merged.columns:
        merged.drop(columns=["name"], inplace=True)

    return merged


def merge_sleeper(nfl_df: pd.DataFrame, sleeper_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join NFL data with Sleeper player metadata using sleeper_id.
    Adds: status, injury_status, depth_chart_order, search_rank.
    """
    sleeper_cols = [
        "sleeper_id", "status", "injury_status", "depth_chart_order", "search_rank",
    ]
    sleeper_cols = [c for c in sleeper_cols if c in sleeper_df.columns]
    sleeper_slim = sleeper_df[sleeper_cols].drop_duplicates(subset=["sleeper_id"])

    if "sleeper_id" not in nfl_df.columns:
        print("Warning: sleeper_id not in NFL data — skipping Sleeper merge.")
        return nfl_df

    # Normalize sleeper_id to string on both sides before joining.
    # import_ids() returns it as float64 (e.g. 12345.0); Sleeper API returns it as str.
    nfl_df = nfl_df.copy()
    nfl_df["sleeper_id"] = nfl_df["sleeper_id"].apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    sleeper_slim = sleeper_slim.copy()
    sleeper_slim["sleeper_id"] = sleeper_slim["sleeper_id"].astype(str)

    merged = nfl_df.merge(sleeper_slim, on="sleeper_id", how="left")

    # Coalesce wopr columns — Sleeper merge can produce wopr_x/wopr_y collision.
    # wopr_y (from nfl_data_py seasonal data, 0–1 range) is the correct one.
    if "wopr_x" in merged.columns and "wopr_y" in merged.columns:
        merged["wopr"] = merged["wopr_y"].fillna(merged["wopr_x"])
        merged.drop(columns=["wopr_x", "wopr_y"], inplace=True)
    elif "wopr_x" in merged.columns:
        merged = merged.rename(columns={"wopr_x": "wopr"})

    return merged


def merge_snap_pct(nfl_df: pd.DataFrame, snap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join season-level avg_snap_pct from fetch_snap_counts() output.
    snap_df must have columns: player_id, season, avg_snap_pct.
    """
    return nfl_df.merge(snap_df, on=["player_id", "season"], how="left")


def compute_age_at_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute player age at the start of the NFL season (September 1st).
    Requires birth_date column (string YYYY-MM-DD).
    """
    df = df.copy()
    if "birth_date" not in df.columns:
        print("Warning: birth_date not found — skipping age computation.")
        return df

    df["birth_date_dt"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df["season_start"] = pd.to_datetime(df["season"].astype(str) + "-09-01")
    df["age_at_season"] = (
        (df["season_start"] - df["birth_date_dt"]).dt.days / 365.25
    ).round(1)
    df.drop(columns=["birth_date_dt", "season_start"], inplace=True)
    return df


def merge_draft_data(nfl_df: pd.DataFrame, draft_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join draft pick data onto the main dataset via player_id (gsis_id).
    Adds: draft_pick, draft_round, age_at_draft, is_undrafted, cfb_player_id.
    Players with no draft record get is_undrafted=1 and null draft features.
    """
    draft_cols = ["player_id", "pick", "round", "age", "cfb_player_id", "season"]
    draft_cols = [c for c in draft_cols if c in draft_df.columns]
    draft_slim = (
        draft_df[draft_cols]
        .dropna(subset=["player_id"])
        .drop_duplicates(subset=["player_id"])
        .rename(columns={"pick": "draft_pick", "round": "draft_round", "age": "age_at_draft",
                         "season": "draft_season"})
    )

    merged = nfl_df.merge(draft_slim, on="player_id", how="left")
    merged["is_undrafted"] = merged["draft_pick"].isna().astype(int)
    return merged


def _parse_height_to_inches(h) -> float:
    """Convert '6-3' (feet-inches) to total inches (75.0). Returns NaN on failure."""
    if pd.isna(h):
        return float("nan")
    try:
        feet, inches = str(h).split("-")
        return int(feet) * 12 + int(inches)
    except Exception:
        return pd.to_numeric(h, errors="coerce")


def merge_combine_data(nfl_df: pd.DataFrame, combine_df: pd.DataFrame,
                       rosters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join combine measurements onto the main dataset.
    Combine data uses pfr_id as key — we route through rosters_df to get player_id.
    Adds: combine_forty, combine_weight, combine_height, combine_vertical, combine_bench.
    Note: combine height is stored as "feet-inches" string (e.g. "6-3") and is
    converted to total inches (75.0) so sklearn's SimpleImputer can process it.
    """
    combine_cols = ["pfr_id", "forty", "wt", "ht", "vertical", "bench"]
    combine_cols = [c for c in combine_cols if c in combine_df.columns]
    combine_slim = (
        combine_df[combine_cols]
        .drop_duplicates(subset=["pfr_id"])
        .rename(columns={"forty": "combine_forty", "wt": "combine_weight",
                         "ht": "combine_height", "vertical": "combine_vertical",
                         "bench": "combine_bench"})
    )
    # Convert height from "6-3" string format to numeric inches
    if "combine_height" in combine_slim.columns:
        combine_slim["combine_height"] = combine_slim["combine_height"].apply(
            _parse_height_to_inches
        )

    # Map pfr_id → player_id via rosters
    if "pfr_id" in rosters_df.columns and "player_id" in rosters_df.columns:
        id_map = rosters_df[["player_id", "pfr_id"]].dropna().drop_duplicates("player_id")
        combine_slim = combine_slim.merge(id_map, on="pfr_id", how="left")
        combine_slim = combine_slim.drop(columns=["pfr_id"]).dropna(subset=["player_id"])
        merged = nfl_df.merge(combine_slim, on="player_id", how="left")
    else:
        print("Warning: pfr_id not in rosters — skipping combine merge.")
        merged = nfl_df

    return merged


def merge_injury_data(nfl_df: pd.DataFrame, injury_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join season-level injury features onto the main dataset.
    injury_df must have: player_id, season, games_missed, ir_flag.
    Adds: games_missed (current season), ir_flag (current season).
    Lagged injury features (prev1) are computed in features.py.
    """
    injury_slim = injury_df[["player_id", "season", "games_missed", "ir_flag"]].copy()
    return nfl_df.merge(injury_slim, on=["player_id", "season"], how="left")


def merge_college_stats(nfl_df: pd.DataFrame, college_df: pd.DataFrame,
                        draft_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join college production stats onto the main dataset for drafted players.
    Routes through draft_df (which has cfb_player_id + player_id) to link
    college stats to NFL player_id.

    College stats are for the final college season — only meaningful for rookies,
    but kept for all seasons as a static player attribute.
    """
    if college_df.empty:
        print("Warning: college stats empty — skipping college merge.")
        return nfl_df

    college_cols = [
        "cfb_player_id", "college_rec_yards", "college_rec_tds", "college_targets",
        "college_rush_yards", "college_rush_tds", "college_rush_atts",
        "college_dominator_rate", "college_years",
    ]
    college_cols = [c for c in college_cols if c in college_df.columns]
    college_slim = college_df[college_cols].dropna(subset=["cfb_player_id"]).copy()
    college_slim["cfb_player_id"] = college_slim["cfb_player_id"].astype(str)

    # Get cfb_player_id → player_id mapping from draft data
    if "cfb_player_id" not in draft_df.columns or "player_id" not in draft_df.columns:
        print("Warning: cfb_player_id not in draft data — skipping college merge.")
        return nfl_df

    id_map = (
        draft_df[["player_id", "cfb_player_id"]]
        .dropna()
        .drop_duplicates("player_id")
        .copy()
    )
    id_map["cfb_player_id"] = id_map["cfb_player_id"].astype(str)

    college_with_id = college_slim.merge(id_map, on="cfb_player_id", how="inner")
    college_with_id = college_with_id.drop(columns=["cfb_player_id"]).drop_duplicates("player_id")

    return nfl_df.merge(college_with_id, on="player_id", how="left")


def build_clean_dataset(
    seasonal_df: pd.DataFrame,
    rosters_df: pd.DataFrame,
    sleeper_df: pd.DataFrame,
    snap_df: pd.DataFrame,
    draft_df: pd.DataFrame = None,
    combine_df: pd.DataFrame = None,
    injury_df: pd.DataFrame = None,
    college_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Full v2 cleaning pipeline: merges all sources, computes age, loads to DB.
    New optional args (draft_df, combine_df, injury_df, college_df) add dynasty features.
    Pass None to skip any optional source (v1 behaviour preserved).
    Returns the merged, cleaned DataFrame.
    """
    print("Cleaning seasonal stats...")
    df = clean_seasonal_stats(seasonal_df)

    print("Merging roster metadata...")
    df = merge_rosters(df, rosters_df)

    print("Merging Sleeper metadata...")
    df = merge_sleeper(df, sleeper_df)

    print("Merging snap pct...")
    df = merge_snap_pct(df, snap_df)

    print("Computing age at season...")
    df = compute_age_at_season(df)

    if draft_df is not None:
        print("Merging draft data...")
        df = merge_draft_data(df, draft_df)

    if combine_df is not None and rosters_df is not None:
        print("Merging combine data...")
        df = merge_combine_data(df, combine_df, rosters_df)

    if injury_df is not None:
        print("Merging injury data...")
        df = merge_injury_data(df, injury_df)

    if college_df is not None and draft_df is not None:
        print("Merging college stats...")
        df = merge_college_stats(df, college_df, draft_df)

    print("Loading to SQLite (nfl_stats)...")
    load_to_db(df, "nfl_stats")

    print(f"Clean dataset: {len(df)} rows, {df.shape[1]} columns")
    return df
