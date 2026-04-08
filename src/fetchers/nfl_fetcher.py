"""
nfl_fetcher.py
Wrappers around nfl_data_py to pull seasonal stats, weekly data, and roster info.

Checkpointing: each season is saved to its own parquet file (e.g. seasonal_stats_2021.parquet)
before being combined. If a fetch is interrupted, already-saved seasons are skipped on retry.

Rate limiting: REQUEST_DELAY_SECONDS is inserted between each season fetch so we don't
hammer the nflverse GitHub CDN with back-to-back requests.
"""

import os
import time
import nfl_data_py as nfl
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
POSITIONS = ["QB", "RB", "WR", "TE"]
DEFAULT_SEASONS = list(range(2012, 2025))

# Pause between each season fetch — keeps us polite to the nflverse CDN
REQUEST_DELAY_SECONDS = 1.0


def _cache_path(name: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    return os.path.join(RAW_DIR, f"{name}.parquet")


def _season_cache_path(prefix: str, season: int) -> str:
    """Per-season checkpoint file, e.g. data/raw/seasonal_stats_2021.parquet"""
    os.makedirs(RAW_DIR, exist_ok=True)
    return os.path.join(RAW_DIR, f"{prefix}_{season}.parquet")


def fetch_seasonal_stats(seasons: list = DEFAULT_SEASONS, force: bool = False) -> pd.DataFrame:
    """
    Pull season-level player stats from nfl_data_py, one season at a time.

    Each season is checkpointed to data/raw/seasonal_stats_{year}.parquet immediately
    after download. Already-saved seasons are skipped on retry — only missing seasons
    are fetched. The combined result is saved to data/raw/seasonal_stats.parquet.

    Note: always iterates through per-season checkpoints rather than reading the
    combined file, so adding new seasons or running tests against a single season
    never masks missing data on subsequent full runs.
    """
    combined_cache = _cache_path("seasonal_stats")
    frames = []
    for i, season in enumerate(seasons):
        season_cache = _season_cache_path("seasonal_stats", season)
        if not force and os.path.exists(season_cache):
            print(f"  [{i+1}/{len(seasons)}] {season} — loaded from checkpoint")
            frames.append(pd.read_parquet(season_cache))
            continue

        print(f"  [{i+1}/{len(seasons)}] {season} — fetching...", end=" ", flush=True)
        df = nfl.import_seasonal_data([season])
        df = df[df["season_type"] == "REG"].copy()
        df.to_parquet(season_cache, index=False)
        frames.append(df)
        print(f"{len(df)} rows saved")

        if i < len(seasons) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(combined_cache, index=False)
    print(f"Combined seasonal stats: {len(combined)} rows saved to {combined_cache}")
    return combined


def fetch_weekly_stats(seasons: list = DEFAULT_SEASONS, force: bool = False) -> pd.DataFrame:
    """
    Pull week-level player stats from nfl_data_py, one season at a time.

    Each season checkpointed to data/raw/weekly_stats_{year}.parquet.
    Combined result saved to data/raw/weekly_stats.parquet.
    Always iterates per-season checkpoints — see fetch_seasonal_stats for rationale.
    """
    combined_cache = _cache_path("weekly_stats")
    frames = []
    for i, season in enumerate(seasons):
        season_cache = _season_cache_path("weekly_stats", season)
        if not force and os.path.exists(season_cache):
            print(f"  [{i+1}/{len(seasons)}] {season} — loaded from checkpoint")
            frames.append(pd.read_parquet(season_cache))
            continue

        print(f"  [{i+1}/{len(seasons)}] {season} — fetching...", end=" ", flush=True)
        df = nfl.import_weekly_data([season])
        df = df[df["season_type"] == "REG"].copy()
        df.to_parquet(season_cache, index=False)
        frames.append(df)
        print(f"{len(df)} rows saved")

        if i < len(seasons) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(combined_cache, index=False)
    print(f"Combined weekly stats: {len(combined)} rows saved to {combined_cache}")
    return combined


def fetch_rosters(force: bool = False) -> pd.DataFrame:
    """
    Pull player metadata and cross-reference IDs via nfl.import_ids().
    Returns one row per player with: player_id (gsis_id), sleeper_id,
    pfr_id, birthdate, position, college, height, weight.
    Cached at data/raw/rosters.parquet.

    Note: import_ids() is a static snapshot (not per-season), so this
    is used as a lookup table joined to seasonal data on player_id.
    """
    cache = _cache_path("rosters")
    if not force and os.path.exists(cache):
        print(f"Loading rosters from cache: {cache}")
        return pd.read_parquet(cache)

    print("Fetching player ID map via import_ids()...")
    df = nfl.import_ids()

    # Rename gsis_id → player_id to match the seasonal/weekly data join key
    df = df.rename(columns={"gsis_id": "player_id", "birthdate": "birth_date"})

    keep_cols = [
        "player_id", "sleeper_id", "pfr_id", "name",
        "position", "birth_date", "college", "height", "weight",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].dropna(subset=["player_id"]).copy()
    df = df[df["position"].isin(POSITIONS)]

    df.to_parquet(cache, index=False)
    print(f"Saved {len(df)} skill-position players to {cache}")
    return df


def fetch_snap_counts(seasons: list = DEFAULT_SEASONS, force: bool = False) -> pd.DataFrame:
    """
    Pull offensive snap count data via nfl.import_snap_counts() and
    aggregate to season-level avg offense_pct per player.

    snap_counts uses pfr_player_id — we join via import_ids() to get
    the nflfastR player_id (gsis_id) for joining with seasonal stats.

    Returns DataFrame with columns: player_id, season, avg_snap_pct.
    Cached at data/raw/snap_counts.parquet.
    """
    cache = _cache_path("snap_counts")

    # Build pfr_id → player_id (gsis_id) mapping once — used for every season
    ids = nfl.import_ids()[["gsis_id", "pfr_id"]].dropna(subset=["gsis_id", "pfr_id"])
    ids = ids.rename(columns={"gsis_id": "player_id"})

    frames = []
    for i, season in enumerate(seasons):
        season_cache = _season_cache_path("snap_counts", season)
        if not force and os.path.exists(season_cache):
            print(f"  [{i+1}/{len(seasons)}] {season} — loaded from checkpoint")
            frames.append(pd.read_parquet(season_cache))
            continue

        print(f"  [{i+1}/{len(seasons)}] {season} — fetching...", end=" ", flush=True)
        snaps = nfl.import_snap_counts([season])
        snaps = snaps[snaps["game_type"] == "REG"].copy()
        snaps = snaps.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="left")
        agg = (
            snaps[snaps["player_id"].notna()]
            .groupby(["player_id", "season"])["offense_pct"]
            .mean()
            .reset_index()
            .rename(columns={"offense_pct": "avg_snap_pct"})
        )
        agg.to_parquet(season_cache, index=False)
        frames.append(agg)
        print(f"{len(agg)} player-season records saved")

        if i < len(seasons) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(cache, index=False)
    print(f"Combined snap counts: {len(combined)} records saved to {cache}")
    return combined


def fetch_draft_picks(seasons: list = DEFAULT_SEASONS, force: bool = False) -> pd.DataFrame:
    """
    Pull NFL draft pick data via nfl.import_draft_picks(), one season at a time.
    Key columns: gsis_id (→ player_id), pick, round, age, college, cfb_player_id, car_av.
    Cached per season and combined at data/raw/draft_picks.parquet.
    """
    combined_cache = _cache_path("draft_picks")
    frames = []
    for i, season in enumerate(seasons):
        season_cache = _season_cache_path("draft_picks", season)
        if not force and os.path.exists(season_cache):
            print(f"  [{i+1}/{len(seasons)}] {season} — loaded from checkpoint")
            frames.append(pd.read_parquet(season_cache))
            continue

        print(f"  [{i+1}/{len(seasons)}] {season} — fetching...", end=" ", flush=True)
        df = nfl.import_draft_picks([season])
        # Rename gsis_id → player_id for consistent join key
        if "gsis_id" in df.columns:
            df = df.rename(columns={"gsis_id": "player_id"})
        df.to_parquet(season_cache, index=False)
        frames.append(df)
        print(f"{len(df)} picks saved")

        if i < len(seasons) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(combined_cache, index=False)
    print(f"Combined draft picks: {len(combined)} rows saved to {combined_cache}")
    return combined


def fetch_combine_data(seasons: list = DEFAULT_SEASONS, force: bool = False) -> pd.DataFrame:
    """
    Pull NFL combine measurements via nfl.import_combine_data(), one season at a time.
    Key columns: draft_ovr, pfr_id, forty, wt, ht, vertical, bench, cone, shuttle.
    join to seasonal data via pfr_id → rosters.pfr_id → player_id.
    Cached per season and combined at data/raw/combine_data.parquet.
    """
    combined_cache = _cache_path("combine_data")
    frames = []
    for i, season in enumerate(seasons):
        season_cache = _season_cache_path("combine_data", season)
        if not force and os.path.exists(season_cache):
            print(f"  [{i+1}/{len(seasons)}] {season} — loaded from checkpoint")
            frames.append(pd.read_parquet(season_cache))
            continue

        print(f"  [{i+1}/{len(seasons)}] {season} — fetching...", end=" ", flush=True)
        df = nfl.import_combine_data([season])
        df.to_parquet(season_cache, index=False)
        frames.append(df)
        print(f"{len(df)} combine records saved")

        if i < len(seasons) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(combined_cache, index=False)
    print(f"Combined combine data: {len(combined)} rows saved to {combined_cache}")
    return combined


def fetch_injuries(seasons: list = DEFAULT_SEASONS, force: bool = False) -> pd.DataFrame:
    """
    Pull weekly injury report data via nfl.import_injuries(), one season at a time.
    Aggregates to season-level features per player:
      - games_missed: weeks with report_status 'Out' or 'IR'
      - ir_flag: 1 if any IR designation that season
    Cached per season and combined at data/raw/injuries.parquet.
    """
    combined_cache = _cache_path("injuries")
    frames = []
    for i, season in enumerate(seasons):
        season_cache = _season_cache_path("injuries", season)
        if not force and os.path.exists(season_cache):
            print(f"  [{i+1}/{len(seasons)}] {season} — loaded from checkpoint")
            frames.append(pd.read_parquet(season_cache))
            continue

        print(f"  [{i+1}/{len(seasons)}] {season} — fetching...", end=" ", flush=True)
        df = nfl.import_injuries([season])
        df = df[df["game_type"] == "REG"].copy() if "game_type" in df.columns else df.copy()

        # report_status values: "Out", "Doubtful", "Questionable", "Probable"
        # IR is a roster transaction — not a report_status value. Use games_missed >= 4
        # as a proxy (IR stints require missing at least 4 games).
        df["is_out"] = df["report_status"].isin(["Out", "Doubtful"]).astype(int)

        agg = (
            df.groupby(["gsis_id", "season"])
            .agg(games_missed=("is_out", "sum"))
            .reset_index()
            .rename(columns={"gsis_id": "player_id"})
        )
        agg["ir_flag"] = (agg["games_missed"] >= 4).astype(int)
        agg.to_parquet(season_cache, index=False)
        frames.append(agg)
        print(f"{len(agg)} player-season injury records saved")

        if i < len(seasons) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(combined_cache, index=False)
    print(f"Combined injuries: {len(combined)} rows saved to {combined_cache}")
    return combined


def aggregate_snap_pct(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Kept for backwards compatibility. snap_pct is not available in weekly
    data — use fetch_snap_counts() instead for offensive snap percentage.
    """
    if "snap_pct" in weekly_df.columns:
        agg = (
            weekly_df[["player_id", "season", "snap_pct"]]
            .dropna(subset=["snap_pct"])
            .groupby(["player_id", "season"])["snap_pct"]
            .mean()
            .reset_index()
            .rename(columns={"snap_pct": "avg_snap_pct"})
        )
        return agg
    print("snap_pct not in weekly data — use fetch_snap_counts() for snap %.")
    return pd.DataFrame(columns=["player_id", "season", "avg_snap_pct"])
