"""
college_fetcher.py
Pulls final-year college production stats for NFL draft prospects via the
College Football Data API (CFBD).

Authentication: CFBD requires a free API key (get one at https://collegefootballdata.com/key).
Set it as an environment variable before running:

    export CFBD_API_KEY="your_key_here"

Or set it in the notebook before calling fetch_college_stats():

    import os
    os.environ["CFBD_API_KEY"] = "your_key_here"

Join path:
  draft_picks.cfb_player_id → CFBD player ID → college season stats

Stats are pulled for the season immediately before each draft class (draft year - 1),
giving us the final college season for each prospect. Additionally, the 3 prior
seasons are checked to compute college_years (how many seasons each player appeared
in, capped at 4).

API response format: LONG — one row per player per statType.
  Receiving statTypes: LONG, REC, TD, YDS, YPR
  Rushing statTypes:   CAR, LONG, TD, YDS, YPC
"""

import os
import time
import requests
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
CFBD_BASE = "https://api.collegefootballdata.com"
REQUEST_DELAY_SECONDS = 1.5  # CFBD asks for polite usage

# How many seasons to look back for college_years count (including final season)
LOOKBACK_YEARS = 4

# statType → output column name for each category
REC_STAT_MAP  = {"YDS": "rec_yards", "TD": "rec_tds", "REC": "rec_atts"}
RUSH_STAT_MAP = {"YDS": "rush_yards", "TD": "rush_tds", "CAR": "rush_atts"}


def _cache_path(name: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    return os.path.join(RAW_DIR, f"{name}.parquet")


def _season_cache_path(prefix: str, season: int) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    return os.path.join(RAW_DIR, f"{prefix}_{season}.parquet")


def _get_cfbd_headers() -> dict:
    """
    Return Authorization header for CFBD API.
    Raises ValueError with setup instructions if key is missing.
    """
    api_key = os.environ.get("CFBD_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "CFBD_API_KEY not set. Get a free key at https://collegefootballdata.com/key\n"
            "Then set it with: os.environ['CFBD_API_KEY'] = 'your_key_here'"
        )
    return {"Authorization": f"Bearer {api_key}"}


def _fetch_cfbd_stats(year: int, category: str) -> list:
    """
    Fetch one category of player stats for one college season from CFBD.
    Returns list of raw stat dicts (long format), or [] on error.
    """
    url = f"{CFBD_BASE}/stats/player/season"
    params = {"year": year, "seasonType": "regular", "category": category}
    try:
        headers = _get_cfbd_headers()
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except ValueError as e:
        print(f"    Error: {e}")
        return []
    except requests.RequestException as e:
        print(f"    Warning: CFBD request failed for {year}/{category}: {e}")
        return []


def _records_to_df(records: list, stat_map: dict) -> pd.DataFrame:
    """
    Convert CFBD long-format records to one row per player (wide format).

    The CFBD API returns one row per player per statType, e.g.:
        {playerId: "123", player: "Name", team: "Alabama", statType: "YDS", stat: "850"}
        {playerId: "123", player: "Name", team: "Alabama", statType: "TD",  stat: "7"}

    stat_map maps statType values to output column names,
    e.g. {"YDS": "rec_yards", "TD": "rec_tds", "REC": "rec_atts"}
    Only statTypes in stat_map are kept.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.rename(columns={"playerId": "cfb_player_id", "player": "player_name"})
    df["stat"] = pd.to_numeric(df["stat"], errors="coerce").fillna(0)

    # Keep only stat types we care about
    df = df[df["statType"].isin(stat_map)]
    if df.empty:
        return pd.DataFrame()

    # Pivot to wide format: one row per player
    id_cols = ["cfb_player_id", "player_name", "team", "conference"]
    id_cols = [c for c in id_cols if c in df.columns]

    pivot = df.pivot_table(
        index=id_cols,
        columns="statType",
        values="stat",
        aggfunc="sum",
    ).reset_index()
    pivot.columns.name = None

    # Rename statType columns to internal names, fill any missing stat types with 0
    pivot = pivot.rename(columns=stat_map)
    for col in stat_map.values():
        if col not in pivot.columns:
            pivot[col] = 0

    # Ensure string columns stay as strings
    for col in ["cfb_player_id", "player_name", "team", "conference"]:
        if col in pivot.columns:
            pivot[col] = pivot[col].astype(str)

    return pivot


def _fetch_year_raw(college_year: int) -> pd.DataFrame:
    """
    Fetch receiving + rushing stats for one college year and return a combined
    wide-format DataFrame. Uses per-year raw cache at college_stats_{year}.parquet.
    Returns empty DataFrame if no data available.
    """
    season_cache = _season_cache_path("college_stats", college_year)
    if os.path.exists(season_cache):
        return pd.read_parquet(season_cache)

    receiving = _fetch_cfbd_stats(college_year, "receiving")
    time.sleep(REQUEST_DELAY_SECONDS)
    rushing = _fetch_cfbd_stats(college_year, "rushing")

    if not receiving and not rushing:
        return pd.DataFrame()

    rec_df  = _records_to_df(receiving, REC_STAT_MAP)
    rush_df = _records_to_df(rushing,   RUSH_STAT_MAP)

    # Outer merge so players with only rushing or only receiving are included
    if not rec_df.empty and not rush_df.empty:
        df = rec_df.merge(
            rush_df[["cfb_player_id"] + list(RUSH_STAT_MAP.values())],
            on="cfb_player_id",
            how="outer",
        )
    elif not rec_df.empty:
        df = rec_df.copy()
        for col in RUSH_STAT_MAP.values():
            df[col] = 0
    elif not rush_df.empty:
        df = rush_df.copy()
        for col in REC_STAT_MAP.values():
            df[col] = 0
    else:
        return pd.DataFrame()

    # Fill missing numeric stats with 0, keep strings clean
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    for col in ["player_name", "team", "conference"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    df["college_season"] = college_year

    df.to_parquet(season_cache, index=False)
    return df


def fetch_college_stats(draft_seasons: list, force: bool = False) -> pd.DataFrame:
    """
    Pull final-year college stats for each draft class, plus up to 3 prior
    seasons to compute college_years (number of seasons each player appeared in).

    For a draft_season (e.g. 2021), the final college season is 2020.
    Prior seasons checked for college_years: 2019, 2018, 2017.

    Returns one row per player per draft class with columns:
        cfb_player_id, player_name, team, conference,
        college_season, draft_season,
        college_rec_yards, college_rec_tds, college_targets,
        college_rush_yards, college_rush_tds, college_rush_atts,
        college_dominator_rate,
        college_years  (1–4: how many seasons player appeared in CFBD data)

    Cached per college season at data/raw/college_stats_{year}.parquet.
    Combined result at data/raw/college_stats.parquet.
    """
    # Fail fast if API key is missing
    try:
        _get_cfbd_headers()
    except ValueError as e:
        print(f"Cannot fetch college stats: {e}")
        return pd.DataFrame()

    # If force=True, delete all existing per-year caches so they regenerate
    if force:
        all_years = set()
        for draft_season in draft_seasons:
            for offset in range(LOOKBACK_YEARS):
                all_years.add(draft_season - 1 - offset)
        for yr in all_years:
            cache = _season_cache_path("college_stats", yr)
            if os.path.exists(cache):
                os.remove(cache)

    combined_cache = _cache_path("college_stats")
    frames = []

    for i, draft_season in enumerate(draft_seasons):
        final_year = draft_season - 1  # last college season before draft

        print(f"  [{i+1}/{len(draft_seasons)}] draft {draft_season} (college {final_year}) — ", end="", flush=True)

        # Fetch final season (main stats)
        final_df = _fetch_year_raw(final_year)
        if final_df.empty:
            print("no data, skipping")
            if i < len(draft_seasons) - 1:
                time.sleep(REQUEST_DELAY_SECONDS)
            continue

        # Fetch prior seasons to build college_years count
        # Collect all cfb_player_ids seen across lookback window
        prior_year_sets: dict[str, set] = {}  # player_id → set of years seen
        for player_id in final_df["cfb_player_id"].unique():
            prior_year_sets[str(player_id)] = {final_year}

        for offset in range(1, LOOKBACK_YEARS):
            prior_year = final_year - offset
            prior_df = _fetch_year_raw(prior_year)
            if prior_df.empty:
                time.sleep(REQUEST_DELAY_SECONDS)
                continue
            for player_id in prior_df["cfb_player_id"].unique():
                pid = str(player_id)
                if pid in prior_year_sets:
                    prior_year_sets[pid].add(prior_year)
            time.sleep(REQUEST_DELAY_SECONDS)

        # Add college_years to final season data
        final_df["cfb_player_id"] = final_df["cfb_player_id"].astype(str)
        final_df["college_years"] = final_df["cfb_player_id"].map(
            lambda pid: len(prior_year_sets.get(pid, {pid}))
        )

        final_df["draft_season"] = draft_season

        # Rename stat columns to final feature names
        final_df = final_df.rename(columns={
            "rec_yards": "college_rec_yards",
            "rec_tds":   "college_rec_tds",
            "rec_atts":  "college_targets",
            "rush_yards": "college_rush_yards",
            "rush_tds":   "college_rush_tds",
            "rush_atts":  "college_rush_atts",
        })

        # Compute dominator rate: player yards+TDs as % of team totals
        final_df["player_yards_tds"] = (
            final_df["college_rec_yards"] + final_df["college_rush_yards"] +
            (final_df["college_rec_tds"] + final_df["college_rush_tds"]) * 20
        )
        team_totals = (
            final_df.groupby("team")["player_yards_tds"]
            .sum()
            .rename("team_yards_tds")
        )
        final_df = final_df.merge(team_totals, on="team", how="left")
        final_df["college_dominator_rate"] = (
            final_df["player_yards_tds"] /
            final_df["team_yards_tds"].replace(0, float("nan"))
        ).round(4)
        final_df.drop(columns=["player_yards_tds", "team_yards_tds"], inplace=True)

        frames.append(final_df)
        n_players = len(final_df)
        avg_years = final_df["college_years"].mean()
        print(f"{n_players} players, avg {avg_years:.1f} college years")

        if i < len(draft_seasons) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    if not frames:
        print("No college stats data fetched.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(combined_cache, index=False)
    print(f"Combined college stats: {len(combined)} rows saved to {combined_cache}")
    return combined
