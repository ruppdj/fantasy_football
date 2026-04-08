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
giving us the final college season for each prospect.
"""

import os
import time
import requests
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
CFBD_BASE = "https://api.collegefootballdata.com"
REQUEST_DELAY_SECONDS = 1.5  # CFBD asks for polite usage

SKILL_CATEGORIES = {"receiving", "rushing"}  # stat categories to pull


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
    Returns list of raw stat dicts, or [] on error.
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


def fetch_college_stats(draft_seasons: list, force: bool = False) -> pd.DataFrame:
    """
    Pull final-year college stats for each draft class.

    For a draft_season (e.g. 2021), pulls college stats from the prior year (2020)
    since players play their final college season before being drafted.

    Returns one row per player with columns:
        cfb_player_id, college_season, player_name, team,
        college_rec_yards, college_rec_tds, college_targets,
        college_rush_yards, college_rush_tds, college_rush_atts,
        college_dominator_rate (computed: player % of team yards+TDs)

    Cached per college season at data/raw/college_stats_{year}.parquet.
    Combined result at data/raw/college_stats.parquet.
    """
    # Fail fast if API key is missing — better than looping and failing on every season
    try:
        _get_cfbd_headers()
    except ValueError as e:
        print(f"Cannot fetch college stats: {e}")
        return pd.DataFrame()

    combined_cache = _cache_path("college_stats")
    frames = []

    for i, draft_season in enumerate(draft_seasons):
        college_year = draft_season - 1  # final college season before draft
        season_cache = _season_cache_path("college_stats", college_year)

        if not force and os.path.exists(season_cache):
            print(f"  [{i+1}/{len(draft_seasons)}] college {college_year} — loaded from checkpoint")
            frames.append(pd.read_parquet(season_cache))
            continue

        print(f"  [{i+1}/{len(draft_seasons)}] college {college_year} — fetching...", end=" ", flush=True)

        # Pull receiving and rushing stats separately then merge
        receiving = _fetch_cfbd_stats(college_year, "receiving")
        time.sleep(REQUEST_DELAY_SECONDS)
        rushing = _fetch_cfbd_stats(college_year, "rushing")

        if not receiving and not rushing:
            print("no data returned, skipping")
            if i < len(draft_seasons) - 1:
                time.sleep(REQUEST_DELAY_SECONDS)
            continue

        def records_to_df(records, prefix):
            rows = []
            for r in records:
                rows.append({
                    "cfb_player_id": r.get("playerId"),
                    "player_name": r.get("player"),
                    "team": r.get("team"),
                    "conference": r.get("conference"),
                    f"{prefix}_yards": r.get("yards", 0),
                    f"{prefix}_tds": r.get("tds", 0),
                    f"{prefix}_atts": r.get("att", r.get("rec", 0)),
                })
            return pd.DataFrame(rows)

        rec_df = records_to_df(receiving, "rec")
        rush_df = records_to_df(rushing, "rush")

        # Merge on player identity
        if not rec_df.empty and not rush_df.empty:
            df = rec_df.merge(
                rush_df[["cfb_player_id", "rush_yards", "rush_tds", "rush_atts"]],
                on="cfb_player_id", how="outer"
            )
        elif not rec_df.empty:
            df = rec_df.copy()
            df["rush_yards"] = 0
            df["rush_tds"] = 0
            df["rush_atts"] = 0
        else:
            df = rush_df.copy()
            df["rec_yards"] = 0
            df["rec_tds"] = 0
            df["rec_atts"] = 0

        df = df.fillna(0)
        df["college_season"] = college_year
        df["draft_season"] = draft_season

        # Rename to final feature names
        df = df.rename(columns={
            "rec_yards": "college_rec_yards",
            "rec_tds": "college_rec_tds",
            "rec_atts": "college_targets",
            "rush_yards": "college_rush_yards",
            "rush_tds": "college_rush_tds",
            "rush_atts": "college_rush_atts",
        })

        # Compute dominator rate: player yards+TDs as % of team totals
        df["player_yards_tds"] = df["college_rec_yards"] + df["college_rush_yards"] + \
                                  (df["college_rec_tds"] + df["college_rush_tds"]) * 20
        team_totals = df.groupby("team")["player_yards_tds"].sum().rename("team_yards_tds")
        df = df.merge(team_totals, on="team", how="left")
        df["college_dominator_rate"] = (
            df["player_yards_tds"] / df["team_yards_tds"].replace(0, float("nan"))
        ).round(4)
        df.drop(columns=["player_yards_tds", "team_yards_tds"], inplace=True)

        df.to_parquet(season_cache, index=False)
        frames.append(df)
        print(f"{len(df)} player records saved")

        if i < len(draft_seasons) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    if not frames:
        print("No college stats data fetched.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(combined_cache, index=False)
    print(f"Combined college stats: {len(combined)} rows saved to {combined_cache}")
    return combined
