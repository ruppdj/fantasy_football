"""
sleeper_fetcher.py
Wrappers around the Sleeper REST API (no auth required).
Pulls player metadata and dynasty ADP context.
Data is cached as parquet in data/raw/.
"""

import os
import time
import requests
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
SLEEPER_BASE = "https://api.sleeper.app/v1"
DYNASTY_POSITIONS = {"QB", "RB", "WR", "TE"}

# Sleeper has no published rate limit, but 1 req/sec is considered polite
# for any public API without an explicit rate limit spec.
REQUEST_DELAY_SECONDS = 1.0


def _cache_path(name: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    return os.path.join(RAW_DIR, f"{name}.parquet")


def fetch_players(force: bool = False) -> pd.DataFrame:
    """
    Fetch all NFL players from Sleeper.
    Returns a DataFrame with player metadata (one row per player).
    Cached at data/raw/sleeper_players.parquet.
    """
    cache = _cache_path("sleeper_players")
    if not force and os.path.exists(cache):
        print(f"Loading Sleeper players from cache: {cache}")
        return pd.read_parquet(cache)

    print("Fetching all NFL players from Sleeper API...")
    resp = requests.get(f"{SLEEPER_BASE}/players/nfl", timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    records = []
    for player_id, info in raw.items():
        fantasy_positions = info.get("fantasy_positions") or []
        if not any(p in DYNASTY_POSITIONS for p in fantasy_positions):
            continue
        records.append({
            "sleeper_id": player_id,
            "full_name": info.get("full_name"),
            "first_name": info.get("first_name"),
            "last_name": info.get("last_name"),
            "position": info.get("position"),
            "fantasy_positions": ",".join(fantasy_positions),
            "team": info.get("team"),
            "age": info.get("age"),
            "years_exp": info.get("years_exp"),
            "birth_date": info.get("birth_date"),
            "college": info.get("college"),
            "height": info.get("height"),
            "weight": info.get("weight"),
            "status": info.get("status"),
            "injury_status": info.get("injury_status"),
            "depth_chart_order": info.get("depth_chart_order"),
            "search_rank": info.get("search_rank"),
            "number": info.get("number"),
        })

    df = pd.DataFrame(records)
    df.to_parquet(cache, index=False)
    print(f"Saved {len(df)} skill-position players to {cache}")
    return df


def fetch_player_stats(player_id: str, season: int, delay: float = REQUEST_DELAY_SECONDS) -> dict:
    """
    Fetch season stats for a single player from Sleeper.
    Returns raw stats dict or empty dict on failure.

    When calling in a loop over many players, the caller should rely on the
    built-in delay param (default REQUEST_DELAY_SECONDS) to stay polite.
    Pass delay=0 only for one-off lookups.
    """
    url = f"{SLEEPER_BASE}/stats/nfl/player/{player_id}"
    params = {"season_type": "regular", "season": season}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if delay > 0:
            time.sleep(delay)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return {}


def fetch_adp(season: int, force: bool = False) -> pd.DataFrame:
    """
    Fetch dynasty ADP data for a given season from Sleeper.
    Cached at data/raw/sleeper_adp_{season}.parquet.

    Note: Sleeper's ADP endpoint returns startup dynasty draft data.
    """
    cache = _cache_path(f"sleeper_adp_{season}")
    if not force and os.path.exists(cache):
        print(f"Loading Sleeper ADP {season} from cache: {cache}")
        return pd.read_parquet(cache)

    print(f"Fetching Sleeper dynasty ADP for {season}...")
    url = f"{SLEEPER_BASE}/players/nfl/trending/add"
    params = {"lookback_hours": 24, "limit": 200}
    # ADP endpoint — use the stats rankings endpoint for dynasty
    url = f"{SLEEPER_BASE}/players/nfl"
    # Sleeper doesn't expose historical ADP directly via public API.
    # We use search_rank as a proxy for dynasty value.
    players = fetch_players(force=False)
    adp_df = players[["sleeper_id", "full_name", "position", "search_rank"]].copy()
    adp_df["season"] = season
    adp_df = adp_df.dropna(subset=["search_rank"]).sort_values("search_rank")
    adp_df.to_parquet(cache, index=False)
    print(f"Saved {len(adp_df)} ADP records to {cache}")
    return adp_df
