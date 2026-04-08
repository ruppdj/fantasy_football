"""
test_fetchers.py
Quick smoke tests for nfl_fetcher and sleeper_fetcher.
Pulls a single season of data and validates expected shape and columns.

Run from the project root:
    python test_fetchers.py
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(__file__))

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def test_nfl_seasonal():
    print("\n=== nfl_fetcher: seasonal stats (2023 only) ===")
    from src.fetchers.nfl_fetcher import fetch_seasonal_stats

    df = fetch_seasonal_stats(seasons=[2023], force=True)

    check("Returns a DataFrame",          df is not None and len(df) > 0,
          f"{len(df)} rows")
    check("Has fantasy_points_ppr col",   "fantasy_points_ppr" in df.columns)
    check("Has player_id col",            "player_id" in df.columns)
    check("Has season col",               "season" in df.columns)
    check("Season == 2023",               (df["season"] == 2023).all())
    check("Only REG season type",
          "season_type" not in df.columns or (df["season_type"] == "REG").all())
    check("Reasonable row count",         50 < len(df) < 5000,
          f"{len(df)} rows")

    print(f"  Columns ({len(df.columns)}): {df.columns.tolist()[:8]} ...")
    return True


def test_nfl_rosters():
    print("\n=== nfl_fetcher: rosters / import_ids ===")
    from src.fetchers.nfl_fetcher import fetch_rosters

    df = fetch_rosters(force=True)

    check("Returns a DataFrame",          df is not None and len(df) > 0,
          f"{len(df)} rows")
    check("Has player_id col",            "player_id" in df.columns)
    check("Has position col",             "position" in df.columns)
    check("Only skill positions",
          df["position"].isin(["QB", "RB", "WR", "TE"]).all())
    check("Has sleeper_id col",           "sleeper_id" in df.columns,
          "needed for Sleeper join")
    check("Has birth_date col",           "birth_date" in df.columns,
          "needed for age computation")
    check("Has pfr_id col",               "pfr_id" in df.columns,
          "needed for snap count join")
    return True


def test_nfl_weekly():
    print("\n=== nfl_fetcher: weekly stats (2023 only) ===")
    from src.fetchers.nfl_fetcher import fetch_weekly_stats

    weekly = fetch_weekly_stats(seasons=[2023], force=True)
    check("Weekly data returned",         len(weekly) > 0, f"{len(weekly)} rows")
    check("Has player_id + season + week",
          all(c in weekly.columns for c in ["player_id", "season", "week"]))
    check("Has fantasy_points_ppr",       "fantasy_points_ppr" in weekly.columns)
    return True


def test_nfl_snap_counts():
    print("\n=== nfl_fetcher: snap counts (2023 only) ===")
    from src.fetchers.nfl_fetcher import fetch_snap_counts

    snap = fetch_snap_counts(seasons=[2023], force=True)
    check("Snap data returned",           len(snap) > 0, f"{len(snap)} rows")
    check("Has player_id col",            "player_id" in snap.columns)
    check("Has season col",               "season" in snap.columns)
    check("Has avg_snap_pct col",         "avg_snap_pct" in snap.columns)
    check("avg_snap_pct in [0, 1]",
          snap["avg_snap_pct"].between(0, 1).all())
    return True


def test_sleeper_players():
    print("\n=== sleeper_fetcher: player metadata ===")
    from src.fetchers.sleeper_fetcher import fetch_players

    df = fetch_players(force=True)

    check("Returns a DataFrame",          df is not None and len(df) > 0,
          f"{len(df)} players")
    check("Has sleeper_id col",           "sleeper_id" in df.columns)
    check("Has full_name col",            "full_name" in df.columns)
    check("Has position col",             "position" in df.columns)
    # fantasy_positions column contains comma-separated values like "WR" or "WR,RB"
    # position (primary) can differ (e.g. a WR listed as "WR" in depth chart)
    # so we validate on fantasy_positions, not position
    skill = {"QB", "RB", "WR", "TE"}
    has_skill = df["fantasy_positions"].dropna().apply(
        lambda x: bool(skill & set(x.split(",")))
    ).all()
    check("fantasy_positions contains skill pos", has_skill)
    check("Reasonable count (>500)",      len(df) > 500,
          f"{len(df)} skill-position players")

    # Spot-check a known player name appears
    names = df["full_name"].dropna().str.lower().tolist()
    found = any("hill" in n for n in names)  # Tyreek Hill
    check("Known player present (Hill)",  found)
    return True


def test_draft_picks():
    print("\n=== nfl_fetcher: draft picks (2023 only) ===")
    from src.fetchers.nfl_fetcher import fetch_draft_picks

    df = fetch_draft_picks(seasons=[2023], force=True)

    check("Returns a DataFrame",          df is not None and len(df) > 0,
          f"{len(df)} rows")
    check("Has player_id col",            "player_id" in df.columns,
          "renamed from gsis_id")
    check("Has season col",               "season" in df.columns)
    check("Has draft_pick col",           "draft_pick" in df.columns,
          "overall pick number")
    check("Has draft_round col",          "draft_round" in df.columns)
    check("Has age_at_draft col",         "age_at_draft" in df.columns)
    check("Season == 2023",               (df["season"] == 2023).all(),
          "single season fetch")
    check("Pick numbers are positive",    (df["draft_pick"] > 0).all())
    check("Round in 1-7",                 df["draft_round"].between(1, 7).all())
    check("Reasonable row count (>200)",  len(df) > 200,
          f"{len(df)} picks in 2023 draft")

    # cfb_player_id is the join key to college stats
    if "cfb_player_id" in df.columns:
        pct = df["cfb_player_id"].notna().mean()
        check("cfb_player_id present",   pct > 0.3,
              f"{pct:.0%} non-null (join key for college stats)")

    print(f"  Sample columns: {df.columns.tolist()[:8]} ...")
    return True


def test_combine_data():
    print("\n=== nfl_fetcher: combine data (2023 only) ===")
    from src.fetchers.nfl_fetcher import fetch_combine_data

    df = fetch_combine_data(seasons=[2023], force=True)

    check("Returns a DataFrame",          df is not None and len(df) > 0,
          f"{len(df)} rows")
    check("Has season col",               "season" in df.columns)
    check("Has combine_forty col",        "combine_forty" in df.columns,
          "40-yard dash time")
    check("Has combine_weight col",       "combine_weight" in df.columns)
    check("Has player_name col",          "player_name" in df.columns or
          any(c in df.columns for c in ["full_name", "name"]))

    if "combine_forty" in df.columns:
        valid_forty = df["combine_forty"].dropna()
        if len(valid_forty) > 0:
            check("Forty times in plausible range (4.0–5.5s)",
                  valid_forty.between(4.0, 5.5).all(),
                  f"{len(valid_forty)} non-null measurements")

    # Combine attendance is sparse — many players skip
    if "combine_forty" in df.columns:
        pct = df["combine_forty"].notna().mean()
        check("Some forty times present", pct > 0.0,
              f"{pct:.0%} non-null (many prospects skip combine)")

    print(f"  Sample columns: {df.columns.tolist()[:8]} ...")
    return True


def test_injuries():
    print("\n=== nfl_fetcher: injuries aggregated (2023 only) ===")
    from src.fetchers.nfl_fetcher import fetch_injuries

    df = fetch_injuries(seasons=[2023], force=True)

    check("Returns a DataFrame",          df is not None and len(df) > 0,
          f"{len(df)} rows")
    check("Has player_id col",            "player_id" in df.columns)
    check("Has season col",               "season" in df.columns)
    check("Has games_missed col",         "games_missed" in df.columns,
          "aggregated from weekly reports")
    check("Has ir_flag col",              "ir_flag" in df.columns,
          "1 if player hit IR any week")
    check("games_missed >= 0",            (df["games_missed"] >= 0).all())
    check("ir_flag is 0 or 1",            df["ir_flag"].isin([0, 1]).all())
    check("Season == 2023",               (df["season"] == 2023).all())

    n_ir = df["ir_flag"].sum()
    avg_missed = df["games_missed"].mean()
    check("Some IR flags present",        n_ir > 0,
          f"{int(n_ir)} players on IR in 2023")
    print(f"  Avg games missed per player-season: {avg_missed:.2f}")
    return True


def test_college_stats():
    print("\n=== college_fetcher: college stats (2022 draft class) ===")
    from src.fetchers.college_fetcher import fetch_college_stats

    # Use a single draft class to keep the test fast
    df = fetch_college_stats(draft_seasons=[2022], force=True)

    if df is None or len(df) == 0:
        check("Returns non-empty DataFrame", False,
              "Got empty result — CFBD API may be unreachable or rate-limited")
        return False

    check("Returns a DataFrame",          len(df) > 0, f"{len(df)} rows")
    check("Has draft_season col",         "draft_season" in df.columns)
    check("Has cfb_player_id col",        "cfb_player_id" in df.columns,
          "join key to draft picks")
    check("Has college_rec_yards col",    "college_rec_yards" in df.columns)
    check("Has college_dominator_rate",   "college_dominator_rate" in df.columns)
    check("draft_season == 2022",         (df["draft_season"] == 2022).all())
    check("college_rec_yards >= 0",       (df["college_rec_yards"] >= 0).all())

    if "college_dominator_rate" in df.columns:
        valid_dom = df["college_dominator_rate"].dropna()
        if len(valid_dom) > 0:
            check("Dominator rate in [0, 1]",
                  valid_dom.between(0, 1).all(),
                  f"{len(valid_dom)} non-null rates")

    print(f"  Sample columns: {df.columns.tolist()[:8]} ...")
    return True


def run_all():
    tests = [
        ("NFL Seasonal Stats",   test_nfl_seasonal),
        ("NFL Rosters",          test_nfl_rosters),
        ("NFL Weekly",           test_nfl_weekly),
        ("NFL Snap Counts",      test_nfl_snap_counts),
        ("Sleeper Players",      test_sleeper_players),
        ("NFL Draft Picks",      test_draft_picks),
        ("NFL Combine Data",     test_combine_data),
        ("NFL Injuries",         test_injuries),
        ("College Stats (CFBD)", test_college_stats),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name} raised an exception:")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
