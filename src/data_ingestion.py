# src/data_ingestion.py
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from nba_api.stats.endpoints import leaguegamelog, boxscoreadvancedv2
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.endpoints import leaguedashteamstats

# ===============================
# Player Game Log (Bulk Fetch)
# ===============================
_bulk_logs_cache = None


# src/data_ingestion.py
import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

_bulk_logs_cache = None

def fetch_bulk_player_game_logs(season='2024-25', retries=3, wait=5):
    global _bulk_logs_cache
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f'{cache_dir}/bulk_logs_{season}.parquet'

    # 1) In-memory cache
    if _bulk_logs_cache is not None:
        return _bulk_logs_cache

    # 2) On-disk cache
    if os.path.exists(cache_file):
        print(f"🔄 Loading bulk logs from cache: {cache_file}")
        _bulk_logs_cache = pd.read_parquet(cache_file)
        return _bulk_logs_cache

    # 3) Fetch from API with retries
    for attempt in range(1, retries + 1):
        try:
            print(f"🌐 Fetching bulk logs (attempt {attempt}/{retries})…")
            lg = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation='P',
                timeout=120
            )
            df = lg.get_data_frames()[0]
            df.columns = df.columns.str.upper()

            # Save to disk for next time
            df.to_parquet(cache_file, index=False)
            print(f"✅ Saved bulk logs to cache: {cache_file}")

            _bulk_logs_cache = df
            return df

        except Exception as e:
            print(f"  ❌ Attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"     retrying in {wait}s…")
                time.sleep(wait)

    # 4) If all else fails, throw error
    raise RuntimeError(
        f"NBA API failed after {retries} retries and no cache file at {cache_file}"
    )


# def fetch_bulk_player_game_logs(season='2024-25'):
#     global _bulk_logs_cache
#     if _bulk_logs_cache is None:
#         print("Fetching bulk player game logs...")
#         lg = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='P', timeout=60)
#         df = lg.get_data_frames()[0]
#         df.columns = df.columns.str.upper()
#         _bulk_logs_cache = df
#     return _bulk_logs_cache






# def fetch_bulk_player_game_logs(season='2024-25', retries=3, wait=5):
#     for attempt in range(retries):
#         try:
#             lg = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='P', timeout=120)
#             df = lg.get_data_frames()[0]
#             df.columns = df.columns.str.upper()
#             return df
#         except Exception as e:
#             print(f"Attempt {attempt + 1}/{retries} failed: {e}. Retrying in {wait} seconds.")
#             time.sleep(wait)
#     raise Exception("NBA API failed after retries.")




def get_player_id(player_name):
    all_players = players.get_players()
    player = next((p for p in all_players if p['full_name'].lower() == player_name.lower()), None)
    return player['id'] if player else None

def get_player_game_logs(player_id, bulk_logs_df):
    return bulk_logs_df[bulk_logs_df['PLAYER_ID'] == player_id].copy()

# ===============================
# Advanced Stats (Parallel Fetch)
# ===============================
def fetch_boxscore_advanced(game_id, player_id):
    try:
        boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id, timeout=60)
        stats = boxscore.player_stats.get_data_frame()
        return stats[stats['PLAYER_ID'] == int(player_id)]
    except:
        return pd.DataFrame()

def get_player_advanced_stats_parallel(player_id, game_ids):
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_boxscore_advanced, game_id, player_id) for game_id in game_ids]
        for future in as_completed(futures):
            result = future.result()
            if not result.empty:
                results.append(result)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()




def get_opponent_stats(season='2024-25', timeout=120):
    """
    Fetch per-team defensive stats (DEF_RATING, OPP_PTS_OFF_TOV, OPP_PTS_2ND_CHANCE).
    """
    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense='Defense',
        per_mode_detailed='PerGame',
        timeout=timeout
    ).get_data_frames()[0]
    return df[['TEAM_ID', 'DEF_RATING', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE']]


def get_opponent_stats_last10(season='2024-25', timeout=120):
    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense='Defense',
        per_mode_detailed='PerGame',
        last_n_games=10,
        timeout=timeout
    ).get_data_frames()[0]

    # keep + rename
    df = df[['TEAM_ID', 'DEF_RATING', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE']]
    df.rename(columns={
        'DEF_RATING': 'DEF_RATING_LAST10',
        'OPP_PTS_OFF_TOV': 'OPP_PTS_OFF_TOV_LAST10',
        'OPP_PTS_2ND_CHANCE': 'OPP_PTS_2ND_CHANCE_LAST10'
    }, inplace=True)
    return df
