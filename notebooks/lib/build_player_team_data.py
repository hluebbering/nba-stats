# fetch_nba_data.py
import os
import time
from time import sleep
import warnings
import pandas as pd
from datetime import datetime, timedelta

from nba_api.stats.endpoints import leaguedashplayerstats, leaguegamelog, boxscoretraditionalv2, commonplayerinfo
from nba_api.stats.static import teams, players

warnings.filterwarnings("ignore")

API_CALL_DELAY = 0.6
DATA_FILE = "data/merged_player_logs.csv"  # CSV name for the merged dataset


def fetch_player_data(season='2024-25', season_type='Regular Season'):
    """
    Fetch player-level statistics and game logs for a given season and season type.
    Returns:
        df_player_stats (pd.DataFrame): Player per-game stats for the season.
        df_player_game_logs (pd.DataFrame): Player game logs for the season.
    """
    print("Fetching player statistics and player game logs...")
    player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, season_type_all_star=season_type,
                                                               per_mode_detailed='PerGame', timeout=60)
    df_player_stats = player_stats.get_data_frames()[0]

    player_game_logs = leaguegamelog.LeagueGameLog(season=season, season_type_all_star=season_type,
                                                   player_or_team_abbreviation='P', timeout=60)
    df_player_game_logs = player_game_logs.get_data_frames()[0]

    return df_player_stats, df_player_game_logs


def fetch_team_stats(season='2024-25', season_type='Regular Season'):
    """
    Fetch team-level statistics by iterating over each game in the season and extracting
    boxscoretraditionalv2 data for teams.
    Returns:
        df_team_stats (pd.DataFrame): All team boxscore data merged into one DataFrame.
    """
    print("Fetching team statistics (boxscoretraditionalv2)...")
    team_game_logs = leaguegamelog.LeagueGameLog(season=season, season_type_all_star=season_type,
                                                 player_or_team_abbreviation='T',timeout=60)
    df_team_game_logs = team_game_logs.get_data_frames()[0]
    game_ids = df_team_game_logs['GAME_ID'].unique()

    df_team_stats = pd.DataFrame()
    for game_id in game_ids:
        sleep(API_CALL_DELAY)
        try:
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id, timeout=60)
            # index=1 => team-level stats
            team_stats = boxscore.get_data_frames()[1]
            df_team_stats = pd.concat([df_team_stats, team_stats], ignore_index=True)
        except Exception as e:
            print(f"Error fetching team stats for game {game_id}: {e}")
            continue

    return df_team_stats




def merge_player_team_stats(df_player_game_logs, df_team_stats):
    """
    Merge player game logs with team stats to get opponent-related stats.
    Merges logs with the player's own team performance and also merges 
    the opponent's team stats for comparison.
    Returns:
        df_player_full (pd.DataFrame): The enriched DataFrame, including opponent info.
    """
    print("Merging player logs with team stats...")

    # Merge logs with team stats (the player's own team)
    df_player_team = pd.merge(df_player_game_logs, df_team_stats,
                              on=['GAME_ID', 'TEAM_ID'], suffixes=('', '_TEAM'))

    # Prepare opponent stats by renaming TEAM_ID -> OPPONENT_TEAM_ID
    df_team_stats_opponent = df_team_stats.copy()
    df_team_stats_opponent.rename(columns={'TEAM_ID': 'OPPONENT_TEAM_ID'}, inplace=True)

    # Merge to get opponent stats
    df_player_full = pd.merge(df_player_team, df_team_stats_opponent,
                              on='GAME_ID', suffixes=('', '_OPP'))

    # Remove rows where TEAM_ID == OPPONENT_TEAM_ID (avoid merging the same team onto itself)
    df_player_full = df_player_full[df_player_full['TEAM_ID'] != df_player_full['OPPONENT_TEAM_ID']]

    return df_player_full


def main(season='2024-25', season_type='Regular Season', data_file=DATA_FILE):
    """
    Main function that checks if CSV file exists. If so, loads from CSV.
    Otherwise, fetches data from the NBA API, merges, and saves CSV.
    Returns df_player_full DataFrame.
    """
    if os.path.exists(DATA_FILE):
        print(f"'{DATA_FILE}' already exists. Loading from CSV.")
        df_player_full = pd.read_csv(DATA_FILE)
    else:
        print(f"'{DATA_FILE}' not found. Fetching fresh data...")

        df_player_stats, df_player_game_logs = fetch_player_data(season, season_type)
        df_team_stats = fetch_team_stats(season, season_type)

        df_player_full = merge_player_team_stats(df_player_game_logs, df_team_stats)
        print("Merged data shape:", df_player_full.shape)

        # Optionally save
        df_player_full.to_csv(DATA_FILE, index=False)
        print(f"Saved merged dataset to '{DATA_FILE}'.")

    # Return the final merged DataFrame
    return df_player_full


if __name__ == "__main__":
    # If you run this file directly, it will check for 'merged_player_logs.csv'
    # in the current folder and either load it or fetch fresh data.
    df_player_full = main()
    print("Data shape:", df_player_full.shape)

    # Next steps (not implemented here):
    # 1. Add a POSITION column to df_player_stats by fetching each player's position.
    # 2. Merge position info into df_player_full.
    # 3. Group by OPPONENT_TEAM_ID and POSITION to compute average PTS allowed to that position.
    # 4. Add that metric as a feature in your final dataset.
