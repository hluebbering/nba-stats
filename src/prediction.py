# src/prediction.py
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2
from src.utils import get_player_id, get_player_team_id, get_team_name
from src.data_ingestion import get_player_game_logs, get_player_advanced_stats_parallel, fetch_bulk_player_game_logs
from src.feature_engineering import feature_engineering_pipeline


def get_next_game_info(team_id, days_ahead=60):
    """
    Search for the next game up to days_ahead into the future.
    Returns (date, opponent_id, home_flag) or (None, None, None).
    """
    next_game_date = datetime.now()
    for _ in range(days_ahead):
        date_str = next_game_date.strftime('%Y-%m-%d')
        try:
            games = scoreboardv2.ScoreboardV2(game_date=date_str).game_header.get_data_frame()
            team_games = games[(games['HOME_TEAM_ID'] == team_id) | (games['VISITOR_TEAM_ID'] == team_id)]
            if not team_games.empty:
                game = team_games.iloc[0]
                opp_id = game['VISITOR_TEAM_ID'] if game['HOME_TEAM_ID'] == team_id else game['HOME_TEAM_ID']
                home_flag = int(game['HOME_TEAM_ID'] == team_id)
                return next_game_date, opp_id, home_flag
        except Exception:
            pass
        next_game_date += timedelta(days=1)
    return None, None, None


def get_team_defense_stats(team_id, season='2024-25', retries=3, wait=5):
    """
    Fetch defensive stats with retry logic.
    Returns a Series with DEF_RATING, OPP_PTS_OFF_TOV, OPP_PTS_2ND_CHANCE, TEAM_ID.
    """
    from nba_api.stats.endpoints import leaguedashteamstats

    for attempt in range(1, retries + 1):
        try:
            df = leaguedashteamstats.LeagueDashTeamStats(
                team_id_nullable=team_id,
                season=season,
                measure_type_detailed_defense='Defense',
                per_mode_detailed='PerGame',
                timeout=120
            ).get_data_frames()[0]
            if df.empty:
                raise ValueError('Empty defense stats')
            return df[['TEAM_ID', 'DEF_RATING', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE']].iloc[0]
        except Exception as e:
            print(f"Defense pull {attempt}/{retries} failed: {e}")
            if attempt < retries:
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
    # Return NaNs if all retries fail
    return pd.Series({'TEAM_ID': team_id, 'DEF_RATING': np.nan, 'OPP_PTS_OFF_TOV': np.nan, 'OPP_PTS_2ND_CHANCE': np.nan})


def predict_next_game(player_name, feature_cols, season='2024-25'):
    player_id = get_player_id(player_name)
    if not player_id:
        print(f"Invalid player: {player_name}")
        return None

    # 1) Logs and advanced stats
    bulk = fetch_bulk_player_game_logs(season)
    logs = get_player_game_logs(player_id, bulk)
    if logs.empty:
        print(f"No logs for {player_name}")
        return None

    game_ids = logs['GAME_ID'].tolist()
    adv = get_player_advanced_stats_parallel(player_id, game_ids)
    if adv.empty:
        print(f"No advanced stats for {player_name}")
        return None

    # Merge and engineer features
    merged = logs.merge(adv, on=['GAME_ID', 'PLAYER_ID'], how='left')
    merged['GAME_DATE'] = pd.to_datetime(merged['GAME_DATE'], errors='coerce')
    df_feat = feature_engineering_pipeline(merged)

    # 2) Next game info
    team_id = get_player_team_id(player_id)
    next_date, opp_id, home = get_next_game_info(team_id)
    if next_date is None:
        print(f"No next game for {player_name}")
        return None

    # 3) Opponent defense stats
    opp_def = get_team_defense_stats(opp_id, season)

    # 4) Prepare latest row for prediction
    latest = df_feat.iloc[-1].copy()
    latest['REST_DAYS'] = (next_date - latest['GAME_DATE']).days
    latest['HOME_GAME'] = home
    latest['GAME_DATE'] = next_date
    latest['OPPONENT_TEAM_ID'] = opp_id
    for col in ['DEF_RATING', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE']:
        latest[col] = opp_def.get(col, np.nan)

    # 5) Predict
    fv = latest[feature_cols].values.reshape(1, -1)
    scaler = joblib.load('lib/scaler.pkl')
    fv_df = pd.DataFrame(fv, columns=feature_cols)
    model = joblib.load('lib/player_points_model.pkl')
    pred = model.predict(scaler.transform(fv_df))[0]
    opp_name = get_team_name(opp_id)
    print(f"Predicted points for {player_name} on {next_date.date()} vs {opp_name}: {pred:.2f}")
    return pred
