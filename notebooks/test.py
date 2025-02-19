# %% [markdown]
# 
# Use Python and the NBA API to develop advanced machine learning model that predicts player performance metrics in upcoming game
# 
# 
# <h3 style="color:black;font-family:'Segoe UI Variable Display';font-size:20px;text-shadow:0.125px 0.25px 0.25px black;margin:0;font-weight:300;line-height:1;">Part 2.0</h3>
# <h3 style="color:black;font-family:'Notes from Paris';font-size:20px;text-shadow:0.125px 0.25px 0.25px black;margin:0;line-height:1;">Part 2.0</h3>
# <h3 style="color:black;font-family:'Juicy Advice Outline';font-size:20px;text-shadow:0.125px 0.25px 0.25px black;margin:0;">Part 2.0</h3>
# <h3 style="color:black;font-family:'Mencken Std';font-size:20px;text-shadow:0.125px 0.25px 0.25px black;line-height:1;margin:0;">Part 2.0</h3>
# <h3 style="color:black;font-family:'Digital-7';font-size:20px;text-shadow:0.125px 0.25px 0.25px black;line-height:1;margin:0;">Part 2.0</h3>
# <h3 style="color:black;font-family:'Proxima Nova';font-size:20px;text-shadow:0.125px 0.25px 0.25px black;line-height:1;margin:0;">Part 2.0</h3>
# <h3 style="color:black;font-family:'Barlow Condensed';font-size:20px;text-shadow:0.125px 0.25px 0.25px black;line-height:1;margin:0;">Part 2.0</h3>
# 
# 
# <h3 style="color:black;font-family:'Lazy Crunch';font-size:20px;text-shadow:0.125px 0.25px 0.25px black;line-height:1;margin:0;">Part 2.0</h3>
# <h3 style="color:black;font-family:'Abril Display';font-size:20px;text-shadow:0.125px 0.25px 0.25px black;margin:0;">Part 2.0</h3>
# 
# 

# %%
import os
import time
import joblib
import warnings
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NBA API
from nba_api.stats.endpoints import (
    playergamelog, boxscoreadvancedv2,
    leaguedashteamstats, scoreboardv2, commonplayerinfo,
    leaguegamefinder, boxscoretraditionalv2
)
from nba_api.stats.static import players, teams

# Scikit-Learn & Modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import Ridge, BayesianRidge
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# %% [markdown]
# Developing a machine learning model to predict NBA player performance metrics like points involves several steps:
# 
# Data Collection: Gather historical and current season data using the NBA API, including advanced statistics such as Player Impact Estimate (PIE), Efficiency (EFF), Player Efficiency Rating (PER), trends, opponent data, and more.
# 
# Data Preprocessing: Clean and preprocess the data to prepare it for modeling.
# 
# Feature Engineering: Create features that capture the important aspects influencing player performance.
# 
# Model Training: Choose and train a suitable machine learning model.
# 
# Model Evaluation: Assess the model's performance and fine-tune as necessary.
# 
# Prediction: Use the trained model to predict future player performance.

# %% [markdown]
# -----
# 
# 
# 
# 
# 1. Utility functions

# %%

# =============================================================================
# 1. Utility & Helper Functions
# =============================================================================

def get_player_id(player_name):
    """Get the NBA player ID given the player's full name."""
    nba_players = players.get_players()
    player = next((p for p in nba_players if p['full_name'].lower() == player_name.lower()), None)
    return player['id'] if player else None

def get_team_abbreviation_id_mapping():
    """Return a dict mapping team abbreviations to team IDs."""
    nba_teams = teams.get_teams()
    return {team['abbreviation']: team['id'] for team in nba_teams}

def get_player_team_id(player_id):
    """Get the player's current team ID."""
    try:
        df = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        return int(df['TEAM_ID'].iloc[0])
    except:
        return None

def get_team_name(team_id):
    """Get the full team name given the team ID."""
    nba_teams = teams.get_teams()
    team = next((t for t in nba_teams if t['id'] == team_id), None)
    return team['full_name'] if team else 'Unknown Team'


# %% [markdown]
# -----------------------------------
# 
# 2. Data Fetching Functions

# %%

# =============================================================================
# 2. Data Fetching Functions
# =============================================================================

def get_player_game_logs(player_id, season='2024-25'):
    """Fetch player game logs for the given season."""
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=60)
        df = gamelog.get_data_frames()[0]
        df.columns = df.columns.str.upper()
        return df
    except:
        return pd.DataFrame()

def get_player_advanced_stats(player_id, season='2024-25'):
    """Fetch advanced stats for all games played by the player in the specified season."""
    gamelog_df = get_player_game_logs(player_id, season)
    adv_stats = []
    for game_id in gamelog_df['GAME_ID']:
        try:
            boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id, timeout=60)
            p_stats = boxscore.player_stats.get_data_frame()
            p_adv = p_stats[p_stats['PLAYER_ID'] == int(player_id)]
            adv_stats.append(p_adv)
        except:
            pass
        time.sleep(0.5)

    if adv_stats:
        df = pd.concat(adv_stats, ignore_index=True)
        return df[['GAME_ID', 'PLAYER_ID', 'USG_PCT', 'PIE', 'TEAM_ID', 'OFF_RATING', 'PACE_PER40']]
    return pd.DataFrame()

def get_opponent_stats(season='2024-25'):
    """Fetch opponent defensive stats for all teams."""
    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season, measure_type_detailed_defense='Defense',
        per_mode_detailed='PerGame', timeout=60
    ).get_data_frames()[0]
    return df[['TEAM_ID', 'DEF_RATING', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE']]


# %% [markdown]
# --------------------
# 
# 3. Feature Engineering Functions

# %%
# =============================================================================
# 3. Feature Engineering
# =============================================================================

def compute_efficiency(df):
    """Compute efficiency metric = (PTS + REB + AST + STL + BLK) - (FGA - FGM) - (FTA - FTM) - TOV."""
    df['EFF'] = (df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK']
                 - (df['FGA'] - df['FGM']) - (df['FTA'] - df['FTM']) - df['TOV'])
    return df

def compute_true_shooting_percentage(df):
    df['TS_DENOM'] = 2 * (df['FGA'] + 0.44 * df['FTA'])
    df['TS_PCT'] = df.apply(
        lambda row: row['PTS'] / row['TS_DENOM'] if row['TS_DENOM'] != 0 else 0, axis=1
    )
    df.drop(columns=['TS_DENOM'], inplace=True)
    return df

def get_player_position(player_id, cache=None):
    if cache is None:
        cache = {}
    if player_id in cache:
        return cache[player_id]
    pos = None
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        raw_pos = info.get('POSITION', [''])[0]
        if isinstance(raw_pos, str):
            main_pos = raw_pos.split('-')[0].title()
            if 'Guard' in main_pos:
                pos = 'G'
            elif 'Forward' in main_pos:
                pos = 'F'
            elif 'Center' in main_pos:
                pos = 'C'
    except:
        pass
    cache[player_id] = pos
    return pos

# %%
def feature_engineering(player_df, adv_df, opp_df, team_map):
    # 1) Basic computations
    player_df = compute_efficiency(player_df)
    player_df = compute_true_shooting_percentage(player_df)

    # 2) Merge advanced stats
    df = pd.merge(player_df, adv_df, on=['GAME_ID', 'PLAYER_ID'], how='left')
    df['OPPONENT_ABBREVIATION'] = df['MATCHUP'].str.split(' ').str[-1]
    df['OPPONENT_TEAM_ID'] = df['OPPONENT_ABBREVIATION'].map(team_map)

    # 3) Merge defensive stats
    if opp_df is not None and not opp_df.empty:
        df = pd.merge(df, opp_df, left_on='OPPONENT_TEAM_ID', right_on='TEAM_ID', how='left')
        for col in ['DEF_RATING', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE']:
            df[col] = df[col].fillna(df[col].mean())

    # 4) Sort by date and parse
    df.sort_values(['PLAYER_NAME', 'GAME_DATE'], inplace=True)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')

    # 5) Convert minutes to float
    def parse_minutes(x):
        if isinstance(x, str) and ':' in x:
            mins, secs = x.split(':')
            return float(mins) + float(secs)/60
        return float(x) if pd.notna(x) else 0

    df['MIN'] = df['MIN'].apply(parse_minutes)
    df['FG_PCT'] = df['FGM'] / df['FGA'].replace(0, np.nan)

    # 6) Compute rolling stats
    rolling_cols = ['PIE', 'USG_PCT', 'PTS', 'REB', 'AST', 'EFF', 'TS_PCT', 'MIN', 'FG_PCT', 'OFF_RATING', 'PACE_PER40']
    for c in rolling_cols:
        df[f'{c}_AVG_LAST_5'] = (df.groupby('PLAYER_NAME')[c]
                                 .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()))
        
    for c in ['PTS', 'USG_PCT', 'MIN']:
        df[f'{c}_VOL_LAST_5'] = (df.groupby('PLAYER_NAME')[c]
                                 .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).std()))

    # 7) Compute season average for PTS
    df['PTS_SEASON_AVG'] = (df.groupby('PLAYER_NAME')['PTS']
                              .transform(lambda x: x.shift(1).expanding().mean()))

    # 8) Additional features
    df['HOME_GAME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['REST_DAYS'] = df.groupby('PLAYER_NAME')['GAME_DATE'].diff().dt.days.fillna(0)

    # Rename and drop extraneous columns
    if 'TEAM_ID_x' in df.columns:
        df.rename(columns={'TEAM_ID_x': 'TEAM_ID'}, inplace=True)
    df.drop(columns=['TEAM_ID_y'], errors='ignore', inplace=True)

    # Forward/back fill to handle any missing rolling stats
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    return df

# %%

# =============================================================================
# 4. Aggregator Functions (Position & Team vs Opponent)
# =============================================================================

def add_player_position(df):
    """Add a 'POSITION' column (G, F, C) to each row."""
    cache = {}
    df['POSITION'] = df['PLAYER_ID'].apply(lambda pid: get_player_position(pid, cache))
    return df.dropna(subset=['POSITION'])

def compute_position_allowed_pts(df):
    """Calculate how many points each team concedes on average to a specific position."""
    agg = df.groupby(['OPPONENT_TEAM_ID', 'POSITION'])['PTS'].mean().reset_index()
    agg.rename(columns={'PTS': 'OPPONENT_POSITION_ALLOWED_PTS'}, inplace=True)
    return agg

def add_opponent_position_allowed_pts(df):
    """Merge the 'OPPONENT_POSITION_ALLOWED_PTS' back to the main DataFrame."""
    df_pos = add_player_position(df)
    agg = compute_position_allowed_pts(df_pos)
    return pd.merge(df_pos, agg, on=['OPPONENT_TEAM_ID', 'POSITION'], how='left')

def compute_team_vs_opponent_allowed_pts(df):
    """Compute average points a TEAM_ID scores vs. a specific OPPONENT_TEAM_ID."""
    agg = (df.groupby(['TEAM_ID', 'OPPONENT_TEAM_ID'])['PTS']
             .mean().reset_index(name='TEAM_VS_OPP_ALLOWED_PTS'))
    return agg

def add_team_vs_opponent_allowed_pts(df):
    """Merge 'TEAM_VS_OPP_ALLOWED_PTS' back to the main DataFrame."""
    agg = compute_team_vs_opponent_allowed_pts(df)
    return pd.merge(df, agg, on=['TEAM_ID', 'OPPONENT_TEAM_ID'], how='left')


# %% [markdown]
# ---------------------------------
# 
# 4. Data Preparation Functions

# %%
# =============================================================================
# 5. Data Splitting & Preparation
# =============================================================================

DEFAULT_FEATURE_COLS = [
    'PIE_AVG_LAST_5', 'USG_PCT_AVG_LAST_5', 'EFF_AVG_LAST_5', 'TS_PCT_AVG_LAST_5',
    'DEF_RATING', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'HOME_GAME', 'REST_DAYS',
    'PTS_AVG_LAST_5', 'REB_AVG_LAST_5', 'AST_AVG_LAST_5', 'FG_PCT_AVG_LAST_5',
    'MIN_AVG_LAST_5', 'OFF_RATING_AVG_LAST_5', 'PACE_PER40_AVG_LAST_5', 'PTS_SEASON_AVG', 
    'OPPONENT_POSITION_ALLOWED_PTS', 'TEAM_VS_OPP_ALLOWED_PTS',
    'PTS_VOL_LAST_5', 'USG_PCT_VOL_LAST_5', 'MIN_VOL_LAST_5',
    #'STARTERS_MISSING'  # newly added feature
]
feature_columns_list = ['PIE_AVG_LAST_5', 'USG_PCT_AVG_LAST_5', 'EFF_AVG_LAST_5', 'TS_PCT_AVG_LAST_5',
    'DEF_RATING', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'HOME_GAME', 'REST_DAYS',
    'PTS_AVG_LAST_5', 'REB_AVG_LAST_5', 'AST_AVG_LAST_5', 'FG_PCT_AVG_LAST_5',
    'MIN_AVG_LAST_5', 'OFF_RATING_AVG_LAST_5', 'PACE_PER40_AVG_LAST_5', 'PTS_SEASON_AVG', 
    'OPPONENT_POSITION_ALLOWED_PTS', 'TEAM_VS_OPP_ALLOWED_PTS',
    'PTS_VOL_LAST_5', 'USG_PCT_VOL_LAST_5', 'MIN_VOL_LAST_5',
    #'STARTERS_MISSING'  # newly added feature
]

# %%
def prepare_data(df, feature_cols=DEFAULT_FEATURE_COLS):
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols].copy()
    X['PLAYER_NAME'] = df['PLAYER_NAME'] # Keep player_name for reference
    y = df['PTS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_test_original = X_test.copy() # Keep a copy of X_test before scaling
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.drop(columns=['PLAYER_NAME']))
    X_test_scaled = scaler.transform(X_test.drop(columns=['PLAYER_NAME']))

    os.makedirs('lib', exist_ok=True)
    joblib.dump(scaler, 'lib/scaler.pkl')
    return X_train_scaled, X_test_scaled, y_train, y_test, X_test_original #X_test['PLAYER_NAME'].values


# %% [markdown]
# ----------------------------
# 
# 5. Model Training and Evaluation

# %%
# =============================================================================
# 6. Modeling & Evaluation
# =============================================================================
def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'Ridge': Ridge(),
        'BayesianRidge': BayesianRidge(),
        #'Lasso': Lasso(),
        #'ElasticNet': ElasticNet(),
        #'LassoLars': LassoLars(),
        #'SGDRegressor': SGDRegressor(),
    }

    best_model = None
    best_rmse = float('inf')

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"\n{name} Performance:\n  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    print(f"\nBest model: {type(best_model).__name__} with RMSE: {best_rmse:.2f}")
    return best_model

def evaluate_model(model, X_test_scaled, y_test, X_test_original):
    preds = model.predict(X_test_scaled) # Make predictions
    rmse = np.sqrt(mean_squared_error(y_test, preds)) # Evaluate metrics
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\nEvaluation on Test Data:\n  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
    
    # Build eval_df 
    eval_df = X_test_original.reset_index(drop=True).copy()
    eval_df['Actual_PTS']    = y_test.reset_index(drop=True)
    eval_df['Predicted_PTS'] = preds
    eval_df['Residual'] = eval_df['Actual_PTS'] - eval_df['Predicted_PTS']
    
    print("\nSample Predictions:")
    print(eval_df[['PLAYER_NAME', 'Actual_PTS', 'Predicted_PTS', 'Residual']].head(10))    
    return eval_df


# %% [markdown]
# --------------------------------
# 
# 7. Model Prediction Functions

# %%
# =============================================================================
# 7. Next-Game Prediction Logic
# =============================================================================

def get_team_defensive_stats(team_id, season='2024-25'):
    try:
        df = leaguedashteamstats.LeagueDashTeamStats(
            team_id_nullable=team_id, season=season,
            measure_type_detailed_defense='Defense', per_mode_detailed='PerGame',
            timeout=60
        ).get_data_frames()[0]
        return df[['TEAM_ID', 'DEF_RATING', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE']].iloc[0]
    except:
        return pd.Series([np.nan]*4, index=['TEAM_ID', 'DEF_RATING', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE'])


def get_next_game_info(player_team_id):
    next_game_date = datetime.now() #+ timedelta(days=1)
    max_days_ahead = 14

    for _ in range(max_days_ahead):
        game_date_str = next_game_date.strftime('%Y-%m-%d')
        try:
            scoreboard = scoreboardv2.ScoreboardV2(game_date=game_date_str)
            games = scoreboard.game_header.get_data_frame()
            team_games = games[(games['HOME_TEAM_ID'] == player_team_id) | (games['VISITOR_TEAM_ID'] == player_team_id)]
            if not team_games.empty:
                next_game = team_games.iloc[0]
                opponent_team_id = next_game['VISITOR_TEAM_ID'] if next_game['HOME_TEAM_ID'] == player_team_id else next_game['HOME_TEAM_ID']
                home_game = 1 if next_game['HOME_TEAM_ID'] == player_team_id else 0
                return next_game_date, opponent_team_id, home_game
        except:
            pass
        next_game_date += timedelta(days=1)

    return None, None, None

# %%
def prepare_features_for_prediction(player_id, player_name, season='2024-25', feature_cols=DEFAULT_FEATURE_COLS, 
                                    df_agg_position=None, df_agg_team_vs_opp=None):
    # 1) Pull player's logs and advanced stats.
    logs_df = get_player_game_logs(player_id, season)
    if logs_df.empty:
        print(f"No game logs for {player_name} in {season}.")
        return None, None
    logs_df['GAME_DATE'] = pd.to_datetime(logs_df['GAME_DATE'])
    logs_df.sort_values('GAME_DATE', ascending=False, inplace=True)
    logs_df['PLAYER_NAME'] = player_name

    # Team & next game
    p_team_id = get_player_team_id(player_id)
    next_game_date, opp_team_id, home_game = get_next_game_info(p_team_id)
    if not next_game_date:
        print(f"No upcoming game found for {player_name}.")
        return None, None

    # Opponent stats
    opp_stats = get_team_defensive_stats(opp_team_id, season)
    # Advanced stats
    adv_df = get_player_advanced_stats(player_id, season)

    # Build a small historical subset
    team_map = get_team_abbreviation_id_mapping()
    recent_logs = logs_df[logs_df['GAME_DATE'] <= logs_df['GAME_DATE'].max()]
    adv_df = adv_df[adv_df['GAME_ID'].isin(recent_logs['GAME_ID'])]
    final_df = feature_engineering(recent_logs, adv_df, None, team_map)

    if final_df.empty:
        return None, None

    # 2) Feature-engineer a 'latest_data' row (from last game).
    # 3) Insert upcoming game info: rest days, home/away, opponent stats.
    latest_data = final_df.iloc[-1].copy()
    latest_data['REST_DAYS'] = (next_game_date - latest_data['GAME_DATE']).days
    latest_data['HOME_GAME'] = home_game
    latest_data['GAME_DATE'] = next_game_date
    latest_data['OPPONENT_TEAM_ID'] = opp_team_id
    latest_data['DEF_RATING'] = opp_stats['DEF_RATING']
    latest_data['OPP_PTS_OFF_TOV'] = opp_stats['OPP_PTS_OFF_TOV']
    latest_data['OPP_PTS_2ND_CHANCE'] = opp_stats['OPP_PTS_2ND_CHANCE']
    
    # 4) Merge aggregator stats for position and team vs. opponent.
    # Position-based aggregator
    if df_agg_position is not None:
        pos = get_player_position(player_id)
        if not pos:  # fallback
            pos = 'G'
        row_match = df_agg_position[
            (df_agg_position['OPPONENT_TEAM_ID'] == opp_team_id) & (df_agg_position['POSITION'] == pos)
        ]
        if not row_match.empty:
            latest_data['OPPONENT_POSITION_ALLOWED_PTS'] = row_match['OPPONENT_POSITION_ALLOWED_PTS'].iloc[0]
        else:
            # fallback to position average
            fallback_pos = df_agg_position[df_agg_position['POSITION'] == pos]
            latest_data['OPPONENT_POSITION_ALLOWED_PTS'] = fallback_pos['OPPONENT_POSITION_ALLOWED_PTS'].mean()

    # Team vs Opp aggregator
    if df_agg_team_vs_opp is not None:
        row_match = df_agg_team_vs_opp[
            (df_agg_team_vs_opp['TEAM_ID'] == p_team_id) & (df_agg_team_vs_opp['OPPONENT_TEAM_ID'] == opp_team_id)
        ]
        if not row_match.empty:
            latest_data['TEAM_VS_OPP_ALLOWED_PTS'] = row_match['TEAM_VS_OPP_ALLOWED_PTS'].iloc[0]
        else:
            latest_data['TEAM_VS_OPP_ALLOWED_PTS'] = df_agg_team_vs_opp['TEAM_VS_OPP_ALLOWED_PTS'].mean()

    # Build the final feature vector
    try:
        fv = latest_data[feature_cols].values.reshape(1, -1)
        return fv, latest_data
    except KeyError as e:
        print(f"Missing columns for {player_name}: {e}")
        return None, None



# %%
def predict_upcoming_points(player_name, season='2024-25', feature_cols=DEFAULT_FEATURE_COLS, 
                            df_agg_position=None, df_agg_team_vs_opp=None):
    player_id = get_player_id(player_name)
    if not player_id:
        print(f"Invalid player: {player_name}")
        return None

    fv, latest_data = prepare_features_for_prediction(player_id, player_name, season, feature_cols=feature_cols,
                                                      df_agg_position=df_agg_position, df_agg_team_vs_opp=df_agg_team_vs_opp)
    if fv is None:
        return None
    
    # Scale & predict
    scaler = joblib.load('lib/scaler.pkl')
    model = joblib.load('lib/player_points_model.pkl')
    fv_scaled = scaler.transform(fv)
    pred = model.predict(fv_scaled)
    opp_name = get_team_name(latest_data['OPPONENT_TEAM_ID'])
    date_str = latest_data['GAME_DATE'].strftime('%Y-%m-%d')

    print(f"Predicted points for {player_name} on {date_str} vs {opp_name}: {pred[0]:.2f}")
    return pred[0]


# %% [markdown]
# -------
# 
# 8. Feature Importance
# 
# -----

# %%
# =============================================================================
# 8. Example Usage 
# =============================================================================

player_names = [
    "LeBron James", "Kevin Durant", "Stephen Curry",
    "Giannis Antetokounmpo", "Luka Dončić", "Joel Embiid",
    "Jayson Tatum", "Nikola Jokić", "Shai Gilgeous-Alexander",
    "Karl-Anthony Towns", "Victor Wembanyama", "Damian Lillard",
    "Donovan Mitchell", "Anthony Davis", "Domantas Sabonis",
    "James Harden", "Kyrie Irving", "Anthony Edwards", "Jimmy Butler",
    "De'Aaron Fox", "Jalen Brunson", "Bronny James", "Tyrese Maxey", 
    "Trae Young", "Pascal Siakam"]

season = '2024-25'
opponent_stats = get_opponent_stats(season)
team_map = get_team_abbreviation_id_mapping()

all_player_data = pd.DataFrame()
for p_name in player_names:
    p_id = get_player_id(p_name)
    if not p_id:
        continue
    p_gamelog = get_player_game_logs(p_id, season)
    adv_stats = get_player_advanced_stats(p_id, season)
    if p_gamelog.empty or adv_stats.empty:
        continue
    
    p_gamelog['PLAYER_NAME'] = p_name
    merged_df = feature_engineering(p_gamelog, adv_stats, opponent_stats, team_map)
    all_player_data = pd.concat([all_player_data, merged_df], ignore_index=True)

# 3) Position & Team aggregator
all_player_data = add_opponent_position_allowed_pts(all_player_data)
all_player_data = add_team_vs_opponent_allowed_pts(all_player_data)


# %%
# X_train_scaled, X_test_scaled, y_train, y_test, p_names_test = prepare_data(all_player_data)
X_train_scaled, X_test_scaled, y_train, y_test, X_test_original = prepare_data(all_player_data) 
best_model = train_and_evaluate_models(X_train_scaled, y_train, X_test_scaled, y_test)
joblib.dump(best_model, 'lib/player_points_model.pkl')

# Build eval_df for residual analysis | after training:
eval_df = evaluate_model(best_model, X_test_scaled, y_test, X_test_original)


# %%
# 7) Predict upcoming game for each player
#    (requires the aggregator dataframes if we want position-based features)
df_agg_position = compute_position_allowed_pts(all_player_data)  # merges OPP_TEAM & POSITION -> mean(PTS)
df_agg_team_opp = compute_team_vs_opponent_allowed_pts(all_player_data)
for name in player_names:
    predict_upcoming_points(
        name, season, df_agg_position=df_agg_position, df_agg_team_vs_opp=df_agg_team_opp)

# %% [markdown]
# -------

# %% [markdown]
# -----
# 
# ###### Evaluation Residuals
# 
# 

# %%

# Now we can do residual analysis
plt.figure(figsize=(8,5))
sns.histplot(eval_df['Residual'], kde=True, bins=20)
plt.title("Histogram of Residuals")
plt.show()


# %% [markdown]
# ###### Starters Missing

# %% [markdown]
# ###### Time Series Cross Validation
# 
# 

# %%
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

def time_series_cv_evaluation(df, feature_cols, target_col='PTS', n_splits=5):
    """Perform time-series cross-validation with n_splits folds."""
    # Sort by date
    df_sorted = df.sort_values(by='GAME_DATE').reset_index(drop=True)
    X_full = df_sorted[feature_cols].copy()
    y_full = df_sorted[target_col].values

    # Prepare cross-validator
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []
    r2_scores = []
    fold_number = 1
    for train_index, val_index in tscv.split(X_full):
        X_train, X_val = X_full.iloc[train_index], X_full.iloc[val_index] # Split
        y_train, y_val = y_full[train_index], y_full[val_index]

        scaler = StandardScaler() # Scale
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)

        # (You can choose whichever model you want here. Let's do a simple RandomForest as an example.)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train_scaled, y_train)

        # Predict
        val_preds = model.predict(X_val_scaled)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        rmse_scores.append(rmse)

        r2 = model.score(X_val_scaled, y_val)
        r2_scores.append(r2)
        
        print(f"Fold {fold_number} RMSE = {rmse:.3f} R-Squared = {r2:.3f}")
        fold_number += 1

    avg_rmse = np.mean(rmse_scores)
    avg_r2 = np.mean(r2_scores)
    print(f"\nAverage RMSE over {n_splits} folds: {avg_rmse:.3f}")
    print(f"\nAverage R-Squared over {n_splits} folds: {avg_r2:.3f}")
    return avg_rmse



# %%
time_series_cv_evaluation(all_player_data, DEFAULT_FEATURE_COLS, target_col='PTS', n_splits=5)

# %% [markdown]
# ----
# 

# %% [markdown]
# ###### Position and Role-Based Features:
# 
# Include data about the player's role (e.g., starter vs. bench), player position, and how the upcoming opponent typically defends that position.
# 

# %%
from lib.build_player_team_data import main
df_player_full = main(season='2024-25', data_file="data/merged_player_team_dataset.csv")

# %% [markdown]
# ------------------------------------

# %% [markdown]
# 
# 
# Hyperparameter Tuning and Bayesian Optimization:
# Instead of a simple grid search, use advanced hyperparameter optimization methods (e.g., Bayesian optimization or Optuna) to find the best parameters for CatBoost, XGBoost, LightGBM, or neural networks.
# Non-Linear Models and Neural Networks:
# Consider deep learning approaches. A simple feed-forward neural network or LSTM/RNN if you structure your data as a time series could capture temporal dependencies more effectively.
# Time-Series Aware Validation:
# 
# Ensure you use proper time-series cross-validation (e.g., TimeSeriesSplit) so the model isn't accidentally leaking future information.
# 
# 
# Betting Lines or Market Data:
# Market-based indicators (like the Vegas over/under for the game) can indirectly capture external knowledge about expected scoring environment.
# 
# 
# Dimensionality Reduction and Feature Selection:
# 
# Feature Importance and Pruning:
# Use model explainability tools (SHAP, feature_importances_) to identify less useful features and remove them.

# %% [markdown]
# -----
# 
# # Star Player Out
# 
# Handling Injuries / Roster Changes
# 
# If a key teammate is absent, a star player’s usage might spike. Consider adding a feature that tracks “number of typical starters missing” for a given game. That often impacts scoring opportunities.
# 
# Explainability
# 
# Tools like SHAP or feature importances from tree-based models can help you see which inputs drive the predictions. This can help you debug or refine features.
# 
# 
# 
# ######  Pipeline Packaging
# 
# Once stable, you can wrap the entire pipeline in a script (or notebook) that daily:
# Pulls updated logs.
# Retrains/updates model (if desired) or just re-scores with the existing model.
# Outputs next-game predictions to a CSV or database.
# 
# 
# 

# %% [markdown]
# --------------
# 
# 
# # Injury Report

# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

def scrape_espn_injuries(url="https://www.espn.com/nba/injuries"):
    """Returns a DataFrame with columns: [TEAM_NAME, PLAYER_NAME, POS, EST_RETURN, STATUS, COMMENT]."""
    # 1) Add headers that mimic a real browser
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/107.0.0.0 Safari/537.36"
        )
    }
    # 2) Make the request
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch {url}, status code: {response.status_code}")
        return pd.DataFrame()

    # 3) Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # 4) The main container for injuries is identified by:
    #    <div class="ResponsiveTable Table__league-injuries"> ...table content...
    #    We'll find all such sections for different teams.

    # "ResponsiveTable Table__league-injuries" is repeated per team
    # We might see multiple <div> blocks with that class
    team_tables = soup.find_all("div", class_="ResponsiveTable Table__league-injuries")

    all_rows = [] # We'll store results in a list of dicts

    for table_div in team_tables:
        # Each 'table_div' should contain a <div class="Table__Title"> for the team name
        # and a <table> with <thead>/<tbody> for the injuries.

        # 1) Get the Team Name
        title_div = table_div.find("div", class_="Table__Title")
        if not title_div:
            # If we can't find the title, skip
            continue

        # The team name is often in: <span class="injuries__teamName ...">TEAM NAME</span>
        team_name_span = title_div.find("span", class_="injuries__teamName")
        if not team_name_span:
            continue
        team_name = team_name_span.get_text(strip=True)

        # 2) The <table> has a <thead> and <tbody> with multiple <tr> rows.
        # Typically: <tbody class="Table__TBODY"><tr> ... <td> ... etc.
        table_tag = table_div.find("table", class_="Table")
        if not table_tag:
            continue

        tbody = table_tag.find("tbody", class_="Table__TBODY")
        if not tbody:
            continue

        # 3) Each row in the <tbody> is one player's injury record
        rows = tbody.find_all("tr", class_="Table__TR")
        for row in rows:
            # We have multiple <td> columns: NAME, POS, EST. RETURN DATE, STATUS, COMMENT
            tds = row.find_all("td", class_="Table__TD")
            if len(tds) < 5:
                # Expect at least 5 columns
                continue

            player_name = tds[0].get_text(strip=True)
            pos = tds[1].get_text(strip=True)
            est_return = tds[2].get_text(strip=True)
            status = tds[3].get_text(strip=True)
            comment = tds[4].get_text(strip=True)

            # Store in a dict
            all_rows.append({
                "TEAM_NAME": team_name,
                "PLAYER_NAME": player_name,
                "POS": pos,
                "EST_RETURN": est_return,
                "STATUS": status, 
                "COMMENT": comment
            })

    # Convert to DataFrame
    df_injury = pd.DataFrame(all_rows)
    return df_injury


df_injury = scrape_espn_injuries("https://www.espn.com/nba/injuries")
if df_injury.empty:
    print("No data or scraping failed.")

# Add a DATA_DATE
df_injury["DATA_DATE"] = datetime.now().strftime("%Y-%m-%d")

# Save to CSV
csv_file = f"data/injury_reports/injury_report_{df_injury['DATA_DATE'].iloc[0]}.csv"
df_injury.to_csv(csv_file, index=False)
print(f"Saved injury data to {csv_file}")



# %%
df_injury[60:85]

# %% [markdown]
# Star Player Identification
# 
# You can use “typical_starters_dict” or any advanced approach to identify who’s considered a “key player” for each team. Then, if they’re out, your model can see a bigger effect on the minutes or usage of the rest of the team.
# 
# Minutes vs. Points
# 
# Often, you’ll first build a minutes model (that uses IS_OUT, TEAM_HAS_STAR_OUT) and outputs MIN_PROJ. Then you feed MIN_PROJ + other features into your points model.
# The partial historical injuries help that minutes model learn “When star is out, player X’s minutes jump by 5.”

# %% [markdown]
# ----

# %% [markdown]
# 4. Pipeline Automation & Data Storage
# 4.1 Scheduled Updates
# Set up a daily or weekly job that:
# Pulls fresh game logs and advanced box scores via the NBA API (or your own data store).
# Updates your training dataset.
# Optionally retrains or re-fits the model.
# Generates new next-game predictions for each player.
# 4.2 Data Storage
# Use a database (e.g., SQLite, PostgreSQL) or a Cloud Data Warehouse (BigQuery, Snowflake) to store:
# Historical game logs
# Player metadata (e.g., birthdate, draft year, position, injuries)
# Team stats / synergy metrics
# This central repository simplifies repeated queries and ensures you have a single source of truth.
# 4.3 Version Control & Logging
# Version your model artifacts (e.g., model_v1.0.pkl, model_v1.1.pkl) and keep a record of:
# Date of training
# Hyperparameters
# Performance metrics
# Log daily predictions and compare them later to actual game results to measure real-time accuracy.

# %% [markdown]
# . Perform a Residual Analysis
# Generate a residual DataFrame: df_residuals = actual_points - predicted_points.
# Plot histograms, scatter plots (residuals vs. minutes or usage), or groupby stats (residuals by team or position).
# Look for patterns that might suggest missing features or systematic biases.

# %% [markdown]
# 

# %%


# %% [markdown]
# ------

# %%
from lib.cleanup_script import remove_markdown_blocks_and_reformat

remove_markdown_blocks_and_reformat("notebooks/test.py", "notebooks/test_cleaned.py")


# %%



