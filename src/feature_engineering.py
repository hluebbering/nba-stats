# src/feature_engineering.py
import numpy as np
import pandas as pd

# ===============================
# Core Feature Engineering Pipeline
# ===============================

def compute_efficiency(df):
    df['EFF'] = (
        df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK']
        - (df['FGA'] - df['FGM']) - (df['FTA'] - df['FTM']) - df['TOV']
    )
    return df


def compute_true_shooting(df):
    df['TS_DENOM'] = 2 * (df['FGA'] + 0.44 * df['FTA'])
    df['TS_PCT'] = np.where(df['TS_DENOM'] != 0, df['PTS'] / df['TS_DENOM'], 0)
    df.drop(columns=['TS_DENOM'], inplace=True)
    return df


def parse_minutes(min_str):
    if isinstance(min_str, str) and ':' in min_str:
        mins, secs = min_str.split(':', 1)
        return float(mins) + float(secs) / 60
    try:
        return float(min_str)
    except:
        return 0.0


def add_rolling_features(df, group_col='PLAYER_ID'):
    # Rolling averages including advanced stats
    roll_cols = ['PTS','REB','AST','EFF','TS_PCT','MIN','USG_PCT','FG_PCT','OFF_RATING','PACE_PER40','PIE']

    for col in roll_cols:
        df[f'{col}_AVG_LAST_5'] = (
            df.groupby(group_col)[col]
              .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
    # Volatility for key metrics
    vol_cols = ['PTS', 'USG_PCT', 'MIN']
    for col in vol_cols:
        df[f'{col}_VOL_LAST_5'] = (
            df.groupby(group_col)[col]
              .transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())
        )
    return df


def feature_engineering_pipeline(df, team_map=None, opp_df=None, opp_last10_df=None):
    # 1. Rename TEAM_ID if merged with suffix
    if 'TEAM_ID_x' in df:
        df.rename(columns={'TEAM_ID_x': 'TEAM_ID'}, inplace=True)
    if 'TEAM_ID_y' in df:
        df.drop(columns=['TEAM_ID_y'], inplace=True)

    # 2. Base stats
    df = compute_efficiency(df)
    df = compute_true_shooting(df)

    # 3. Parse minutes and shooting percentages
    if 'MIN' in df.columns:
        df['MIN'] = df['MIN'].apply(parse_minutes)
    else:
        df['MIN'] = 0.0
    df['FG_PCT'] = df['FGM'] / df['FGA'].replace(0, np.nan)

    # 4. Opponent mapping and home indicator
    if team_map is not None and 'MATCHUP' in df.columns:
        df['OPPONENT_ABBREV'] = df['MATCHUP'].str.split().str[-1]
        df['OPPONENT_TEAM_ID'] = df['OPPONENT_ABBREV'].map(team_map)
        df['HOME_GAME'] = df['MATCHUP'].str.contains(r'vs\.').astype(int)

    # 5. Merge opponent defensive stats
    if opp_df is not None and 'OPPONENT_TEAM_ID' in df.columns:
        df = df.merge(
            opp_df,
            left_on='OPPONENT_TEAM_ID',
            right_on='TEAM_ID',
            how='left',
            suffixes=('', '_opp')
        )
        # option A: assign back
        for col in ['DEF_RATING','OPP_PTS_OFF_TOV','OPP_PTS_2ND_CHANCE']:
            df[col] = df[col].fillna(df[col].mean())
            
    
    # 5-b. Merge LAST-10 opponent defense
    if opp_last10_df is not None and 'OPPONENT_TEAM_ID' in df.columns:
        df = df.merge(
            opp_last10_df,
            left_on='OPPONENT_TEAM_ID',
            right_on='TEAM_ID',
            how='left',
            suffixes=('', '_L10')
        )

        # Drop duplicate TEAM_ID from the right-side merge
        if 'TEAM_ID_L10' in df.columns:
            df.drop(columns=['TEAM_ID_L10'], inplace=True)

        # Fill NaNs in the new last-10 columns
        fill_cols = [
            'DEF_RATING_LAST10',
            'OPP_PTS_OFF_TOV_LAST10',
            'OPP_PTS_2ND_CHANCE_LAST10'
        ]
        df[fill_cols] = df[fill_cols].fillna(df[fill_cols].mean())
        df[fill_cols] = df[fill_cols].fillna(0)   # second pass


# -------------------------------------------------




    # 6. Sort and compute rest days
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
    df.sort_values(['PLAYER_ID', 'GAME_DATE'], inplace=True)
    df['REST_DAYS'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days.fillna(0)

    # 7. Rolling features
    df = add_rolling_features(df)

    # 8. Season average for points
    df['PTS_SEASON_AVG'] = (
        df.groupby('PLAYER_ID')['PTS']
          .transform(lambda x: x.shift(1).expanding().mean())
    )

    # 9. Forward/backward fill missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df
