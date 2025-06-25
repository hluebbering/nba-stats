# src/aggregators.py
import pandas as pd
from src.utils import get_player_position

# === PLAYER POSITION ===
def add_player_position(df):
    """Add a 'POSITION' column (G, F, C) for each row based on player role."""
    cache = {}
    def _get(pos_id):
        if pos_id in cache:
            return cache[pos_id]
        position = get_player_position(pos_id)
        cache[pos_id] = position
        return position

    df['POSITION'] = df['PLAYER_ID'].apply(_get)
    return df.dropna(subset=['POSITION'])





# === AGGREGATOR: OPPONENT POSITION ALLOWED PTS ===
def compute_position_allowed_pts(df_pos):
    """Compute average points conceded by each team to each position."""
    agg = df_pos.groupby(['OPPONENT_TEAM_ID', 'POSITION'])['PTS'].mean().reset_index()
    agg.rename(columns={'PTS': 'OPPONENT_POSITION_ALLOWED_PTS'}, inplace=True)
    return agg

def add_opponent_position_allowed_pts(df):
    """Merge opponent position allowed points into main DataFrame."""
    df_pos = add_player_position(df)
    agg = compute_position_allowed_pts(df_pos)
    return pd.merge(df_pos, agg, on=['OPPONENT_TEAM_ID', 'POSITION'], how='left')


# aggregators.py
def add_opponent_position_allowed_pts(df):
    # 1️⃣ ensure each row has a POSITION (G / F / C)
    df_pos = add_player_position(df)          # ← this was missing
    
    # 2️⃣ build the aggregate table
    agg = compute_position_allowed_pts(df_pos)
    
    # 3️⃣ left-join so we never lose rows
    df_pos = df_pos.merge(agg, on=['OPPONENT_TEAM_ID', 'POSITION'], how='left')
    
    # 4️⃣ fill any still-missing values with the global mean
    df_pos['OPPONENT_POSITION_ALLOWED_PTS'] = (
        df_pos['OPPONENT_POSITION_ALLOWED_PTS']
          .fillna(df_pos['OPPONENT_POSITION_ALLOWED_PTS'].mean())
    )
    return df_pos





# === AGGREGATOR: TEAM VS OPPONENT ALLOWED PTS ===
def compute_team_vs_opponent_allowed_pts(df):
    """Compute average points a team scores vs each opponent."""
    agg = df.groupby(['TEAM_ID', 'OPPONENT_TEAM_ID'])['PTS'].mean().reset_index(name='TEAM_VS_OPP_ALLOWED_PTS')
    return agg

def add_team_vs_opponent_allowed_pts(df):
    """Merge team vs opponent allowed points into DataFrame."""
    agg = compute_team_vs_opponent_allowed_pts(df)
    return pd.merge(df, agg, on=['TEAM_ID', 'OPPONENT_TEAM_ID'], how='left')
