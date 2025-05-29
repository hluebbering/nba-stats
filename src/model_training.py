# src/model_training.py
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from catboost import CatBoostRegressor
import pandas as pd

DEFAULT_FEATURE_COLS = [
    'PIE_AVG_LAST_5', 'USG_PCT_AVG_LAST_5', 'EFF_AVG_LAST_5', 'TS_PCT_AVG_LAST_5',
    'DEF_RATING', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'HOME_GAME', 'REST_DAYS',
    'PTS_AVG_LAST_5', 'REB_AVG_LAST_5', 'AST_AVG_LAST_5', 'FG_PCT_AVG_LAST_5', 'MIN_AVG_LAST_5',
    'OFF_RATING_AVG_LAST_5', 'PACE_PER40_AVG_LAST_5', 'PTS_SEASON_AVG',
    'OPPONENT_POSITION_ALLOWED_PTS', 'TEAM_VS_OPP_ALLOWED_PTS',
    'PTS_VOL_LAST_5', 'USG_PCT_VOL_LAST_5', 'MIN_VOL_LAST_5'
]


DEFAULT_FEATURE_COLS = [
    'PIE_AVG_LAST_5', 'USG_PCT_AVG_LAST_5', 'EFF_AVG_LAST_5', 'TS_PCT_AVG_LAST_5',
    'DEF_RATING', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'HOME_GAME', 'REST_DAYS',
    'PTS_AVG_LAST_5', 'REB_AVG_LAST_5', 'AST_AVG_LAST_5', 'FG_PCT_AVG_LAST_5',
    'MIN_AVG_LAST_5', 'OFF_RATING_AVG_LAST_5', 'PACE_PER40_AVG_LAST_5', 'PTS_SEASON_AVG',
    'OPPONENT_POSITION_ALLOWED_PTS', 'TEAM_VS_OPP_ALLOWED_PTS', 'PTS_VOL_LAST_5',
    'USG_PCT_VOL_LAST_5', 'MIN_VOL_LAST_5'
]




def prepare_data(df, feature_cols=DEFAULT_FEATURE_COLS):
    # Keep only columns that exist in df
    existing_features = [col for col in feature_cols if col in df.columns]
    missing_features = set(feature_cols) - set(existing_features)
    if missing_features:
        print(f"Warning: Missing features skipped: {missing_features}")

    df = df.dropna(subset=existing_features)
    X = df[existing_features].copy()
    y = df['PTS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, 'lib/scaler.pkl')
    return X_train_scaled, X_test_scaled, y_train, y_test, X_test.reset_index(drop=True)





def train_models(X_train, y_train, X_test, y_test):
    models = {
        'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'Ridge': Ridge(),
        'BayesianRidge': BayesianRidge()
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
    joblib.dump(best_model, 'lib/player_points_model.pkl')
    return best_model

def evaluate_model(model, X_test_scaled, y_test, X_test_original):
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\nEvaluation on Test Data:\n  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    eval_df = X_test_original.copy()
    eval_df['Actual_PTS'] = y_test.reset_index(drop=True)
    eval_df['Predicted_PTS'] = preds
    eval_df['Residual'] = eval_df['Actual_PTS'] - eval_df['Predicted_PTS']

    return eval_df
