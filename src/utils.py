# src/utils.py
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonplayerinfo

# === PLAYER & TEAM LOOKUPS ===

def get_player_id(name):
    """Get the NBA player ID given the player's full name."""
    all_players = players.get_players()
    player = next((p for p in all_players if p['full_name'].lower() == name.lower()), None)
    return player['id'] if player else None


def get_player_position(player_id):
    """Get simplified position (G, F, C) for a player ID."""
    try:
        df = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        raw_pos = df.get('POSITION', [''])[0]
        main_pos = raw_pos.split('-')[0]
        if 'Guard' in main_pos:
            return 'G'
        if 'Forward' in main_pos:
            return 'F'
        if 'Center' in main_pos:
            return 'C'
    except:
        pass
    return None


def get_player_team_id(player_id):
    """Get the player's current team ID."""
    try:
        df = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        return int(df['TEAM_ID'].iloc[0])
    except:
        return None


def get_team_abbreviation_id_mapping():
    """Return a dict mapping team abbreviations to team IDs."""
    nba_teams = teams.get_teams()
    return {team['abbreviation']: team['id'] for team in nba_teams}


def get_team_name(team_id):
    """Get the full team name given the team ID."""
    nba_teams = teams.get_teams()
    team = next((t for t in nba_teams if t['id'] == team_id), None)
    return team['full_name'] if team else 'Unknown Team'
