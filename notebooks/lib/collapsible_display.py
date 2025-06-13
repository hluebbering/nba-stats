import pandas as pd
from IPython.display import display, HTML

# Data dictionary with categories and variable descriptions
nba_data_dict = {
    "Identifiers": {
        "data": {
            "PLAYER_ID": "Unique Identifier for the Player",
            "PLAYER_NAME": "Name of the Player"
        }
    },

    "Target Variable": {
        #"icon": "https://3dmodelsworld.com/wp-content/uploads/2022/12/basketball-ball-pbr-3d-model-physically-based-rendering-ta.webp",
        "icon": "./assets/images/bball3d.png",
        "data": {
            "PIE": "Player Impact Estimate"
        }
    },

    "Player Stats (Totals)": {
        "icon": "./assets/images/antman.png",
        "data": {
            "MIN": "Minutes Played",
            "PTS": "Points Scored",
            "REB": "Total Rebounds",
            "OREB": "Offensive Rebounds",
            "DREB": "Defensive Rebounds",
            "AST": "Assists",
            "STL": "Steals",
            "BLK": "Blocks",
            "TOV": "Turnovers",
            "PF": "Personal Fouls",
            "FGA": "Field Goals Attempted",
            "FGM": "Field Goals Made",
            "FTA": "Free Throws Attempted",
            "FTM": "Free Throws Made",
            "FG3A": "3-Pointers Attempted",
            "FG3M": "3-Pointers Made",
            "PLUS_MINUS": "Plus/Minus Rating",
            "FANTASY_PTS": "Fantasy Points"
        }
    },


    "Per-Game Stats": {
        "icon": "./assets/images/score-board.svg",
        "data": {
            "GAMES_PLAYED": "Games Played",
            "PTS_per_Game": "Points per Game",
            "REB_per_Game": "Rebounds per Game",
            "AST_per_Game": "Assists per Game",
            "STL_per_Game": "Steals per Game",
            "BLK_per_Game": "Blocks per Game",
            "TOV_per_Game": "Turnovers per Game",
            "PF_per_Game": "Personal Fouls per Game"
        }
    },

    "Advanced Metrics": {
        "icon": "./assets/images/courtside.png",
        "data": {
            "TS%": "True Shooting Percentage",
            "eFG%": "Effective Field Goal Percentage",
            "AST_TOV_Ratio": "Assist-to-Turnover Ratio"
        }
    },

    "Game Context": {
        "icon": "./assets/images/scoreboard4.png",
        "data": {
            "HOME": "Home/Away Indicator",
            "W/L": "Win/Loss Indicator"
        }
    },
    
    
    "Team Stats": {
        "icon": "https://cdn.inspireuplift.com/uploads/images/seller_products/1684438777_1.png",
        "data": {
        "FG_PCT_TEAM": "Team Field Goal Percentage",
        "FG3_PCT_TEAM": "Team 3-Point Percentage",
        "FT_PCT_TEAM": "Team Free Throw Percentage",
        "REB_TEAM": "Total Team Rebounds",
        "AST_TEAM": "Total Team Assists",
        "TO": "Team Turnovers",
        "FGA_TEAM": "Team Field Goals Attempted",
        "FGM_TEAM": "Team Field Goals Made",
        "FTA_TEAM": "Team Free Throws Attempted",
        "FTM_TEAM": "Team Free Throws Made",
        "FG3A_TEAM": "Team 3-Pointers Attempted",
        "FG3M_TEAM": "Team 3-Pointers Made",
        "PLUS_MINUS_TEAM": "Team Plus/Minus Rating"
        }
    },
    
    
    "Opponent Stats": {
        "icon": "./assets/images/jokic.png",
        
        "data": {
        "FG_PCT_OPPONENT": "Opponent Field Goal Percentage",
        "FG3_PCT_OPPONENT": "Opponent 3-Point Percentage",
        "FT_PCT_OPPONENT": "Opponent Free Throw Percentage",
        "REB_OPPONENT": "Opponent Total Rebounds",
        "AST_OPPONENT": "Opponent Total Assists",
        "TO_OPPONENT": "Opponent Turnovers",
        "FGA_OPPONENT": "Opponent Field Goals Attempted",
        "FGM_OPPONENT": "Opponent Field Goals Made",
        "FTA_OPPONENT": "Opponent Free Throws Attempted",
        "FTM_OPPONENT": "Opponent Free Throws Made",
        "FG3A_OPPONENT": "Opponent 3-Pointers Attempted",
        "FG3M_OPPONENT": "Opponent 3-Pointers Made",
        "PLUS_MINUS_OPPONENT": "Opponent Plus/Minus Rating"
        }
    },
    
    "Performance Ratios": {
        "icon": "./assets/images/shooting7.png",
        "data": {
        "FG_PCT_Ratio": "Field Goal Percentage Ratio (Team/Opponent)",
        "FG3_PCT_Ratio": "3-Point Percentage Ratio (Team/Opponent)",
        "FT_PCT_Ratio": "Free Throw Percentage Ratio (Team/Opponent)",
        "REB_Ratio": "Rebound Ratio (Team/Opponent)",
        "AST_Ratio": "Assist Ratio (Team/Opponent)",
        "TOV_Ratio": "Turnover Ratio (Team/Opponent)",
        "PLUS_MINUS_Ratio": "Plus/Minus Ratio (Team/Opponent)"
        }
    }
}



def display_collapsible_data_dict(data_dict=nba_data_dict):
    """
    Display a compact, collapsible, and 3D-styled HTML view of a data dictionary using a single NBA icon.
    
    Parameters:
    - data_dict: Dictionary with categories as keys and sub-dictionaries containing variable-description pairs as values.
    """
    # Define the NBA logo URL
    nba_logo_url = "https://cdn.nba.com/logos/leagues/logo-nba.svg"
    nba_logo_url = "https://pngimg.com/uploads/nba/nba_PNG8.png"
    nba_logo_url = "https://images.squarespace-cdn.com/content/v1/601b29d0be770940c3a3053b/1613529644365-LHGHHI1CRSZ59L6BYDNE/NBA_Logoman_word.png"
    
    html_content = f"""
    <style>
        /* Importing Custom Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Open+Sans:wght@300;600&display=swap');
        
        /* Container Styling */
        .table-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
            font-family: 'Roboto', sans-serif;
            font-size: 0.85em;
            max-width: 100%;
            margin: 0 auto;
        }}

        /* 3D Card Effect */
        .table-section {{
            width: 31%;
            border-radius: 6px;
            background-color: #ffffff;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15), 0 1px 4px rgba(0, 0, 0, 0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .table-section:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2), 0 4px 8px rgba(0, 0, 0, 0.15);
        }}

        /* Category Header with Gradient and Font Customization */
        .category-header {{
            background: linear-gradient(135deg, #C8102E, #a60d24);
            color: #ffffff;
            padding: 8px;
            font-weight: 600;
            font-family: 'Open Sans', sans-serif;
            letter-spacing: 0.25px;
            text-shadow: 0 1px 0 black;
            font-size: 11px;
            cursor: pointer;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            transition: background-color 0.3s ease;
        }}
        .category-header:hover {{
            background: linear-gradient(135deg, #a60d24, #900b1d);
        }}

        /* Collapsible Content with Slide Effect */
        .category-content {{
            display: none;
            background-color: #ffffff;
            font-family: 'Roboto', sans-serif;
            animation: slideDown 0.4s ease-out forwards;
        }}
        @keyframes slideDown {{
            from {{ max-height: 0; opacity: 0; }}
            to {{ max-height: 600px; opacity: 1; }}
        }}

        /* Table Styling with Custom Fonts */
        .table-container table {{
            width: 100%;
            border-collapse: collapse;
            margin: 0;
        }}
        .table-container th, .table-container td {{
            padding: 6px 8px;
            text-align: left;
            font-family: 'Roboto', sans-serif;
            border-bottom: 1px solid #eee;
        }}
        .table-container th {{
            background-color: #002D72;
            color: #ffffff;
            font-weight: 700;
            font-family: 'Open Sans', sans-serif;
        }}
        .table-container tr:nth-child(even) {{
            background-color: #fafafa;
        }}
        .table-container tr:hover td {{
            background-color: #f1f5f9;
            box-shadow: inset 0px 1px 3px rgba(0,0,0,0.1);
        }}

        /* NBA Logo Styling */
        .nba-logo {{
            width: 26px;
            height: 16px;
            object-fit: contain;
            transform: scale(1.4);
            margin-right: 10px;
            margin-left: -6px;
            vertical-align: middle;
            filter: drop-shadow(0.5px 0.5px 0.25px white) saturate(1.25) contrast(1.035) drop-shadow(-0.125px -0.0125px 0.125px #090909ff);
        }}
    </style>

    <script>
        function toggleContent(id) {{
            var content = document.getElementById(id);
            content.style.display = content.style.display === "none" || content.style.display === "" ? "block" : "none";
        }}
    </script>

    <div class="table-container">
    """
    
    # Generate collapsible sections with NBA logo
    section_id = 0
    for category, content in data_dict.items():
        
        icon_url = content.get("icon", nba_logo_url)
        variables = content["data"]  # Get the actual data
        
        html_content += f"""
        <div class="table-section">
            <div class="category-header" onclick="toggleContent('section-{section_id}')">
                <img src="{icon_url}" class="nba-logo"> {category}
            </div>
            <div class="category-content" id="section-{section_id}">
                <table>
                    <tr><th>Variable</th><th>Description</th></tr>
        """
        for var, desc in variables.items():
            html_content += f"<tr><td>{var}</td><td>{desc}</td></tr>"
        
        html_content += """
                </table>
            </div>
        </div>
        """
        section_id += 1

    html_content += "</div>"
    display(HTML(html_content))


# Call the function with example data
# display_collapsible_data_dict(data_dict, title="NBA Player Data Dictionary")
