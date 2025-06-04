import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://raw.githubusercontent.com/wacckky/ELECTIVE_FINALS/26ac0f1d8e3a74ed1cab850abda7273503534215/background.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }
    .stApp * { color: white !important; }
    section[data-testid="stSidebar"] * { color: white !important; }
    input, select, textarea, .stSlider > div, .stSelectbox div[data-baseweb="select"] * {
        color: black !important;
    }
    div[data-baseweb="select"] span { color: black !important; }
    .stSlider label, .stSlider div[data-testid="stTickBar"] span { color: black !important; }
    div.stButton > button {
        color: white !important;
        background-color: #FF4B4B !important;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #FF3333 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Load model and scaler ===
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("xgb_model2.pkl")
    scaler = joblib.load("xgb_scaler2.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# === Load data ===
@st.cache_data
def load_data():
    df = pd.read_csv("MAINDATA.csv", encoding="ISO-8859-1")
    df = df.drop_duplicates()
    df['Player'] = df['Player'].str.strip().str.title()
    df['Tm'] = df['Tm'].str.strip().str.upper()
    df['Opp'] = df['Opp'].str.strip().str.upper()
    df['Res'] = df['Res'].map({'W': 1, 'L': 0})
    
    numeric_cols = ['MP', 'PTS', 'TRB', 'AST', 'TOV', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA',
                    'ORB', 'DRB', 'PF']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(df.mean(numeric_only=True), inplace=True)

    if {'PTS', 'TRB', 'AST', 'TOV'}.issubset(df.columns):
        df['EFFICIENCY'] = df['PTS'] + df['TRB'] + df['AST'] - df['TOV']

    return df

df = load_data()

# Map for display purposes only
team_name_map = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BRK": "Brooklyn Nets",
    "CHI": "Chicago Bulls", "CHO": "Charlotte Hornets", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"
}
reverse_map = {v: k for k, v in team_name_map.items()}

# === Define feature columns ===
categorical_cols = ['Player', 'Tm', 'Opp']
numeric_cols = ['MP', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA',
                'ORB', 'DRB', 'TOV', 'PF', 'EFFICIENCY', 'Res']

@st.cache_data
def get_training_columns():
    df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
    feature_cols = list(df_encoded.columns) + numeric_cols
    return feature_cols

training_columns = get_training_columns()

# === UI ===
st.title("NBA Player Performance Predictor")
st.write("Select a team, player, and opponent to predict stats.")

teams = sorted(df['Tm'].unique())
teams_full_names = [team_name_map[t] for t in teams]
selected_team_full = st.selectbox("Select Player's Team:", teams_full_names)
selected_team = reverse_map[selected_team_full]

players_from_team = sorted(df[df['Tm'] == selected_team]['Player'].unique())
selected_player = st.selectbox("Select Player:", players_from_team)

opponents = sorted(df['Opp'].unique())
opponents_full_names = [team_name_map[o] for o in opponents]
selected_opponent_full = st.selectbox("Select Opponent Team:", opponents_full_names)
selected_opponent = reverse_map[selected_opponent_full]

if st.button("Predict Performance"):
    player_data = df[(df['Player'] == selected_player) & (df['Opp'] == selected_opponent)]

    if player_data.empty:
        st.warning(f"No game data for {selected_player} vs {selected_opponent_full}")
        player_only = df[df['Player'] == selected_player]
        if not player_only.empty:
            st.info(f"This player has played vs: {', '.join([team_name_map[o] for o in player_only['Opp'].unique()])}")
    else:
        avg_numeric = player_data[numeric_cols].mean().to_frame().T
        avg_cat = player_data[categorical_cols].iloc[0:1]

        cat_encoded = pd.get_dummies(avg_cat, drop_first=True)
        X_input = pd.concat([cat_encoded.reset_index(drop=True), avg_numeric.reset_index(drop=True)], axis=1)
        X_input = X_input.reindex(columns=training_columns, fill_value=0)

        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)
        pts, trb, ast = prediction[0]
        pts = round(pts)
        trb = round(trb)
        ast = round(ast)

        st.subheader(f"📊 Prediction for {selected_player} vs {selected_opponent_full}")
        st.markdown("### 🔴 **Points**")
        st.slider("Predicted Points", min_value=0, max_value=60, value=int(pts), disabled=True, key="points_slider")

        st.markdown("### 🔵 **Rebounds**")
        st.slider("Predicted Rebounds", min_value=0, max_value=30, value=int(trb), disabled=True, key="rebounds_slider")

        st.markdown("### 🟢 **Assists**")
        st.slider("Predicted Assists", min_value=0, max_value=30, value=int(ast), disabled=True, key="assists_slider")
