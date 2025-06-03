import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load the model and scaler ===
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
    
    # Convert numeric columns
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

# Prepare columns used for training
categorical_cols = ['Player', 'Tm', 'Opp']
numeric_cols = ['MP', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA',
                'ORB', 'DRB', 'TOV', 'PF', 'EFFICIENCY', 'Res']

# Get training feature columns to align inputs later
@st.cache_data
def get_training_columns():
    df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
    feature_cols = list(df_encoded.columns) + numeric_cols
    return feature_cols

training_columns = get_training_columns()
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://raw.githubusercontent.com/wacckky/ELECTIVE_FINALS/26ac0f1d8e3a74ed1cab850abda7273503534215/background.jpg
");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
""", unsafe_allow_html=True)

# === UI ===
st.title("NBA Player Performance Predictor")
st.write("Select a team, player, and opponent to predict stats.")

teams = sorted(df['Tm'].unique())
selected_team = st.selectbox("Select Player's Team:", teams)

players_from_team = sorted(df[df['Tm'] == selected_team]['Player'].unique())
selected_player = st.selectbox("Select Player:", players_from_team)

opponents = sorted(df['Opp'].unique())
selected_opponent = st.selectbox("Select Opponent Team:", opponents)

if st.button("Predict Performance"):

    # Filter data for the player/opponent combo
    player_data = df[(df['Player'] == selected_player) & (df['Opp'] == selected_opponent)]

    if player_data.empty:
        st.warning(f"No game data for {selected_player} vs {selected_opponent}")
        player_only = df[df['Player'] == selected_player]
        if not player_only.empty:
            st.info(f"This player has played vs: {', '.join(player_only['Opp'].unique())}")
    else:
        # Get average stats for numeric and categorical features for that combo
        avg_numeric = player_data[numeric_cols].mean().to_frame().T
        avg_cat = player_data[categorical_cols].iloc[0:1]  # Just one row for categorical

        # One-hot encode the categorical features
        cat_encoded = pd.get_dummies(avg_cat, drop_first=True)

        # Combine numeric and encoded categorical features
        X_input = pd.concat([cat_encoded.reset_index(drop=True), avg_numeric.reset_index(drop=True)], axis=1)

        # Align columns with training data - add missing columns with 0
        X_input = X_input.reindex(columns=training_columns, fill_value=0)

        # Scale input
        X_scaled = scaler.transform(X_input)

        # Predict
        prediction = model.predict(X_scaled)
        pts, trb, ast = prediction[0]

        # Round results to whole numbers
        pts = round(pts)
        trb = round(trb)
        ast = round(ast)
        import streamlit.components.v1 as components

        st.subheader(f"📊 Prediction for {selected_player} vs {selected_opponent}")

        # Styled labels and disabled sliders
        st.markdown("### 🔴 **Points**")
        st.slider("Predicted Points", min_value=0, max_value=60, value=int(pts), disabled=True, key="points_slider")

        st.markdown("### 🔵 **Rebounds**")
        st.slider("Predicted Rebounds", min_value=0, max_value=30, value=int(trb), disabled=True, key="rebounds_slider")

        st.markdown("### 🟢 **Assists**")
        st.slider("Predicted Assists", min_value=0, max_value=30, value=int(ast), disabled=True, key="assists_slider")


