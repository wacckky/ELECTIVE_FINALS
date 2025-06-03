import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load the model ===
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()

# === Load and process data ===
@st.cache_data
def load_data():
    df = pd.read_csv("MAINDATA.csv", encoding="ISO-8859-1")
    df = df.drop_duplicates()
    df['Player'] = df['Player'].str.strip().str.title()
    df['Tm'] = df['Tm'].str.strip().str.upper()
    df['Opp'] = df['Opp'].str.strip().str.upper()

    cols_to_numeric = ['MP', 'PTS', 'TRB', 'AST', 'TOV']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(df.mean(numeric_only=True), inplace=True)

    if {'PTS', 'TRB', 'AST', 'TOV'}.issubset(df.columns):
        df['EFFICIENCY'] = df['PTS'] + df['TRB'] + df['AST'] - df['TOV']

    return df

df = load_data()

# === UI Layout ===
st.title("🏀 NBA Player Performance Predictor")
st.write("Use the dropdowns below to select a **team**, a **player**, and an **opponent team** to predict performance.")

# Dropdown: Select Team
teams = sorted(df['Tm'].unique())
selected_team = st.selectbox("Select Player's Team:", teams)

# Dropdown: Select Player from Team
players_from_team = sorted(df[df['Tm'] == selected_team]['Player'].unique())
selected_player = st.selectbox("Select Player:", players_from_team)

# Dropdown: Select Opponent
opponents = sorted(df['Opp'].unique())
selected_opponent = st.selectbox("Select Opponent Team:", opponents)

# Prediction Trigger
if st.button("Predict Performance"):
    player_data = df[(df['Player'] == selected_player) & (df['Opp'] == selected_opponent)]

    if player_data.empty:
        st.warning(f"No game data for {selected_player} vs {selected_opponent}")
        player_only = df[df['Player'] == selected_player]
        if not player_only.empty:
            st.info(f"This player has played vs: {', '.join(player_only['Opp'].unique())}")
    else:
        numeric_df = df.select_dtypes(include=[np.number])
        X_columns = numeric_df.drop(columns=['PTS', 'TRB', 'AST']).columns
        avg_features = player_data[X_columns].mean().to_frame().T

        prediction = model.predict(avg_features)
        predicted_pts, predicted_trb, predicted_ast = prediction[0]

        st.subheader(f"📊 Prediction for {selected_player} vs {selected_opponent}")
        st.write(f"**Points**: {predicted_pts:.2f}")
        st.write(f"**Rebounds**: {predicted_trb:.2f}")
        st.write(f"**Assists**: {predicted_ast:.2f}")
