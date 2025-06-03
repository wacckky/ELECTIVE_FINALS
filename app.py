import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load the model (cached) ===
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()

# === Load data ===
@st.cache_data
def load_data():
    df = pd.read_csv("MAINDATA.csv", encoding="ISO-8859-1")
    df = df.drop_duplicates()
    df['Player'] = df['Player'].str.strip().str.title()
    df['Tm'] = df['Tm'].str.strip().str.upper()

    cols_to_numeric = ['MP', 'PTS', 'TRB', 'AST', 'TOV']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(df.mean(numeric_only=True), inplace=True)

    if {'PTS', 'TRB', 'AST', 'TOV'}.issubset(df.columns):
        df['EFFICIENCY'] = df['PTS'] + df['TRB'] + df['AST'] - df['TOV']

    return df

df = load_data()

# === UI ===
st.title("🏀 NBA Player Performance Predictor")
st.write("Predict a player's average **points**, **rebounds**, and **assists** vs a specific opponent.")

player_name = st.text_input("Enter Player Name:")
opponent_team = st.text_input("Enter Opponent Team (3-letter code, e.g. LAL, BOS):")

if st.button("Predict"):
    player = player_name.strip().title()
    opp = opponent_team.strip().upper()
    
    player_data = df[(df['Player'] == player) & (df['Opp'] == opp)]

    if player_data.empty:
        st.warning(f"No data found for {player} vs {opp}")
        player_only = df[df['Player'] == player]
        if not player_only.empty:
            st.info(f"But this player has played vs: {', '.join(player_only['Opp'].unique())}")
    else:
        numeric_df = df.select_dtypes(include=[np.number])
        X_columns = numeric_df.drop(columns=['PTS', 'TRB', 'AST']).columns
        avg_features = player_data[X_columns].mean().to_frame().T

        prediction = model.predict(avg_features)
        predicted_pts, predicted_trb, predicted_ast = prediction[0]

        st.subheader(f"📊 Prediction for {player} vs {opp}")
        st.write(f"**Points**: {predicted_pts:.2f}")
        st.write(f"**Rebounds**: {predicted_trb:.2f}")
        st.write(f"**Assists**: {predicted_ast:.2f}")
