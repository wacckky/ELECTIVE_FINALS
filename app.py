import streamlit as st
import pandas as pd
import joblib

# Load model and data
xgb_model = joblib.load("xgb_model.pkl")
df = pd.read_csv("MAINDATA.csv", encoding="ISO-8859-1")

# Clean and preprocess
df['Player'] = df['Player'].str.strip().str.title()
df['Tm'] = df['Tm'].str.strip().str.upper()
df['Opp'] = df['Opp'].str.strip().str.upper()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

# Sidebar selections
teams = sorted(df['Tm'].unique())
selected_team = st.sidebar.selectbox("Select Team", teams)

players = sorted(df[df['Tm'] == selected_team]['Player'].unique())
selected_player = st.sidebar.selectbox("Select Player", players)

opponent_teams = sorted(df['Opp'].unique())
selected_opponent = st.sidebar.selectbox("Select Opponent Team", opponent_teams)

st.title("🏀 NBA Player Performance Predictor")

if st.button("Predict Stats"):
    player_data = df[(df['Player'] == selected_player) & (df['Opp'] == selected_opponent)]

    if player_data.empty:
        recent_games = df[df['Player'] == selected_player].sort_values(by='Date', ascending=False).head(5)
        if recent_games.empty:
            st.error(f"❌ No data found for player: {selected_player}")
        else:
            st.warning(f"⚠️ No games found for {selected_player} vs {selected_opponent}. Using last 5 games instead.")
            avg_input = recent_games.drop(columns=['PTS', 'TRB', 'AST']).mean().to_frame().T
            prediction = xgb_model.predict(avg_input)
    else:
        avg_input = player_data.drop(columns=['PTS', 'TRB', 'AST']).mean().to_frame().T
        prediction = xgb_model.predict(avg_input)

    predicted_pts = round(prediction[0][0])
    predicted_trb = round(prediction[0][1])
    predicted_ast = round(prediction[0][2])

    st.subheader(f"📊 Predicted Stats for {selected_player}")
    st.write(f"- Points: {predicted_pts}")
    st.write(f"- Rebounds: {predicted_trb}")
    st.write(f"- Assists: {predicted_ast}")
