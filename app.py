import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

st.title("🏀 NBA Player Performance Predictor")

# Load the dataset
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

    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    numeric_df = numeric_df.dropna(axis=1, how='all')

    return df, numeric_df

df, numeric_df = load_data()

# Prepare features and target
y = numeric_df[['PTS', 'TRB', 'AST']]
X = numeric_df.drop(['PTS', 'TRB', 'AST'], axis=1)

# Train the XGBoost model
@st.cache_resource
def train_model(X, y):
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultiOutputRegressor(XGBRegressor(random_state=42, n_estimators=100))
    model.fit(X_train, y_train)
    return model

model = train_model(X, y)

# UI Inputs
player_name = st.text_input("Enter player name:")
opponent_team = st.text_input("Enter opponent team:")

if st.button("Predict Performance"):
    player_name = player_name.strip().title()
    opponent_team = opponent_team.strip().upper()
    player_data = df[(df['Player'] == player_name) & (df['Opp'] == opponent_team)]

    if player_data.empty:
        player_only = df[df['Player'] == player_name]
        if player_only.empty:
            st.error(f"❌ Player '{player_name}' not found.")
        else:
            st.warning(f"⚠️ No games found for '{player_name}' against '{opponent_team}'.")
            st.write("Teams this player has played against:")
            st.write(player_only['Opp'].unique())
    else:
        avg_features = player_data[X.columns].mean().to_frame().T
        prediction = model.predict(avg_features)[0]
        pts, trb, ast = prediction

        st.subheader(f"📊 Predicted Averages vs {opponent_team}")
        st.write(f"- Points: `{pts:.2f}`")
        st.write(f"- Rebounds: `{trb:.2f}`")
        st.write(f"- Assists: `{ast:.2f}`")
