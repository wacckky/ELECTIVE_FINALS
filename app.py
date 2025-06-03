import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data(show_spinner=True)
def load_and_process_data(uploaded_file):
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

    # Drop duplicates
    df = df.drop_duplicates()

    # Clean text columns
    df['Player'] = df['Player'].str.strip().str.title()
    df['Tm'] = df['Tm'].str.strip().str.upper()

    # Convert relevant columns to numeric
    cols_to_numeric = ['MP', 'PTS', 'TRB', 'AST', 'TOV']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing numeric values with column means
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Create Efficiency feature if columns exist
    if {'PTS', 'TRB', 'AST', 'TOV'}.issubset(df.columns):
        df['EFFICIENCY'] = df['PTS'] + df['TRB'] + df['AST'] - df['TOV']

    return df

@st.cache_data(show_spinner=True)
def train_models(df):
    numeric_df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

    y = numeric_df[['PTS', 'TRB', 'AST']]
    X = numeric_df.drop(['PTS', 'TRB', 'AST'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = MultiOutputRegressor(XGBRegressor(random_state=42, n_estimators=100))
    xgb_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    return xgb_model, rf_model, X, y, X_test, y_test

def predict_player_performance_avg(df, X_cols, model, player_name, opponent_team):
    player_name = player_name.strip().title()
    opponent_team = opponent_team.strip().upper()

    player_data = df[(df['Player'] == player_name) & (df['Opp'] == opponent_team)]

    if player_data.empty:
        player_only = df[df['Player'] == player_name]
        if player_only.empty:
            return None, f"❌ Player '{player_name}' not found in dataset."
        else:
            teams = player_only['Opp'].unique()
            return None, (f"⚠️ No games found for '{player_name}' against '{opponent_team}'.\n"
                          f"Player has played against these teams: {', '.join(teams)}")
    
    avg_features = player_data[X_cols].mean().to_frame().T
    prediction = model.predict(avg_features)
    predicted_pts, predicted_trb, predicted_ast = prediction[0]

    return (predicted_pts, predicted_trb, predicted_ast), None


def main():
    st.title("🏀 NBA Player Performance Predictor")

    uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])
    if uploaded_file:
        df = load_and_process_data(uploaded_file)
        st.success("Data loaded and processed!")

        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Train models
        with st.spinner("Training models..."):
            xgb_model, rf_model, X, y, X_test, y_test = train_models(df)
        st.success("Models trained!")

        st.subheader("Model Evaluation (XGBoost)")
        xgb_pred = xgb_model.predict(X_test)
        for i, target in enumerate(['PTS', 'TRB', 'AST']):
            mse = mean_squared_error(y_test.iloc[:, i], xgb_pred[:, i])
            r2 = r2_score(y_test.iloc[:, i], xgb_pred[:, i])
            st.write(f"{target} - RMSE: {mse**0.5:.2f}, R²: {r2:.2f}")

        st.subheader("Make a Prediction")
        player_name = st.text_input("Enter Player Name")
        opponent_team = st.text_input("Enter Opponent Team (Abbreviation, e.g., LAL, BOS)")

        model_choice = st.radio("Select Model", ("XGBoost", "Random Forest"))

        if st.button("Predict"):
            if not player_name or not opponent_team:
                st.error("Please enter both player name and opponent team.")
            else:
                model = xgb_model if model_choice == "XGBoost" else rf_model
                prediction, msg = predict_player_performance_avg(df, X.columns, model, player_name, opponent_team)
                if msg:
                    st.warning(msg)
                else:
                    pts, trb, ast = prediction
                    st.write(f"### Predicted Stats for {player_name} vs {opponent_team}:")
                    st.write(f"- Points: {pts:.2f}")
                    st.write(f"- Rebounds: {trb:.2f}")
                    st.write(f"- Assists: {ast:.2f}")

        st.subheader("Correlation Heatmap")
        numeric_df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        st.pyplot()

        st.subheader("Points Distribution")
        plt.figure(figsize=(8, 6))
        sns.histplot(numeric_df['PTS'], bins=20, kde=True)
        st.pyplot()

        st.subheader("Minutes Played vs Points")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='MP', y='PTS', data=numeric_df)
        st.pyplot()

if __name__ == "__main__":
    main()
