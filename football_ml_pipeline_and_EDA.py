# football_ml_pipeline.py

import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_DIR = "dataset"
SEASONS = ["18_19", "19_20", "20_21", "21_22", "22_23", "23_24"]
LEAGUES = ["E0", "F1", "D1", "I1", "SP1"]  # EPL, Ligue 1, Bundesliga, Serie A, La Liga

# --- Load Data ---
def load_all_data():
    all_data = []
    for season in SEASONS:
        for league in LEAGUES:
            path = os.path.join(DATA_DIR, f"{league}_{season}.xlsx")
            if os.path.exists(path):
                df = pd.read_excel(path)
                df['Season'] = season
                df['League'] = league
                all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# --- Feature Engineering ---
def engineer_features(df):
    df = df.dropna(subset=['FTR', 'HomeTeam', 'AwayTeam'])
    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A', 'Season', 'League']]

    # Implied Probabilities
    df['Imp_H'] = 1 / df['B365H']
    df['Imp_D'] = 1 / df['B365D']
    df['Imp_A'] = 1 / df['B365A']
    
    # Normalize to sum to 1
    total = df[['Imp_H', 'Imp_D', 'Imp_A']].sum(axis=1)
    df['Imp_H'] /= total
    df['Imp_D'] /= total
    df['Imp_A'] /= total

    # Encode result
    df['Result'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

    # Encode teams
    le = LabelEncoder()
    df['HomeTeam'] = le.fit_transform(df['HomeTeam'])
    df['AwayTeam'] = le.fit_transform(df['AwayTeam'])

    return df.dropna()

# --- Exploratory Data Analysis ---
def perform_eda(df):
    print("\n--- EDA ---")
    print(df.describe())

    # Histograms
    df['Result'].value_counts().plot(kind='bar', title='Match Outcome Distribution')
    plt.xlabel("Result (0=Home, 1=Draw, 2=Away)")
    plt.ylabel("Count")
    plt.show()

    # Scatter plot of odds vs. goals
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='B365H', y='FTHG', data=df)
    plt.title('Home Odds vs Full Time Home Goals')
    plt.xlabel('Home Odds')
    plt.ylabel('Home Goals')
    plt.show()

    # Correlation matrix
    plt.figure(figsize=(10, 6))
    corr = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Imp_H', 'Imp_D', 'Imp_A', 'Result']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.show()

# --- Split Train/Test ---
def split_data(df):
    train_df = df[df['Season'] < '2324']
    test_df = df[df['Season'] == '2324']
    X_train = train_df.drop(columns=['FTR', 'FTHG', 'FTAG', 'Result'])
    y_train = train_df['Result']
    X_test = test_df.drop(columns=['FTR', 'FTHG', 'FTAG', 'Result'])
    y_test = test_df['Result']
    return X_train, X_test, y_train, y_test

# --- Model Training and Evaluation ---
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf')
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        print(f"\n{name}")
        print(f"Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
        print(classification_report(y_test, preds))

# --- Main Pipeline ---
def main():
    df = load_all_data()
    df = engineer_features(df)
    perform_eda(df)
    X_train, X_test, y_train, y_test = split_data(df)
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
