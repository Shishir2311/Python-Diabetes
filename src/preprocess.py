import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv("../data/diabetes.csv")
    return df

def preprocess_data(df):

    # Add family history if missing (genetic proxy)
    if "family_history" not in df.columns:
        df["family_history"] = ((df["Age"] > 45) & (df["BMI"] > 28)).astype(int)

    # Remove missing values
    df = df.dropna()

    # Separate features and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
