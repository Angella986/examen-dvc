import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_PATH = Path("data/raw/raw.csv")
PROCESSED_DIR = Path("data/processed")

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    # La cible est la dernière colonne
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

if __name__ == "__main__":
    main()