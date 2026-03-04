import pandas as pd
import json
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import pickle

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")
DATA_DIR = Path("data")

def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Charger les données de test
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")

    # Charger le modèle entraîné
    with open(MODELS_DIR / "trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Faire les prédictions
    predictions = model.predict(X_test)

    # Sauvegarder les prédictions
    pred_df = pd.DataFrame({
        "y_test": y_test.values.ravel(),
        "y_pred": predictions
    })
    pred_df.to_csv(DATA_DIR / "predictions.csv", index=False)

    # Calculer les métriques
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    scores = {
        "mse": mse,
        "r2": r2
    }

    # Sauvegarder les métriques
    with open(METRICS_DIR / "scores.json", "w") as f:
        json.dump(scores, f, indent=4)

if __name__ == "__main__":
    main()