import pandas as pd
from sklearn.linear_model import Ridge
from pathlib import Path
import pickle

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Charger les données normalisées
    X_train = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")

    # Charger les meilleurs paramètres
    with open(MODELS_DIR / "best_params.pkl", "rb") as f:
        best_params = pickle.load(f)

    # Créer le modèle final avec les meilleurs paramètres
    model = Ridge(**best_params)

    # Entraîner le modèle
    model.fit(X_train, y_train.values.ravel())

    # Sauvegarder le modèle entraîné
    with open(MODELS_DIR / "trained_model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()