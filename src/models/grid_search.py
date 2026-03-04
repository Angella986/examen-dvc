import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import pickle

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Charger les données normalisées
    X_train = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")

    # Définir le modèle
    model = Ridge()

    # Définir la grille d'hyperparamètres
    param_grid = {
        "alpha": [0.1, 1.0, 10.0, 50.0, 100.0]
    }

    # GridSearch
    grid = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train.values.ravel())

    # Sauvegarder les meilleurs paramètres
    best_params = grid.best_params_

    with open(MODELS_DIR / "best_params.pkl", "wb") as f:
        pickle.dump(best_params, f)

if __name__ == "__main__":
    main()