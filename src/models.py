"""Modelos candidatos e grades de hiperparâmetros.

Um baseline trivial (classe majoritária), um baseline simples (regressão
logística) e cinco modelos de famílias diferentes. As grades abaixo são
exatamente as percorridas pela busca exaustiva (GridSearchCV) em src/train.py.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .config import SEED


def get_candidates() -> dict[str, dict]:
    """Retorna {nome: {"estimator", "grid", "family", "label"}}."""
    return {
        "dummy": {
            "label": "Classe majoritária",
            "family": "Baseline trivial",
            "estimator": DummyClassifier(strategy="most_frequent"),
            "grid": {},
        },
        "logreg": {
            "label": "Regressão logística",
            "family": "Baseline linear",
            "estimator": LogisticRegression(max_iter=5000, random_state=SEED),
            "grid": {
                "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100],
                "model__class_weight": [None, "balanced"],
            },
        },
        "knn": {
            "label": "k-NN",
            "family": "Baseado em instâncias",
            "estimator": KNeighborsClassifier(),
            "grid": {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31, 41],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        },
        "svm": {
            "label": "SVM (RBF)",
            "family": "Máquina de vetores de suporte",
            "estimator": SVC(kernel="rbf", probability=True, random_state=SEED),
            "grid": {
                "model__C": [0.1, 0.3, 1, 3, 10, 30, 100],
                "model__gamma": ["scale", 0.003, 0.01, 0.03, 0.1, 0.3],
                "model__class_weight": [None, "balanced"],
            },
        },
        "rf": {
            "label": "Random Forest",
            "family": "Ensemble (bagging)",
            "estimator": RandomForestClassifier(random_state=SEED, n_jobs=1),
            "grid": {
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 4, 6, 8, 12],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", 0.5],
            },
        },
        "gb": {
            "label": "Gradient Boosting",
            "family": "Ensemble (boosting)",
            "estimator": GradientBoostingClassifier(random_state=SEED),
            "grid": {
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 1.0],
            },
        },
        "mlp": {
            "label": "MLP",
            "family": "Rede neural",
            "estimator": MLPClassifier(max_iter=3000, random_state=SEED),
            "grid": {
                "model__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
                "model__alpha": [1e-4, 1e-2, 1e-1, 1.0],
                "model__learning_rate_init": [1e-3, 1e-2],
            },
        },
    }
