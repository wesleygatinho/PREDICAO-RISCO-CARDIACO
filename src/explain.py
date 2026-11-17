"""Explicação das predições com valores SHAP no espaço dos atributos originais.

O modelo explicado é o Pipeline completo (pré-processamento + classificador),
visto como uma função dos 11 atributos que o usuário informa. Assim cada
atributo recebe uma única contribuição, em vez de uma por coluna one-hot.

O SHAP trabalha com matrizes numéricas; por isso as categóricas são convertidas
em códigos inteiros apenas para o mascaramento e reconvertidas em texto antes de
chamar o Pipeline. O pré-processamento aplicado continua sendo o do arquivo salvo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from . import config


class PipelineExplainer:
    """Calcula valores SHAP (explicador por permutação) para o Pipeline salvo."""

    def __init__(self, pipeline, background: pd.DataFrame, seed: int = config.SEED):
        self.pipeline = pipeline
        self.features = list(config.FEATURES)
        self._codes = {c: list(config.CATEGORIES[c].keys()) for c in config.CAT_FEATURES}
        self._background = self._encode(background[self.features])
        masker = shap.maskers.Independent(self._background, max_samples=len(self._background))
        self._explainer = shap.explainers.Permutation(self._predict, masker, seed=seed)
        self.base_value = float(self._predict(self._background).mean())

    # ------------------------------------------------------------------ codificação
    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(X), len(self.features)), dtype=float)
        for j, col in enumerate(self.features):
            if col in self._codes:
                out[:, j] = [self._codes[col].index(v) for v in X[col]]
            else:
                out[:, j] = X[col].astype(float).to_numpy()
        return out

    def _decode(self, A: np.ndarray) -> pd.DataFrame:
        data = {}
        for j, col in enumerate(self.features):
            if col in self._codes:
                data[col] = [self._codes[col][int(round(v))] for v in A[:, j]]
            else:
                data[col] = A[:, j]
        return pd.DataFrame(data, columns=self.features)

    def _predict(self, A: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(self._decode(np.asarray(A)))[:, 1]

    # ------------------------------------------------------------------ API pública
    def shap_values(self, X: pd.DataFrame, max_evals: int = 500) -> np.ndarray:
        """Matriz (n_amostras, 11) com a contribuição de cada atributo para P(doença)."""
        explanation = self._explainer(self._encode(X[self.features]), max_evals=max_evals, silent=True)
        return np.asarray(explanation.values)

    def explain_one(self, row: pd.DataFrame) -> pd.DataFrame:
        """Contribuições de um único paciente, ordenadas por magnitude."""
        values = self.shap_values(row)[0]
        out = pd.DataFrame(
            {
                "atributo": self.features,
                "rotulo": [config.LABELS_PT[c] for c in self.features],
                "valor": [row.iloc[0][c] for c in self.features],
                "contribuicao": values,
            }
        )
        return out.reindex(out["contribuicao"].abs().sort_values(ascending=False).index).reset_index(drop=True)
