"""Camada de serviço da aplicação: carrega o modelo salvo, prediz, explica e registra.

Separada da interface para poder ser testada sem abrir o navegador.
"""
from __future__ import annotations

import csv
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src import config

PREDICTIONS_LOG = config.LOGS_DIR / "predicoes.csv"
FEEDBACK_LOG = config.LOGS_DIR / "feedback.csv"


class ModelNotFoundError(FileNotFoundError):
    pass


@dataclass
class Prediction:
    id: str
    probability: float      # P(doença cardíaca) estimada pelo modelo
    threshold: float        # limiar de decisão definido no treino
    positive: bool          # probability >= threshold
    label: str


class RiskModel:
    """Pipeline salvo + metadados. Não reimplementa nenhum pré-processamento."""

    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        metadata_path: Path = config.METADATA_PATH,
        background_path: Path = config.BACKGROUND_PATH,
    ):
        if not Path(model_path).exists() or not Path(metadata_path).exists():
            raise ModelNotFoundError(
                "Modelo não encontrado em models/. Gere-o com:  python -m src.train"
            )
        self.pipeline = joblib.load(model_path)
        with open(metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.threshold = float(self.metadata["limiar"])
        self._background_path = Path(background_path)
        self._explainer = None

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        proba = self.pipeline.predict_proba(X[config.FEATURES])[:, 1]
        return pd.Series(proba, index=X.index, name="probabilidade")

    def predict_one(self, row: pd.DataFrame) -> Prediction:
        p = float(self.predict_proba(row).iloc[0])
        positive = p >= self.threshold
        return Prediction(
            id=uuid.uuid4().hex[:12],
            probability=p,
            threshold=self.threshold,
            positive=positive,
            label="Risco elevado" if positive else "Risco baixo",
        )

    def explain(self, row: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Contribuição SHAP de cada atributo e o valor base (média de referência)."""
        if self._explainer is None:
            from src.explain import PipelineExplainer  # import tardio: shap demora a carregar

            background = pd.read_csv(self._background_path)
            self._explainer = PipelineExplainer(self.pipeline, background)
        return self._explainer.explain_one(row), self._explainer.base_value


# --------------------------------------------------------------------------- registros
def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            writer.writeheader()
        writer.writerow(row)


def log_prediction(pred: Prediction, record: dict, model_name: str, path: Path | None = None) -> None:
    """Registra a predição. Não há nome, documento ou outro identificador do paciente."""
    _append(
        path or PREDICTIONS_LOG,
        {
            "id": pred.id,
            "data_hora_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "modelo": model_name,
            **{c: record[c] for c in config.FEATURES},
            "probabilidade": round(pred.probability, 6),
            "limiar": pred.threshold,
            "classe_predita": int(pred.positive),
        },
    )


def log_feedback(prediction_id: str, useful: str, comment: str = "", path: Path | None = None) -> None:
    _append(
        path or FEEDBACK_LOG,
        {
            "id_predicao": prediction_id,
            "data_hora_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "avaliacao": useful,
            "comentario": comment.strip()[:500],
        },
    )
