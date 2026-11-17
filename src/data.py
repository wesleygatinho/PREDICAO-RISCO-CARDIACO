"""Leitura do conjunto de dados e divisão treino/teste."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_data(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Lê o CSV original e confere se o esquema é o esperado."""
    df = pd.read_csv(path)
    expected = config.FEATURES + [config.TARGET]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em {path}: {missing}")
    return df[expected]


def split_data(df: pd.DataFrame):
    """Hold-out estratificado com semente fixa.

    O conjunto de teste é separado aqui, uma única vez, e só volta a ser usado
    na avaliação final. Nenhuma estatística de pré-processamento, nenhum
    hiperparâmetro e nenhum limiar de decisão é ajustado com ele.
    """
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED
    )
