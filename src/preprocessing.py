"""Pipeline de pré-processamento.

Todas as etapas são transformadores do scikit-learn dentro de um ColumnTransformer.
Como o pré-processador é o primeiro passo de um Pipeline, as estatísticas (medianas,
médias, desvios, categorias) são aprendidas apenas nos dados em que o Pipeline é
ajustado: o conjunto de treino ou, na validação cruzada, as dobras de treino.
Isso impede vazamento de dados.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor(add_missing_indicator: bool = False) -> ColumnTransformer:
    """Constrói o pré-processador (ainda não ajustado).

    - RestingBP e Cholesterol: o valor 0 é tratado como ausente e imputado pela
      mediana do treino; depois, padronização.
    - Age, MaxHR e Oldpeak: imputação pela mediana (proteção para entradas
      incompletas) e padronização.
    - FastingBS: já é 0/1, passa direto.
    - Categóricas: one-hot; binárias viram uma única coluna; categoria nunca
      vista no treino vira vetor de zeros em vez de erro.

    `add_missing_indicator=True` acrescenta colunas indicadoras de ausência
    (usado apenas no experimento de sensibilidade descrito no artigo).
    """
    zero_as_missing = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    missing_values=0, strategy="median", add_indicator=add_missing_indicator
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(
        handle_unknown="ignore", drop="if_binary", sparse_output=False
    )
    return ColumnTransformer(
        transformers=[
            ("num_zero", zero_as_missing, config.NUM_ZERO_AS_MISSING),
            ("num", numeric, config.NUM_FEATURES),
            ("bin", "passthrough", config.BIN_FEATURES),
            ("cat", categorical, config.CAT_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, add_missing_indicator: bool = False) -> Pipeline:
    """Pré-processamento + estimador em um único objeto serializável."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(add_missing_indicator)),
            ("model", estimator),
        ]
    )
