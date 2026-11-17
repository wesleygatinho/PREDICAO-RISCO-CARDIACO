"""Testes do pipeline de dados, do pré-processamento e do modelo salvo."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import config
from src.data import load_data, split_data
from src.preprocessing import build_pipeline, build_preprocessor
from src.train import choose_threshold, corrected_resampled_ttest, metrics_at


@pytest.fixture(scope="module")
def df():
    return load_data()


@pytest.fixture(scope="module")
def splits(df):
    return split_data(df)


# --------------------------------------------------------------------------- dados
def test_esquema_do_conjunto_de_dados(df):
    assert df.shape == (918, 12)
    assert list(df.columns) == config.FEATURES + [config.TARGET]
    assert set(df[config.TARGET].unique()) == {0, 1}
    for col, options in config.CATEGORIES.items():
        assert set(df[col].unique()) <= set(options), col


def test_divisao_reprodutivel_estratificada_e_disjunta(df, splits):
    X_train, X_test, y_train, y_test = splits
    X_train2, X_test2, _, _ = split_data(df)
    assert X_train.index.equals(X_train2.index) and X_test.index.equals(X_test2.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_test) == len(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


# --------------------------------------------------------------------------- pré-processamento
def test_estatisticas_ajustadas_apenas_no_treino(df, splits):
    """As medianas de imputação devem vir do treino, não do conjunto completo."""
    X_train, X_test, _, _ = splits
    pre = build_preprocessor().fit(X_train)
    imputer = pre.named_transformers_["num_zero"].named_steps["imputer"]
    scaler = pre.named_transformers_["num"].named_steps["scaler"]

    chol_train = X_train.loc[X_train["Cholesterol"] > 0, "Cholesterol"]
    bp_train = X_train.loc[X_train["RestingBP"] > 0, "RestingBP"]
    assert imputer.statistics_ == pytest.approx([bp_train.median(), chol_train.median()])
    assert scaler.mean_ == pytest.approx(X_train[config.NUM_FEATURES].mean().to_numpy())
    # se o teste tivesse entrado no ajuste, a média seria a do conjunto completo
    assert not np.allclose(scaler.mean_, df[config.NUM_FEATURES].mean().to_numpy())


def test_zero_no_colesterol_e_imputado_pela_mediana(splits):
    X_train, _, y_train, _ = splits
    pipe = build_pipeline(LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    median = X_train.loc[X_train["Cholesterol"] > 0, "Cholesterol"].median()
    row = X_train.iloc[[0]].copy()
    row["Cholesterol"] = 0
    row_median = row.copy()
    row_median["Cholesterol"] = median
    assert pipe.predict_proba(row)[0, 1] == pytest.approx(pipe.predict_proba(row_median)[0, 1])


def test_saida_do_preprocessador_sem_nan_e_com_colunas_esperadas(splits):
    X_train, X_test, _, _ = splits
    pre = build_preprocessor().fit(X_train)
    Z = pre.transform(X_test)
    assert Z.shape == (len(X_test), 18)
    assert not np.isnan(Z).any()
    names = list(pre.get_feature_names_out())
    assert "Sex_M" in names and "ST_Slope_Flat" in names and "FastingBS" in names


def test_categoria_desconhecida_nao_quebra_o_pipeline(splits):
    X_train, _, y_train, _ = splits
    pipe = build_pipeline(LogisticRegression(max_iter=2000)).fit(X_train, y_train)
    row = X_train.iloc[[0]].copy()
    row["ChestPainType"] = "XYZ"
    p = pipe.predict_proba(row)[0, 1]
    assert 0.0 <= p <= 1.0


# --------------------------------------------------------------------------- funções de avaliação
def test_metricas_para_um_limiar():
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.3])
    m = metrics_at(y, p, 0.5)
    assert (m["vp"], m["fn"], m["fp"], m["vn"]) == (2, 1, 1, 2)
    assert m["precisao"] == pytest.approx(2 / 3)
    assert m["revocacao"] == pytest.approx(2 / 3)
    assert m["especificidade"] == pytest.approx(2 / 3)


def test_escolha_do_limiar_prioriza_revocacao():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.35 * y + rng.normal(0.35, 0.18, 500), 0, 1)
    t_f2, _ = choose_threshold(y, p, beta=2.0)
    t_f05, _ = choose_threshold(y, p, beta=0.5)
    assert 0.05 <= t_f2 <= 0.95
    assert t_f2 < t_f05  # beta maior -> limiar menor -> mais revocação


def test_teste_t_corrigido():
    a = np.array([0.91, 0.93, 0.92, 0.94, 0.90])
    t, p = corrected_resampled_ttest(a, a, 80, 20)
    assert (t, p) == (0.0, 1.0)
    t, p = corrected_resampled_ttest(a + np.array([0.05, 0.06, 0.05, 0.04, 0.05]), a, 80, 20)
    assert t > 0 and 0 < p < 0.05


# --------------------------------------------------------------------------- modelo salvo
@pytest.fixture(scope="module")
def saved():
    if not config.MODEL_PATH.exists():
        pytest.skip("modelo não treinado; execute: python -m src.train")
    with open(config.METADATA_PATH, encoding="utf-8") as f:
        return joblib.load(config.MODEL_PATH), json.load(f)


def test_modelo_salvo_contem_pre_processamento_e_classificador(saved):
    pipeline, meta = saved
    assert list(pipeline.named_steps) == ["preprocess", "model"]
    assert meta["atributos"] == config.FEATURES
    assert 0.05 <= meta["limiar"] <= 0.95


def test_modelo_salvo_reproduz_as_metricas_registradas(saved, splits):
    pipeline, meta = saved
    _, X_test, _, y_test = splits
    proba = pipeline.predict_proba(X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert roc_auc_score(y_test, proba) == pytest.approx(
        meta["metricas_teste_limiar_escolhido"]["auc_roc"], abs=1e-9
    )
    m = metrics_at(y_test, proba, meta["limiar"])
    assert m["revocacao"] == pytest.approx(meta["metricas_teste_limiar_escolhido"]["revocacao"])


def test_modelo_supera_o_baseline_trivial(saved, splits):
    pipeline, _ = saved
    _, X_test, _, y_test = splits
    assert roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]) > 0.85


def test_amostra_de_referencia_do_shap_vem_do_treino(splits):
    if not config.BACKGROUND_PATH.exists():
        pytest.skip("modelo não treinado")
    X_train, X_test, _, _ = splits
    background = pd.read_csv(config.BACKGROUND_PATH)
    assert list(background.columns) == config.FEATURES
    train_rows = set(map(tuple, X_train.astype(str).to_numpy()))
    assert all(tuple(r) in train_rows for r in background.astype(str).to_numpy())
