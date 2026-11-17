"""Testes da aplicação: validação de entradas, serviço de predição e interface."""
from __future__ import annotations

import pandas as pd
import pytest

from app.validation import ValidationError, record_to_frame, validate_dataframe, validate_record
from src import config

VALID = {
    "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 239,
    "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 160, "ExerciseAngina": "N",
    "Oldpeak": 1.2, "ST_Slope": "Flat",
}


# --------------------------------------------------------------------------- validação
def test_registro_valido_passa_sem_erros():
    clean, errors = validate_record(VALID)
    assert errors == []
    assert clean == VALID
    assert list(record_to_frame(clean).columns) == config.FEATURES


@pytest.mark.parametrize(
    "campo, valor, trecho",
    [
        ("Age", 150, "fora da faixa"),
        ("Age", -3, "fora da faixa"),
        ("Age", None, "obrigatório"),
        ("Age", "abc", "não é um número válido"),
        ("Age", 54.5, "número inteiro"),
        ("RestingBP", 0, "fora da faixa"),
        ("MaxHR", 500, "fora da faixa"),
        ("Oldpeak", 20, "fora da faixa"),
        ("Cholesterol", 5000, "fora da faixa"),
        ("Sex", "X", "opções aceitas"),
        ("ChestPainType", None, "obrigatório"),
        ("FastingBS", None, "obrigatório"),
        ("ST_Slope", "Sideways", "opções aceitas"),
        ("FastingBS", 2, "0 (não) ou 1 (sim)"),
    ],
)
def test_entradas_invalidas_geram_mensagem_clara(campo, valor, trecho):
    _, errors = validate_record({**VALID, campo: valor})
    assert len(errors) == 1
    assert errors[0].startswith(config.LABELS_PT[campo])
    assert trecho in errors[0]


def test_varios_erros_sao_reportados_juntos():
    _, errors = validate_record({**VALID, "Age": 200, "Sex": "?", "MaxHR": None})
    assert len(errors) == 3


def test_colesterol_nao_medido_e_aceito_como_zero():
    for value in (0, None, ""):
        clean, errors = validate_record({**VALID, "Cholesterol": value})
        assert errors == [] and clean["Cholesterol"] == 0


def test_virgula_decimal_e_caixa_das_categorias_sao_normalizadas():
    clean, errors = validate_record({**VALID, "Oldpeak": "1,5", "Sex": "m", "ST_Slope": "flat"})
    assert errors == []
    assert clean["Oldpeak"] == 1.5 and clean["Sex"] == "M" and clean["ST_Slope"] == "Flat"


def test_arquivo_sem_colunas_obrigatorias_e_recusado():
    with pytest.raises(ValidationError) as exc:
        validate_dataframe(pd.DataFrame({"Age": [50], "Sex": ["M"]}))
    assert "colunas obrigatórias" in exc.value.messages[0]


def test_arquivo_vazio_e_recusado():
    with pytest.raises(ValidationError):
        validate_dataframe(pd.DataFrame(columns=config.FEATURES))


def test_arquivo_com_linhas_validas_e_invalidas():
    df = pd.DataFrame([VALID, {**VALID, "Age": 300}, {**VALID, "Cholesterol": None}])
    valid, errors = validate_dataframe(df)
    assert len(valid) == 2
    assert list(valid.index) == [2, 4]          # número da linha no arquivo (cabeçalho = 1)
    assert len(errors) == 1 and errors[0].startswith("Linha 3:")


# --------------------------------------------------------------------------- serviço
@pytest.fixture(scope="module")
def model():
    if not config.MODEL_PATH.exists():
        pytest.skip("modelo não treinado; execute: python -m src.train")
    from app.service import RiskModel

    return RiskModel()


def test_servico_usa_o_pipeline_salvo(model):
    assert list(model.pipeline.named_steps) == ["preprocess", "model"]
    assert model.threshold == model.metadata["limiar"]


def test_predicao_individual(model):
    clean, _ = validate_record(VALID)
    pred = model.predict_one(record_to_frame(clean))
    assert 0.0 <= pred.probability <= 1.0
    assert pred.positive == (pred.probability >= model.threshold)
    assert pred.label in ("Risco elevado", "Risco baixo")


def test_perfis_extremos_sao_ordenados_como_esperado(model):
    alto = {**VALID, "Age": 65, "ChestPainType": "ASY", "ExerciseAngina": "Y", "Oldpeak": 2.5,
            "ST_Slope": "Flat", "MaxHR": 110, "FastingBS": 1}
    baixo = {**VALID, "Age": 35, "Sex": "F", "ChestPainType": "ATA", "ExerciseAngina": "N",
             "Oldpeak": 0.0, "ST_Slope": "Up", "MaxHR": 180}
    p_alto = model.predict_one(record_to_frame(alto)).probability
    p_baixo = model.predict_one(record_to_frame(baixo)).probability
    assert p_alto > 0.8 and p_baixo < 0.2


def test_explicacao_soma_ate_a_probabilidade(model):
    clean, _ = validate_record(VALID)
    row = record_to_frame(clean)
    contrib, base = model.explain(row)
    assert set(contrib["atributo"]) == set(config.FEATURES)
    total = base + contrib["contribuicao"].sum()
    assert total == pytest.approx(model.predict_one(row).probability, abs=1e-6)


def test_registro_de_predicao_e_feedback(model, tmp_path):
    from app.service import log_feedback, log_prediction

    clean, _ = validate_record(VALID)
    pred = model.predict_one(record_to_frame(clean))
    pred_log, fb_log = tmp_path / "predicoes.csv", tmp_path / "feedback.csv"
    log_prediction(pred, clean, "rf", path=pred_log)
    log_prediction(pred, clean, "rf", path=pred_log)
    log_feedback(pred.id, "Sim", "coerente", path=fb_log)

    logged = pd.read_csv(pred_log)
    assert len(logged) == 2
    assert set(config.FEATURES) <= set(logged.columns)
    assert logged.loc[0, "probabilidade"] == pytest.approx(pred.probability, abs=1e-6)
    fb = pd.read_csv(fb_log)
    assert fb.loc[0, "id_predicao"] == pred.id and fb.loc[0, "avaliacao"] == "Sim"


# --------------------------------------------------------------------------- interface
@pytest.fixture()
def app_test(model, tmp_path, monkeypatch):
    """Executa a interface em memória, com os registros apontando para uma pasta temporária."""
    from streamlit.testing.v1 import AppTest

    import app.service as service

    monkeypatch.setattr(service, "PREDICTIONS_LOG", tmp_path / "predicoes.csv")
    monkeypatch.setattr(service, "FEEDBACK_LOG", tmp_path / "feedback.csv")
    at = AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    return at.run()


def test_interface_abre_sem_excecao_e_mostra_limitacoes(app_test):
    assert not app_test.exception
    assert any("não substitui avaliação médica" in w.value for w in app_test.warning)


def test_interface_formulario_vazio_mostra_erros(app_test):
    at = app_test
    next(b for b in at.button if b.label == "Calcular risco").click().run()
    assert not at.exception
    assert any("Corrija os campos" in e.value for e in at.error)
    assert len(at.metric) == 0


def test_interface_formulario_valido_mostra_probabilidade(app_test, tmp_path):
    at = app_test
    numbers = {n.label: n for n in at.number_input}
    selects = {s.label: s for s in at.selectbox}
    numbers["Idade (anos)"].set_value(62)
    numbers["Pressão arterial em repouso (mm Hg)"].set_value(150)
    numbers["Colesterol sérico (mg/dl)"].set_value(260)
    numbers["Frequência cardíaca máxima no esforço (bpm)"].set_value(115)
    numbers["Depressão de ST no esforço — oldpeak (mm)"].set_value(2.0)
    selects["Sexo"].set_value("M")
    selects["Tipo de dor no peito"].set_value("ASY")
    selects["Glicemia de jejum acima de 120 mg/dl?"].set_value(1)
    selects["ECG em repouso"].set_value("Normal")
    selects["Angina induzida por exercício?"].set_value("Y")
    selects["Inclinação do segmento ST no pico do esforço"].set_value("Flat")
    next(b for b in at.button if b.label == "Calcular risco").click().run()

    assert not at.exception
    assert not any("Corrija os campos" in e.value for e in at.error)
    assert at.metric[0].label == "Probabilidade estimada de doença cardíaca"
    assert at.metric[0].value.endswith("%")
    assert any("Risco elevado" in e.value for e in at.error)
    assert (tmp_path / "predicoes.csv").exists()
