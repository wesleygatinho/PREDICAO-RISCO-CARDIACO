"""Configuração central do projeto: caminhos, semente, colunas e dicionário de dados.

Tudo o que precisa ser igual no treino, na avaliação, na aplicação e nos testes
fica aqui, para existir uma única fonte de verdade.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- caminhos
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = ROOT / "logs"

MODEL_PATH = MODELS_DIR / "modelo_final.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
BACKGROUND_PATH = MODELS_DIR / "shap_background.csv"

# --------------------------------------------------------------------------- protocolo
SEED = 42
TEST_SIZE = 0.20          # hold-out estratificado, separado antes de qualquer análise de modelo
CV_SPLITS = 5             # validação cruzada estratificada no conjunto de treino
CV_REPEATS = 3            # repetida 3 vezes -> 15 estimativas por configuração
SELECTION_METRIC = "roc_auc"   # métrica usada para escolher hiperparâmetros e o modelo final
THRESHOLD_BETA = 2.0      # limiar de decisão escolhido maximizando F-beta (beta=2 prioriza revocação)

# --------------------------------------------------------------------------- colunas
TARGET = "HeartDisease"

# Numéricas em que o valor 0 é um código de "não medido" (fisiologicamente impossível).
NUM_ZERO_AS_MISSING = ["RestingBP", "Cholesterol"]
# Demais numéricas contínuas/discretas.
NUM_FEATURES = ["Age", "MaxHR", "Oldpeak"]
# Binária já codificada como 0/1.
BIN_FEATURES = ["FastingBS"]
# Categóricas nominais (texto).
CAT_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

# Ordem original das colunas no CSV.
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]
NUMERIC_ALL = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# --------------------------------------------------------------------------- dicionário de dados
# Usado pela AED, pela interface (rótulos e ajuda) e pela validação de entradas.
CATEGORIES = {
    "Sex": {"M": "Masculino", "F": "Feminino"},
    "ChestPainType": {
        "ASY": "Assintomático",
        "ATA": "Angina atípica",
        "NAP": "Dor não anginosa",
        "TA": "Angina típica",
    },
    "RestingECG": {
        "Normal": "Normal",
        "ST": "Anormalidade de onda ST-T",
        "LVH": "Hipertrofia ventricular esquerda",
    },
    "ExerciseAngina": {"N": "Não", "Y": "Sim"},
    "ST_Slope": {"Up": "Ascendente", "Flat": "Plano", "Down": "Descendente"},
}

LABELS_PT = {
    "Age": "Idade",
    "Sex": "Sexo",
    "ChestPainType": "Tipo de dor no peito",
    "RestingBP": "Pressão arterial em repouso",
    "Cholesterol": "Colesterol sérico",
    "FastingBS": "Glicemia de jejum > 120 mg/dl",
    "RestingECG": "ECG em repouso",
    "MaxHR": "Frequência cardíaca máxima",
    "ExerciseAngina": "Angina induzida por exercício",
    "Oldpeak": "Depressão de ST (oldpeak)",
    "ST_Slope": "Inclinação do segmento ST",
    "HeartDisease": "Doença cardíaca",
}

UNITS = {
    "Age": "anos",
    "RestingBP": "mm Hg",
    "Cholesterol": "mg/dl",
    "MaxHR": "bpm",
    "Oldpeak": "mm",
}

# Faixas aceitas pela aplicação. São limites de plausibilidade clínica um pouco mais
# largos do que o intervalo observado no conjunto de dados (ver reports/eda_resumo.json).
VALID_RANGES = {
    "Age": (18, 100),
    "RestingBP": (60, 260),
    "Cholesterol": (50, 700),
    "MaxHR": (40, 230),
    "Oldpeak": (-4.0, 8.0),
}

CLASS_NAMES = {0: "Sem doença cardíaca", 1: "Com doença cardíaca"}
