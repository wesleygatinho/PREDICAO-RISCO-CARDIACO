"""Validação das entradas da aplicação.

Estas funções só conferem e normalizam o que o usuário informou (tipos, faixas,
categorias). Nenhuma transformação de pré-processamento do modelo é feita aqui:
imputação, padronização e codificação vêm do Pipeline salvo em models/.
"""
from __future__ import annotations

import math

import pandas as pd

from src import config

INTEGER_FIELDS = ["Age", "RestingBP", "Cholesterol", "MaxHR"]
ALLOW_NOT_MEASURED = ["Cholesterol"]   # 0 = "não medido" (mesma convenção do conjunto de dados)


class ValidationError(ValueError):
    """Erro de validação com a lista de mensagens a exibir ao usuário."""

    def __init__(self, messages: list[str]):
        super().__init__("; ".join(messages))
        self.messages = messages


def _to_number(value):
    """Converte para float aceitando vírgula decimal; devolve None se não for número."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
        if value == "":
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def validate_record(record: dict) -> tuple[dict, list[str]]:
    """Valida um paciente. Retorna (registro normalizado, lista de erros)."""
    errors: list[str] = []
    clean: dict = {}

    for col, (lo, hi) in config.VALID_RANGES.items():
        label = config.LABELS_PT[col]
        unit = config.UNITS.get(col, "")
        raw = record.get(col)
        number = _to_number(raw)
        if number is None:
            if col in ALLOW_NOT_MEASURED and (raw is None or str(raw).strip() == ""):
                clean[col] = 0
                continue
            if raw is None or str(raw).strip() == "":
                errors.append(f"{label}: campo obrigatório não preenchido.")
            else:
                errors.append(f"{label}: '{raw}' não é um número válido.")
            continue
        if col in ALLOW_NOT_MEASURED and number == 0:
            clean[col] = 0
            continue
        if col in INTEGER_FIELDS and number != int(number):
            errors.append(f"{label}: informe um número inteiro (recebido {raw}).")
            continue
        if not lo <= number <= hi:
            errors.append(
                f"{label}: {number:g} {unit} está fora da faixa aceita ({lo:g} a {hi:g} {unit})."
            )
            continue
        clean[col] = int(number) if col in INTEGER_FIELDS else float(number)

    fasting = _to_number(record.get("FastingBS"))
    if record.get("FastingBS") is None or str(record.get("FastingBS")).strip() == "":
        errors.append(f"{config.LABELS_PT['FastingBS']}: campo obrigatório não preenchido.")
    elif fasting not in (0.0, 1.0):
        errors.append(
            f"{config.LABELS_PT['FastingBS']}: use 0 (não) ou 1 (sim); recebido '{record.get('FastingBS')}'."
        )
    else:
        clean["FastingBS"] = int(fasting)

    for col, options in config.CATEGORIES.items():
        raw = record.get(col)
        value = str(raw).strip() if raw is not None else ""
        match = next((k for k in options if k.lower() == value.lower()), None)
        if value == "":
            errors.append(f"{config.LABELS_PT[col]}: campo obrigatório não preenchido.")
        elif match is None:
            errors.append(
                f"{config.LABELS_PT[col]}: valor '{value}' inválido; opções aceitas: {', '.join(options)}."
            )
        else:
            clean[col] = match

    return clean, errors


def record_to_frame(clean: dict) -> pd.DataFrame:
    """Monta o DataFrame de uma linha, na ordem de colunas usada no treino."""
    return pd.DataFrame([clean], columns=config.FEATURES)


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Valida um arquivo com vários pacientes. Retorna (linhas válidas, erros)."""
    missing = [c for c in config.FEATURES if c not in df.columns]
    if missing:
        raise ValidationError(
            [f"O arquivo não tem as colunas obrigatórias: {', '.join(missing)}.",
             f"Colunas esperadas: {', '.join(config.FEATURES)}."]
        )
    if df.empty:
        raise ValidationError(["O arquivo não contém nenhuma linha de dados."])

    rows, errors, index = [], [], []
    for i, rec in enumerate(df[config.FEATURES].to_dict(orient="records"), start=2):
        rec = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
        clean, errs = validate_record(rec)
        if errs:
            errors.extend(f"Linha {i}: {e}" for e in errs)
        else:
            rows.append(clean)
            index.append(i)
    valid = pd.DataFrame(rows, columns=config.FEATURES, index=index)
    return valid, errors
