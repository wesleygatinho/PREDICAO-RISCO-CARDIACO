"""Interface web (Streamlit) para o modelo de risco de doença cardíaca.

Executar a partir da raiz do repositório:

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.service import (  # noqa: E402
    ModelNotFoundError,
    RiskModel,
    log_feedback,
    log_prediction,
)
from app.validation import (  # noqa: E402
    ValidationError,
    record_to_frame,
    validate_dataframe,
    validate_record,
)
from src import config  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"

st.set_page_config(page_title="Triagem de risco cardíaco", page_icon="🫀", layout="wide")


# --------------------------------------------------------------------------- utilidades
def pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f}%".replace(".", ",")


def num(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def describe_value(col: str, value) -> str:
    """Valor do atributo em linguagem de usuário."""
    if col in config.CATEGORIES:
        return config.CATEGORIES[col][value]
    if col == "FastingBS":
        return "Sim" if int(value) == 1 else "Não"
    if col == "Cholesterol" and float(value) == 0:
        return "não medido"
    unit = config.UNITS.get(col, "")
    return f"{float(value):g} {unit}".strip()


@st.cache_resource(show_spinner="Carregando o modelo salvo...")
def load_model() -> RiskModel:
    return RiskModel()


def template_csv() -> bytes:
    example = pd.DataFrame(
        [
            [54, "M", "ASY", 140, 239, 0, "Normal", 160, "N", 1.2, "Flat"],
            [41, "F", "ATA", 130, 204, 0, "Normal", 172, "N", 0.0, "Up"],
        ],
        columns=config.FEATURES,
    )
    return example.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- cabeçalho
st.title("🫀 Triagem de risco de doença cardíaca")
st.caption(
    "Protótipo acadêmico — Inteligência Artificial (DEIN0118), UFMA. "
    "Estima a probabilidade de doença cardíaca a partir de 11 dados clínicos de rotina."
)

st.warning(
    "**Esta ferramenta não faz diagnóstico e não substitui avaliação médica.** "
    "O resultado é uma estimativa estatística para apoiar a priorização de pacientes "
    "e deve ser interpretado por um profissional de saúde.",
    icon="⚠️",
)

with st.expander("Para quem é, o que faz e quais são as limitações", expanded=False):
    st.markdown(
        """
**Para quem é.** Profissionais e estudantes da área de saúde que queiram uma
segunda leitura, rápida, sobre quais pacientes merecem investigação cardiológica
prioritária. Não foi feita para uso direto por pacientes.

**O que faz.** Recebe idade, sexo, tipo de dor no peito, pressão arterial,
colesterol, glicemia de jejum, ECG de repouso e três medidas do teste de esforço.
Devolve a probabilidade estimada de doença cardíaca, a classificação em risco
baixo ou elevado e os atributos que mais pesaram naquela estimativa.

**Limitações.**
- O modelo foi treinado com 918 registros históricos de cinco bases (Estados Unidos
  e Europa). Não foi validado em pacientes brasileiros nem em uso clínico real.
- 79% dos registros de treino são de homens; o desempenho em mulheres é menos
  confiável porque há poucos casos para medir.
- Em 19% dos registros o colesterol não foi medido, e esses casos se concentram em
  uma das bases de origem. O modelo usa a mediana do treino nesses casos.
- O alvo dos dados é a presença de doença cardíaca diagnosticada, usada como
  indicador de risco de insuficiência cardíaca; a ferramenta não prevê eventos futuros.
- O limiar de decisão foi escolhido para priorizar sensibilidade: prefere-se
  encaminhar um paciente saudável a deixar passar um doente. Isso aumenta os
  falsos positivos.
"""
    )

try:
    model = load_model()
except ModelNotFoundError as exc:
    st.error(f"{exc}", icon="🚫")
    st.stop()

meta = model.metadata

with st.sidebar:
    st.header("Modelo em uso")
    st.write(f"**{meta['rotulo']}**")
    st.write(f"Limiar de decisão: **{num(model.threshold, 2)}**")
    m = meta["metricas_teste_limiar_escolhido"]
    st.write(
        f"No conjunto de teste ({meta['n_teste']} pacientes): "
        f"revocação {num(m['revocacao'])}, precisão {num(m['precisao'])}, "
        f"AUC-ROC {num(m['auc_roc'])}."
    )
    st.divider()
    do_log = st.checkbox(
        "Registrar predições (sem identificação)", value=True,
        help="Grava os valores informados e o resultado em logs/predicoes.csv para auditoria. "
             "Nenhum nome ou documento é pedido ou armazenado.",
    )
    st.caption("Não digite nomes, documentos ou outros dados que identifiquem o paciente.")

tab_one, tab_batch, tab_about = st.tabs(["Paciente individual", "Arquivo com vários pacientes", "Sobre o modelo"])


# --------------------------------------------------------------------------- paciente individual
with tab_one:
    with st.form("form_paciente"):
        st.subheader("Dados do paciente")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Idade (anos)", value=None, step=1, placeholder="ex.: 54")
            sex = st.selectbox(
                "Sexo", list(config.CATEGORIES["Sex"]), index=None, placeholder="Selecione",
                format_func=lambda k: config.CATEGORIES["Sex"][k],
            )
            chest = st.selectbox(
                "Tipo de dor no peito", list(config.CATEGORIES["ChestPainType"]), index=None,
                placeholder="Selecione", format_func=lambda k: config.CATEGORIES["ChestPainType"][k],
            )
            fasting = st.selectbox(
                "Glicemia de jejum acima de 120 mg/dl?", [0, 1], index=None, placeholder="Selecione",
                format_func=lambda v: "Sim" if v == 1 else "Não",
            )
        with c2:
            bp = st.number_input("Pressão arterial em repouso (mm Hg)", value=None, step=1, placeholder="ex.: 130")
            chol = st.number_input(
                "Colesterol sérico (mg/dl)", value=None, step=1, placeholder="ex.: 220",
                help="Se o exame não foi feito, deixe em branco e marque a opção abaixo.",
            )
            chol_missing = st.checkbox("Colesterol não medido")
            ecg = st.selectbox(
                "ECG em repouso", list(config.CATEGORIES["RestingECG"]), index=None,
                placeholder="Selecione", format_func=lambda k: config.CATEGORIES["RestingECG"][k],
            )
        with c3:
            maxhr = st.number_input("Frequência cardíaca máxima no esforço (bpm)", value=None, step=1, placeholder="ex.: 150")
            angina = st.selectbox(
                "Angina induzida por exercício?", list(config.CATEGORIES["ExerciseAngina"]), index=None,
                placeholder="Selecione", format_func=lambda k: config.CATEGORIES["ExerciseAngina"][k],
            )
            oldpeak = st.number_input(
                "Depressão de ST no esforço — oldpeak (mm)", value=None, step=0.1, format="%.1f",
                placeholder="ex.: 1.0",
            )
            slope = st.selectbox(
                "Inclinação do segmento ST no pico do esforço", list(config.CATEGORIES["ST_Slope"]),
                index=None, placeholder="Selecione", format_func=lambda k: config.CATEGORIES["ST_Slope"][k],
            )
        submitted = st.form_submit_button("Calcular risco", type="primary")

    if submitted:
        raw = {
            "Age": age, "Sex": sex, "ChestPainType": chest, "RestingBP": bp,
            "Cholesterol": 0 if chol_missing else chol, "FastingBS": fasting, "RestingECG": ecg,
            "MaxHR": maxhr, "ExerciseAngina": angina, "Oldpeak": oldpeak, "ST_Slope": slope,
        }
        errors = []
        if chol_missing and chol is not None:
            errors.append(
                "Colesterol sérico: você informou um valor e marcou 'não medido'. Mantenha apenas um dos dois."
            )
        if not chol_missing and chol is None:
            errors.append(
                "Colesterol sérico: informe o valor ou marque 'Colesterol não medido'."
            )
        elif not chol_missing and chol == 0:
            errors.append(
                "Colesterol sérico: 0 mg/dl não é um valor possível; se o exame não foi feito, marque 'Colesterol não medido'."
            )
        clean, field_errors = validate_record(raw)
        errors += [e for e in field_errors if not (e.startswith("Colesterol") and errors)]
        if errors:
            st.session_state.pop("resultado", None)
            st.error(
                "Não foi possível calcular. Corrija os campos abaixo e tente de novo:\n\n"
                + "\n".join(f"- {e}" for e in errors),
                icon="🚫",
            )
        else:
            row = record_to_frame(clean)
            pred = model.predict_one(row)
            with st.spinner("Calculando a explicação..."):
                contrib, base = model.explain(row)
            if do_log:
                log_prediction(pred, clean, meta["modelo"])
            st.session_state["resultado"] = {
                "pred": pred, "contrib": contrib, "base": base, "registro": clean, "feedback": False,
            }

    res = st.session_state.get("resultado")
    if res:
        pred, contrib, base = res["pred"], res["contrib"], res["base"]
        st.divider()
        st.subheader("Resultado")
        k1, k2, k3 = st.columns([1.1, 1, 1.4])
        k1.metric("Probabilidade estimada de doença cardíaca", pct(pred.probability))
        k2.metric("Limiar de decisão", pct(pred.threshold, 0))
        with k3:
            if pred.positive:
                st.error(
                    "**Risco elevado** — probabilidade igual ou acima do limiar. "
                    "Sugere priorizar a avaliação cardiológica.", icon="🔴",
                )
            else:
                st.success(
                    "**Risco baixo** — probabilidade abaixo do limiar. "
                    "Não descarta doença: sintomas e julgamento clínico prevalecem.", icon="🟢",
                )
        st.progress(min(max(pred.probability, 0.0), 1.0))
        st.caption(
            f"Como ler: entre pacientes com este perfil, o modelo estima que cerca de "
            f"{round(pred.probability * 100)} em cada 100 tenham doença cardíaca. "
            f"A confiança na classe indicada é {pct(pred.probability if pred.positive else 1 - pred.probability)}. "
            "A probabilidade vem de um modelo estatístico e tem margem de erro."
        )

        st.subheader("O que mais pesou nesta estimativa")
        chart_df = contrib.assign(
            descricao=[f"{r.rotulo}: {describe_value(r.atributo, r.valor)}" for r in contrib.itertuples()],
            efeito=["Aumenta o risco" if v >= 0 else "Reduz o risco" for v in contrib["contribuicao"]],
            pontos=(contrib["contribuicao"] * 100).round(1),
        )
        chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusEnd=3, height=16)
            .encode(
                x=alt.X("pontos:Q", title="Contribuição para a probabilidade (pontos percentuais)"),
                y=alt.Y(
                    "descricao:N", sort=None, title=None,
                    axis=alt.Axis(labelLimit=420, labelOverlap=False, labelFontSize=13),
                ),
                color=alt.Color(
                    "efeito:N", title=None,
                    scale=alt.Scale(domain=["Aumenta o risco", "Reduz o risco"], range=[ORANGE, BLUE]),
                    legend=alt.Legend(orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("descricao:N", title="Atributo"),
                    alt.Tooltip("pontos:Q", title="Contribuição (p.p.)"),
                    alt.Tooltip("efeito:N", title="Efeito"),
                ],
            )
            .properties(height=alt.Step(30))
        )
        st.altair_chart(chart, use_container_width=True)
        up = chart_df[chart_df["pontos"] > 0].head(3)["descricao"].tolist()
        down = chart_df[chart_df["pontos"] < 0].head(3)["descricao"].tolist()
        if up:
            st.markdown("**Principais fatores que aumentaram o risco:** " + "; ".join(up) + ".")
        if down:
            st.markdown("**Principais fatores que reduziram o risco:** " + "; ".join(down) + ".")
        st.caption(
            f"Valores SHAP: partem da probabilidade média de um grupo de referência do treino "
            f"({pct(base)}) e somam a contribuição de cada atributo até chegar à probabilidade deste "
            f"paciente ({pct(pred.probability)}). Mostram o que o modelo usou, não relações de causa e efeito."
        )

        st.subheader("Esta estimativa foi útil?")
        if res["feedback"]:
            st.success("Obrigado, sua avaliação foi registrada.", icon="✅")
        else:
            with st.form("form_feedback"):
                useful = st.radio(
                    "A estimativa parece coerente com o caso?",
                    ["Sim", "Não", "Não sei avaliar"], horizontal=True, index=None,
                )
                comment = st.text_area(
                    "Comentário (opcional, sem dados que identifiquem o paciente)", max_chars=500
                )
                if st.form_submit_button("Enviar avaliação"):
                    if useful is None:
                        st.error("Escolha uma das opções antes de enviar.", icon="🚫")
                    else:
                        log_feedback(pred.id, useful, comment)
                        res["feedback"] = True
                        st.rerun()


# --------------------------------------------------------------------------- lote
with tab_batch:
    st.subheader("Predição para vários pacientes")
    st.write(
        "Envie um arquivo CSV com uma linha por paciente e as colunas abaixo "
        "(mesmos nomes e códigos do conjunto de dados original)."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Coluna": c,
                    "Significado": config.LABELS_PT[c],
                    "Valores aceitos": (
                        ", ".join(f"{k} = {v}" for k, v in config.CATEGORIES[c].items())
                        if c in config.CATEGORIES
                        else "0 = não, 1 = sim" if c == "FastingBS"
                        else f"{config.VALID_RANGES[c][0]:g} a {config.VALID_RANGES[c][1]:g} {config.UNITS[c]}"
                        + (" (0 ou vazio = não medido)" if c == "Cholesterol" else "")
                    ),
                }
                for c in config.FEATURES
            ]
        ),
        hide_index=True, use_container_width=True,
    )
    st.download_button("Baixar modelo de arquivo (CSV)", template_csv(), "modelo_pacientes.csv", "text/csv")

    uploaded = st.file_uploader("Arquivo CSV", type=["csv"])
    if uploaded is not None:
        try:
            try:
                batch = pd.read_csv(uploaded, sep=None, engine="python")
            except Exception as exc:  # arquivo ilegível
                raise ValidationError([f"Não foi possível ler o arquivo como CSV ({exc})."]) from exc
            if len(batch) > 5000:
                raise ValidationError(["O arquivo tem mais de 5.000 linhas; divida-o em partes menores."])
            valid, batch_errors = validate_dataframe(batch)
        except ValidationError as exc:
            st.error("Arquivo recusado:\n\n" + "\n".join(f"- {e}" for e in exc.messages), icon="🚫")
        else:
            if batch_errors:
                shown = batch_errors[:20]
                extra = len(batch_errors) - len(shown)
                st.error(
                    f"{len(batch) - len(valid)} linha(s) com erro foram ignoradas:\n\n"
                    + "\n".join(f"- {e}" for e in shown)
                    + (f"\n- ... e mais {extra} erro(s)." if extra > 0 else ""),
                    icon="🚫",
                )
            if len(valid):
                proba = model.predict_proba(valid)
                out = valid.copy()
                out.insert(0, "linha_do_arquivo", valid.index)
                out["probabilidade"] = proba.round(4).to_numpy()
                out["classificacao"] = ["Risco elevado" if p >= model.threshold else "Risco baixo" for p in proba]
                st.success(
                    f"{len(valid)} paciente(s) avaliados; "
                    f"{int((proba >= model.threshold).sum())} com risco elevado "
                    f"(limiar {num(model.threshold, 2)}).",
                    icon="✅",
                )
                st.dataframe(
                    out.sort_values("probabilidade", ascending=False),
                    hide_index=True, use_container_width=True,
                    column_config={
                        "probabilidade": st.column_config.ProgressColumn(
                            "probabilidade", min_value=0.0, max_value=1.0, format="%.3f"
                        )
                    },
                )
                st.download_button(
                    "Baixar resultados (CSV)", out.to_csv(index=False).encode("utf-8"),
                    "predicoes.csv", "text/csv",
                )
            else:
                st.info("Nenhuma linha válida para avaliar.")


# --------------------------------------------------------------------------- sobre
with tab_about:
    st.subheader("Como o modelo foi construído")
    m_thr, m_050 = meta["metricas_teste_limiar_escolhido"], meta["metricas_teste_limiar_050"]
    st.markdown(
        f"""
- **Dados:** Heart Failure Prediction Dataset (fedesoriano, Kaggle), 918 pacientes,
  11 atributos clínicos; alvo: presença de doença cardíaca.
- **Protocolo:** {meta['n_treino']} pacientes para treino e {meta['n_teste']} para teste
  (divisão estratificada, semente {meta['semente']}); hiperparâmetros escolhidos por validação
  cruzada estratificada apenas no treino.
- **Modelo final:** {meta['rotulo']} ({meta['familia']}), escolhido por ter a {meta['criterio_selecao']}.
- **Pré-processamento:** o mesmo Pipeline salvo no treino faz a imputação, a padronização e a
  codificação; a aplicação apenas o carrega do arquivo `models/modelo_final.joblib`.
- **Limiar de decisão:** {num(model.threshold, 2)} ({meta['criterio_limiar']}).
- **Treinado em:** {meta['treinado_em']} · scikit-learn {meta['versoes']['scikit-learn']}.
"""
    )
    st.markdown("**Desempenho no conjunto de teste** (pacientes nunca vistos no treino)")
    rows = []
    for name, mm in ((f"Limiar {num(model.threshold, 2)} (em uso)", m_thr), ("Limiar 0,50", m_050)):
        rows.append(
            {
                "Configuração": name,
                "Revocação": num(mm["revocacao"]), "Precisão": num(mm["precisao"]),
                "Especificidade": num(mm["especificidade"]), "F1": num(mm["f1"]),
                "AUC-ROC": num(mm["auc_roc"]),
                "Falsos negativos": mm["fn"], "Falsos positivos": mm["fp"],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Revocação: fração dos doentes que o modelo identifica. Precisão: fração dos alertas que "
        "eram de fato doentes. Com apenas 184 pacientes no teste, essas medidas têm incerteza "
        "de alguns pontos percentuais."
    )
    with st.expander("Hiperparâmetros do modelo final"):
        st.json(meta["hiperparametros"])
