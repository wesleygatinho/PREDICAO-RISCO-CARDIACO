# Predição de risco de insuficiência cardíaca

Trabalho final de Aprendizado de Máquina — Inteligência Artificial (DEIN0118), UFMA, 2026.2.

**Autores:** TODO: Nome do Autor 1, Nome do Autor 2
**Vídeo de demonstração:** TODO: colar aqui o link do vídeo (até 5 minutos)
**Aplicação pública:** TODO: colar aqui o link, se for implantada (ver "Implantação pública")

Classificação binária em saúde: a partir de 11 dados clínicos de rotina, o modelo estima a
probabilidade de um paciente ter doença cardíaca, usada como indicador de risco de
insuficiência cardíaca. O repositório contém o treino reprodutível, o modelo salvo, uma
aplicação web (Streamlit) e o artigo no formato SBC.

> **Aviso.** Protótipo acadêmico. Não faz diagnóstico e não substitui avaliação médica.

## Resultado em uma tabela

Conjunto de teste (184 pacientes, separado antes de qualquer ajuste), limiar 0,50:

| Modelo | Acurácia | Precisão | Revocação | F1 | AUC-ROC | AUC-ROC na validação cruzada |
|---|---|---|---|---|---|---|
| Classe majoritária (baseline trivial) | 0,554 | 0,554 | 1,000 | 0,713 | 0,500 | 0,500 ± 0,000 |
| Regressão logística (baseline) | 0,875 | 0,899 | 0,873 | 0,886 | 0,931 | 0,920 ± 0,027 |
| k-NN | 0,891 | 0,902 | 0,902 | 0,902 | 0,938 | 0,922 ± 0,029 |
| SVM (RBF) | 0,880 | 0,885 | 0,902 | 0,893 | 0,936 | 0,922 ± 0,027 |
| **Random Forest (modelo final)** | 0,848 | 0,849 | 0,882 | 0,865 | 0,923 | **0,930 ± 0,024** |
| Gradient Boosting | 0,891 | 0,902 | 0,902 | 0,902 | 0,931 | 0,928 ± 0,024 |
| MLP | 0,870 | 0,875 | 0,892 | 0,883 | 0,930 | 0,927 ± 0,024 |

O modelo final foi escolhido pela maior AUC-ROC média na validação cruzada, regra fixada antes
de olhar o teste. No teste, os seis modelos ficam praticamente empatados e o Random Forest não é
o melhor; essa diferença está dentro da incerteza de um teste com 184 pacientes e é discutida
no artigo. Com o limiar de decisão ajustado para priorizar revocação (0,29, escolhido só com
dados de treino), o modelo final atinge revocação 0,951 e precisão 0,782 no teste
(5 falsos negativos e 27 falsos positivos).

## Dados

- **Conjunto:** Heart Failure Prediction Dataset, de fedesoriano (Kaggle, 2021):
  <https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction>
- **Licença:** Open Database License (ODbL) para a base; conteúdo © dos autores originais.
- **Origem:** combinação de cinco bases do UCI Heart Disease (Cleveland, Hungria, Suíça,
  Long Beach VA e Statlog), 918 pacientes após remoção de duplicatas.
- **Privacidade:** não há nome, documento, data ou outro identificador de paciente.
- O arquivo `data/heart.csv` (36 kB) está no repositório. SHA-256:
  `948420b084d8a3a0ca42b8419fce9aee175879e43f8aedf712377899a67aa49b`.

## Estrutura

```
data/heart.csv              conjunto de dados original
src/config.py               caminhos, semente, colunas e faixas de validação
src/data.py                 leitura e divisão treino/teste
src/preprocessing.py        pipeline de pré-processamento (ajustado só no treino)
src/models.py               modelos candidatos e grades de hiperparâmetros
src/eda.py                  análise exploratória e figuras
src/train.py                busca de hiperparâmetros, comparação, limiar e persistência
src/evaluate.py             curvas ROC, matriz de confusão, SHAP, subgrupos, erros
src/explain.py              valores SHAP no espaço dos atributos originais
app/streamlit_app.py        interface web
app/service.py              carrega o modelo salvo, prediz, explica e registra
app/validation.py           validação das entradas
models/modelo_final.joblib  pipeline final (pré-processamento + classificador)
models/metadata.json        limiar, hiperparâmetros, métricas e versões
models/shap_background.csv  50 pacientes de treino usados como referência do SHAP
notebooks/                  notebook da análise exploratória
reports/                    tabelas, figuras e registros dos experimentos
tests/                      testes automatizados (pytest)
artigo/                     artigo no modelo SBC (LaTeX e PDF)
scripts/capturar_telas.py   capturas de tela usadas no artigo (opcional)
```

## Instalação (máquina limpa)

Requer Python 3.11 ou mais novo.

Windows (PowerShell):

```powershell
git clone <URL-DO-REPOSITORIO>
cd predicao-insuficiencia-cardiaca
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux ou macOS:

```bash
git clone <URL-DO-REPOSITORIO>
cd predicao-insuficiencia-cardiaca
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Executar a aplicação

O modelo treinado já está em `models/`, então basta:

```bash
streamlit run app/streamlit_app.py
```

O navegador abre em <http://localhost:8501>. A aplicação tem três abas:

1. **Paciente individual:** formulário com os 11 dados, probabilidade estimada, classificação
   (risco baixo ou elevado), gráfico com a contribuição de cada atributo (SHAP) e um campo para
   avaliar se a estimativa foi útil.
2. **Arquivo com vários pacientes:** envio de um CSV, validação linha a linha e download dos
   resultados.
3. **Sobre o modelo:** dados, protocolo, métricas no teste e hiperparâmetros.

Entradas inválidas (campo vazio, valor fora da faixa, categoria desconhecida, arquivo sem as
colunas esperadas) geram mensagens que dizem qual campo corrigir e qual é a faixa aceita.

A aplicação carrega o `Pipeline` salvo em `models/modelo_final.joblib` e não reimplementa
nenhuma etapa de pré-processamento.

### Registros e feedback

Com a opção "Registrar predições" marcada, cada predição é gravada em `logs/predicoes.csv`
(valores informados, probabilidade, limiar e classe; nenhum identificador de paciente) e cada
avaliação do usuário em `logs/feedback.csv`. A pasta `logs/` não é versionada.

## Reproduzir o treino

```bash
python -m src.eda        # análise exploratória e figuras 1 a 5   (segundos)
python -m src.train      # busca de hiperparâmetros e modelo final (cerca de 4 minutos)
python -m src.evaluate   # avaliação, SHAP e análise de erros     (cerca de 5 minutos)
```

A semente aleatória é 42 em todas as etapas (`src/config.py`). Os resultados ficam em
`reports/` (`resultados_cv.csv`, `resultados_teste.csv`, `resumo_experimentos.json`,
`busca_hiperparametros/*.csv`, `figures/*.png`) e o modelo em `models/`.

Protocolo, em resumo:

1. divisão estratificada 80/20 (734 treino, 184 teste);
2. pré-processamento dentro do `Pipeline`, ajustado apenas nos dados de treino de cada dobra;
3. busca em grade com validação cruzada estratificada de 5 dobras repetida 3 vezes;
4. modelo final = maior AUC-ROC média na validação cruzada;
5. limiar de decisão = máximo F2 nas predições out-of-fold do treino;
6. avaliação única no conjunto de teste.

## Testes

```bash
pytest
```

São 42 testes: esquema dos dados, divisão reprodutível, ausência de vazamento no
pré-processamento, métricas, modelo salvo, validação de entradas, registro de predições e a
interface (executada em memória com `streamlit.testing`).

## Implantação pública (opcional)

No [Streamlit Community Cloud](https://streamlit.io/cloud): criar um app apontando para este
repositório, arquivo principal `app/streamlit_app.py`, Python 3.11. Nenhuma variável de ambiente
é necessária. Em hospedagem gratuita o disco é temporário, então os arquivos de `logs/` se
perdem quando o app reinicia.

## Artigo

`artigo/artigo.pdf` (modelo SBC). Para recompilar: `cd artigo` e
`pdflatex artigo && bibtex artigo && pdflatex artigo && pdflatex artigo`.

## Uso de IA generativa

Ver a seção "Declaração de uso de IA" do artigo.
