"""Captura as telas da aplicação usadas no artigo (artigo/figuras/tela_*.png).

Ferramenta de apoio, fora do fluxo de treino. Requer a aplicação rodando em
http://localhost:8501 e o pacote playwright (pip install playwright), usando o
Microsoft Edge já instalado no Windows:

    streamlit run app/streamlit_app.py
    python scripts/capturar_telas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artigo" / "figuras"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"

PACIENTE = {
    "numeros": {
        "Idade (anos)": "62",
        "Pressão arterial em repouso (mm Hg)": "150",
        "Colesterol sérico (mg/dl)": "260",
        "Frequência cardíaca máxima no esforço (bpm)": "115",
        "Depressão de ST no esforço — oldpeak (mm)": "2.0",
    },
    "selecoes": {
        "Sexo": "Masculino",
        "Tipo de dor no peito": "Assintomático",
        "Glicemia de jejum acima de 120 mg/dl?": "Sim",
        "ECG em repouso": "Normal",
        "Angina induzida por exercício?": "Sim",
        "Inclinação do segmento ST no pico do esforço": "Plano",
    },
}


def abrir(page: Page) -> None:
    page.goto(URL)
    page.get_by_role("button", name="Calcular risco").wait_for(timeout=60_000)
    page.wait_for_timeout(800)


def preencher_numero(page: Page, rotulo: str, valor: str) -> None:
    campo = page.get_by_label(rotulo, exact=True)
    campo.click()
    campo.fill(valor)
    campo.press("Tab")


def selecionar(page: Page, rotulo: str, opcao: str) -> None:
    page.get_by_role("combobox", name=rotulo, exact=False).first.click()
    page.get_by_role("option", name=opcao, exact=True).click()


def recorte(page: Page, inicio: str, fim: str | None, destino: Path, margem: int = 12) -> None:
    """Salva a região do bloco principal entre dois títulos de referência.

    `fim` pode ser um título (corta antes dele) ou "alerta" (corta depois do
    último quadro de alerta da página).
    """
    main = page.locator('[data-testid="stMainBlockContainer"]').bounding_box()
    y0 = page.get_by_role("heading", name=inicio, exact=False).first.bounding_box()["y"] - margem
    if fim == "alerta":
        box = page.locator('[data-testid="stAlert"]').last.bounding_box()
        y1 = box["y"] + box["height"] + margem
    elif fim:
        y1 = page.get_by_role("heading", name=fim, exact=False).first.bounding_box()["y"] - margem
    else:
        y1 = main["y"] + main["height"]
    page.screenshot(
        path=str(destino),
        clip={"x": main["x"] - 8, "y": y0, "width": main["width"] + 16, "height": y1 - y0},
    )
    print("salvo:", destino.relative_to(ROOT))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(
            viewport={"width": 1500, "height": 2600}, device_scale_factor=2,
            color_scheme="light", locale="pt-BR",
        )
        page = context.new_page()

        # 1) entradas inválidas -> mensagens de erro
        abrir(page)
        page.get_by_role("checkbox", name="Registrar predições").uncheck(force=True)
        preencher_numero(page, "Idade (anos)", "150")
        preencher_numero(page, "Pressão arterial em repouso (mm Hg)", "20")
        selecionar(page, "Sexo", "Masculino")
        page.get_by_role("button", name="Calcular risco").click()
        page.get_by_text("Não foi possível calcular").wait_for(timeout=30_000)
        page.wait_for_timeout(500)
        recorte(page, "Dados do paciente", "alerta", OUT / "tela_erros.png")

        # 2) paciente válido -> formulário preenchido e resultado
        abrir(page)
        page.get_by_role("checkbox", name="Registrar predições").uncheck(force=True)
        for rotulo, valor in PACIENTE["numeros"].items():
            preencher_numero(page, rotulo, valor)
        for rotulo, opcao in PACIENTE["selecoes"].items():
            selecionar(page, rotulo, opcao)
        page.wait_for_timeout(300)
        recorte(page, "Triagem de risco de doença cardíaca", "Dados do paciente", OUT / "tela_cabecalho.png")
        page.get_by_role("button", name="Calcular risco").click()
        page.get_by_text("Esta estimativa foi útil?").wait_for(timeout=120_000)
        recorte(page, "Dados do paciente", "Resultado", OUT / "tela_formulario.png")
        page.wait_for_timeout(2500)  # o gráfico termina de desenhar
        recorte(page, "Resultado", "Esta estimativa foi útil?", OUT / "tela_resultado.png")

        browser.close()


if __name__ == "__main__":
    main()
