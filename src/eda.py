"""Análise exploratória de dados (AED).

Uso:  python -m src.eda

Gera as figuras em reports/figures/ e um resumo numérico em reports/eda_resumo.json.
O diagnóstico de estrutura e qualidade usa o conjunto completo; a análise das
relações entre atributos e alvo usa apenas o conjunto de treino, para que nenhuma
decisão de modelagem seja tomada olhando para o conjunto de teste.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, plotting
from .data import load_data, split_data

FIG = config.FIGURES_DIR


# --------------------------------------------------------------------------- diagnóstico
def iqr_outliers(s: pd.Series) -> int:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def quality_summary(df: pd.DataFrame) -> dict:
    """Estrutura, tipos, ausentes, duplicados, valores impossíveis e outliers."""
    out = {
        "n_linhas": int(df.shape[0]),
        "n_colunas": int(df.shape[1]),
        "tipos": {c: str(t) for c, t in df.dtypes.items()},
        "ausentes_explicitos": {c: int(v) for c, v in df.isna().sum().items()},
        "linhas_duplicadas": int(df.duplicated().sum()),
        "zeros_impossiveis": {
            c: int((df[c] == 0).sum()) for c in config.NUM_ZERO_AS_MISSING
        },
        "oldpeak_negativo": int((df["Oldpeak"] < 0).sum()),
        "outliers_iqr": {c: iqr_outliers(df[c]) for c in config.NUMERIC_ALL},
        "outliers_iqr_colesterol_sem_zeros": iqr_outliers(
            df.loc[df["Cholesterol"] > 0, "Cholesterol"]
        ),
        "descritivas": json.loads(df[config.NUMERIC_ALL].describe().round(2).to_json()),
        "alvo_contagem": {int(k): int(v) for k, v in df[config.TARGET].value_counts().items()},
        "alvo_proporcao": {
            int(k): round(float(v), 4)
            for k, v in df[config.TARGET].value_counts(normalize=True).items()
        },
        "sexo_contagem": {k: int(v) for k, v in df["Sex"].value_counts().items()},
        "prevalencia_colesterol_zero": round(
            float(df.loc[df["Cholesterol"] == 0, config.TARGET].mean()), 4
        ),
        "prevalencia_colesterol_medido": round(
            float(df.loc[df["Cholesterol"] > 0, config.TARGET].mean()), 4
        ),
    }
    return out


def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in config.FEATURES + [config.TARGET]:
        if col in config.CATEGORIES:
            tipo = "categórica nominal"
            valores = ", ".join(f"{k} ({v})" for k, v in config.CATEGORIES[col].items())
        elif col in ("FastingBS", config.TARGET):
            tipo = "binária"
            valores = "0, 1"
        else:
            tipo = "numérica"
            valores = f"{df[col].min():g} a {df[col].max():g} {config.UNITS.get(col, '')}".strip()
        obs = ""
        if col in config.NUM_ZERO_AS_MISSING:
            n0 = int((df[col] == 0).sum())
            obs = f"{n0} zeros = não medido ({n0 / len(df):.1%})"
        if col == config.TARGET:
            obs = "alvo"
        rows.append(
            {
                "coluna": col,
                "descricao": config.LABELS_PT[col],
                "tipo": tipo,
                "valores": valores,
                "pct_ausente_explicito": round(float(df[col].isna().mean() * 100), 2),
                "observacao": obs,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figuras
def fig_target(df: pd.DataFrame) -> None:
    """Distribuição do alvo, geral e por sexo."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1, 1.25]})

    counts = df[config.TARGET].value_counts().sort_index()
    ax = axes[0]
    bars = ax.bar(
        ["Sem doença (0)", "Com doença (1)"],
        counts.values,
        color=[plotting.BLUE, plotting.ORANGE],
        width=0.55,
    )
    for b, v in zip(bars, counts.values):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 8,
            f"{v}\n({v / len(df):.1%})".replace(".", ","),
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylim(0, counts.max() * 1.28)
    ax.set_ylabel("Pacientes")
    ax.set_title("(a) Classes do alvo")

    ax = axes[1]
    tab = pd.crosstab(df["Sex"], df[config.TARGET]).loc[["F", "M"]]
    x = np.arange(len(tab))
    w = 0.36
    for i, cls in enumerate([0, 1]):
        bars = ax.bar(
            x + (i - 0.5) * (w + 0.03), tab[cls].values, width=w,
            color=plotting.CLASS_COLORS[cls],
            label="Sem doença" if cls == 0 else "Com doença",
        )
        for b, v, tot in zip(bars, tab[cls].values, tab.sum(axis=1).values):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 8, f"{v}\n({v / tot:.0%})",
                ha="center", va="bottom", fontsize=8,
            )
    ax.set_xticks(x, [f"Feminino (n={tab.loc['F'].sum()})", f"Masculino (n={tab.loc['M'].sum()})"])
    ax.set_ylim(0, tab.values.max() * 1.3)
    ax.set_ylabel("Pacientes")
    ax.set_title("(b) Classes por sexo")
    ax.legend(loc="upper left")
    fig.tight_layout()
    plotting.save(fig, FIG / "fig01_distribuicao_alvo.png")


def fig_numeric_by_class(train: pd.DataFrame) -> None:
    """Boxplots das numéricas por classe (conjunto de treino)."""
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.6))
    for ax, col in zip(axes, config.NUMERIC_ALL):
        data = [train.loc[train[config.TARGET] == c, col].values for c in (0, 1)]
        bp = ax.boxplot(
            data, widths=0.55, patch_artist=True,
            medianprops={"color": plotting.INK, "linewidth": 1.2},
            whiskerprops={"color": plotting.INK_2, "linewidth": 0.8},
            capprops={"color": plotting.INK_2, "linewidth": 0.8},
            flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": plotting.MUTED,
                        "markeredgecolor": "none", "alpha": 0.7},
        )
        for patch, c in zip(bp["boxes"], (0, 1)):
            patch.set_facecolor(plotting.CLASS_COLORS[c])
            patch.set_edgecolor("white")
            patch.set_alpha(0.9)
        ax.set_xticks([1, 2], ["Sem", "Com"])
        unit = config.UNITS.get(col, "")
        ax.set_title(f"{col}", fontsize=9)
        ax.set_ylabel(unit, fontsize=8)
    fig.supxlabel("Doença cardíaca", fontsize=9, color=plotting.INK_2)
    fig.tight_layout()
    plotting.save(fig, FIG / "fig02_numericas_por_classe.png")


def fig_categorical_prevalence(train: pd.DataFrame) -> None:
    """Proporção de pacientes com doença cardíaca em cada categoria (treino)."""
    cols = config.CAT_FEATURES + config.BIN_FEATURES
    overall = train[config.TARGET].mean()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex=True)
    for ax, col in zip(axes.ravel(), cols):
        g = train.groupby(col)[config.TARGET].agg(["mean", "count"]).sort_values("mean")
        labels = [f"{idx}  (n={n})" for idx, n in zip(g.index, g["count"])]
        bars = ax.barh(labels, g["mean"].values, color=plotting.ORANGE, height=0.6)
        for b, v in zip(bars, g["mean"].values):
            ax.text(
                v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.0%}", va="center", fontsize=8,
                zorder=5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
            )
        ax.axvline(overall, color=plotting.INK_2, linewidth=0.9, linestyle="--", zorder=1)
        ax.set_xlim(0, 1.12)
        ax.set_title(col, fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.supxlabel(
        f"Proporção com doença cardíaca (linha tracejada: média do treino, {overall:.0%})",
        fontsize=9, color=plotting.INK_2,
    )
    fig.tight_layout()
    plotting.save(fig, FIG / "fig03_categoricas_por_classe.png")


def fig_correlation(train: pd.DataFrame) -> pd.DataFrame:
    """Correlação de Spearman entre numéricas, FastingBS e o alvo (treino).

    Os zeros de Cholesterol e RestingBP são tratados como ausentes aqui, para
    que o código de "não medido" não distorça os coeficientes.
    """
    d = train[config.NUMERIC_ALL + config.BIN_FEATURES + [config.TARGET]].astype(float)
    for c in config.NUM_ZERO_AS_MISSING:
        d.loc[d[c] == 0, c] = np.nan
    corr = d.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    im = ax.imshow(corr.values, cmap=plotting.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(
                j, i, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=7.5,
                color="white" if abs(v) > 0.6 else plotting.INK,
            )
    # separador branco entre células
    ax.set_xticks(np.arange(-0.5, len(corr), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(corr), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.set_label("Correlação de Spearman", fontsize=8)
    fig.tight_layout()
    plotting.save(fig, FIG / "fig04_correlacao.png")
    return corr


def fig_cholesterol_zero(df: pd.DataFrame) -> None:
    """Histograma do colesterol evidenciando o pico em zero (código de ausência)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    bins = np.arange(0, 620, 20)
    ax.hist(
        [df.loc[df[config.TARGET] == 0, "Cholesterol"], df.loc[df[config.TARGET] == 1, "Cholesterol"]],
        bins=bins, stacked=True, color=[plotting.BLUE, plotting.ORANGE],
        label=["Sem doença", "Com doença"], edgecolor="white", linewidth=0.6,
    )
    n0 = int((df["Cholesterol"] == 0).sum())
    ax.annotate(
        f"{n0} registros com valor 0\n(colesterol não medido)",
        xy=(10, n0 * 0.9), xytext=(110, n0 * 0.82), fontsize=8,
        arrowprops={"arrowstyle": "-", "color": plotting.INK_2, "linewidth": 0.8},
    )
    ax.set_xlabel("Colesterol sérico (mg/dl)")
    ax.set_ylabel("Pacientes")
    ax.set_title("(a) Distribuição do colesterol")
    ax.legend(loc="upper right")

    ax = axes[1]
    rates = [
        df.loc[df["Cholesterol"] > 0, config.TARGET].mean(),
        df.loc[df["Cholesterol"] == 0, config.TARGET].mean(),
    ]
    ns = [int((df["Cholesterol"] > 0).sum()), n0]
    bars = ax.bar(
        [f"Medido\n(n={ns[0]})", f"Não medido\n(n={ns[1]})"], rates,
        color=plotting.ORANGE, width=0.5,
    )
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Com doença cardíaca")
    ax.set_title("(b) Prevalência por situação")
    fig.tight_layout()
    plotting.save(fig, FIG / "fig05_colesterol_ausente.png")


# --------------------------------------------------------------------------- execução
def main() -> None:
    plotting.apply_style()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    train = X_train.assign(**{config.TARGET: y_train})

    resumo = quality_summary(df)
    resumo["n_treino"] = int(len(X_train))
    resumo["n_teste"] = int(len(X_test))
    resumo["prevalencia_por_categoria_treino"] = {
        col: {
            str(k): {"n": int(v["count"]), "prevalencia": round(float(v["mean"]), 4)}
            for k, v in train.groupby(col)[config.TARGET].agg(["mean", "count"]).iterrows()
        }
        for col in config.CAT_FEATURES + config.BIN_FEATURES
    }

    data_dictionary(df).to_csv(config.REPORTS_DIR / "dicionario_dados.csv", index=False)
    fig_target(df)
    fig_numeric_by_class(train)
    fig_categorical_prevalence(train)
    corr = fig_correlation(train)
    fig_cholesterol_zero(df)
    resumo["correlacao_spearman_com_alvo_treino"] = {
        k: round(float(v), 3) for k, v in corr[config.TARGET].drop(config.TARGET).items()
    }
    resumo["maior_correlacao_entre_atributos_treino"] = round(
        float(
            corr.drop(index=config.TARGET, columns=config.TARGET)
            .where(~np.eye(len(corr) - 1, dtype=bool))
            .abs()
            .max()
            .max()
        ),
        3,
    )

    with open(config.REPORTS_DIR / "eda_resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    print(f"Linhas: {resumo['n_linhas']} | colunas: {resumo['n_colunas']}")
    print(f"Ausentes explícitos: {sum(resumo['ausentes_explicitos'].values())} | duplicadas: {resumo['linhas_duplicadas']}")
    print(f"Zeros impossíveis: {resumo['zeros_impossiveis']}")
    print(f"Outliers (IQR): {resumo['outliers_iqr']}")
    print(f"Alvo: {resumo['alvo_proporcao']}")
    print(f"Figuras salvas em {FIG}")


if __name__ == "__main__":
    main()
