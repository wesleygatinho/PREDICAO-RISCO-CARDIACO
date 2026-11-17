"""Avaliação final, interpretação e análise de erros.

Uso:  python -m src.evaluate      (depois de python -m src.train)

Lê o modelo salvo e as predições de teste geradas pelo treino e produz:
curvas ROC, matrizes de confusão, curva de limiar, calibração, importância por
permutação, valores SHAP, análise de subgrupos e perfil dos erros.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve

from . import config, plotting
from .data import load_data, split_data
from .explain import PipelineExplainer
from .models import get_candidates
from .train import metrics_at

FIG = config.FIGURES_DIR


def fmt(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}".replace(".", ",")


# --------------------------------------------------------------------------- figuras
def fig_roc(test_proba: pd.DataFrame, labels: dict, final_name: str) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    y = test_proba["y_true"].to_numpy()
    order = sorted(
        [c for c in test_proba.columns if c.startswith("proba_")],
        key=lambda c: -roc_auc_score(y, test_proba[c]),
    )
    colors = iter(plotting.SERIES)
    for col in order:
        name = col.replace("proba_", "")
        auc = roc_auc_score(y, test_proba[col])
        if name == "dummy":
            continue
        fpr, tpr, _ = roc_curve(y, test_proba[col])
        ax.plot(
            fpr, tpr, color=next(colors),
            linewidth=2.4 if name == final_name else 1.3,
            label=f"{labels[name]} (AUC = {fmt(auc)})" + (" *" if name == final_name else ""),
        )
    ax.plot([0, 1], [0, 1], color=plotting.MUTED, linewidth=1, linestyle="--",
            label="Classe majoritária (AUC = 0,500)")
    ax.set_xlabel("Taxa de falsos positivos (1 − especificidade)")
    ax.set_ylabel("Taxa de verdadeiros positivos (revocação)")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0, 1.01)
    ax.grid(axis="both", visible=True)
    ax.legend(loc="lower right", fontsize=7.5, title="* modelo final", title_fontsize=7.5)
    fig.tight_layout()
    plotting.save(fig, FIG / "fig06_curvas_roc.png")


def fig_confusion(y, proba, threshold: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.9))
    for ax, thr in zip(axes, (0.5, threshold)):
        m = metrics_at(y, proba, thr)
        cm = np.array([[m["vn"], m["fp"]], [m["fn"], m["vp"]]])
        ax.imshow(cm, cmap=plotting.SEQ_BLUE, vmin=0, vmax=cm.max() * 1.15)
        names = [["VN", "FP"], ["FN", "VP"]]
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]}\n{names[i][j]}", ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() * 0.55 else plotting.INK,
                )
        ax.set_xticks([0, 1], ["Sem doença", "Com doença"])
        ax.set_yticks([0, 1], ["Sem doença", "Com doença"], rotation=90, va="center")
        ax.set_xlabel("Classe predita")
        ax.set_ylabel("Classe real")
        ax.set_title(f"Limiar = {fmt(thr, 2)}  (revocação {fmt(m['revocacao'])})", fontsize=9)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    plotting.save(fig, FIG / "fig07_matriz_confusao.png")


def fig_threshold_and_calibration(curve: pd.DataFrame, threshold: float, y, proba) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axes[0]
    ax.plot(curve["limiar"], curve["revocacao"], color=plotting.ORANGE, label="Revocação")
    ax.plot(curve["limiar"], curve["precisao"], color=plotting.BLUE, label="Precisão")
    ax.plot(curve["limiar"], curve["fbeta"], color=plotting.AQUA, label="F2")
    ax.axvline(threshold, color=plotting.INK_2, linestyle="--", linewidth=1)
    ax.text(threshold + 0.015, 0.52, f"limiar = {fmt(threshold, 2)}", fontsize=8, color=plotting.INK_2)
    ax.set_xlabel("Limiar de decisão")
    ax.set_ylabel("Valor (predições out-of-fold, treino)")
    ax.set_ylim(0.5, 1.01)
    ax.set_title("(a) Escolha do limiar")
    ax.legend(loc="lower left")

    ax = axes[1]
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=8, strategy="quantile")
    ax.plot([0, 1], [0, 1], color=plotting.MUTED, linestyle="--", linewidth=1, label="Calibração perfeita")
    ax.plot(mean_pred, frac_pos, color=plotting.ORANGE, marker="o", markersize=5,
            markeredgecolor="white", label="Modelo final (teste)")
    ax.set_xlabel("Probabilidade prevista (média da faixa)")
    ax.set_ylabel("Fração observada com doença")
    ax.set_title("(b) Calibração no teste")
    ax.grid(axis="both", visible=True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    plotting.save(fig, FIG / "fig08_limiar_calibracao.png")
    return {"prob_media_prevista": mean_pred.round(4).tolist(), "fracao_observada": frac_pos.round(4).tolist()}


def fig_importance(perm: pd.DataFrame, shap_abs: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    ax = axes[0]
    p = perm.sort_values("media")
    ax.barh([config.LABELS_PT[c] for c in p.index], p["media"], xerr=p["dp"], color=plotting.BLUE,
            height=0.62, error_kw={"ecolor": plotting.INK_2, "elinewidth": 0.8, "capsize": 2})
    ax.set_xlabel("Queda média da AUC-ROC ao permutar")
    ax.set_title("(a) Importância por permutação")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)

    ax = axes[1]
    s = shap_abs.sort_values()
    ax.barh([config.LABELS_PT[c] for c in s.index], s.values, color=plotting.BLUE, height=0.62)
    ax.set_xlabel("Média de |SHAP| (probabilidade)")
    ax.set_title("(b) Valores SHAP")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    fig.tight_layout()
    plotting.save(fig, FIG / "fig09_importancia_atributos.png")


def fig_shap_dependence(X: pd.DataFrame, shap_values: np.ndarray) -> None:
    """Contribuição SHAP em função do valor do atributo, para os 4 mais influentes."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top = [config.FEATURES[i] for i in np.argsort(-mean_abs)[:4]]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=True)
    rng = np.random.default_rng(config.SEED)
    for ax, col in zip(axes, top):
        j = config.FEATURES.index(col)
        sv = shap_values[:, j]
        colors = np.where(sv >= 0, plotting.ORANGE, plotting.BLUE)
        if col in config.CATEGORIES or col in config.BIN_FEATURES:
            cats = list(config.CATEGORIES[col].keys()) if col in config.CATEGORIES else [0, 1]
            pos = X[col].map({c: i for i, c in enumerate(cats)}).to_numpy(dtype=float)
            ax.scatter(pos + rng.uniform(-0.18, 0.18, len(pos)), sv, s=9, c=colors, alpha=0.75, linewidths=0)
            ax.set_xticks(range(len(cats)), [str(c) for c in cats])
            ax.set_xlim(-0.5, len(cats) - 0.5)
        else:
            ax.scatter(X[col], sv, s=9, c=colors, alpha=0.75, linewidths=0)
        ax.axhline(0, color=plotting.AXIS, linewidth=0.8)
        ax.set_title(col, fontsize=9)
    axes[0].set_ylabel("Contribuição SHAP")
    fig.supxlabel("Valor do atributo (laranja: aumenta o risco; azul: reduz)", fontsize=8.5,
                  color=plotting.INK_2)
    fig.tight_layout()
    plotting.save(fig, FIG / "fig10_shap_dependencia.png")


# --------------------------------------------------------------------------- análises
def subgroup_table(X: pd.DataFrame, y: pd.Series, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    groups = {
        "Todos": np.ones(len(X), dtype=bool),
        "Sexo = M": (X["Sex"] == "M").to_numpy(),
        "Sexo = F": (X["Sex"] == "F").to_numpy(),
        "Idade < 55": (X["Age"] < 55).to_numpy(),
        "Idade >= 55": (X["Age"] >= 55).to_numpy(),
        "Colesterol medido": (X["Cholesterol"] > 0).to_numpy(),
        "Colesterol não medido": (X["Cholesterol"] == 0).to_numpy(),
    }
    rows = []
    for name, mask in groups.items():
        yt, pr = y.to_numpy()[mask], proba[mask]
        m = metrics_at(yt, pr, threshold) if len(np.unique(yt)) == 2 else None
        rows.append(
            {
                "subgrupo": name,
                "n": int(mask.sum()),
                "positivos": int(yt.sum()),
                "prevalencia": float(yt.mean()),
                **({k: m[k] for k in ["precisao", "revocacao", "especificidade", "f1", "auc_roc", "fn", "fp"]}
                   if m else {}),
            }
        )
    return pd.DataFrame(rows)


def error_profile(X: pd.DataFrame, y: pd.Series, proba: np.ndarray, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = X.copy()
    d["y_true"] = y.to_numpy()
    d["proba"] = proba
    d["pred"] = (proba >= threshold).astype(int)
    d["grupo"] = np.select(
        [(d.y_true == 1) & (d.pred == 1), (d.y_true == 1) & (d.pred == 0),
         (d.y_true == 0) & (d.pred == 1), (d.y_true == 0) & (d.pred == 0)],
        ["VP", "FN", "FP", "VN"],
        default="",
    )
    chol = d["Cholesterol"].where(d["Cholesterol"] > 0)
    prof = d.groupby("grupo").agg(
        n=("y_true", "size"),
        idade_media=("Age", "mean"),
        maxhr_media=("MaxHR", "mean"),
        oldpeak_medio=("Oldpeak", "mean"),
        proba_media=("proba", "mean"),
        pct_masculino=("Sex", lambda s: (s == "M").mean()),
        pct_assintomatico=("ChestPainType", lambda s: (s == "ASY").mean()),
        pct_angina_exercicio=("ExerciseAngina", lambda s: (s == "Y").mean()),
        pct_st_ascendente=("ST_Slope", lambda s: (s == "Up").mean()),
        pct_st_plano=("ST_Slope", lambda s: (s == "Flat").mean()),
        pct_glicemia_alta=("FastingBS", "mean"),
    )
    prof["colesterol_medio_medido"] = chol.groupby(d["grupo"]).mean()
    prof["pct_colesterol_nao_medido"] = (d["Cholesterol"] == 0).groupby(d["grupo"]).mean()
    prof = prof.reindex(["VP", "FN", "VN", "FP"])
    errors = d[d["grupo"].isin(["FN", "FP"])].sort_values(["grupo", "proba"])
    return prof, errors


# --------------------------------------------------------------------------- execução
def main() -> None:
    plotting.apply_style()
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = joblib.load(config.MODEL_PATH)
    with open(config.METADATA_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    threshold = meta["limiar"]
    final_name = meta["modelo"]
    labels = {k: v["label"] for k, v in get_candidates().items()}

    test_proba = pd.read_csv(config.REPORTS_DIR / "predicoes_teste.csv", index_col="indice")
    assert (test_proba.index == X_test.index).all(), "predições não alinhadas ao conjunto de teste"
    proba = pipeline.predict_proba(X_test)[:, 1]
    assert np.allclose(proba, test_proba[f"proba_{final_name}"]), "modelo salvo difere do avaliado no treino"

    fig_roc(test_proba, labels, final_name)
    fig_confusion(y_test, proba, threshold)
    curve = pd.read_csv(config.REPORTS_DIR / "curva_limiar_oof.csv")
    calib = fig_threshold_and_calibration(curve, threshold, y_test, proba)

    # importância por permutação no teste (queda de AUC ao embaralhar cada atributo original)
    perm = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc", n_repeats=50, random_state=config.SEED, n_jobs=-1
    )
    perm_df = pd.DataFrame(
        {"media": perm.importances_mean, "dp": perm.importances_std}, index=config.FEATURES
    ).sort_values("media", ascending=False)
    perm_df.to_csv(config.REPORTS_DIR / "importancia_permutacao.csv", index_label="atributo")

    # SHAP no espaço dos atributos originais
    background = pd.read_csv(config.BACKGROUND_PATH)
    if "n_jobs" in pipeline.named_steps["model"].get_params():
        # só acelera a predição em lote exigida pelo SHAP; não altera o modelo
        pipeline.named_steps["model"].set_params(n_jobs=-1)
    explainer = PipelineExplainer(pipeline, background)
    shap_values = explainer.shap_values(X_test)
    shap_df = pd.DataFrame(shap_values, columns=config.FEATURES, index=X_test.index)
    shap_df.to_csv(config.REPORTS_DIR / "shap_teste.csv", index_label="indice")
    shap_abs = shap_df.abs().mean().sort_values(ascending=False)
    # verificação de aditividade: valor base + soma das contribuições = probabilidade prevista
    recon = explainer.base_value + shap_values.sum(axis=1)
    additivity_err = float(np.abs(recon - proba).max())

    fig_importance(perm_df, shap_abs)
    fig_shap_dependence(X_test, shap_values)

    sub = subgroup_table(X_test, y_test, proba, threshold)
    sub.to_csv(config.REPORTS_DIR / "subgrupos_teste.csv", index=False)
    prof, errors = error_profile(X_test, y_test, proba, threshold)
    prof.to_csv(config.REPORTS_DIR / "perfil_erros.csv", index_label="grupo")
    errors.to_csv(config.REPORTS_DIR / "erros_teste.csv", index_label="indice")

    out = {
        "modelo_final": final_name,
        "limiar": threshold,
        "importancia_permutacao": {k: {"media": round(float(v["media"]), 4), "dp": round(float(v["dp"]), 4)}
                                   for k, v in perm_df.iterrows()},
        "shap_media_abs": {k: round(float(v), 4) for k, v in shap_abs.items()},
        "shap_valor_base": round(explainer.base_value, 4),
        "shap_erro_max_aditividade": additivity_err,
        "calibracao_teste": calib,
        "subgrupos": json.loads(sub.round(4).to_json(orient="records")),
        "perfil_erros": json.loads(prof.round(3).to_json(orient="index")),
    }
    with open(config.REPORTS_DIR / "resumo_avaliacao.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    pd.set_option("display.width", 200)
    print("Importância por permutação (queda de AUC):")
    print(perm_df.round(4).to_string())
    print("\nMédia |SHAP|:")
    print(shap_abs.round(4).to_string())
    print(f"\nErro máximo de aditividade do SHAP: {additivity_err:.2e}")
    print("\nSubgrupos (teste):")
    print(sub.round(3).to_string(index=False))
    print("\nPerfil dos erros (teste):")
    print(prof.round(2).T.to_string())


if __name__ == "__main__":
    main()
