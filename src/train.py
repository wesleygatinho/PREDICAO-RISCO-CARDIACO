"""Treino, comparação e persistência dos modelos.

Uso:  python -m src.train

Protocolo
---------
1. Hold-out estratificado 80/20 com semente fixa; o teste fica intocado até o fim.
2. No treino, busca exaustiva de hiperparâmetros (GridSearchCV) com validação
   cruzada estratificada 5x3 (5 dobras, 3 repetições). O pré-processamento está
   dentro do Pipeline, logo é reajustado em cada dobra de treino (sem vazamento).
3. O modelo final é o de maior AUC-ROC média na validação cruzada.
4. O limiar de decisão do modelo final é escolhido nas predições out-of-fold do
   treino, maximizando F2 (revocação pesa o dobro da precisão).
5. Só então todos os modelos são avaliados uma única vez no conjunto de teste.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

from . import config
from .data import load_data, split_data
from .models import get_candidates
from .preprocessing import build_pipeline

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


# --------------------------------------------------------------------------- métricas
def metrics_at(y_true, proba, threshold: float) -> dict:
    """Métricas de classificação binária para um dado limiar de decisão."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "limiar": float(threshold),
        "acuracia": float(accuracy_score(y_true, pred)),
        "precisao": float(precision_score(y_true, pred, zero_division=0)),
        "revocacao": float(recall_score(y_true, pred, zero_division=0)),
        "especificidade": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp),
    }


def choose_threshold(y_true, proba, beta: float = config.THRESHOLD_BETA) -> tuple[float, pd.DataFrame]:
    """Limiar que maximiza F-beta nas predições out-of-fold do treino."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "limiar": t,
                "precisao": precision_score(y_true, pred, zero_division=0),
                "revocacao": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "fbeta": fbeta_score(y_true, pred, beta=beta, zero_division=0),
            }
        )
    curve = pd.DataFrame(rows)
    best = curve["fbeta"].max()
    # em caso de empate, o maior limiar (menos falsos positivos)
    threshold = float(curve.loc[np.isclose(curve["fbeta"], best), "limiar"].max())
    return threshold, curve


def corrected_resampled_ttest(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int) -> tuple[float, float]:
    """Teste t reamostrado corrigido (Nadeau e Bengio, 2003) para CV repetida.

    Corrige a variância pela sobreposição dos conjuntos de treino entre dobras.
    Retorna (estatística t, p-valor bilateral).
    """
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / n + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return float(t), float(p)


def bootstrap_ci(y_true, proba, threshold: float, n_boot: int = 2000, seed: int = config.SEED) -> dict:
    """Intervalo de confiança de 95% (percentil) por bootstrap do conjunto de teste."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    keys = ["acuracia", "precisao", "revocacao", "especificidade", "f1", "f2", "auc_roc"]
    samples = {k: [] for k in keys}
    n = len(y_true)
    while len(samples["f1"]) < n_boot:
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = metrics_at(y_true[idx], proba[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()
    }


def sha256_of(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- execução
def main() -> None:
    t0 = time.time()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    search_dir = config.REPORTS_DIR / "busca_hiperparametros"
    search_dir.mkdir(exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)} | semente: {config.SEED}")

    cv = RepeatedStratifiedKFold(
        n_splits=config.CV_SPLITS, n_repeats=config.CV_REPEATS, random_state=config.SEED
    )
    n_cv = config.CV_SPLITS * config.CV_REPEATS
    n_fold_test = len(X_train) // config.CV_SPLITS
    n_fold_train = len(X_train) - n_fold_test

    candidates = get_candidates()
    fitted, cv_rows, test_rows, fold_auc, best_params = {}, [], [], {}, {}
    test_proba = pd.DataFrame({"y_true": y_test}, index=X_test.index)

    for name, spec in candidates.items():
        t1 = time.time()
        search = GridSearchCV(
            build_pipeline(spec["estimator"]),
            param_grid=spec["grid"],
            scoring=SCORING,
            refit=config.SELECTION_METRIC,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        res = pd.DataFrame(search.cv_results_)
        i = search.best_index_

        # busca documentada: uma linha por configuração avaliada
        keep = [c for c in res.columns if c.startswith("param_")] + [
            f"mean_test_{m}" for m in SCORING
        ] + [f"std_test_{m}" for m in SCORING] + ["mean_train_roc_auc", "rank_test_roc_auc"]
        res[keep].sort_values("rank_test_roc_auc").to_csv(search_dir / f"{name}.csv", index=False)

        row = {"modelo": name, "rotulo": spec["label"], "familia": spec["family"],
               "n_configuracoes": int(len(res))}
        for m in SCORING:
            row[f"{m}_media"] = float(res.loc[i, f"mean_test_{m}"])
            row[f"{m}_dp"] = float(res.loc[i, f"std_test_{m}"])
        row["roc_auc_treino_media"] = float(res.loc[i, "mean_train_roc_auc"])
        cv_rows.append(row)
        fold_auc[name] = np.array([res.loc[i, f"split{k}_test_roc_auc"] for k in range(n_cv)])
        best_params[name] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        fitted[name] = search.best_estimator_

        proba = search.best_estimator_.predict_proba(X_test)[:, 1]
        test_proba[f"proba_{name}"] = proba
        test_rows.append({"modelo": name, "rotulo": spec["label"], **metrics_at(y_test, proba, 0.5)})
        print(
            f"[{name:7s}] {len(res):3d} configs | AUC CV {row['roc_auc_media']:.4f} ± {row['roc_auc_dp']:.4f}"
            f" | F1 CV {row['f1_media']:.4f} | {time.time() - t1:5.1f}s | {best_params[name]}"
        )

    cv_table = pd.DataFrame(cv_rows)
    test_table = pd.DataFrame(test_rows)

    # ---------------------------------------------------------------- escolha do modelo final
    final_name = cv_table.sort_values("roc_auc_media", ascending=False).iloc[0]["modelo"]
    final = fitted[final_name]
    print(f"\nModelo final (maior AUC-ROC média na CV): {final_name}")

    comparisons = {}
    for name in candidates:
        if name == final_name:
            continue
        t, p = corrected_resampled_ttest(fold_auc[final_name], fold_auc[name], n_fold_train, n_fold_test)
        comparisons[name] = {
            "diferenca_media_auc": float((fold_auc[final_name] - fold_auc[name]).mean()),
            "t": t, "p_valor": p,
        }

    # ---------------------------------------------------------------- limiar de decisão (só treino)
    oof_cv = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.SEED)
    oof = cross_val_predict(clone(final), X_train, y_train, cv=oof_cv, method="predict_proba", n_jobs=-1)[:, 1]
    threshold, curve = choose_threshold(y_train.to_numpy(), oof)
    curve.to_csv(config.REPORTS_DIR / "curva_limiar_oof.csv", index=False)
    pd.DataFrame({"y_true": y_train, "proba_oof": oof}, index=X_train.index).to_csv(
        config.REPORTS_DIR / "oof_treino_modelo_final.csv", index_label="indice"
    )
    oof_metrics = metrics_at(y_train, oof, threshold)
    print(f"Limiar escolhido (max F2 out-of-fold): {threshold:.2f}")

    # ---------------------------------------------------------------- avaliação final no teste
    final_proba = test_proba[f"proba_{final_name}"].to_numpy()
    final_050 = metrics_at(y_test, final_proba, 0.5)
    final_thr = metrics_at(y_test, final_proba, threshold)
    ci = bootstrap_ci(y_test, final_proba, threshold)
    ci_050 = bootstrap_ci(y_test, final_proba, 0.5)

    # ---------------------------------------------------------------- sensibilidade: indicador de ausência
    sensitivity = {}
    for name in dict.fromkeys(["logreg", final_name]):
        est = clone(candidates[name]["estimator"]).set_params(**best_params[name])
        scores = cross_val_score(
            build_pipeline(est, add_missing_indicator=True), X_train, y_train,
            scoring="roc_auc", cv=cv, n_jobs=-1,
        )
        sensitivity[name] = {
            "auc_sem_indicador_media": float(fold_auc[name].mean()),
            "auc_sem_indicador_dp": float(fold_auc[name].std()),
            "auc_com_indicador_media": float(scores.mean()),
            "auc_com_indicador_dp": float(scores.std()),
        }

    # ---------------------------------------------------------------- persistência
    joblib.dump(final, config.MODEL_PATH)
    background = (
        X_train.assign(_y=y_train)
        .groupby("_y", group_keys=False)
        .sample(n=25, random_state=config.SEED)
        .drop(columns="_y")
    )
    background.to_csv(config.BACKGROUND_PATH, index=False)

    cv_table.to_csv(config.REPORTS_DIR / "resultados_cv.csv", index=False)
    test_table.to_csv(config.REPORTS_DIR / "resultados_teste.csv", index=False)
    test_proba.to_csv(config.REPORTS_DIR / "predicoes_teste.csv", index_label="indice")

    metadata = {
        "modelo": final_name,
        "rotulo": candidates[final_name]["label"],
        "familia": candidates[final_name]["family"],
        "hiperparametros": best_params[final_name],
        "limiar": threshold,
        "criterio_limiar": "máximo F2 nas predições out-of-fold do treino",
        "criterio_selecao": "maior AUC-ROC média na validação cruzada estratificada 5x3",
        "atributos": config.FEATURES,
        "alvo": config.TARGET,
        "semente": config.SEED,
        "n_treino": int(len(X_train)),
        "n_teste": int(len(X_test)),
        "prevalencia_treino": float(y_train.mean()),
        "metricas_teste_limiar_escolhido": final_thr,
        "metricas_teste_limiar_050": final_050,
        "ic95_teste_limiar_escolhido": ci,
        "treinado_em": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_sha256": sha256_of(config.DATA_PATH),
        "versoes": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    summary = {
        "protocolo": {
            "semente": config.SEED,
            "teste": config.TEST_SIZE,
            "cv": f"{config.CV_SPLITS} dobras x {config.CV_REPEATS} repetições",
            "metrica_selecao": config.SELECTION_METRIC,
            "n_treino": int(len(X_train)), "n_teste": int(len(X_test)),
            "positivos_treino": int(y_train.sum()), "positivos_teste": int(y_test.sum()),
        },
        "melhores_hiperparametros": best_params,
        "cv": cv_rows,
        "teste_limiar_050": test_rows,
        "modelo_final": final_name,
        "comparacao_auc_cv_final_vs_outros": comparisons,
        "limiar": threshold,
        "oof_treino_limiar_escolhido": oof_metrics,
        "final_teste_limiar_050": final_050,
        "final_teste_limiar_escolhido": final_thr,
        "final_teste_ic95_limiar_escolhido": ci,
        "final_teste_ic95_limiar_050": ci_050,
        "sensibilidade_indicador_ausencia": sensitivity,
        "tempo_total_s": round(time.time() - t0, 1),
    }
    with open(config.REPORTS_DIR / "resumo_experimentos.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\nTeste (limiar 0,50):")
    print(
        test_table[["rotulo", "acuracia", "precisao", "revocacao", "f1", "auc_roc"]]
        .round(4).to_string(index=False)
    )
    print(f"\nModelo final no teste com limiar {threshold:.2f}: "
          + ", ".join(f"{k}={final_thr[k]:.4f}" for k in ["precisao", "revocacao", "f1", "f2", "auc_roc"]))
    print(f"Matriz de confusão: VN={final_thr['vn']} FP={final_thr['fp']} FN={final_thr['fn']} VP={final_thr['vp']}")
    print(f"Sensibilidade (indicador de ausência): {json.dumps(sensitivity, indent=1)}")
    print(f"\nModelo salvo em {config.MODEL_PATH}")
    print(f"Tempo total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
