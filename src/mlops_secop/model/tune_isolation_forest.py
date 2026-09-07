"""
Busqueda de hiperparametros con Optuna para el Isolation Forest de
deteccion de contratos atipicos, con tracking y registro en MLflow.

Este script:
1. Carga una MUESTRA de `features.parquet` (no el dataset completo) para
   la busqueda de hiperparametros. Esto es seguro porque IsolationForest
   ya subsamplea internamente por arbol (ver docstring de
   train_isolation_forest.py) -- la busqueda no necesita ver las 6.2M
   filas para converger a una buena combinacion de hiperparametros, y
   trabajar sobre una muestra hace cada trial mucho mas rapido. Lo que
   si escala con el tamano del dataset es el *scoring* (calcular el
   anomaly_score de cada fila), no el entrenamiento en si.
2. Genera un lote de contratos SINTETICOS deliberadamente atipicos, con
   tres patrones de perturbacion distintos (no uno solo), y los inyecta
   en la muestra -- formaliza como metrica automatica la validacion
   manual que ya se hizo para elegir n_estimators=500.
3. Para cada combinacion de hiperparametros que prueba Optuna, entrena
   un IsolationForest sobre la muestra + sinteticos, y mide que fraccion
   de los sinteticos queda en el 1% superior de anomaly_score
   (recall@top-1%). Optuna maximiza este valor.
4. Cada trial se registra en MLflow (run anidado) con sus parametros y
   su recall.

IMPORTANTE -- hallazgo del 7/9: se diagnostico que esta metrica de
recall sobre outliers sinteticos NO es confiable para este dataset --
incluso la configuracion ya validada manualmente por el equipo
(n_estimators=500, defaults) obtiene mal puntaje en ella (ver
diagnostico en el resumen del proyecto). Por eso `run_tuning()` se deja
documentado como el intento de busqueda automatica (util para mostrar
el proceso y el hallazgo en la sustentacion), pero el modelo que
realmente se entrena, trackea, explica y registra como final es el de
`run_baseline_final()`, que usa la configuracion validada a mano.

5. `explain_model()`: explica el modelo final con SHAP sobre una
   muestra pequena y dirigida (los contratos mas atipicos + un puñado
   de contratos normales), no sobre las 6.2M filas -- el objetivo es
   explicar por que el modelo marca ciertos contratos, no recalcular
   SHAP para todos.

Nota sobre MLflow y el Model Registry: el Model Registry de MLflow NO
funciona con el backend de archivos por defecto (`mlruns/`) -- requiere
un backend con base de datos. Por eso este script fija explicitamente
`sqlite:///mlflow.db` como tracking URI: sigue siendo 100% local (un
solo archivo SQLite en la raiz del repo), sin necesidad de levantar un
servidor de MLflow.

Ejecucion manual (PowerShell, con uv):
    uv run python -m mlops_secop.model.tune_isolation_forest

Para inspeccionar los resultados despues:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from scipy import sparse

from mlops_secop.model.train_isolation_forest import (
    CATEGORICAL_COLS,
    MODEL_DIR,
    NUMERIC_COLS,
    load_features,
    prepare_matrix,
    save_artifacts,
    score_contracts,
    train_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------------------------------------------------------------------
# Configuracion de la busqueda -- todos son parametros de run_tuning() con
# estos valores como default, para poder sobreescribirlos en los tests
# sin tocar estas constantes.
# --------------------------------------------------------------------------

#: Tamano de la muestra para la busqueda de hiperparametros.
SAMPLE_SIZE = 400_000

#: Cuantos contratos sinteticos se inyectan en cada trial.
N_SYNTHETIC_ANOMALIES = 150

#: Percentil superior de anomaly_score en el que se espera encontrar los
#: sinteticos.
TOP_PERCENTILE = 0.01

N_TRIALS = 40
RANDOM_STATE = 42

MLFLOW_EXPERIMENT_NAME = "isolation_forest_secop"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "isolation_forest_secop"

#: Configuracion ya validada manualmente por el equipo (ver
#: train_isolation_forest.py) -- es la que se usa en run_baseline_final().
BASELINE_PARAMS: dict = {
    "n_estimators": 500,
    "max_samples": "auto",
    "max_features": 1.0,
}


# --------------------------------------------------------------------------
# Contratos sinteticos -- formaliza la validacion manual de n_estimators
# --------------------------------------------------------------------------


def make_synthetic_anomalies(
    df: pd.DataFrame, n: int, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Genera `n` contratos sinteticos deliberadamente atipicos, a partir de
    filas reales tomadas al azar, perturbando sus columnas numericas con
    tres patrones distintos de anomalia (a partes iguales):

    - "valor_extremo": valor_contrato y valor_vs_promedio_categoria
      multiplicados por un factor grande -- simula sobrecosto.
    - "duracion_extrema": duracion_dias multiplicada por un factor
      grande -- simula un contrato anormalmente largo.
    - "concentracion_extrema": concentracion_proveedor forzada muy por
      encima del maximo observado -- simula un proveedor que concentra
      sospechosamente la contratacion.

    Usar tres patrones (en vez de uno solo, como en la validacion manual
    original con 20 casos) evita que la busqueda de hiperparametros
    sobreajuste a detectar un unico tipo de anomalia.
    """
    rng = np.random.default_rng(random_state)
    base_rows = df.sample(n=n, random_state=random_state, replace=False).reset_index(
        drop=True
    )

    patterns = np.array(["valor_extremo", "duracion_extrema", "concentracion_extrema"])
    assigned = rng.choice(patterns, size=n)

    synthetic = base_rows.copy()
    synthetic["numero_del_contrato"] = [f"SYNTH-{i:04d}" for i in range(n)]

    valor_mask = assigned == "valor_extremo"
    synthetic.loc[valor_mask, "valor_contrato"] *= 50.0
    synthetic.loc[valor_mask, "valor_vs_promedio_categoria"] *= 50.0

    duracion_mask = assigned == "duracion_extrema"
    synthetic.loc[duracion_mask, "duracion_dias"] *= 20.0

    concentracion_mask = assigned == "concentracion_extrema"
    max_concentracion = df["concentracion_proveedor"].max()
    synthetic.loc[concentracion_mask, "concentracion_proveedor"] = (
        max_concentracion * 5.0
    )

    synthetic["_synthetic_pattern"] = assigned
    return synthetic


def recall_at_top_percentile(
    scored: pd.DataFrame,
    synthetic_ids: set[str],
    top_percentile: float = TOP_PERCENTILE,
) -> float:
    """
    Fraccion de los contratos sinteticos que cae dentro del top
    `top_percentile` de anomaly_score sobre el conjunto evaluado.

    `scored` debe venir ordenado descendentemente por anomaly_score
    (asi lo devuelve score_contracts en train_isolation_forest.py).
    """
    n_top = max(1, int(len(scored) * top_percentile))
    top_ids = set(scored.head(n_top)["numero_del_contrato"])
    detected = len(synthetic_ids & top_ids)
    return detected / len(synthetic_ids)


# --------------------------------------------------------------------------
# Funcion objetivo de Optuna
# --------------------------------------------------------------------------


def make_objective(search_df: pd.DataFrame, synthetic_ids: set[str]):
    """
    Cierra sobre la muestra (ya con los sinteticos inyectados) para que
    Optuna solo tenga que llamar `objective(trial)`.

    El encoder se reajusta dentro de cada trial (no se comparte entre
    trials): evita estado mutable compartido si en el futuro se corre
    el study con paralelismo (`n_jobs>1`).
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_samples": trial.suggest_int("max_samples", 256, 4096, step=256),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }

        X, _encoder = prepare_matrix(search_df)
        model = train_model(
            X, contamination="auto", random_state=RANDOM_STATE, **params
        )

        scored = score_contracts(model, X, search_df)
        recall = recall_at_top_percentile(scored, synthetic_ids)

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("recall_at_top_1pct", recall)

        return recall

    return objective


# --------------------------------------------------------------------------
# Explicabilidad con SHAP -- solo sobre un puñado de contratos
# --------------------------------------------------------------------------


def explain_model(
    model,
    X: sparse.csr_matrix,
    scored: pd.DataFrame,
    feature_names: list[str],
    n_examples: int = 20,
) -> pd.DataFrame:
    """
    Explica el modelo final con SHAP sobre una muestra pequena y
    dirigida: los `n_examples` contratos MAS atipicos + `n_examples`
    contratos normales al azar -- no el dataset completo. TreeExplainer
    es rapido, pero calcular shap_values() sobre 6.2M filas seria
    trabajo desperdiciado: el objetivo es explicar por que el modelo
    marca ciertos contratos, no auditar todos.

    `scored.index` corresponde exactamente a las posiciones de fila en
    `X` (mismo orden que el DataFrame original antes de ordenar por
    anomaly_score), asi que se puede indexar X directamente con el.
    """
    import shap

    top_positions = scored.head(n_examples).index.to_numpy()
    normal_positions = (
        scored[~scored["is_anomaly"]]
        .sample(
            n=min(n_examples, int((~scored["is_anomaly"]).sum())),
            random_state=RANDOM_STATE,
        )
        .index.to_numpy()
    )
    sample_positions = np.concatenate([top_positions, normal_positions])
    X_sample = X[sample_positions].toarray()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)

    logger.info(
        "Variables mas influyentes segun SHAP (sobre %s atipicos + %s normales):\n%s",
        n_examples,
        len(normal_positions),
        importance.to_string(index=False),
    )
    return importance


# --------------------------------------------------------------------------
# Orquestacion -- busqueda exploratoria con Optuna (documentada, ver nota
# al inicio del modulo: no se usa su resultado como modelo final)
# --------------------------------------------------------------------------


def run_tuning(
    sample_size: int = SAMPLE_SIZE,
    n_synthetic: int = N_SYNTHETIC_ANOMALIES,
    n_trials: int = N_TRIALS,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    full_df = load_features()
    sample_df = full_df.sample(
        n=min(sample_size, len(full_df)), random_state=RANDOM_STATE
    )

    synthetic_df = make_synthetic_anomalies(sample_df, n_synthetic, RANDOM_STATE)
    synthetic_ids = set(synthetic_df["numero_del_contrato"])
    search_df = pd.concat([sample_df, synthetic_df], ignore_index=True)

    with mlflow.start_run(run_name="optuna_search_exploratorio"):
        mlflow.log_param("sample_size", len(sample_df))
        mlflow.log_param("n_synthetic_anomalies", n_synthetic)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.enqueue_trial(
            {"n_estimators": 500, "max_samples": 4096, "max_features": 1.0}
        )
        study.optimize(make_objective(search_df, synthetic_ids), n_trials=n_trials)

        best_params = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_recall_at_top_1pct", study.best_value)

    logger.info("Mejores hiperparametros encontrados: %s", best_params)
    logger.info("Recall en el top 1%% (muestra): %.2f", study.best_value)
    logger.info(
        "NOTA: este resultado es exploratorio. El modelo final se entrena "
        "con run_baseline_final(), no con estos hiperparametros -- ver "
        "docstring del modulo."
    )

    return {"best_params": best_params, "best_recall_sample": study.best_value}


# --------------------------------------------------------------------------
# Modelo final -- usa la configuracion ya validada manualmente por el
# equipo (ver nota al inicio del modulo)
# --------------------------------------------------------------------------


def run_baseline_final(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    register_model: bool = True,
    explain: bool = True,
) -> dict:
    """
    Entrena, trackea, explica y registra el modelo BASE ya validado
    manualmente (n_estimators=500, resto en default) -- no el resultado
    de la busqueda de Optuna. Ver docstring del modulo / registro del
    proyecto: la busqueda automatica con outliers sinteticos no resulto
    ser una metrica confiable para este dataset (el propio baseline
    obtiene mal puntaje en ella), asi que se usa el modelo validado
    manualmente como el modelo final de esta iteracion.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    full_df = load_features()

    with mlflow.start_run(run_name="final_model_baseline_validado"):
        mlflow.log_params(BASELINE_PARAMS)

        X_full, encoder_full = prepare_matrix(full_df)
        final_model = train_model(
            X_full, contamination="auto", random_state=RANDOM_STATE, **BASELINE_PARAMS
        )

        scored_full = score_contracts(final_model, X_full, full_df)
        n_anomalies = int(scored_full["is_anomaly"].sum())

        mlflow.log_metric("total_contratos", len(full_df))
        mlflow.log_metric("n_anomalias_detectadas", n_anomalies)
        mlflow.log_metric("pct_anomalias", 100 * n_anomalies / len(full_df))

        save_artifacts(final_model, encoder_full)

        if register_model:
            mlflow.sklearn.log_model(
                final_model,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME,
            )
        else:
            mlflow.sklearn.log_model(final_model, artifact_path="model")

        if explain:
            feature_names = NUMERIC_COLS + list(
                encoder_full.get_feature_names_out(CATEGORICAL_COLS)
            )
            importance = explain_model(final_model, X_full, scored_full, feature_names)
            importance_path = MODEL_DIR / "shap_feature_importance.csv"
            importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path))

    summary = {
        "params": BASELINE_PARAMS,
        "total_contratos": len(full_df),
        "n_anomalias_detectadas": n_anomalies,
    }
    logger.info("Resumen del modelo final (baseline validado): %s", summary)
    return summary


if __name__ == "__main__":
    run_baseline_final()
