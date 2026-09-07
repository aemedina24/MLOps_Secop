"""
Entrenamiento del modelo Isolation Forest para detección de contratos
atípicos, sobre `data/processed/secop_ii/features.parquet`.

Este script:
1. Carga `features.parquet`, seleccionando explícitamente columnas (no
   `SELECT *`), igual que en `build_features.py` -- mismo motivo:
   máquina de desarrollo con RAM limitada (~6GB).
2. Separa tres grupos de columnas:
   - Identificadores (`numero_del_contrato`, `url_contrato`): NUNCA
     entran al modelo, se conservan aparte para poder rastrear un
     contrato real cuando se marque como atípico.
   - Numéricas: entran directo, sin escalar (Isolation Forest es
     basado en árboles de decisión, insensible a la escala de las
     variables -- a diferencia de modelos basados en distancia como
     k-NN o SVM, donde el escalado sería obligatorio).
   - Categóricas: se codifican con one-hot encoding disperso
     (`scipy.sparse`), NO con `pd.get_dummies` (que produce un
     DataFrame denso). Con ~123 columnas dummy estimadas sobre 6.2M
     filas, una matriz densa ocuparía varios GB solo en ceros; la
     versión dispersa solo almacena las posiciones con un 1.
3. Combina numéricas + categóricas en una única matriz dispersa
   (`scipy.sparse.hstack`) -- IsolationForest de scikit-learn acepta
   matrices dispersas (CSR/CSC) directamente como entrada.
4. Entrena el modelo con hiperparámetros por defecto razonables (sin
   Optuna todavía -- ese ajuste fino viene después de confirmar que el
   pipeline básico entrena y produce resultados con sentido).
5. Calcula el `anomaly_score` de cada contrato y lo une de vuelta a los
   identificadores, para poder inspeccionar manualmente los contratos
   más atípicos como validación (mismo patrón que usamos para validar
   `valor_vs_promedio_categoria` en build_features.py).
6. Guarda el modelo entrenado y el encoder (ambos necesarios para poder
   aplicar el modelo a datos nuevos más adelante, ej. en inferencia).

Por qué IsolationForest tolera bien el volumen completo (6.2M filas)
sin submuestrear: cada árbol del bosque ya subsample internamente
(parámetro `max_samples`, por defecto min(256, n_filas) POR ÁRBOL), así
que el costo de entrenamiento no depende linealmente del tamaño total
del dataset -- es la codificación one-hot (paso 2-3), no el
entrenamiento en sí, lo que representaba el riesgo real de memoria.

Ejecución manual (PowerShell, con uv):
    uv run python -m mlops_secop.model.train_isolation_forest
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

from mlops_secop.config import FEATURES_OUTPUT_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Definición de columnas -- refleja el esquema real de salida de
# build_features.py (nombres YA renombrados vía COLUMN_RENAME_MAP donde
# aplica, ej. modalidad_de_contratacion en vez de
# modalidad_de_contrataci_n).
# --------------------------------------------------------------------------

#: Columnas de solo referencia -- nunca entran al modelo. Debe coincidir
#: con las que build_features.py incluye como identificadores en su
#: propio IDENTIFIER_COLS (no se importa directamente para evitar
#: acoplar train_isolation_forest.py a la estructura interna de
#: build_features.py, pero deben mantenerse sincronizadas a mano).
#: `objeto_a_contratar` viaja como referencia (no como feature) para
#: que un auditor pueda distinguir items reales distintos bajo el mismo
#: numero_del_contrato -- ver docstring de build_features.IDENTIFIER_COLS.
IDENTIFIER_COLS: list[str] = [
    "numero_del_contrato",
    "url_contrato",
    "objeto_a_contratar",
]
NUMERIC_COLS: list[str] = [
    "valor_contrato",
    "valor_vs_promedio_categoria",
    "duracion_dias",
    "mes_firma",
    "dia_semana_firma",
    "concentracion_proveedor",
]

CATEGORICAL_COLS: list[str] = [
    "modalidad_de_contratacion",
    "departamento_entidad",
    "nivel_entidad",
    "tipo_de_contrato",
    "origen",
    "tipo_documento_proveedor",
]

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
ENCODER_PATH = MODEL_DIR / "onehot_encoder.joblib"

# Hiperparámetros por defecto -- punto de partida razonable, antes de
# afinar con Optuna en una iteración posterior. n_estimators=500 (no
# el default de scikit-learn, que es 100) se eligió tras validar con
# outliers sintéticos deliberados: con ~99 columnas categóricas
# dispersas (one-hot) frente a solo 6 numéricas, IsolationForest
# selecciona features al azar en cada división del árbol, así que la
# mayoría de las divisiones caen sobre columnas categóricas poco
# informativas para el patrón "valor atípico" -- diluyendo la señal.
# Con 100 árboles, solo 1 de 20 outliers deliberados quedaba en el
# top 100 de resultados; con 500, subió a 10 de 20. n_estimators debe
# incluirse en el espacio de búsqueda de Optuna en la siguiente
# iteración, en vez de quedar fijo aquí.
DEFAULT_N_ESTIMATORS = 500
DEFAULT_CONTAMINATION = "auto"
DEFAULT_RANDOM_STATE = 42


def load_features(path: str | None = None) -> pd.DataFrame:
    """
    Carga features.parquet, seleccionando explícitamente solo las
    columnas necesarias (identificadores + numéricas + categóricas) --
    mismo motivo de memoria limitada que en build_features.py.

    `path` NO usa FEATURES_OUTPUT_PATH como valor por defecto del
    parámetro -- los valores por defecto se evalúan una sola vez, en el
    momento en que se define la función, así que quedarían "congelados"
    con el valor que tenía FEATURES_OUTPUT_PATH al importar el módulo.
    Esto rompería monkeypatch.setattr(...) en los tests (el mismo
    problema que ya resolvió config.py usando funciones en vez de
    constantes de módulo para las rutas configurables por entorno).
    Referenciar la constante DENTRO del cuerpo de la función sí permite
    que los tests la sobreescriban correctamente.
    """
    if path is None:
        path = FEATURES_OUTPUT_PATH
    columns = IDENTIFIER_COLS + NUMERIC_COLS + CATEGORICAL_COLS
    df = pd.read_parquet(path, columns=columns)
    logger.info("Cargadas %s filas de %s", len(df), path)
    return df


def prepare_matrix(
    df: pd.DataFrame, encoder: OneHotEncoder | None = None
) -> tuple[sparse.csr_matrix, OneHotEncoder]:
    """
    Construye la matriz dispersa de entrada al modelo.

    Si `encoder` es None, se ajusta uno nuevo (entrenamiento). Si se
    pasa un encoder ya ajustado, se reutiliza (inferencia sobre datos
    nuevos) -- así esta función sirve tanto para entrenar como para
    puntuar contratos futuros con el mismo encoding.
    """
    numeric_matrix = sparse.csr_matrix(df[NUMERIC_COLS].to_numpy(dtype=float))

    if encoder is None:
        encoder = OneHotEncoder(
            sparse_output=True, handle_unknown="ignore", dtype="float64"
        )
        categorical_matrix = encoder.fit_transform(df[CATEGORICAL_COLS])
    else:
        categorical_matrix = encoder.transform(df[CATEGORICAL_COLS])

    X = sparse.hstack([numeric_matrix, categorical_matrix], format="csr")
    logger.info(
        "Matriz de entrada: %s filas x %s columnas (%s numéricas + %s "
        "categóricas codificadas), %.1f%% de densidad",
        X.shape[0],
        X.shape[1],
        len(NUMERIC_COLS),
        categorical_matrix.shape[1],
        100 * X.nnz / (X.shape[0] * X.shape[1]),
    )
    return X, encoder


def train_model(
    X: sparse.csr_matrix,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    contamination: str | float = DEFAULT_CONTAMINATION,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def score_contracts(
    model: IsolationForest, X: sparse.csr_matrix, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcula el anomaly_score de cada contrato y lo une con sus
    identificadores para poder inspeccionar los resultados.

    decision_function: valores más NEGATIVOS = más atípico. Se invierte
    el signo (anomaly_score = -decision_function) para que "más alto =
    más atípico", más intuitivo para un auditor no técnico.
    """
    raw_scores = model.decision_function(X)
    result = df[IDENTIFIER_COLS].copy()
    result["anomaly_score"] = -raw_scores
    result["is_anomaly"] = model.predict(X) == -1
    return result.sort_values("anomaly_score", ascending=False)


def save_artifacts(model: IsolationForest, encoder: OneHotEncoder) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    logger.info("Modelo guardado en %s", MODEL_PATH)
    logger.info("Encoder guardado en %s", ENCODER_PATH)


def run_training() -> dict:
    df = load_features()
    X, encoder = prepare_matrix(df)

    logger.info("Entrenando IsolationForest...")
    model = train_model(X)

    scored = score_contracts(model, X, df)
    save_artifacts(model, encoder)

    n_anomalies = int(scored["is_anomaly"].sum())
    summary = {
        "total_contratos": len(df),
        "n_anomalias_detectadas": n_anomalies,
        "pct_anomalias": round(100 * n_anomalies / len(df), 4),
        "model_path": str(MODEL_PATH),
        "encoder_path": str(ENCODER_PATH),
    }
    logger.info("Resumen de entrenamiento: %s", summary)
    logger.info(
        "Top 10 contratos mas atipicos:\n%s",
        scored.head(10).to_string(index=False),
    )
    return summary


if __name__ == "__main__":
    run_training()
