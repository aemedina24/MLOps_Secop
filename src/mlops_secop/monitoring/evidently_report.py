"""
Reporte de drift de datos con Evidently, comparando contratos firmados
en 2023 (referencia) contra contratos firmados en 2025-2026 (actual),
sobre las columnas numéricas y categóricas usadas por el modelo.

Por qué esta comparación específica: valida empíricamente una
limitación metodológica ya documentada del pipeline -- `valor_contrato`
no se ajusta por inflación (IPC), y `valor_vs_promedio_categoria` se
calcula sobre todo el histórico 2022-2026 mezclado en pesos nominales.
Es ESPERABLE ver drift real en variables monetarias por esta razón
conocida, no una sorpresa ni un error del pipeline -- este reporte
sirve como evidencia documentada de esa limitación para la sección de
gobernanza, no como una alerta de que "algo se rompió".

Columnas incluidas: las 6 numéricas + las 6 categóricas que consume
`train_isolation_forest.py` (mismas listas, para que el reporte de
drift hable exactamente de lo que el modelo ve). Se excluyen las
columnas de identificador (numero_del_contrato, url_contrato,
objeto_a_contratar) -- no aportan a un análisis de drift estadístico,
son de solo referencia.

Ejecución manual (PowerShell, con uv):
    uv run python -m mlops_secop.monitoring.evidently_report
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from mlops_secop.config import FEATURES_OUTPUT_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Mismas columnas que usa el modelo (train_isolation_forest.py) -- el
# reporte de drift debe hablar exactamente de lo que el modelo consume,
# no de un subconjunto distinto elegido aparte.
# --------------------------------------------------------------------------

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

DATE_COL_FOR_SPLIT = "fecha_de_firma_del_contrato"

REFERENCE_YEAR = 2023
CURRENT_YEARS = (2025, 2026)

#: Proporción de columnas que deben mostrar drift individual para
#: considerar que el DATASET completo tiene drift -- no basta con que
#: una sola columna drifte. 0.5 replica el default de Evidently.
DRIFT_SHARE_THRESHOLD = 0.5

REPORT_OUTPUT_PATH = Path("reports/data_drift_2023_vs_2025_2026.html")


def load_split_by_year(path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga features.parquet y separa dos subconjuntos por año de
    fecha_de_firma_del_contrato: referencia (2023) y actual (2025-2026).

    `path` no usa FEATURES_OUTPUT_PATH como valor por defecto del
    parámetro -- los defaults se evalúan una sola vez al definir la
    función, lo que rompería monkeypatch.setattr(...) en los tests
    (mismo problema ya resuelto en train_isolation_forest.load_features).
    """
    if path is None:
        path = FEATURES_OUTPUT_PATH

    columns = NUMERIC_COLS + CATEGORICAL_COLS + [DATE_COL_FOR_SPLIT]
    df = pd.read_parquet(path, columns=columns)

    year = pd.to_datetime(df[DATE_COL_FOR_SPLIT]).dt.year
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS

    reference = df.loc[year == REFERENCE_YEAR, feature_cols].reset_index(drop=True)
    current = df.loc[year.isin(CURRENT_YEARS), feature_cols].reset_index(drop=True)

    logger.info(
        "Referencia (%s): %s filas | Actual (%s): %s filas",
        REFERENCE_YEAR,
        len(reference),
        CURRENT_YEARS,
        len(current),
    )
    return reference, current


def _column_is_drifted(metric_entry: dict) -> bool:
    """
    Determina si una columna tiene drift, interpretando correctamente
    la DIRECCIÓN del valor según el método estadístico que Evidently
    eligió para ella.

    Esto es necesario porque Evidently selecciona automáticamente el
    método según el tamaño de la muestra: con datasets grandes usa
    distancias (Wasserstein, Jensen-Shannon), donde valores MÁS ALTOS
    significan MÁS drift; con datasets pequeños usa tests de hipótesis
    (Kolmogorov-Smirnov, chi-cuadrado, Z-test), que reportan p-valores,
    donde valores MÁS BAJOS significan MÁS drift (se rechaza la
    hipótesis de que ambas distribuciones son iguales). Confirmado
    empíricamente: el mismo par de datasets, con distinto tamaño de
    muestra, usó métodos distintos con direcciones opuestas -- una
    primera versión de este código asumía "mayor = más drift" siempre,
    lo que habría invertido silenciosamente el resultado en datasets
    pequeños (como los que usan estos mismos tests).
    """
    method = metric_entry["config"].get("method", "")
    value = metric_entry["value"]
    threshold = metric_entry["config"].get("threshold", 0.1)

    is_p_value_method = "p_value" in method.lower() or "p-value" in method.lower()
    if is_p_value_method:
        return value < threshold
    return value > threshold


def generate_drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    """
    Genera el snapshot de Evidently. `current_data` va primero,
    `reference_data` segundo -- orden que exige la API de Evidently
    (fácil de confundir: el primer argumento posicional es el dataset
    que se evalúa, no el punto de comparación).
    """
    report = Report([DataDriftPreset()])
    return report.run(current_data=current, reference_data=reference)


def run_drift_report(output_path: Path | None = None) -> dict:
    if output_path is None:
        output_path = REPORT_OUTPUT_PATH

    reference, current = load_split_by_year()

    if len(reference) == 0 or len(current) == 0:
        raise ValueError(
            f"Uno de los dos grupos quedó vacío (referencia={len(reference)}, "
            f"actual={len(current)}) -- revisa REFERENCE_YEAR/CURRENT_YEARS "
            f"contra las fechas reales disponibles en features.parquet."
        )

    snapshot = generate_drift_report(reference, current)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))
    logger.info("Reporte HTML guardado en %s", output_path)

    result_dict = snapshot.dict()

    # Se calcula "drifted" columna por columna con _column_is_drifted
    # (que respeta la dirección correcta del método) en vez de usar el
    # conteo agregado de Evidently (metrics[0], DriftedColumnsCount):
    # ese conteo usa su propia lógica interna no siempre documentada
    # con el mismo detalle, y preferimos que nuestro n_drifted_columns
    # y drifted_columns sean mutuamente consistentes y auditables.
    per_column_entries = [
        m for m in result_dict["metrics"][1:] if "column" in m.get("config", {})
    ]
    per_column_drift = {m["config"]["column"]: m["value"] for m in per_column_entries}
    drifted_columns = [
        m["config"]["column"] for m in per_column_entries if _column_is_drifted(m)
    ]
    n_total = len(per_column_entries)
    drifted_share = len(drifted_columns) / n_total if n_total else 0.0

    summary = {
        "reference_rows": len(reference),
        "current_rows": len(current),
        "n_columns_evaluated": n_total,
        "n_drifted_columns": len(drifted_columns),
        "drifted_share": drifted_share,
        "dataset_drift_detected": drifted_share >= DRIFT_SHARE_THRESHOLD,
        "drifted_columns": drifted_columns,
        "per_column_drift_score": per_column_drift,
        "report_path": str(output_path),
    }
    logger.info("Resumen de drift: %s", summary)
    return summary


if __name__ == "__main__":
    run_drift_report()
