"""
Construcción de features de "alerta" para el modelo de detección de
contratos atípicos (Isolation Forest), a partir de
`data/processed/secop_ii/contracts.parquet`.

Este script:
1. Filtra los contratos según las reglas de negocio de `config.py`
   (`ESTADOS_VALIDOS`, `VALOR_MIN`/`VALOR_MAX`).
2. Convierte `valor_contrato` a numérico y las tres columnas de fecha
   (`fecha_de_firma_del_contrato`, `fecha_inicio_ejecuci_n`,
   `fecha_fin_ejecuci_n`) a tipo `DATE` real — en `contracts.parquet`
   llegan como texto (`VARCHAR`), heredado deliberadamente de la capa
   RAW, que nunca transforma valores.
3. Calcula las features de alerta:
   - `valor_vs_promedio_categoria`: razón entre el valor del contrato y
     el promedio de su categoría (`CATEGORIA_COLS` en config.py). Un
     valor de 3.0 significa "este contrato vale 3 veces el promedio de
     contratos con su misma modalidad y departamento".
   - `duracion_dias`: días entre inicio y fin de ejecución.
   - `concentracion_proveedor`: cuántos contratos tiene ese mismo
     proveedor (`documento_proveedor`) en todo el histórico filtrado.
   - `mes_firma`, `dia_semana_firma`: para detectar estacionalidad o
     patrones de fechas atípicas (ej. concentración de firmas a fin de
     año o fin de semana).
4. Renombra las columnas truncadas por Socrata según
   `COLUMN_RENAME_MAP` (solo en este dataset derivado, no en RAW ni en
   `contracts.parquet`).
5. Guarda el resultado en `data/processed/secop_ii/features.parquet`.

Por qué DuckDB (consistente con process_secop.py): el volumen filtrado
sigue siendo del orden de millones de filas (~6.2M en la última
verificación), así que se mantiene el mismo enfoque out-of-core que ya
se usa en el resto del pipeline, evitando el MemoryError que motivó esa
decisión originalmente.

Ejecución manual (PowerShell, con uv):
    uv run python -m mlops_secop.features.build_features
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from mlops_secop.config import (
    CATEGORIA_COLS,
    COLUMN_RENAME_MAP,
    CONTRACTS_PARQUET_PATH,
    ESTADOS_VALIDOS,
    FEATURES_OUTPUT_PATH,
    VALOR_MAX,
    VALOR_MIN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _estados_sql_list() -> str:
    """Formatea ESTADOS_VALIDOS como lista SQL de strings entre comillas."""
    return ", ".join(f"'{estado}'" for estado in ESTADOS_VALIDOS)


def _categoria_partition_by() -> str:
    """
    Columnas de CATEGORIA_COLS formateadas para una cláusula
    PARTITION BY / GROUP BY.
    """
    return ", ".join(CATEGORIA_COLS)


def _base_filtered_cte() -> str:
    """
    CTE (Common Table Expression) con el filtro de negocio y las
    conversiones de tipo aplicadas una sola vez, reutilizado por el resto
    de la consulta.

    LOWER() en el filtro de estado es obligatorio: el dataset real tiene
    inconsistencias de mayúsculas/minúsculas confirmadas empíricamente
    (ver config.py).
    """
    return f"""
        base AS (
            SELECT
                *,
                TRY_CAST(valor_contrato AS DOUBLE) AS valor_contrato_num,
                TRY_CAST(fecha_de_firma_del_contrato AS DATE)
                    AS fecha_de_firma_del_contrato_dt,
                TRY_CAST(fecha_inicio_ejecuci_n AS DATE)
                    AS fecha_inicio_ejecuci_n_dt,
                TRY_CAST(fecha_fin_ejecuci_n AS DATE)
                    AS fecha_fin_ejecuci_n_dt
            FROM read_parquet('{CONTRACTS_PARQUET_PATH}')
            WHERE LOWER(estado_del_proceso) IN ({_estados_sql_list()})
              AND TRY_CAST(valor_contrato AS DOUBLE)
                  BETWEEN {VALOR_MIN} AND {VALOR_MAX}
        )
    """


def _build_query() -> str:
    """
    Construye la consulta completa de features.

    Usa una ventana (AVG(...) OVER (PARTITION BY ...)) en vez de un JOIN
    manual contra una tabla de promedios: es más simple de leer y DuckDB
    la optimiza igual de bien para este volumen de datos.
    """
    partition_by = _categoria_partition_by()

    return f"""
        WITH {_base_filtered_cte()},
        con_concentracion AS (
            SELECT
                *,
                COUNT(*) OVER (PARTITION BY documento_proveedor)
                    AS concentracion_proveedor,
                AVG(valor_contrato_num) OVER (PARTITION BY {partition_by})
                    AS promedio_categoria
            FROM base
        )
        SELECT
            * EXCLUDE (
                valor_contrato, valor_contrato_num,
                fecha_de_firma_del_contrato, fecha_de_firma_del_contrato_dt,
                fecha_inicio_ejecuci_n, fecha_inicio_ejecuci_n_dt,
                fecha_fin_ejecuci_n, fecha_fin_ejecuci_n_dt,
                promedio_categoria
            ),
            valor_contrato_num AS valor_contrato,
            fecha_de_firma_del_contrato_dt AS fecha_de_firma_del_contrato,
            fecha_inicio_ejecuci_n_dt AS fecha_inicio_ejecuci_n,
            fecha_fin_ejecuci_n_dt AS fecha_fin_ejecuci_n,
            valor_contrato_num / NULLIF(promedio_categoria, 0)
                AS valor_vs_promedio_categoria,
            DATE_DIFF(
                'day', fecha_inicio_ejecuci_n_dt, fecha_fin_ejecuci_n_dt
            ) AS duracion_dias,
            MONTH(fecha_de_firma_del_contrato_dt) AS mes_firma,
            DAYOFWEEK(fecha_de_firma_del_contrato_dt) AS dia_semana_firma
        FROM con_concentracion
    """


def build_features() -> dict:
    con = duckdb.connect()
    try:
        input_rows = con.execute(
            f"""
            SELECT COUNT(*) FROM read_parquet('{CONTRACTS_PARQUET_PATH}')
            WHERE LOWER(estado_del_proceso) IN ({_estados_sql_list()})
              AND TRY_CAST(valor_contrato AS DOUBLE)
                  BETWEEN {VALOR_MIN} AND {VALOR_MAX}
            """
        ).fetchone()[0]
        logger.info(
            "Contratos que pasan el filtro de negocio (estado + rango de valor): %s",
            input_rows,
        )

        query = _build_query()
        con.execute(f"COPY ({query}) TO '{FEATURES_OUTPUT_PATH}' (FORMAT PARQUET)")

        output_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{FEATURES_OUTPUT_PATH}')"
        ).fetchone()[0]
    finally:
        con.close()

    # El renombrado de columnas truncadas (COLUMN_RENAME_MAP) se aplica
    # como paso final, sobre el Parquet ya escrito, para no complicar el
    # SQL principal con alias adicionales.
    _rename_columns(FEATURES_OUTPUT_PATH)

    summary = {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "output_path": FEATURES_OUTPUT_PATH,
        "categoria_cols": CATEGORIA_COLS,
    }
    logger.info("Resumen de features: %s", summary)
    return summary


def _rename_columns(parquet_path: str) -> None:
    """
    Aplica COLUMN_RENAME_MAP sobre el Parquet ya escrito.

    Escribe a un archivo temporal y luego reemplaza el original -- NUNCA
    se lee y escribe el mismo archivo en la misma consulta COPY: DuckDB
    evalúa la lectura de forma perezosa, así que leer y sobrescribir el
    mismo path en una sola operación puede corromper el archivo (el
    escritor podría empezar a truncar el archivo mientras el lector
    todavía no terminó de leerlo).
    """
    path = Path(parquet_path)
    tmp_path = path.with_suffix(".tmp.parquet")

    con = duckdb.connect()
    try:
        select_exprs = [
            f'"{old_name}" AS "{new_name}"'
            for old_name, new_name in COLUMN_RENAME_MAP.items()
        ]
        rename_select = ", ".join(select_exprs) if select_exprs else "*"
        exclude_cols = ", ".join(f'"{c}"' for c in COLUMN_RENAME_MAP)

        con.execute(
            f"""
            COPY (
                SELECT * EXCLUDE ({exclude_cols}), {rename_select}
                FROM read_parquet('{path.as_posix()}')
            ) TO '{tmp_path.as_posix()}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()

    tmp_path.replace(path)


if __name__ == "__main__":
    build_features()
