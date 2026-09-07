"""
Construcción de features de "alerta" para el modelo de detección de
contratos atípicos (Isolation Forest), a partir de
`data/processed/secop_ii/contracts.parquet`.

Este script:
1. Filtra los contratos según las reglas de negocio de `config.py`
   (`ESTADOS_VALIDOS`, `VALOR_MIN`/`VALOR_MAX`), y exige que
   `fecha_inicio_ejecuci_n` y `fecha_fin_ejecuci_n` estén ambas
   presentes y en orden correcto (Isolation Forest no acepta NaN;
   ~624 contratos excluidos por fechas faltantes o invertidas, dato
   erróneo de origen, ~0.01% del histórico).
2. Selecciona EXPLÍCITAMENTE solo las columnas necesarias -- no usa
   `SELECT *` sobre las 22 columnas originales. `contracts.parquet`
   incluye varios campos de texto libre largo (`objeto_a_contratar`,
   `objeto_del_proceso`, `nombre_de_la_entidad`) que no aportan señal
   al modelo y multiplican el volumen de datos que hay que mover en
   memoria. En una máquina con RAM limitada (confirmado: ~6GB totales)
   esto fue la diferencia entre completar la consulta o agotar la
   memoria, incluso después de migrar a GROUP BY + JOIN y configurar
   spill a disco explícito.
3. Convierte `valor_contrato` a numérico y las tres columnas de fecha a
   tipo `DATE` real -- en `contracts.parquet` llegan como texto
   (`VARCHAR`), heredado deliberadamente de la capa RAW.
4. Calcula las features de alerta:
   - `valor_vs_promedio_categoria`: razón entre el valor del contrato y
     el promedio de su categoría (`CATEGORIA_COLS` en config.py).
   - `duracion_dias`: días entre inicio y fin de ejecución.
   - `concentracion_proveedor`: cuántos contratos tiene ese mismo
     proveedor en todo el histórico filtrado.
   - `mes_firma`, `dia_semana_firma`: estacionalidad / fechas atípicas.
5. Conserva `numero_del_contrato` y `url_contrato` como identificadores
   de referencia (NO son features del modelo) -- sin esto, cuando el
   modelo marque un contrato como atípico no habría forma de saber a
   cuál contrato real corresponde para que un auditor lo revise.
6. Renombra las columnas truncadas por Socrata según
   `COLUMN_RENAME_MAP` (solo en este dataset derivado).
7. Guarda el resultado en `data/processed/secop_ii/features.parquet`.

Por qué GROUP BY + JOIN y no funciones de ventana: se intentó primero
con `OVER (PARTITION BY ...)`, pero causó `OutOfMemoryException` sobre
el volumen completo. El patrón GROUP BY + JOIN (igual que
process_secop.py) calcula agregados sobre grupos pequeños que luego se
unen de vuelta al dataset filtrado, en vez de mantener el particionado
completo en memoria a la vez.

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

# --------------------------------------------------------------------------
# Selección explícita de columnas
# --------------------------------------------------------------------------

ADDITIONAL_CATEGORICAL_COLS: list[str] = [
    "nivel_entidad",
    "tipo_de_contrato",
    "origen",
    "tipo_documento_proveedor",
]

IDENTIFIER_COLS: list[str] = [
    "numero_del_contrato",
    "url_contrato",
]

_RAW_COLUMNS_NEEDED: list[str] = list(
    dict.fromkeys(
        [
            "estado_del_proceso",
            "valor_contrato",
            "fecha_de_firma_del_contrato",
            "fecha_inicio_ejecuci_n",
            "fecha_fin_ejecuci_n",
            "documento_proveedor",
            *CATEGORIA_COLS,
            *ADDITIONAL_CATEGORICAL_COLS,
            *IDENTIFIER_COLS,
        ]
    )
)


def _estados_sql_list() -> str:
    return ", ".join(f"'{estado}'" for estado in ESTADOS_VALIDOS)


def _categoria_partition_by() -> str:
    return ", ".join(CATEGORIA_COLS)


def _raw_columns_select() -> str:
    return ", ".join(_RAW_COLUMNS_NEEDED)


def _where_clause() -> str:
    return f"""
        LOWER(estado_del_proceso) IN ({_estados_sql_list()})
        AND TRY_CAST(valor_contrato AS DOUBLE)
            BETWEEN {VALOR_MIN} AND {VALOR_MAX}
        AND TRY_CAST(fecha_inicio_ejecuci_n AS DATE) IS NOT NULL
        AND TRY_CAST(fecha_fin_ejecuci_n AS DATE) IS NOT NULL
        AND TRY_CAST(fecha_fin_ejecuci_n AS DATE)
            >= TRY_CAST(fecha_inicio_ejecuci_n AS DATE)
    """


def _base_filtered_cte() -> str:
    return f"""
        base AS (
            SELECT
                {_raw_columns_select()},
                TRY_CAST(valor_contrato AS DOUBLE) AS valor_contrato_num,
                TRY_CAST(fecha_de_firma_del_contrato AS DATE)
                    AS fecha_de_firma_del_contrato_dt,
                TRY_CAST(fecha_inicio_ejecuci_n AS DATE)
                    AS fecha_inicio_ejecuci_n_dt,
                TRY_CAST(fecha_fin_ejecuci_n AS DATE)
                    AS fecha_fin_ejecuci_n_dt
            FROM read_parquet('{CONTRACTS_PARQUET_PATH}')
            WHERE {_where_clause()}
        )
    """


def _build_query() -> str:
    partition_by = _categoria_partition_by()
    identifier_select = ", ".join(f"base.{c}" for c in IDENTIFIER_COLS)
    categorical_select = ", ".join(f"base.{c}" for c in ADDITIONAL_CATEGORICAL_COLS)

    return f"""
        WITH {_base_filtered_cte()},
        agg_categoria AS (
            SELECT
                {partition_by},
                AVG(valor_contrato_num) AS promedio_categoria
            FROM base
            GROUP BY {partition_by}
        ),
        agg_proveedor AS (
            SELECT
                documento_proveedor,
                COUNT(*) AS concentracion_proveedor
            FROM base
            GROUP BY documento_proveedor
        )
        SELECT
            {identifier_select},
            base.documento_proveedor,
            base.{CATEGORIA_COLS[0]},
            base.{CATEGORIA_COLS[1]},
            {categorical_select},
            base.valor_contrato_num AS valor_contrato,
            base.fecha_de_firma_del_contrato_dt AS fecha_de_firma_del_contrato,
            base.fecha_inicio_ejecuci_n_dt AS fecha_inicio_ejecuci_n,
            base.fecha_fin_ejecuci_n_dt AS fecha_fin_ejecuci_n,
            base.valor_contrato_num / NULLIF(agg_categoria.promedio_categoria, 0)
                AS valor_vs_promedio_categoria,
            DATE_DIFF(
                'day', base.fecha_inicio_ejecuci_n_dt, base.fecha_fin_ejecuci_n_dt
            ) AS duracion_dias,
            MONTH(base.fecha_de_firma_del_contrato_dt) AS mes_firma,
            DAYOFWEEK(base.fecha_de_firma_del_contrato_dt) AS dia_semana_firma,
            agg_proveedor.concentracion_proveedor AS concentracion_proveedor
        FROM base
        JOIN agg_categoria USING ({partition_by})
        JOIN agg_proveedor USING (documento_proveedor)
    """


def build_features() -> dict:
    con = duckdb.connect()
    try:
        tmp_dir = Path(FEATURES_OUTPUT_PATH).parent / ".duckdb_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
        con.execute("PRAGMA memory_limit='1.5GB'")

        input_rows = con.execute(
            f"""
            SELECT COUNT(*) FROM read_parquet('{CONTRACTS_PARQUET_PATH}')
            WHERE {_where_clause()}
            """
        ).fetchone()[0]
        logger.info(
            "Contratos que pasan el filtro de negocio (estado + rango de "
            "valor + duracion valida): %s",
            input_rows,
        )

        query = _build_query()
        con.execute(f"COPY ({query}) TO '{FEATURES_OUTPUT_PATH}' (FORMAT PARQUET)")

        output_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{FEATURES_OUTPUT_PATH}')"
        ).fetchone()[0]
    finally:
        con.close()

    _rename_columns(FEATURES_OUTPUT_PATH)

    summary = {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "output_path": FEATURES_OUTPUT_PATH,
        "categoria_cols": CATEGORIA_COLS,
        "additional_categorical_cols": ADDITIONAL_CATEGORICAL_COLS,
        "identifier_cols": IDENTIFIER_COLS,
    }
    logger.info("Resumen de features: %s", summary)
    return summary


def _rename_columns(parquet_path: str) -> None:
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
