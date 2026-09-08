"""
Genera las tablas de referencia necesarias para calcular, en tiempo de
inferencia (API de Fase 4), las dos features que dependen de agregados
historicos: `valor_vs_promedio_categoria` y `concentracion_proveedor`.

Un contrato NUEVO no trae estos valores de fabrica -- se derivan del
historico ya procesado en `features.parquet`. Estas tablas se calculan
UNA VEZ (no en cada arranque de la API, para no requerir cargar
`features.parquet` completo -- ~937MB -- dentro del contenedor Docker)
y se versionan con DVC igual que el modelo y el encoder.

Ejecucion manual (PowerShell, con uv), despues de correr
build_features.py:
    uv run python -m mlops_secop.model.build_reference_tables
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from mlops_secop.config import CATEGORIA_COLS, COLUMN_RENAME_MAP, FEATURES_OUTPUT_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
CATEGORIA_PROMEDIOS_PATH = MODEL_DIR / "categoria_promedios.parquet"
PROVEEDOR_CONTEOS_PATH = MODEL_DIR / "proveedor_conteos.parquet"

# CATEGORIA_COLS en config.py usa los nombres crudos (truncados por
# Socrata, ej. modalidad_de_contrataci_n); features.parquet ya tiene el
# nombre renombrado via COLUMN_RENAME_MAP. Se traduce aqui para no
# duplicar la lista a mano y no desincronizarse si config.py cambia.
_CATEGORIA_COLS_FEATURES: list[str] = [
    COLUMN_RENAME_MAP.get(c, c) for c in CATEGORIA_COLS
]


def build_reference_tables() -> dict:
    """
    Recalcula, sobre `features.parquet` (ya filtrado por las reglas de
    negocio), los mismos dos agregados que build_features.py calcula
    internamente (`agg_categoria`, `agg_proveedor`) -- pero esta vez los
    guarda como artefactos propios en vez de solo usarlos para un JOIN
    interno, para poder reutilizarlos despues sobre un contrato nuevo
    que todavia no existe en el historico.
    """
    con = duckdb.connect()
    try:
        partition_by = ", ".join(_CATEGORIA_COLS_FEATURES)

        con.execute(
            f"""
            COPY (
                SELECT {partition_by}, AVG(valor_contrato) AS promedio_categoria
                FROM read_parquet('{FEATURES_OUTPUT_PATH}')
                GROUP BY {partition_by}
            ) TO '{CATEGORIA_PROMEDIOS_PATH.as_posix()}' (FORMAT PARQUET)
            """
        )
        con.execute(
            f"""
            COPY (
                SELECT documento_proveedor, COUNT(*) AS concentracion_proveedor
                FROM read_parquet('{FEATURES_OUTPUT_PATH}')
                GROUP BY documento_proveedor
            ) TO '{PROVEEDOR_CONTEOS_PATH.as_posix()}' (FORMAT PARQUET)
            """
        )

        n_categorias = con.execute(
            f"SELECT COUNT(*) FROM read_parquet("
            f"'{CATEGORIA_PROMEDIOS_PATH.as_posix()}')"
        ).fetchone()[0]
        n_proveedores = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{PROVEEDOR_CONTEOS_PATH.as_posix()}')"
        ).fetchone()[0]
    finally:
        con.close()

    summary = {
        "n_combinaciones_categoria": n_categorias,
        "n_proveedores_distintos": n_proveedores,
        "categoria_promedios_path": str(CATEGORIA_PROMEDIOS_PATH),
        "proveedor_conteos_path": str(PROVEEDOR_CONTEOS_PATH),
    }
    logger.info("Tablas de referencia generadas: %s", summary)
    return summary


if __name__ == "__main__":
    build_reference_tables()
