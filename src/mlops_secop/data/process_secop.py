"""
Deduplicación y consolidación de la capa RAW de contratos SECOP II hacia
`data/processed/`.

Este script:
1. Lee todos los archivos Parquet en `data/raw/secop_ii/ingestion_date=*/`,
   ignorando cualquier otra carpeta (como `_checkpoints/`).
2. Añade la columna `ingestion_date` extraída del nombre de la partición
   (Hive partitioning), sin materializar el dataset completo en memoria.
3. Deduplica por fila completa (todas las columnas salvo `ingestion_date`),
   quedándose con el `ingestion_date` más reciente cuando hay duplicados
   exactos. No se usa `numero_del_contrato` como clave: en el dataset SECOP
   Integrado, un mismo `numero_del_contrato` puede agrupar cientos de
   ítems/entregas/proveedores distintos bajo un contrato marco (confirmado
   empíricamente: un caso con 303 `objetos_a_contratar` distintos bajo un
   mismo `numero_del_contrato`).
4. Guarda el resultado consolidado en `data/processed/secop_ii/contracts.parquet`.

Procesamiento "out-of-core" con DuckDB
---------------------------------------
Con ~5 millones de filas repartidas en ~1.000 archivos Parquet, cargar todo
con pandas (`pd.concat` de todas las particiones) agota la memoria disponible:
cada string en pandas es un objeto Python individual con overhead propio, así
que el tamaño real en memoria puede ser 5-10 veces mayor que el Parquet en
disco. DuckDB lee y agrega los Parquet directamente desde disco, de forma
vectorizada y con spill-to-disk automático cuando hace falta, sin necesidad
de materializar el dataset completo en RAM al mismo tiempo.

Variables de entorno opcionales:
- SECOP_RAW_DIR        (default: data/raw/secop_ii)
- SECOP_PROCESSED_DIR  (default: data/processed/secop_ii)

Ejecución manual (PowerShell, con uv):
    uv run python -m mlops_secop.data.process_secop
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _raw_dir() -> Path:
    return Path(os.environ.get("SECOP_RAW_DIR", "data/raw/secop_ii"))


def _processed_dir() -> Path:
    return Path(os.environ.get("SECOP_PROCESSED_DIR", "data/processed/secop_ii"))


def _raw_glob(raw_root: Path) -> str:
    """
    Patrón glob hacia los archivos Parquet dentro de las particiones
    `ingestion_date=YYYY-MM-DD/part-*.parquet`.

    No apuntamos el glob a `raw_root` directamente porque esa carpeta
    también contiene `_checkpoints/`, que no es una partición de datos;
    exigir el prefijo `part-*.parquet` dentro de cada `ingestion_date=*`
    lo evita.
    """
    return (raw_root / "ingestion_date=*" / "part-*.parquet").as_posix()


def _warn_empty_partitions(raw_root: Path) -> list[Path]:
    """
    Recorre las carpetas `ingestion_date=*` (sin leer su contenido) y
    registra un warning por cada una que no tenga archivos parquet.

    Es solo un chequeo de metadata (listar nombres de archivo), no carga
    datos, así que no afecta el uso de memoria del out-of-core con DuckDB.
    """
    partition_dirs = sorted(raw_root.glob("ingestion_date=*"))
    if not partition_dirs:
        raise FileNotFoundError(
            f"No se encontraron particiones 'ingestion_date=*' en {raw_root}"
        )

    for partition_dir in partition_dirs:
        if not any(partition_dir.glob("part-*.parquet")):
            logger.warning("Partición sin archivos parquet: %s", partition_dir)

    return partition_dirs


def _dedup_query(raw_glob: str) -> str:
    """
    Construye la consulta SQL de deduplicación sobre los Parquet crudos.

    `GROUP BY ALL` agrupa por todas las expresiones del SELECT que no son
    una agregación (es decir, todas las columnas salvo `ingestion_date`),
    lo cual equivale exactamente a deduplicar por fila completa. `MAX
    (ingestion_date)` conserva la fecha de ingesta más reciente para cada
    grupo de duplicados, igual que el comportamiento anterior basado en
    pandas (sort ascendente + `drop_duplicates(keep="last")`).

    `hive_types_autocast = 0` evita que DuckDB convierta automáticamente
    `ingestion_date` (extraído del nombre de carpeta) a un tipo DATE;
    se mantiene como texto "YYYY-MM-DD", igual que en la implementación
    anterior.
    """
    return f"""
        SELECT * EXCLUDE (ingestion_date), MAX(ingestion_date) AS ingestion_date
        FROM read_parquet(
            '{raw_glob}',
            hive_partitioning = true,
            hive_types_autocast = 0
        )
        GROUP BY ALL
    """


def run_processing() -> dict:
    raw_root = _raw_dir()
    processed_root = _processed_dir()
    processed_root.mkdir(parents=True, exist_ok=True)

    _warn_empty_partitions(raw_root)

    raw_glob = _raw_glob(raw_root)
    output_path = processed_root / "contracts.parquet"

    con = duckdb.connect()
    try:
        input_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{raw_glob}')"
        ).fetchone()[0]
        logger.info("Total filas RAW combinadas: %s", input_rows)

        con.execute(
            f"COPY ({_dedup_query(raw_glob)}) "
            f"TO '{output_path.as_posix()}' (FORMAT PARQUET)"
        )

        output_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{output_path.as_posix()}')"
        ).fetchone()[0]
    finally:
        con.close()

    duplicates_removed = input_rows - output_rows
    logger.info(
        "Deduplicación: %s filas -> %s filas (%s duplicados eliminados)",
        input_rows,
        output_rows,
        duplicates_removed,
    )
    logger.info("Guardado dataset deduplicado en %s", output_path)

    summary = {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "duplicates_removed": duplicates_removed,
        "output_path": str(output_path),
    }
    logger.info("Resumen de processing: %s", summary)
    return summary


if __name__ == "__main__":
    run_processing()
