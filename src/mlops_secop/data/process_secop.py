"""
Deduplicación y consolidación de la capa RAW de contratos SECOP II hacia
`data/processed/`.

Este script:
1. Lee todos los archivos Parquet en `data/raw/secop_ii/ingestion_date=*/`,
   ignorando cualquier otra carpeta (como `_checkpoints/`).
2. Añade la columna `ingestion_date` extraída del nombre de la partición.
3. Deduplica por fila completa (todas las columnas salvo `ingestion_date`),
   quedándose con la fila del `ingestion_date` más reciente cuando hay
   duplicados exactos. No se usa `numero_del_contrato` como clave: en el
   dataset SECOP Integrado, un mismo `numero_del_contrato` puede agrupar
   cientos de ítems/entregas/proveedores distintos bajo un contrato marco
   (ver detalle empírico en el docstring de `_deduplicate()`).
4. Guarda el resultado consolidado en `data/processed/secop_ii/contracts.parquet`.

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

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _raw_dir() -> Path:
    return Path(os.environ.get("SECOP_RAW_DIR", "data/raw/secop_ii"))


def _processed_dir() -> Path:
    return Path(os.environ.get("SECOP_PROCESSED_DIR", "data/processed/secop_ii"))


def _load_raw(raw_root: Path) -> pd.DataFrame:
    """
    Lee todas las particiones `ingestion_date=YYYY-MM-DD/part-*.parquet`
    y agrega `ingestion_date` como columna, extraída del nombre de carpeta.

    No usamos pd.read_parquet(raw_root) directamente porque esa carpeta
    también contiene `_checkpoints/`, que no es una partición de datos.
    """
    partition_dirs = sorted(raw_root.glob("ingestion_date=*"))
    if not partition_dirs:
        raise FileNotFoundError(
            f"No se encontraron particiones 'ingestion_date=*' en {raw_root}"
        )

    frames: list[pd.DataFrame] = []
    for partition_dir in partition_dirs:
        ingestion_date = partition_dir.name.split("=", 1)[1]
        part_files = sorted(partition_dir.glob("part-*.parquet"))
        if not part_files:
            logger.warning("Partición sin archivos parquet: %s", partition_dir)
            continue

        df_partition = pd.concat(
            (pd.read_parquet(f) for f in part_files), ignore_index=True
        )
        df_partition["ingestion_date"] = ingestion_date
        frames.append(df_partition)
        logger.info(
            "Leídas %s filas de %s (%s archivos)",
            len(df_partition),
            partition_dir.name,
            len(part_files),
        )

    df = pd.concat(frames, ignore_index=True)
    logger.info("Total filas RAW combinadas: %s", len(df))
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplica por fila completa (todas las columnas salvo ingestion_date).

    No usamos numero_del_contrato como clave porque, en el dataset SECOP
    Integrado (rpmr-utcd), un mismo numero_del_contrato puede agrupar
    múltiples ítems/entregas/proveedores distintos bajo un contrato marco
    (confirmado empíricamente: ver ejemplo con 303 objetos_a_contratar
    distintos bajo un mismo numero_del_contrato). Comparar la fila completa
    evita fusionar registros que en realidad son legítimamente distintos.
    """
    before = len(df)

    df_sorted = df.sort_values("ingestion_date", ascending=True)
    compare_cols = [c for c in df_sorted.columns if c != "ingestion_date"]

    df_deduped = df_sorted.drop_duplicates(subset=compare_cols, keep="last")

    after = len(df_deduped)
    logger.info(
        "Deduplicación: %s filas -> %s filas (%s duplicados eliminados)",
        before,
        after,
        before - after,
    )
    return df_deduped


def run_processing() -> dict:
    raw_root = _raw_dir()
    processed_root = _processed_dir()
    processed_root.mkdir(parents=True, exist_ok=True)

    df = _load_raw(raw_root)
    df_deduped = _deduplicate(df)

    output_path = processed_root / "contracts.parquet"
    df_deduped.to_parquet(output_path, index=False)
    logger.info("Guardado dataset deduplicado en %s", output_path)

    summary = {
        "input_rows": len(df),
        "output_rows": len(df_deduped),
        "duplicates_removed": len(df) - len(df_deduped),
        "output_path": str(output_path),
    }
    logger.info("Resumen de processing: %s", summary)
    return summary


if __name__ == "__main__":
    run_processing()
