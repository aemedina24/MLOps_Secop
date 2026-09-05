"""
Tests de integración para la capa de processing de SECOP (motor DuckDB).

A diferencia de la versión anterior basada en pandas (que probaba
`_deduplicate()` con DataFrames en memoria), estas pruebas escriben
particiones Parquet reales en un directorio temporal y ejecutan
`run_processing()` de punta a punta, verificando el Parquet de salida.
Esto ejercita la consulta SQL real (incluyendo `GROUP BY ALL` sobre
`read_parquet`) en vez de solo la lógica equivalente en pandas.
"""

import pandas as pd
import pytest

from mlops_secop.data import process_secop


def _write_partition(raw_root, ingestion_date, rows):
    """Escribe una partición `ingestion_date=.../part-000.parquet` con `rows`."""
    partition_dir = raw_root / f"ingestion_date={ingestion_date}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(partition_dir / "part-000.parquet", index=False)


@pytest.fixture
def raw_and_processed(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    monkeypatch.setenv("SECOP_RAW_DIR", str(raw_root))
    monkeypatch.setenv("SECOP_PROCESSED_DIR", str(processed_root))
    return raw_root, processed_root


def test_removes_exact_duplicate_rows(raw_and_processed):
    """Dos filas 100% idénticas (mismo ingestion_date) deben colapsar en una sola."""
    raw_root, processed_root = raw_and_processed
    _write_partition(
        raw_root,
        "2026-09-01",
        {
            "numero_del_contrato": ["A-1", "A-1"],
            "nit_de_la_entidad": ["123", "123"],
            "valor_contrato": ["1000", "1000"],
        },
    )

    summary = process_secop.run_processing()

    result = pd.read_parquet(processed_root / "contracts.parquet")
    assert len(result) == 1
    assert summary["input_rows"] == 2
    assert summary["output_rows"] == 1
    assert summary["duplicates_removed"] == 1


def test_keeps_rows_that_share_key_but_differ_elsewhere(raw_and_processed):
    """
    Dos filas con el mismo numero_del_contrato pero distintas en otra
    columna (ej. distinto valor_contrato, como en un contrato marco con
    múltiples ítems) NO deben fusionarse.
    """
    raw_root, processed_root = raw_and_processed
    _write_partition(
        raw_root,
        "2026-09-01",
        {
            "numero_del_contrato": ["A-1", "A-1"],
            "nit_de_la_entidad": ["123", "123"],
            "valor_contrato": ["1000", "2000"],  # <- difiere
        },
    )

    process_secop.run_processing()

    result = pd.read_parquet(processed_root / "contracts.parquet")
    assert len(result) == 2


def test_keeps_most_recent_ingestion_date_on_conflict(raw_and_processed):
    """
    Si hay duplicados exactos en distintas particiones (mismo contenido,
    distinto ingestion_date), debe quedarse con el ingestion_date más
    reciente. También verifica que ingestion_date se preserve como texto
    "YYYY-MM-DD" y no se convierta a un tipo DATE.
    """
    raw_root, processed_root = raw_and_processed
    row = {
        "numero_del_contrato": ["A-1"],
        "nit_de_la_entidad": ["123"],
        "valor_contrato": ["1000"],
    }
    _write_partition(raw_root, "2026-09-01", row)
    _write_partition(raw_root, "2026-09-02", row)

    process_secop.run_processing()

    result = pd.read_parquet(processed_root / "contracts.parquet")
    assert len(result) == 1
    assert result["ingestion_date"].iloc[0] == "2026-09-02"


def test_does_not_remove_rows_when_no_duplicates_exist(raw_and_processed):
    """Un dataset sin filas repetidas debe salir intacto."""
    raw_root, processed_root = raw_and_processed
    _write_partition(
        raw_root,
        "2026-09-01",
        {
            "numero_del_contrato": ["A-1", "A-2", "A-3"],
            "nit_de_la_entidad": ["123", "456", "789"],
            "valor_contrato": ["1000", "2000", "3000"],
        },
    )

    process_secop.run_processing()

    result = pd.read_parquet(processed_root / "contracts.parquet")
    assert len(result) == 3


def test_raises_when_no_partitions_found(raw_and_processed):
    """Si no hay particiones ingestion_date=*, debe fallar explícitamente."""
    raw_root, _processed_root = raw_and_processed
    raw_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        process_secop.run_processing()
