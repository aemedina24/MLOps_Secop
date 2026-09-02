"""Tests unitarios para la capa de processing de SECOP."""

import pandas as pd

from mlops_secop.data.process_secop import _deduplicate


def test_deduplicate_removes_exact_duplicate_rows():
    """Dos filas 100% idénticas deben colapsar en una sola."""
    df = pd.DataFrame(
        {
            "numero_del_contrato": ["A-1", "A-1"],
            "nit_de_la_entidad": ["123", "123"],
            "valor_contrato": ["1000", "1000"],
            "ingestion_date": ["2026-09-01", "2026-09-01"],
        }
    )

    result = _deduplicate(df)

    assert len(result) == 1


def test_deduplicate_keeps_rows_that_share_key_but_differ_elsewhere():
    """
    Dos filas con el mismo numero_del_contrato pero distintas en otra
    columna (ej. distinto objeto_a_contratar, como en un contrato marco
    con múltiples ítems) NO deben fusionarse.
    """
    df = pd.DataFrame(
        {
            "numero_del_contrato": ["A-1", "A-1"],
            "nit_de_la_entidad": ["123", "123"],
            "valor_contrato": ["1000", "2000"],  # <- difiere
            "ingestion_date": ["2026-09-01", "2026-09-01"],
        }
    )

    result = _deduplicate(df)

    assert len(result) == 2


def test_deduplicate_keeps_most_recent_ingestion_date_on_conflict():
    """
    Si hay duplicados exactos salvo por ingestion_date, debe quedarse
    con la copia del ingestion_date más reciente.
    """
    df = pd.DataFrame(
        {
            "numero_del_contrato": ["A-1", "A-1"],
            "nit_de_la_entidad": ["123", "123"],
            "valor_contrato": ["1000", "1000"],
            "ingestion_date": ["2026-09-01", "2026-09-02"],
        }
    )

    result = _deduplicate(df)

    assert len(result) == 1
    assert result["ingestion_date"].iloc[0] == "2026-09-02"


def test_deduplicate_does_not_remove_rows_when_no_duplicates_exist():
    """Un DataFrame sin filas repetidas debe salir intacto."""
    df = pd.DataFrame(
        {
            "numero_del_contrato": ["A-1", "A-2", "A-3"],
            "nit_de_la_entidad": ["123", "456", "789"],
            "valor_contrato": ["1000", "2000", "3000"],
            "ingestion_date": ["2026-09-01", "2026-09-01", "2026-09-01"],
        }
    )

    result = _deduplicate(df)

    assert len(result) == 3
