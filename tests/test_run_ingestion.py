"""
Tests de orquestacion de run_ingestion.
Mockea SocrataClient.paginate para no depender de red ni del token real.
Los pasos ya cubiertos por separado (_resolve_date_range en
test_ingest_secop.py, _save_page_as_parquet en test_ingest_parquet.py)
no se vuelven a probar aqui en detalle.
"""

from datetime import UTC, datetime, timedelta

from mlops_secop.data import ingest_secop as m


def _fake_paginate_two_pages(self, **kwargs):
    yield [{"nivel_entidad": "Nacional"}, {"nivel_entidad": "Territorial"}]
    yield [{"nivel_entidad": "Nacional"}]


def _fake_paginate_empty(self, **kwargs):
    return
    yield


def test_run_ingestion_counts_rows_and_pages_from_mocked_pages(tmp_path, monkeypatch):
    monkeypatch.setenv("SOCRATA_APP_TOKEN", "fake-token-for-test")
    monkeypatch.setenv("SECOP_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("SECOP_CHECKPOINT_PATH", str(tmp_path / "checkpoint.json"))
    monkeypatch.setattr(m.SocrataClient, "paginate", _fake_paginate_two_pages)

    # Forzamos que el rango de fechas produzca UNA sola ventana de chunking,
    # para que el mock se llame exactamente una vez. Usamos una fecha
    # relativa a "ahora" (no una fecha fija) para que este test nunca
    # expire: con una fecha fija, el rango [fecha, mañana) eventualmente
    # supera CHUNK_DAYS con el paso del tiempo y genera más de una ventana.
    monkeypatch.setattr(m, "DEFAULT_START_DATE", datetime.now(UTC) - timedelta(days=5))

    summary = m.run_ingestion()
    assert summary["total_rows"] == 3
    assert summary["total_pages"] == 2


def test_run_ingestion_writes_checkpoint_even_when_no_rows_found(tmp_path, monkeypatch):
    """
    Con el checkpoint incremental por ventana, cada ventana que termina sin
    excepciones se confirma en el checkpoint, incluso si no trajo filas —
    así una ventana genuinamente vacía no se vuelve a consultar para
    siempre en corridas futuras.
    """
    checkpoint_path = tmp_path / "checkpoint.json"
    monkeypatch.setenv("SOCRATA_APP_TOKEN", "fake-token-for-test")
    monkeypatch.setenv("SECOP_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("SECOP_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setattr(m.SocrataClient, "paginate", _fake_paginate_empty)
    monkeypatch.setattr(m, "DEFAULT_START_DATE", datetime.now(UTC) - timedelta(days=5))

    m.run_ingestion()

    assert checkpoint_path.exists() is True
