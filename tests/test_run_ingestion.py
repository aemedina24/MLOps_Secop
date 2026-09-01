"""
Tests de orquestacion de run_ingestion.

Mockea SocrataClient.paginate para no depender de red ni del token real.
Los pasos ya cubiertos por separado (_resolve_date_range en
test_ingest_secop.py, _save_page_as_parquet en test_ingest_parquet.py)
no se vuelven a probar aqui en detalle.
"""

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

    summary = m.run_ingestion()

    assert summary["total_rows"] == 3
    assert summary["total_pages"] == 2


def test_run_ingestion_does_not_update_checkpoint_when_no_rows(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.json"
    monkeypatch.setenv("SOCRATA_APP_TOKEN", "fake-token-for-test")
    monkeypatch.setenv("SECOP_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("SECOP_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setattr(m.SocrataClient, "paginate", _fake_paginate_empty)

    m.run_ingestion()

    assert checkpoint_path.exists() is False
