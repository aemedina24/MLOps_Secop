"""
Tests de la lógica de resolución de rango de fechas y checkpoint de la
ingesta de SECOP II.

Deliberadamente NO llaman a la API real: evitan red, flakiness y
dependencia del SOCRATA_APP_TOKEN en CI. Cubren solo la lógica pura
(cálculo de fechas, lectura/escritura de checkpoint, lista de columnas).
La verificación contra la API real es un paso manual/de integración,
no parte de la suite de pytest.
"""

from datetime import UTC, datetime, timedelta

import pytest

from mlops_secop.data import ingest_secop as m


@pytest.fixture(autouse=True)
def _isolated_checkpoint(tmp_path, monkeypatch):
    """Aísla cada test en su propio checkpoint temporal."""
    checkpoint_file = tmp_path / "last_extraction.json"
    monkeypatch.setenv("SECOP_CHECKPOINT_PATH", str(checkpoint_file))
    yield checkpoint_file


def test_first_run_starts_at_2022_01_01():
    desde, _hasta = m._resolve_date_range()
    assert desde == m.DEFAULT_START_DATE


def test_hasta_is_start_of_tomorrow():
    _desde, hasta = m._resolve_date_range()
    now = datetime.now(UTC)
    expected = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    assert hasta == expected


def test_incremental_run_starts_at_checkpoint_minus_overlap():
    """
    El rango incremental no arranca exactamente en el checkpoint: retrocede
    OVERLAP_DAYS para recapturar contratos que SECOP registró con retraso
    respecto a su fecha real de firma (late-arriving data). Los duplicados
    que esto genera se resuelven en la capa de processing (_deduplicate).
    """
    checkpoint_dt = datetime(2024, 6, 1, tzinfo=UTC)
    m._write_checkpoint(checkpoint_dt)

    desde, _hasta = m._resolve_date_range()
    assert desde == checkpoint_dt - timedelta(days=m.OVERLAP_DAYS)


def test_checkpoint_never_goes_before_default_start():
    corrupted_checkpoint = datetime(2019, 1, 1, tzinfo=UTC)
    m._write_checkpoint(corrupted_checkpoint)

    desde, _hasta = m._resolve_date_range()
    assert desde == m.DEFAULT_START_DATE


def test_overlap_cannot_push_desde_before_default_start():
    """
    Si el checkpoint está cerca de DEFAULT_START_DATE, restarle OVERLAP_DAYS
    podría llevar desde antes del inicio oficial del dataset. La protección
    de DEFAULT_START_DATE debe seguir aplicando incluso con el solapamiento.
    """
    near_start = m.DEFAULT_START_DATE + timedelta(days=2)
    m._write_checkpoint(near_start)

    desde, _hasta = m._resolve_date_range()
    assert desde == m.DEFAULT_START_DATE


def test_selected_columns_match_spec_exactly():
    expected = {
        "nivel_entidad",
        "codigo_entidad_en_secop",
        "nombre_de_la_entidad",
        "nit_de_la_entidad",
        "departamento_entidad",
        "municipio_entidad",
        "estado_del_proceso",
        "modalidad_de_contrataci_n",
        "objeto_a_contratar",
        "objeto_del_proceso",
        "tipo_de_contrato",
        "fecha_de_firma_del_contrato",
        "fecha_inicio_ejecuci_n",
        "fecha_fin_ejecuci_n",
        "numero_del_contrato",
        "numero_de_proceso",
        "valor_contrato",
        "nom_raz_social_contratista",
        "url_contrato",
        "origen",
        "tipo_documento_proveedor",
        "documento_proveedor",
    }
    assert set(m.COLUMNS) == expected
    assert len(m.COLUMNS) == 22
