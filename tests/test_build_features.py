"""
Tests para `mlops_secop.features.build_features`.

Usan datasets Parquet sintéticos y pequeños (no el `contracts.parquet`
real de millones de filas), construidos a mano para cada caso, siguiendo
el mismo patrón que el resto de la suite (tests rápidos, sin depender de
datos reales ni de red).

Las columnas se escriben como texto (string), igual que llegan realmente
en `contracts.parquet` (la capa RAW nunca transforma tipos) -- así el
test también cubre que `build_features` hace las conversiones de tipo
correctamente, no solo que "funciona" con datos ya limpios.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlops_secop.features import build_features as bf

DEFAULT_ROW = {
    "estado_del_proceso": "celebrado",
    "valor_contrato": "5000000",
    "fecha_de_firma_del_contrato": "2024-01-15",
    "fecha_inicio_ejecuci_n": "2024-02-01",
    "fecha_fin_ejecuci_n": "2024-03-01",
    "documento_proveedor": "900123456",
    "modalidad_de_contrataci_n": "Contratacion directa",
    "departamento_entidad": "Antioquia",
    "nivel_entidad": "Territorial",
    "tipo_de_contrato": "Prestacion de servicios",
    "origen": "SECOPII",
    "tipo_documento_proveedor": "NIT",
    "numero_del_contrato": "CTO-001",
    "url_contrato": "https://example.com/CTO-001",
}


def _row(**overrides: str) -> dict:
    """Crea una fila de contrato sintética, con overrides sobre el default."""
    row = dict(DEFAULT_ROW)
    row.update(overrides)
    return row


def _write_contracts_parquet(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """
    Redirige CONTRACTS_PARQUET_PATH y FEATURES_OUTPUT_PATH a rutas
    temporales.

    Importante: build_features.py hace `from mlops_secop.config import
    CONTRACTS_PARQUET_PATH, ...`, así que el nombre queda vinculado al
    namespace del propio módulo build_features en el momento del import.
    Parchar mlops_secop.config no tendría efecto aquí -- hay que parchar
    el atributo directamente en el módulo bf.
    """
    contracts_path = tmp_path / "contracts.parquet"
    features_path = tmp_path / "features.parquet"
    monkeypatch.setattr(bf, "CONTRACTS_PARQUET_PATH", str(contracts_path))
    monkeypatch.setattr(bf, "FEATURES_OUTPUT_PATH", str(features_path))
    return contracts_path, features_path


def test_identifier_columns_preserved(isolated_paths):
    """
    numero_del_contrato y url_contrato deben llegar intactos al output
    -- son la única forma de rastrear un contrato real cuando el modelo
    lo marque como atipico. No son features, son referencia.
    """
    contracts_path, features_path = isolated_paths
    _write_contracts_parquet(
        contracts_path,
        [_row(numero_del_contrato="CTO-999", url_contrato="https://x.co/999")],
    )

    bf.build_features()

    result = pd.read_parquet(features_path)
    assert result["numero_del_contrato"].iloc[0] == "CTO-999"
    assert result["url_contrato"].iloc[0] == "https://x.co/999"


def test_filters_out_reversed_dates(isolated_paths):
    """
    fecha_fin anterior a fecha_inicio es un error de captura confirmado
    en el dato de origen (619 casos en el histórico real) -- esos
    contratos se excluyen por completo, no solo se les pone NULL en
    duracion_dias.
    """
    contracts_path, features_path = isolated_paths
    rows = [
        _row(
            documento_proveedor="A",
            fecha_inicio_ejecuci_n="2024-03-01",
            fecha_fin_ejecuci_n="2024-01-01",  # invertida
        ),
        _row(
            documento_proveedor="B",
            fecha_inicio_ejecuci_n="2024-01-01",
            fecha_fin_ejecuci_n="2024-03-01",  # correcta
        ),
    ]
    _write_contracts_parquet(contracts_path, rows)

    summary = bf.build_features()

    assert summary["output_rows"] == 1
    result = pd.read_parquet(features_path)
    assert result["documento_proveedor"].iloc[0] == "B"


def test_filters_out_rows_with_null_dates(isolated_paths):
    """
    Isolation Forest no acepta NaN -- fecha_inicio y fecha_fin deben
    estar ambas presentes para que el contrato entre al dataset de
    features, no solo tener el orden correcto cuando existen.
    """
    contracts_path, features_path = isolated_paths
    rows = [_row(fecha_inicio_ejecuci_n="no-es-una-fecha")]
    _write_contracts_parquet(contracts_path, rows)

    summary = bf.build_features()

    assert summary["output_rows"] == 0


def test_filters_out_invalid_estado(isolated_paths):
    contracts_path, features_path = isolated_paths
    rows = [
        _row(estado_del_proceso="celebrado"),
        _row(estado_del_proceso="cancelado"),  # no esta en ESTADOS_VALIDOS
    ]
    _write_contracts_parquet(contracts_path, rows)

    summary = bf.build_features()

    assert summary["output_rows"] == 1


def test_estado_filter_is_case_insensitive(isolated_paths):
    """
    El dataset real tiene inconsistencias de mayusculas/minusculas
    confirmadas empiricamente (ver config.py) -- el filtro debe seguir
    incluyendo el contrato sin importar el casing exacto.
    """
    contracts_path, features_path = isolated_paths
    rows = [_row(estado_del_proceso="CELEBRADO")]
    _write_contracts_parquet(contracts_path, rows)

    summary = bf.build_features()

    assert summary["output_rows"] == 1


def test_filters_out_valor_outside_range(isolated_paths):
    contracts_path, features_path = isolated_paths
    rows = [
        _row(valor_contrato="500000"),  # por debajo de VALOR_MIN
        _row(valor_contrato="5000000"),  # dentro del rango
        _row(valor_contrato="300000000"),  # por encima de VALOR_MAX
    ]
    _write_contracts_parquet(contracts_path, rows)

    summary = bf.build_features()

    assert summary["output_rows"] == 1


def test_valor_vs_promedio_categoria_computed_correctly(isolated_paths):
    contracts_path, features_path = isolated_paths
    # Misma categoria (modalidad + departamento) para los tres, valores
    # 1M / 2M / 3M -> promedio = 2M -> ratios esperados 0.5 / 1.0 / 1.5
    rows = [
        _row(valor_contrato="1000000", documento_proveedor="A"),
        _row(valor_contrato="2000000", documento_proveedor="B"),
        _row(valor_contrato="3000000", documento_proveedor="C"),
    ]
    _write_contracts_parquet(contracts_path, rows)

    bf.build_features()

    result = pd.read_parquet(features_path).sort_values("valor_contrato")
    ratios = result["valor_vs_promedio_categoria"].tolist()
    assert ratios == pytest.approx([0.5, 1.0, 1.5])


def test_duracion_dias_computed_correctly(isolated_paths):
    contracts_path, features_path = isolated_paths
    rows = [
        _row(
            fecha_inicio_ejecuci_n="2024-01-01",
            fecha_fin_ejecuci_n="2024-01-31",
        )
    ]
    _write_contracts_parquet(contracts_path, rows)

    bf.build_features()

    result = pd.read_parquet(features_path)
    assert result["duracion_dias"].iloc[0] == 30


def test_concentracion_proveedor_counts_correctly(isolated_paths):
    contracts_path, features_path = isolated_paths
    rows = [
        _row(documento_proveedor="900123456"),
        _row(documento_proveedor="900123456"),
        _row(documento_proveedor="900123456"),
        _row(documento_proveedor="800999999"),
    ]
    _write_contracts_parquet(contracts_path, rows)

    bf.build_features()

    result = pd.read_parquet(features_path)
    concentracion_por_proveedor = result.groupby("documento_proveedor")[
        "concentracion_proveedor"
    ].first()
    assert concentracion_por_proveedor["900123456"] == 3
    assert concentracion_por_proveedor["800999999"] == 1


def test_column_rename_map_applied(isolated_paths):
    contracts_path, features_path = isolated_paths
    _write_contracts_parquet(contracts_path, [_row()])

    bf.build_features()

    columns = set(pd.read_parquet(features_path).columns)
    for old_name, new_name in bf.COLUMN_RENAME_MAP.items():
        assert old_name not in columns
        assert new_name in columns


def test_date_columns_are_real_dates_not_strings(isolated_paths):
    """
    Verifica el tipo de columna vía DuckDB (DESCRIBE), no vía el dtype
    que pandas infiere al leer el Parquet: pandas representa columnas
    DATE de Parquet como objetos `datetime.date` individuales (dtype
    `object`), no como `datetime64`, así que `is_datetime64_any_dtype`
    daría un falso negativo aquí incluso con el tipo correcto en disco.
    """
    contracts_path, features_path = isolated_paths
    _write_contracts_parquet(contracts_path, [_row()])

    bf.build_features()

    import duckdb

    schema = duckdb.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{features_path.as_posix()}')"
    ).fetchdf()
    date_cols = schema.set_index("column_name")["column_type"]
    assert date_cols["fecha_de_firma_del_contrato"] == "DATE"
    assert date_cols["fecha_inicio_ejecucion"] == "DATE"
    assert date_cols["fecha_fin_ejecucion"] == "DATE"
