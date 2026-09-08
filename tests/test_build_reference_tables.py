"""
Tests para `mlops_secop.model.build_reference_tables`.

Verifica que las tablas de referencia (promedio por categoria,
concentracion por proveedor) coincidan con un calculo independiente en
pandas sobre el mismo dataset sintetico -- no solo que el script corra
sin errores.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlops_secop.model import build_reference_tables as brt


@pytest.fixture
def synthetic_features_path(tmp_path) -> str:
    df = pd.DataFrame(
        {
            "modalidad_de_contratacion": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
            ],
            "departamento_entidad": [
                "X",
                "X",
                "Y",
                "X",
                "X",
                "X",
                "Y",
            ],
            "valor_contrato": [
                10.0,
                20.0,
                100.0,
                5.0,
                15.0,
                25.0,
                200.0,
            ],
            "documento_proveedor": [
                "p1",
                "p1",
                "p2",
                "p3",
                "p3",
                "p3",
                "p4",
            ],
        }
    )
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    return str(path)


def test_build_reference_tables_matches_manual_groupby(
    synthetic_features_path, tmp_path, monkeypatch
):
    monkeypatch.setattr(brt, "FEATURES_OUTPUT_PATH", synthetic_features_path)
    categoria_path = tmp_path / "categoria_promedios.parquet"
    proveedor_path = tmp_path / "proveedor_conteos.parquet"
    monkeypatch.setattr(brt, "CATEGORIA_PROMEDIOS_PATH", categoria_path)
    monkeypatch.setattr(brt, "PROVEEDOR_CONTEOS_PATH", proveedor_path)

    summary = brt.build_reference_tables()

    df = pd.read_parquet(synthetic_features_path)
    expected_categoria = (
        df.groupby(["modalidad_de_contratacion", "departamento_entidad"])[
            "valor_contrato"
        ]
        .mean()
        .reset_index(name="promedio_categoria")
        .sort_values(["modalidad_de_contratacion", "departamento_entidad"])
        .reset_index(drop=True)
    )
    actual_categoria = (
        pd.read_parquet(categoria_path)
        .sort_values(["modalidad_de_contratacion", "departamento_entidad"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        actual_categoria, expected_categoria, check_dtype=False
    )

    expected_proveedor = (
        df.groupby("documento_proveedor")
        .size()
        .reset_index(name="concentracion_proveedor")
        .sort_values("documento_proveedor")
        .reset_index(drop=True)
    )
    actual_proveedor = (
        pd.read_parquet(proveedor_path)
        .sort_values("documento_proveedor")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        actual_proveedor, expected_proveedor, check_dtype=False
    )

    assert summary["n_combinaciones_categoria"] == len(expected_categoria)
    assert summary["n_proveedores_distintos"] == len(expected_proveedor)
