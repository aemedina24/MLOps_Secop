"""
Tests para `mlops_secop.monitoring.model_drift_report`.

A diferencia de los tests de `evidently_report.py` (que comparan
features crudas), aquí se entrena un modelo Isolation Forest REAL
(no un mock) sobre datos sintéticos, se guarda con joblib, y se valida
que el reporte de model drift lo carga y aplica correctamente -- para
confirmar que la integración con `train_isolation_forest.py` funciona
de punta a punta, no solo que cada pieza compila por separado.
"""

from __future__ import annotations

import random

import joblib
import pandas as pd
import pytest

from mlops_secop.model.train_isolation_forest import prepare_matrix, train_model
from mlops_secop.monitoring import evidently_report as er
from mlops_secop.monitoring import model_drift_report as mdr


def _make_rows(n: int, seed: int, valor_shift: float = 1.0) -> list[dict]:
    rng = random.Random(seed)
    modalidades = [f"modalidad_{i}" for i in range(10)]
    departamentos = [f"depto_{i}" for i in range(15)]
    rows = []
    for _ in range(n):
        rows.append(
            {
                "valor_contrato": rng.uniform(1_000_000, 200_000_000) * valor_shift,
                "valor_vs_promedio_categoria": rng.gauss(1.0, 0.3),
                "duracion_dias": rng.randint(30, 365),
                "mes_firma": rng.randint(1, 12),
                "dia_semana_firma": rng.randint(0, 6),
                "concentracion_proveedor": rng.randint(1, 20),
                "modalidad_de_contratacion": rng.choice(modalidades),
                "departamento_entidad": rng.choice(departamentos),
                "nivel_entidad": rng.choice(["Nacional", "Territorial"]),
                "tipo_de_contrato": rng.choice(["Prestacion", "Obra", "Suministro"]),
                "origen": rng.choice(["SECOPI", "SECOPII"]),
                "tipo_documento_proveedor": rng.choice(["NIT", "Cedula"]),
            }
        )
    return rows


@pytest.fixture
def trained_model_and_data(tmp_path):
    """
    Entrena un modelo real (no mock) sobre datos sintéticos, lo guarda
    con joblib, y prepara un features.parquet con 2023 (referencia) y
    2025-2026 (actual) para que run_model_drift_report los use.
    """
    training_rows = _make_rows(1000, seed=1)
    training_df = pd.DataFrame(training_rows)
    X, encoder = prepare_matrix(training_df)
    model = train_model(X, n_estimators=50)

    model_path = tmp_path / "model.joblib"
    encoder_path = tmp_path / "encoder.joblib"
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)

    rows_2023 = _make_rows(300, seed=2, valor_shift=1.0)
    for r in rows_2023:
        r["fecha_de_firma_del_contrato"] = "2023-06-15"
    rows_2025 = _make_rows(150, seed=3, valor_shift=1.0)
    for r in rows_2025:
        r["fecha_de_firma_del_contrato"] = "2025-06-15"
    rows_2026 = _make_rows(150, seed=4, valor_shift=1.0)
    for r in rows_2026:
        r["fecha_de_firma_del_contrato"] = "2026-06-15"

    features_df = pd.DataFrame(rows_2023 + rows_2025 + rows_2026)
    features_path = tmp_path / "features.parquet"
    features_df.to_parquet(features_path, index=False)

    return {
        "model_path": model_path,
        "encoder_path": encoder_path,
        "features_path": features_path,
    }


def test_score_group_returns_expected_columns(trained_model_and_data):
    model = joblib.load(trained_model_and_data["model_path"])
    encoder = joblib.load(trained_model_and_data["encoder_path"])
    df = pd.DataFrame(_make_rows(50, seed=5))

    scored = mdr._score_group(df, model, encoder)

    assert set(scored.columns) == {"anomaly_score", "is_anomaly"}
    assert len(scored) == 50
    assert scored["is_anomaly"].dtype == bool


def test_run_model_drift_report_end_to_end(
    trained_model_and_data, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        er, "FEATURES_OUTPUT_PATH", str(trained_model_and_data["features_path"])
    )
    output_path = tmp_path / "model_drift_report.html"

    summary = mdr.run_model_drift_report(
        output_path=output_path,
        model_path=trained_model_and_data["model_path"],
        encoder_path=trained_model_and_data["encoder_path"],
    )

    assert output_path.exists()
    assert summary["reference_rows"] == 300
    assert summary["current_rows"] == 300
    assert 0.0 <= summary["reference_anomaly_rate"] <= 1.0
    assert 0.0 <= summary["current_anomaly_rate"] <= 1.0
    assert summary["score_drift_detected"] in (True, False)
