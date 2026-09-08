"""
Tests de la API de inferencia (Fase 4).

Construye un modelo/encoder/tablas de referencia SINTETICOS (no los
artefactos reales de 6.2M filas) y los inyecta via monkeypatch sobre
`mlops_secop.api.inference`, siguiendo el mismo patron ya usado en
test_train_isolation_forest.py. Esto permite probar la API completa
(incluyendo el endpoint HTTP) sin depender de tener los datos reales
descargados.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops_secop.api import inference as inf_module
from mlops_secop.api.main import app
from mlops_secop.model import train_isolation_forest as tif

MODALIDADES = ["Contratación directa", "Licitación pública"]
DEPARTAMENTOS = ["Cundinamarca", "Antioquia"]
PROVEEDOR_CONOCIDO = "900123456"


def _synthetic_training_df(n_rows: int = 300) -> pd.DataFrame:
    rng = random.Random(7)
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "numero_del_contrato": f"CTO-{i}",
                "url_contrato": f"https://example.com/CTO-{i}",
                "objeto_a_contratar": f"Objeto {i}",
                "departamento_entidad": rng.choice(DEPARTAMENTOS),
                "nivel_entidad": "Territorial",
                "tipo_de_contrato": "Prestación de servicios",
                "origen": "SECOPII",
                "tipo_documento_proveedor": "NIT",
                "valor_contrato": float(rng.randint(1_000_000, 200_000_000)),
                "valor_vs_promedio_categoria": max(0.1, rng.gauss(1.0, 0.3)),
                "duracion_dias": rng.randint(30, 365),
                "mes_firma": rng.randint(1, 12),
                "dia_semana_firma": rng.randint(0, 6),
                "concentracion_proveedor": rng.randint(1, 10),
                "modalidad_de_contratacion": rng.choice(MODALIDADES),
            }
        )
    # Un proveedor conocido, para poder probar el caso "proveedor visto".
    rows[0]["documento_proveedor"] = PROVEEDOR_CONOCIDO
    for i in range(1, n_rows):
        rows[i]["documento_proveedor"] = f"prov-{i}"
    return pd.DataFrame(rows)


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """
    Entrena un modelo pequeño con datos sinteticos, genera las tablas
    de referencia correspondientes, y apunta todos los artefactos de
    `mlops_secop.api.inference` a esos archivos temporales.
    """
    df = _synthetic_training_df()
    X, encoder = tif.prepare_matrix(df)
    model = tif.train_model(X, n_estimators=50)

    model_path = tmp_path / "model.joblib"
    encoder_path = tmp_path / "encoder.joblib"
    import joblib

    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)

    categoria_promedios = (
        df.groupby(["modalidad_de_contratacion", "departamento_entidad"])[
            "valor_contrato"
        ]
        .mean()
        .reset_index(name="promedio_categoria")
    )
    categoria_promedios_path = tmp_path / "categoria_promedios.parquet"
    categoria_promedios.to_parquet(categoria_promedios_path, index=False)

    proveedor_conteos = (
        df.groupby("documento_proveedor")
        .size()
        .reset_index(name="concentracion_proveedor")
    )
    proveedor_conteos_path = tmp_path / "proveedor_conteos.parquet"
    proveedor_conteos.to_parquet(proveedor_conteos_path, index=False)

    monkeypatch.setattr(inf_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(inf_module, "ENCODER_PATH", encoder_path)
    monkeypatch.setattr(
        inf_module, "CATEGORIA_PROMEDIOS_PATH", categoria_promedios_path
    )
    monkeypatch.setattr(inf_module, "PROVEEDOR_CONTEOS_PATH", proveedor_conteos_path)
    inf_module.get_inference_service.cache_clear()

    with TestClient(app) as client:
        yield client

    inf_module.get_inference_service.cache_clear()


def _contrato_payload(**overrides) -> dict:
    payload = {
        "numero_del_contrato": "CTO-TEST-1",
        "valor_contrato": 45_000_000,
        "fecha_de_firma_del_contrato": "2026-03-10",
        "fecha_inicio_ejecucion": "2026-03-15",
        "fecha_fin_ejecucion": "2026-09-15",
        "documento_proveedor": PROVEEDOR_CONOCIDO,
        "modalidad_de_contratacion": MODALIDADES[0],
        "departamento_entidad": DEPARTAMENTOS[0],
        "nivel_entidad": "Territorial",
        "tipo_de_contrato": "Prestación de servicios",
        "origen": "SECOPII",
        "tipo_documento_proveedor": "NIT",
    }
    payload.update(overrides)
    return payload


def test_health_reports_modelo_cargado(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["modelo_cargado"] is True


def test_predict_returns_score_and_flag(api_client):
    response = api_client.post("/predict", json=_contrato_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["numero_del_contrato"] == "CTO-TEST-1"
    assert isinstance(body["anomaly_score"], float)
    assert isinstance(body["is_anomaly"], bool)


def test_predict_with_unseen_provider_and_category_does_not_crash(api_client):
    """
    Un contrato con proveedor y combinacion modalidad/departamento
    nunca vistos en el historico debe usar los valores de fallback
    (concentracion_proveedor=1, promedio global) en vez de fallar.
    """
    response = api_client.post(
        "/predict",
        json=_contrato_payload(
            documento_proveedor="proveedor-nuevo-999",
            modalidad_de_contratacion="Modalidad inexistente",
            departamento_entidad="Departamento inexistente",
        ),
    )
    assert response.status_code == 200


def test_predict_rejects_fecha_fin_antes_de_inicio(api_client):
    response = api_client.post(
        "/predict",
        json=_contrato_payload(
            fecha_inicio_ejecucion="2026-09-15",
            fecha_fin_ejecucion="2026-03-15",
        ),
    )
    assert response.status_code == 422


def test_predict_rejects_valor_no_positivo(api_client):
    response = api_client.post("/predict", json=_contrato_payload(valor_contrato=0))
    assert response.status_code == 422


def test_health_reports_modelo_no_cargado_si_faltan_artefactos(tmp_path, monkeypatch):
    monkeypatch.setattr(inf_module, "MODEL_PATH", tmp_path / "no-existe.joblib")
    inf_module.get_inference_service.cache_clear()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["modelo_cargado"] is False
    inf_module.get_inference_service.cache_clear()
