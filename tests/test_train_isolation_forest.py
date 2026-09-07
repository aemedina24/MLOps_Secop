"""
Tests para `mlops_secop.model.train_isolation_forest`.

El test más importante de este archivo (`test_detects_deliberate_outliers`)
no verifica solo que el código corra sin errores -- verifica que el
modelo realmente detecta anomalías inyectadas deliberadamente. Esto
importa porque durante el desarrollo se confirmó empíricamente que con
pocos árboles (`n_estimators` bajo) el modelo puede correr sin errores
y aun así fallar en detectar outliers reales, por la dilución de señal
de las columnas categóricas dispersas (ver comentario en
DEFAULT_N_ESTIMATORS). Un test que solo revisa "no crashea" no habría
detectado ese problema.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from mlops_secop.model import train_isolation_forest as tif


def _make_synthetic_features(n_rows: int = 2000, n_outliers: int = 20) -> pd.DataFrame:
    """
    Genera un dataset sintético con `n_outliers` filas deliberadamente
    atípicas en múltiples dimensiones a la vez (valor muy por encima
    del promedio de su categoría, duración extremadamente larga,
    proveedor que solo aparece una vez), y el resto de filas con
    valores "normales" alrededor de una distribución razonable.
    """
    rng = random.Random(42)
    modalidades = [f"modalidad_{i}" for i in range(10)]
    departamentos = [f"depto_{i}" for i in range(15)]

    rows = []
    for i in range(n_rows):
        is_outlier = i < n_outliers
        if is_outlier:
            valor_ratio = rng.uniform(15.0, 30.0)
            duracion = rng.randint(2000, 5000)
            concentracion = 1
        else:
            valor_ratio = max(0.05, rng.gauss(1.0, 0.3))
            duracion = rng.randint(30, 365)
            concentracion = rng.randint(2, 20)

        rows.append(
            {
                "numero_del_contrato": f"CTO-{i}",
                "url_contrato": f"https://example.com/CTO-{i}",
                "objeto_a_contratar": f"Objeto de contratacion {i}",
                "departamento_entidad": rng.choice(departamentos),
                "nivel_entidad": rng.choice(["Nacional", "Territorial"]),
                "tipo_de_contrato": rng.choice(["Prestacion", "Obra", "Suministro"]),
                "origen": rng.choice(["SECOPI", "SECOPII"]),
                "tipo_documento_proveedor": rng.choice(["NIT", "Cedula"]),
                "valor_contrato": float(rng.randint(1_000_000, 200_000_000)),
                "valor_vs_promedio_categoria": valor_ratio,
                "duracion_dias": duracion,
                "mes_firma": rng.randint(1, 12),
                "dia_semana_firma": rng.randint(0, 6),
                "concentracion_proveedor": concentracion,
                "modalidad_de_contratacion": rng.choice(modalidades),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return _make_synthetic_features()


def test_prepare_matrix_shape(synthetic_df):
    X, encoder = tif.prepare_matrix(synthetic_df)

    assert X.shape[0] == len(synthetic_df)
    # columnas numericas + columnas dummy del encoder
    expected_cols = len(tif.NUMERIC_COLS) + sum(
        len(cats) for cats in encoder.categories_
    )
    assert X.shape[1] == expected_cols


def test_prepare_matrix_is_sparse(synthetic_df):
    X, _ = tif.prepare_matrix(synthetic_df)
    # nnz (non-zero count) debe ser bastante menor que el total de
    # celdas -- confirma que efectivamente se está usando una matriz
    # dispersa, no una representación densa disfrazada.
    assert X.nnz < X.shape[0] * X.shape[1]


def test_encoder_reused_for_new_data(synthetic_df):
    """
    Un encoder ya ajustado debe poder aplicarse a datos nuevos (caso de
    uso de inferencia futura), sin volver a ajustarse.
    """
    X_train, encoder = tif.prepare_matrix(synthetic_df)
    X_new, encoder_again = tif.prepare_matrix(synthetic_df.head(5), encoder=encoder)

    assert encoder_again is encoder
    assert X_new.shape[1] == X_train.shape[1]


def test_save_and_load_artifacts_roundtrip(synthetic_df, tmp_path, monkeypatch):
    monkeypatch.setattr(tif, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(tif, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(tif, "ENCODER_PATH", tmp_path / "encoder.joblib")

    X, encoder = tif.prepare_matrix(synthetic_df)
    model = tif.train_model(X, n_estimators=50)
    tif.save_artifacts(model, encoder)

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "encoder.joblib").exists()

    import joblib

    loaded_model = joblib.load(tmp_path / "model.joblib")
    loaded_encoder = joblib.load(tmp_path / "encoder.joblib")

    X_reloaded, _ = tif.prepare_matrix(synthetic_df, encoder=loaded_encoder)
    # El modelo recargado debe producir exactamente los mismos scores.
    np.testing.assert_array_almost_equal(
        model.decision_function(X), loaded_model.decision_function(X_reloaded)
    )


def test_score_contracts_returns_identifiers_and_score(synthetic_df):
    X, encoder = tif.prepare_matrix(synthetic_df)
    model = tif.train_model(X, n_estimators=50)

    scored = tif.score_contracts(model, X, synthetic_df)

    assert set(scored.columns) == {
        "numero_del_contrato",
        "url_contrato",
        "objeto_a_contratar",
        "anomaly_score",
        "is_anomaly",
    }
    assert len(scored) == len(synthetic_df)
    # Debe venir ordenado de mas atipico a menos atipico.
    assert scored["anomaly_score"].is_monotonic_decreasing


def test_detects_deliberate_outliers(synthetic_df):
    """
    Confirma que el modelo REALMENTE detecta anomalías inyectadas
    deliberadamente, no solo que el pipeline corre sin errores.

    Usa DEFAULT_N_ESTIMATORS (no un valor bajo arbitrario) -- este test
    habría fallado con el valor por defecto original (100) antes de
    ajustarlo a 500 tras la validación empírica documentada en el
    módulo. Umbral relajado (top 30%, no top 5%) porque el dataset
    sintético es pequeño (2000 filas) comparado con el real (6.2M),
    donde el bosque tiene mucho más volumen para promediar el ruido.
    """
    n_outliers = 20
    X, encoder = tif.prepare_matrix(synthetic_df)
    model = tif.train_model(X, n_estimators=tif.DEFAULT_N_ESTIMATORS)

    scored = tif.score_contracts(model, X, synthetic_df)
    outlier_ids = {f"CTO-{i}" for i in range(n_outliers)}

    top_30_pct = scored.head(int(len(scored) * 0.30))
    found = outlier_ids & set(top_30_pct["numero_del_contrato"])

    assert len(found) >= n_outliers * 0.5, (
        f"Solo se detectaron {len(found)}/{n_outliers} outliers "
        f"deliberados en el top 30% -- posible regresion en la "
        f"capacidad de deteccion del modelo."
    )


def test_run_training_end_to_end(synthetic_df, tmp_path, monkeypatch):
    """
    Ejercita run_training() completo, no solo sus funciones internas
    por separado. Esto habria atrapado un bug real que ocurrio durante
    el desarrollo: un error de tipeo en el nombre de una variable
    (n_anomalias vs n_anomalies) dentro del diccionario de resumen, que
    ningun test de funciones individuales detecto porque ese codigo
    especifico solo se ejecuta dentro de run_training().
    """
    contracts_path = tmp_path / "synthetic_features.parquet"
    synthetic_df.to_parquet(contracts_path, index=False)

    monkeypatch.setattr(tif, "FEATURES_OUTPUT_PATH", str(contracts_path))
    monkeypatch.setattr(tif, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(tif, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(tif, "ENCODER_PATH", tmp_path / "encoder.joblib")

    summary = tif.run_training()

    assert summary["total_contratos"] == len(synthetic_df)
    assert "n_anomalias_detectadas" in summary
    assert summary["n_anomalias_detectadas"] >= 0
    assert (tmp_path / "model.joblib").exists()
