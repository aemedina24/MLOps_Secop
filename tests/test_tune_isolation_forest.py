from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_secop.model import train_isolation_forest as tif
from mlops_secop.model import tune_isolation_forest as tune


def _make_fake_features_df(n: int = 300, random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    return pd.DataFrame(
        {
            "numero_del_contrato": [f"C-{i}" for i in range(n)],
            "url_contrato": [f"http://example.com/{i}" for i in range(n)],
            "objeto_a_contratar": [f"objeto-{i}" for i in range(n)],
            "valor_contrato": rng.lognormal(mean=17, sigma=1.0, size=n),
            "valor_vs_promedio_categoria": rng.normal(1.0, 0.3, size=n).clip(0.1, 3),
            "duracion_dias": rng.normal(180, 60, size=n).clip(1, 720),
            "mes_firma": rng.integers(1, 13, size=n),
            "dia_semana_firma": rng.integers(0, 7, size=n),
            "concentracion_proveedor": rng.integers(1, 20, size=n),
            "modalidad_de_contratacion": rng.choice(
                [f"modalidad_{i}" for i in range(5)], size=n
            ),
            "departamento_entidad": rng.choice(
                [f"depto_{i}" for i in range(8)], size=n
            ),
            "nivel_entidad": rng.choice(["nacional", "territorial"], size=n),
            "tipo_de_contrato": rng.choice([f"tipo_{i}" for i in range(5)], size=n),
            "origen": rng.choice(["origen_a", "origen_b"], size=n),
            "tipo_documento_proveedor": rng.choice(["nit", "cc"], size=n),
        }
    )


def test_make_synthetic_anomalies_returns_requested_count_with_unique_ids():
    df = _make_fake_features_df()
    synthetic = tune.make_synthetic_anomalies(df, n=10, random_state=1)

    assert len(synthetic) == 10
    assert synthetic["numero_del_contrato"].nunique() == 10
    assert set(synthetic["_synthetic_pattern"]) <= {
        "valor_extremo",
        "duracion_extrema",
        "concentracion_extrema",
    }


def test_make_synthetic_anomalies_actually_perturbs_values():
    df = _make_fake_features_df()
    synthetic = tune.make_synthetic_anomalies(df, n=30, random_state=1)

    valor_rows = synthetic[synthetic["_synthetic_pattern"] == "valor_extremo"]
    assert not valor_rows.empty
    # El valor sintetico debe ser mucho mayor que cualquier valor real observado.
    assert valor_rows["valor_contrato"].min() > df["valor_contrato"].max()


def test_recall_at_top_percentile_perfect_case():
    scored = pd.DataFrame(
        {
            "numero_del_contrato": ["SYNTH-0", "SYNTH-1", "C-1", "C-2", "C-3"],
            "anomaly_score": [0.9, 0.8, 0.1, 0.05, 0.01],
        }
    )
    recall = tune.recall_at_top_percentile(
        scored, synthetic_ids={"SYNTH-0", "SYNTH-1"}, top_percentile=0.4
    )
    assert recall == 1.0


def test_recall_at_top_percentile_zero_case():
    scored = pd.DataFrame(
        {
            "numero_del_contrato": ["C-1", "C-2", "SYNTH-0"],
            "anomaly_score": [0.9, 0.8, 0.01],
        }
    )
    recall = tune.recall_at_top_percentile(
        scored, synthetic_ids={"SYNTH-0"}, top_percentile=0.34
    )
    assert recall == 0.0


def test_run_tuning_returns_best_params_and_recall_in_range(tmp_path, monkeypatch):
    """
    run_tuning() es exploratorio (ver docstring del modulo): solo se
    verifica que corre de punta a punta y devuelve valores en rango
    razonable, no que "gane" contra el baseline.
    """
    features_path = tmp_path / "features.parquet"
    _make_fake_features_df(n=300, random_state=2).to_parquet(features_path)
    monkeypatch.setattr(tif, "FEATURES_OUTPUT_PATH", str(features_path))

    db_path = (tmp_path / "test_mlflow_tuning.db").as_posix()
    summary = tune.run_tuning(
        sample_size=250,
        n_synthetic=6,
        n_trials=2,
        tracking_uri=f"sqlite:///{db_path}",
    )

    assert 0.0 <= summary["best_recall_sample"] <= 1.0
    assert "n_estimators" in summary["best_params"]


def test_run_baseline_final_trains_saves_and_registers_model(tmp_path, monkeypatch):
    features_path = tmp_path / "features.parquet"
    _make_fake_features_df(n=300, random_state=2).to_parquet(features_path)

    monkeypatch.setattr(tif, "FEATURES_OUTPUT_PATH", str(features_path))
    monkeypatch.setattr(tif, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(
        tif, "MODEL_PATH", tmp_path / "models" / "isolation_forest.joblib"
    )
    monkeypatch.setattr(
        tif, "ENCODER_PATH", tmp_path / "models" / "onehot_encoder.joblib"
    )
    monkeypatch.setattr(tune, "MODEL_DIR", tmp_path / "models")

    db_path = (tmp_path / "test_mlflow_baseline.db").as_posix()
    summary = tune.run_baseline_final(
        tracking_uri=f"sqlite:///{db_path}",
        register_model=True,
        explain=True,
    )

    assert summary["total_contratos"] == 300
    assert summary["params"] == tune.BASELINE_PARAMS
    assert (tmp_path / "models" / "isolation_forest.joblib").exists()
    assert (tmp_path / "models" / "onehot_encoder.joblib").exists()
    assert (tmp_path / "models" / "shap_feature_importance.csv").exists()
