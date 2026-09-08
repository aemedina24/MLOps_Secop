"""
Tests para `mlops_secop.monitoring.evidently_report`.

El test más importante (`test_detects_deliberate_drift_but_not_stable_columns`)
no verifica solo que el código corre sin errores -- confirma que el
reporte distingue correctamente entre una columna con drift real
inyectado deliberadamente y columnas que no cambiaron, replicando el
mismo principio de validación que ya usamos para
test_detects_deliberate_outliers en test_train_isolation_forest.py.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest

from mlops_secop.monitoring import evidently_report as er


def _make_synthetic_features(
    n_per_group: int = 500, valor_shift: float = 1.0, seed: int = 42
) -> pd.DataFrame:
    """
    Genera filas sintéticas para un solo grupo (referencia o actual).
    `valor_shift` multiplica valor_contrato -- usarlo distinto entre
    dos llamadas simula un drift real inyectado deliberadamente.

    `seed` DEBE variar entre llamadas de distintos grupos (2023 vs 2025
    vs 2026): con una semilla fija compartida, distintas llamadas
    generan secuencias de números aleatorios idénticas fila por fila,
    lo que produce "drift" artificial por duplicación de patrones en
    vez de reflejar variación real entre grupos -- este bug se detectó
    empíricamente al validar este mismo test (columnas sin ningún shift
    deliberado se marcaban como drifted por esta razón).
    """
    rng = random.Random(seed)
    modalidades = [f"modalidad_{i}" for i in range(10)]
    departamentos = [f"depto_{i}" for i in range(15)]

    rows = []
    for _ in range(n_per_group):
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
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_features_path(tmp_path):
    """
    Crea un features.parquet sintético con 3 años: 2023 (referencia),
    2025 y 2026 (actual). valor_contrato en 2025-2026 tiene un shift
    deliberado de +40% respecto a 2023 (simula inflación real) -- el
    resto de columnas no cambia entre grupos.
    """
    df_2023 = _make_synthetic_features(n_per_group=500, valor_shift=1.0, seed=1)
    df_2023["fecha_de_firma_del_contrato"] = "2023-06-15"

    df_2025 = _make_synthetic_features(n_per_group=250, valor_shift=1.4, seed=2)
    df_2025["fecha_de_firma_del_contrato"] = "2025-06-15"

    df_2026 = _make_synthetic_features(n_per_group=250, valor_shift=1.4, seed=3)
    df_2026["fecha_de_firma_del_contrato"] = "2026-06-15"

    df_2022 = _make_synthetic_features(n_per_group=100, valor_shift=1.0, seed=4)
    df_2022["fecha_de_firma_del_contrato"] = "2022-06-15"

    df_all = pd.concat([df_2022, df_2023, df_2025, df_2026], ignore_index=True)
    path = tmp_path / "synthetic_features.parquet"
    df_all.to_parquet(path, index=False)
    return path


def test_load_split_by_year_selects_correct_groups(synthetic_features_path):
    reference, current = er.load_split_by_year(str(synthetic_features_path))

    assert len(reference) == 500  # solo 2023
    assert len(current) == 500  # 250 (2025) + 250 (2026), excluye 2022

    expected_cols = set(er.NUMERIC_COLS + er.CATEGORICAL_COLS)
    assert set(reference.columns) == expected_cols
    assert set(current.columns) == expected_cols
    # la columna de fecha usada solo para filtrar no debe quedar en el
    # dataframe final que se le pasa a Evidently.
    assert er.DATE_COL_FOR_SPLIT not in reference.columns


def test_load_split_by_year_raises_when_path_missing_via_default(monkeypatch):
    """
    load_features NO debe usar FEATURES_OUTPUT_PATH como default de
    parámetro (se evaluaría una sola vez al definir la función) -- este
    test confirma que monkeypatch.setattr sí tiene efecto real, leyendo
    la constante dentro del cuerpo de la función, no como default.
    """
    monkeypatch.setattr(er, "FEATURES_OUTPUT_PATH", "/ruta/que/no/existe.parquet")
    with pytest.raises(FileNotFoundError):
        er.load_split_by_year()


def test_run_drift_report_raises_on_empty_group(tmp_path, monkeypatch):
    """
    Si el año de referencia o actual no tiene ninguna fila (ej. dataset
    de prueba muy pequeño, o años mal configurados), debe fallar con un
    error claro en vez de generar un reporte vacío silenciosamente.
    """
    df = _make_synthetic_features(n_per_group=10)
    df["fecha_de_firma_del_contrato"] = "2024-06-15"  # ni 2023 ni 2025-2026
    path = tmp_path / "empty_groups.parquet"
    df.to_parquet(path, index=False)

    monkeypatch.setattr(er, "FEATURES_OUTPUT_PATH", str(path))

    with pytest.raises(ValueError, match="quedó vacío"):
        er.run_drift_report(output_path=tmp_path / "report.html")


def test_detects_deliberate_drift_via_comparison(synthetic_features_path):
    """
    Valida que valor_contrato queda marcado como drifted cuando SÍ hay
    un shift deliberado (+40%, escenario real de esta fixture), y NO
    lo está en un baseline sin shift (mismo generador).

    Usa er._column_is_drifted (no una comparación cruda de valores)
    porque Evidently cambia de método estadístico según el tamaño de
    muestra: con estos volúmenes (500 filas) usa tests de p-valor,
    donde valores BAJOS significan drift -- lo opuesto a las métricas
    de distancia (Wasserstein) que usa con datasets más grandes. Una
    comparación ingenua "mayor score = más drift" da resultados
    invertidos aqui; se confirmó empíricamente durante el desarrollo.
    """
    reference, current = er.load_split_by_year(str(synthetic_features_path))
    snapshot_with_shift = er.generate_drift_report(reference, current)
    entry_with_shift = next(
        m
        for m in snapshot_with_shift.dict()["metrics"][1:]
        if m["config"].get("column") == "valor_contrato"
    )

    # Mismo generador, mismo tamaño, pero SIN shift -- baseline de
    # "no debería haber drift real".
    baseline_reference = _make_synthetic_features(n_per_group=500, seed=10)
    baseline_current = _make_synthetic_features(n_per_group=500, seed=11)
    snapshot_baseline = er.generate_drift_report(baseline_reference, baseline_current)
    entry_baseline = next(
        m
        for m in snapshot_baseline.dict()["metrics"][1:]
        if m["config"].get("column") == "valor_contrato"
    )

    assert er._column_is_drifted(entry_with_shift), (
        f"valor_contrato con shift deliberado de +40% deberia marcarse "
        f"como drifted (metodo={entry_with_shift['config'].get('method')}, "
        f"value={entry_with_shift['value']})"
    )
    assert not er._column_is_drifted(entry_baseline), (
        f"valor_contrato SIN shift no deberia marcarse como drifted "
        f"(falso positivo) (metodo={entry_baseline['config'].get('method')}, "
        f"value={entry_baseline['value']})"
    )


def test_run_drift_report_end_to_end(synthetic_features_path, tmp_path, monkeypatch):
    monkeypatch.setattr(er, "FEATURES_OUTPUT_PATH", str(synthetic_features_path))
    output_path = tmp_path / "report.html"

    summary = er.run_drift_report(output_path=output_path)

    assert output_path.exists()
    assert summary["reference_rows"] == 500
    assert summary["current_rows"] == 500
    assert summary["n_columns_evaluated"] == len(er.NUMERIC_COLS) + len(
        er.CATEGORICAL_COLS
    )
    assert summary["n_drifted_columns"] == len(summary["drifted_columns"])
    assert "valor_contrato" in summary["per_column_drift_score"]
