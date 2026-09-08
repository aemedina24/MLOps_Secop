"""
Reporte de MODEL drift: compara la distribución de `anomaly_score` y la
tasa de contratos marcados como atípicos que produce el MODELO YA
ENTRENADO, entre contratos de 2023 (referencia) y 2025-2026 (actual).

Diferencia con `evidently_report.py` (DATA drift): ese reporte compara
las FEATURES de entrada entre grupos. Este compara las PREDICCIONES del
modelo. Son complementarios, no redundantes -- un modelo puede producir
predicciones estables aunque las features de entrada hayan cambiado (si
el modelo es robusto a ese cambio), o viceversa. El objetivo aquí es
ver si el MISMO modelo, sin reentrenar, se comporta distinto sobre
datos de periodos distintos -- no evaluar un modelo nuevo.

Reutiliza el modelo y encoder ya entrenados desde disco (los mismos que
`train_isolation_forest.py` guarda) y la función `load_split_by_year`
de `evidently_report.py`, para no duplicar la lógica de separación por
año ni la de codificación de features.

Ejecución manual (PowerShell, con uv):
    uv run python -m mlops_secop.monitoring.model_drift_report
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from mlops_secop.model.train_isolation_forest import (
    ENCODER_PATH,
    MODEL_PATH,
    prepare_matrix,
)
from mlops_secop.monitoring.evidently_report import (
    _column_is_drifted,
    load_split_by_year,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPORT_OUTPUT_PATH = Path("reports/model_drift_2023_vs_2025_2026.html")


def _score_group(df: pd.DataFrame, model, encoder) -> pd.DataFrame:
    """
    Aplica el modelo YA ENTRENADO (no se reentrena aquí) a un grupo de
    contratos, usando el encoder ya ajustado -- igual que se haría en
    inferencia real sobre datos nuevos.
    """
    X, _ = prepare_matrix(df, encoder=encoder)
    raw_scores = model.decision_function(X)
    return pd.DataFrame(
        {
            "anomaly_score": -raw_scores,
            "is_anomaly": model.predict(X) == -1,
        }
    )


def run_model_drift_report(
    output_path: Path | None = None,
    model_path: Path | None = None,
    encoder_path: Path | None = None,
) -> dict:
    if output_path is None:
        output_path = REPORT_OUTPUT_PATH
    if model_path is None:
        model_path = MODEL_PATH
    if encoder_path is None:
        encoder_path = ENCODER_PATH

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    reference_features, current_features = load_split_by_year()

    reference_scores = _score_group(reference_features, model, encoder)
    current_scores = _score_group(current_features, model, encoder)

    report = Report([DataDriftPreset()])
    snapshot = report.run(
        current_data=current_scores[["anomaly_score"]],
        reference_data=reference_scores[["anomaly_score"]],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))
    logger.info("Reporte HTML de model drift guardado en %s", output_path)

    result_dict = snapshot.dict()
    score_entry = next(
        m
        for m in result_dict["metrics"][1:]
        if m["config"].get("column") == "anomaly_score"
    )
    score_drift_detected = _column_is_drifted(score_entry)

    reference_rate = float(reference_scores["is_anomaly"].mean())
    current_rate = float(current_scores["is_anomaly"].mean())
    rate_change_pct = (
        ((current_rate - reference_rate) / reference_rate * 100)
        if reference_rate > 0
        else None
    )

    summary = {
        "reference_rows": len(reference_scores),
        "current_rows": len(current_scores),
        "reference_anomaly_rate": reference_rate,
        "current_anomaly_rate": current_rate,
        "anomaly_rate_change_pct": rate_change_pct,
        "score_drift_detected": score_drift_detected,
        "score_drift_value": score_entry["value"],
        "report_path": str(output_path),
    }
    logger.info("Resumen de model drift: %s", summary)
    return summary


if __name__ == "__main__":
    run_model_drift_report()
