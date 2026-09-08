"""
API de inferencia (Fase 4): expone el modelo Isolation Forest ya
entrenado y versionado con DVC a traves de HTTP, en vez de que cada
consumidor tenga que cargar los artefactos manualmente desde un
notebook o script.

El modelo se carga desde los archivos versionados con DVC
(`models/isolation_forest.joblib` + `onehot_encoder.joblib`), NO desde
el MLflow Model Registry en vivo -- decision tomada para simplificar el
empaquetado en Docker (no depende de tener `mlflow.db` disponible en
tiempo de ejecucion). En un despliegue de produccion real, este mismo
punto es donde se cambiaria a `mlflow.sklearn.load_model(...)` para
cargar dinamicamente la version marcada como "Production" en el
Registry.

Ejecucion manual (PowerShell, con uv):
    uv run uvicorn mlops_secop.api.main:app --reload
Luego abrir http://127.0.0.1:8000/docs para la interfaz interactiva.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from mlops_secop.api.inference import ModeloNoDisponibleError, get_inference_service
from mlops_secop.api.schemas import ContratoInput, HealthOutput, PrediccionOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps SECOP -- API de deteccion de contratos atipicos",
    description=(
        "Recibe los datos de un contrato de SECOP II y devuelve un "
        "score de anomalia (mas alto = mas atipico) mas una bandera "
        "is_anomaly, calculados con el modelo Isolation Forest "
        "registrado en el MLflow Model Registry como "
        "'isolation_forest_secop'."
    ),
    version="1.0.0",
)


@app.get("/health", response_model=HealthOutput)
def health() -> HealthOutput:
    """
    Chequeo de vida basico. Intenta cargar el servicio de inferencia
    (sin fallar el proceso si el modelo no esta disponible todavia) para
    que un orquestador (Docker/Kubernetes) pueda distinguir "la API
    esta arriba" de "la API esta arriba Y lista para predecir".
    """
    try:
        get_inference_service()
        modelo_cargado = True
    except ModeloNoDisponibleError:
        modelo_cargado = False
    return HealthOutput(status="ok", modelo_cargado=modelo_cargado)


@app.post("/predict", response_model=PrediccionOutput)
def predict(contrato: ContratoInput) -> PrediccionOutput:
    """
    Recibe los datos crudos de un contrato nuevo y devuelve su score de
    anomalia. Las features derivadas de agregados historicos
    (valor_vs_promedio_categoria, concentracion_proveedor) se calculan
    internamente contra tablas de referencia precalculadas -- ver
    mlops_secop.api.inference.InferenceService.
    """
    try:
        servicio = get_inference_service()
    except ModeloNoDisponibleError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return servicio.predict(contrato)
