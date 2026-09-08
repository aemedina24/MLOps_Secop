# Imagen de la API de inferencia (Fase 4) -- sirve el modelo Isolation
# Forest ya entrenado y versionado con DVC. NO incluye el pipeline de
# ingesta/procesamiento/entrenamiento (esos pasos se corren fuera del
# contenedor, sobre datos completos, en la máquina de desarrollo).
#
# Nota honesta de simplificación: pyproject.toml no separa las
# dependencias de entrenamiento (dvc, mlflow, optuna, shap) de las que
# la API realmente necesita en tiempo de ejecución (fastapi, uvicorn,
# pandas, scikit-learn, duckdb, joblib) -- por tiempo, esta imagen
# instala TODAS las dependencias "no-dev". Una mejora futura razonable
# seria separar un extra "api" en pyproject.toml para una imagen mas
# liviana.
FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copiar solo los archivos de dependencias primero -- aprovecha el
# cache de capas de Docker: si despues solo cambia el codigo (src/),
# no hay que reinstalar todas las dependencias de nuevo.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --no-dev

# Solo los 4 artefactos que la API necesita para predecir -- no toda
# la carpeta models/ (que tambien tiene punteros .dvc, .gitkeep y el
# CSV de importancia de SHAP, irrelevantes para servir el modelo).
COPY models/isolation_forest.joblib \
     models/onehot_encoder.joblib \
     models/categoria_promedios.parquet \
     models/proveedor_conteos.parquet \
     ./models/

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "mlops_secop.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
