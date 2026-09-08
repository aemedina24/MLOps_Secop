# Estrategia de Reentrenamiento — Modelo de Detección de Contratos Atípicos

Este documento define **cuándo** y **cómo** se reentrena el modelo
`isolation_forest_secop`, y qué validaciones debe pasar un modelo nuevo
antes de reemplazar al que está en producción.

---

## 1. Disparadores de reentrenamiento (triggers)

### 1.1 — Programado (calendario)

El pipeline de ingesta (`ingest_secop.py`) trae datos nuevos de forma
incremental. Se recomienda reentrenar el modelo **cada vez que el
histórico crezca de forma significativa** (sugerido: trimestral, o
cuando el volumen de contratos nuevos supere ~5% del histórico actual),
no en cada ejecución de la ingesta — reentrenar en cada corrida
incremental sería sobreingeniería para el volumen de cambio real de
este dataset.

### 1.2 — Por drift detectado (basado en monitoreo)

Los reportes de `evidently_report.py` (data drift) y
`model_drift_report.py` (model drift) definen umbrales objetivos para
disparar una revisión:

| Señal | Umbral sugerido | Acción |
|---|---|---|
| % de columnas de entrada con drift (`data_drift_report`) | ≥ 50% (`DRIFT_SHARE_THRESHOLD` en `evidently_report.py`) | Revisar si el sesgo de IPC (ver `GOBERNANZA.md`, sección 4.1) ya justifica por sí solo el drift, o si hay una causa nueva que investigar |
| Cambio en la tasa de anomalías detectadas (`model_drift_report`) | Cambio relativo > 30% entre periodos | Investigar antes de reentrenar a ciegas -- un cambio grande en la tasa puede ser una señal real de que el mundo cambió, o un síntoma del problema de reproducibilidad ya documentado (`GOBERNANZA.md`, sección 4.3) |

**Importante:** un drift detectado no dispara un reentrenamiento
automático. Dado que ya se documentó que el pipeline tiene un sesgo
conocido de IPC (que **debería** producir drift esperado con el tiempo)
y un problema de reproducibilidad no resuelto de raíz, un salto en las
métricas de drift requiere primero **diagnóstico manual** para
distinguir "esto es el sesgo conocido funcionando como se espera" de
"pasó algo nuevo que amerita atención".

### 1.3 — Manual (por decisión del equipo)

Cualquier cambio deliberado en el pipeline que afecte el
entrenamiento debe ir acompañado de un reentrenamiento explícito,
incluyendo (no limitado a):

- Cambios en `config.py` (rango de valor, estados válidos, columnas de
  categoría).
- Cambios en `build_features.py` (nuevas features, cambios en el
  cálculo de las existentes).
- Corrección de la limitación de IPC (sección 4.1 de `GOBERNANZA.md`),
  si se implementa.
- Cambios en los hiperparámetros del modelo.

---

## 2. Proceso de reentrenamiento

1. **Regenerar `features.parquet`** con el pipeline reproducible de
   DVC:
   ```powershell
   uv run dvc repro features
   ```
2. **Entrenar el modelo** con los hiperparámetros vigentes (o los
   nuevos, si el reentrenamiento es por un cambio deliberado de
   hiperparámetros):
   ```powershell
   uv run python -m mlops_secop.model.train_isolation_forest
   ```
3. **Validar el modelo nuevo** (ver sección 3) antes de promoverlo.
4. **Versionar el modelo nuevo con DVC**:
   ```powershell
   uv run dvc add models/isolation_forest.joblib models/onehot_encoder.joblib
   ```
5. **Registrar en MLflow** como una nueva versión del modelo
   (`isolation_forest_secop`), sin sobreescribir el registro de la
   versión anterior — MLflow Model Registry versiona automáticamente.
6. **Comitear** `dvc.lock` (o los `.dvc` actualizados) y abrir Pull
   Request, siguiendo el flujo estándar del proyecto
   (`CONTRIBUTING.md`).

---

## 3. Validación antes de promover un modelo nuevo a producción

Dado el hallazgo documentado en `GOBERNANZA.md` (sección 4.2): **no
existe una métrica automática confiable para decidir si un modelo es
"mejor" que otro** en este problema (sin ground truth real, y con la
métrica sintética de Optuna ya descartada por no correlacionar con
calidad real). La validación es, por diseño, manual:

1. **Revisión manual de los contratos más atípicos** — igual que se
   hizo para el modelo actual: inspeccionar el top 15-20 de
   `anomaly_score` más alto y confirmar que tienen sentido de negocio
   (montos, entidades, modalidades que un auditor humano consideraría
   razonablemente atípicas).
2. **Comparar la tasa de anomalías** del modelo nuevo contra el modelo
   anterior sobre el mismo periodo — un cambio muy grande (ej. de 0.01%
   a 5%) es señal de alerta, no necesariamente de mejora.
3. **Correr el reporte de model drift** (`model_drift_report.py`)
   comparando el modelo nuevo contra el anterior sobre el mismo
   conjunto de datos, para entender qué tan distinto se comporta antes
   de reemplazarlo.
4. **Confirmar que la suite de tests pasa** (`uv run pytest tests\ -v`)
   — no valida la calidad del modelo en sí, pero confirma que el
   pipeline que lo produjo sigue funcionando correctamente.

Solo tras esta revisión manual, promover el modelo nuevo actualizando
su alias/stage en MLflow Model Registry (ej. de `None` a
`Production`), sin eliminar la versión anterior — permite revertir
rápidamente si algo sale mal.

---

## 4. Plan de rollback

Si un modelo recién promovido muestra un comportamiento inesperado en
producción:

1. Revertir el alias de `Production` en MLflow Model Registry a la
   versión anterior (no requiere reentrenar ni tocar código).
2. Si el problema viene de los datos (`features.parquet` corrupto o
   con un bug real, no solo el sesgo de IPC ya conocido), usar
   `dvc checkout` con el commit anterior de `dvc.lock` para volver a
   la versión de datos previa.
3. Documentar el incidente siguiendo el mismo formato usado en
   `GOBERNANZA.md` para los incidentes de `dvc.lock` ya registrados —
   mantener la misma disciplina de transparencia sobre qué pasó y por
   qué.

---

## 5. Limitación reconocida de esta estrategia

Esta estrategia asume que alguien del equipo **ejecuta manualmente**
los pasos anteriores cuando corresponde — no hay automatización de
"reentrenar y promover solo" (eso sería la Fase 7, "Ciclo MLOps
Completo", fuera de alcance de esta entrega). Los umbrales de la
sección 1.2 son puntos de partida razonables, no valores validados
estadísticamente contra datos históricos de reentrenamientos reales
(que aún no existen para este proyecto) — deben revisarse con la
experiencia de las primeras veces que se ejecute este proceso.
