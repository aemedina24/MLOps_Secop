# Gobernanza del Modelo — Detección de Contratos Atípicos SECOP II

Este documento responde a las preguntas mínimas de gobernanza que
cualquier modelo en producción debe poder contestar: **qué modelo está
corriendo, con qué datos y código se generó, qué tan bien funciona, y
qué limitaciones conocidas tiene** — para que cualquier decisión basada
en sus resultados (marcar un contrato como "candidato a revisión") se
tome con el contexto completo, no como una caja negra.

---

## 1. Modelo en producción

| Campo | Valor |
|---|---|
| Nombre en MLflow Model Registry | `isolation_forest_secop` |
| Versión | 1 |
| Estado | `READY` |
| Algoritmo | Isolation Forest (scikit-learn) |
| Run ID de MLflow | No disponible de forma centralizada -- ver nota abajo |

> **Nota sobre el tracking de MLflow:** `mlruns/` y `mlflow.db` están en
> `.gitignore` (correcto: son bases de datos locales, no deberían
> versionarse en Git). Esto significa que **el tracking de experimentos
> es local a la máquina de cada colaborador** -- el Run ID exacto del
> modelo registrado como v1/READY vive únicamente en el `mlflow.db` de
> quien lo entrenó, no es recuperable desde otra máquina sin que esa
> persona lo comparta directamente. Confirmado empíricamente: al
> intentar consultar el registro desde una segunda máquina (con
> `features.parquet`/`models/*.joblib` ya sincronizados vía DVC), tanto
> `mlflow.db` como `mlruns/` aparecieron vacíos de ese registro.
>
> **Implicación de gobernanza:** el versionamiento actual (DVC para
> datos/modelo, Git para código) es suficiente para *reproducir* el
> modelo, pero no para *auditar centralizadamente* el historial de
> experimentos entre colaboradores -- cada quien tiene su propia
> "bitácora" local de MLflow. Para un entorno de producción real, esto
> requeriría un servidor de MLflow compartido (no solo `sqlite:///mlflow.db`
> local), fuera del alcance de este proyecto académico.

### Hiperparámetros

```
n_estimators   = 500
max_samples    = 'auto'
max_features   = 1.0
contamination  = 'auto'
random_state   = 42
```

Estos son los mismos hiperparámetros que la validación manual del
equipo confirmó como correctos (ver sección 4.2) — **no** son el
resultado "óptimo" que devolvió la búsqueda de Optuna, por una razón
documentada explícitamente ahí.

### Código y datos que generaron este modelo

| Artefacto | Ubicación | Hash MD5 (DVC) |
|---|---|---|
| Dataset de entrenamiento | `data/processed/secop_ii/features.parquet` | `62f1e95dd5a98eb82a5667baf5423a4c` (936 MB) |
| Modelo entrenado | `models/isolation_forest.joblib` | `6ef5345b67974feb33d99d2b3f8fb652` (3.96 MB) |
| Encoder (one-hot) | `models/onehot_encoder.joblib` | Ver `models/onehot_encoder.joblib.dvc` |
| Código de entrenamiento | `src/mlops_secop/model/train_isolation_forest.py` | `git log -- src/mlops_secop/model/train_isolation_forest.py` |

**Nota de trazabilidad:** el modelo y el encoder se versionan con `dvc
add` directo (archivos `.dvc` en `models/`), NO como stages de
`dvc.yaml`/`dvc.lock` como sí ocurre con `process` y `features`. Esto
significa que, a diferencia del dataset de features, **el modelo no se
regenera automáticamente con `dvc repro`** si cambia el código de
entrenamiento -- hay que volver a correr el entrenamiento y hacer
`dvc add` manualmente. Es una asimetría real en el pipeline, documentada
aquí para que quede explícita: no es un descuido, es cómo quedó
diseñado, pero vale la pena evaluarlo como mejora futura (agregar
`train_isolation_forest.py` como stage de `dvc.yaml` para que todo el
pipeline sea igualmente reproducible de punta a punta).

```powershell
# Para verificar estos hashes en cualquier momento:
Get-Content models\isolation_forest.joblib.dvc
Get-Content dvc.lock | Select-String -Pattern "features.parquet" -Context 3,3
```

**Por qué esto importa:** con DVC versionando estos tres artefactos
juntos, cualquier persona puede reproducir exactamente este modelo con
`dvc pull` + `dvc checkout`, sin depender de regenerar nada localmente
— lo cual es precisamente la lección del incidente de reproducibilidad
(sección 4.3).

---

## 2. Datos usados

- **Fuente:** SECOP Integrado (`rpmr-utcd`), Datos Abiertos Colombia.
- **Rango temporal del histórico:** 2022-01-01 a la fecha de la última
  ingesta.
- **Filtro de negocio aplicado** (ver `config.py`): solo contratos en
  estados que representan ejecución real o cierre (`ESTADOS_VALIDOS`),
  con `valor_contrato` entre $1,000,000 y $200,000,000, y con fechas de
  ejecución (`fecha_inicio_ejecucion`, `fecha_fin_ejecucion`) presentes
  y en orden correcto.
- **Volumen final tras filtros:** 6,225,097 contratos.

---

## 3. Métricas del modelo registrado

| Métrica | Valor |
|---|---|
| Contratos evaluados | 6,225,097 |
| Anomalías detectadas | 556 |
| Porcentaje de anomalías | 0.0089% |
| Método de validación | Revisión manual del equipo (no una métrica automática — ver sección 4.2) |

Estas cifras corresponden a la corrida registrada oficialmente en
MLflow como versión 1 del modelo. Dado el hallazgo de la sección 4.3,
**el conteo exacto de anomalías puede variar levemente si el modelo se
reentrena desde cero** en vez de partir de los artefactos ya
versionados en DVC — por eso el número oficial es el que quedó
registrado en MLflow para esta versión específica, no una constante
universal del pipeline.

---

## 4. Limitaciones conocidas

### 4.1 — El modelo no ajusta `valor_contrato` por inflación (IPC)

**El problema:** `valor_vs_promedio_categoria` compara cada contrato
contra el promedio de su categoría (modalidad + departamento)
calculado sobre **todo el histórico 2022-2026 mezclado, en pesos
nominales**. Con inflación acumulada significativa en ese periodo, esto
introduce un sesgo sistemático:

- Los contratos de 2025-2026 tienden a verse artificialmente "más caros
  que el promedio" solo por ser recientes → **posibles falsos
  positivos**.
- Los contratos de 2022-2023 pueden verse artificialmente normales
  aunque hayan tenido sobrecostos reales en su momento → **posibles
  falsos negativos**.

**Evidencia empírica de esta limitación:** se generó un reporte de
drift con Evidently comparando contratos firmados en 2023 (referencia)
contra 2025-2026 (actual) sobre las 12 columnas que usa el modelo
(`reports/data_drift_2023_vs_2025_2026.html`). Resultado:

| Métrica del reporte | Valor |
|---|---|
| Columnas evaluadas | 12 |
| Columnas con drift detectado | 7 (58.3%) |
| Drift de dataset detectado | Sí |

Dos de las columnas con drift son precisamente las relacionadas con
valor monetario: `valor_contrato` y `valor_vs_promedio_categoria` — 
consistente con la hipótesis de sesgo por inflación, no una sorpresa.

**Segunda pieza de evidencia — model drift:** además del drift en los
datos de entrada, se comparó cómo se comporta el **modelo ya
entrenado** (sin reentrenar) sobre contratos de 2023 vs 2025-2026
(`reports/model_drift_2023_vs_2025_2026.html`,
`model_drift_report.py`). Resultado:

| Métrica del reporte | 2023 (referencia) | 2025-2026 (actual) |
|---|---|---|
| Contratos evaluados | 1,202,009 | 2,156,337 |
| Tasa de anomalías detectadas | 0.0033% | 0.0053% |
| Cambio relativo en la tasa | +60.3% | |
| Drift detectado en `anomaly_score` | Sí | |

El modelo marca proporcionalmente **60% más contratos como atípicos**
en el periodo reciente que en 2023, sin haber sido reentrenado entre
ambos periodos — es decir, el mismo modelo se comporta distinto según
la época que evalúa. Esto es consistente con (y refuerza) la hipótesis
del sesgo de IPC: si `valor_vs_promedio_categoria` está sesgado hacia
arriba en contratos recientes, es esperable que el modelo marque
proporcionalmente más de ellos como atípicos, sin que eso signifique
necesariamente más irregularidad real en 2025-2026 que en 2023.

**Nota de interpretación importante:** este cambio del 60% en la tasa
de anomalías **no debe interpretarse automáticamente como una señal de
que el modelo necesita reentrenarse** (ver `ESTRATEGIA_REENTRENAMIENTO.md`,
sección 1.2) — es exactamente el comportamiento que se anticiparía
dado el sesgo ya conocido, no una sorpresa que amerite una respuesta
reactiva sin diagnóstico previo.

**Solución identificada pero pospuesta por tiempo:** normalizar
`valor_vs_promedio_categoria` dentro del mismo año de firma (agregar el
año a `CATEGORIA_COLS` en `build_features.py`) en vez de deflactar con
el IPC real del DANE — más simple (no requiere una fuente de datos
externa nueva) y resuelve la mayor parte del sesgo. Queda como mejora
futura, documentada aquí para que cualquier interpretación de los
resultados del modelo tenga en cuenta esta limitación mientras tanto.

**Alternativa evaluada y descartada:** restringir el dataset a 2023+
(excluyendo 2022). Se descartó porque no resuelve el sesgo (sigue
habiendo inflación dentro de 2023-2026) y reduce el tamaño de muestra
sin beneficio técnico claro.

### 4.2 — La métrica de Optuna no correlaciona con calidad real del modelo

**El problema:** se intentó afinar los hiperparámetros de Isolation
Forest con Optuna, usando una métrica de recall sobre anomalías
sintéticas inyectadas artificialmente en los datos. Al evaluar esta
métrica, se encontró que **no correlaciona con la calidad real del
modelo en este dataset** — la configuración de hiperparámetros que el
equipo ya había validado manualmente como razonable obtuvo una
calificación mala bajo esta métrica sintética.

**Decisión tomada:** el modelo que quedó en producción usa exactamente
los hiperparámetros validados manualmente por el equipo (sección 1),
no el resultado "óptimo" que devolvió la búsqueda de Optuna. El
experimento de Optuna quedó documentado en el código
(`src/mlops_secop/model/tune_isolation_forest.py`) como algo que se
intentó, junto con la razón por la que no se usó su resultado.

**Implicación para gobernanza:** sin ground truth real de fraude
confirmado, la validación del modelo depende de la revisión manual del
equipo (inspeccionar manualmente los contratos marcados como más
atípicos y confirmar que tienen sentido de negocio), no de una métrica
automática optimizable. Esto es una limitación real del proceso de
validación, no solo de esta iteración del modelo.

### 4.3 — Reproducibilidad del conteo exacto de anomalías

**El problema:** se entrenó el modelo final de forma independiente en
una segunda máquina, con el mismo código y los mismos hiperparámetros
exactos (`n_estimators=500, max_samples='auto', max_features=1.0,
contamination='auto', random_state=42`), sobre un `features.parquet`
regenerado de forma independiente (mismo número de filas: 6,225,097).
Resultado: **556 anomalías detectadas en una corrida, 1,919 en otra**,
a pesar de que el código fuente y las versiones de librerías clave
(`pandas`, `scikit-learn`, `numpy`) eran idénticas.

**Hipótesis más probable:** `features.parquet` se regeneró de forma
independiente en cada máquina con DuckDB. La columna
`valor_vs_promedio_categoria` se calcula con un `GROUP BY` + promedio
que DuckDB paraleliza — la suma de números de punto flotante no es
asociativa, así que el orden de agregación entre ejecuciones paralelas
puede producir diferencias mínimas en esa columna. Como
`contamination='auto'` de Isolation Forest usa un umbral **fijo**
sobre `decision_function` para decidir qué es anomalía, y el modelo ya
es sensible cerca de ese umbral (ver 4.2), una diferencia mínima en los
valores de entrada puede mover cientos de contratos de un lado al otro
del corte.

**Mitigación ya aplicada:** se versionó `features.parquet` y
`models/*.joblib` con DVC (ver sección 1), para que todo el equipo
parta del mismo archivo exacto en vez de regenerarlo cada quien por su
lado. Con esto, el modelo es reproducible bit a bit **siempre que se
use `dvc pull` + `dvc checkout`** en vez de recalcular el pipeline
desde cero en cada máquina.

**Incidente relacionado — `dvc.lock` desincronizado (dos veces):**
esta misma arquitectura de versionamiento tuvo dos incidentes reales
donde `dvc.lock` quedó desincronizado del código real (una vez por un
cambio de rendimiento en `build_features.py` no sincronizado a tiempo,
corregido en el PR que introdujo Optuna/MLflow/SHAP; una segunda vez
por el mismo motivo, detectado y corregido en un PR posterior). Lección
operativa: sincronizar `dvc.lock` (`dvc repro` o `dvc commit`, según
corresponda) **antes** de abrir cualquier Pull Request que toque un
archivo que sea dependencia de un stage de DVC, no después.

---

## 5. Monitoreo y reentrenamiento

| Componente | Ubicación | Qué mide |
|---|---|---|
| Data drift | `src/mlops_secop/monitoring/evidently_report.py`, `reports/data_drift_2023_vs_2025_2026.html` | Cambios en la distribución de las 12 columnas de entrada del modelo |
| Model drift | `src/mlops_secop/monitoring/model_drift_report.py`, `reports/model_drift_2023_vs_2025_2026.html` | Cambios en `anomaly_score` y en la tasa de anomalías que produce el mismo modelo, sin reentrenar |
| Estrategia de reentrenamiento | `docs/ESTRATEGIA_REENTRENAMIENTO.md` | Cuándo reentrenar (disparadores), cómo validar un modelo nuevo antes de promoverlo, y plan de rollback |

---

## 6. Resumen para quien vaya a usar o auditar este modelo

- El modelo detecta candidatos a revisión manual, **no** confirma
  fraude — un `anomaly_score` alto significa "estadísticamente
  atípico", no "irregular confirmado".
- Los resultados están sesgados hacia contratos recientes por el
  problema de inflación no corregido (sección 4.1) — al revisar la
  lista de candidatos, tener presente que contratos de 2025-2026 pueden
  estar sobrerrepresentados por esta razón, no necesariamente porque
  sean más irregulares.
- Reentrenar el modelo desde cero (en vez de partir de los artefactos
  versionados en DVC) puede cambiar el conteo exacto de anomalías —
  esto es esperado, no un error, mientras la causa (sección 4.3) no se
  resuelva de raíz.
