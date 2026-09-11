# 🚀 MLOps_Secop: Pipeline End-to-End de Machine Learning

## 📌 Arquitectura del Sistema

```text
                    ┌─────────────────────────┐
                    │         GitHub          │
                    │   Código + PR + CI/CD   │
                    └────────────┬────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────┐
│                    PROYECTO MLOps                       │
│                                                         │
│  uv ──────────────── Entornos + uv.lock                 │
│  DVC ─────────────── Datos + artefactos                 │
│  MLflow ──────────── Tracking + Model Registry          │
│  AutoML ──────────── Selección y optimización           │
│  FastAPI + Docker ── API REST + Empaquetado             │
│  Evidently ───────── Data/Model Drift + Monitoreo       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
---
## 🛠️ Separación de Responsabilidades

| Dimensión | Herramienta | Responsabilidad |
|---|---|---|
| Código y CI/CD | Git / GitHub | Control de versiones del código fuente, ramas, commits, Pull Requests y automatización CI/CD. |
| Dependencias | uv | Gestión reproducible del entorno mediante pyproject.toml y uv.lock. |
| Datos y artefactos | DVC | Versionamiento y trazabilidad de datasets y artefactos de Machine Learning. |
| Experimentos | MLflow | Tracking de experimentos, parámetros, métricas, artefactos y Model Registry. |
| Calidad | Ruff + pytest |Linting, formateo y ejecución de pruebas automatizadas.  |
| Serving | FastAPI + Docker | Exposición del modelo mediante API REST y empaquetado reproducible. |
| Monitoreo | Evidently | Detección de Data Drift, Model Drift y seguimiento del comportamiento del modelo. |

---
## 📁 Estructura del Proyecto

```text
El código productivo vive exclusivamente en src/.
MLOps_Secop/
│
├── .github/
│   └── workflows/          # Pipelines de CI/CD con GitHub Actions
│
├── data/
│   ├── raw/                # Datos originales
│   └── processed/          # Datos procesados
│
├── docs/                   # Gobernanza y estrategia de reentrenamiento
│   ├── GOBERNANZA.md
│   └── ESTRATEGIA_REENTRENAMIENTO.md
│
├── notebooks/              # Exploración y análisis inicial (EDA)
│
├── src/
│   └── mlops_secop/
│       ├── config.py       # Configuración central del proyecto
│       ├── data/           # Ingesta y procesamiento de datos
│       ├── features/       # Ingeniería de características
│       ├── model/          # Entrenamiento, tuning (Optuna) y SHAP
│       ├── api/            # API de inferencia (FastAPI)
│       └── monitoring/     # Data/Model Drift (Evidently)
│
├── tests/                  # Pruebas automatizadas con pytest
│
├── models/                 # Artefactos de modelos (versionados con DVC)
│
├── reports/                # Reportes HTML de Evidently (data/model drift)
│
├── Makefile                # Automatización de comandos
├── pyproject.toml          # Configuración y dependencias del proyecto
├── uv.lock                 # Versiones exactas de dependencias
├── Dockerfile              # Imagen del servicio
├── docker-compose.yml      # Orquestación de servicios
├── .gitignore              # Archivos excluidos de Git
└── README.md               # Documentación del proyecto
```
---
## 🗺️ Hoja de Ruta del Proyecto

### ✅ Fase 1 — Cimientos y Calidad

- [ ] Git + GitHub (protección formal de rama `main` pendiente de activar en GitHub Settings)
- [x] Entorno reproducible con `uv` (`pyproject.toml` + `uv.lock`)
- [x] Dependencias dev separadas (`pytest`, `ruff`, `pre-commit`)
- [x] Ruff configurado (reglas E, F, I, UP)
- [x] pytest configurado con tests reales
- [x] pre-commit activo
- [x] Estructura `src/` (código productivo separado de notebooks)

**Para contribuir al proyecto, ver [`CONTRIBUTING.md`](./CONTRIBUTING.md)**
(flujo de Git, convención de commits, troubleshooting).

### 📦 Fase 2 — Control de Versiones y Datos

- [x] Configuración del repositorio en `GitHub`
- [x] Configuración de `DVC`
- [x] Versionamiento de datasets
- [x] Organización de datos en `raw/` y `processed/`
- [x] Creación del pipeline de datos

### 🤖 Fase 3 — Machine Learning y Experiment Tracking

- [x] Exploración y análisis de los datos (EDA)
- [x] Feature Engineering
- [x] Implementación de modelos base
- [x] Implementación de AutoML
- [ ] Evaluación y comparación de modelos
- [x] Configuración de `MLflow`
- [x] Tracking de experimentos
- [x] Registro de modelos mediante MLflow Model Registry

### 🚀 Fase 4 — Model Serving y Contenedores

- [x] Creación de API de inferencia con `FastAPI`
- [x] Creación del `Dockerfile`
- [x] Contenerización del modelo
- [x] Configuración de `docker-compose`
- [x] Pruebas de la API

### 🔄 Fase 5 — CI/CD

- [x] Configuración de GitHub Actions
- [x] Automatización de pruebas
- [x] Automatización de Ruff
- [ ] Validación automática de Pull Requests (pendiente: branch protection)
- [x] Construcción automática de imagen Docker
- [x] Pipeline de despliegue (publicación de la imagen en GHCR)

> **Nota:** la configuración base de CI/CD (GitHub Actions con `uv sync`
> + `ruff` + `pytest`) se adelantó desde la Fase 1, antes de completar
> Fase 2, porque se detectó que ningún Pull Request tenía validación
> automática. Ver `.github/workflows/ci.yml`.
>
> **Nota sobre el build de Docker en CI:** el job `docker-build` construye
> la imagen en cada PR usando archivos vacíos como reemplazo de los 4
> artefactos de `models/` (versionados con DVC en Google Drive, sin
> credenciales configuradas en el runner). Esto valida que el `Dockerfile`
> y las dependencias siguen funcionando, pero no prueba el modelo real —
> eso ya lo cubren `test_api.py` y la validación manual end-to-end del
> equipo.
>
> **Nota sobre el "Pipeline de despliegue":** en cada `push` a `main` (no
> en los PRs), el job `docker-build` etiqueta y publica la imagen ya
> construida en GitHub Container Registry
> (`ghcr.io/aemedina24/mlops-secop-api`), usando el `GITHUB_TOKEN`
> automático del workflow — sin necesidad de credenciales nuevas. Esto
> cierra el ciclo build → publish de un pipeline de despliegue real, pero
> con la misma limitación de arriba: la imagen publicada usa los
> artefactos de modelo vacíos (placeholder), así que sirve como evidencia
> de que el pipeline de publicación funciona de punta a punta, **no**
> como una imagen lista para servir predicciones reales. Para correr la
> API con el modelo real, sigue la Guía de Inicio Rápido (`dvc pull` +
> `docker compose up --build`) más abajo. Desplegar automáticamente esa
> imagen en un servidor o servicio en la nube quedó fuera de alcance: no
> hay un ambiente real donde correrla para este proyecto académico.
>
> **Importante si vas a probar la imagen de GHCR directamente**
> (`docker pull` + `docker run`, sin clonar el repo ni hacer `dvc pull`):
> como sus 4 archivos de `models/` están vacíos, `/health` y `/predict`
> van a devolver un error 500 (el `joblib.load()` interno falla al leer
> un archivo vacío) — **eso es el comportamiento esperado**, no un bug.
> Esa imagen solo demuestra que el pipeline de CI/CD publica
> correctamente; para ver la API respondiendo con predicciones reales,
> usa la Guía de Inicio Rápido más abajo (clonar + `dvc pull` + `docker
> compose up --build`), que sí arma la imagen con el modelo entrenado.

### 📊 Fase 6 — Monitoreo y Gobernanza

- [x] Configuración de `Evidently`
- [x] Monitoreo de Data Drift
- [x] Monitoreo de Model Drift
- [x] Seguimiento del rendimiento del modelo
- [x] Generación de reportes
- [x] Definición de estrategia de reentrenamiento

**Documentación completa de gobernanza y reentrenamiento:**
[`docs/GOBERNANZA.md`](./docs/GOBERNANZA.md) (qué modelo está en
producción, con qué datos/código, métricas y limitaciones conocidas) y
[`docs/ESTRATEGIA_REENTRENAMIENTO.md`](./docs/ESTRATEGIA_REENTRENAMIENTO.md)
(disparadores de reentrenamiento, proceso de validación y rollback).

### 🔁 Fase 7 — Ciclo MLOps Completo

- [ ] Automatización del pipeline completo
- [ ] Integración DVC + MLflow
- [ ] Integración CI/CD + entrenamiento
- [ ] Versionamiento de modelos
- [ ] Validación automática del modelo
- [ ] Estrategia de reentrenamiento automático
- [ ] Documentación final del proyecto

---
## 🎓 ¿Cómo evaluar este proyecto?

Este proyecto no tiene sustentación en vivo — se evalúa clonando y
revisando el repositorio directamente. Hay tres niveles de revisión,
de menos a más esfuerzo, y solo uno de ellos ejecuta el modelo con
predicciones reales:

1. **Revisión estática (sin ejecutar nada):** leer este README, la
   documentación de gobernanza
   ([`docs/GOBERNANZA.md`](./docs/GOBERNANZA.md) y
   [`docs/ESTRATEGIA_REENTRENAMIENTO.md`](./docs/ESTRATEGIA_REENTRENAMIENTO.md)),
   y el historial de Pull Requests + GitHub Actions del repo (CI
   corriendo en cada PR, branch protection activa en `main`). Alcanza
   para ver todo el proceso de MLOps sin correr una sola línea de
   código.
2. **Correr la API con el modelo real (la única forma de ver
   predicciones reales):** seguir la Guía de Inicio Rápido de abajo —
   clonar, `uv sync`, `dvc pull` (trae el modelo/datos desde Google
   Drive, compartido como "cualquiera con el enlace, Lector") y
   `docker compose up --build`. Con eso, `http://localhost:8000/docs`
   expone `/predict` funcionando con el modelo entrenado.
3. **Bajar la imagen publicada en GHCR
   (`docker pull ghcr.io/aemedina24/mlops-secop-api:latest`):** prueba
   que el pipeline de CI/CD publica correctamente sin necesitar ningún
   acceso ni configuración — pero **no** sirve predicciones reales (ver
   la nota en la Fase 5 más arriba): sus artefactos de modelo son
   placeholders vacíos a propósito, así que `/health` y `/predict` van
   a devolver error 500 si se corre standalone. Es evidencia del
   pipeline de despliegue, no un atajo para probar el modelo.

---
## 💻 Guía de Inicio Rápido

1. Clonar el repositorio

   ```powershell
   git clone https://github.com/aemedina24/MLOps_Secop.git
   cd MLOps_Secop
   ```

2. Sincronizar el entorno

   ```powershell
   uv sync
   ```

3. Activar los hooks de pre-commit

   ```powershell
   uv run pre-commit install
   ```

4. Configurar tu token de Socrata (obligatorio para la ingesta)

   El pipeline de ingesta (`src/mlops_secop/data/ingest_secop.py`) requiere
   un `SOCRATA_APP_TOKEN` propio de cada persona — **no se comparte ni se
   sube a Git**, cada quien debe generar el suyo:

   1. Copia la plantilla de variables de entorno:

      ```powershell
      # Windows (PowerShell)
      copy .env.example .env
      ```

      ```bash
      # macOS / Linux
      cp .env.example .env
      ```

   2. Crea una cuenta gratuita en [datos.gov.co](https://www.datos.gov.co/).

   3. Ve a tu perfil → **Mis Aplicaciones** → **Crear Nueva Aplicación**.

   4. Copia el **App Token** generado (no el token secreto) y pégalo en tu `.env`:

      ```
      SOCRATA_APP_TOKEN=tu_token_aqui
      ```

   5. El archivo `.env` nunca se versiona (ya está en `.gitignore`); si
      clonas el proyecto en otra máquina, repite este paso ahí también.

5. Descargar los datos y modelos versionados con DVC

   ```powershell
   uv run dvc pull
   ```

   > **Importante:** los datos procesados y los artefactos de `models/`
   > (el modelo entrenado, el encoder, las tablas de referencia) están
   > versionados con DVC en un remoto de Google Drive, **no en Git**. Para
   > que `dvc pull` funcione, tu cuenta de Google necesita permiso de
   > lectura sobre esa carpeta de Drive — pídele acceso a alguien del
   > equipo si es la primera vez que clonas el repositorio. Sin este paso,
   > `docker compose up --build` fallará al intentar copiar los archivos
   > de `models/` (ver Fase 4 más abajo).

> **Nota (Windows):** `make` no viene instalado por defecto. Instalalo
> con [Chocolatey](https://chocolatey.org/install) (`choco install make -y`,
> requiere PowerShell como Administrador) o usa los comandos equivalentes
> directamente: `uv run ruff check . --fix && uv run ruff format .` (para
> `make quality`) y `uv run pytest` (para `make test`).

---
## 🔧 Comandos Principales

El proyecto utiliza un Makefile para simplificar las tareas frecuentes.

`make quality`

Ejecuta las herramientas de calidad de código.

`make test`

Ejecuta la suite de pruebas automatizadas.

`make setup`

Configura el entorno inicial del proyecto.

---
## 🎯 Objetivo del Proyecto

MLOps_Secop construye un pipeline reproducible de extremo a extremo para
detectar contratos potencialmente atípicos o irregulares en los datos
abiertos de contratación pública de SECOP II. Como no existe una fuente
de datos con contratos ya confirmados como fraudulentos, el problema se
aborda como **detección de anomalías no supervisada** (Isolation
Forest): el modelo aprende el comportamiento típico de un contrato
según su tipo, entidad, valor y modalidad, y señala qué tan atípico es
uno nuevo respecto a ese patrón, explicando con SHAP qué variables
influyen en cada predicción. Un `anomaly_score` alto es un **candidato
a revisión manual por un auditor, no una acusación automática de
fraude**.

Más allá del modelo en sí, el proyecto aplica de punta a punta las
prácticas de MLOps del curso: versionamiento de datos y modelos con
DVC, tracking de experimentos con MLflow, una API de inferencia
contenerizada (FastAPI + Docker), y monitoreo de drift con gobernanza
documentada (Evidently) — ver [`docs/GOBERNANZA.md`](./docs/GOBERNANZA.md)
para el detalle completo, incluyendo las limitaciones conocidas del
modelo actual.

