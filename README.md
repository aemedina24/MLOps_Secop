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

| Componente | Estado | Detalle |
|---|---|---|
| Git + GitHub | 🟡 | Flujo de PRs en uso; protección formal de rama `main` pendiente de activar en GitHub Settings |
| uv + pyproject.toml + uv.lock | ✅ | Entorno reproducible |
| Dependencias dev separadas | ✅ | `[dependency-groups] dev` (pytest, ruff, pre-commit) |
| Ruff configurado | ✅ | `[tool.ruff]` — reglas E, F, I, UP, line-length 88 |
| pytest configurado + tests reales | ✅ | `[tool.pytest.ini_options]` + `tests/test_mlops_secop.py` |
| pre-commit activo | ✅ | Verificado en cada commit |
| Estructura `src/` | ✅ | Código productivo separado de notebooks |

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
- [ ] Construcción automática de imagen Docker
- [ ] Pipeline de despliegue

> **Nota:** la configuración base de CI/CD (GitHub Actions con `uv sync`
> + `ruff` + `pytest`) se adelantó desde la Fase 1, antes de completar
> Fase 2, porque se detectó que ningún Pull Request tenía validación
> automática. Ver `.github/workflows/ci.yml`.

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

El objetivo de MLOps_Secop es construir un sistema reproducible y mantenible para desarrollar soluciones de Machine Learning utilizando datos de contratación pública del SECOP.

