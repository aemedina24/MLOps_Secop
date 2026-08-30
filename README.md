
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
├── notebooks/              # Exploración y análisis inicial (EDA)
│
├── src/
│   └── mlops_secop/
│       ├── data/           # Ingesta y procesamiento de datos
│       ├── features/       # Ingeniería de características
│       ├── models/         # Definición y gestión de modelos
│       ├── training/       # Entrenamiento
│       └── inference/      # Inferencia y predicciones
│
├── tests/                  # Pruebas automatizadas con pytest
│
├── configs/                # Configuraciones del proyecto
│
├── models/                 # Artefactos de modelos
│
├── reports/                # Reportes y resultados
│
├── scripts/                # Scripts auxiliares
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
| Git + GitHub | ✅ | `main` protegida, PRs obligatorios |
| uv + pyproject.toml + uv.lock | ✅ | Entorno reproducible |
| Dependencias dev separadas | ✅ | `[dependency-groups] dev` (pytest, ruff, pre-commit) |
| Ruff configurado | ✅ | `[tool.ruff]` — reglas E, F, I, UP, line-length 88 |
| pytest configurado + tests reales | ✅ | `[tool.pytest.ini_options]` + `tests/test_mlops_secop.py` |
| pre-commit activo | ✅ | Verificado en cada commit |
| Estructura `src/` | ✅ | Código productivo separado de notebooks |

**Para contribuir al proyecto, ver [`CONTRIBUTING.md`](./CONTRIBUTING.md)**
(flujo de Git, convención de commits, troubleshooting).

### 📦 Fase 2 — Control de Versiones y Datos

- [ ] Configuración del repositorio en `GitHub`
- [ ] Configuración de `DVC`
- [ ] Versionamiento de datasets
- [ ] Organización de datos en `raw/` y `processed/`
- [ ] Creación del pipeline de datos

### 🤖 Fase 3 — Machine Learning y Experiment Tracking

- [ ] Exploración y análisis de los datos (EDA)
- [ ] Feature Engineering
- [ ] Implementación de modelos base
- [ ] Implementación de AutoML
- [ ] Evaluación y comparación de modelos
- [ ] Configuración de `MLflow`
- [ ] Tracking de experimentos
- [ ] Registro de modelos mediante MLflow Model Registry

### 🚀 Fase 4 — Model Serving y Contenedores

- [ ] Creación de API de inferencia con `FastAPI`
- [ ] Creación del `Dockerfile`
- [ ] Contenerización del modelo
- [ ] Configuración de `docker-compose`
- [ ] Pruebas de la API

### 🔄 Fase 5 — CI/CD

- [ ] Configuración de GitHub Actions
- [ ] Automatización de pruebas
- [ ] Automatización de Ruff
- [ ] Validación automática de Pull Requests
- [ ] Construcción automática de imagen Docker
- [ ] Pipeline de despliegue

### 📊 Fase 6 — Monitoreo y Gobernanza

- [ ] Configuración de `Evidently`
- [ ] Monitoreo de Data Drift
- [ ] Monitoreo de Model Drift
- [ ] Seguimiento del rendimiento del modelo
- [ ] Generación de reportes
- [ ] Definición de estrategia de reentrenamiento

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
git clone https://github.com/aemedina24/MLOps_Secop.git
cd MLOps_Secop
2. Sincronizar el entorno
uv sync
3. Activar los hooks de pre-commit
uv run pre-commit install

---
## 🔧 Comandos Principales

El proyecto utiliza un Makefile para simplificar las tareas frecuentes.

make quality

Ejecuta las herramientas de calidad de código.

make test

Ejecuta la suite de pruebas automatizadas.

make setup

Configura el entorno inicial del proyecto.
---
## 🎯 Objetivo del Proyecto

El objetivo de MLOps_Secop es construir un sistema reproducible y mantenible para desarrollar soluciones de Machine Learning utilizando datos de contratación pública del SECOP.