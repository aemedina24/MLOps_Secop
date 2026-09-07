"""
Configuración centralizada del proyecto MLOps_Secop — única fuente de verdad.

Este módulo agrupa dos tipos de configuración, deliberadamente separados:

1. Constantes de dominio SECOP (fijas, no dependen del entorno de
   ejecución): las columnas del dataset, la columna de fecha usada para
   particionar la ingesta, la fecha de inicio del histórico, y los
   tamaños de ventana de la ingesta incremental. No cambian entre
   desarrollo/CI/producción, así que se definen como constantes simples
   de módulo.

2. Configuración de entorno (rutas, parámetros de la API): se resuelven
   con FUNCIONES, no como constantes de módulo. Esto es intencional: si
   fueran constantes (`RAW_DIR = Path(os.environ.get(...))`), su valor
   quedaría fijado la primera vez que Python importa el módulo, y ya no
   se podría sobreescribir después con variables de entorno ni con
   `monkeypatch.setenv(...)` en los tests. Como funciones, el valor se
   lee en el momento en que se llaman, no en el momento en que se
   importan.

Por qué existe este archivo: antes de esta refactorización,
`ingest_secop.py` y `process_secop.py` definían por separado valores
como la ruta de datos crudos (`SECOP_RAW_DIR`), cada uno con su propio
default `"data/raw/secop_ii"` repetido. Si alguno de los dos cambiaba
sin actualizar el otro, los dos pipelines podían terminar apuntando a
carpetas distintas sin ningún aviso. Centralizar esos valores aquí hace
que cambiarlos sea una sola edición, no una búsqueda manual por todo el
proyecto.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

# --------------------------------------------------------------------------
# Dominio SECOP — constantes fijas (no configurables por entorno)
# --------------------------------------------------------------------------

#: Las 22 columnas exigidas del dataset SECOP Integrado (rpmr-utcd) en
#: Datos Abiertos Colombia. Se seleccionan explícitamente en la API (en
#: vez de traer todas las columnas) para blindarnos contra cambios de
#: esquema del dataset público y mantener el contrato de datos explícito.
COLUMNS: list[str] = [
    "nivel_entidad",
    "codigo_entidad_en_secop",
    "nombre_de_la_entidad",
    "nit_de_la_entidad",
    "departamento_entidad",
    "municipio_entidad",
    "estado_del_proceso",
    "modalidad_de_contrataci_n",
    "objeto_a_contratar",
    "objeto_del_proceso",
    "tipo_de_contrato",
    "fecha_de_firma_del_contrato",
    "fecha_inicio_ejecuci_n",
    "fecha_fin_ejecuci_n",
    "numero_del_contrato",
    "numero_de_proceso",
    "valor_contrato",
    "nom_raz_social_contratista",
    "url_contrato",
    "origen",
    "tipo_documento_proveedor",
    "documento_proveedor",
]

#: Columna de fecha usada para particionar la ingesta en ventanas de
#: tiempo (ver `ingest_secop._iter_date_chunks`).
DATE_COLUMN = "fecha_de_firma_del_contrato"

#: Fecha de inicio del histórico cuando no existe un checkpoint previo.
DEFAULT_START_DATE = datetime(2022, 1, 1, tzinfo=UTC)

#: Tamaño de cada ventana de fecha en la ingesta. Mantiene el $offset de
#: la API de Socrata siempre acotado dentro de cada ventana, en vez de
#: crecer sin límite sobre todo el histórico.
CHUNK_DAYS = 15

#: Días que se retrocede el checkpoint en cada corrida incremental, para
#: recapturar contratos que la entidad pública registró en SECOP con
#: retraso respecto a su fecha real de firma ("late-arriving data").
OVERLAP_DAYS = 5


# --------------------------------------------------------------------------
# Configuración de entorno — funciones, no constantes de módulo.
# --------------------------------------------------------------------------


def raw_dir() -> Path:
    """Carpeta de datos crudos, particionada por `ingestion_date=YYYY-MM-DD`."""
    return Path(os.environ.get("SECOP_RAW_DIR", "data/raw/secop_ii"))


def processed_dir() -> Path:
    """Carpeta de salida del dataset procesado y deduplicado."""
    return Path(os.environ.get("SECOP_PROCESSED_DIR", "data/processed/secop_ii"))


def checkpoint_path() -> Path:
    """Ruta del checkpoint de la última extracción incremental exitosa."""
    return Path(
        os.environ.get(
            "SECOP_CHECKPOINT_PATH", "data/raw/_checkpoints/last_extraction.json"
        )
    )


def api_base_url() -> str:
    """URL base del portal Socrata (Datos Abiertos Colombia)."""
    return os.environ.get("SECOP_API_BASE_URL", "https://www.datos.gov.co/resource")


def dataset_id() -> str:
    """Identificador del dataset SECOP Integrado en el portal Socrata."""
    return os.environ.get("SECOP_DATASET_ID", "rpmr-utcd")


def page_size() -> int:
    """Tamaño de página por defecto para la paginación de la API SODA."""
    return int(os.environ.get("SECOP_PAGE_SIZE", "5000"))
