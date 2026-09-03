"""
Ingesta RAW de contratos SECOP II desde la API de Datos Abiertos Colombia
(portal Socrata, dataset rpmr-utcd, "SECOP Integrado").

Este script:
1. Calcula el rango temporal (incremental si hay checkpoint, o desde
   2022-01-01 en la primera ejecución).
2. Divide ese rango en ventanas de tiempo pequeñas (CHUNK_DAYS) y pagina
   la API dentro de cada ventana por separado. Esto evita que el $offset
   crezca sin límite en datasets grandes: Socrata recorre internamente
   todas las filas anteriores al offset pedido, así que offsets altos
   (millones de filas) empiezan a superar cualquier timeout razonable.
   Particionar por fecha mantiene el offset siempre acotado dentro de
   cada ventana.
3. Guarda cada página como Parquet en `data/raw/` SIN transformar los
   valores originales.
4. Actualiza un checkpoint con la fecha de la última extracción exitosa,
   habilitando ejecuciones incrementales futuras.

Variables de entorno requeridas:
- SOCRATA_APP_TOKEN

Variables de entorno opcionales:
- SECOP_API_BASE_URL     (default: https://www.datos.gov.co/resource)
- SECOP_DATASET_ID       (default: rpmr-utcd)
- SECOP_PAGE_SIZE        (default: 5000)
- SECOP_RAW_DIR          (default: data/raw/secop_ii)
- SECOP_CHECKPOINT_PATH  (default: data/raw/_checkpoints/last_extraction.json)

Ejecución manual (PowerShell, con uv):
    uv run python -m mlops_secop.data.ingest_secop
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from mlops_secop.data.socrata_client import SocrataClient, SocrataClientConfig

# Carga las variables de .env hacia el entorno del proceso, si el archivo
# existe. En producción (CI/CD, Docker) las variables normalmente ya vienen
# inyectadas por la plataforma, así que esto es un no-op inofensivo ahí —
# solo tiene efecto real en desarrollo local.
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 1. Configuración fija del dataset (las 22 columnas exigidas, sin SELECT *)
# --------------------------------------------------------------------------

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

DATE_COLUMN = "fecha_de_firma_del_contrato"
DEFAULT_START_DATE = datetime(2022, 1, 1, tzinfo=UTC)

# Tamaño de cada ventana de fecha para acotar el $offset. 15 días es un
# punto de partida razonable para un dataset de millones de filas
# distribuidas en varios años; si en el futuro alguna ventana sigue
# teniendo timeouts, se puede bajar este valor sin tocar el resto del
# diseño.
CHUNK_DAYS = 15


def _load_config() -> SocrataClientConfig:
    token = os.environ.get("SOCRATA_APP_TOKEN")
    if not token:
        raise RuntimeError(
            "SOCRATA_APP_TOKEN no está definido. Define la variable de "
            "entorno antes de ejecutar la ingesta (ver .env.example). "
            "Nunca escribas el token directamente en el código."
        )

    return SocrataClientConfig(
        base_url=os.environ.get(
            "SECOP_API_BASE_URL", "https://www.datos.gov.co/resource"
        ),
        dataset_id=os.environ.get("SECOP_DATASET_ID", "rpmr-utcd"),
        app_token=token,
    )


# --------------------------------------------------------------------------
# 2. Checkpoint — habilita la extracción incremental
# --------------------------------------------------------------------------


def _checkpoint_path() -> Path:
    return Path(
        os.environ.get(
            "SECOP_CHECKPOINT_PATH", "data/raw/_checkpoints/last_extraction.json"
        )
    )


def _read_checkpoint() -> datetime | None:
    path = _checkpoint_path()
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return datetime.fromisoformat(payload["last_extraction_end"])


def _write_checkpoint(extraction_end: datetime) -> None:
    path = _checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "last_extraction_end": extraction_end.isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _resolve_date_range() -> tuple[datetime, datetime]:
    """
    Resuelve el rango [desde, hasta) de la extracción actual.

    - hasta = inicio del día SIGUIENTE al momento de ejecución. Se calcula
      en tiempo de ejecución con datetime.now(); nunca es un valor fijo
      en el código, tal como exige el requisito.
    - desde = checkpoint de la última extracción exitosa si existe,
      o 2022-01-01 en la primera ejecución.
    """
    now = datetime.now(UTC)
    hasta = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    checkpoint = _read_checkpoint()
    desde = checkpoint if checkpoint is not None else DEFAULT_START_DATE

    if desde < DEFAULT_START_DATE:
        # Protección explícita: sin importar lo que diga un checkpoint
        # corrupto o manual, nunca extraemos antes de 2022-01-01.
        desde = DEFAULT_START_DATE

    return desde, hasta


def _soda_datetime(dt: datetime) -> str:
    """Formatea un datetime al formato floating timestamp que espera SODA."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _iter_date_chunks(
    desde: datetime, hasta: datetime, chunk_days: int = CHUNK_DAYS
) -> list[tuple[datetime, datetime]]:
    """
    Divide el rango [desde, hasta) en ventanas consecutivas de tamaño fijo.

    Por qué existe esta función: sin ella, una sola consulta cubriría todo
    el rango y el $offset crecería sin límite a medida que se avanza en
    la paginación (ver docstring del módulo). Al particionar por fecha,
    cada ventana empieza su propia paginación en offset=0.
    """
    chunks: list[tuple[datetime, datetime]] = []
    current = desde
    delta = timedelta(days=chunk_days)
    while current < hasta:
        chunk_end = min(current + delta, hasta)
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks


# --------------------------------------------------------------------------
# 3. Persistencia RAW (Parquet, sin transformar valores)
# --------------------------------------------------------------------------


def _save_page_as_parquet(page: list[dict], run_dir: Path, page_number: int) -> Path:
    """
    Guarda una página como Parquet SIN transformar los valores originales.

    Todas las columnas se conservan como string, que es como llegan desde
    la API SODA (JSON). Convertir tipos (fechas a datetime, valor_contrato
    a numérico, etc.) es responsabilidad de una capa de `processing`
    posterior, no de la capa RAW.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    file_path = run_dir / f"part-{page_number:05d}.parquet"

    normalized_rows = [{col: row.get(col) for col in COLUMNS} for row in page]

    table = pa.Table.from_pylist(
        normalized_rows, schema=pa.schema([(col, pa.string()) for col in COLUMNS])
    )
    pq.write_table(table, file_path)
    logger.info("Guardadas %s filas en %s", len(page), file_path)
    return file_path


# --------------------------------------------------------------------------
# 4. Orquestación
# --------------------------------------------------------------------------


def run_ingestion(page_size: int | None = None) -> dict:
    config = _load_config()
    page_size = page_size or int(os.environ.get("SECOP_PAGE_SIZE", "5000"))

    desde, hasta = _resolve_date_range()
    logger.info("Rango de extracción: %s -> %s", desde.isoformat(), hasta.isoformat())

    raw_root = Path(os.environ.get("SECOP_RAW_DIR", "data/raw/secop_ii"))
    run_dir = raw_root / f"ingestion_date={datetime.now(UTC):%Y-%m-%d}"

    client = SocrataClient(config)

    total_rows = 0
    total_pages = 0
    written_files: list[str] = []
    page_number = 0

    chunks = _iter_date_chunks(desde, hasta)
    logger.info(
        "Extracción dividida en %s ventana(s) de %s días para evitar "
        "offsets grandes en la API",
        len(chunks),
        CHUNK_DAYS,
    )

    for chunk_index, (chunk_desde, chunk_hasta) in enumerate(chunks, start=1):
        where_clause = (
            f"{DATE_COLUMN} >= '{_soda_datetime(chunk_desde)}' "
            f"AND {DATE_COLUMN} < '{_soda_datetime(chunk_hasta)}'"
        )
        logger.info(
            "Ventana %s/%s: %s -> %s",
            chunk_index,
            len(chunks),
            chunk_desde.isoformat(),
            chunk_hasta.isoformat(),
        )

        for page in client.paginate(
            select=COLUMNS,
            where=where_clause,
            order_by=f"{DATE_COLUMN} ASC",
            page_size=page_size,
        ):
            page_number += 1
            file_path = _save_page_as_parquet(page, run_dir, page_number)
            written_files.append(str(file_path))
            total_rows += len(page)
            total_pages += 1

    if total_rows > 0:
        # El checkpoint solo avanza si TODAS las ventanas terminaron sin
        # excepciones. Si la extracción falla a mitad de camino, el
        # checkpoint queda intacto y la siguiente corrida reintenta desde
        # el principio: la ingesta es reanudable, aunque no incremental
        # dentro de una misma corrida fallida.
        _write_checkpoint(hasta)
    else:
        logger.info(
            "No se encontraron registros nuevos en el rango solicitado. "
            "El checkpoint NO se actualiza."
        )

    summary = {
        "desde": desde.isoformat(),
        "hasta": hasta.isoformat(),
        "total_rows": total_rows,
        "total_pages": total_pages,
        "output_dir": str(run_dir),
        "files": written_files,
    }
    logger.info("Resumen de ingesta: %s", summary)
    return summary


if __name__ == "__main__":
    run_ingestion()
