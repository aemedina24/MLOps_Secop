"""
Cliente genérico para consumir APIs de Socrata (SODA) con paginación,
reintentos y timeout configurables.

Este módulo NO conoce nada específico de SECOP II: es reutilizable para
cualquier dataset publicado en un portal Socrata (datos.gov.co,
data.cityofnewyork.us, etc.). La lógica específica de SECOP vive en
`ingest_secop.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class SocrataClientConfig:
    base_url: str
    dataset_id: str
    app_token: str
    timeout_seconds: float = 90.0
    max_retries: int = 3
    backoff_factor: float = 1.5


class SocrataClient:
    """Cliente de paginación para endpoints SODA de Socrata."""

    def __init__(self, config: SocrataClientConfig):
        self._config = config
        self._session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=self._config.max_retries,
            backoff_factor=self._config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"X-App-Token": self._config.app_token})
        return session

    def _endpoint(self) -> str:
        return f"{self._config.base_url}/{self._config.dataset_id}.json"

    def paginate(
        self,
        select: list[str],
        where: str,
        order_by: str,
        page_size: int = 5000,
    ) -> Iterator[list[dict]]:
        """
        Generador que produce UNA página (lista de registros) a la vez.

        Por qué un generador y no una lista acumulada: el requisito pide
        "evitar cargar innecesariamente todo el dataset en memoria" y
        "permitir procesamiento por lotes". Un generador permite que el
        llamador procese/guarde cada página y la descarte antes de pedir
        la siguiente.

        Usa $limit/$offset con $order explícito. Esto es importante:
        sin ORDER BY, la paginación por $offset en Socrata no garantiza
        un orden estable entre llamadas, lo que podría duplicar u omitir
        filas entre páginas.
        """
        offset = 0
        params_base = {
            "$select": ",".join(select),
            "$where": where,
            "$order": order_by,
            "$limit": page_size,
        }

        while True:
            params = {**params_base, "$offset": offset}
            logger.info("Solicitando página offset=%s limit=%s", offset, page_size)

            response = self._session.get(
                self._endpoint(),
                params=params,
                timeout=self._config.timeout_seconds,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Error HTTP {response.status_code} consultando "
                    f"{self._config.dataset_id}: {response.text[:500]}"
                )

            page = response.json()

            if not isinstance(page, list):
                raise RuntimeError(
                    f"Respuesta inesperada de la API (no es una lista): {page}"
                )

            if not page:
                logger.info("Página vacía recibida. Fin de la paginación.")
                break

            yield page

            if len(page) < page_size:
                # Última página parcial: no tiene sentido seguir pidiendo,
                # nos ahorramos una llamada extra que devolvería vacío.
                break

            offset += page_size
